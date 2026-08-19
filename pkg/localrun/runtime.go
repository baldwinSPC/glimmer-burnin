package localrun

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"os"
	"os/exec"
	"sort"
	"strings"
	"sync"
)

// ContainerRuntime runs one container to completion.
//
// An interface so the engine can be tested without a container runtime, and so
// an embedding agent that already has its own way of launching containers — the
// control plane's readiness validator does — can supply that instead of being
// made to shell out.
type ContainerRuntime interface {
	// Run executes the spec and returns what the container produced. It returns
	// an error only when the container could not be run at all; a container that
	// ran and exited non-zero is a successful Run with a non-zero ExitCode,
	// because that exit code is the runner contract's verdict and not a failure
	// of this layer.
	Run(ctx context.Context, spec RunSpec) (Execution, error)
	// Name identifies the runtime for messages.
	Name() string
}

// StreamingRuntime is a ContainerRuntime that can report stdout as it arrives
// rather than only when the container exits.
//
// A SEPARATE, OPTIONAL interface rather than a method on ContainerRuntime,
// because ContainerRuntime is exported and a third party may already implement
// it; adding a method would break every such implementation for a feature it
// may not need. The engine type-asserts, and a runtime that does not implement
// this simply produces no checkpoints — which is a degradation of evidence,
// never of the verdict.
//
// onOutput is called with EVERYTHING read so far, not the new fragment. The
// runner contract is last-occurrence-wins over the whole stream, so a consumer
// handed only the tail would parse a metric out of the middle of a line and
// report whichever fragment happened to land in that read.
type StreamingRuntime interface {
	ContainerRuntime
	RunStreaming(ctx context.Context, spec RunSpec, onOutput func(stdoutSoFar string)) (Execution, error)
}

// Execution is what a container produced.
type Execution struct {
	ExitCode int
	Stdout   string
	Stderr   string
	// TimedOut says the container was killed at its deadline rather than
	// exiting. Kept distinct because a killed runner reported no verdict, and
	// the engine must not read the kill signal's exit code as one.
	TimedOut bool
}

// CLIRuntime drives docker, podman or nerdctl through os/exec.
//
// No container SDK: the three CLIs are stable, present wherever the runtimes
// are, and shelling out keeps this package free of a dependency that would have
// to be licence-reviewed and kept current for a feature whose whole job is to
// run someone else's image.
type CLIRuntime struct {
	// Binary is the executable, e.g. "docker".
	Binary string
	// GPUFlagStyle selects how accelerator access is requested. Empty means
	// autodetected at construction.
	GPUFlagStyle GPUFlagStyle
}

// GPUFlagStyle is how a runtime is asked for accelerators.
type GPUFlagStyle string

const (
	// GPUFlagDockerNative is `--gpus all`, which docker supports through the
	// NVIDIA container toolkit.
	GPUFlagDockerNative GPUFlagStyle = "gpus"
	// GPUFlagCDI is `--device nvidia.com/gpu=all`, the Container Device
	// Interface form that podman and nerdctl prefer.
	GPUFlagCDI GPUFlagStyle = "cdi"
)

// runtimeCandidates is the detection order.
//
// docker first because it is the one CI exercises and the one most sites have;
// the others are best-effort and documented as such.
var runtimeCandidates = []struct {
	binary string
	style  GPUFlagStyle
}{
	{"docker", GPUFlagDockerNative},
	{"podman", GPUFlagCDI},
	{"nerdctl", GPUFlagCDI},
}

// ErrNoRuntime is returned when no container runtime could be found.
var ErrNoRuntime = errors.New("no container runtime found")

// DetectRuntime finds a usable runtime.
//
// It checks the binary is on PATH and nothing more. Whether the daemon is
// running is not probed here: the first Run will say so, with the runtime's own
// error, which is more informative than anything this could synthesise.
func DetectRuntime() (*CLIRuntime, error) {
	for _, c := range runtimeCandidates {
		if path, err := exec.LookPath(c.binary); err == nil {
			return &CLIRuntime{Binary: path, GPUFlagStyle: c.style}, nil
		}
	}
	names := make([]string, 0, len(runtimeCandidates))
	for _, c := range runtimeCandidates {
		names = append(names, c.binary)
	}
	return nil, fmt.Errorf("%w: looked for %s on PATH", ErrNoRuntime, strings.Join(names, ", "))
}

// NewRuntime returns a runtime for a named binary.
func NewRuntime(binary string) (*CLIRuntime, error) {
	for _, c := range runtimeCandidates {
		if c.binary == binary {
			path, err := exec.LookPath(binary)
			if err != nil {
				return nil, fmt.Errorf("%s is not on PATH: %w", binary, err)
			}
			return &CLIRuntime{Binary: path, GPUFlagStyle: c.style}, nil
		}
	}
	return nil, fmt.Errorf("unknown container runtime %q", binary)
}

// Name identifies the runtime.
func (r *CLIRuntime) Name() string { return r.Binary }

// Run executes one container.
func (r *CLIRuntime) Run(ctx context.Context, spec RunSpec) (Execution, error) {
	return r.RunStreaming(ctx, spec, nil)
}

// RunStreaming executes one container, reporting stdout as it arrives.
//
// ONE implementation, with the streaming as the general case and Run passing a
// nil callback, rather than two paths that could disagree about deadlines, exit
// codes or stderr. A second copy of the "was this a deadline kill or a
// verdict?" logic is exactly the kind of divergence this project spends its
// tests catching.
func (r *CLIRuntime) RunStreaming(ctx context.Context, spec RunSpec, onOutput func(string)) (Execution, error) {
	// The deadline is enforced here rather than left to the runtime, so that a
	// hung runner is killed on the same schedule regardless of which CLI is
	// driving and whether it honours its own timeout flags.
	if spec.Timeout > 0 {
		var cancel context.CancelFunc
		ctx, cancel = context.WithTimeout(ctx, spec.Timeout)
		defer cancel()
	}

	args := r.args(spec)
	cmd := exec.CommandContext(ctx, r.Binary, args...)

	stdout := &liveBuffer{notify: onOutput}
	var stderr bytes.Buffer
	cmd.Stdout = stdout
	cmd.Stderr = &stderr
	// Never inherit the caller's stdin: a runner that reads from it would block
	// forever in a pipeline, and none of them has any business doing so.
	cmd.Stdin = nil

	err := cmd.Run()
	out := Execution{Stdout: stdout.String(), Stderr: stderr.String()}

	switch {
	case err == nil:
		out.ExitCode = 0

	case errors.Is(ctx.Err(), context.DeadlineExceeded):
		// Killed at its deadline. The exit code here is the signal, not a
		// verdict, so it is replaced with one that cannot be mistaken for the
		// runner contract's 0/1/2 — the same reasoning behind the operator
		// recording a deadline kill as Error rather than reading the code.
		out.TimedOut = true
		out.ExitCode = -1
		out.Stderr = strings.TrimSpace(fmt.Sprintf(
			"container exceeded its %s deadline and was killed — no verdict was reported. %s",
			spec.Timeout, out.Stderr))

	case isExitError(err, &out):
		// A non-zero exit is the runner's verdict, delivered.

	default:
		return Execution{}, fmt.Errorf("running %s: %w", r.Binary, err)
	}

	return out, nil
}

// liveBuffer accumulates a command's stdout and announces each write.
//
// os/exec writes to cmd.Stdout from a goroutine it owns, and the engine reads
// the accumulated text from its own, so every access is under the mutex. Before
// this existed the whole of stdout was invisible until the process exited,
// which is why a multi-hour soak could be killed at hour six having published
// nothing at all — the evidence was sitting in a buffer nobody could read.
//
// The notification carries the whole stream rather than the fragment just
// written: a container's stdout arrives in whatever chunks the pipe delivers,
// which split lines in the middle, and the runner contract is defined over
// complete `key=value` lines.
type liveBuffer struct {
	mu     sync.Mutex
	buf    bytes.Buffer
	notify func(string)
}

func (b *liveBuffer) Write(p []byte) (int, error) {
	b.mu.Lock()
	n, err := b.buf.Write(p)
	var snapshot string
	if b.notify != nil {
		snapshot = b.buf.String()
	}
	b.mu.Unlock()

	// Called OUTSIDE the lock. The callback parses and may deliver over a
	// network, and holding the buffer's lock across that would block the
	// container's own output — turning a slow sink into a stalled runner.
	if b.notify != nil {
		b.notify(snapshot)
	}
	return n, err
}

func (b *liveBuffer) String() string {
	b.mu.Lock()
	defer b.mu.Unlock()
	return b.buf.String()
}

// isExitError extracts an exit code from a command failure.
func isExitError(err error, out *Execution) bool {
	var ee *exec.ExitError
	if errors.As(err, &ee) {
		out.ExitCode = ee.ExitCode()
		return true
	}
	return false
}

// args builds the command line.
//
// Ordering is deterministic so a test can assert on it and a user can read a
// logged command without hunting.
func (r *CLIRuntime) args(spec RunSpec) []string {
	args := []string{"run", "--rm"}

	if spec.HostNetwork {
		args = append(args, "--network", "host")
	}
	if spec.Privileged {
		args = append(args, "--privileged")
	}
	if spec.UnlimitedMemlock {
		// There is no PodSpec field for this at all, which is why the fabric
		// runners document a node-level containerd change in-cluster. Here it
		// is one flag.
		args = append(args, "--ulimit", "memlock=-1:-1")
	}

	switch spec.GPUAccess {
	case GPUNvidia:
		if r.GPUFlagStyle == GPUFlagCDI {
			args = append(args, "--device", "nvidia.com/gpu=all")
		} else {
			args = append(args, "--gpus", "all")
		}
	case GPUAMD:
		// AMD is device nodes rather than a runtime integration.
		args = append(args, "--device", "/dev/kfd", "--device", "/dev/dri")
	}

	for _, m := range spec.Mounts {
		v := m.Source + ":" + m.Target
		if m.ReadOnly {
			v += ":ro"
		}
		args = append(args, "-v", v)
	}
	for _, d := range spec.Devices {
		args = append(args, "--device", d)
	}

	for _, k := range sortedKeys(spec.Env) {
		args = append(args, "-e", k+"="+spec.Env[k])
	}

	if len(spec.Command) > 0 {
		args = append(args, "--entrypoint", spec.Command[0])
	}

	args = append(args, spec.Image)

	if len(spec.Command) > 1 {
		args = append(args, spec.Command[1:]...)
	}
	args = append(args, spec.Args...)
	return args
}

func sortedKeys(m map[string]string) []string {
	out := make([]string, 0, len(m))
	for k := range m {
		out = append(out, k)
	}
	sort.Strings(out)
	return out
}

// PreflightMounts checks the host paths a test declares actually exist.
//
// The kubelet refuses to start a pod whose assert-only hostPath is missing, and
// this reproduces that check rather than letting a runner start and degrade.
// The failure is a configuration Error either way; catching it before the
// container runs makes the message name the path instead of leaving a runner to
// report something downstream of it.
func PreflightMounts(spec RunSpec) error {
	var missing []string
	for _, m := range spec.Mounts {
		if _, err := os.Stat(m.Source); err != nil {
			missing = append(missing, m.Source)
		}
	}
	if len(missing) > 0 {
		return fmt.Errorf("declared host path(s) not present: %s", strings.Join(missing, ", "))
	}
	return nil
}
