//go:build unix

package localrun

import (
	"context"
	"errors"
	"os"
	"path/filepath"
	"strings"
	"testing"

	api "github.com/baldwinSPC/glimmer-burnin/api/v1alpha1"
)

// The line podman-docker's /usr/bin/docker writes to stderr on every call.
const shimBanner = "Emulate Docker CLI using podman. Create /etc/containers/nodocker to quiet msg."

// installFakes writes each named script into a fresh directory and makes that
// directory the whole of PATH.
func installFakes(t *testing.T, scripts map[string]string) string {
	t.Helper()
	dir := t.TempDir()
	for name, body := range scripts {
		if err := os.WriteFile(filepath.Join(dir, name), []byte("#!/bin/sh\n"+body), 0o755); err != nil {
			t.Fatal(err)
		}
	}
	t.Setenv("PATH", dir)
	return dir
}

// podman-docker's shim, as Debian and Fedora ship it: the banner, then podman.
const shimScript = `echo "` + shimBanner + `" >&2
exec podman "$@"
`

const podmanScript = `if [ "$1" = --version ]; then echo "podman version 5.4.2"; exit 0; fi
`

// #594: --runtime auto on a host whose docker is podman-docker's shim runs
// podman itself, with podman's device flags.
func TestAutoDetectRunsPodmanBehindThePodmanDockerShim(t *testing.T) {
	dir := installFakes(t, map[string]string{"docker": shimScript, "podman": podmanScript})
	rt, err := DetectRuntime()
	if err != nil {
		t.Fatalf("DetectRuntime: %v", err)
	}
	if want := filepath.Join(dir, "podman"); rt.Binary != want || rt.GPUFlagStyle != GPUFlagCDI {
		t.Errorf("detected %s (%s), want %s (%s)", rt.Binary, rt.GPUFlagStyle, want, GPUFlagCDI)
	}
}

// A shim whose podman is not on PATH is still podman, so it keeps the shim and
// takes podman's device flags.
func TestAutoDetectKeepsAShimWithNoPodmanOnPathAsPodman(t *testing.T) {
	dir := installFakes(t, map[string]string{"docker": `echo "` + shimBanner + `" >&2
echo "podman version 5.4.2"
`})
	rt, err := DetectRuntime()
	if err != nil {
		t.Fatalf("DetectRuntime: %v", err)
	}
	if want := filepath.Join(dir, "docker"); rt.Binary != want || rt.GPUFlagStyle != GPUFlagCDI {
		t.Errorf("detected %s (%s), want %s (%s)", rt.Binary, rt.GPUFlagStyle, want, GPUFlagCDI)
	}
}

// A real docker is still preferred over podman, with docker's own flags.
func TestAutoDetectKeepsARealDocker(t *testing.T) {
	dir := installFakes(t, map[string]string{
		"docker": `echo "Docker version 27.3.1, build ce12230"
`,
		"podman": podmanScript,
	})
	rt, err := DetectRuntime()
	if err != nil {
		t.Fatalf("DetectRuntime: %v", err)
	}
	if want := filepath.Join(dir, "docker"); rt.Binary != want || rt.GPUFlagStyle != GPUFlagDockerNative {
		t.Errorf("detected %s (%s), want %s (%s)", rt.Binary, rt.GPUFlagStyle, want, GPUFlagDockerNative)
	}
}

// runThroughShim runs one test through a CLIRuntime whose binary is a docker
// that prints the shim's banner and then behaves as body says.
func runThroughShim(t *testing.T, body string) (TestResult, Execution) {
	t.Helper()
	dir := installFakes(t, map[string]string{"docker": `echo "` + shimBanner + `" >&2
` + body})
	rt := &CLIRuntime{Binary: filepath.Join(dir, "docker")}
	exec, err := rt.Run(context.Background(), RunSpec{Image: "example.invalid/runner:test"})
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	rep, err := Run(context.Background(), onePlan(PlannedTest{Name: "t", Spec: testSpec(), Required: true}, 0), rt, Hooks{})
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	if len(rep.Results) != 1 {
		t.Fatalf("got %d results, want 1", len(rep.Results))
	}
	return rep.Results[0], exec
}

// The banner is not part of what the runtime said about the container.
func TestTheShimsBannerIsNotTheRuntimesOutput(t *testing.T) {
	res, exec := runThroughShim(t, `echo tflops=101
`)
	if res.Phase != api.RunPassed {
		t.Errorf("phase = %s (%q), want Passed", res.Phase, res.Message)
	}
	if strings.Contains(exec.Stderr, "Emulate Docker CLI") {
		t.Errorf("Stderr = %q, want the shim's banner removed", exec.Stderr)
	}
}

// A runtime that could not start the container says why after its progress
// lines, and that line is the message, never the banner.
func TestARuntimeFailureShowsTheRuntimesError(t *testing.T) {
	rows := []struct {
		name, stderr, want string
	}{
		{"podman", "Trying to pull example.invalid/runner:test...\nError: initializing source docker://example.invalid/runner:test: pinging container registry example.invalid: no such host",
			"Error: initializing source docker://example.invalid/runner:test"},
		{"docker", "Unable to find image 'example.invalid/runner:test' locally\ndocker: Error response from daemon: pull access denied for example.invalid/runner.\nSee 'docker run --help'.",
			"docker: Error response from daemon: pull access denied"},
	}
	for _, r := range rows {
		t.Run(r.name, func(t *testing.T) {
			res, _ := runThroughShim(t, `printf '%s\n' "`+r.stderr+`" >&2
exit 125
`)
			if res.Phase != api.RunError {
				t.Fatalf("phase = %s, want Error", res.Phase)
			}
			if !strings.Contains(res.Message, r.want) || strings.Contains(res.Message, "Emulate Docker CLI") {
				t.Errorf("message = %q, want it to carry %q and not the shim's banner", res.Message, r.want)
			}
		})
	}
}

// A stop that fails through the shim reports the runtime's error, not the
// banner.
func TestAFailedStopThroughTheShimReportsTheRuntimesError(t *testing.T) {
	out := shimBanner + "\nError: no container with name or ID \"burnin-x\" found: no such container\n"
	if got, want := failure([]byte(out), errors.New("exit status 125")), `Error: no container with name or ID "burnin-x" found: no such container`; got != want {
		t.Errorf("failure = %q, want %q", got, want)
	}
}
