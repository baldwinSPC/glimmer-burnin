//go:build unix

package localrun

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"sync"
	"testing"
	"time"
)

// stubRuntime is a CLIRuntime driving testdata/stubruntime.sh, whose containers
// outlive the client that started them, as a daemon's do.
type stubRuntime struct {
	rt    *CLIRuntime
	state string

	mu   sync.Mutex
	logs []string
}

func newStubRuntime(t *testing.T, ticks int, daemonDown bool) *stubRuntime {
	t.Helper()
	bin := filepath.Join(t.TempDir(), "docker")
	src, err := os.ReadFile(filepath.Join("testdata", "stubruntime.sh"))
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(bin, src, 0o755); err != nil {
		t.Fatal(err)
	}
	s := &stubRuntime{state: t.TempDir()}
	t.Setenv("STUB_STATE", s.state)
	t.Setenv("STUB_TICKS", fmt.Sprint(ticks))
	if daemonDown {
		t.Setenv("STUB_DAEMON_DOWN", "1")
	}
	s.rt = &CLIRuntime{Binary: bin, RunID: "0123456789abcdef", Logf: func(format string, args ...any) {
		s.mu.Lock()
		defer s.mu.Unlock()
		s.logs = append(s.logs, fmt.Sprintf(format, args...))
	}}
	return s
}

// startForeign marks a container this runtime did not start as running.
func (s *stubRuntime) startForeign(t *testing.T, name string) {
	t.Helper()
	if err := os.WriteFile(filepath.Join(s.state, "running."+name), nil, 0o644); err != nil {
		t.Fatal(err)
	}
}

// calls is every invocation of the runtime, one argv per line.
func (s *stubRuntime) calls(t *testing.T) []string {
	t.Helper()
	b, err := os.ReadFile(filepath.Join(s.state, "argv"))
	if err != nil && !os.IsNotExist(err) {
		t.Fatal(err)
	}
	return strings.Split(strings.TrimSpace(string(b)), "\n")
}

// running is the names of the containers still running.
func (s *stubRuntime) running(t *testing.T) []string {
	t.Helper()
	m, err := filepath.Glob(filepath.Join(s.state, "running.*"))
	if err != nil {
		t.Fatal(err)
	}
	var names []string
	for _, p := range m {
		names = append(names, strings.TrimPrefix(filepath.Base(p), "running."))
	}
	return names
}

// started is the name the run call gave its container.
func (s *stubRuntime) started(t *testing.T) string {
	t.Helper()
	for _, c := range s.calls(t) {
		f := strings.Fields(c)
		if len(f) > 0 && f[0] == "run" {
			if i := slices.Index(f, "--name"); i >= 0 && i+1 < len(f) {
				return f[i+1]
			}
			t.Fatalf("the run call named no container: %s", c)
		}
	}
	t.Fatal("no run call")
	return ""
}

// waitStarted blocks until the run call's container is running.
func (s *stubRuntime) waitStarted(t *testing.T) {
	t.Helper()
	deadline := time.Now().Add(10 * time.Second)
	for time.Now().Before(deadline) {
		if slices.ContainsFunc(s.running(t), func(n string) bool { return strings.HasPrefix(n, "burnin-") }) {
			return
		}
		time.Sleep(20 * time.Millisecond)
	}
	t.Fatalf("the container never started; calls: %q", s.calls(t))
}

// stopCalls is every kill and rm the runtime received.
func (s *stubRuntime) stopCalls(t *testing.T) []string {
	t.Helper()
	var out []string
	for _, c := range s.calls(t) {
		if strings.HasPrefix(c, "kill ") || strings.HasPrefix(c, "rm ") {
			out = append(out, c)
		}
	}
	return out
}

func (s *stubRuntime) logged() []string {
	s.mu.Lock()
	defer s.mu.Unlock()
	return slices.Clone(s.logs)
}

// #584: cancelling the context SIGKILLs the runtime's client, and the container
// it started belongs to the daemon. The stop names that container and only it:
// a container someone else started on the same host is still running after.
func TestACancelledRunStopsTheContainerItStartedAndNothingElse(t *testing.T) {
	s := newStubRuntime(t, 0, false)
	s.startForeign(t, "someone-elses")

	ctx, cancel := context.WithCancel(context.Background())
	done := make(chan error, 1)
	go func() {
		_, err := s.rt.Run(ctx, RunSpec{Image: "stub:1"})
		done <- err
	}()
	s.waitStarted(t)
	name := s.started(t)
	cancel()
	select {
	case <-done:
	case <-time.After(StopTimeout + 5*time.Second):
		t.Fatal("Run did not return after its context was cancelled")
	}

	if got, want := s.stopCalls(t), []string{"kill " + name, "rm --force " + name}; !slices.Equal(got, want) {
		t.Errorf("stop calls = %q, want %q", got, want)
	}
	if got := s.running(t); !slices.Equal(got, []string{"someone-elses"}) {
		t.Errorf("running after the cancel = %q, want only the container this runtime did not start", got)
	}
	if logs := s.logged(); len(logs) != 1 || !strings.Contains(logs[0], "killed container "+name) {
		t.Errorf("logged %q, want one line saying %s was killed", logs, name)
	}
}

// The test's own deadline ends the client the same way, so it stops the
// container the same way, and "was killed" in the message is then true.
func TestADeadlineStopsTheContainerToo(t *testing.T) {
	s := newStubRuntime(t, 0, false)
	out, err := s.rt.Run(context.Background(), RunSpec{Image: "stub:1", Timeout: 500 * time.Millisecond})
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	name := s.started(t)
	if !out.TimedOut || out.ExitCode != -1 {
		t.Errorf("TimedOut=%v ExitCode=%d, want a deadline kill", out.TimedOut, out.ExitCode)
	}
	if got, want := s.stopCalls(t), []string{"kill " + name, "rm --force " + name}; !slices.Equal(got, want) {
		t.Errorf("stop calls = %q, want %q", got, want)
	}
	if got := s.running(t); len(got) != 0 {
		t.Errorf("still running after the deadline: %q", got)
	}
	if !strings.Contains(out.Stderr, "deadline and was killed") {
		t.Errorf("Stderr = %q, want the deadline kill named", out.Stderr)
	}
}

// A container that exits by itself is not stopped: there is nothing to stop,
// and a kill naming it would be a kill nobody asked for.
func TestAContainerThatExitsByItselfIsNotStopped(t *testing.T) {
	s := newStubRuntime(t, 4, false)
	out, err := s.rt.Run(context.Background(), RunSpec{Image: "stub:1", Timeout: time.Minute})
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	if out.ExitCode != 0 || !strings.Contains(out.Stdout, "tflops=101") {
		t.Fatalf("exit %d stdout %q, want the runner's own exit", out.ExitCode, out.Stdout)
	}
	if got := s.stopCalls(t); len(got) != 0 {
		t.Errorf("a container that exited by itself was stopped: %q", got)
	}
	if logs := s.logged(); len(logs) != 0 {
		t.Errorf("logged %q for a container nothing stopped", logs)
	}
}

// A context that ended before the client started leaves no container, so
// nothing is stopped and no name is used.
func TestARunCancelledBeforeItStartsStopsNothing(t *testing.T) {
	s := newStubRuntime(t, 0, false)
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if _, err := s.rt.Run(ctx, RunSpec{Image: "stub:1"}); err == nil {
		t.Fatal("Run with a cancelled context returned no error")
	}
	if got := s.calls(t); !slices.Equal(got, []string{""}) {
		t.Errorf("the runtime was called: %q", got)
	}
	if logs := s.logged(); len(logs) != 0 {
		t.Errorf("logged %q for a run that never started", logs)
	}
}

// When neither call can establish the container stopped, the line and the
// deadline message say it may still be running, and name the label that finds
// it. Saying "killed" there would be the claim #584 found false.
func TestAStopThatCannotBeConfirmedSaysSo(t *testing.T) {
	s := newStubRuntime(t, 0, true)
	out, err := s.rt.Run(context.Background(), RunSpec{Image: "stub:1", Timeout: 500 * time.Millisecond})
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	name := s.started(t)
	label := LabelRun + "=0123456789abcdef"
	logs := s.logged()
	if len(logs) != 1 || !strings.Contains(logs[0], "container "+name+" may still be running") ||
		!strings.Contains(logs[0], "Cannot connect") || !strings.Contains(logs[0], label) {
		t.Errorf("logged %q, want one line saying %s may still be running, why, and %s", logs, name, label)
	}
	if strings.Contains(out.Stderr, "was killed") || !strings.Contains(out.Stderr, "may still be running") {
		t.Errorf("Stderr = %q, want it to say the container may still be running", out.Stderr)
	}
}

// Every container is started under its own name, which is what a stop names,
// and with the run's label when the runtime has one.
func TestEveryContainerIsNamedAndLabelledWithItsRun(t *testing.T) {
	labelled := strings.Join((&CLIRuntime{Binary: "docker", RunID: "abc"}).args(RunSpec{Image: "i:1"}, "burnin-x"), " ")
	if !strings.HasPrefix(labelled, "run --rm --name burnin-x --label "+LabelRun+"=abc ") {
		t.Errorf("args = %s, want the name and the run label", labelled)
	}
	plain := strings.Join((&CLIRuntime{Binary: "docker"}).args(RunSpec{Image: "i:1"}, "burnin-x"), " ")
	if strings.Contains(plain, "--label") || !strings.Contains(plain, "--name burnin-x") {
		t.Errorf("args = %s, want the name and no label for a runtime with no run id", plain)
	}
	if a, b := containerName(), containerName(); a == b || !strings.HasPrefix(a, "burnin-") {
		t.Errorf("containerName gave %q and %q, want two distinct burnin- names", a, b)
	}
}
