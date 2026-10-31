//go:build unix

package main

import (
	"bytes"
	"os"
	"os/exec"
	"path/filepath"
	"slices"
	"strings"
	"syscall"
	"testing"
	"time"
)

// TestHelperProcessBurnin is the burnin CLI when BURNIN_HELPER_ARGS is set, so
// a test can signal a real `burnin run` process. It does nothing otherwise.
func TestHelperProcessBurnin(t *testing.T) {
	args := os.Getenv("BURNIN_HELPER_ARGS")
	if args == "" {
		return
	}
	os.Args = append([]string{"burnin"}, strings.Split(args, "\x1f")...)
	main()
	os.Exit(0)
}

// #584, as a caller stops a run: SIGTERM to the CLI's process group, then
// SIGKILL to the group after five seconds. The runtime's client relays the
// SIGTERM to a PID 1 that drops it, so before the fix the CLI exited and its
// container kept the load running. After it, the CLI kills that container by
// name before it exits, inside the five seconds, and a container it did not
// start is still running.
func TestASignalledRunStopsItsContainerInsideTheCallersGrace(t *testing.T) {
	binDir, state := t.TempDir(), t.TempDir()
	src, err := os.ReadFile(filepath.Join("..", "..", "pkg", "localrun", "testdata", "stubruntime.sh"))
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(binDir, "docker"), src, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(state, "running.someone-elses"), nil, 0o644); err != nil {
		t.Fatal(err)
	}

	cli := exec.Command(os.Args[0], "-test.run=^TestHelperProcessBurnin$")
	cli.Env = append(os.Environ(),
		"BURNIN_HELPER_ARGS="+strings.Join([]string{"run", "-f", writeSuite(t, oneProfile), "--node", "n1", "--runtime", "docker"}, "\x1f"),
		"PATH="+binDir+string(os.PathListSeparator)+os.Getenv("PATH"),
		"STUB_STATE="+state,
	)
	var stderr bytes.Buffer
	cli.Stderr = &stderr
	cli.SysProcAttr = &syscall.SysProcAttr{Setpgid: true}
	if err := cli.Start(); err != nil {
		t.Fatal(err)
	}
	exited := make(chan struct{})
	go func() { _ = cli.Wait(); close(exited) }()

	name := ""
	for deadline := time.Now().Add(20 * time.Second); name == "" && time.Now().Before(deadline); {
		if m, _ := filepath.Glob(filepath.Join(state, "running.burnin-*")); len(m) > 0 {
			name = strings.TrimPrefix(filepath.Base(m[0]), "running.")
			break
		}
		time.Sleep(20 * time.Millisecond)
	}
	if name == "" {
		_ = syscall.Kill(-cli.Process.Pid, syscall.SIGKILL)
		<-exited
		t.Fatalf("the run never started its container; stderr:\n%s", stderr.String())
	}

	_ = syscall.Kill(-cli.Process.Pid, syscall.SIGTERM)
	select {
	case <-exited:
	case <-time.After(5 * time.Second):
		_ = syscall.Kill(-cli.Process.Pid, syscall.SIGKILL)
		<-exited
		t.Errorf("the CLI was still running five seconds after SIGTERM and was SIGKILLed; stderr:\n%s", stderr.String())
	}

	b, _ := os.ReadFile(filepath.Join(state, "argv"))
	var stops []string
	for _, c := range strings.Split(strings.TrimSpace(string(b)), "\n") {
		if strings.HasPrefix(c, "run ") && !strings.Contains(c, "--name "+name+" --label burnin.glimmer.ai/run=") {
			t.Errorf("the run call did not carry its name and the run label: %s", c)
		}
		if strings.HasPrefix(c, "kill ") || strings.HasPrefix(c, "rm ") {
			stops = append(stops, c)
		}
	}
	if want := []string{"kill " + name, "rm --force " + name}; !slices.Equal(stops, want) {
		t.Errorf("stop calls = %q, want %q", stops, want)
	}
	if _, err := os.Stat(filepath.Join(state, "running."+name)); err == nil {
		t.Errorf("container %s is still running after the CLI was stopped", name)
	}
	if _, err := os.Stat(filepath.Join(state, "running.someone-elses")); err != nil {
		t.Errorf("a container the CLI did not start was stopped: %v", err)
	}
	if !strings.Contains(stderr.String(), "killed container "+name) {
		t.Errorf("stderr does not say the container was killed:\n%s", stderr.String())
	}
	t.Logf("the CLI's stderr:\n%s", stderr.String())
}
