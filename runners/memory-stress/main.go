// Command burnin-memory-stress is the entrypoint of the runner image for the
// "memory-stress" TestKind: it stresses HOST memory with stressapptest and
// reports what the DIMMs did.
//
// SPDX-License-Identifier: Apache-2.0
// Copyright the Glimmer authors.
//
// This file is original work licensed under Apache-2.0. It executes
// stressapptest (also Apache-2.0), which is built from source in the image and
// shipped alongside this binary; see the README for the licence of every
// component.
//
// Output contract (pkg/runner):
//
//	exit 0  STRESSAPPTEST_PASS            the memory was exercised and was clean
//	exit 1  STRESSAPPTEST_FAIL: <reason>  the memory returned wrong data
//	exit 2  STRESSAPPTEST_SKIP: <reason>  the test does not apply to this node
//	exit 3  STRESSAPPTEST_ERROR: <reason> the runner could not judge the memory
//
// Metrics are key=value lines on stdout and are ALWAYS printed before the
// decision, so a failing, interrupted or erroring run still yields its evidence.
package main

import (
	"bufio"
	"errors"
	"fmt"
	"io"
	"os"
	"os/exec"
	"os/signal"
	"strconv"
	"sync"
	"syscall"
	"time"
)

// Exit codes are the runner contract, not an internal detail: 0/1/2 are Pass,
// Fail and Skip, and anything else is an Error — hardware left unjudged.
const (
	exitPass  = 0
	exitFail  = 1
	exitSkip  = 2
	exitError = 3
)

const (
	markerPass  = "STRESSAPPTEST_PASS"
	markerFail  = "STRESSAPPTEST_FAIL"
	markerSkip  = "STRESSAPPTEST_SKIP"
	markerError = "STRESSAPPTEST_ERROR"
)

const (
	// heartbeatSeconds is how often the elapsed time is re-emitted while the
	// test runs. The operator can fetch a running pod's logs to checkpoint its
	// metrics, and the parser is last-occurrence-wins, so a periodic line makes
	// an interrupted multi-hour soak report how far it actually got instead of
	// reporting nothing at all.
	heartbeatSeconds = 60
	// killGraceSeconds is how long a terminated stressapptest gets to exit
	// before the process group is killed outright.
	killGraceSeconds = 10
)

// stressapptestRef is stamped at build time (-X main.stressapptestRef=...) with
// the exact upstream ref the image was built from, so a stored verdict can be
// traced to the tool that produced it.
var stressapptestRef = "unknown"

// hardCapGrace is how long past BURNIN_DURATION_SECONDS a run may live before
// the wrapper kills it. A variable rather than a constant only so the test can
// shrink it: the kill path is the one that decides whether an interrupted run
// can masquerade as a passing one, and it is worth exercising.
var hardCapGrace = time.Duration(hardCapGraceSeconds) * time.Second

// productionRoot is where the real /proc and /sys are. It is spelled out rather
// than left empty because filepath.Join("", "proc", "meminfo") is the RELATIVE
// "proc/meminfo": that resolved correctly only for as long as the image set no
// WORKDIR and the process therefore happened to start in /. A WORKDIR, a
// command:/workingDir: in the pod spec, or a wrapper entrypoint would each have
// turned this runner into exit 3 on every node at once — an unjudged fleet
// rather than a visible crash. The seam itself stays: tests inject a temp dir.
const productionRoot = "/"

func main() {
	code, reason := execute(os.Stdout, os.Stderr, os.LookupEnv, productionRoot)
	fmt.Fprintln(os.Stdout, marker(code, reason))
	os.Exit(code)
}

// marker is the last line of stdout: the human-readable outcome, and what the
// parser records as the result's Message.
func marker(code int, reason string) string {
	switch code {
	case exitPass:
		return markerPass
	case exitFail:
		return markerFail + ": " + reason
	case exitSkip:
		return markerSkip + ": " + reason
	default:
		return markerError + ": " + reason
	}
}

// execute runs one test and returns its exit code and the reason for it.
//
// getenv and root are parameters rather than package-level lookups so the whole
// path — sizing, execution, metric emission and the decision — can be exercised
// in a unit test against a fake /proc and a fake stressapptest. The metric key
// names below are a contract with the operator's parser; nothing but running
// this end to end proves they still match.
func execute(stdout, stderr io.Writer, getenv getenvFunc, root string) (int, string) {
	em := &emitter{w: stdout}

	cfg, err := loadConfig(getenv)
	if err != nil {
		return exitError, err.Error()
	}

	// Emitted before anything can go wrong, so even a skipped or erroring run
	// records which tool was asked to do what.
	em.text("tool", "stressapptest")
	em.text("tool_version", stressapptestRef)
	em.int("duration_requested_s", int64(cfg.durationSeconds))

	sys, err := readSysInfo(root)
	if err != nil {
		return exitError, err.Error()
	}
	p, err := planRun(cfg, sys)
	if err != nil {
		var re *runnerError
		if errors.As(err, &re) {
			return re.code, re.msg
		}
		return exitError, err.Error()
	}

	em.int("threads", int64(p.threads))
	if p.coveredPct > 0 {
		em.float("memory_covered_pct", p.coveredPct)
	}

	sizing := "auto-sized"
	if p.explicitSize {
		sizing = "sized by BURNIN_MEMORY_MB"
	}
	fmt.Fprintf(stderr,
		"memory-stress: %s — testing %d MiB (%.1f%% of the node's %d MiB) with %d copy threads for %ds of a %ds budget; cgroup limit %d MiB, available %d MiB\n",
		sizing, p.testMB, p.coveredPct, sys.memTotalBytes/mib, p.threads, p.satSeconds,
		cfg.durationSeconds, sys.limitBytes/mib, sys.memAvailableBytes/mib)

	res, wall, runErr := runStressapptest(cfg, p, em, stderr)

	// Every measurement first, decision second.
	em.float("elapsed_s", wall.Seconds())
	if res.toolSeconds > 0 {
		em.float("sat_run_s", res.toolSeconds)
	}
	if res.completed || res.sdcDetections > 0 {
		em.int("sdc_count", res.sdcDetections)
	}
	if res.completed {
		// Only reported when a completion summary was seen. A run that died
		// early has unknown counters, and printing hardware_incidents=0 for it
		// would let a threshold read "no errors" off a test that never
		// finished looking.
		em.int("hardware_incidents", res.hardwareIncidents)
		em.int("miscompares", res.miscompares)
		em.float("read_bandwidth_mbs", res.readMBs())
		em.float("write_bandwidth_mbs", res.writeMBs())
		em.text("sat_status", string(res.status))
	}

	return decide(res, runErr)
}

// runStressapptest runs the tool to completion and returns what it reported,
// the wall clock of the whole test body (allocation and pattern fill included),
// and an error if the run did not finish on its own terms.
func runStressapptest(cfg config, p plan, em *emitter, logTo io.Writer) (satResult, time.Duration, error) {
	var res satResult

	cmd := exec.Command(cfg.binary, p.args(cfg)...)
	// Its own process group, so a kill reaches every thread the tool spawned
	// rather than leaving one behind holding the region.
	cmd.SysProcAttr = &syscall.SysProcAttr{Setpgid: true}

	pr, pw, err := os.Pipe()
	if err != nil {
		return res, 0, fmt.Errorf("cannot create a pipe for stressapptest output: %w", err)
	}
	cmd.Stdout = pw
	cmd.Stderr = pw

	fmt.Fprintf(logTo, "memory-stress: exec %s %v\n", cfg.binary, p.args(cfg))
	start := time.Now()
	if err := cmd.Start(); err != nil {
		pw.Close()
		pr.Close()
		return res, 0, fmt.Errorf("cannot start %s: %w", cfg.binary, err)
	}
	// The parent's write end must go, or the reader below never sees EOF.
	pw.Close()

	// reaped is closed once cmd.Wait() has returned. After that the child's pid
	// may be recycled by the kernel, so nothing may signal it again — an
	// escalation timer that fired late would otherwise SIGKILL an unrelated
	// process group on the node.
	reaped := make(chan struct{})

	var mu sync.Mutex
	killedBecause := ""
	kill := func(why string) {
		mu.Lock()
		first := killedBecause == ""
		if first {
			killedBecause = why
		}
		mu.Unlock()
		if !first || cmd.Process == nil {
			return
		}
		// A negative pid addresses the process group set up by Setpgid, so the
		// signal reaches every thread the tool spawned rather than leaving one
		// behind holding the region.
		pid := cmd.Process.Pid
		_ = syscall.Kill(-pid, syscall.SIGTERM)
		select {
		case <-reaped:
		case <-time.After(killGraceSeconds * time.Second):
			_ = syscall.Kill(-pid, syscall.SIGKILL)
		}
	}

	// The pod's activeDeadlineSeconds is duration + 120s. Ending a wedged run
	// ourselves, inside that window, is what lets the metrics gathered so far
	// reach stdout — a kubelet deadline kill takes the log's last words with it.
	hardCap := time.Duration(cfg.durationSeconds)*time.Second + hardCapGrace
	capTimer := time.AfterFunc(hardCap, func() {
		kill(fmt.Sprintf("stressapptest was still running after %s and was killed", hardCap))
	})
	defer capTimer.Stop()

	sigs := make(chan os.Signal, 1)
	signal.Notify(sigs, syscall.SIGTERM, syscall.SIGINT)
	defer signal.Stop(sigs)

	stop := make(chan struct{})
	var wg sync.WaitGroup
	wg.Add(1)
	go func() {
		defer wg.Done()
		select {
		case s := <-sigs:
			// Deliberately NOT a clean exit 0. A signal means the test did not
			// finish, and an entrypoint that exits 0 on SIGTERM turns a
			// deadline kill into a passing verdict for untested hardware.
			kill("the runner received " + s.String() + " before the test finished")
		case <-stop:
		}
	}()
	wg.Add(1)
	go func() {
		defer wg.Done()
		ticker := time.NewTicker(heartbeatSeconds * time.Second)
		defer ticker.Stop()
		for {
			select {
			case t := <-ticker.C:
				em.float("elapsed_s", t.Sub(start).Seconds())
			case <-stop:
				return
			}
		}
	}()

	scanner := bufio.NewScanner(pr)
	scanner.Buffer(make([]byte, 0, 64*1024), 1024*1024)
	for scanner.Scan() {
		line := scanner.Text()
		res.feed(line)
		// Forwarded PREFIXED, never raw. The operator parses metrics out of the
		// pod log, which merges stdout and stderr, so an unprefixed tool line
		// could be mistaken for a measurement. "sat| " contains a space, which
		// the parser refuses to read as a metric key.
		fmt.Fprintf(logTo, "sat| %s\n", line)
	}
	scanErr := scanner.Err()
	waitErr := cmd.Wait()
	close(reaped)
	pr.Close()

	// Stop the helpers before the caller emits the final metrics, so a
	// heartbeat cannot land after — and, being last, overwrite — the real
	// elapsed time.
	close(stop)
	wg.Wait()
	wall := time.Since(start)

	mu.Lock()
	why := killedBecause
	mu.Unlock()
	switch {
	case why != "":
		return res, wall, errors.New(why)
	case scanErr != nil:
		return res, wall, fmt.Errorf("could not read stressapptest output: %w", scanErr)
	case waitErr != nil:
		// stressapptest exits 1 both when it finds errors and when it fails
		// procedurally, so this is not a verdict on its own — decide() weighs
		// the evidence first and only falls back to this message.
		return res, wall, fmt.Errorf("stressapptest exited with %v", waitErr)
	}
	return res, wall, nil
}

// emitter writes the key=value metric lines that are this runner's report.
//
// It is mutex-guarded because the heartbeat writes from its own goroutine, and
// two interleaved Fprintf calls would produce a line that parses as neither
// metric.
type emitter struct {
	mu sync.Mutex
	w  io.Writer
}

func (e *emitter) metric(name, value string) {
	e.mu.Lock()
	defer e.mu.Unlock()
	fmt.Fprintf(e.w, "%s=%s\n", name, value)
}

func (e *emitter) text(name, value string) { e.metric(name, value) }

func (e *emitter) int(name string, value int64) { e.metric(name, strconv.FormatInt(value, 10)) }

func (e *emitter) float(name string, value float64) {
	e.metric(name, strconv.FormatFloat(value, 'f', 2, 64))
}
