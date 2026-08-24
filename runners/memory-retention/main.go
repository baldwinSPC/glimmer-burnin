// Command burnin-memory-retention is the entrypoint of the runner image for
// the "memory-retention" TestKind: it writes a pattern into a region of host
// memory, holds it UNTOUCHED for a duration, then reads it back — a
// memtest86+-shaped bit-fade test, not memtest86+ itself (which needs a
// reboot and cannot run inside either of this project's container
// dispatchers; see docs/vendors/nvidia.md).
//
// SPDX-License-Identifier: Apache-2.0
// Copyright the Glimmer authors.
//
// This file is original work licensed under Apache-2.0 and uses no
// third-party source or tool.
//
// WHAT THIS CATCHES THAT memory-stress DOES NOT
// -----------------------------------------------
// memory-stress (stressapptest) pounds a region with continuous read/write
// traffic and catches faults that show up UNDER ACCESS. A weak DRAM cell that
// loses its charge over time — because DRAM refresh is marginal, because a
// row is disturbed by its neighbours, because the controller's timing is
// wrong at the margin — never gets the chance to fail that way: continuous
// access refreshes every cell as a side effect of testing it. This kind holds
// a pattern UNTOUCHED for the bulk of its budget instead, which is the one
// thing continuous-access testing structurally cannot do.
//
// Two patterns are used, in order: a value and its bitwise complement over
// the SAME cells (0xFF then 0x00) — memtest86+'s own "bit fade, 2 patterns"
// shape. A cell stuck at 1 is invisible to a pattern that expects 1 there;
// testing the complement is what catches it.
//
// Output contract (pkg/runner):
//
//	exit 0  MEMORY_RETENTION_PASS            every pattern held clean
//	exit 1  MEMORY_RETENTION_FAIL: <reason>  the memory returned wrong data
//	exit 2  MEMORY_RETENTION_SKIP: <reason>  the test does not apply to this node
//	exit 3  MEMORY_RETENTION_ERROR: <reason> the runner could not judge the memory
//
// Metrics are key=value lines on stdout and are always printed before the
// decision, so a failing, interrupted or erroring run still yields its
// evidence.
package main

import (
	"fmt"
	"io"
	"os"
	"os/signal"
	"sync"
	"syscall"
	"time"
)

const (
	exitPass  = 0
	exitFail  = 1
	exitSkip  = 2
	exitError = 3
)

const (
	markerPass  = "MEMORY_RETENTION_PASS"
	markerFail  = "MEMORY_RETENTION_FAIL"
	markerSkip  = "MEMORY_RETENTION_SKIP"
	markerError = "MEMORY_RETENTION_ERROR"
)

// productionRoot is where the real /proc and /sys are. Spelled out rather
// than left empty for the same reason memory-stress's own productionRoot is:
// filepath.Join("", "proc", "meminfo") is the RELATIVE "proc/meminfo", which
// only resolves correctly for as long as nothing gives this process a working
// directory other than "/". The seam stays testable: tests inject a temp dir.
const productionRoot = "/"

func main() {
	code, reason := execute(os.Stdout, os.Stderr, os.LookupEnv, productionRoot)
	fmt.Fprintln(os.Stdout, marker(code, reason))
	os.Exit(code)
}

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
// getenv and root are parameters rather than package-level lookups so config
// resolution and sizing can be exercised in a unit test against a fake
// /proc — the same seam memory-stress uses. The pattern loop itself
// (allocation, fill, hold, verify) is exercised in its own unit tests against
// small in-memory buffers; nothing here re-tests that arithmetic.
func execute(stdout, stderr io.Writer, getenv getenvFunc, root string) (int, string) {
	em := &emitter{w: stdout}

	cfg, err := loadConfig(getenv)
	if err != nil {
		return exitError, err.Error()
	}
	em.int("duration_requested_s", int64(cfg.durationSeconds))

	sys, err := readSysInfo(root)
	if err != nil {
		return exitError, err.Error()
	}

	p, err := planRun(cfg, sys)
	if err != nil {
		var re *runnerError
		if as(err, &re) {
			return re.code, re.msg
		}
		return exitError, err.Error()
	}

	em.int("region_mb", p.regionMB)
	em.int("patterns_planned", int64(len(patternBytes)))
	em.int("hold_seconds", int64(p.holdSeconds))
	sizing := "auto-sized"
	if p.explicitRegion {
		sizing = "sized by BURNIN_RETENTION_MB"
	}
	holdSource := "derived from BURNIN_DURATION_SECONDS"
	if p.explicitHold {
		holdSource = "set by BURNIN_RETENTION_HOLD_SECONDS"
	}
	fmt.Fprintf(stderr,
		"memory-retention: %s — testing %d MiB for %ds per pattern (%s), %d pattern(s), of a %ds budget; "+
			"cgroup limit %d MiB, available %d MiB\n",
		sizing, p.regionMB, p.holdSeconds, holdSource, len(patternBytes), cfg.durationSeconds,
		sys.limitBytes/mib, sys.memAvailableBytes/mib)

	buf := make([]byte, p.regionMB*mib)

	locked, lockErr := lockMemory(buf)
	em.text("retention_memory_locked", boolStr(locked))
	if lockErr != nil {
		fmt.Fprintf(stderr, "memory-retention: mlock failed, continuing best-effort: %v\n", lockErr)
	}

	stop, cleanup := newStopSignal(cfg.durationSeconds)
	defer cleanup()

	res, runErr := runTest(buf, p, stop, em, func(f string, a ...any) { fmt.Fprintf(stderr, "memory-retention: "+f+"\n", a...) })

	em.int("retention_patterns_completed", int64(res.patternsCompleted))
	em.int("retention_bit_flips", res.bitFlips)
	em.int("retention_bytes_flipped", res.bytesFlipped)
	if res.firstOffsetValid {
		em.int("retention_first_flip_offset", res.firstOffset)
		em.text("retention_first_flip_pattern", fmt.Sprintf("0x%02X", res.firstOffsetPattern))
	}

	return decide(res, runErr)
}

// decide turns the accumulated evidence into a verdict.
//
// A bit flip is unambiguous: this runner wrote every byte itself and held the
// region untouched, so any byte reading back wrong IS memory corruption,
// found regardless of whether the whole plan finished — the same
// record-before-kill reasoning this project's soak engine applies to a
// miscompare. Absence of a flip is NOT the same fact: it only means "clean so
// far", and is a verdict only once every planned pattern has actually been
// checked. An interruption with nothing found yet is unjudged, not clean.
func decide(res execResult, runErr error) (int, string) {
	if res.bitFlips > 0 {
		return exitFail, fmt.Sprintf(
			"%d bit(s) across %d byte(s) read back wrong after being held untouched — the first at "+
				"offset %d against pattern 0x%02X — this is memory corruption, not a transient",
			res.bitFlips, res.bytesFlipped, res.firstOffset, res.firstOffsetPattern)
	}
	if runErr != nil {
		return exitError, fmt.Sprintf(
			"interrupted after %d of %d pattern(s) completed clean, before the test could finish "+
				"looking: %v", res.patternsCompleted, len(patternBytes), runErr)
	}
	if res.patternsCompleted < len(patternBytes) {
		// Should not be reachable without runErr set, but decide() must never
		// call a plan it did not finish clean unless it can say why.
		return exitError, fmt.Sprintf(
			"only %d of %d pattern(s) completed; the remainder was not judged",
			res.patternsCompleted, len(patternBytes))
	}
	return exitPass, ""
}

// as is local rather than errors.As, kept trivially small so this file's only
// import beyond the standard fmt/os/sync/syscall/time set is what it already
// has — a *runnerError is the only error type this package ever wraps this
// way, so a type switch is exactly as capable and pulls in nothing new.
func as(err error, target **runnerError) bool {
	re, ok := err.(*runnerError)
	if !ok {
		return false
	}
	*target = re
	return true
}

// emitter writes the key=value metric lines that are this runner's report.
//
// Unlike memory-stress's own emitter, this one needs no mutex: nothing here
// runs a background goroutine that also writes metrics (the stop signal's
// goroutine only ever sets an internal reason, never touches em), so every
// call is already on the single execution path.
type emitter struct{ w io.Writer }

func (e *emitter) metric(name, value string) { fmt.Fprintf(e.w, "%s=%s\n", name, value) }

func (e *emitter) text(name, value string) { e.metric(name, value) }

func (e *emitter) int(name string, value int64) { e.metric(name, fmt.Sprintf("%d", value)) }

func boolStr(b bool) string {
	if b {
		return "true"
	}
	return "false"
}

// lockMemory best-effort mlocks buf so the pages under test cannot be
// reclaimed or swapped out during the hold — a page round-tripped through
// swap is not "held untouched in DRAM" any more, which would let a real
// retention fault hide behind a false pass, or a swap round-trip masquerade
// as one. It is deliberately non-fatal: mlock needs CAP_IPC_LOCK or a raised
// RLIMIT_MEMLOCK that most clusters do not grant a burn-in pod by default —
// the same posture this project already accepts for nccl's own memlock — and
// refusing to run without it would make this kind unusable anywhere that
// grant is not arranged. memory_locked=false is reported honestly instead of
// silently passed over.
func lockMemory(buf []byte) (bool, error) {
	if len(buf) == 0 {
		return false, nil
	}
	if err := syscall.Mlock(buf); err != nil {
		return false, err
	}
	return true, nil
}

// newStopSignal wires SIGTERM/SIGINT and a hard deadline into one poll-based
// stop check, mirroring memory-stress's own kill/reaped idiom: a single
// mutex-guarded reason, set exactly once, so every caller of stop() — the
// pattern loop's hold-sleep ticks and its between-pattern check — observes
// the same answer without racing to set it twice. hardCap sits inside the
// pod's own activeDeadlineSeconds grace (duration + 120s) so THIS process
// ends the run on its own terms and the metrics gathered so far still reach
// stdout, rather than losing them to a kubelet SIGKILL.
//
// Poll-based rather than a channel the caller selects on: runTest's hold is
// naturally a tick loop already (it has to re-check for a signal at least
// once a second to be responsive), and a plain function is trivial to fake in
// a test — no goroutine, no channel plumbing to synchronise with.
func newStopSignal(durationSeconds int) (stop func() (string, bool), cleanup func()) {
	var mu sync.Mutex
	reason := ""
	stopped := make(chan struct{})
	set := func(why string) {
		mu.Lock()
		defer mu.Unlock()
		if reason == "" {
			reason = why
			close(stopped)
		}
	}
	read := func() (string, bool) {
		select {
		case <-stopped:
			mu.Lock()
			defer mu.Unlock()
			return reason, true
		default:
			return "", false
		}
	}

	sigs := make(chan os.Signal, 1)
	signal.Notify(sigs, syscall.SIGTERM, syscall.SIGINT)
	done := make(chan struct{})
	go func() {
		select {
		case s := <-sigs:
			// Deliberately not a clean pass: a signal means the plan did not
			// run to completion, and an entrypoint that treated it as success
			// would turn a deadline kill into a passing verdict for host
			// memory nobody actually finished checking.
			set("received " + s.String() + " before the test finished")
		case <-done:
		}
	}()

	hardCap := time.Duration(durationSeconds)*time.Second + hardCapGraceSeconds*time.Second
	capTimer := time.AfterFunc(hardCap, func() {
		set(fmt.Sprintf("still running after %s and was stopped by its own deadline", hardCap))
	})

	return read, func() {
		close(done)
		signal.Stop(sigs)
		capTimer.Stop()
	}
}
