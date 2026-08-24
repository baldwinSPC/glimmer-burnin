// Command xpu-diag is the runner for the "xpu-diag" TestKind.
//
// UNVERIFIED (#172). It wraps Intel XPU Manager's `xpu-smi diag -j` and
// translates the result into the burn-in runner contract, the same role
// dcgm-diag fills for NVIDIA — but it has never executed against a real Intel
// Data Center GPU, because nobody on this project has one. See
// docs/dev/spike-intel-xpu-manager.md and this file's own comments for what is
// and is not established. Do not point a default image at this kind's
// BuiltInKinds entry; pkg/runnerimages deliberately carries none.
//
//	0  the diagnostic ran and every component reported PASS
//	1  the diagnostic ran and at least one component reported FAIL — a
//	   verdict about the hardware
//	3  anything else: xpu-smi could not be run at all, a component never
//	   reached a PASS/FAIL result, or the JSON could not be parsed — the
//	   hardware is UNJUDGED
//
// There is deliberately no exit 2 (Skip) path yet. XPU Manager's own component
// result enum (xpum_diag_task_result_t) has no distinct "not applicable to
// this hardware" value the way DCGM's diagnostic does, so inventing one here
// would be this runner declaring a distinction the vendor's API does not
// offer — see decide.go's decide().
//
// Exit 3 is never used for a hardware judgement and exit 1 is never used for a
// runner problem, same as every other runner in this project.
//
// Every metric is printed BEFORE the marker on every path, so a failing or
// erroring run still yields its evidence. Prose goes to stderr; stdout carries
// only key=value lines and the final marker.
package main

import (
	"bytes"
	"context"
	"fmt"
	"os"
	"os/exec"
	"strconv"
	"time"
)

const (
	exitPass  = 0
	exitFail  = 1
	exitError = 3
)

const (
	markerPass  = "XPU_DIAG_PASS"
	markerFail  = "XPU_DIAG_FAIL"
	markerError = "XPU_DIAG_ERROR"
)

func main() {
	os.Exit(run(os.Stdout))
}

func run(stdout *os.File) int {
	// The absolute path, not a bare name on PATH: the final image is a
	// minimal distroless base with no guarantee /usr/local/bin is on it, and
	// this is the same reason memory-bw's wrapper takes nvbandwidth's path as
	// a build-time constant rather than trusting PATH resolution.
	bin := envOr("BURNIN_XPU_SMI_BIN", "/usr/local/bin/xpu-smi")
	deviceID := envOr("BURNIN_XPU_DEVICE_ID", "0")
	level := envOr("BURNIN_XPU_DIAG_LEVEL", "1")
	attempt := envInt("BURNIN_ATTEMPT", 1)

	// A soft deadline, not a claim about how long any given level actually
	// takes. dcgm-diag's README states measured per-level budgets from real
	// DCGM runs; this runner has no equivalent measurement to state, so it
	// only bounds worst-case runtime rather than pretending to size the level
	// to the duration the way dcgm-diag does. See the spike report.
	deadline := time.Duration(envInt("BURNIN_DURATION_SECONDS", 600)) * time.Second

	metric("diag_level", level)
	metric("device_id", deviceID)
	metric("attempt", strconv.Itoa(attempt))

	ctx, cancel := context.WithTimeout(context.Background(), deadline)
	defer cancel()

	cmd := exec.CommandContext(ctx, bin, "diag", "-d", deviceID, "-l", level, "-j")
	var out, errOut bytes.Buffer
	cmd.Stdout = &out
	cmd.Stderr = &errOut

	start := time.Now()
	runErr := cmd.Run()
	metric("elapsed_s", strconv.FormatFloat(time.Since(start).Seconds(), 'f', 2, 64))

	if ctx.Err() == context.DeadlineExceeded {
		return finish(stdout, exitError, markerError, fmt.Sprintf("timed out waiting for xpu-smi diag after %s", deadline))
	}

	if out.Len() == 0 {
		logf("xpu-smi produced no stdout; stderr: %s", errOut.String())
		reason := "xpu-smi diag produced no output"
		if runErr != nil {
			reason = "xpu-smi diag: " + runErr.Error()
		}
		return finish(stdout, exitError, markerError, reason)
	}

	d := decide(out.Bytes())
	metric("tests_run", strconv.Itoa(d.Total))
	metric("tests_passed", strconv.Itoa(d.Passed))
	metric("tests_failed", strconv.Itoa(d.Failed))
	metric("tests_incomplete", strconv.Itoa(d.Incomplete))

	switch d.Verdict {
	case verdictPass:
		return finish(stdout, exitPass, markerPass, d.Reason)
	case verdictFail:
		return finish(stdout, exitFail, markerFail, d.Reason)
	default:
		return finish(stdout, exitError, markerError, d.Reason)
	}
}

// finish prints the marker line to stdout and returns the exit code, mirroring
// dcgm-diag's report.go finish(): "MARKER: reason", the only line format
// pkg/runner's marker scanners are guaranteed to recognise.
func finish(stdout *os.File, code int, marker, reason string) int {
	if reason == "" {
		fmt.Fprintln(stdout, marker)
	} else {
		fmt.Fprintf(stdout, "%s: %s\n", marker, sanitize(reason))
	}
	return code
}
