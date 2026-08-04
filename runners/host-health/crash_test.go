package main

import (
	"bytes"
	"context"
	"strings"
	"testing"

	"github.com/baldwinSPC/glimmer-burnin/pkg/contract"
	"github.com/baldwinSPC/glimmer-burnin/pkg/runner"
)

// What a crashed host-health leaves behind — issue #112.
//
// The observed failure was a host-health test settling Skipped on a real node,
// on a runner that cannot skip. #105 fixed the mapping (an exit 2 with no _SKIP
// declaration is an Error), which means a recurrence is REPORTED correctly. It
// did not make a recurrence DIAGNOSABLE, and that is what these tests hold: at
// the time, every metric was buffered in the emitter and written in one block at
// the very end, so a runner that died produced stdout with nothing in it at all —
// no metrics, no verdict, nothing naming the probe it was inside.
//
// The two halves are tested separately because they fail differently. A panic is
// recoverable and the runner converts it into its own contract; a runtime fatal
// error and the kernel OOM killer are not recoverable by anything, and the only
// thing that can survive them is output already on the wire.

// A recovered panic is an Error that explains itself, not a bare stack frame.
func TestPanicIsReportedAsAnExplainedError(t *testing.T) {
	f := newE2EFixture(t, healthyGPU(), "0")
	f.cfg.run = func(context.Context, string, []string) (string, error) {
		panic("nvidia-smi probe exploded")
	}

	var buf bytes.Buffer
	code := run(f.cfg, &buf)
	out := buf.String()

	if code != exitError {
		t.Fatalf("exit = %d, want %d — a crashed runner leaves the node unjudged\n%s", code, exitError, out)
	}
	// The camouflage this whole issue is about: exit 2 is the skip code, and a
	// runner that crashes into it reports hardware it never looked at as out of
	// scope.
	if code == exitNotApplicable {
		t.Fatal("a panic must never leave the process exiting 2, the skip code")
	}

	res := runner.Parse("host-health", out, code)
	if res.Verdict != runner.VerdictError {
		t.Errorf("verdict = %q, want Error", res.Verdict)
	}
	if res.UndeclaredSkip {
		t.Error("the runner exited 3 deliberately, so nothing should look like an undeclared skip")
	}

	// The message is the operator's stored record, and it must say the runner
	// crashed rather than quote the last line of a stack trace.
	if !strings.Contains(res.Message, "panicked") {
		t.Errorf("parsed message does not say the runner crashed: %q", res.Message)
	}
	if !strings.Contains(res.Message, stageProbeFirst) {
		t.Errorf("parsed message does not name the stage it died in: %q", res.Message)
	}

	// Recorded as a metric as well, because a metric cannot be overwritten by a
	// later stack frame the way Message can.
	if got := res.Metrics["hostHealthPanic"]; !strings.Contains(got, "nvidia-smi probe exploded") {
		t.Errorf("hostHealthPanic = %q, want the panic value", got)
	}
	if got := res.Metrics["hostHealthStage"]; got != stageProbeFirst {
		t.Errorf("hostHealthStage = %q, want %q", got, stageProbeFirst)
	}

	// The evidence gathered BEFORE the crash survives, which the old
	// write-everything-at-the-end shape threw away.
	if _, ok := res.Metrics["hostHealthVersion"]; !ok {
		t.Errorf("metrics collected before the panic were lost:\n%s", out)
	}
	if len(res.InvalidNames) != 0 {
		t.Errorf("the stack trace was parsed as metrics: %v", res.InvalidNames)
	}
}

// The half no recover can reach: a runtime fatal error or the kernel OOM killer.
//
// Neither can be intercepted, so the only thing that can survive one is output
// already written. This asserts that the breadcrumb for a stage is on the wire
// BEFORE that stage does any work — the output captured at the instant the probe
// starts is exactly what a SIGKILL at that instant would leave in the pod log.
func TestStageIsOnTheWireBeforeTheStageRuns(t *testing.T) {
	f := newE2EFixture(t, healthyGPU(), "0")
	inner := f.cfg.run

	var buf bytes.Buffer
	var atProbe string
	var probed bool
	f.cfg.run = func(ctx context.Context, name string, args []string) (string, error) {
		if !probed {
			// Captured BEFORE the probe does anything, so this is byte for byte
			// what a SIGKILL at this instant would leave in the pod's log.
			probed, atProbe = true, buf.String()
		}
		return inner(ctx, name, args)
	}
	if code := run(f.cfg, &buf); code != exitPass {
		t.Fatalf("exit = %d, want a clean run\n%s", code, buf.String())
	}
	// Tracked separately from the captured text: an EMPTY capture is the
	// interesting failure (nothing was on the wire yet) and must not be
	// indistinguishable from the probe never having run at all.
	if !probed {
		t.Fatal("the GPU probe never ran, so this test proves nothing")
	}

	// 137 is what the kubelet records for a SIGKILLed container.
	res := runner.Parse("host-health", atProbe, 137)
	if got := res.Metrics["hostHealthStage"]; got != stageProbeFirst {
		t.Errorf("a kill during the first probe records hostHealthStage=%q, want %q\ncaptured:\n%s",
			got, stageProbeFirst, atProbe)
	}
	if res.Verdict != runner.VerdictError {
		t.Errorf("verdict = %q, want Error for a killed runner", res.Verdict)
	}
}

// A completed run always ends at the same stage. Stated so that a reader of a
// stored result knows what the ABSENCE of "done" means: the runner did not get
// to the end, whatever the exit code says.
func TestCompletedRunReachesDone(t *testing.T) {
	f := newE2EFixture(t, healthyGPU(), "0")
	var buf bytes.Buffer
	code := run(f.cfg, &buf)

	res := runner.Parse("host-health", buf.String(), code)
	if got := res.Metrics["hostHealthStage"]; got != stageDone {
		t.Errorf("hostHealthStage = %q, want %q\n%s", got, stageDone, buf.String())
	}
	if _, crashed := res.Metrics["hostHealthPanic"]; crashed {
		t.Error("a clean run must not report a panic")
	}
}

// The stage marker must never be mistaken for a metric a profile could gate on.
// It is a label, it is always "done" on any run that finished, and a gate on it
// would fail closed on every node forever.
func TestStageAndPanicAreEvidenceOnly(t *testing.T) {
	for _, name := range []string{"hostHealthStage", "hostHealthPanic"} {
		if contract.SafeToThresholdOn(name) {
			t.Errorf("%s must be registered as Evidence so the threshold linter refuses a gate on it", name)
		}
	}
}
