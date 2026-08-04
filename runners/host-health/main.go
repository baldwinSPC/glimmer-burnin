// Command host-health is the entrypoint of the runner image for the
// "host-health" TestKind: a passive read of the host- and driver-side fault
// counters that move while every performance test still passes.
//
// It applies no load and measures no throughput. Its whole value is that a row
// being remapped, a link retraining, or a single Xid is a part on its way out
// long before any benchmark notices, so this runner is cheap enough to attach
// to every profile.
//
// Three rules govern everything below.
//
//  1. NEVER emit a zero that was not measured. A counter this runner could not
//     read is OMITTED, not reported as 0. pkg/verdict fails a threshold whose
//     metric is missing, so omission fails closed; a fabricated 0 would silently
//     satisfy acceptance and is the exact false-negative this project forbids.
//
//     There is a third case between those two, and it is not a loophole in the
//     rule but the reason the rule needs three states. Where the driver was
//     ASKED and answered "N/A" for every GPU — GB10 exposes no ECC to NVML at
//     all, since its unified LPDDR5X has on-die ECC only — the counter is
//     emitted as the reserved value `n/a` (see pkg/runner). That is the runner
//     declaring a property of the part, and it is the ONLY thing that can let a
//     profile's RequiredIfMeasurable gate be reported as not-evaluated rather
//     than failing a healthy node. A counter we could not read for any other
//     reason — no driver, a timeout, a missing column, some GPUs answering and
//     others not — is still OMITTED: "there is nothing here to measure" and "we
//     could not look" are different claims, and only the first is ours to make.
//
//  2. NEVER skip. Host health applies to every node, so exit 2 is unreachable
//     here (see exitNotApplicable). A node whose counters cannot be read is
//     UNJUDGED, not "not applicable": that is exit 3 (Error), which the operator
//     keeps distinct from Fail.
//
//     "Unreachable" is a claim about what run() RETURNS, and the PROCESS can
//     exit 2 without it: that is what the Go runtime does for an unrecovered
//     panic, and for every runtime fatal error — out of memory, concurrent map
//     write, stack exhaustion — which no deferred recover can intercept. This
//     runner is where that was caught (#103), precisely because it is the one
//     that claims never to skip; on a runner with a legitimate exit 2 the same
//     crash is indistinguishable from a real one. It is not fixed here and
//     cannot be: the fix is in pkg/runner, which honours exit 2 as Skip only
//     when the runner also PRINTS a _SKIP declaration. This runner prints none,
//     by design, so a crash of it is recorded Error and the hardware is left
//     unjudged rather than certified out of scope. Do not add a marker here.
//
//  3. Degrade per probe, not as a whole. Every probe is independent and optional;
//     the runner emits whatever subset of counters this host actually exposes.
//     That is also what keeps it vendor-neutral — the NVML probe contributing
//     nothing on a non-NVIDIA node is a normal outcome, not an error.
//
// The runner contract is exit code plus key=value lines on stdout. Metrics are
// printed BEFORE the verdict line, so a failing run still yields its evidence.
//
// Dependencies: the standard library ONLY. The image build compiles this
// directory as a throwaway module (the publish workflow's build context is the
// runner directory, which cannot see the repository's go.mod), so a non-stdlib
// import in a non-test file breaks the image build. Test files are free to
// import the repository's packages — they are removed before the image build,
// and metricnames_test.go uses pkg/contract and pkg/runner to prove that every
// key emitted here parses into a legal canonical metric name.
package main

import (
	"context"
	"fmt"
	"io"
	"os"
	"sort"
	"strconv"
	"strings"
	"time"
)

// Exit codes are the runner contract.
const (
	exitPass = 0
	exitFail = 1
	// exitNotApplicable is declared for completeness and is deliberately never
	// returned. Every node has host health; "we could not look" is exitError.
	exitNotApplicable = 2
	exitError         = 3
)

// emissionVersion identifies the shape of the metric set below. A consumer
// charting these counters over months needs to know when the set changed;
// bump it whenever a metric's MEANING changes (adding a new key does not).
const emissionVersion = "1"

// Defaults. The window is the interval over which event counters are differenced.
const (
	defaultWindowSeconds = 15
	// maxWindowSeconds caps BURNIN_DURATION_SECONDS. The reconciler defaults an
	// unset duration to 600s, and a passive counter read has no business
	// occupying ten minutes of a node's burn-in: this runner is meant to finish
	// in well under a minute so a profile can afford to run it everywhere.
	maxWindowSeconds = 30
	// maxTotalSeconds is the runner's own deadline. It exists so a wedged probe
	// (an unresponsive driver makes nvidia-smi hang, which is itself a symptom)
	// cannot turn into a pod killed at its activeDeadlineSeconds — that would
	// destroy the evidence and be reported as an infrastructure Error.
	maxTotalSeconds = 55
	// probeTimeoutSeconds bounds one external command.
	probeTimeoutSeconds = 8
	// closingReserveSeconds is held back from the observation window so the
	// second sample and the printing always fit inside maxTotalSeconds.
	closingReserveSeconds = 15
)

// gatedKeys are the counters this runner decides pass/fail on. They are exactly
// the five the host-health TestKind is defined by (see the alias table in
// pkg/runner), which is what makes the runner's own exit code agree with the
// profile thresholds it is written for (each `Equal 0`).
//
// Everything else this runner prints is EVIDENCE: real, worth storing, and
// available to a stricter profile as a threshold, but not something the runner
// condemns a node for on its own. Where that line falls is a policy decision,
// and policy belongs in the BurnInTest, not baked into an image a fleet pins.
var gatedKeys = []string{
	keyXidCount,
	keyECCErrors,
	keyRowsRemapped,
	keyPCIeReplayCount,
	keyNICLinkDown,
}

// Runner-side metric keys. These are the runner's own spelling; pkg/runner maps
// them to canonical names (xid_count -> xidEvents, and so on). They are
// constants so the gating list above cannot drift from what the probes emit.
const (
	keyXidCount        = "xid_count"
	keyECCErrors       = "ecc_errors"
	keyRowsRemapped    = "rows_remapped"
	keyPCIeReplayCount = "pcie_replay_count"
	keyNICLinkDown     = "nic_link_down"
)

// unmeasurableValue is the runner contract's reserved metric value: this
// hardware cannot produce this measurement at all. It must stay in step with
// pkg/runner.Unmeasurable, which is asserted by a test — the runner's non-test
// sources are standard-library-only, so it cannot import the constant.
const unmeasurableValue = "n/a"

func main() {
	os.Exit(run(configFromEnv(time.Now()), os.Stdout))
}

// config is everything the probes need. It is a struct rather than a pile of
// globals so the whole runner can be exercised in a unit test against a fake
// sysfs tree, a fake kernel log and a fake nvidia-smi.
type config struct {
	// window is the interval over which event counters are differenced.
	window time.Duration
	// deadline bounds the entire run. Zero means unbounded (tests).
	deadline time.Time

	kmsgPath     string
	kernLogPaths []string
	sysfsRoot    string
	nvidiaSMI    string
	probeTimeout time.Duration

	// run executes an external command. Injectable so tests never fork.
	run commandRunner
	// now and sleep are injectable for the same reason.
	now   func() time.Time
	sleep func(time.Duration)
}

func configFromEnv(start time.Time) config {
	cfg := config{
		window:       time.Duration(clampWindowSeconds(envInt("BURNIN_DURATION_SECONDS", 0))) * time.Second,
		deadline:     start.Add(maxTotalSeconds * time.Second),
		kmsgPath:     envStr("BURNIN_KMSG_PATH", "/dev/kmsg"),
		kernLogPaths: splitPaths(envStr("BURNIN_KERN_LOG_PATHS", "/var/log/kern.log:/var/log/messages:/var/log/syslog")),
		sysfsRoot:    envStr("BURNIN_SYSFS_ROOT", "/sys"),
		nvidiaSMI:    envStr("BURNIN_NVIDIA_SMI", "nvidia-smi"),
		probeTimeout: probeTimeoutSeconds * time.Second,
		run:          execRunner,
		now:          time.Now,
		sleep:        time.Sleep,
	}
	return cfg
}

// clampWindowSeconds turns the pod's BURNIN_DURATION_SECONDS into an
// observation window. The duration is honoured as an upper bound the caller
// asked for, never as a floor: this runner is passive, and a profile that asked
// for a ten-minute compute soak did not thereby ask to stare at sysfs for ten
// minutes.
func clampWindowSeconds(requested int) int {
	if requested <= 0 {
		return defaultWindowSeconds
	}
	if requested > maxWindowSeconds {
		return maxWindowSeconds
	}
	return requested
}

// run performs the whole test and returns the exit code. Output is written to w
// in contract order: every metric first, then the single verdict line.
func run(cfg config, w io.Writer) int {
	out := newEmitter()
	start := cfg.now()
	out.set("host_health_version", emissionVersion)
	// Reserve the position; the value is overwritten below with the window that
	// was actually observed.
	out.setInt("observation_window_s", int64(cfg.window/time.Second))

	// The kernel log is opened FIRST so the window it observes encloses every
	// other probe: a Xid logged while nvidia-smi is running must land inside
	// the window, not in the gap before it.
	klog := newKernelLogProbe(cfg)
	klog.baseline()

	gpu0 := probeGPU(cfg)
	aer0 := probeAER(cfg.sysfsRoot)
	nic0 := probeNIC(cfg.sysfsRoot)

	// The window that was actually observed, which is not always the one that
	// was asked for: the runner's own deadline can cut it short, and every
	// windowed counter below is a rate over THIS interval.
	out.setInt("observation_window_s", int64(cfg.sleepWindow().Round(time.Second)/time.Second))

	gpu1 := probeGPU(cfg)
	aer1 := probeAER(cfg.sysfsRoot)
	nic1 := probeNIC(cfg.sysfsRoot)
	klog.collect()

	klog.emit(out)
	emitGPU(out, gpu0, gpu1)
	emitAER(out, aer0, aer1)
	emitNIC(out, nic0, nic1)
	reconcilePCIeReplay(out, gpu0, gpu1, aer0, aer1)

	out.set("elapsed_s", strconv.FormatFloat(cfg.now().Sub(start).Seconds(), 'f', 1, 64))

	// The verdict is computed before anything is written, but written last: the
	// metrics are the evidence for it and must survive a truncated log.
	code, verdict := decide(out)
	out.set("node_ready", strconv.FormatBool(code == exitPass))

	out.writeTo(w)
	fmt.Fprintln(w, verdict)
	return code
}

// sleepWindow waits out the observation window without ever crossing the
// runner's own deadline, and reports how long it actually waited.
func (c config) sleepWindow() time.Duration {
	d := c.window
	if !c.deadline.IsZero() {
		if budget := c.deadline.Sub(c.now()) - closingReserveSeconds*time.Second; budget < d {
			d = budget
		}
	}
	if d <= 0 {
		return 0
	}
	c.sleep(d)
	return d
}

// decide applies the gate. It returns the exit code and the single verdict line.
func decide(out *emitter) (int, string) {
	var measured, unmeasurable, faults []string
	for _, k := range gatedKeys {
		raw, ok := out.get(k)
		if !ok {
			continue // not measured on this host — omitted, so it fails closed upstream
		}
		if raw == unmeasurableValue {
			// A DECLARATION, not a measurement. It cannot pass the node on its
			// own and it cannot fail it either; what it means for acceptance is
			// the profile's call, made by pkg/verdict against the threshold's
			// Applicability. Counting it as measured here would let a part with
			// no ECC hardware satisfy this runner's own gate on nothing.
			unmeasurable = append(unmeasurable, k)
			continue
		}
		n, err := strconv.ParseFloat(raw, 64)
		if err != nil {
			continue
		}
		measured = append(measured, k)
		if n > 0 {
			faults = append(faults, k+"="+raw)
		}
	}

	if len(measured) == 0 {
		// Not a hardware verdict: we could not look. Error, never Fail, and
		// never Skip — host health applies to every node.
		msg := "HOST_HEALTH_ERROR: no host-health counter could be read on this node " +
			"(no NVML, no readable kernel log, no PCIe AER, no NIC link counters) — the node is unjudged, not healthy"
		if len(unmeasurable) > 0 {
			msg += "; unmeasurable on this hardware: " + strings.Join(unmeasurable, ", ")
		}
		return exitError, msg
	}
	if len(faults) > 0 {
		return exitFail, "HOST_HEALTH_FAULT: " + strings.Join(faults, " ")
	}
	return exitPass, "HOST_HEALTH_OK"
}

// ---------------------------------------------------------------------------
// emitter
// ---------------------------------------------------------------------------

// emitter collects key=value metrics in insertion order.
//
// Deterministic order matters more than it looks: these lines are what a human
// reads out of `kubectl logs` when a node is condemned, and two runs of the
// same runner on the same node should diff cleanly.
type emitter struct {
	order []string
	vals  map[string]string
}

func newEmitter() *emitter {
	return &emitter{vals: map[string]string{}}
}

// set records a metric. Re-setting a key overwrites it in place; the parser is
// last-occurrence-wins, so keeping the position stable keeps the printed order
// and the parsed result telling the same story.
func (e *emitter) set(key, value string) {
	if _, seen := e.vals[key]; !seen {
		e.order = append(e.order, key)
	}
	e.vals[key] = value
}

func (e *emitter) setInt(key string, v int64) { e.set(key, strconv.FormatInt(v, 10)) }

// setIntIf records a counter only when it was actually measured. This is rule 1
// of the package comment in one function: `ok` false means the counter is
// omitted, never zeroed.
func (e *emitter) setIntIf(key string, v int64, ok bool) {
	if ok {
		e.setInt(key, v)
	}
}

// setCounter records a counter according to what the driver actually said,
// which is rule 1's third case. Measured counters get their number; a counter
// the hardware has no way to produce gets the sentinel, which is a claim about
// the part rather than about this run; a counter we simply could not read is
// omitted and fails closed upstream.
func (e *emitter) setCounter(key string, v int64, state counterState) {
	switch state {
	case counterOK:
		e.setInt(key, v)
	case counterUnmeasurable:
		e.set(key, unmeasurableValue)
	}
}

func (e *emitter) get(key string) (string, bool) {
	v, ok := e.vals[key]
	return v, ok
}

func (e *emitter) writeTo(w io.Writer) {
	for _, k := range e.order {
		fmt.Fprintf(w, "%s=%s\n", k, e.vals[k])
	}
}

// keys returns the emitted keys, sorted. Used by tests.
func (e *emitter) keys() []string {
	out := append([]string(nil), e.order...)
	sort.Strings(out)
	return out
}

// ---------------------------------------------------------------------------
// small helpers
// ---------------------------------------------------------------------------

type commandRunner func(ctx context.Context, name string, args []string) (string, error)

func envStr(name, def string) string {
	if v := strings.TrimSpace(os.Getenv(name)); v != "" {
		return v
	}
	return def
}

func envInt(name string, def int) int {
	v := strings.TrimSpace(os.Getenv(name))
	if v == "" {
		return def
	}
	n, err := strconv.Atoi(v)
	if err != nil {
		return def
	}
	return n
}

func splitPaths(s string) []string {
	var out []string
	for _, p := range strings.Split(s, ":") {
		if p = strings.TrimSpace(p); p != "" {
			out = append(out, p)
		}
	}
	return out
}

// delta differences a monotonic counter across the window. It reports ok only
// when BOTH samples were taken, and clamps a decrease to zero: a counter that
// went backwards was reset (driver reload, device rebind), which is not
// evidence of negative faults.
func delta(before, after int64, beforeOK, afterOK bool) (int64, bool) {
	if !beforeOK || !afterOK {
		return 0, false
	}
	if after < before {
		return 0, true
	}
	return after - before, true
}
