package v1alpha1

import (
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

// RunPhase is the lifecycle of a BurnInRun, and doubles as the phase of an
// individual TestResult.
// +kubebuilder:validation:Enum=Pending;Running;Passed;Failed;Error;Skipped;Cancelled
type RunPhase string

const (
	RunPending RunPhase = "Pending"
	RunRunning RunPhase = "Running"
	RunPassed  RunPhase = "Passed"
	RunFailed  RunPhase = "Failed"
	// RunError means the machinery malfunctioned (runner crash, unpullable
	// image): the hardware is unjudged. Distinct from Failed, which is a real
	// verdict about the hardware.
	RunError RunPhase = "Error"
	// RunSkipped (tests only) means the test does not apply to this hardware —
	// exit code 2 in the runner contract. A node that cannot run a test has
	// not failed it; collapsing Skip into Fail is the false-negative class
	// that made healthy Sparks look broken.
	RunSkipped RunPhase = "Skipped"
	// RunCancelled means a human or an external control plane stopped this run
	// on purpose (spec.cancel). It is terminal and it is NOT a verdict: the
	// hardware was not judged, and a cancelled run must never be reported or
	// summarised as Failed. It is kept distinct from Error for the opposite
	// reason — Error says our machinery broke, Cancelled says it did exactly
	// what it was told. A fleet operator has to be able to tell "I stopped
	// this" from "this stopped itself".
	RunCancelled RunPhase = "Cancelled"
)

// IsTerminal reports whether a phase is final — no further work will be done
// and the phase will not change again.
//
// It includes RunSkipped, which the run-scoped helper in the controller does
// not need to: a run never takes that phase, but a TestResult does, and a
// skipped test is finished. The two therefore agree on every phase a RUN can
// actually hold, and this one is additionally correct for a TestResult.
func (p RunPhase) IsTerminal() bool {
	switch p {
	case RunPassed, RunFailed, RunError, RunSkipped, RunCancelled:
		return true
	default:
		return false
	}
}

// Admission conditions.
//
// A refused run is terminal with phase Error, because Error is exactly what it
// means: the machinery declined to run, and the hardware was never judged. It
// deliberately does NOT get a phase of its own — a new terminal phase would have
// to be understood by every consumer of the delivery envelope before any of them
// could tell it from a pass, and a consumer that does not recognise a phase is a
// consumer that might treat it as one. The condition below is what makes the
// CAUSE machine-readable without moving the verdict.
const (
	// ConditionRunAdmitted says whether this run was allowed to take its
	// targets. False with reason TargetsBusy means another run already holds
	// one of them; the message names it.
	ConditionRunAdmitted = "Admitted"

	// ReasonRunAdmitted: the targets were free (or the run was forced).
	ReasonRunAdmitted = "TargetsAvailable"
	// ReasonRunForced: the targets were NOT free and spec.force admitted the
	// run anyway. Recorded separately from ReasonRunAdmitted so a verdict
	// produced under deliberate contention is identifiable as such.
	ReasonRunForced = "AdmittedByForce"
	// ReasonTargetsBusy: refused, because an already-active run holds a target.
	ReasonTargetsBusy = "TargetsBusy"
)

// CancelPolicy decides what happens to work already in flight when a run is
// cancelled.
// +kubebuilder:validation:Enum=Graceful;Immediate
type CancelPolicy string

const (
	// CancelGraceful stops launching new work and lets the executions already
	// running finish, so their results and metrics are recorded. This is the
	// default because an in-flight burn-in is expensive evidence: a soak that
	// is four hours in has already cost the fleet those four hours, and
	// throwing the result away means paying for them again.
	CancelGraceful CancelPolicy = "Graceful"
	// CancelImmediate deletes in-flight test pods now. Use it when the reason
	// for cancelling is the load itself — a facility power or cooling event,
	// where waiting for a thermal soak to finish is the opposite of what is
	// wanted. Executions killed this way are recorded as Cancelled, never as
	// Failed: the operator stopped them, the hardware did not fail them.
	CancelImmediate CancelPolicy = "Immediate"
)

// BurnInRunSpec is one execution of a BurnInProfile against a target.
type BurnInRunSpec struct {
	// ProfileRef names the BurnInProfile to run (same namespace).
	// +kubebuilder:validation:Required
	ProfileRef string `json:"profileRef"`

	// Target selects the node(s) under test.
	// +kubebuilder:validation:Required
	Target TargetSelector `json:"target"`

	// MaxConcurrentNodes caps how many nodes this run exercises at the same
	// time. DEFAULT 1 — strictly sequential.
	//
	// THIS IS A FACILITY SAFETY INTERLOCK, NOT A PERFORMANCE KNOB. Burn-in
	// deliberately drives hardware to its power and thermal limits; that is
	// the whole point of the test. Raising this multiplies real electrical
	// load and real heat in a real room. Ten nodes at full power draw is a
	// rack-level power event, and every accelerator in the run holding maximum
	// load at once is a cooling event — either can trip a breaker, exceed a
	// PDU's budget, or push inlet temperatures past what the room can carry.
	// The consequences land on hardware that was never under test.
	//
	// The default of 1 is the safe floor, and it is deliberately inconvenient:
	// a large sweep is slow at 1, and the operator raising it is the person
	// who has to have checked the circuit and the cooling headroom first. It
	// must never be raised automatically, defaulted higher "for throughput",
	// or inferred from cluster size. If a run is too slow, the answer is a
	// facility conversation, not a bigger number.
	//
	// There is a correctness reason too, on top of the facility one: concurrent
	// nodes contend for shared fabric and shared power, so a node measured
	// beside nine others under full load is not measured under the same
	// conditions as one measured alone. A bandwidth or clock verdict taken
	// under contention is not comparable to one taken sequentially.
	//
	// +kubebuilder:default=1
	// +kubebuilder:validation:Minimum=1
	MaxConcurrentNodes *int32 `json:"maxConcurrentNodes,omitempty"`

	// Force admits this run even when one of its targets is already held by
	// another active BurnInRun. DEFAULT false, and it should stay false.
	//
	// THIS IS THE ESCAPE HATCH ON A FACILITY INTERLOCK, not a way to go faster.
	// MaxConcurrentNodes bounds how many nodes ONE run drives at full power;
	// nothing about it bounds how many runs drive the same node, and two runs
	// each honouring a cap of 1 on the same node are two full-power soaks on one
	// machine — twice the power, twice the heat, and each run believing it is
	// compliant. That is what admission refuses, and it is why the refusal is
	// the default rather than a warning.
	//
	// The refusal is also a correctness gate, not only a facility one. Two soaks
	// contending for the same accelerators, memory bandwidth and fabric measure
	// each other rather than the hardware, so both report degraded numbers and a
	// healthy node earns a failing verdict — or, as observed on real hardware,
	// the kubelet kills one of them and the run reports exit 137 against a node
	// that was never at fault.
	//
	// Setting it is a statement that the overlap is deliberate and that the
	// power and cooling headroom has been checked by the person setting it. It
	// is recorded on the run and in the exported envelope, so a verdict produced
	// under contention can be identified as such months later.
	//
	// It does NOT disable the cordon ownership rules. A forced run still refuses
	// to take, overwrite or release a node another run has stamped, and it still
	// records the node's PRE-BURN-IN schedulability rather than the cordon it
	// finds — a forced overlap must not be able to launder another run's cordon
	// into permanent prior state.
	//
	// +kubebuilder:default=false
	Force *bool `json:"force,omitempty"`

	// RetryOnErrorLimit is how many extra attempts a test gets after an ERROR.
	// Default 0 — no retries.
	//
	// Retries apply to the Error phase ONLY: image pull failure, node
	// disappearing mid-test, runner crash, an exit code outside the runner
	// contract. Those say our machinery malfunctioned and the hardware was
	// never judged, and re-running is the correct response.
	//
	// A RETRY MUST NEVER RE-RUN A FAILED TEST. A Failed verdict is a
	// measurement: the hardware was judged and did not meet the threshold.
	// Re-running it until it passes launders a real hardware fault into an
	// acceptance, which is the single worst outcome this operator can produce
	// — a bad part shipped to production with a clean certificate. Marginal
	// hardware is exactly the hardware that passes on the second try. If a
	// test is genuinely flaky in a way retries would paper over, the fix is a
	// better threshold or a repeat count, both of which make the gate
	// stricter, not looser.
	//
	// Skipped is likewise never retried: a test that does not apply to this
	// hardware will not start applying to it on the next attempt.
	//
	// Each retry re-runs the full test, so the cost is bounded by
	// DeadlineSeconds. Attempts are recorded individually on the TestResult so
	// a node that only passes after three retries is visible as such, not
	// indistinguishable from one that passed first time.
	//
	// +kubebuilder:default=0
	// +kubebuilder:validation:Minimum=0
	RetryOnErrorLimit *int32 `json:"retryOnErrorLimit,omitempty"`

	// DeadlineSeconds bounds the WHOLE run, measured from status.startedAt:
	// every test, every node, every repeat and every retry together. nil or 0
	// means no deadline.
	//
	// Exceeding it drives the run to Error, not Failed and not Cancelled. Error
	// is correct because a run that ran out of time did not judge the hardware
	// it had not reached yet, and reporting that as a hardware verdict would
	// condemn parts that were never tested. It is not Cancelled because nobody
	// asked for it — Cancelled is reserved for an explicit request, so that a
	// consumer can tell an operator's decision from a bound being hit.
	//
	// Tests that DID complete before the deadline keep their real results.
	// A deadline stops a run; it does not retract evidence already gathered.
	//
	// This is the outer bound that keeps a run from occupying nodes forever:
	// per-test DurationSeconds bounds one execution, but repeats, retries and
	// a long node list multiply, and MaxConcurrentNodes=1 makes a wide sweep
	// long by design.
	//
	// +kubebuilder:validation:Minimum=0
	DeadlineSeconds *int32 `json:"deadlineSeconds,omitempty"`

	// Suspend pauses the run: no new test pods are launched. Executions
	// already in flight are left alone to finish and record their results —
	// suspending is not a reason to discard evidence.
	//
	// Suspend is REVERSIBLE and NON-TERMINAL. A suspended run produces no
	// verdict and reaches no terminal phase; clearing the flag resumes it from
	// where it stopped, with the results it had already recorded intact. Use
	// it for a maintenance window or a facility power constraint, where the
	// run should continue later rather than be redone.
	//
	// Cancel dominates Suspend: a run that is both suspended and cancelled
	// cancels, because a suspended run that could never be cancelled would be
	// unstoppable without deleting the object.
	//
	// +kubebuilder:default=false
	Suspend *bool `json:"suspend,omitempty"`

	// Baseline declares that this run MEASURES rather than certifies.
	//
	// Running thresholdless to gather baselines already works — a profile with
	// no thresholds passes on exit 0 and records its metrics. What did not work
	// was telling the two apart AFTERWARDS: a baseline sweep and an acceptance
	// run both deliver phase Passed, so a consumer gating cluster admission on
	// "the last run passed" would certify hardware against a run that gated
	// nothing. That is a fail-open in the reporting path, and this flag is what
	// closes it.
	//
	// IT DOES NOT SUPPRESS THRESHOLDS, and must never be made to. A run marked
	// baseline whose profile carries a gate is REFUSED at start, as a config
	// Error naming the test. Suppression would be a threshold-laundering switch —
	// a way to turn a failing acceptance run into a passing measurement run by
	// flipping a boolean — and this operator's whole premise is that a verdict
	// cannot be edited into existence. If a profile has gates it is not a
	// baseline; use a different profile.
	//
	// The flag rides into every delivered envelope, so a consumer can never
	// mistake a sweep for a certification.
	//
	// +kubebuilder:default=false
	Baseline *bool `json:"baseline,omitempty"`

	// Cancel requests that this run stop for good. It is the CANCEL_BURNIN
	// verb: an external control plane sets it, the run drives to Cancelled,
	// and every node the run cordoned is released.
	//
	// Cancel is ONE-WAY. Once the controller has observed it and begun
	// cancelling, clearing the flag does not resurrect the run; a run that
	// could un-cancel would race its own cleanup and could leave a node
	// cordoned with nothing left to uncordon it. To run again, create a new
	// BurnInRun — which is also what makes the audit trail honest, since the
	// new run has its own fingerprint and its own evidence.
	//
	// Cancelled is terminal and is NOT a hardware verdict. Tests that finished
	// before the cancel keep their real results; tests that had not run are
	// left unjudged rather than being marked failed.
	//
	// +kubebuilder:default=false
	Cancel *bool `json:"cancel,omitempty"`

	// CancelPolicy decides the fate of in-flight executions when Cancel is
	// set. Ignored unless Cancel is true.
	// +kubebuilder:default=Graceful
	CancelPolicy CancelPolicy `json:"cancelPolicy,omitempty"`

	// CancelReason is free-form text recorded on the run and carried in the
	// exported envelope, so a consumer reading a Cancelled verdict months
	// later can tell "facility power event" from "operator typo'd the target"
	// from "superseded by a newer run". A cancellation with no reason is the
	// one result nobody can act on.
	// +kubebuilder:validation:MaxLength=512
	CancelReason string `json:"cancelReason,omitempty"`

	// TTLSecondsAfterFinished garbage-collects the run (and its pods) this long
	// after it reaches a terminal phase. 0 keeps it.
	TTLSecondsAfterFinished *int32 `json:"ttlSecondsAfterFinished,omitempty"`
}

// AttemptTrigger says why an execution happened. It is what keeps the repeat
// rule and the retry rule from being confused for one another after the fact.
// +kubebuilder:validation:Enum=Initial;Repeat;ErrorRetry;Segment
type AttemptTrigger string

const (
	// AttemptInitial is the first execution of a test on a node.
	AttemptInitial AttemptTrigger = "Initial"
	// AttemptRepeat is an execution demanded by BurnInTestSpec.RepeatCount.
	// Repeats make the gate STRICTER: every one must pass.
	AttemptRepeat AttemptTrigger = "Repeat"
	// AttemptSegment is the next window of a soak divided by
	// BurnInTestSpec.Soak. It is neither a repeat nor a retry, and recording it
	// as either would misstate the history: the previous execution succeeded
	// AND contributed to the aggregate the verdict will be read from, where a
	// repeat is a whole test run again and a retry is a measurement that never
	// happened.
	AttemptSegment AttemptTrigger = "Segment"
	// AttemptErrorRetry is an execution granted by
	// BurnInRunSpec.RetryOnErrorLimit after a previous attempt ended in Error.
	// A retry may only ever follow an Error. If a retry is ever seen following
	// a Failed attempt, machinery has laundered a hardware verdict, and this
	// field is the evidence that it happened.
	AttemptErrorRetry AttemptTrigger = "ErrorRetry"
)

// TestAttempt is one execution of a test on a node — one pod, one exit code,
// one set of metrics.
//
// Attempts are recorded individually rather than collapsed into the final
// phase because the shape of the attempt history IS the hardware signal. A node
// that passed on the first try and a node that passed on the fourth try after
// three errors are both "Passed" at the test level and are very different
// parts. Keeping the history is also the only way to audit the retry rule:
// every retry's predecessor must be an Error, and that is checkable here.
type TestAttempt struct {
	// Attempt is the 1-based index of this execution within the test's
	// attempt sequence on this node.
	// +kubebuilder:validation:Minimum=1
	Attempt int32 `json:"attempt"`

	// Trigger says why this execution happened.
	Trigger AttemptTrigger `json:"trigger"`

	// Node is the node this execution ran on. For Pair and Group scope it is
	// the rank-0 / initiating node; Nodes on the TestResult carries the full set.
	Node string `json:"node,omitempty"`

	// PodName is the pod that carried this execution, so a stored result can
	// still be traced back to logs and events.
	PodName string `json:"podName,omitempty"`

	// Phase is this execution's own outcome. It is not the test's outcome: a
	// test with repeats is Passed only if EVERY attempt passed, and a test
	// with retries is Passed if the LAST attempt passed after Errors.
	Phase RunPhase `json:"phase,omitempty"`

	// ExitCode is the runner's raw exit code (0 pass, 1 fail, 2 skip, anything
	// else Error). Recorded raw so an out-of-contract code stays visible as
	// the contract violation it is instead of being flattened into Error.
	ExitCode *int32 `json:"exitCode,omitempty"`

	StartedAt  *metav1.Time `json:"startedAt,omitempty"`
	FinishedAt *metav1.Time `json:"finishedAt,omitempty"`

	// LastCheckpointAt is when this execution's in-progress metrics were last
	// refreshed, if BurnInTestSpec.CheckpointIntervalSeconds is enabled. While
	// the attempt is running, Metrics below are checkpoint evidence, not a
	// judged result.
	LastCheckpointAt *metav1.Time `json:"lastCheckpointAt,omitempty"`

	// Metrics are this execution's parsed metrics. Per-attempt rather than
	// merged: merging would silently discard the run where the bandwidth
	// dipped, which is the run that mattered.
	Metrics map[string]string `json:"metrics,omitempty"`

	// Message is a human-readable outcome for this execution.
	Message string `json:"message,omitempty"`
}

// TestResult is the outcome of one test within a run, aggregated over its
// attempts.
type TestResult struct {
	Name string `json:"name"`
	// Kind is the executed test's kind. It is EMPTY on a synthetic result — one
	// that is about the run rather than about any hardware: "resolve" when the
	// spec could not be resolved, "admission" when the fleet was busy. Every
	// real BurnInTest has a Kind, so an empty one is how the controller (and a
	// consumer) tells the two apart; a real test named "resolve" cannot shadow
	// it.
	Kind TestKind `json:"kind"`
	// Scope is the topology the test ran at, from the pinned plan. It is ABSENT
	// on a synthetic result, and the schema has to allow that: an unresolvable
	// testRef is not a Node, a Pair or a Group, and when this field was a
	// required enum the operator wrote a status its own CRD refused — the
	// reconcile errored, requeued, built the same status and was refused again,
	// forever, and the run sat with an empty status looking queued rather than
	// broken (#391). Inventing a scope to satisfy the schema would be a lie a
	// consumer might act on. Every real result carries one by construction: it
	// is copied from the planned test, whose scope the apiserver defaults and
	// plan.go validates.
	// +optional
	Scope TestScope `json:"scope,omitempty"`
	// Phase is the test's verdict across all attempts. With repeats it is the
	// AND of every attempt; with retries it is the outcome of the last attempt
	// after the Errors that preceded it.
	Phase RunPhase `json:"phase"`
	Nodes []string `json:"nodes,omitempty"`

	// StartedAt is when the FIRST attempt began executing, and FinishedAt is
	// when the LAST attempt ended, so the pair always spans the test's real
	// occupancy of the hardware — including the time lost to errored attempts
	// and the gaps between repeats. Anything narrower would under-report how
	// long a node was actually held.
	//
	// StartedAt is set when execution actually begins, never at scheduling
	// time: a pod that sat unschedulable for an hour did not test anything for
	// that hour, and counting the wait as test time would make a stuck run
	// look like a slow one.
	StartedAt  *metav1.Time `json:"startedAt,omitempty"`
	FinishedAt *metav1.Time `json:"finishedAt,omitempty"`

	// LastCheckpointAt is when the running attempt last refreshed Metrics.
	// Non-nil with a nil FinishedAt means the numbers below are a mid-flight
	// sample and no threshold has been applied to them yet.
	LastCheckpointAt *metav1.Time `json:"lastCheckpointAt,omitempty"`

	// Metrics are the parsed numeric results the thresholds were evaluated
	// against (e.g. busBandwidthGBs: "23.4", eccErrors: "0"). With repeats
	// these are the metrics of the attempt that decided the verdict — the
	// first failing attempt if the test failed, otherwise the last — so the
	// numbers here always explain the phase beside them. Per-attempt numbers
	// are in Attempts.
	Metrics map[string]string `json:"metrics,omitempty"`

	// Message is a human-readable outcome (a failing threshold, the real error).
	Message string `json:"message,omitempty"`

	// Violations lists EVERY threshold the deciding attempt did not satisfy, in
	// spec order — the same attempt Metrics came from, so the two always explain
	// the same execution. Empty unless Phase is Failed.
	//
	// Message names only the first, because that field is frozen; this is the
	// complete picture. Read Cause before acting: a Failed test can mix a
	// hardware shortfall with a broken threshold, and only the former is a
	// reason to touch a node.
	Violations []Violation `json:"violations,omitempty"`

	// NotEvaluated are the thresholds that were neither satisfied nor violated,
	// because they were never applied — a RequiredIfMeasurable gate on a metric
	// the runner declared unmeasurable.
	//
	// It is recorded as DATA rather than left in Message because the distinction
	// it carries is the one a consumer most needs and can least infer from
	// prose: a gate that did not run is not a gate that passed. A Passed test
	// whose ECC gate never applied must never be indistinguishable from one whose
	// ECC gate applied and was satisfied.
	NotEvaluated []NotEvaluated `json:"notEvaluated,omitempty"`

	// Unmeasurable are the metric names the runner POSITIVELY DECLARED it cannot
	// measure on this hardware (the reserved `n/a` value).
	//
	// This is a claim about the PART, not about the run, and it is the only thing
	// that lets a RequiredIfMeasurable gate be reported as not-evaluated rather
	// than failing a healthy node. It must survive to a consumer intact:
	// "this part has no ECC" and "this part reported zero ECC errors" are
	// different statements about different hardware, and a consumer that cannot
	// tell them apart will eventually treat one as the other.
	Unmeasurable []string `json:"unmeasurable,omitempty"`

	// Artifacts are the non-metric evidence the runner returned, by REFERENCE.
	//
	// The payload lives in a run-owned ConfigMap, never here: a status is read
	// on every reconcile by every watcher, and a megabyte of dcgmi JSON inlined
	// into it would be re-transmitted on each one. The reference carries what a
	// consumer needs in order to decide whether to fetch — what it is, how big,
	// and a digest to verify what it got.
	//
	// A ref with Dropped set is evidence that was REFUSED, kept deliberately.
	// Omitting it would make "the runner produced no artifact" and "the artifact
	// was too large to keep" the same observation, and only one of those is a
	// reason to go and look at the node.
	Artifacts []ArtifactRef `json:"artifacts,omitempty"`

	// VariantAxes are the matrix labels this execution came from — {precision:
	// fp4}, {class: soak} — or empty for a test with no variants.
	//
	// Echoed so a report can group a sweep's cells without parsing them back out
	// of test names. Names are result identity and are stable, but they are
	// STRINGS: recovering "which precision was this" by splitting on a hyphen
	// works until a variant is called "fp8-dense", and then it works wrongly and
	// silently.
	// +optional
	VariantAxes map[string]string `json:"variantAxes,omitempty"`

	// RepeatsRequired is the resolved BurnInTestSpec.RepeatCount for this run:
	// how many passing executions this test owes per node. Pinned onto the
	// result so a verdict stays readable after the BurnInTest is edited.
	// +kubebuilder:validation:Minimum=0
	RepeatsRequired int32 `json:"repeatsRequired,omitempty"`

	// RepeatsCompleted is how many of those executions have passed so far.
	// Passed requires RepeatsCompleted == RepeatsRequired; a test that stopped
	// short is not passing, it is unfinished.
	// +kubebuilder:validation:Minimum=0
	RepeatsCompleted int32 `json:"repeatsCompleted,omitempty"`

	// ErrorRetries is how many attempts were spent recovering from Error. It
	// is tracked separately from repeats so retries can never be mistaken for
	// satisfying RepeatCount: an errored attempt performed no measurement and
	// therefore owes its repeat again.
	// +kubebuilder:validation:Minimum=0
	ErrorRetries int32 `json:"errorRetries,omitempty"`

	// SegmentsRequired is how many pod executions this test's soak was divided
	// into, resolved from BurnInTestSpec.Soak at run start and pinned here.
	// Zero means the test is not segmented and ran as one pod, which is every
	// test that does not set spec.soak.
	//
	// It is also the flag the reconciler branches on: a result with a positive
	// SegmentsRequired renders its verdict over AggregatedMetrics rather than
	// over any one attempt.
	// +kubebuilder:validation:Minimum=0
	SegmentsRequired int32 `json:"segmentsRequired,omitempty"`

	// SegmentsCompleted is how many segments have finished CLEANLY and
	// contributed to AggregatedMetrics.
	//
	// It does not advance for an errored segment, and that is the property the
	// whole design rests on: an Error means the segment measured nothing, so it
	// re-runs the SAME index and contributes nothing. A counter that advanced on
	// an error would silently shorten the soak by every interruption it suffered
	// — which is the failure this feature exists to prevent, reintroduced as
	// bookkeeping.
	// +kubebuilder:validation:Minimum=0
	SegmentsCompleted int32 `json:"segmentsCompleted,omitempty"`

	// SegmentSeconds is the length of one window, pinned from the plan alongside
	// SegmentsRequired.
	//
	// Recorded because SegmentsCompleted is a count and a count is not a
	// duration: "40 of 288" says how far a soak got only if the reader also knows
	// how long a window is, and neither this status nor the delivery envelope
	// carries the run's spec. Zero for an unsegmented test.
	// +kubebuilder:validation:Minimum=0
	SegmentSeconds int32 `json:"segmentSeconds,omitempty"`

	// AggregatedMetrics is the running combination of every clean segment's
	// metrics, folded by the per-metric rule in pkg/contract.AggregationFor.
	//
	// IT IS PERSISTED, and that is what makes attempt truncation and controller
	// restarts safe: the verdict is read from here and never from the attempt
	// list, so attempts may be capped and a manager may die at any point without
	// the soak losing what it measured. A metric no segment reported is absent
	// from it — a soak must never manufacture a number it did not measure, and
	// fail-closed evaluation of the absence at the end is the correct outcome.
	AggregatedMetrics map[string]string `json:"aggregatedMetrics,omitempty"`

	// TruncatedAttempts counts the passing segments dropped from Attempts to
	// keep the status writable.
	//
	// A 72-hour soak at 15-minute segments is 288 attempt records, each with a
	// metrics map, in a status that lives in etcd and is copied into a
	// 900 KiB-capped ConfigMap sink. Attempts keeps the first, every
	// non-passing one, and the most recent passing ones; this says how many
	// uneventful segments are not shown, so an elided history is visible as
	// elided rather than as a soak that ran fewer segments than it did.
	// +kubebuilder:validation:Minimum=0
	TruncatedAttempts int32 `json:"truncatedAttempts,omitempty"`

	// Attempts is the per-execution history, in order.
	Attempts []TestAttempt `json:"attempts,omitempty"`
}

// Violation is one threshold a test did not satisfy, as recorded on a result.
//
// It mirrors verdict.Violation rather than reusing it, and that is forced
// rather than chosen: pkg/verdict imports this package, so importing it back
// would be a cycle. The controller maps between them in one place. If a field
// is added there, add it here too — a violation the API cannot express is a
// violation an operator never sees.
//
// Cause and Kind are deliberately NOT +kubebuilder:validation:Enum. An enum on
// a STATUS field turns every future violation kind into a rejected status
// write, which would wedge the run that discovered it — the failure mode would
// be a burn-in that cannot record why it failed. The vocabulary is documented
// in pkg/verdict and validated there, where getting it wrong costs a test
// rather than a run.
// ArtifactRef points at one piece of non-metric evidence a runner returned.
//
// It is a POINTER and not the payload; see TestResult.Artifacts for why.
type ArtifactRef struct {
	// Name is what the runner called it, e.g. "dcgmi.json", after sanitising.
	Name string `json:"name"`
	// MediaType is what the payload IS, so a renderer need not sniff it.
	MediaType string `json:"mediaType,omitempty"`
	// SizeBytes is the payload's TRUE size — including for a ref that was
	// dropped for exceeding a cap, which is the case where it matters most,
	// because "how much evidence did we lose" is the next question asked.
	// +kubebuilder:validation:Minimum=0
	SizeBytes int64 `json:"sizeBytes,omitempty"`
	// Digest is "sha256:<hex>" over the stored payload, so a consumer that
	// fetches it separately can verify it got what this verdict was about.
	Digest string `json:"digest,omitempty"`
	// ConfigMap is the run-owned ConfigMap holding the payload, and Key is the
	// entry within it. Both empty when Dropped is set.
	ConfigMap string `json:"configMap,omitempty"`
	Key       string `json:"key,omitempty"`
	// Dropped says why the evidence was not kept, and is empty when it was.
	// An artifact is evidence ABOUT a verdict and never part of one, so a
	// failure to store one is recorded here and changes no phase.
	Dropped string `json:"dropped,omitempty"`
}

// NotEvaluated is a threshold that was never applied.
//
// It mirrors pkg/verdict.NotEvaluated and pkg/contract.NotEvaluated. Three
// mirrors is unfortunate and deliberate: pkg/verdict must not import this
// package (it would be a cycle) and pkg/contract must stay free of Kubernetes
// types so a consumer can decode a verdict without client-go. The three are held
// in step by a test rather than by anyone remembering.
type NotEvaluated struct {
	// Metric is the threshold's metric name, as written in the profile.
	Metric string `json:"metric"`
	// Reason is why it was not applied (see pkg/verdict.ReasonUnmeasurable).
	Reason string `json:"reason"`
}

type Violation struct {
	// Index is the threshold's position in the test's spec.thresholds, so a
	// report can point at the offending entry rather than at the test.
	// +kubebuilder:validation:Minimum=0
	Index int32 `json:"index"`

	// Metric is the threshold's metric name, as written in the profile.
	Metric string `json:"metric"`

	// Cause says who should act, and is the field to read first:
	//
	//   Measurement — the hardware fell short. This one is about the part.
	//   Evidence    — the runner's report could not support a judgement
	//                 (metric missing, unparseable, non-finite, contradictory,
	//                 or declared unmeasurable under a gate that requires it).
	//                 The node is unjudged, not condemned.
	//   Authoring   — the threshold itself is broken. No hardware is implicated
	//                 and no node should be touched; the profile needs fixing.
	Cause string `json:"cause"`

	// Kind is the specific route, finer-grained than Cause (e.g. Unsatisfied,
	// NotReported, ThresholdValueNonNumeric). See pkg/verdict.
	Kind string `json:"kind"`

	// Reason is the full sentence explaining this violation. For the first
	// violation it is exactly the leading text of Message.
	Reason string `json:"reason,omitempty"`
}

const (
	// ConditionPlanExpanded reports how much work the pinned plan actually is,
	// when variants expanded it beyond the entries the profile lists.
	//
	// A profile with one gemm entry and four precision variants across eight
	// nodes at maxConcurrentNodes 1 is thirty-two sequential executions. That
	// arithmetic is not visible anywhere in the profile, and the usual way to
	// discover it is to wait. It is a condition rather than an Event because an
	// Event expires in an hour and a multi-day sweep does not.
	ConditionPlanExpanded = "PlanExpanded"

	// ConditionThresholdsSound reports what the threshold linter
	// (verdict.ValidateThresholds) made of the gates this run is about to
	// apply, evaluated once against the PINNED plan so it describes the
	// thresholds that actually decided the verdict rather than whatever the
	// profile says later.
	//
	// It is advisory and it is deliberately NOT a phase. False does not stop
	// the run and does not make it Failed: an unsound gate is one whose verdict
	// may not mean what its author intended, and refusing to run on suspicion
	// would let this operator veto profiles it merely does not understand. The
	// gates it CAN prove unusable — a name the parser drops, a value that is not
	// a finite number — never reach a condition at all; they refuse the run at
	// plan time as a config Error, because a threshold nothing can satisfy is a
	// broken profile and must never be reported as broken hardware.
	//
	// It is a condition rather than an Event on purpose. The advisory has to
	// outlive the run it describes: an Event expires in an hour and the verdict
	// is kept for as long as anyone might read it, and an acceptance record with
	// no surviving note that one of its gates was unsound is exactly the
	// situation this is here to prevent.
	ConditionThresholdsSound = "ThresholdsSound"

	// ReasonThresholdsReviewed means the linter found nothing to say. It is not
	// a claim that the thresholds are correctly CALIBRATED — no linter can judge
	// that, and the gate this project got most wrong was one it would have
	// passed clean.
	ReasonThresholdsReviewed = "Reviewed"
	// ReasonVariantsExpanded is set when a profile entry became several
	// executions.
	ReasonVariantsExpanded = "VariantsExpanded"

	// ReasonUnsoundThresholds means at least one gate evaluates but its verdict
	// does not mean what its author appears to intend — an exact comparison on a
	// continuous measurement, a gate on a metric the registry records as
	// evidence, or a gate on a metric no runner of that kind is known to emit.
	ReasonUnsoundThresholds = "UnsoundThresholds"
)

// BurnInRunStatus tracks execution and the exported verdict.
type BurnInRunStatus struct {
	// +kubebuilder:default=Pending
	Phase RunPhase `json:"phase,omitempty"`
	// Fingerprint is the NodeFingerprint captured at run start (what hardware
	// this verdict actually applies to).
	Fingerprint map[string]string `json:"fingerprint,omitempty"`
	StartedAt   *metav1.Time      `json:"startedAt,omitempty"`
	FinishedAt  *metav1.Time      `json:"finishedAt,omitempty"`
	Results     []TestResult      `json:"results,omitempty"`
	Passed      int32             `json:"passed,omitempty"`
	Failed      int32             `json:"failed,omitempty"`
	// Errored counts tests that produced no verdict because the machinery
	// malfunctioned. It is counted separately from Failed and must never be
	// folded into it: a summary that shows "0 failed" while eight tests never
	// ran reads as a clean sweep of hardware nothing actually measured.
	Errored int32 `json:"errored,omitempty"`
	// Skipped counts tests that do not apply to this hardware (runner exit 2).
	// Also never folded into Failed — a node that cannot run a test has not
	// failed it.
	Skipped int32 `json:"skipped,omitempty"`

	// CordonedNodes are the nodes this run cordoned and therefore owes an
	// uncordon. It is the durable record the cleanup path works from: without
	// it, a controller restart between cordoning a node and finishing the run
	// has no way to know which nodes it made unschedulable, and the fleet
	// silently loses capacity nothing will ever give back. Only nodes the run
	// itself cordoned are listed, so a node an administrator had already
	// cordoned is never uncordoned on the run's way out.
	CordonedNodes []string `json:"cordonedNodes,omitempty"`

	// PriorUnschedulable is each target node's schedulability AS THE RUN FOUND
	// IT, keyed by node name, captured once when the run starts and never
	// recomputed. True means the node was already out of the scheduler before
	// this run touched it, and must be left that way.
	//
	// It exists because the answer to "was this node already cordoned?" has to
	// be a property of the RUN, not a reading taken at some arbitrary later
	// moment. A run cordons and releases a node once per wave, so the same
	// question used to be re-asked several times per run and answered from
	// whatever spec.unschedulable happened to say at that instant. Any cordon
	// the run could not attribute to itself — its own, seen after a manager
	// restart or an annotation it no longer recognises; a node an operator
	// cordoned while the burn-in was already holding it — was then recorded as
	// "pre-existing" and faithfully made PERMANENT on the way out, which is a
	// stranded cordon with a paper trail saying it was intentional.
	//
	// Captured before the first cordon and never overwritten, so it survives a
	// restart, is identical on every reconcile, and cannot be contaminated by
	// the run's own footprint. Nodes absent from the map (a target that did not
	// exist at start, or a run that began under an older operator) fall back to
	// reading the node, which is the old behaviour and no worse than it.
	PriorUnschedulable map[string]bool `json:"priorUnschedulable,omitempty"`

	// SinkStatus records the last export attempt per sink.
	SinkStatus         map[string]string  `json:"sinkStatus,omitempty"`
	ObservedGeneration int64              `json:"observedGeneration,omitempty"`
	Conditions         []metav1.Condition `json:"conditions,omitempty"`
}

// +kubebuilder:object:root=true
// +kubebuilder:subresource:status
// +kubebuilder:resource:shortName=bir,categories=burnin
// +kubebuilder:printcolumn:name="Profile",type=string,JSONPath=`.spec.profileRef`
// +kubebuilder:printcolumn:name="Phase",type=string,JSONPath=`.status.phase`
// +kubebuilder:printcolumn:name="Baseline",type=boolean,JSONPath=`.spec.baseline`,priority=1
// +kubebuilder:printcolumn:name="Passed",type=integer,JSONPath=`.status.passed`
// +kubebuilder:printcolumn:name="Failed",type=integer,JSONPath=`.status.failed`
// +kubebuilder:printcolumn:name="Age",type=date,JSONPath=`.metadata.creationTimestamp`

// BurnInRun is one execution of a BurnInProfile against a target.
type BurnInRun struct {
	metav1.TypeMeta   `json:",inline"`
	metav1.ObjectMeta `json:"metadata,omitempty"`
	Spec              BurnInRunSpec   `json:"spec,omitempty"`
	Status            BurnInRunStatus `json:"status,omitempty"`
}

// +kubebuilder:object:root=true

// BurnInRunList contains a list of BurnInRun.
type BurnInRunList struct {
	metav1.TypeMeta `json:",inline"`
	metav1.ListMeta `json:"metadata,omitempty"`
	Items           []BurnInRun `json:"items"`
}
