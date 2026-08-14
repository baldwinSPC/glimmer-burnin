// SPDX-License-Identifier: Apache-2.0
// Copyright the Glimmer authors.

package contract

// THE WORDS A CONSUMER SWITCHES ON.
//
// The envelope told consumers to read Violation.Cause first — it is the field
// that decides whether an engineer is dispatched — and then exported no
// constant for any of its values. A consumer acting on the structured verdict
// had to hard-code "Measurement", "Evidence", "Authoring", or import
// pkg/verdict for the typed versions (#295).
//
// A hand-written literal is not a hypothetical risk here. `cause` routes a
// verdict to a human or away from one, and a typo silently sends a hardware
// shortfall down the "nobody acts" branch with the compiler unable to say so.
// That is the same shape as every fabricated-evidence defect this project has
// fixed: not a wrong answer today, a foreseeable one.
//
// These live HERE rather than only in pkg/verdict because the envelope is what
// a third party decodes, and its vocabulary belongs in its own package. Same
// reasoning that moved the acceptance vocabulary in #274/#342: the words a
// consumer needs should not require importing the thing that produced them.
// pkg/verdict aliases these, so the producer and the document cannot drift.
//
// THE OPEN-WORLD RULE SURVIVES. Every type below is a named string, so an
// unrecognised value decodes rather than erroring — a consumer built against
// this version still parses a document from a newer one, and reads a cause it
// does not know rather than failing. Do NOT switch these to integers, and do
// not add a validating UnmarshalJSON: a receiver that rejects an unknown cause
// would drop the whole delivery over a word it did not need to understand.

// Cause says WHO SHOULD ACT on a violation, and is the field to read first.
type Cause string

const (
	// CauseMeasurement is the hardware falling short of a bar applied to a real
	// reading. This is the one that dispatches an engineer.
	CauseMeasurement Cause = "Measurement"
	// CauseEvidence is a runner report that cannot support a judgement at all.
	// The node is UNJUDGED, not condemned — re-run it; do not replace it.
	CauseEvidence Cause = "Evidence"
	// CauseAuthoring is a broken threshold. No hardware is implicated and no
	// node should be touched: fix the profile.
	CauseAuthoring Cause = "Authoring"
)

// Causes is every cause this version emits, for a consumer building a
// switch and for the agreement test in internal/sink.
var Causes = []Cause{CauseMeasurement, CauseEvidence, CauseAuthoring}

// ViolationKind is the specific route to a violation, finer-grained than Cause.
//
// A consumer that only needs to know who acts reads Cause. A consumer building
// a report reads this to say WHY.
type ViolationKind string

const (
	// KindUnsatisfied — a measurement was taken and it did not meet the bar.
	KindUnsatisfied ViolationKind = "Unsatisfied"
	// KindNotReported — the runner never emitted the gated metric. Fails
	// closed: a missing measurement must never satisfy an acceptance.
	KindNotReported ViolationKind = "NotReported"
	// KindNonNumeric — the runner emitted something that is not a number.
	KindNonNumeric ViolationKind = "NonNumeric"
	// KindNonFinite — NaN or an infinity, which compares false against
	// everything and would make NotEqual a gate that always passes.
	KindNonFinite ViolationKind = "NonFinite"
	// KindContradictory — the runner reported the same metric as both measured
	// and unmeasurable.
	KindContradictory ViolationKind = "Contradictory"
	// KindUnmeasurableRequired — the runner declared the metric unmeasurable
	// and the threshold is Required rather than RequiredIfMeasurable.
	KindUnmeasurableRequired ViolationKind = "UnmeasurableRequired"
	// KindThresholdValueNonNumeric — the THRESHOLD's value is not a number.
	KindThresholdValueNonNumeric ViolationKind = "ThresholdValueNonNumeric"
	// KindThresholdValueNonFinite — the THRESHOLD's value is NaN or infinite.
	KindThresholdValueNonFinite ViolationKind = "ThresholdValueNonFinite"
)

// ViolationKinds is every kind this version emits.
var ViolationKinds = []ViolationKind{
	KindUnsatisfied, KindNotReported, KindNonNumeric, KindNonFinite,
	KindContradictory, KindUnmeasurableRequired,
	KindThresholdValueNonNumeric, KindThresholdValueNonFinite,
}

// Phase is a run's or a test's outcome.
//
// The distinctions here are the project's central discipline and a consumer
// must not collapse them: PhaseError says our machinery broke and the hardware
// is UNJUDGED; PhaseFailed is a verdict about the hardware; PhaseSkipped says
// the test does not apply to this part; PhaseCancelled says a human stopped
// the run and nothing was judged. Only PhaseFailed condemns anything.
type Phase string

const (
	PhasePending   Phase = "Pending"
	PhaseRunning   Phase = "Running"
	PhasePassed    Phase = "Passed"
	PhaseFailed    Phase = "Failed"
	PhaseError     Phase = "Error"
	PhaseSkipped   Phase = "Skipped"
	PhaseCancelled Phase = "Cancelled"
)

// Phases is every phase this version emits.
var Phases = []Phase{
	PhasePending, PhaseRunning, PhasePassed, PhaseFailed,
	PhaseError, PhaseSkipped, PhaseCancelled,
}

// Cause reports which cause this kind belongs to.
//
// The SINGLE mapping between the two, so a Violation's Kind and Cause can never
// disagree — and exported here so a consumer classifying a kind it received
// gets the same answer the producer did, rather than reimplementing this switch
// and drifting from it.
//
// An unrecognised kind answers CauseEvidence, which is the open-world rule
// applied to a decision: a kind from a newer version means we could not
// establish anything, never that the hardware is at fault. Erring towards
// "unjudged" sends nobody to replace a working part.
func (k ViolationKind) Cause() Cause {
	switch k {
	case KindUnsatisfied:
		return CauseMeasurement
	case KindThresholdValueNonNumeric, KindThresholdValueNonFinite:
		return CauseAuthoring
	default:
		return CauseEvidence
	}
}
