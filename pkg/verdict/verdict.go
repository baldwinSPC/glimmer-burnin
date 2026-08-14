// Package verdict evaluates parsed test metrics against a test's thresholds.
// It performs no I/O, so the pass/fail logic every BurnInRun depends on is
// unit-testable in isolation.
//
// IT IS NOT KUBERNETES-FREE, unlike pkg/contract and pkg/runner, and this
// comment used to claim otherwise. Evaluate takes []contract.Threshold,
// so importing this package brings api/v1alpha1 and with it apimachinery and
// controller-runtime — nine modules a consumer links. Issue #274 tracks whether
// to move the threshold types down into pkg/contract and make this genuinely
// standalone; that breaks every caller and is not something to do silently.
//
// This is public API. It exists outside internal/ so that a burn-in dispatcher
// which is not this operator — notably a pre-Kubernetes, agent-native path —
// can reach the same verdict from the same metrics. One brain, two dispatchers:
// if two callers can disagree about whether a node passed, the contract has
// already failed. Changing the semantics below changes acceptance for every
// consumer, so treat them as frozen absent a deliberate, versioned decision.
//
// Importing this package does not invert the dependency rule: consumers depend
// on this project; this project must never import them.
//
// # Comparison semantics
//
// A metric and a threshold value are both parsed as float64 and compared with
// no tolerance whatsoever. For GreaterThanOrEqual and LessThanOrEqual that is
// unremarkable — a floor is a floor, and the two together express a band.
//
// EQUAL AND NOTEQUAL ARE EXACT, AND THEY ARE FOR DIMENSIONLESS COUNTERS.
// eccErrors, throttleEvents, miscompares, xidEvents: integers, where exactly
// zero is the only acceptable answer, and where every value a runner can emit
// round-trips through float64 without loss. On those, exact equality is not a
// hazard — it is the only correct comparison.
//
// On a CONTINUOUS measurement they are a trap. `sustainedClockPct Equal 83.22`
// requires a sampled average to reproduce one decimal string exactly; it will
// not, so the gate fails on every healthy node forever, and the failure is
// reported in the same shape as a hardware verdict. The two-sided gate the
// author actually wanted is GreaterThanOrEqual plus LessThanOrEqual.
//
// There is deliberately NO epsilon, and this is the considered answer to that
// question rather than an omission:
//
//   - It would silently change the meaning of every counter threshold already
//     written. `eccErrors Equal 0` is the archetypal gate of this project and it
//     means exactly zero; a tolerance around zero is a tolerance for ECC errors.
//   - An epsilon small enough to be safe for counters does not rescue the case
//     that motivates it. No clock average lands within 1e-9 of 83.22 either. The
//     tolerance such an author needs is domain knowledge (±0.5 percentage
//     points, say) that only they have, and the API can already say it.
//   - Two dispatchers evaluate these thresholds. An epsilon is invisible in the
//     spec, so a profile's meaning would depend on which evaluator ran it
//     rather than on what the profile says — which is the exact disagreement
//     this package exists to prevent.
//
// The rule is enforced by being discoverable BEFORE a run rather than by being
// silently applied during one: ValidateThresholds reports Equal/NotEqual on a
// continuous metric (and other unusable gates) at authoring time, while the
// author can still fix it. Evaluation itself is unchanged and still fails
// closed.
//
// That guidance is only worth something if something runs it, so in this
// operator three surfaces do. The CRD's own patterns refuse the malformed
// spellings at the apiserver; the reconciler lints the pinned plan before a node
// is cordoned, refusing a threshold nothing could satisfy as a config Error and
// recording an unsound one as an advisory condition on the BurnInRun; and a test
// over config/samples holds the project's own examples to the same bar. A second
// dispatcher owes its authors an equivalent surface — a warning nobody executes
// is not discoverability.
package verdict

import (
	"fmt"
	"math"
	"strconv"
	"strings"

	"github.com/baldwinSPC/glimmer-burnin/pkg/contract"
)

// ReasonUnmeasurable is why a threshold went un-evaluated: the runner declared,
// for this execution on this hardware, that the metric cannot be measured.
// Today it is the only such reason.
const ReasonUnmeasurable = "unmeasurable on this hardware"

// NotEvaluated is a threshold that was neither satisfied nor violated, because
// it was never applied. It is a first-class part of the outcome rather than a
// silent omission: a gate that was not evaluated must never be reportable as a
// gate that passed.
type NotEvaluated struct {
	// Metric is the threshold's metric name.
	Metric string
	// Reason is why it was not evaluated (see ReasonUnmeasurable).
	Reason string
}

// Cause says who should act on a violation. It is the axis that separates a
// statement about the hardware from a statement about the machinery or about
// the profile — a separation that matters because all three arrive as the same
// Failed test, and only one of them is a reason to touch a node.
type Cause string

const (
	// CauseMeasurement is the hardware falling short of a bar that was applied
	// to a real measurement. This is the only cause that is evidence about the
	// part.
	CauseMeasurement Cause = "Measurement"

	// CauseEvidence is a runner report that cannot support a judgement at all:
	// the metric is missing, unparseable, non-finite, self-contradictory, or
	// declared unmeasurable under a gate that requires it. The node is unjudged,
	// not condemned — though the threshold still fails closed.
	CauseEvidence Cause = "Evidence"

	// CauseAuthoring is a broken threshold. No hardware is implicated and no
	// node should be touched; the profile needs fixing. ValidateThresholds
	// catches these before a run, and this cause is what they look like when
	// one slips through anyway.
	CauseAuthoring Cause = "Authoring"
)

// ViolationKind is the specific route by which a threshold went unsatisfied.
// It is finer-grained than Cause so the taxonomy can gain routes without
// reclassifying the ones already there.
type ViolationKind string

const (
	// KindUnsatisfied is a real measurement that failed the comparison.
	KindUnsatisfied ViolationKind = "Unsatisfied"
	// KindNotReported is a metric absent from both the metrics and the
	// unmeasurable set. Absence is not a declaration.
	KindNotReported ViolationKind = "NotReported"
	// KindNonNumeric is a metric value that does not parse as a float64.
	KindNonNumeric ViolationKind = "NonNumeric"
	// KindNonFinite is a metric that parsed as NaN or ±Inf.
	KindNonFinite ViolationKind = "NonFinite"
	// KindContradictory is a metric both reported and declared unmeasurable.
	KindContradictory ViolationKind = "Contradictory"
	// KindUnmeasurableRequired is a metric declared unmeasurable under an
	// applicability that requires it.
	KindUnmeasurableRequired ViolationKind = "UnmeasurableRequired"
	// KindThresholdValueNonNumeric is a threshold whose Value does not parse.
	KindThresholdValueNonNumeric ViolationKind = "ThresholdValueNonNumeric"
	// KindThresholdValueNonFinite is a threshold whose Value is NaN or ±Inf.
	KindThresholdValueNonFinite ViolationKind = "ThresholdValueNonFinite"
)

// Cause reports which cause this kind belongs to. It is the single mapping
// between the two, so a Violation's Kind and Cause can never disagree. An
// unrecognised kind answers CauseEvidence: the conservative reading is that we
// could not establish anything, never that the hardware is at fault.
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

// Violation is one threshold that was not satisfied. Its shape deliberately
// mirrors Problem in lint.go — Index, Metric, a classification, and a Reason
// that is a full sentence — so the package has one vocabulary for "here is
// every problem, classified" rather than two.
type Violation struct {
	// Index is the threshold's position in the slice it was found in.
	Index int
	// Metric is the threshold's metric name, as written.
	Metric string
	// Cause says who should act. Always Kind.Cause().
	Cause Cause
	// Kind is the specific route.
	Kind ViolationKind
	// Reason is the human-readable explanation. For the first violation this is
	// exactly Outcome.Message.
	Reason string
}

// Outcome is the result of evaluating a test's thresholds.
type Outcome struct {
	// Passed reports that no threshold was violated. Un-evaluated thresholds do
	// not make it false — but a caller reporting Passed without also reporting
	// NotEvaluated is misrepresenting the run, which is why they are returned
	// together.
	Passed bool
	// Message names the FIRST violated threshold. Empty when Passed.
	//
	// FROZEN. Every threshold is now evaluated (see Violations), but this field
	// keeps the value it had when evaluation stopped at the first violation,
	// character for character. Callers that want the whole picture read
	// Violations; callers that predate it observe no change.
	Message string
	// NotEvaluated lists the thresholds that were not applied, in spec order.
	//
	// FROZEN, and deliberately INCOMPLETE when a threshold was violated: it
	// holds the un-evaluated thresholds found BEFORE the first violation, which
	// is what stopping there used to yield. A gate sitting after the first
	// violation is absent here even though it too went un-applied — read
	// Violations and this together for the complete picture, or see GEP-0561
	// for why the truncation was kept rather than quietly widened.
	NotEvaluated []NotEvaluated
	// Violations lists EVERY threshold that was not satisfied, in spec order,
	// each classified by Cause. Empty exactly when Passed.
	//
	// This is the field to read. Message is one of these; this is all of them,
	// so an operator sees every gate a node missed in one run instead of
	// rediscovering them one burn-in cycle at a time.
	Violations []Violation
}

// NotEvaluatedMessage renders the un-evaluated thresholds for a human-readable
// report, or "" when every threshold was evaluated.
func (o Outcome) NotEvaluatedMessage() string {
	if len(o.NotEvaluated) == 0 {
		return ""
	}
	parts := make([]string, 0, len(o.NotEvaluated))
	for _, ne := range o.NotEvaluated {
		parts = append(parts, fmt.Sprintf("%s is %s", ne.Metric, ne.Reason))
	}
	return "not evaluated: " + strings.Join(parts, "; ")
}

// maxNamedViolations bounds how many metrics ViolationSummary spells out. The
// summary rides in a status field and a delivered envelope, so it has to stay
// readable for a profile that gates forty metrics; the count is always exact
// even when the names are elided.
const maxNamedViolations = 4

// causeWord renders a Cause for a one-line human summary. It is deliberately
// plain English rather than the constant's name: the reader is an engineer
// deciding whether to walk to a rack, not a consumer of this API.
func causeWord(c Cause) string {
	switch c {
	case CauseMeasurement:
		return "hardware"
	case CauseAuthoring:
		return "profile error"
	default:
		return "not measured"
	}
}

// ViolationSummary renders the violations BEYOND the first for a human-readable
// report, or "" when there are fewer than two — in which case Message already
// says everything there is to say.
//
// It exists here rather than in a caller so both dispatchers render the same
// sentence from the same outcome. It is deliberately NOT "join every reason
// with a semicolon": each violation contributes its metric and its cause, which
// is what a reader needs to decide who to send, and the reasons themselves stay
// in Violations for anyone who wants them.
//
// Naming the cause is the point. A profile error and a thermal ceiling arrive
// as the same Failed test, and only one of them is a reason to touch a node.
func (o Outcome) ViolationSummary() string {
	if len(o.Violations) < 2 {
		return ""
	}
	rest := o.Violations[1:]

	named := rest
	if len(named) > maxNamedViolations {
		named = named[:maxNamedViolations]
	}
	parts := make([]string, 0, len(named))
	for _, v := range named {
		parts = append(parts, fmt.Sprintf("%s (%s)", v.Metric, causeWord(v.Cause)))
	}
	if elided := len(rest) - len(named); elided > 0 {
		parts = append(parts, fmt.Sprintf("and %d more", elided))
	}

	gates := "gates"
	if len(rest) == 1 {
		gates = "gate"
	}
	return fmt.Sprintf("and %d more %s failed: %s", len(rest), gates, strings.Join(parts, ", "))
}

// Evaluate applies a test's thresholds to one execution's results.
//
// metrics are the values the runner reported; unmeasurable is the set of metric
// names the runner EXPLICITLY declared it cannot measure on this hardware (the
// "n/a" sentinel — see pkg/runner). The two are disjoint. A test with no
// thresholds passes: the runner exiting 0 is the only gate.
//
// The semantics, exactly:
//
//   - the metric is present and satisfies the comparison → pass
//   - the metric is present and violates it → FAILURE
//   - the metric is ABSENT from both → FAILURE, always, whatever the
//     Applicability says. A missing measurement must never silently satisfy
//     acceptance, and a runner that forgot to emit is not hardware that cannot
//     measure.
//   - the metric is declared unmeasurable, Applicability Required (the default,
//     and anything unrecognised) → FAILURE. An unmeasured quantity has not been
//     shown to be within limits.
//   - the metric is declared unmeasurable, Applicability RequiredIfMeasurable →
//     the threshold is NOT EVALUATED. It appears in Outcome.NotEvaluated, and
//     it is the caller's job to surface that; it is not a pass.
//
// EVERY threshold is evaluated, and Violations holds all of them. Message and
// NotEvaluated nonetheless report exactly what they reported when evaluation
// stopped at the first violation: Message is Violations[0].Reason, and
// NotEvaluated is truncated at the first violation. That is deliberate — this
// package is public API with two dispatchers, and widening a field they already
// read would change acceptance reporting for both. See GEP-0561.
func Evaluate(
	metrics map[string]string,
	unmeasurable map[string]bool,
	thresholds []contract.Threshold,
) Outcome {
	var out Outcome

	// Accumulated in full; truncated at the first violation on the way out, so
	// the frozen field keeps its old value while Violations stays complete.
	var notEvaluated []NotEvaluated
	notEvaluatedAtFirstViolation := -1

	fail := func(i int, th contract.Threshold, kind ViolationKind, format string, args ...any) {
		if len(out.Violations) == 0 {
			notEvaluatedAtFirstViolation = len(notEvaluated)
		}
		out.Violations = append(out.Violations, Violation{
			Index:  i,
			Metric: th.Metric,
			Cause:  kind.Cause(),
			Kind:   kind,
			Reason: fmt.Sprintf(format, args...),
		})
	}

	for i, th := range thresholds {
		raw, reported := metrics[th.Metric]

		if unmeasurable[th.Metric] {
			if reported {
				// The runner said both things about one metric. Which claim is
				// true is unknowable from here, so the gate fails closed rather
				// than picking the convenient one.
				fail(i, th, KindContradictory, "metric %q was reported as %q AND declared unmeasurable; the runner's output is self-contradictory", th.Metric, raw)
				continue
			}
			// Only the runner's explicit declaration reaches here; absence is
			// handled below and stays a failure regardless of Applicability.
			if th.Applicability == contract.RequiredIfMeasurable {
				notEvaluated = append(notEvaluated, NotEvaluated{
					Metric: th.Metric,
					Reason: ReasonUnmeasurable,
				})
				continue
			}
			fail(i, th, KindUnmeasurableRequired, "metric %q is %s and this threshold is %s; "+
				"set applicability=RequiredIfMeasurable if the gate should be skipped where the hardware cannot measure it",
				th.Metric, ReasonUnmeasurable, applicabilityOf(th))
			continue
		}

		if !reported {
			fail(i, th, KindNotReported, "metric %q was not reported by the runner", th.Metric)
			continue
		}
		got, err := strconv.ParseFloat(raw, 64)
		if err != nil {
			fail(i, th, KindNonNumeric, "metric %q=%q is not numeric", th.Metric, raw)
			continue
		}
		if math.IsNaN(got) || math.IsInf(got, 0) {
			// strconv accepts "NaN" and "Inf". A non-finite value is not a
			// measurement, and it must not be treated as one: NaN compares false
			// against everything, so it fails a floor closed but SATISFIES
			// NotEqual against every value on earth. That is the one way this
			// comparison could hand out an acceptance nobody measured.
			fail(i, th, KindNonFinite, "metric %q=%q is not a finite measurement", th.Metric, raw)
			continue
		}
		want, err := strconv.ParseFloat(th.Value, 64)
		if err != nil {
			fail(i, th, KindThresholdValueNonNumeric, "threshold value %q for %q is not numeric", th.Value, th.Metric)
			continue
		}
		if math.IsNaN(want) || math.IsInf(want, 0) {
			fail(i, th, KindThresholdValueNonFinite, "threshold value %q for %q is not a finite number", th.Value, th.Metric)
			continue
		}
		if !compare(got, th.Comparison, want) {
			fail(i, th, KindUnsatisfied, "%s: got %g, need %s %g", th.Metric, got, th.Comparison, want)
			continue
		}
	}

	if len(out.Violations) == 0 {
		out.Passed = true
		out.NotEvaluated = notEvaluated
		return out
	}
	out.Message = out.Violations[0].Reason
	out.NotEvaluated = notEvaluated[:notEvaluatedAtFirstViolation]
	return out
}

// applicabilityOf reports the threshold's effective applicability. An unset or
// unrecognised value is Required: the relaxation must be asked for explicitly.
func applicabilityOf(th contract.Threshold) contract.Applicability {
	if th.Applicability == contract.RequiredIfMeasurable {
		return contract.RequiredIfMeasurable
	}
	return contract.Required
}

// compare applies the comparison with no tolerance. EQ and NEQ are exact by
// design — they gate dimensionless counters, where exactness is the point, and
// they are the wrong tool for a continuous measurement. The package doc says
// why there is no epsilon, and ValidateThresholds says so to the author of a
// threshold that misuses them. Callers reach this only with finite values.
func compare(got float64, op contract.Comparison, want float64) bool {
	switch op {
	case contract.GTE:
		return got >= want
	case contract.LTE:
		return got <= want
	case contract.EQ:
		return got == want
	case contract.NEQ:
		return got != want
	default:
		return false // unknown comparison fails closed
	}
}
