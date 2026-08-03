// Package verdict evaluates parsed test metrics against a test's thresholds.
// It is deliberately pure (no k8s, no I/O) so the pass/fail logic every
// BurnInRun depends on is unit-testable in isolation.
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
package verdict

import (
	"fmt"
	"math"
	"strconv"
	"strings"

	burninv1alpha1 "github.com/baldwinSPC/glimmer-burnin/api/v1alpha1"
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

// Outcome is the result of evaluating a test's thresholds.
type Outcome struct {
	// Passed reports that no threshold was violated. Un-evaluated thresholds do
	// not make it false — but a caller reporting Passed without also reporting
	// NotEvaluated is misrepresenting the run, which is why they are returned
	// together.
	Passed bool
	// Message names the FIRST violated threshold. Empty when Passed.
	Message string
	// NotEvaluated lists the thresholds that were not applied, in spec order.
	NotEvaluated []NotEvaluated
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
// Evaluation stops at the first violation, so Message names one threshold. The
// un-evaluated thresholds found before it are still returned — they are
// evidence about the run either way.
func Evaluate(
	metrics map[string]string,
	unmeasurable map[string]bool,
	thresholds []burninv1alpha1.Threshold,
) Outcome {
	var out Outcome
	fail := func(format string, args ...any) Outcome {
		out.Passed = false
		out.Message = fmt.Sprintf(format, args...)
		return out
	}

	for _, th := range thresholds {
		raw, reported := metrics[th.Metric]

		if unmeasurable[th.Metric] {
			if reported {
				// The runner said both things about one metric. Which claim is
				// true is unknowable from here, so the gate fails closed rather
				// than picking the convenient one.
				return fail("metric %q was reported as %q AND declared unmeasurable; the runner's output is self-contradictory", th.Metric, raw)
			}
			// Only the runner's explicit declaration reaches here; absence is
			// handled below and stays a failure regardless of Applicability.
			if th.Applicability == burninv1alpha1.RequiredIfMeasurable {
				out.NotEvaluated = append(out.NotEvaluated, NotEvaluated{
					Metric: th.Metric,
					Reason: ReasonUnmeasurable,
				})
				continue
			}
			return fail("metric %q is %s and this threshold is %s; "+
				"set applicability=RequiredIfMeasurable if the gate should be skipped where the hardware cannot measure it",
				th.Metric, ReasonUnmeasurable, applicabilityOf(th))
		}

		if !reported {
			return fail("metric %q was not reported by the runner", th.Metric)
		}
		got, err := strconv.ParseFloat(raw, 64)
		if err != nil {
			return fail("metric %q=%q is not numeric", th.Metric, raw)
		}
		if math.IsNaN(got) || math.IsInf(got, 0) {
			// strconv accepts "NaN" and "Inf". A non-finite value is not a
			// measurement, and it must not be treated as one: NaN compares false
			// against everything, so it fails a floor closed but SATISFIES
			// NotEqual against every value on earth. That is the one way this
			// comparison could hand out an acceptance nobody measured.
			return fail("metric %q=%q is not a finite measurement", th.Metric, raw)
		}
		want, err := strconv.ParseFloat(th.Value, 64)
		if err != nil {
			return fail("threshold value %q for %q is not numeric", th.Value, th.Metric)
		}
		if math.IsNaN(want) || math.IsInf(want, 0) {
			return fail("threshold value %q for %q is not a finite number", th.Value, th.Metric)
		}
		if !compare(got, th.Comparison, want) {
			return fail("%s: got %g, need %s %g", th.Metric, got, th.Comparison, want)
		}
	}
	out.Passed = true
	return out
}

// applicabilityOf reports the threshold's effective applicability. An unset or
// unrecognised value is Required: the relaxation must be asked for explicitly.
func applicabilityOf(th burninv1alpha1.Threshold) burninv1alpha1.Applicability {
	if th.Applicability == burninv1alpha1.RequiredIfMeasurable {
		return burninv1alpha1.RequiredIfMeasurable
	}
	return burninv1alpha1.Required
}

// compare applies the comparison with no tolerance. EQ and NEQ are exact by
// design — they gate dimensionless counters, where exactness is the point, and
// they are the wrong tool for a continuous measurement. The package doc says
// why there is no epsilon, and ValidateThresholds says so to the author of a
// threshold that misuses them. Callers reach this only with finite values.
func compare(got float64, op burninv1alpha1.Comparison, want float64) bool {
	switch op {
	case burninv1alpha1.GTE:
		return got >= want
	case burninv1alpha1.LTE:
		return got <= want
	case burninv1alpha1.EQ:
		return got == want
	case burninv1alpha1.NEQ:
		return got != want
	default:
		return false // unknown comparison fails closed
	}
}
