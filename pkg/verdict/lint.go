package verdict

import (
	"fmt"
	"math"
	"strconv"

	burninv1alpha1 "github.com/baldwinSPC/glimmer-burnin/api/v1alpha1"
	"github.com/baldwinSPC/glimmer-burnin/pkg/contract"
)

// Severity says how a caller deciding whether to admit a profile should treat a
// problem. Both are worth refusing; they differ in what goes wrong if nobody
// does.
type Severity string

const (
	// SeverityMalformed is a threshold that cannot be evaluated at all. It fails
	// closed on every node, forever, regardless of the hardware — so it reads as
	// a fleet-wide hardware verdict when it is a typo in a profile.
	SeverityMalformed Severity = "Malformed"

	// SeverityUnsound is a threshold that evaluates but whose verdict does not
	// mean what its author intends. This is the more dangerous of the two,
	// because the result looks exactly like a real judgement: a gate on a metric
	// that is only evidence, or an exact comparison against a continuous
	// measurement, produces a pass or a fail that nobody should act on.
	SeverityUnsound Severity = "Unsound"
)

// Problem is one threshold that cannot decide what its author meant.
type Problem struct {
	// Index is the threshold's position in the slice it was found in, so a
	// caller can point at the offending entry rather than at the test.
	Index int
	// Metric is the threshold's metric name, as written.
	Metric string
	// Severity says which kind of unusable this is.
	Severity Severity
	// Reason is a full sentence naming the fix, not just the fault.
	Reason string
}

func (p Problem) Error() string {
	return fmt.Sprintf("threshold %d (%s, %s): %s", p.Index, p.Metric, p.Severity, p.Reason)
}

// ValidateThresholds reviews thresholds for gates that cannot decide anything,
// and is meant to run at AUTHORING TIME — admission of a BurnInTest or
// BurnInProfile, a CI check over a repository of profiles, a linter in an
// editor — while the author is still there to fix them.
//
// That timing is the whole point. Every problem it reports is one that surfaces
// at verdict time as an ordinary-looking test outcome: a permanent failure that
// reads as a bad part, or a pass that was never really measured. By then a node
// has been condemned or accepted, the author is somewhere else, and the shape of
// the report gives nobody a reason to suspect the profile rather than the
// hardware. The same threshold rejected at admission costs one error message.
//
// It is advisory and it changes nothing about evaluation. Evaluate applies the
// thresholds it is given and still fails closed; this function does not filter,
// rewrite, or relax them, and a caller that ignores it gets exactly today's
// behaviour. An empty result means no problem THIS package can see — a
// threshold can still be wrong about the hardware in ways only its author knows.
//
// It reports every problem it finds rather than the first, including several on
// one threshold: an author fixing a profile should not have to rediscover them
// one run at a time.
func ValidateThresholds(thresholds []burninv1alpha1.Threshold) []Problem {
	var problems []Problem

	for i, th := range thresholds {
		add := func(sev Severity, format string, args ...any) {
			problems = append(problems, Problem{
				Index:    i,
				Metric:   th.Metric,
				Severity: sev,
				Reason:   fmt.Sprintf(format, args...),
			})
		}

		// A name that fails the contract's grammar is not merely untidy: the
		// runner's parser drops metrics it cannot validate, so a threshold
		// naming one waits for a value that can never arrive and fails closed on
		// every node.
		if err := contract.ValidateMetricName(th.Metric); err != nil {
			add(SeverityMalformed,
				"%s; a metric whose name fails the contract grammar is dropped by the runner's parser, so this gate can never be satisfied", err)
		}

		want, valErr := strconv.ParseFloat(th.Value, 64)
		switch {
		case valErr != nil:
			add(SeverityMalformed, "value %q is not numeric; a threshold value is compared as float64", th.Value)
		case math.IsNaN(want) || math.IsInf(want, 0):
			// "NaN" and "Inf" parse. NaN in particular compares false against
			// everything, which quietly turns NotEqual into a gate that always
			// passes.
			add(SeverityMalformed, "value %q is not a finite number; it cannot express a bound", th.Value)
		}

		switch th.Comparison {
		case burninv1alpha1.GTE, burninv1alpha1.LTE:
			// A floor or a ceiling on any metric is sound; the pair of them is
			// how a continuous quantity gets a band.

		case burninv1alpha1.EQ, burninv1alpha1.NEQ:
			// The intended use is a dimensionless counter against a whole
			// number: eccErrors Equal 0. Anything else is asking a continuous
			// quantity to reproduce a decimal string exactly.
			unit := contract.UnitOf(th.Metric)
			switch {
			case unit != contract.UnitNone:
				add(SeverityUnsound,
					"%s and %s compare exactly, with no tolerance, and %q carries the unit %q — it is a continuous measurement, and no run reproduces a decimal exactly, so this gate fails on every healthy node and the failure reads as a hardware verdict. Use %s and/or %s (both, for a band); reserve exact comparison for dimensionless counters such as eccErrors",
					burninv1alpha1.EQ, burninv1alpha1.NEQ, th.Metric, unit, burninv1alpha1.GTE, burninv1alpha1.LTE)
			case valErr == nil && want != math.Trunc(want):
				add(SeverityUnsound,
					"%s and %s compare exactly, and the value %q is not a whole number; exact comparison is for dimensionless counters, whose values are integers. Use %s and/or %s to gate a fractional quantity",
					burninv1alpha1.EQ, burninv1alpha1.NEQ, th.Value, burninv1alpha1.GTE, burninv1alpha1.LTE)
			}

		default:
			add(SeverityMalformed,
				"comparison %q is not one of %s, %s, %s, %s; an unrecognised comparison fails closed, so this gate fails on every node",
				th.Comparison, burninv1alpha1.GTE, burninv1alpha1.LTE, burninv1alpha1.EQ, burninv1alpha1.NEQ)
		}

		// The registry's own opinion about what may decide acceptance. An
		// unregistered name answers true — a third-party runner's author knows
		// what their measurement means and this package does not.
		if !contract.SafeToThresholdOn(th.Metric) {
			reason := "the metric registry classifies it as evidence rather than acceptance"
			if m, ok := contract.Lookup(th.Metric); ok && m.Description != "" {
				reason = m.Description
			}
			add(SeverityUnsound,
				"%q is recorded as evidence and is not safe to decide acceptance on: %s", th.Metric, reason)
		}
	}

	return problems
}
