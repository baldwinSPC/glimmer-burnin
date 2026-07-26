// Package verdict evaluates parsed test metrics against a test's thresholds.
// It is deliberately pure (no k8s, no I/O) so the pass/fail logic every
// BurnInRun depends on is unit-testable in isolation.
package verdict

import (
	"fmt"
	"strconv"

	burninv1alpha1 "github.com/baldwinSPC/glimmer-burnin/api/v1alpha1"
)

// Evaluate reports whether every threshold passes against metrics, and a
// human-readable message naming the first failure. A test with no thresholds
// passes (the runner exiting 0 is the only gate). A threshold referencing a
// metric the runner did not emit FAILS closed — a missing measurement must not
// silently pass acceptance.
func Evaluate(metrics map[string]string, thresholds []burninv1alpha1.Threshold) (bool, string) {
	for _, th := range thresholds {
		raw, ok := metrics[th.Metric]
		if !ok {
			return false, fmt.Sprintf("metric %q was not reported by the runner", th.Metric)
		}
		got, err := strconv.ParseFloat(raw, 64)
		if err != nil {
			return false, fmt.Sprintf("metric %q=%q is not numeric", th.Metric, raw)
		}
		want, err := strconv.ParseFloat(th.Value, 64)
		if err != nil {
			return false, fmt.Sprintf("threshold value %q for %q is not numeric", th.Value, th.Metric)
		}
		if !compare(got, th.Comparison, want) {
			return false, fmt.Sprintf("%s: got %g, need %s %g", th.Metric, got, th.Comparison, want)
		}
	}
	return true, ""
}

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
