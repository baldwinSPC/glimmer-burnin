package controller

import (
	"fmt"
	"math"
	"sort"
	"strconv"

	burninv1alpha1 "github.com/baldwinSPC/glimmer-burnin/api/v1alpha1"
	"github.com/baldwinSPC/glimmer-burnin/pkg/contract"
	"github.com/baldwinSPC/glimmer-burnin/pkg/verdict"
)

// ─── Segmented soaks ──────────────────────────────────────────────────────────
//
// A soak is divided into a sequence of shorter pod executions so that an
// eviction, a drain, a kubelet restart or a reboot costs ONE SEGMENT rather than
// the whole run. See BurnInTestSpec.Soak for what the field promises; this file
// holds the arithmetic and the folding, and completeAttempt holds the decision.
//
// The decision is deliberately NOT here. There is exactly one place a verdict is
// rendered for both Node and Pair scope, and adding a second — even one that
// only decides "keep going" — would be the beginning of two answers to the same
// question. Everything in this file is pure: it computes windows, folds maps and
// says which violations are provable, and none of it writes a phase.

// minSegmentSeconds is the floor SoakSpec.SegmentSeconds is validated against.
//
// It is enforced HERE as well as by the CRD's Minimum marker, because the
// reconciler must not assume apiserver defaulting or validation has happened: a
// run built in Go, or one pinned before this field existed, still has to be
// executed under the policy the field documentation promises. A one-second
// segment on a seven-day soak is 604,800 pod creations.
const minSegmentSeconds int32 = 300

// keptPassingSegments is how many uneventful segments stay in the attempt
// history. See TestResult.TruncatedAttempts for what this bounds and why it is
// safe: the verdict reads the persisted aggregate, never the attempt list.
const keptPassingSegments = 16

// segmentSeconds is one segment's window, or 0 when the test is not segmented.
func segmentSeconds(spec *burninv1alpha1.BurnInTestSpec) int32 {
	if spec.Soak == nil || spec.Soak.SegmentSeconds <= 0 {
		return 0
	}
	return spec.Soak.SegmentSeconds
}

// segmentsRequired is how many pod executions a test's soak divides into, or 0
// when it is not segmented.
//
// It rounds UP, and every segment is a full window, so a soak runs at least the
// duration it asked for and may overrun by less than one segment. The
// alternative — a short remainder segment — is worse twice over: it is below the
// floor SegmentSeconds exists to enforce, and a counter summed over a
// ten-second window is not comparable to the same counter summed over a
// fifteen-minute one, which is exactly what the Sum aggregation rule assumes.
func segmentsRequired(spec *burninv1alpha1.BurnInTestSpec) int32 {
	seg := segmentSeconds(spec)
	if seg <= 0 {
		return 0
	}
	duration := spec.DurationSeconds
	if duration <= 0 {
		duration = defaultDurationSeconds
	}
	n := duration / seg
	if duration%seg != 0 {
		n++
	}
	if n < 1 {
		n = 1
	}
	return n
}

// abortEarly reports whether this test asked to stop the moment the aggregate
// provably violates a gate. Nil is false: ending a soak early is a decision an
// author makes, never a default.
func abortEarly(spec *burninv1alpha1.BurnInTestSpec) bool {
	return spec.Soak != nil && spec.Soak.AbortEarly != nil && *spec.Soak.AbortEarly
}

// executionDurationSeconds is how long ONE POD of this test runs for.
//
// This is the single place the segment window reaches the cluster, and both
// consumers must read it or they disagree about the same pod: podForTest sizes
// BURNIN_DURATION_SECONDS and activeDeadlineSeconds from it, and podOverdue
// sizes the operator-side window it reaps an unschedulable pod after. A
// segmented soak whose pod deadline came from the whole duration would be
// segmented in name only — the eviction would still cost the week, because
// nothing would end the pod at its segment boundary.
func executionDurationSeconds(spec *burninv1alpha1.BurnInTestSpec) int32 {
	if seg := segmentSeconds(spec); seg > 0 {
		return seg
	}
	if spec.DurationSeconds > 0 {
		return spec.DurationSeconds
	}
	return defaultDurationSeconds
}

// ─── Folding ──────────────────────────────────────────────────────────────────

// foldMetrics combines one clean segment's metrics into the running aggregate.
//
// HOW a metric combines is declared beside its NAME, in
// pkg/contract.AggregationFor, and is never decided here. That is the whole
// reason the hint exists: a per-metric switch inside the reconciler would be
// contract-shaped knowledge in the one place this project keeps it out of, and
// it would have to be rewritten by anyone adding a metric. An unregistered name
// answers Last, which is the honest default — inventing Sum semantics for
// somebody else's measurement would be this operator deciding what it means.
//
// The result is a fresh map. The aggregate is read back out of a persisted
// status and handed to the evaluator, and sharing it with whatever the caller
// passed in would let a later parse rewrite a verdict's inputs.
func foldMetrics(agg, next map[string]string) map[string]string {
	if len(next) == 0 {
		return agg
	}
	out := make(map[string]string, len(agg)+len(next))
	for name, raw := range agg {
		out[name] = raw
	}
	for name, raw := range next {
		prior, seen := out[name]
		if !seen {
			out[name] = raw
			continue
		}
		out[name] = combineMetric(contract.AggregationFor(name), prior, raw)
	}
	return out
}

// combineMetric applies one aggregation rule to two readings of one metric.
//
// Min and Max return one of the two ORIGINAL STRINGS rather than a reformatted
// number, so a runner's own spelling of its measurement survives the fold and a
// stored verdict still shows the value the runner printed.
//
// A value that is not a FINITE number poisons the metric for Sum, Min and Max,
// and the poison is deliberately sticky: a segment that produced something
// nothing can add or order has broken the evidence for that metric, and letting
// a later clean segment launder it away would turn a broken measurement into an
// acceptance. verdict.Evaluate fails such a value closed under every comparison,
// which is where the poison is meant to land. Last is exempt because Last means
// last — it is the rule for a nameplate or a lifetime total the runner reports
// absolutely, and there the newest reading is the answer.
func combineMetric(rule contract.Aggregation, prior, next string) string {
	switch rule {
	case contract.AggSum, contract.AggMin, contract.AggMax:
	default:
		return next
	}

	a, aOK := finiteValue(prior)
	if !aOK {
		return prior
	}
	b, bOK := finiteValue(next)
	if !bOK {
		return next
	}

	switch rule {
	case contract.AggSum:
		return strconv.FormatFloat(a+b, 'f', -1, 64)
	case contract.AggMin:
		if b < a {
			return next
		}
		return prior
	case contract.AggMax:
		if b > a {
			return next
		}
		return prior
	}
	return next
}

func finiteValue(raw string) (float64, bool) {
	v, err := strconv.ParseFloat(raw, 64)
	if err != nil || math.IsNaN(v) || math.IsInf(v, 0) {
		return 0, false
	}
	return v, true
}

// foldUnmeasurable accumulates what the runner DECLARED it could not measure.
//
// A name stays in the set only while no segment has produced a value for it: if
// one segment declared eccErrors unmeasurable and another reported it, the part
// evidently can measure it, and the gate applies to the aggregate. Keeping both
// claims would make verdict.Evaluate report the runner as self-contradictory,
// which would be this operator inventing a contradiction out of two honest
// readings of two different windows.
func foldUnmeasurable(prior []string, declared map[string]bool, agg map[string]string) []string {
	set := make(map[string]bool, len(prior)+len(declared))
	for _, name := range prior {
		set[name] = true
	}
	for name := range declared {
		set[name] = true
	}
	out := make([]string, 0, len(set))
	for name := range set {
		if _, measured := agg[name]; measured {
			continue
		}
		out = append(out, name)
	}
	sort.Strings(out)
	if len(out) == 0 {
		return nil
	}
	return out
}

// unmeasurableSet is the evaluator's shape of the accumulated declaration.
func unmeasurableSet(names []string) map[string]bool {
	if len(names) == 0 {
		return nil
	}
	out := make(map[string]bool, len(names))
	for _, name := range names {
		out[name] = true
	}
	return out
}

// ─── Aborting early ───────────────────────────────────────────────────────────

// provableViolations keeps only the violations of the aggregate SO FAR that no
// later segment could retract.
//
// This is what makes AbortEarly safe to have at all. The aggregate is a moving
// object: a Min falls as segments arrive, a Max rises, a Sum grows. A violation
// is provable exactly when the aggregation is monotone IN THE GATED DIRECTION,
// so the four cases below are the whole list and each is a one-line proof —
//
//	Sum under LessThanOrEqual   a count only grows, so a sum already over its
//	                            ceiling is over it forever.
//	Sum under Equal             once the sum has PASSED the value it can never
//	                            return to it; a sum still short of it can.
//	Min under GreaterThanOrEqual  a minimum only falls, so a floor already
//	                            broken stays broken. The worst window IS the
//	                            verdict, which is what Min means.
//	Max under LessThanOrEqual   a maximum only rises, so a ceiling already
//	                            exceeded stays exceeded.
//
// Everything else stays silent, and the case that matters is a Min-aggregated
// gauge under a ceiling: a later segment can pull the minimum back under the
// bound, so a violation seen at segment three is not a verdict. Aborting a week
// of burn-in on a reading the next fifteen minutes would have retracted is the
// wrong trade in both directions — it wastes the burn-in and it condemns the
// part.
//
// Only a MEASUREMENT violation qualifies. A metric no segment has reported yet
// is Evidence, not a shortfall, and a later segment may well report it; a broken
// threshold is Authoring and implicates no hardware at all. Fail-closed still
// applies at the END of the soak, where an absence really is final.
func provableViolations(
	agg map[string]string,
	out verdict.Outcome,
	thresholds []burninv1alpha1.Threshold,
) []verdict.Violation {
	var provable []verdict.Violation
	for _, v := range out.Violations {
		if v.Cause != verdict.CauseMeasurement {
			continue
		}
		if v.Index < 0 || v.Index >= len(thresholds) {
			continue
		}
		if monotoneBreach(agg, thresholds[v.Index]) {
			provable = append(provable, v)
		}
	}
	return provable
}

// monotoneBreach is the four-case proof above, applied to one threshold.
func monotoneBreach(agg map[string]string, th burninv1alpha1.Threshold) bool {
	switch contract.AggregationFor(th.Metric) {
	case contract.AggSum:
		switch th.Comparison {
		case burninv1alpha1.LTE:
			return true
		case burninv1alpha1.EQ:
			// Only once the sum is PAST the value. Short of it, the next segment
			// can still land on it exactly — which is the ordinary case for
			// `Equal 0`, where any sum above zero is already unreachable.
			got, gotOK := finiteValue(agg[th.Metric])
			want, wantOK := finiteValue(th.Value)
			return gotOK && wantOK && got > want
		}
	case contract.AggMin:
		return th.Comparison == burninv1alpha1.GTE
	case contract.AggMax:
		return th.Comparison == burninv1alpha1.LTE
	}
	return false
}

// ─── Attempt history ──────────────────────────────────────────────────────────

// truncateAttempts caps a segmented result's attempt history.
//
// It keeps the FIRST attempt, EVERY non-passing attempt, and the most recent
// keptPassingSegments passing ones; the rest are counted in TruncatedAttempts.
// A 72-hour soak at fifteen-minute segments is 288 attempt records, each with a
// metrics map, in a status that lives in etcd and is copied into a
// 900 KiB-capped ConfigMap sink.
//
// It is safe precisely because the verdict is read from AggregatedMetrics, which
// is persisted, and never from the attempt list. Two things it must never drop
// and does not: the non-passing attempts, which are already bounded by the retry
// budget and are the whole reason the history is worth keeping, and the LAST
// attempt, which nextAttempt reads to decide which segment comes next — dropping
// it would restart a seven-day soak at segment one.
//
// It applies to segmented results only. A repeated test's history is bounded by
// a RepeatCount its author wrote down; a soak's is derived from a duration and
// can be hundreds of records nobody asked for.
func truncateAttempts(res *burninv1alpha1.TestResult) {
	if res.SegmentsRequired <= 0 || len(res.Attempts) <= keptPassingSegments+1 {
		return
	}

	var passing []int
	for i := range res.Attempts {
		if i == 0 {
			continue
		}
		if res.Attempts[i].Phase == burninv1alpha1.RunPassed {
			passing = append(passing, i)
		}
	}
	drop := len(passing) - keptPassingSegments
	if drop <= 0 {
		return
	}
	dropped := make(map[int]bool, drop)
	for _, i := range passing[:drop] {
		dropped[i] = true
	}

	kept := make([]burninv1alpha1.TestAttempt, 0, len(res.Attempts)-drop)
	for i := range res.Attempts {
		if dropped[i] {
			continue
		}
		kept = append(kept, res.Attempts[i])
	}
	res.Attempts = kept
	res.TruncatedAttempts += int32(drop)
}

// ─── Plan-time refusals ───────────────────────────────────────────────────────

// validateSoak refuses a segmented test that could not mean what it says.
//
// Every rule refuses at START, before a node is cordoned, for the reason
// validateTopology does: a soak that is wrong about its own arithmetic does not
// fail loudly, it produces a plausible-looking verdict about a burn-in that did
// not happen, and days later.
func validateSoak(t plannedTest) error {
	s := t.Spec.Soak
	if s == nil {
		return nil
	}
	if t.Spec.Kind.BurstOnly() {
		return fmt.Errorf(
			"test %q sets spec.soak but kind %q is burst-only: its runner does one bounded burst of work and "+
				"ignores BURNIN_DURATION_SECONDS entirely, so segmenting it would run the same millisecond-long "+
				"measurement over and over and report it as a soak. Pair it with a duration-bearing kind in the "+
				"profile instead — a burst is a correctness gate and a soak is a different test",
			t.Name, t.Spec.Kind)
	}
	if s.SegmentSeconds < minSegmentSeconds {
		return fmt.Errorf(
			"test %q sets spec.soak.segmentSeconds to %d, below the %d-second floor — a segment boundary costs a "+
				"pod teardown, a pod creation and a scheduling round-trip, and below that the run spends more time "+
				"changing pods than measuring hardware",
			t.Name, s.SegmentSeconds, minSegmentSeconds)
	}
	if t.Spec.DurationSeconds <= 0 {
		return fmt.Errorf(
			"test %q sets spec.soak but no spec.durationSeconds — a soak that does not say how long it soaks cannot "+
				"be divided into segments, and defaulting it would decide the length of somebody's burn-in for them",
			t.Name)
	}
	if s.SegmentSeconds > t.Spec.DurationSeconds {
		return fmt.Errorf(
			"test %q asks for %d-second segments of a %d-second soak — a segment longer than the soak is a soak "+
				"that burns the fleet longer than it was asked to; lower spec.soak.segmentSeconds, or raise "+
				"spec.durationSeconds to what you actually want burned in",
			t.Name, s.SegmentSeconds, t.Spec.DurationSeconds)
	}
	if n := repeatsRequired(&t.Spec); n > 1 {
		return fmt.Errorf(
			"test %q sets both spec.soak and spec.repeatCount %d, and this operator will not guess which one the "+
				"verdict is about — a repeat re-runs a whole test and ANDs the verdicts, while a segment is one "+
				"window of a single verdict rendered over the aggregate. Express the length you want as "+
				"spec.durationSeconds and leave repeatCount at 1",
			t.Name, n)
	}
	return nil
}
