package sink

import (
	"strconv"
	"time"

	burninv1alpha1 "github.com/baldwinSPC/glimmer-burnin/api/v1alpha1"
	"github.com/baldwinSPC/glimmer-burnin/pkg/contract"
)

// EnvelopeFor builds the delivery document for a run.
//
// reason and eventKey together identify the delivery, and the eventKey must be
// derived from what happened rather than from when it happened: the DeliveryID
// is hashed from it, and a key that moves with the clock would defeat the
// receiver's ability to dedupe a retry. Use PhaseKey, TestKey, or CheckpointKey
// rather than composing one by hand.
func EnvelopeFor(
	run *burninv1alpha1.BurnInRun,
	profile string,
	reason contract.Reason,
	eventKey string,
	now time.Time,
	cluster *contract.ClusterRef,
	baseline bool,
) *contract.Envelope {
	env := &contract.Envelope{
		Version:    contract.Version,
		Baseline:   baseline,
		DeliveryID: contract.NewDeliveryID(string(run.UID), reason, eventKey),
		Reason:     reason,
		SentAt:     now.UTC(),
		Run: contract.RunRef{
			Namespace: run.Namespace,
			Name:      run.Name,
			UID:       string(run.UID),
			Profile:   profile,
		},
		Phase:       string(run.Status.Phase),
		Fingerprint: run.Status.Fingerprint,
		Cluster:     cluster,
		Summary: contract.Summary{
			Passed:  run.Status.Passed,
			Failed:  run.Status.Failed,
			Errored: run.Status.Errored,
			Skipped: run.Status.Skipped,
		},
	}
	// Present only where they mean something, so a consumer can read their
	// presence as a signal rather than checking a sentinel.
	if run.Status.Phase == burninv1alpha1.RunCancelled {
		env.CancelReason = run.Spec.CancelReason
	}
	if reason == contract.ReasonCheckpoint {
		// The sequence is already the event key — it is what makes DeliveryID
		// stable across retries — so it is derived here rather than threaded
		// through another parameter that could disagree with it.
		if n, err := strconv.Atoi(eventKey); err == nil {
			env.CheckpointSequence = n
		}
	}
	for _, r := range run.Status.Results {
		env.Results = append(env.Results, testResult(r))
	}
	return env
}

func testResult(r burninv1alpha1.TestResult) contract.TestResult {
	out := contract.TestResult{
		Name:    r.Name,
		Kind:    string(r.Kind),
		Scope:   string(r.Scope),
		Phase:   string(r.Phase),
		Nodes:   r.Nodes,
		Metrics: r.Metrics,
		Message: r.Message,
		// The structured half of the verdict. Message names only the first
		// violation and is frozen there; without these a consumer had to parse
		// English to learn whether a node was implicated, and a run that failed
		// three gates delivered as one sentence about the first.
		Unmeasurable: append([]string(nil), r.Unmeasurable...),
		VariantAxes:  r.VariantAxes,
	}
	// Present only for a segmented soak, so a consumer reads its absence as "this
	// was one pod" rather than comparing a zero against a sentinel. Keyed on
	// SegmentsRequired, which is the same flag the reconciler branches on — a
	// result carries it if and only if its verdict was rendered over the fold.
	if r.SegmentsRequired > 0 {
		out.Segments = &contract.SegmentSummary{
			Completed:         r.SegmentsCompleted,
			Required:          r.SegmentsRequired,
			SegmentSeconds:    r.SegmentSeconds,
			TruncatedAttempts: r.TruncatedAttempts,
		}
	}
	for _, v := range r.Violations {
		out.Violations = append(out.Violations, contract.Violation{
			Index: v.Index, Metric: v.Metric, Cause: v.Cause, Kind: v.Kind, Reason: v.Reason,
		})
	}
	for _, a := range r.Applied {
		out.Applied = append(out.Applied, contract.AppliedGate{
			Index: a.Index, Metric: a.Metric, Comparison: a.Comparison, Value: a.Value,
		})
	}
	for _, a := range r.Artifacts {
		// By reference, never by value. A consumer that only wants the verdict
		// is not made to carry a megabyte of dcgmi JSON to get it; one that
		// wants the evidence fetches the named ConfigMap and checks the digest.
		// A ref whose Dropped is set is delivered too — evidence that was
		// refused is itself a fact about the run.
		out.Artifacts = append(out.Artifacts, contract.ArtifactRef{
			Name: a.Name, MediaType: a.MediaType, SizeBytes: a.SizeBytes,
			Digest: a.Digest, ConfigMap: a.ConfigMap, Key: a.Key, Dropped: a.Dropped,
		})
	}
	for _, n := range r.NotEvaluated {
		out.NotEvaluated = append(out.NotEvaluated, contract.NotEvaluated{
			Metric: n.Metric, Reason: n.Reason,
		})
	}
	if r.StartedAt != nil {
		t := r.StartedAt.Time.UTC()
		out.StartedAt = &t
	}
	if r.FinishedAt != nil {
		t := r.FinishedAt.Time.UTC()
		out.FinishedAt = &t
	}
	return out
}

// PhaseKey identifies a phase-change delivery. A run enters each phase once, so
// the phase alone is a stable key.
func PhaseKey(phase burninv1alpha1.RunPhase) string { return string(phase) }

// TestKey identifies a test-completion delivery. Test names are unique within a
// profile, so the name is stable across retries of the same completion.
func TestKey(testName string) string { return testName }

// CheckpointKey identifies the nth checkpoint of a run.
//
// Checkpoints are numbered rather than timestamped precisely so a retry keeps
// its identity: a timestamped key would mint a new DeliveryID on every attempt
// and defeat deduplication, turning a flaky endpoint into a flood of
// near-identical records.
func CheckpointKey(sequence int) string { return strconv.Itoa(sequence) }
