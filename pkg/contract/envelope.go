// Package contract defines the versioned document this operator delivers to a
// BurnInSink, and the metric-naming rules that document's contents obey.
//
// It is deliberately free of Kubernetes types. A consumer of burn-in results
// needs to decode an envelope, not to reconcile a cluster; making it import
// client-go to read a verdict would be a tax on every adopter. Everything here
// is plain Go with JSON tags, so the package is cheap to depend on.
//
// This is the ONLY integration seam with an external control plane. Changing a
// field's meaning changes it for every consumer at once, so the envelope is
// versioned and additive: add fields, do not repurpose them.
package contract

import (
	"crypto/sha256"
	"encoding/hex"
	"strconv"
	"time"
)

// Version identifies the envelope schema. Consumers should reject an envelope
// whose Version they do not recognise rather than guess at its shape.
const Version = "burnin.glimmer.ai/v1alpha1"

// Reason is why a delivery was sent. It exists so a receiver can tell a routine
// progress update from a terminal verdict without diffing state.
type Reason string

const (
	// ReasonPhaseChanged is sent when a run enters a new phase, including the
	// terminal one. This is the delivery a consumer gates acceptance on.
	ReasonPhaseChanged Reason = "RunPhaseChanged"
	// ReasonTestCompleted is sent when a single test within a run finishes.
	ReasonTestCompleted Reason = "TestCompleted"
	// ReasonCheckpoint is a periodic cumulative summary of a still-running run.
	// A multi-day thermal soak would otherwise deliver nothing between start and
	// finish, leaving a consumer unable to distinguish "soaking" from "wedged".
	ReasonCheckpoint Reason = "Checkpoint"
)

// Reasons is every delivery reason this contract defines.
//
// It exists so a test can be TOTAL over them. A test that hard-codes the list
// stops covering the day somebody adds a fourth, and the properties asserted
// per reason — that the summary tallies every execution, that the envelope
// validates — are exactly the ones a new delivery path would be easiest to get
// wrong on and hardest to notice.
//
// Adding a Reason means adding it here, and TestReasonsIsComplete reads this
// file's own constants to make sure you did — otherwise forgetting would
// silently narrow every test that ranges over it, which is the failure mode a
// list like this exists to prevent and the one it would otherwise cause.
var Reasons = []Reason{ReasonPhaseChanged, ReasonTestCompleted, ReasonCheckpoint}

// RunRef identifies the run a delivery describes.
type RunRef struct {
	Namespace string `json:"namespace"`
	Name      string `json:"name"`
	// UID is the Kubernetes object UID. It is the stable identity across the
	// run's lifetime; Name alone can be reused after a run is garbage-collected.
	UID string `json:"uid"`
	// Profile is the BurnInProfile this run executed.
	Profile string `json:"profile,omitempty"`
}

// TestResult is one test's outcome, flattened out of the CRD status.
type TestResult struct {
	Name       string     `json:"name"`
	Kind       string     `json:"kind"`
	Scope      string     `json:"scope,omitempty"`
	Phase      string     `json:"phase"`
	Nodes      []string   `json:"nodes,omitempty"`
	StartedAt  *time.Time `json:"startedAt,omitempty"`
	FinishedAt *time.Time `json:"finishedAt,omitempty"`
	// Metrics are the parsed numeric results thresholds were evaluated against.
	// Keys obey the naming rules in metrics.go.
	Metrics map[string]string `json:"metrics,omitempty"`
	// Message is the human-readable outcome: the failing threshold, or the
	// underlying error when Phase is Error.
	//
	// It names only the FIRST violation, because that field is frozen. Violations
	// below is the complete picture; a consumer must not parse this.
	Message string `json:"message,omitempty"`

	// Violations are every threshold this test failed, in spec order.
	//
	// Before this existed a consumer saw one sentence of prose and had to parse
	// English to learn whether a node was implicated — and a run that failed
	// three gates for three different reasons delivered as one sentence about the
	// first. Read Cause before acting: a Failed test can mix a hardware shortfall
	// with a broken threshold, and only the former is a reason to touch a node.
	Violations []Violation `json:"violations,omitempty"`

	// NotEvaluated are thresholds that were never applied, so a consumer can tell
	// a gate that did not run from a gate that passed without reading prose.
	NotEvaluated []NotEvaluated `json:"notEvaluated,omitempty"`

	// Unmeasurable are the metric names the runner positively declared it cannot
	// measure on this hardware. A claim about the PART: "this part has no ECC"
	// and "this part reported zero ECC errors" are different statements, and a
	// consumer that cannot distinguish them will eventually treat one as the
	// other.
	Unmeasurable []string `json:"unmeasurable,omitempty"`

	// Artifacts are the non-metric evidence the runner returned, by reference.
	Artifacts []ArtifactRef `json:"artifacts,omitempty"`

	// VariantAxes are the matrix labels this execution came from — {precision:
	// fp4}, {class: soak} — or empty for a test with no variants. A consumer
	// groups a sweep's cells by these rather than by parsing test names, which
	// stops working the moment a variant is called "fp8-dense".
	VariantAxes map[string]string `json:"variantAxes,omitempty"`

	// Segments describes a SEGMENTED SOAK, and is absent for an ordinary
	// execution — absence is the signal, so a consumer reads `null` as "this was
	// one pod" rather than comparing a zero against a sentinel.
	//
	// Without it, "40 of 288 segments" and "288 of 288" arrive looking identical,
	// and they are different statements about a node. Metrics alone cannot
	// separate them: the same map is a fold across 288 windows in one case and a
	// single reading in the other, and a consumer has no way to know which it is
	// holding.
	Segments *SegmentSummary `json:"segments,omitempty"`
}

// Violation is one threshold a test failed.
//
// It mirrors api/v1alpha1.Violation and pkg/verdict's violation shape. The
// duplication is deliberate — this package must stay free of Kubernetes types so
// a consumer can decode a verdict without client-go — and the three are held in
// step by TestMirroredStructsAgree rather than by anyone remembering.
type Violation struct {
	// Index is the threshold's position in the test's spec.thresholds.
	Index int32 `json:"index"`
	// Metric is the threshold's metric name, as written in the profile.
	Metric string `json:"metric"`
	// Cause says WHO SHOULD ACT and is the field to read first:
	// Measurement (the hardware fell short), Evidence (the runner's report could
	// not support a judgement — the node is unjudged, not condemned), or
	// Authoring (the threshold itself is broken; no node should be touched).
	Cause string `json:"cause"`
	// Kind is the specific route, finer-grained than Cause.
	Kind string `json:"kind"`
	// Reason is the human-readable detail.
	Reason string `json:"reason,omitempty"`
}

// ArtifactRef points at one piece of non-metric evidence a runner returned.
//
// Mirrors v1alpha1.ArtifactRef; internal/sink.TestMirroredStructsAgree keeps the
// two in step. The payload is NOT in the envelope: a consumer that wants it
// fetches the named ConfigMap, and one that only wants the verdict is not made
// to carry a megabyte of dcgmi JSON to get it.
type ArtifactRef struct {
	Name      string `json:"name"`
	MediaType string `json:"mediaType,omitempty"`
	SizeBytes int64  `json:"sizeBytes,omitempty"`
	Digest    string `json:"digest,omitempty"`
	ConfigMap string `json:"configMap,omitempty"`
	Key       string `json:"key,omitempty"`
	Dropped   string `json:"dropped,omitempty"`
}

// SegmentSummary says how a segmented soak was executed, and how much of it
// actually burned.
//
// A soak is divided into a sequence of shorter pods so that an eviction costs
// one window instead of the week. That is a SCHEDULING decision and it
// deliberately does not change the verdict — thresholds are applied once, at the
// end, to the fold — so a consumer cannot tell from Phase or Metrics how a test
// was run. This is where it is told.
//
// HOW THE METRICS COMBINED is not repeated here. Each metric's rule is declared
// beside its name in the registry and answered by AggregationFor, which is a
// single source of truth a consumer can query; copying 40 rules onto every
// result would let the envelope and the registry disagree about the same name.
// A consumer reading a result whose Segments is non-nil should treat Metrics as
// a FOLD and call AggregationFor to learn how each key was combined.
type SegmentSummary struct {
	// Completed counts only the segments that finished CLEANLY and contributed to
	// the fold. It does not advance for an interruption, so
	// Completed < Required on a terminal result means the soak was cut short —
	// which is the fact this whole type exists to make reachable.
	Completed int32 `json:"completed"`
	// Required is how many segments the declared duration was divided into.
	Required int32 `json:"required"`
	// SegmentSeconds is the length of one window. Present so a consumer can
	// reconstruct how much time Completed represents without knowing the run's
	// spec, which it does not receive.
	SegmentSeconds int32 `json:"segmentSeconds,omitempty"`
	// TruncatedAttempts is how many passing segments were dropped from the
	// attempt history to keep the status writable.
	//
	// Without it a consumer counting delivered attempts silently undercounts, and
	// never learns that it did. The verdict is unaffected — it is read from the
	// persisted fold and never from the attempt list — but "there were 288 of
	// these and you are seeing 17" is something only the operator can say.
	TruncatedAttempts int32 `json:"truncatedAttempts,omitempty"`
}

// NotEvaluated is a threshold that was neither satisfied nor violated because it
// was never applied.
type NotEvaluated struct {
	Metric string `json:"metric"`
	Reason string `json:"reason"`
}

// Summary is the run's tally, in units of per-(test, node) EXECUTIONS — a
// 2-test profile against 3 nodes finishes with up to 6 in these counters, so
// consumers must not gate on passed == number-of-tests.
//
// ALL FOUR TERMINAL OUTCOMES ARE COUNTED, one entry per Results element. That
// changed in v0.6.0 and the change is the reason this paragraph is worded
// carefully: through v0.5.0 the struct carried only Passed and Failed, and this
// comment told consumers that "Skipped and Errored executions appear in neither
// counter; Results carries their phases". A consumer who followed that advice
// wrote a walk over Results to recover them — correctly — and if it now ALSO
// adds the new counters it double-counts every execution that did not pass.
//
// So: read the counters, or walk Results, but not both. The two are the same
// tally by construction.
//
// The four are kept separate rather than folded together for the reason
// recount() keeps them separate in the status: an Errored execution measured
// nothing and a Skipped one did not apply, and neither is a hardware verdict.
//
// On a CHECKPOINT the run has not finished, so executions still in flight are
// counted in none of the four and the sum is less than the number of Results.
type Summary struct {
	Passed int32 `json:"passed"`
	Failed int32 `json:"failed"`
	// Errored and Skipped complete the tally. Without them the summary counted
	// two of the four phases and could not answer the question every consumer
	// asks first — "how many executions did not pass" — which is the one thing a
	// summary is for. A summary reading "0 failed" beside eight executions that
	// never ran would be a clean sweep of hardware nobody measured.
	Errored int32 `json:"errored"`
	Skipped int32 `json:"skipped"`
}

// ClusterRef identifies the cluster a delivery came from.
//
// Ingest is per-cluster and authenticated per-cluster, so a receiver can infer
// origin from the credential — which works right up until anyone aggregates,
// replays or archives envelopes, at which point the document is not
// self-describing. A stored envelope should be portable evidence on its own.
//
// Both fields are optional and an operator with neither configured emits
// neither. Name is NEVER guessed: it comes from an explicit --cluster-name flag,
// because a guessed cluster identity that is wrong is worse than an absent one —
// it attributes a verdict to the wrong fleet.
type ClusterRef struct {
	Name string `json:"name,omitempty"`
	// UID is the kube-system namespace UID, which is the closest thing
	// Kubernetes has to a stable cluster identity. Read once at manager start.
	UID string `json:"uid,omitempty"`
}

// Envelope is the delivered document.
type Envelope struct {
	Version string `json:"version"`

	// Baseline says this run MEASURED rather than certified.
	//
	// Without it a thresholdless sweep and an acceptance run both deliver
	// phase Passed, and a consumer gating admission on "the last run passed"
	// certifies hardware against a run that gated nothing. The operator refuses
	// to combine the flag with thresholds, so a true value here means the run
	// carried no gates at all — it is a statement about what the verdict is
	// WORTH, not a hint.
	Baseline bool `json:"baseline,omitempty"`
	// DeliveryID is the idempotency key. It is DERIVED, not random: a retry of
	// the same delivery carries the same ID, so a receiver that has already
	// applied it can discard the duplicate. See NewDeliveryID.
	DeliveryID string    `json:"deliveryId"`
	Reason     Reason    `json:"reason"`
	SentAt     time.Time `json:"sentAt"`

	Run RunRef `json:"run"`
	// Cluster identifies where this delivery came from. Optional; see ClusterRef.
	Cluster *ClusterRef `json:"cluster,omitempty"`
	Phase   string      `json:"phase"`

	// CancelReason is the reason recorded on a cancelled run.
	//
	// Present only when Phase is Cancelled. Without it a cancelled delivery gives
	// a consumer no way to tell an operator-requested stop from a deadline
	// expiry, and those mean opposite things about the hardware: one is a human
	// deciding to stop, the other is a test that ran out of time and measured
	// nothing.
	CancelReason string `json:"cancelReason,omitempty"`

	// CheckpointSequence orders checkpoints within one run.
	//
	// Present only when Reason is Checkpoint. The value already exists — it is
	// what makes DeliveryID stable across retries — and without it a consumer
	// receiving checkpoints out of order cannot put them back in order, which is
	// exactly what a long soak's progress record is for.
	CheckpointSequence int `json:"checkpointSequence,omitempty"`
	// Fingerprint is the hardware this verdict applies to, captured at run
	// start. A verdict without it is not portable evidence.
	Fingerprint map[string]string `json:"fingerprint,omitempty"`
	Results     []TestResult      `json:"results,omitempty"`
	Summary     Summary           `json:"summary"`
}

// NewDeliveryID derives the idempotency key for a delivery.
//
// It must be a pure function of what the delivery describes — never of the
// wall clock or a random source — because retries depend on it being stable.
// eventKey distinguishes deliveries within a run: the target phase for a phase
// change, the test name for a test completion, the sequence number for a
// checkpoint.
func NewDeliveryID(runUID string, reason Reason, eventKey string) string {
	sum := sha256.Sum256([]byte(runUID + "\x00" + string(reason) + "\x00" + eventKey))
	// 128 bits is far past collision concerns for a per-run key, and a shorter
	// string keeps logs and receiver-side dedupe tables readable.
	return hex.EncodeToString(sum[:16])
}

// Validate reports whether an envelope is well-formed enough to deliver.
// It is applied before sending: shipping a malformed document to a consumer's
// ingest endpoint is worse than failing locally where the error is visible.
func (e *Envelope) Validate() error {
	switch {
	case e.Version == "":
		return &ValidationError{Field: "version", Reason: "must be set"}
	case e.Version != Version:
		return &ValidationError{Field: "version", Reason: "unknown envelope version " + e.Version}
	case e.DeliveryID == "":
		return &ValidationError{Field: "deliveryId", Reason: "must be set; retries depend on a stable key"}
	case e.Reason == "":
		return &ValidationError{Field: "reason", Reason: "must be set"}
	case e.Run.UID == "":
		return &ValidationError{Field: "run.uid", Reason: "must be set; name alone is not stable identity"}
	case e.SentAt.IsZero():
		return &ValidationError{Field: "sentAt", Reason: "must be set"}
	}
	for i, r := range e.Results {
		for name := range r.Metrics {
			if err := ValidateMetricName(name); err != nil {
				return &ValidationError{
					Field:  "results[" + strconv.Itoa(i) + "].metrics",
					Reason: err.Error(),
				}
			}
		}
	}
	return nil
}

// ValidationError names the offending field so a failure is actionable from a
// log line alone.
type ValidationError struct {
	Field  string
	Reason string
}

func (e *ValidationError) Error() string {
	return "invalid envelope: " + e.Field + ": " + e.Reason
}
