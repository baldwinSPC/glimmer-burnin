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

// NotEvaluated is a threshold that was neither satisfied nor violated because it
// was never applied.
type NotEvaluated struct {
	Metric string `json:"metric"`
	Reason string `json:"reason"`
}

// Summary is the run's tally, in units of per-(test, node) EXECUTIONS — a
// 2-test profile against 3 nodes finishes with up to 6 in these counters, so
// consumers must not gate on passed == number-of-tests. Skipped and Errored
// executions appear in neither counter; Results carries their phases.
type Summary struct {
	Passed int32 `json:"passed"`
	Failed int32 `json:"failed"`
}

// Envelope is the delivered document.
type Envelope struct {
	Version string `json:"version"`
	// DeliveryID is the idempotency key. It is DERIVED, not random: a retry of
	// the same delivery carries the same ID, so a receiver that has already
	// applied it can discard the duplicate. See NewDeliveryID.
	DeliveryID string    `json:"deliveryId"`
	Reason     Reason    `json:"reason"`
	SentAt     time.Time `json:"sentAt"`

	Run   RunRef `json:"run"`
	Phase string `json:"phase"`
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
