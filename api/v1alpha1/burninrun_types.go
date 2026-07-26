package v1alpha1

import (
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

// RunPhase is the lifecycle of a BurnInRun.
// +kubebuilder:validation:Enum=Pending;Running;Passed;Failed;Error;Cancelled
type RunPhase string

const (
	RunPending   RunPhase = "Pending"
	RunRunning   RunPhase = "Running"
	RunPassed    RunPhase = "Passed"
	RunFailed    RunPhase = "Failed"
	RunError     RunPhase = "Error"
	RunCancelled RunPhase = "Cancelled"
)

// BurnInRunSpec is one execution of a BurnInProfile against a target.
type BurnInRunSpec struct {
	// ProfileRef names the BurnInProfile to run (same namespace).
	// +kubebuilder:validation:Required
	ProfileRef string `json:"profileRef"`

	// Target selects the node(s) under test.
	// +kubebuilder:validation:Required
	Target TargetSelector `json:"target"`

	// TTLSecondsAfterFinished garbage-collects the run (and its pods) this long
	// after it reaches a terminal phase. 0 keeps it.
	TTLSecondsAfterFinished *int32 `json:"ttlSecondsAfterFinished,omitempty"`
}

// TestResult is the outcome of one test within a run.
type TestResult struct {
	Name       string       `json:"name"`
	Kind       TestKind     `json:"kind"`
	Scope      TestScope    `json:"scope"`
	Phase      RunPhase     `json:"phase"`
	Nodes      []string     `json:"nodes,omitempty"`
	StartedAt  *metav1.Time `json:"startedAt,omitempty"`
	FinishedAt *metav1.Time `json:"finishedAt,omitempty"`
	// Metrics are the parsed numeric results the thresholds were evaluated
	// against (e.g. busBandwidthGBs: "23.4", eccErrors: "0").
	Metrics map[string]string `json:"metrics,omitempty"`
	// Message is a human-readable outcome (a failing threshold, the real error).
	Message string `json:"message,omitempty"`
}

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
