package v1alpha1

import (
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

// BurnInProfileSpec is an ordered acceptance suite: which tests, in what order,
// and the overall verdict policy. A profile is what an operator "runs" against
// hardware ("acceptance", "quick-smoke", "pair-network").
type BurnInProfileSpec struct {
	// Tests are executed in order. Each references a BurnInTest by name (same
	// namespace) or inlines a spec.
	// +kubebuilder:validation:MinItems=1
	Tests []ProfileTest `json:"tests"`

	// FailFast stops the run at the first failing test (default false — a full
	// acceptance sweep usually wants every result even after one failure).
	FailFast bool `json:"failFast,omitempty"`

	// Sinks names BurnInSinks (same namespace) that results are exported to.
	Sinks []string `json:"sinks,omitempty"`
}

// ProfileTest references a BurnInTest or inlines one.
type ProfileTest struct {
	// TestRef names an existing BurnInTest.
	TestRef string `json:"testRef,omitempty"`
	// Inline defines the test directly (mutually exclusive with TestRef).
	Inline *BurnInTestSpec `json:"inline,omitempty"`
	// Required marks whether this test's failure fails the whole profile
	// (default true). Informational tests can be marked false.
	// +kubebuilder:default=true
	Required *bool `json:"required,omitempty"`
}

type BurnInProfileStatus struct {
	ObservedGeneration int64 `json:"observedGeneration,omitempty"`
}

// +kubebuilder:object:root=true
// +kubebuilder:subresource:status
// +kubebuilder:resource:shortName=bip,categories=burnin
// +kubebuilder:printcolumn:name="Tests",type=integer,JSONPath=`.spec.tests[*]`

// BurnInProfile is an ordered acceptance suite of BurnInTests.
type BurnInProfile struct {
	metav1.TypeMeta   `json:",inline"`
	metav1.ObjectMeta `json:"metadata,omitempty"`
	Spec              BurnInProfileSpec   `json:"spec,omitempty"`
	Status            BurnInProfileStatus `json:"status,omitempty"`
}

// +kubebuilder:object:root=true

// BurnInProfileList contains a list of BurnInProfile.
type BurnInProfileList struct {
	metav1.TypeMeta `json:",inline"`
	metav1.ListMeta `json:"metadata,omitempty"`
	Items           []BurnInProfile `json:"items"`
}

// TargetSelector picks the node(s) a run targets.
type TargetSelector struct {
	// NodeNames explicitly names nodes. For a profile containing a Pair-scope
	// test exactly two are required, and they must be distinct; the operator
	// pairs them for the point-to-point tests, running the server on the first
	// and the client on the second. The requirement is ENFORCED at run start —
	// a Pair test against one node, or three, is a terminal configuration Error
	// naming the count it got, never a run that quietly measures something
	// else.
	NodeNames []string `json:"nodeNames,omitempty"`
	// NodeSelector selects nodes by label (e.g. glimmer.ai/interconnect=roce-200g).
	// The same exactly-two rule applies to what it resolves to.
	NodeSelector map[string]string `json:"nodeSelector,omitempty"`
	// Tolerations applied to every test pod so it can land on tainted GPU nodes.
	Tolerations []corev1.Toleration `json:"tolerations,omitempty"`
}
