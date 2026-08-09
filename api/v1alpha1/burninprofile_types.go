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

	// Variants expands this entry into one execution per cell of a matrix.
	//
	// compute-smoke runs an NVFP4 GEMM. That is one cell of a matrix the API
	// had no word for: the same kernel at FP8, BF16, TF32 and FP64 answers
	// different questions about the same silicon, and asking all five meant
	// authoring five BurnInTest objects differing by one environment variable,
	// each with its own name, its own thresholds, and its own copy of every
	// other field. The same shape recurs on every axis worth sweeping — message
	// size for a fabric test, buffer size for a bandwidth test, duration class
	// for a soak.
	//
	// Empty means one execution, exactly as before. A profile with no variants
	// plans identically to one written before variants existed.
	// +optional
	Variants []TestVariant `json:"variants,omitempty"`
}

// TestVariant is one cell of a test's matrix.
//
// # Why a list and not a cartesian product
//
// A `matrix.axes` block that multiplied out was considered and rejected. It
// needs exclusion lists for the cells that make no sense, a cell-naming scheme,
// and per-cell thresholds anyway — at which point it has collapsed back into
// this list with extra machinery in front of it. Variants are the primitive;
// cartesian expansion can be layered on later as client-side sugar that emits
// them.
//
// # Why the controller never reads Axes
//
// Axes are OPAQUE labels. The controller injects them and echoes them and does
// not interpret them, ever. That is what keeps the vendor seam where it belongs:
// a precision, a message size or a duration class means something to a runner
// and nothing to a reconciler, and a controller that learned to branch on
// `precision: fp4` would be the `if nvidia {}` this project refuses, wearing a
// different hat.
type TestVariant struct {
	// Name suffixes the parent test's name and is this variant's RESULT
	// IDENTITY: the execution is planned as "<test>-<name>".
	//
	// Two variants of one test with the same name are refused at plan time by
	// the same unique-name check that refuses two tests with the same name,
	// because the failure is identical — the second would never run and its
	// verdict would silently overwrite the first's.
	// +kubebuilder:validation:MinLength=1
	// +kubebuilder:validation:Pattern=`^[a-z0-9]([-a-z0-9]*[a-z0-9])?$`
	Name string `json:"name"`

	// Axes describe what varies, as opaque labels: {precision: fp4},
	// {class: soak}, {messageBytes: "8"}.
	//
	// Each becomes BURNIN_VARIANT_<AXIS>=<value> in the runner's environment,
	// upper-cased, and is echoed onto the TestResult and into the delivery
	// envelope so a report can group a sweep's cells without parsing them back
	// out of names.
	// +optional
	Axes map[string]string `json:"axes,omitempty"`

	// The overlays. Nil INHERITS the parent test's value; non-nil REPLACES it.
	//
	// Thresholds replace rather than merge, and that is the one that matters.
	// An FP4 throughput floor is not an edited FP8 floor. Merging by metric name
	// would silently retain a gate the author believed they had replaced — and a
	// node failed against a threshold nobody can find in the profile is the
	// worst kind of verdict this project can produce, because there is nothing
	// to read that explains it.
	//
	// A variant that means to apply NO thresholds sets an empty (non-nil) list.
	// +optional
	DurationSeconds *int32 `json:"durationSeconds,omitempty"`
	// +optional
	Args []string `json:"args,omitempty"`
	// +optional
	Env []corev1.EnvVar `json:"env,omitempty"`
	// +optional
	Thresholds []Threshold `json:"thresholds,omitempty"`
	// +optional
	// +kubebuilder:validation:Minimum=1
	RepeatCount *int32 `json:"repeatCount,omitempty"`
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
