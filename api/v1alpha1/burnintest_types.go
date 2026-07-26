package v1alpha1

import (
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

// TestScope is the topology a test exercises.
//
// Pair (and Group) scope is first-class in v1: acceptance of an interconnect —
// NCCL over RoCE/IB, ib_write_bw, GPUDirect — is only meaningful across at least
// two nodes, so the operator schedules and correlates multi-node tests, not just
// per-node ones.
// +kubebuilder:validation:Enum=Node;Pair;Group
type TestScope string

const (
	// ScopeNode runs on a single node (gpu-burn, thermal soak, DCGM diag).
	ScopeNode TestScope = "Node"
	// ScopePair runs across exactly two nodes (point-to-point fabric: ib_write_bw,
	// 2-rank NCCL all-reduce, GPUDirect RDMA).
	ScopePair TestScope = "Pair"
	// ScopeGroup runs across N>=2 nodes (collective NCCL at cluster scale).
	ScopeGroup TestScope = "Group"
)

// TestKind is a well-known category. The kind selects the default runner image
// and result parser; Runner may override the image. Unknown kinds are allowed
// (custom runners) but get no built-in parsing.
type TestKind string

const (
	KindGPUBurn      TestKind = "gpu-burn"       // Node: sustained FP compute + ECC watch
	KindComputeSmoke TestKind = "compute-smoke"  // Node: quick arch-correct kernel (sm_121 FP4 / FP16 proxy)
	KindDCGMDiag     TestKind = "dcgm-diag"      // Node: NVIDIA DCGM diagnostics
	KindThermalSoak  TestKind = "thermal-soak"   // Node: hold load, assert no throttle/trip
	KindNCCL         TestKind = "nccl"           // Pair/Group: all-reduce bus-bandwidth over the fabric
	KindIBWriteBW    TestKind = "ib-write-bw"    // Pair: perftest ib_write_bw across the RDMA link
	KindGPUDirect    TestKind = "gpudirect-rdma" // Pair: GPUDirect RDMA path validation
	KindCustom       TestKind = "custom"         // any image; no built-in parsing
)

// BurnInTestSpec defines one acceptance test.
type BurnInTestSpec struct {
	// Kind selects built-in runner defaults + result parsing.
	// +kubebuilder:validation:Required
	Kind TestKind `json:"kind"`

	// Scope is the topology the test needs.
	// +kubebuilder:default=Node
	Scope TestScope `json:"scope,omitempty"`

	// DurationSeconds bounds a single execution (0 = the kind's default).
	// +kubebuilder:validation:Minimum=0
	DurationSeconds int32 `json:"durationSeconds,omitempty"`

	// Runner overrides the image/command for this test. When empty the operator
	// uses the built-in image for Kind. This is the vendor/heterogeneity seam:
	// a new accelerator or NIC ships a runner image, not a controller change.
	Runner *RunnerSpec `json:"runner,omitempty"`

	// Thresholds are pass/fail gates evaluated against parsed metrics
	// (e.g. busBandwidthGBs>=20, eccErrors==0, throttleEvents==0). A test with no
	// thresholds passes on a clean (exit 0) run.
	Thresholds []Threshold `json:"thresholds,omitempty"`

	// Resources requested per test pod (e.g. nvidia.com/gpu, rdma/rdma_shared_device_a).
	Resources corev1.ResourceRequirements `json:"resources,omitempty"`

	// HostNetwork runs the test pod on the host network (needed for direct host
	// RDMA in some fabrics). Defaults false.
	HostNetwork bool `json:"hostNetwork,omitempty"`
}

// RunnerSpec is an explicit test container.
type RunnerSpec struct {
	// +kubebuilder:validation:Required
	Image   string          `json:"image"`
	Command []string        `json:"command,omitempty"`
	Args    []string        `json:"args,omitempty"`
	Env     []corev1.EnvVar `json:"env,omitempty"`
	// Privileged is sometimes required for RDMA/GPUDirect paths.
	Privileged bool `json:"privileged,omitempty"`
}

// Comparison is the operator applied by a Threshold.
// +kubebuilder:validation:Enum=GreaterThanOrEqual;LessThanOrEqual;Equal;NotEqual
type Comparison string

const (
	GTE Comparison = "GreaterThanOrEqual"
	LTE Comparison = "LessThanOrEqual"
	EQ  Comparison = "Equal"
	NEQ Comparison = "NotEqual"
)

// Threshold gates a named metric emitted by the runner's result parser.
type Threshold struct {
	// +kubebuilder:validation:Required
	Metric string `json:"metric"` // e.g. "busBandwidthGBs", "eccErrors", "latencyUs"
	// +kubebuilder:validation:Required
	Comparison Comparison `json:"comparison"`
	// +kubebuilder:validation:Required
	Value string `json:"value"` // numeric compared as float64
}

// BurnInTestStatus is intentionally minimal — BurnInTest is a reusable
// definition; execution state lives on BurnInRun.
type BurnInTestStatus struct {
	// ObservedGeneration is the last spec generation reconciled.
	ObservedGeneration int64 `json:"observedGeneration,omitempty"`
}

// +kubebuilder:object:root=true
// +kubebuilder:subresource:status
// +kubebuilder:resource:shortName=bit,categories=burnin
// +kubebuilder:printcolumn:name="Kind",type=string,JSONPath=`.spec.kind`
// +kubebuilder:printcolumn:name="Scope",type=string,JSONPath=`.spec.scope`

// BurnInTest is a single, reusable hardware acceptance test.
type BurnInTest struct {
	metav1.TypeMeta   `json:",inline"`
	metav1.ObjectMeta `json:"metadata,omitempty"`
	Spec              BurnInTestSpec   `json:"spec,omitempty"`
	Status            BurnInTestStatus `json:"status,omitempty"`
}

// +kubebuilder:object:root=true

// BurnInTestList contains a list of BurnInTest.
type BurnInTestList struct {
	metav1.TypeMeta `json:",inline"`
	metav1.ListMeta `json:"metadata,omitempty"`
	Items           []BurnInTest `json:"items"`
}
