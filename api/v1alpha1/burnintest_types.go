package v1alpha1

import (
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

// TestScope is the topology a test exercises.
//
// Acceptance of an interconnect — NCCL over RoCE/IB, ib_write_bw, GPUDirect —
// is only meaningful across at least two nodes, so the operator schedules and
// correlates multi-node tests, not just per-node ones.
//
// Node and Pair execute. Group does not yet, and a Group test is recorded as a
// terminal Error rather than skipped: a required acceptance test the operator
// cannot run must never let hardware pass by omission.
// +kubebuilder:validation:Enum=Node;Pair;Group
type TestScope string

const (
	// ScopeNode runs on a single node (gpu-burn, thermal soak, DCGM diag).
	ScopeNode TestScope = "Node"

	// ScopePair runs across exactly two nodes (point-to-point fabric:
	// ib_write_bw, 2-rank NCCL all-reduce, GPUDirect RDMA).
	//
	// A Pair test is ONE execution over TWO nodes, not two executions. The
	// operator runs a server pod on the first target and a client pod on the
	// second, rendezvous'd through a headless Service, and records a SINGLE
	// TestResult naming both nodes. That is not a bookkeeping convenience: a
	// point-to-point measurement is a property of the LINK, and a verdict
	// attributed to one endpoint would send an engineer to replace the wrong
	// part — or, worse, would let one node's result be read as an acceptance of
	// hardware that was never independently measured.
	//
	// Exactly two target nodes are required, and they must be distinct.
	ScopePair TestScope = "Pair"

	// ScopeGroup runs across N>=2 nodes (collective NCCL at cluster scale).
	// Not executed by this operator version.
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

	// KindMemoryBW is a Node-scope memory-bandwidth measurement: device-local
	// STREAM-style bandwidth plus the host<->device and device<->device copy
	// paths (memoryBandwidthGBs, hostToDeviceBandwidthGBs,
	// deviceToHostBandwidthGBs, deviceToDeviceBandwidthGBs).
	//
	// It exists as its own kind because a bandwidth shortfall and a compute
	// shortfall have different root causes and different remediations: a part
	// that computes at full rate but reads memory at half rate is usually a
	// link, a seating, or a downgraded PCIe/NVLink negotiation, not a bad die.
	// Folding it into gpu-burn would report one number for two faults.
	KindMemoryBW TestKind = "memory-bw"

	// KindHostHealth is a Node-scope read of host- and driver-side fault
	// counters over the test window (xidEvents, pcieReplayErrors,
	// nicLinkDownEvents, remappedRows, eccErrors).
	//
	// It is passive: it applies no load and measures no throughput, so it is
	// cheap enough to run alongside every profile. Its value is that these
	// counters move while every performance test still passes — a link
	// retraining or a row being remapped is a part on its way out, and this is
	// the kind that turns that into an acceptance signal rather than something
	// discovered after the node is in production.
	KindHostHealth TestKind = "host-health"

	// KindClockProbe is a Node-scope sustained-clock-under-load gate: it holds
	// a known load and asserts the accelerator actually sustains its clocks
	// (sustainedClockPct, smClockMHz, memClockMHz against ratedBoostClockMHz).
	//
	// This is the most important kind added in this revision. It catches a GPU
	// pinned in a low P-state that reports perfectly healthy utilization — the
	// node looks busy, every liveness and health check passes, and it simply
	// runs at a fraction of its rated speed. On GB10 this is the USB-C
	// Power-Delivery failure mode: an under-spec or degraded PD supply, or a
	// cable negotiating a lower contract, silently caps the power budget and
	// the part clocks down. Nothing else in the suite sees it — compute-smoke
	// passes, thermal-soak passes (a slow part runs cool), dcgm-diag passes —
	// because every one of them asserts correctness or health, and none of
	// them asserts speed. A fleet with this fault delivers correct results,
	// slowly, forever, and the loss shows up only in the training-job
	// wall-clock.
	//
	// Two rules follow from the failure mode and belong in the runner, not
	// here: the probe must warm up before sampling (an unwarmed part is
	// legitimately at idle clocks), and it must skip (exit 2) rather than fail
	// where it cannot read a rated clock, since a missing nameplate is an
	// unjudged part, not a slow one.
	KindClockProbe TestKind = "clockprobe"

	// KindMemoryStress is a Node-scope HOST memory stress test: it exercises
	// system RAM and reports hardware incidents and miscompares
	// (memoryErrors, miscompares, iterationsCompleted).
	//
	// Host DIMM faults present as accelerator flakiness — a corrupted staging
	// buffer becomes a wrong answer that looks like a bad GPU — so accepting a
	// node means accepting its host memory too.
	//
	// The sanctioned tool is stressapptest (Apache-2.0). stress-ng and fio are
	// GPL and cannot ship in a runner image; see the licensing rules in
	// CLAUDE.md before substituting a tool here.
	KindMemoryStress TestKind = "memory-stress"

	KindCustom TestKind = "custom" // any image; no built-in parsing
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

	// RepeatCount is how many times this test executes sequentially on each
	// node. EVERY execution must pass for the test to pass — the repeats are
	// an AND, not a best-of.
	//
	// This is the intermittent-fault gate. The faults that matter most in
	// burn-in are the ones that do not reproduce on the first try: a marginal
	// solder joint, a link that retrains under thermal cycling, a DIMM that
	// miscompares once an hour. A single clean pass does not distinguish a
	// good part from a part that fails one run in five, and running the test
	// N times is what makes that distinction. Repeats are sequential, never
	// parallel: the point is to cycle the hardware, and two concurrent copies
	// on one node would contend for the very resource under test.
	//
	// Do not confuse this with RetryOnErrorLimit on the run. A repeat re-runs
	// a test that already produced a verdict and makes the verdict stricter; a
	// retry re-runs a test that produced no verdict at all. A repeat can turn
	// a Passed into a Failed, and must never turn a Failed into a Passed.
	//
	// +kubebuilder:default=1
	// +kubebuilder:validation:Minimum=1
	RepeatCount *int32 `json:"repeatCount,omitempty"`

	// CheckpointIntervalSeconds is how often a long execution publishes its
	// metrics so far onto the TestResult. 0 or nil disables checkpointing.
	//
	// A multi-hour thermal soak that is cancelled, evicted, or lost to a node
	// reboot at minute 200 otherwise reports nothing at all: the runner's
	// metrics only reach the status when the pod terminates. Checkpointing
	// keeps the evidence gathered so far, so an interrupted soak still tells
	// you what the temperatures and clocks were doing.
	//
	// A checkpoint is EVIDENCE, never a verdict. Thresholds are evaluated once,
	// against the final metrics of a completed execution — a mid-run sample
	// that dips below a threshold is not a failure, because the run is not
	// over. Checkpoints overwrite in place rather than accumulating, so a long
	// soak cannot grow its status object without bound.
	//
	// +kubebuilder:validation:Minimum=0
	CheckpointIntervalSeconds *int32 `json:"checkpointIntervalSeconds,omitempty"`

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
	// Image overrides the built-in image for the test's Kind. Empty means "use
	// the default for this Kind", and a Kind with no default refuses at plan
	// time asking for this field rather than pull-failing per node.
	//
	// It is optional because RunnerSpec is no longer only an image override: a
	// test that needs nothing but a readinessProbe or a hostPath would otherwise
	// be forced to pin an image it did not want to change, and pinning one to
	// get a mount is how a fleet ends up on a stale runner tag by accident.
	// +optional
	Image   string          `json:"image,omitempty"`
	Command []string        `json:"command,omitempty"`
	Args    []string        `json:"args,omitempty"`
	Env     []corev1.EnvVar `json:"env,omitempty"`
	// Privileged is sometimes required for RDMA/GPUDirect paths.
	//
	// It is NOT sufficient to reach host device nodes — see HostPaths, which is
	// the field that actually gets them into the container.
	Privileged bool `json:"privileged,omitempty"`

	// HostPaths are paths on the NODE made visible inside the runner container.
	//
	// This is the most security-relevant field in this API and it is deliberately
	// the most explicit one. Every entry hands the runner a piece of the host's
	// own filesystem; it is not made safe by the pod being short-lived, and a
	// writable mount of a path that only needed reading is a real escalation.
	// Nothing is mounted unless it is named here — there is no implicit host
	// access anywhere in this operator.
	//
	// A burn-in runner legitimately needs it, which is why the field exists
	// rather than being refused. Hardware acceptance measures the host, and the
	// two facts below were measured on a real cluster, not inferred:
	//
	//   - RDMA fabric tests are IMPOSSIBLE without it. `privileged: true` plus
	//     `hostNetwork: true` does NOT give a pod the host's user-verbs device
	//     nodes: /dev/infiniband inside such a pod holds only rdma_cm and umad*,
	//     while the host also has uverbs*. uverbs is what ibv_create_cq opens, so
	//     ib_write_bw dies with "Couldn't create CQ / Failed to create CQs" and
	//     the nccl runner fails the same way. Mounting the host's
	//     /dev/infiniband is what makes uverbs* appear in the pod.
	//   - Xid scanning needs /dev/kmsg, which is not in a container's default
	//     /dev. Without it host-health reports xid_source=none and OMITS
	//     xidEvents, so a gate on that metric fails closed and certifies nothing.
	//
	// The shape is deliberately narrow rather than a general
	// []corev1.Volume + []corev1.VolumeMount. Every mount a burn-in runner has
	// ever needed is a host device or a host log path, and a curated field states
	// that intent where a general volume list would bury it: a reviewer can read
	// what a BurnInTest takes from the node, and from which direction, without
	// decoding a volume source. It also lets this API refuse what the general
	// form cannot — Type here admits only the assert-only hostPath types, so a
	// BurnInTest can never ask the kubelet to CREATE a path on a node.
	//
	// +optional
	// +listType=map
	// +listMapKey=mountPath
	HostPaths []HostPathMount `json:"hostPaths,omitempty"`

	// ReadinessProbe is what a Pair-scope SERVER runner uses to say it is
	// actually able to accept a connection.
	//
	// It exists because "the container is running" and "the socket is
	// listening" are different claims, and the gap between them is exactly
	// where a fabric test produces its most misleading result: an ib_write_bw
	// or nccl client that connects a moment too early fails with a connection
	// error, and a connection error on a fabric test reads as a bad link. The
	// operator will not start the client pod until the server pod is Ready, so
	// a server that declares a probe here converts that gate from "the kubelet
	// started the process" into "the process answered on its port".
	//
	// Without a probe a pod is Ready as soon as its containers start, which is
	// the weaker guarantee. A server runner whose listener takes any
	// appreciable time to bind should ship one — a tcpSocket probe on the
	// runner's own port is usually enough.
	//
	// It is honoured at every scope; it is only load-bearing at Pair.
	// +optional
	ReadinessProbe *corev1.Probe `json:"readinessProbe,omitempty"`
}

// HostPathMount makes ONE path on the node visible inside the runner container.
//
// It is honoured identically at Node scope and at Pair scope, on BOTH pods of a
// pair: a fabric test that could reach the verbs devices from only one end of
// the link would measure nothing.
//
// See RunnerSpec.HostPaths for what this costs and why a burn-in runner needs
// it anyway. The short version: this is host access, it is granted only where
// named, and it is granted read-only unless the entry says otherwise.
type HostPathMount struct {
	// Path is the absolute path ON THE HOST, e.g. "/dev/infiniband".
	// +kubebuilder:validation:Required
	// +kubebuilder:validation:MinLength=1
	// +kubebuilder:validation:Pattern=`^/`
	Path string `json:"path"`

	// MountPath is where that path appears INSIDE the runner container.
	//
	// It is required rather than defaulted to Path, because a mount whose two
	// ends are written out is a mount a reviewer can check. It is also this
	// list's identity: two entries may not share a MountPath.
	// +kubebuilder:validation:Required
	// +kubebuilder:validation:MinLength=1
	// +kubebuilder:validation:Pattern=`^/`
	MountPath string `json:"mountPath"`

	// ReadOnly mounts the path without write access. It DEFAULTS TO TRUE, which
	// is the direction a security-relevant default has to fall: an author who
	// needs the runner to write to the host has to say so, and an author who
	// forgets gets the harmless mount rather than the dangerous one.
	//
	// It is a pointer precisely so that "unset" is distinguishable from "false".
	// A plain bool would make a Go-constructed object — or one written before
	// this field existed — silently take the writable form.
	//
	// Which way round the real cases fall:
	//
	//   - /dev/kmsg is READ-ONLY. host-health only ever reads the ring buffer,
	//     and a burn-in runner has no business writing to the kernel log.
	//   - /dev/infiniband must be WRITABLE (readOnly: false). The user-verbs
	//     device nodes are opened read-write by ibv_open_device, so a read-only
	//     mount fails in the same place, and with much the same message, as no
	//     mount at all.
	//
	// +kubebuilder:default=true
	// +optional
	ReadOnly *bool `json:"readOnly,omitempty"`

	// Type asserts what the path must already BE on the node, and the kubelet
	// refuses the pod if it is anything else.
	//
	// The permitted values are the assert-only ones. corev1's DirectoryOrCreate
	// and FileOrCreate are deliberately NOT accepted: a burn-in run touches every
	// node in a fleet, and an operator that creates paths on hosts to satisfy its
	// own mounts would mutate the fleet it was sent to measure. Leaving Type
	// unset keeps Kubernetes' default of no check at all, which also lets the
	// container runtime create a missing path — so set it whenever you know what
	// the path is.
	//
	// Setting it converts "the runner saw an empty directory and reported
	// nothing" into a pod that refuses to start, which the run records as an
	// infrastructure Error. That is the better failure: an Error says the
	// hardware was never judged, where an empty mount looks like a measurement.
	//
	// +kubebuilder:validation:Enum=Directory;File;Socket;CharDevice;BlockDevice
	// +optional
	Type *corev1.HostPathType `json:"type,omitempty"`
}

// Comparison is the operator applied by a Threshold. Metric and value are both
// parsed as float64 and compared with NO TOLERANCE — there is no epsilon
// anywhere in this API, and that is what makes each operator right for exactly
// one kind of metric.
//
// GreaterThanOrEqual and LessThanOrEqual are how a MEASUREMENT is gated: a
// floor, a ceiling, or the two together for a band. Anything with a unit —
// bandwidth, temperature, latency, clocks, a percentage — belongs here.
//
// Equal and NotEqual are EXACT, and they exist for DIMENSIONLESS COUNTERS:
// eccErrors Equal 0, throttleEvents Equal 0, miscompares Equal 0. On a counter
// exactness is the whole point, because a tolerance around zero ECC errors is a
// tolerance for ECC errors.
//
// DO NOT use Equal or NotEqual on a continuous measurement.
// `sustainedClockPct Equal 83.22` asks an averaged sample to reproduce a decimal
// string exactly; it will not, so the gate fails on every healthy node forever
// and each failure is reported in the same shape as a hardware verdict. A metric
// whose name ends in a unit suffix (Gbps, GBs, MBs, Us, Ms, S, C, W, Pct, MHz,
// Tflops) is continuous by construction — gate it with GreaterThanOrEqual
// and/or LessThanOrEqual instead. pkg/verdict.ValidateThresholds reports this
// misuse at authoring time, before a run has condemned anything.
// +kubebuilder:validation:Enum=GreaterThanOrEqual;LessThanOrEqual;Equal;NotEqual
type Comparison string

const (
	// GTE and LTE gate a measurement; use both to express a band.
	GTE Comparison = "GreaterThanOrEqual"
	LTE Comparison = "LessThanOrEqual"
	// EQ and NEQ are exact comparisons, for dimensionless counters only.
	EQ  Comparison = "Equal"
	NEQ Comparison = "NotEqual"
)

// Applicability says what a threshold means on hardware that cannot produce the
// metric at all. It is NOT a way to make a gate optional — see the two values.
// +kubebuilder:validation:Enum=Required;RequiredIfMeasurable
type Applicability string

const (
	// Required is the default and keeps the fail-closed behaviour this project
	// is built on: the metric must be reported, and it must satisfy the
	// comparison. A metric the runner did not emit is a FAILURE, and so is a
	// metric the runner declared unmeasurable — an unmeasured quantity has not
	// been shown to be within limits, and acceptance must not assume it is.
	Required Applicability = "Required"

	// RequiredIfMeasurable keeps every bit of the Required behaviour EXCEPT the
	// one case where the runner has EXPLICITLY declared, for this execution on
	// this hardware, that the metric cannot be measured (the `metric=n/a`
	// sentinel in the runner stdout contract — see pkg/runner). In that one
	// case the threshold is not evaluated, and the run reports it as
	// not-evaluated rather than as satisfied.
	//
	// Absence is NOT a declaration. A runner that simply omits the metric — it
	// crashed before emitting, its probe timed out, someone renamed a key —
	// still fails the threshold exactly as under Required. Only the runner,
	// having looked at the hardware, can relax this gate; a profile author
	// choosing this value cannot.
	//
	// The case it exists for: NVIDIA GB10 / DGX Spark exposes no ECC to NVML at
	// all (its unified LPDDR5X has on-die ECC only), so eccErrors and
	// remappedRows are unmeasurable there — while on an ECC-capable part the
	// same gate must still fail a GPU whose ECC was switched off or whose
	// counters have moved.
	RequiredIfMeasurable Applicability = "RequiredIfMeasurable"
)

// Threshold gates a named metric emitted by the runner's result parser.
//
// A threshold naming a metric the runner did not emit is a FAILURE, never a
// pass: a missing measurement must not silently satisfy acceptance.
//
// Choosing the comparison matters as much as choosing the number — see
// Comparison. In short: dimensional metrics (a name ending in a unit suffix)
// take GreaterThanOrEqual/LessThanOrEqual; dimensionless counters take Equal or
// NotEqual, which are exact.
type Threshold struct {
	// Metric is the canonical metric name to gate, as the runner's result
	// parser emits it — e.g. "busBandwidthGBs", "eccErrors", "latencyUs".
	// Names are lowerCamelCase and a dimensional metric ends in a unit suffix;
	// a name that breaks that grammar is dropped during parsing, so a threshold
	// naming one can never be satisfied.
	// +kubebuilder:validation:Required
	Metric string `json:"metric"`

	// +kubebuilder:validation:Required
	Comparison Comparison `json:"comparison"` // no doc comment on purpose: controller-gen prefers a field's over its type's, and the type doc is what belongs in the CRD

	// Value is the number the metric is compared against, parsed as float64.
	// It must be finite: "NaN" and "Inf" parse as numbers but express no bound,
	// and NaN in particular compares false against everything, which would turn
	// NotEqual into a gate that always passes.
	//
	// Under Equal/NotEqual the comparison is exact, so this should be a whole
	// number counted by a counter. Under GreaterThanOrEqual/LessThanOrEqual it
	// is a bound and any finite value is meaningful.
	// +kubebuilder:validation:Required
	Value string `json:"value"`

	// Applicability decides what happens when the hardware cannot produce this
	// metric. Defaults to Required, which is today's fail-closed behaviour.
	// +kubebuilder:default=Required
	// +optional
	Applicability Applicability `json:"applicability,omitempty"`
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
