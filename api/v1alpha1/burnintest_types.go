package v1alpha1

import (
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

// Capability is a Linux capability RunnerSpec.Capabilities may add.
//
// Enumerated rather than a free string (#302), matching VendorImage's own
// reasoning: a typo'd capability name is accepted by the apiserver either
// way, but an enum turns it into a validation error instead of a silently
// ineffective grant that looks like broken hardware at run time. Widen this
// the moment a runner legitimately needs another one.
//
// CapabilitySYSLOG does NOT make /dev/kmsg readable in-cluster by itself —
// see its own doc comment.
// +kubebuilder:validation:Enum=SYSLOG
type Capability string

const (
	// CapabilitySYSLOG is CAP_SYSLOG. #134 measured that reading /dev/kmsg
	// under the Ubuntu/Debian default kernel.dmesg_restrict=1 needs CAP_SYSLOG
	// AND a device-cgroup grant AND uid 0, all three at once.
	//
	// This field can only ever supply the CAP_SYSLOG third. #302 measured
	// (in-cluster, on real GB10 hardware) that the device-cgroup grant does
	// NOT follow from a HostPaths CharDevice mount the way it does under
	// pkg/localrun's `docker run --device`: internal/controller/pods.go maps a
	// HostPathMount to a plain corev1.HostPathVolumeSource, which the kubelet
	// mounts without adding a device-cgroup rule — runc's default allowlist
	// does not cover /dev/kmsg (major:minor 1:11). So `runAsUser: 0` +
	// `capabilities: [SYSLOG]` + a HostPaths CharDevice mount for /dev/kmsg
	// still reads xidSource=none in-cluster; only `privileged: true` grants
	// the device cgroup this path needs, and CapabilitySYSLOG buys nothing
	// over that until the operator has some other way to grant a single
	// device (a device plugin, or CDI, both open ended in #302).
	//
	// Use this field for a capability whose need does NOT route through a
	// missing device-cgroup grant.
	CapabilitySYSLOG Capability = "SYSLOG"
)

// TestScope is the topology a test exercises.
//
// Acceptance of an interconnect — NCCL over RoCE/IB, ib_write_bw, GPUDirect —
// is only meaningful across at least two nodes, so the operator schedules and
// correlates multi-node tests, not just per-node ones.
//
// Node, Pair and Group all execute. A scope this operator version does not
// recognise is recorded as a terminal Error rather than skipped: a required
// acceptance test the operator cannot run must never let hardware pass by
// omission.
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

	// ScopeGroup runs a collective across EVERY target node (NCCL at cluster
	// scale), one rank per node.
	//
	// A Group test is ONE execution over N nodes. Target i is rank i, all of
	// them rendezvous'd through a single headless Service, and the result is a
	// SINGLE TestResult naming every node — for the same reason a Pair produces
	// one: a collective is a property of the GROUP, and splitting it per node
	// would produce N claims no single node can support. On a collective that is
	// even more misleading than on a link, because every healthy rank blocks
	// waiting for the faulty one and would report the same timeout.
	//
	// Rank 0 is the root: it starts first, publishes whatever bootstrap handle
	// the collective needs, and the other ranks are not created until it reports
	// Ready. The runner is told BURNIN_RANK, BURNIN_NRANKS, BURNIN_ROOT_HOST and
	// BURNIN_ROOT_NODE, and nothing else — no rank list and no topology.
	//
	// At least two distinct target nodes are required, and spec.maxConcurrentNodes
	// must be at least the number of them: a group holds every one of its nodes
	// for the whole test, so it costs one slot per rank and at any smaller cap it
	// could never start. All three are refused at run start rather than mid-flight.
	//
	// It is executed WITHOUT JobSet and without OpenMPI. Every rank is pinned by
	// hostname to a node this operator has already admitted and cordoned, so
	// there is no placement decision left for a gang scheduler to make; see the
	// package comment in internal/controller/group.go for the full reasoning.
	ScopeGroup TestScope = "Group"
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
	//
	// At Group scope it also buys margin for the RENDEZVOUS. Every rank's pod
	// deadline is this plus a fixed grace, and a rank spends part of that window
	// waiting for the rest of the cohort to be scheduled and to pull the runner
	// image — so on a cold image cache a large group can lose real test time to
	// the wait. Raising this, or pre-pulling the runner image onto the target
	// nodes, are the two things that help; see issue #122 and the note in
	// internal/controller/group.go.
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

	// Soak divides this test's DurationSeconds into a sequence of shorter pod
	// executions. Unset means one pod for the whole duration, which is what
	// every test did before this field existed.
	//
	// +optional
	Soak *SoakSpec `json:"soak,omitempty"`

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

// SoakSpec turns ONE long execution into a SEQUENCE of shorter ones.
//
// A seven-day soak used to be a single 604,920-second pod, and everything that
// normally happens to a pod over a week is fatal to one: a kubelet restart, a
// drain, an eviction, an image-GC pass, a reboot. Each of those ended the run as
// an Error, and with retryOnErrorLimit set the retry started the week over from
// zero. Segmenting costs ONE SEGMENT instead.
//
// It is OPT-IN PER TEST and that is not negotiable, because whether a duration
// means anything at all is the RUNNER's property and only the test's author
// knows it. host-health clamps its window to 30 seconds and compute-smoke is
// declared burst-only (see TestKind.BurstOnly); segmenting either produces a
// sequence of identical short measurements and calls it a week. A default-on
// version of this field would do exactly that to every profile in the fleet.
//
// # What changes about the verdict, and it is the whole feature
//
// THE VERDICT IS RENDERED ONCE, OVER THE AGGREGATE. Per segment only the exit
// code decides: 0 folds the segment's metrics into TestResult.AggregatedMetrics
// and the test stays open, 1 settles the test Failed immediately, 2 settles it
// Skipped, and anything else is an Error that consumes retryOnErrorLimit,
// re-runs THE SAME segment index and contributes nothing. Thresholds are
// evaluated once, on the last segment, against the accumulated aggregate —
// never per segment, because a soak's gate is a statement about the soak.
//
// How each metric combines is declared beside its NAME in
// pkg/contract.Metric.Aggregation, never here and never in the reconciler: Sum
// for a per-window counter, Min for a floor, Max for a ceiling, Last for a
// nameplate or a lifetime total the runner already reports absolutely. An
// unregistered name combines Last, because inventing Sum semantics for somebody
// else's measurement would be this operator deciding what it means.
//
// elapsedS sums, which is what makes `elapsedS >= 0.95 * durationSeconds` an
// honest gate on whether the soak actually soaked.
type SoakSpec struct {
	// SegmentSeconds is one pod's window. Each segment gets
	// BURNIN_DURATION_SECONDS=<segmentSeconds> and its own
	// activeDeadlineSeconds, so an eviction or a reboot costs one segment
	// rather than the whole run.
	//
	// The floor is five minutes and it is a real bound rather than a round
	// number: a segment boundary costs a pod teardown, a pod creation and a
	// scheduling round-trip, and below five minutes the run spends more time
	// changing pods than it spends measuring hardware.
	//
	// EVERY SEGMENT IS A FULL WINDOW, including the last, so a soak runs at
	// least its DurationSeconds and may overrun by up to one segment. A short
	// remainder segment would be the worse choice twice over: it is below the
	// floor this field exists to enforce, and a counter summed over a
	// ten-second window is not comparable to the same counter summed over a
	// fifteen-minute one, which is precisely what the aggregation rules assume.
	//
	// It must not exceed DurationSeconds — a segment longer than the soak is a
	// soak that burns longer than it was asked to — and the run is refused at
	// start if it does.
	//
	// +kubebuilder:validation:Required
	// +kubebuilder:validation:Minimum=300
	SegmentSeconds int32 `json:"segmentSeconds"`

	// AbortEarly stops the soak as Failed the moment the aggregate so far
	// PROVABLY violates a gate. Default false.
	//
	// "Provably" is the whole of it, and it is narrower than it sounds. It fires
	// only where the aggregation is monotone IN THE GATED DIRECTION, so that no
	// later segment could retract the violation: a Sum counter under
	// LessThanOrEqual (or under Equal, once the sum has passed the value), a Min
	// under a GreaterThanOrEqual floor, a Max under a LessThanOrEqual ceiling.
	// Everywhere else it stays silent — a Min under a ceiling can be pulled back
	// under it by a later segment, and aborting a week of burn-in on a violation
	// the next fifteen minutes would have retracted is the wrong trade.
	//
	// It is evaluated at SEGMENT BOUNDARIES ONLY, from harvested metrics, and
	// never from a checkpoint's parse of a bounded log tail. A checkpoint is
	// evidence and never a verdict; ending a soak on a truncated read would make
	// it one.
	//
	// A metric that has not been reported by any segment yet never aborts
	// anything. Fail-closed still applies at the end — a gated metric no segment
	// ever emitted fails the test — but absence mid-soak is not yet an absence.
	//
	// +kubebuilder:default=false
	// +optional
	AbortEarly *bool `json:"abortEarly,omitempty"`
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
	Image string `json:"image,omitempty"`

	// ImagesByVendor selects the image by the ACCELERATOR VENDOR of the node the
	// test lands on, so one profile can serve a mixed fleet.
	//
	//	runner:
	//	  imagesByVendor:
	//	    - vendor: nvidia
	//	      image: ghcr.io/.../glimmer-burnin-memory-bw:v1
	//	    - vendor: amd
	//	      image: ghcr.io/.../glimmer-burnin-memory-bw-rocm:v1
	//
	// This is the project's vendor-neutrality rule expressed in the API rather
	// than worked around. Heterogeneity is DATA — fingerprints and runner images
	// — and never a controller branch, and the reconciler still holds that line:
	// resolution is a lookup, not an `if amd {}`. What changes is that the data
	// finally has somewhere to go. Before this, `memory-bw` meant nvbandwidth
	// full stop, and a mixed fleet needed either N profiles that drift apart or
	// an explicit image that gives up on one profile serving everything.
	//
	// Resolution order is explicit > byVendor > the kind's built-in default. A
	// node whose vendor has neither is a PLAN-TIME error naming the vendor and
	// this field — never a skip, because a node silently not being tested is how
	// a fleet gets certified without being measured.
	//
	// A LIST OF PAIRS, not a map keyed by vendor, and the reason is worth
	// recording: Kubernetes cannot validate map KEYS. A structural schema
	// describes `additionalProperties` — the values — and nothing else, so a
	// typo'd `nvidai:` would be accepted by the apiserver and resolve to no
	// image on every NVIDIA node in the fleet. The only way to check keys is a
	// CEL rule, and a CEL rule over a map whose key length the schema cannot
	// bound is estimated so expensively that the apiserver REFUSES TO INSTALL
	// THE ENTIRE CRD — measured, not guessed: 15.3x over budget even with
	// maxProperties set. As a list, `vendor` is an ordinary enumerated string
	// the apiserver validates for free, and listMapKey rejects duplicates.
	//
	// +optional
	// +listType=map
	// +listMapKey=vendor
	// +kubebuilder:validation:MaxItems=8
	ImagesByVendor []VendorImage `json:"imagesByVendor,omitempty"`

	Command []string        `json:"command,omitempty"`
	Args    []string        `json:"args,omitempty"`
	Env     []corev1.EnvVar `json:"env,omitempty"`
	// Privileged is sometimes required for RDMA/GPUDirect paths.
	//
	// It is NOT sufficient to reach host device nodes — see HostPaths, which is
	// the field that actually gets them into the container.
	Privileged bool `json:"privileged,omitempty"`

	// RunAsUser overrides the uid the runner container runs as. Unset uses the
	// image's own USER, which for every runner this project publishes is the
	// non-root 65532.
	//
	// IT IS A PRIVILEGE GRANT AND IT IS SEPARATE FROM Privileged ON PURPOSE,
	// because the two are not the same thing and one does not imply the other.
	// Linux drops a process's effective capabilities when it switches away from
	// root, so a container that is `privileged: true` and runs as uid 65532 does
	// NOT hold CAP_SYSLOG — which is exactly what reading /dev/kmsg needs on any
	// host with kernel.dmesg_restrict=1. That combination looks sufficient, is
	// not, and produces a probe that silently reports nothing (issue #134).
	//
	// Set it to 0 only for a test that genuinely needs a root-owned host
	// resource, and prefer the alternative where one exists: a kernel log the
	// image's own uid can read, named through the runner's environment, needs no
	// privilege at all. Every runner here degrades honestly without it — a source
	// it could not read is reported as unread, never as clean.
	// +optional
	RunAsUser *int64 `json:"runAsUser,omitempty"`

	// Capabilities are Linux capabilities ADDED to the runner container,
	// narrower than Privileged for a capability need that does not also
	// require a device-cgroup grant.
	//
	// IT DOES NOT, BY ITSELF, MAKE /dev/kmsg READABLE IN-CLUSTER, even
	// combined with a HostPaths CharDevice mount and RunAsUser: 0 — see
	// CapabilitySYSLOG's own doc comment and #302. This field exists because
	// #134's truth table names CAP_SYSLOG as one of three grants /dev/kmsg
	// needs at once, and until this field existed there was no way to request
	// it without asking for Privileged: true's much larger blast radius too —
	// but #302 then measured that the operator's own device-cgroup grant is
	// missing regardless, so requesting CAP_SYSLOG here still leaves
	// /dev/kmsg at EPERM. host-health's kmsg recipe still needs
	// Privileged: true in-cluster today; this field is for whatever capability
	// need arrives next that is not gated on a missing device-cgroup rule.
	//
	// Add-only, deliberately: a Drop list is a hardening feature and a
	// different conversation from the narrower-grant one this field answers.
	//
	// Enumerated rather than a free string, matching ImagesByVendor's own
	// reasoning (#274/#257): a typo'd capability name is accepted by the
	// apiserver either way, but an enum makes it a validation error instead of
	// a silently ineffective grant that looks like broken hardware. Widen the
	// enum the moment a runner legitimately needs another one.
	//
	// ONLY MEANINGFUL WITH RunAsUser: 0. A capability added to the bounding
	// set does nothing for a non-root uid without ambient capabilities — the
	// same measurement that showed Privileged alone fails for this image's
	// default uid 65532. Setting Capabilities without RunAsUser: 0 is refused
	// at plan time (buildPlan) rather than producing a probe that silently
	// reads nothing and reads as broken hardware.
	// +optional
	// +listType=set
	// +kubebuilder:validation:MaxItems=4
	Capabilities []Capability `json:"capabilities,omitempty"`

	// PriorityClassName points this test's pods at a PriorityClass the SITE
	// manages.
	//
	// Passthrough only. This operator never creates a PriorityClass: a
	// cluster-scoped object minted by a test-runner is a footgun, and it is a
	// permission the operator should not need — its RBAC does not grow for this
	// field, which is the point.
	//
	// It is worth setting on contended hardware, where a multi-day soak is
	// otherwise one preemption away from an Error. Every runner pod already
	// carries the autoscaler's safe-to-evict: false annotation, which handles
	// consolidation; this handles contention, which is a different mechanism
	// and needs a different answer.
	// +optional
	PriorityClassName string `json:"priorityClassName,omitempty"`

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

	// Prepare seeds a scratch volume from an IMAGE before the runner starts —
	// an initContainer, in effect, for the one case a burn-in runner
	// genuinely needs one: a vendor tool the runner image may not
	// redistribute (dcgm-diag's DCGM; AMD's RVS would be the same case) but
	// that the SITE is entitled to pull, most often because the identical
	// image is already running as a DaemonSet on every GPU node in the
	// cluster (#375).
	//
	// Before this field, the only route to that binary was HostPaths — which
	// means preparing every node in the fleet by hand before the operator can
	// measure it, exactly the gap dcgm-diag's own README documents as its
	// unsatisfying option. An initContainer using the site's own DCGM image
	// needs no host mount and no per-node preparation at all: the vendor
	// binary the runner needs is copied out of a container the site is
	// already entitled to run, immediately before the runner that needs it
	// starts.
	//
	//	runner:
	//	  prepare:
	//	    - image: nvcr.io/nvidia/cloud-native/dcgm:4.5.2-1-ubuntu22.04
	//	      copyFrom: /usr
	//	      mountPath: /usr/local/dcgm
	//
	// This is NOT a general initContainer field, on purpose, for the same
	// reason HostPaths is a curated shape rather than []corev1.Volume +
	// []corev1.VolumeMount: every case this field exists for is "copy a path
	// out of an image", and a curated field states that intent where a
	// general initContainer list would let a BurnInTest run arbitrary
	// pre-runner logic — a much larger surface than seeding a volume.
	//
	// Each step gets its own emptyDir, populated by copying CopyFrom out of
	// Image, then mounted READ-ONLY into the runner container at MountPath —
	// the runner only ever needs to READ a vendor tool it did not build, and
	// a scratch volume it could write to is a privilege the runner does not
	// need and this field does not grant. The copy happens once, in the
	// initContainer, before the runner container is created at all; the
	// runner never sees Image or CopyFrom, only the result at MountPath.
	//
	// +optional
	// +listType=map
	// +listMapKey=mountPath
	// +kubebuilder:validation:MaxItems=4
	Prepare []PrepareStep `json:"prepare,omitempty"`

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

// VendorImage pins one accelerator vendor's runner image.
type VendorImage struct {
	// Vendor is the accelerator vendor, matching NodeFingerprint's vocabulary.
	//
	// Enumerated by the apiserver, which is the whole reason this is a list
	// rather than a map: a typo here is rejected at apply time rather than
	// resolving to no image on every node of that vendor at 3am.
	//
	// +kubebuilder:validation:Enum=nvidia;amd;intel;tenstorrent;habana
	Vendor string `json:"vendor"`

	// Image is the runner image for that vendor.
	//
	// +kubebuilder:validation:MinLength=1
	Image string `json:"image"`
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

// PrepareStep copies ONE path out of an image into the runner container,
// before the runner starts. See RunnerSpec.Prepare for what this is for and
// why it is not a general initContainer.
type PrepareStep struct {
	// Image is pulled and run as an initContainer purely to be copied from —
	// its own entrypoint is never invoked; the step overrides it with a copy
	// command instead.
	// +kubebuilder:validation:Required
	// +kubebuilder:validation:MinLength=1
	Image string `json:"image"`

	// CopyFrom is the path INSIDE Image to copy — recursively, contents and
	// all — into MountPath.
	// +kubebuilder:validation:Required
	// +kubebuilder:validation:MinLength=1
	// +kubebuilder:validation:Pattern=`^/`
	CopyFrom string `json:"copyFrom"`

	// MountPath is where the copied contents appear INSIDE THE RUNNER
	// container, read-only. This list's identity: two entries may not share
	// a MountPath, the same rule and the same reason as HostPathMount's.
	// +kubebuilder:validation:Required
	// +kubebuilder:validation:MinLength=1
	// +kubebuilder:validation:Pattern=`^/`
	MountPath string `json:"mountPath"`
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
