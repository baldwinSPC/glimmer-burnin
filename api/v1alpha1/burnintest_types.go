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

// TestKind is a well-known category. The kind selects the default runner image
// and result parser; Runner may override the image. Unknown kinds are allowed
// (custom runners) but get no built-in parsing.
type TestKind string

const (
	KindGPUBurn TestKind = "gpu-burn" // Node: sustained FP compute + ECC watch

	// KindComputeSmoke is a Node-scope BURST: one arch-correct block-scaled
	// NVFP4 GEMM, checked against a host reference, in milliseconds.
	//
	// It is deliberately not a soak, and it is the only kind that IGNORES
	// DurationSeconds as a runtime budget — see BurstOnly. What it proves is
	// that the real FP4 tensor-core instruction path executed on this part and
	// produced the right answer; that is a correctness statement, and holding
	// the same GEMM in a loop for two minutes would not make it a stronger one.
	// It would make it a worse soak than the kinds that exist to be soaks:
	// thermal-soak and gpu-burn both run on the shared duration-honouring load
	// wrapper, and clockprobe holds a load specifically to judge sustained
	// clocks. Converging compute-smoke onto that wrapper would duplicate all
	// three and leave the fleet with no cheap correctness gate at all.
	//
	// So: one kind, one job. Pair it with a soak in a profile when a profile
	// wants both — which is what config/samples/node-acceptance.yaml does.
	KindComputeSmoke TestKind = "compute-smoke"

	KindDCGMDiag    TestKind = "dcgm-diag"      // Node: NVIDIA DCGM diagnostics
	KindThermalSoak TestKind = "thermal-soak"   // Node: hold load, assert no throttle/trip
	KindNCCL        TestKind = "nccl"           // Pair/Group: all-reduce bus-bandwidth over the fabric
	KindIBWriteBW   TestKind = "ib-write-bw"    // Pair: perftest ib_write_bw across the RDMA link
	KindGPUDirect   TestKind = "gpudirect-rdma" // Pair: GPUDirect RDMA path validation

	// KindTCPBaseline is a Pair-scope plain-TCP throughput measurement.
	//
	// It exists to answer the question that always follows an RDMA failure: is
	// the FABRIC broken, or is this node's networking broken in general? The
	// RDMA runners need /dev/infiniband, memlock headroom and a working verbs
	// stack, and a misconfiguration in any of those reports a fabric fault that
	// is really a configuration fault. A plain TCP test over the same pair
	// separates the two in seconds.
	//
	// It is the only fabric-adjacent kind that needs no accelerator and no RDMA
	// hardware at all, which makes it the natural first test in a profile and
	// the natural thing to reach for when something else fails.
	//
	// Its runner carries a MANAGEMENT-PATH GUARD and will skip or error rather
	// than load the interface carrying the node's default route: a burn-in that
	// saturates the path the cluster is managed through takes the fleet out
	// while measuring it.
	KindTCPBaseline TestKind = "tcp-baseline"

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

	// KindDiskIO is a Node-scope storage measurement: sequential throughput and
	// operation latency against a site-declared path, with the page cache
	// bypassed.
	//
	// Storage is otherwise unmeasured by this suite, and on an AI node it is not
	// a minor subsystem: checkpoint writes, dataset streaming and shared-
	// filesystem reads are where a training run stalls when the NVMe underneath
	// it is degrading. A drive with a thermal-throttling controller passes every
	// other test here.
	//
	// Its runner is standard-library-only, which is a LICENSING decision: fio,
	// IOR and elbencho are all GPL, and a copyleft dependency would make this
	// project unpublishable.
	//
	// It writes, so its safety rules are part of the kind: no declared path
	// means no test, it only ever writes a file it created, and it refuses a
	// write that would leave the filesystem near full.
	KindDiskIO TestKind = "disk-io"

	// KindFingerprintProbe reports what the hardware says about itself: PCI
	// vendor and device IDs read from a read-only sysfs mount.
	//
	// Every other way this operator learns a node's identity depends on
	// something else having described it — a device plugin's labels, or the
	// extended resources it advertises. Both are silent on a node where the
	// plugin never came up, and there the node simply looks like it has no
	// accelerator. PCI IDs are reported by the device over a bus and need no
	// software to have chosen to describe it.
	//
	// Its output is EVIDENCE. The runner has no failure path: it passes when it
	// could look and skips when it could not. A count may be gated; an identity
	// string may not, and pkg/contract marks them so the linter says so.
	KindFingerprintProbe TestKind = "fingerprint-probe"

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
	// the part clocks down. Most of the suite is blind to it — compute-smoke
	// passes, dcgm-diag passes, memory-bw passes — because every one of them
	// asserts correctness or health, and none of them asserts speed. A fleet
	// with this fault delivers correct results, slowly, forever, and the loss
	// shows up only in the training-job wall-clock.
	//
	// thermal-soak is the exception: it applies a sustained-clock floor of its
	// own (default 60% of rated boost) and fails below it, and a wedge lands
	// well under that. So this kind is not the only thing that can catch a
	// wedge; what it adds is cost and attribution. It is a ~60s enrollment
	// gate against a soak measured in tens of minutes, and it isolates the
	// clock signal so a wedge is reported AS a wedge — pdWedgeSuspected,
	// throttleClassification, powerLimitRatioPct — rather than as a soak that
	// did not pass. That thermal-soak fires on a genuinely wedged part is
	// inferred from its source, not yet observed on wedged hardware.
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

// BuiltInKinds are the kinds whose runner image, result parser and metric names
// THIS PROJECT owns. They are the closed half of a deliberately open enum:
// TestKind accepts any string, so a site can point its own image at the runner
// contract without a release of this operator, and KindCustom exists to say so
// explicitly.
//
// The distinction has one consumer and it is worth the hand-kept list. A
// threshold naming a metric pkg/contract has never heard of is unremarkable on
// somebody else's runner — the registry is an open world precisely so a
// third-party measurement can be gated on — but on one of these kinds it is
// either a registry entry this project forgot or a gate no runner will ever
// satisfy. Only the second reading is dangerous, and neither is decidable
// without knowing whose runner is about to be asked. See
// verdict.ValidateThresholdsForKind.
//
// ADD A NEW KIND HERE when you add its constant above.
// TestBuiltInKindsCoversEveryDeclaredKind fails otherwise.
var BuiltInKinds = []TestKind{
	KindGPUBurn,
	KindComputeSmoke,
	KindDCGMDiag,
	KindThermalSoak,
	KindNCCL,
	KindIBWriteBW,
	KindGPUDirect,
	KindTCPBaseline,
	KindMemoryBW,
	KindDiskIO,
	KindFingerprintProbe,
	KindHostHealth,
	KindClockProbe,
	KindMemoryStress,
}

var builtInKinds = func() map[TestKind]bool {
	m := make(map[TestKind]bool, len(BuiltInKinds))
	for _, k := range BuiltInKinds {
		m[k] = true
	}
	return m
}()

// IsBuiltIn reports whether this project ships the runner for this kind.
//
// KindCustom answers false because that is what it means, and an UNRECOGNISED
// string answers false too — a kind this operator has never heard of is somebody
// else's runner by definition, and guessing otherwise would apply this project's
// expectations to an image it has never seen. Nothing that fails closed depends
// on this answer; it only decides how much advice an author is owed.
func (k TestKind) IsBuiltIn() bool {
	return builtInKinds[k]
}

// BurstOnly reports whether this kind's runner does one bounded burst of work
// and therefore does NOT use DurationSeconds as a runtime budget.
//
// The operator injects BURNIN_DURATION_SECONDS into every runner at every
// scope, and every runner in this repository reads it — except compute-smoke,
// whose GEMM finishes in milliseconds no matter what it is asked for. That was
// silently true for a long time while a shipped sample asked the kind for 120
// seconds and got milliseconds (issue #25), which is the shape of dishonesty
// this method exists to make impossible: the fact is now declared, the sample
// guard in api/v1alpha1/samples_test.go refuses a duration on such a kind, and
// runners/pins_test.go refuses the reverse — a kind claiming to honour a
// duration whose runner source never reads the variable.
//
// DurationSeconds is not meaningless for a burst kind: it still bounds the
// POD, via the deadline the reconciler derives from it, so an image that hangs
// before reaching its kernel is still killed. What it does not do is decide how
// long the test runs. Setting it on a burst kind therefore states a deadline
// and nothing else, and a profile author who meant "burn this node in for two
// minutes" wants a soak kind instead.
//
// KindCustom is never burst-only: the whole point of it is an image this
// project knows nothing about, and claiming its runner ignores a duration would
// be a guess about somebody else's code.
// KindFingerprintProbe is burst-only for a different reason from
// KindComputeSmoke: not because the work is a bounded burst, but because there
// is no work at all. It reads sysfs once. Holding a node for two minutes to
// re-read the same immutable file would occupy hardware and measure nothing —
// so a profile that wants both an identity and a soak pairs it with a soak
// kind, exactly as it would with compute-smoke.
func (k TestKind) BurstOnly() bool {
	return k == KindComputeSmoke || k == KindFingerprintProbe
}

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
	// +kubebuilder:validation:Enum=nvidia;amd;intel;tenstorrent
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
// misuse at authoring time, before a run has condemned anything: the operator
// runs it against the pinned plan at run start and records what it found on the
// BurnInRun's ThresholdsSound condition.
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
	//
	// The pattern is the apiserver's half of that rule and it is deliberately
	// only the half a regex can express: lowerCamelCase, which is what
	// contract.ValidateMetricName checks first. The other half — that a name
	// this project OWNS declares the unit it is registered with — needs the
	// registry and is enforced at plan time, where a violation refuses the run
	// as a config Error rather than condemning a node.
	// +kubebuilder:validation:Required
	// +kubebuilder:validation:Pattern=`^[a-z][a-zA-Z0-9]*$`
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
	//
	// The pattern admits exactly the finite decimal forms ("20", "10.8", "-3",
	// "1e9") and refuses the ones that parse as float64 without expressing a
	// bound ("NaN", "Inf", "+Inf"), along with the ones that do not parse at all
	// ("twenty"). Every one of those is a gate that fails on every node forever
	// and reports it in the shape of a hardware verdict, so the apiserver is the
	// right place to say no — while the author is still holding the file.
	// +kubebuilder:validation:Required
	// +kubebuilder:validation:Pattern=`^[+-]?([0-9]+(\.[0-9]*)?|\.[0-9]+)([eE][+-]?[0-9]+)?$`
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
