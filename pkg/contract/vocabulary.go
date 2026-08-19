// SPDX-License-Identifier: Apache-2.0
// Copyright the Glimmer authors.

package contract

// THE ACCEPTANCE VOCABULARY — the words a threshold is written in.
//
// These types lived in api/v1alpha1 until the hoist proposed in the GEP-0178
// amendment and tracked as #274. They define what a gate SAYS, which is a
// statement about hardware and not about Kubernetes — but living in the CRD
// package meant pkg/verdict, the shared verdict brain, imported the CRD to get
// its own vocabulary and dragged apimachinery and controller-runtime into every
// consumer. #274 measured that: 0, 0 and 9 Kubernetes modules for contract,
// runner and verdict.
//
// The cost fell hardest exactly where the project can least afford it. GEP-0178
// specifies that a separate, pre-Kubernetes path adopts this brain precisely so a
// bare-metal run and an in-cluster run cannot disagree about the same hardware
// — and that path would have linked controller-runtime it has no cluster to
// use. One brain, two dispatchers, and only one of them is a Kubernetes client.
//
// api/v1alpha1 keeps every one of these names as a type ALIAS, so the CRD wire
// format is unchanged, controller-gen reads the kubebuilder markers here
// (crd generation already scans the whole module), and no caller had to change
// to keep compiling. The window for doing it this cheaply was open because no
// external consumer had adopted pkg/verdict yet: the only external consumer
// imports only pkg/contract today, so there were zero callers to break. It
// would not have stayed open.
//
// hack/invariants/kubernetesfree_test.go is the ledger that now enforces the
// result rather than asserting it in prose.

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

	// KindFabricSoak is a Pair-scope RDMA link soak: the same measurement
	// ib-write-bw takes, iterated over hours.
	//
	// ib-write-bw answers "does this link work right now" and passes on a link
	// with a marginal optic, a badly seated cable, or a transceiver that fails
	// once it is warm. Those present in production as a slow training job or an
	// occasional collective timeout rather than as a link that is down, and they
	// are found by running the fabric for hours.
	//
	// What it reports is not the peak: the WORST window, the spread across
	// windows, the count of iterations that failed, and the port error-counter
	// delta measured across the run rather than the NIC's lifetime total.
	KindFabricSoak TestKind = "fabric-soak"

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

	// KindGemmSweep is a Node-scope GEMM across PRECISIONS, one execution per
	// precision, selected by the `precision` variant axis.
	//
	// compute-smoke proves one thing precisely: that a GB10's NVFP4
	// block-scaled path executes and produces correct numbers. That narrowness
	// is deliberate and stays — but it leaves the rest of the compute surface
	// unmeasured. A part can pass FP4 and fail FP64. Tensor-core paths and
	// CUDA-core paths are different silicon and fail independently, and
	// precision-specific defects are exactly what an acceptance suite should
	// find before a training run does.
	//
	// It is ONE kind with a variant axis rather than five kinds, because the
	// question "does this part compute correctly at precision X" is one question
	// asked five times. The runner reads BURNIN_VARIANT_PRECISION; the
	// controller never interprets it.
	//
	// Thresholds are PER VARIANT and are not interchangeable: an FP4 throughput
	// floor and an FP64 throughput floor are different numbers about different
	// hardware paths, and a single floor across them would be meaningless.
	KindGemmSweep TestKind = "gemm-sweep"

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
	KindGemmSweep,
	KindDCGMDiag,
	KindThermalSoak,
	KindNCCL,
	KindIBWriteBW,
	KindGPUDirect,
	KindTCPBaseline,
	KindMemoryBW,
	KindDiskIO,
	KindFabricSoak,
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

// sustainedLoadKinds are the kinds whose runner deliberately HOLDS a load for
// the length of its window, rather than measuring something and stopping.
//
// ADD A NEW KIND HERE, or to the list in the test, when you add its constant
// above. TestEveryKindDeclaresWhetherItHoldsLoad fails otherwise, because
// forgetting is the failure mode that matters: a new soak kind that nobody
// classified is a soak that gets drained.
var sustainedLoadKinds = map[TestKind]bool{
	// The shared duration-honouring load wrapper (runners/*/soak_core.cuh).
	KindThermalSoak: true,
	KindGPUBurn:     true,

	// Holds a known, steady, clock-bound load for its whole window — that is
	// how it judges sustained clocks at all. Short, but short is not cool: on a
	// chassis that has already heat-soaked it reaches the trip point too.
	KindClockProbe: true,

	// The same RDMA measurement ib-write-bw takes, iterated over HOURS, and it
	// exists specifically to find the parts that fail once they are warm.
	KindFabricSoak: true,

	// stressapptest for the whole window. Host RAM rather than the
	// accelerator — which matters here rather than exempting it, because the
	// watchdog reading this declaration judges the host CPU package too.
	KindMemoryStress: true,

	// Levels 3 and 4 run DCGM's targeted_stress and sm_stress plugins for
	// ~15 minutes, and the level is a runner env var. Whether THIS execution
	// drives load is therefore not a property of the kind — so it is declared
	// by its worst case, because the operator must not read a runner's
	// environment to decide what a kind is.
	KindDCGMDiag: true,
}

// DrivesSustainedLoad reports whether this kind's runner holds the part under
// deliberate load for its window, rather than taking a measurement and stopping.
//
// It exists for ONE consumer: the heat declaration the operator writes on a node
// it is loading (AnnotationHeatExpected), which tells a node-local thermal
// watchdog that a trip here is the test working rather than the cooling failing.
// It is a statement about what the RUNNER does, and it belongs beside BurstOnly
// for the same reason that one does — the fact is the kind's, and a controller
// deciding it from a name would go stale the moment a runner changed.
//
// It is not the inverse of BurstOnly. host-health honours a duration and applies
// no load at all; memory-bw and gemm-sweep take measurements that end when the
// measurement ends; nccl, ib-write-bw, gpudirect-rdma and tcp-baseline are short
// fabric measurements, and fabric-soak is the one that iterates for hours.
//
// KindCustom answers FALSE, and an unrecognised kind with it. The whole point of
// custom is an image this project knows nothing about, and this answer would
// suppress a safety action on somebody else's behalf — which is a worse guess to
// get wrong than BurstOnly's. A site running a custom soak asserts the hold on
// the node itself instead.
func (k TestKind) DrivesSustainedLoad() bool {
	return sustainedLoadKinds[k]
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

// DeepCopyInto is hand-written, and deliberately not generated.
//
// Threshold is embedded in CRD specs, so apimachinery's deepcopy contract
// applies to it — but pkg/contract is NOT a Kubernetes API package and must not
// become one to satisfy that. Marking it for controller-gen made the generator
// try to deepcopy Envelope's time.Time fields, which it cannot: it expects
// metav1.Time. That is the correct refusal, and the right answer is these six
// lines rather than reshaping a neutral package around a generator.
//
// Threshold is four string-typed fields, so a shallow copy IS a deep copy —
// which is exactly what controller-gen produced when the type lived in
// api/v1alpha1 (`*out = *in`). If a reference-typed field is ever added here,
// this must be updated, and api/v1alpha1's own generated code will not catch it
// for you.
func (in *Threshold) DeepCopyInto(out *Threshold) {
	*out = *in
}

// DeepCopy returns a new Threshold with the same value.
func (in *Threshold) DeepCopy() *Threshold {
	if in == nil {
		return nil
	}
	out := new(Threshold)
	in.DeepCopyInto(out)
	return out
}
