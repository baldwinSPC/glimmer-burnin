package contract

import (
	"fmt"
	"sort"
	"strings"
)

// Metric naming rules.
//
// Metric names cross a repository boundary: a runner emits them, this operator
// evaluates thresholds against them, and an external control plane stores and
// charts them. Nothing reconciles a disagreement, so the name IS the contract.
//
// These are CANONICAL names — the ones that appear in a BurnInRun's status and
// in a delivered envelope. They are not the runner's stdout format. A runner
// prints whatever key=value pairs suit it (the compute-smoke runner uses
// snake_case, e.g. "nonfinite_count"), and the TestKind's result parser
// normalises those keys to the canonical names below. Keeping the two separate
// means a runner's output format can change without breaking every stored
// verdict, and lets already-published runner images stay valid.
//
// The rules:
//
//  1. A metric carrying a physical unit MUST end in a registered unit suffix
//     (see UnitSuffixes). "bandwidth" is not a metric name; "bandwidthGbps" is.
//     A bare unit is not a name either: "tflops" says nothing about what was
//     measured, so the canonical form is "throughputTflops".
//  2. A dimensionless counter MUST NOT carry a unit suffix.
//  3. Names are lowerCamelCase.
//
// Rule 1 is why bandwidthGbps and busBandwidthGBs coexist rather than one being
// renamed to match the other. They are not the same quantity: ib_write_bw
// reports raw link throughput, while NCCL bus bandwidth is a derived figure
// scaled by the collective's communication pattern. Normalising them to a
// single unit would make two different measurements look interchangeable, which
// is a worse failure than an inconsistent-looking pair of names. The unit suffix
// is what keeps them honestly distinguishable.
//
// The same reasoning governs when to mint a new name rather than reuse one.
// Sharing a name across two differently-obtained figures is the failure the
// convention exists to prevent, so an unwarmed single-launch throughput
// (throughputTflops) and a figure sustained across a multi-minute burn
// (sustainedThroughputTflops) are separate metrics even though both are TFLOPS.
// Collapsing them would make one of them thresholdable by accident.
//
// # Why a metric with no number in it is registered at all
//
// Some of what a runner reports is a LABEL, not a measurement: a GPU's name, a
// driver version, clockprobe's "pdWedgeSuspected=true|false|unknown". Those
// names obey the grammar, carry no unit suffix, and are perfectly legitimate
// evidence — but a threshold is evaluated by parsing the value as a float64, so
// a gate on one of them fails closed on EVERY node, forever, and the failure
// reads as a hardware verdict on healthy hardware.
//
// Left unregistered, nothing catches that: the grammar passes, UnitOf answers
// UnitNone (which is the legitimate dimensionless-counter case), and
// SafeToThresholdOn answers true because the registry is an open world. So a
// label-valued metric this project OWNS is registered with ThresholdUse
// Evidence precisely so SafeToThresholdOn answers false and an authoring-time
// linter refuses the gate while its author is still there to fix it. Its
// Description is the text the linter reports, so it names the values the metric
// actually takes and says what to gate on instead.
//
// This is a registration rule for FIRST-PARTY runners only. A third-party
// runner's label-valued metric stays outside the registry and stays permitted,
// which is the open-world rule doing its job.

// Unit is a physical unit a metric can be expressed in.
type Unit string

const (
	UnitGigabitsPerSecond  Unit = "Gbps" // link/wire throughput, as interconnect vendors quote it
	UnitGigabytesPerSecond Unit = "GBs"  // memory and collective bandwidth, as NCCL/DCGM report it
	UnitMegabytesPerSecond Unit = "MBs"
	UnitMicroseconds       Unit = "Us"
	UnitMilliseconds       Unit = "Ms"
	UnitSeconds            Unit = "S"
	UnitCelsius            Unit = "C"
	UnitWatts              Unit = "W"
	UnitPercent            Unit = "Pct"
	UnitTeraflops          Unit = "Tflops"
	// UnitMegahertz is how every driver and every clock probe quotes a core or
	// memory clock, so it is the unit the measurement arrives in. The alternative
	// — normalising clocks to a percentage of the rated boost clock — throws away
	// the absolute figure and requires the runner to know a per-SKU nameplate
	// constant it may not have. sustainedClockPct carries the portable ratio;
	// this unit carries the measurement it was computed from.
	UnitMegahertz Unit = "MHz"

	// UnitNone marks a dimensionless counter. It has no suffix by rule 2.
	UnitNone Unit = ""
)

// UnitSuffixes are the suffixes a dimensional metric name may end in.
//
// Order matters: longer suffixes are checked first so "busBandwidthGBs" is not
// mistaken for a name ending in the UnitSeconds suffix "S".
var UnitSuffixes = []Unit{
	UnitGigabitsPerSecond,
	UnitGigabytesPerSecond,
	UnitMegabytesPerSecond,
	UnitTeraflops,
	UnitMegahertz,
	UnitMicroseconds,
	UnitMilliseconds,
	UnitPercent,
	UnitCelsius,
	UnitWatts,
	UnitSeconds,
}

// ThresholdUse says whether a registered metric is a number acceptance may be
// decided on, or one that is only worth recording.
//
// The distinction is not cosmetic. throughputTflops from compute-smoke is a
// single unwarmed kernel launch: it proves the arch-correct instruction path
// executed, and its magnitude is dominated by launch overhead. A profile author
// who writes `throughputTflops >= 90` gets a threshold that passes or fails on
// scheduling noise, and every resulting verdict — pass or fail — is meaningless
// while looking exactly like a hardware judgement. Marking such metrics lets a
// caller reject that threshold at admission, when the author can still fix it,
// instead of at verdict time when a node has already been condemned.
//
// It is a tri-state rather than a bool so that "nobody decided" is
// distinguishable from "decided: not thresholdable". A new registry entry that
// forgets the field is caught by TestRegistryIsSelfConsistent rather than
// silently inheriting whichever answer the zero value happens to encode.
type ThresholdUse string

const (
	// ThresholdUseUnspecified is the zero value: the metric has not been
	// classified. It is not a valid state for a registered metric.
	ThresholdUseUnspecified ThresholdUse = ""
	// ThresholdUseAcceptance means a threshold against this metric decides a
	// real property of the hardware.
	ThresholdUseAcceptance ThresholdUse = "Acceptance"
	// ThresholdUseEvidence means the metric is recorded as context — a liveness
	// signal, or a nameplate constant read back from the driver — and is not
	// safe to threshold on. The metric's Description says why.
	ThresholdUseEvidence ThresholdUse = "Evidence"
)

// Aggregation says how a metric COMBINES when the same test runs more than once
// and one verdict must be rendered across all of it.
//
// Nothing consumes this yet. It exists because the answer is a property of the
// MEASURAND and belongs beside the name, not in a per-metric switch inside the
// reconciler — that would be exactly the contract-shaped knowledge CLAUDE.md
// keeps out of it, and it would have to be rewritten by anyone adding a metric.
//
// The rules are not interchangeable and picking the wrong one is silent:
//
//	Sum   a per-window count. Two windows of three miscompares is six.
//	Min   a floor, where the WORST window is the verdict. A soak that held
//	      83% of rated clock for eleven hours and 40% for one hour is a part
//	      that dropped to 40%; averaging it would certify the drop away.
//	Max   a ceiling, same reasoning inverted: peak temperature is the peak
//	      across the whole soak, not the peak of the last segment.
//	Last  a nameplate, an identity, or a LIFETIME total the runner already
//	      reports absolutely. Summing a lifetime total across windows
//	      multiplies it by the number of windows.
type Aggregation string

const (
	// AggUnspecified is the zero value and is not valid for a registered
	// metric — TestRegistryIsSelfConsistent refuses it, so "nobody decided"
	// cannot inherit whichever answer the zero value happens to encode. That is
	// the same discipline ThresholdUse uses and for the same reason.
	AggUnspecified Aggregation = ""
	AggSum         Aggregation = "Sum"
	AggMin         Aggregation = "Min"
	AggMax         Aggregation = "Max"
	AggLast        Aggregation = "Last"
)

// AggregationFor returns how a metric combines across windows.
//
// An UNREGISTERED metric aggregates Last, deliberately. The registry is an open
// world and a third-party runner's name is legitimate, but inventing Sum or Min
// semantics for a name nothing has declared would be this operator deciding what
// somebody else's measurement means — and getting it wrong silently, which is
// the failure mode every rule in this file exists to prevent.
func AggregationFor(name string) Aggregation {
	if m, ok := registry[name]; ok {
		return m.Aggregation
	}
	return AggLast
}

// Metric is a registered, canonical metric.
type Metric struct {
	Name string
	Unit Unit
	// Description says what the number means, not merely what it is called.
	// Two bandwidth figures in different units are only safe to store side by
	// side if a reader can tell which one they are looking at.
	Description string
	// ThresholdUse says whether acceptance may be decided on this metric.
	// Every registered metric must state one explicitly.
	ThresholdUse ThresholdUse
	// Aggregation says how the metric combines across repeated windows. Every
	// registered metric must state one explicitly; see Aggregation.
	Aggregation Aggregation
}

// SafeToThresholdOn reports whether a threshold against this metric decides
// something real. An unclassified entry answers false: the registry not having
// an opinion is not the same as the metric being sound to gate a fleet on.
func (m Metric) SafeToThresholdOn() bool {
	return m.ThresholdUse == ThresholdUseAcceptance
}

// registry holds the metrics the project itself emits. It is intentionally not
// a closed world: a runner may emit a metric that is not registered, and that
// is allowed so long as the name obeys the grammar. The registry exists to stop
// the names we DO own from drifting apart, not to forbid new measurements.
var registry = map[string]Metric{
	// --- Interconnect and collective bandwidth -----------------------------
	"bandwidthGbps": {
		Name: "bandwidthGbps", Unit: UnitGigabitsPerSecond,
		Description:  "raw RDMA write throughput on a link, as reported by ib_write_bw",
		Aggregation:  AggMin,
		ThresholdUse: ThresholdUseAcceptance,
	},
	"tcpThroughputGbps": {
		Name: "tcpThroughputGbps", Unit: UnitGigabitsPerSecond,
		Description:  "TCP throughput between two nodes, as reported by iperf3. Deliberately NOT bandwidthGbps: that is an RDMA verbs measurement and this is a kernel-stack one over a different path, and a profile running both would otherwise have the second silently overwrite the first — which is exactly the comparison the test exists to enable",
		Aggregation:  AggMin,
		ThresholdUse: ThresholdUseAcceptance,
	},
	"peakBandwidthGbps": {
		Name: "peakBandwidthGbps", Unit: UnitGigabitsPerSecond,
		Description:  "best single-iteration RDMA write throughput on a link; the average (bandwidthGbps) is the figure a link is accepted on, since one good iteration does not survive a marginal cable",
		Aggregation:  AggMin,
		ThresholdUse: ThresholdUseAcceptance,
	},
	"busBandwidthGBs": {
		Name: "busBandwidthGBs", Unit: UnitGigabytesPerSecond,
		Description:  "NCCL bus bandwidth: algorithm bandwidth scaled by the collective's communication pattern",
		Aggregation:  AggMin,
		ThresholdUse: ThresholdUseAcceptance,
	},
	"algBandwidthGBs": {
		Name: "algBandwidthGBs", Unit: UnitGigabytesPerSecond,
		Description:  "NCCL algorithm bandwidth: message size over collective time, with no scaling for the communication pattern; unlike busBandwidthGBs it is not comparable across collectives or rank counts",
		Aggregation:  AggMin,
		ThresholdUse: ThresholdUseAcceptance,
	},

	// --- Memory copy and stress bandwidth ----------------------------------
	"hostToDeviceBandwidthGBs": {
		Name: "hostToDeviceBandwidthGBs", Unit: UnitGigabytesPerSecond,
		Description:  "measured copy bandwidth from host memory into device memory; bounded by the host link (PCIe, or C2C on a coherent part)",
		Aggregation:  AggMin,
		ThresholdUse: ThresholdUseAcceptance,
	},
	"deviceToHostBandwidthGBs": {
		Name: "deviceToHostBandwidthGBs", Unit: UnitGigabytesPerSecond,
		Description:  "measured copy bandwidth from device memory back to host memory; asymmetry with hostToDeviceBandwidthGBs is normal and is why the two directions are separate metrics",
		Aggregation:  AggMin,
		ThresholdUse: ThresholdUseAcceptance,
	},
	"deviceToDeviceBandwidthGBs": {
		Name: "deviceToDeviceBandwidthGBs", Unit: UnitGigabytesPerSecond,
		Description:  "on-device copy bandwidth, device memory to device memory; bounded by the memory subsystem rather than by the host link",
		Aggregation:  AggMin,
		ThresholdUse: ThresholdUseAcceptance,
	},
	"memoryBandwidthGBs": {
		Name: "memoryBandwidthGBs", Unit: UnitGigabytesPerSecond,
		Description:  "sustained device memory bandwidth achieved over the whole measurement window, not a peak sample",
		Aggregation:  AggMin,
		ThresholdUse: ThresholdUseAcceptance,
	},
	"readBandwidthMBs": {
		Name: "readBandwidthMBs", Unit: UnitMegabytesPerSecond,
		Description:  "sustained read throughput reported by the host memory stress tool",
		Aggregation:  AggMin,
		ThresholdUse: ThresholdUseAcceptance,
	},
	"writeBandwidthMBs": {
		Name: "writeBandwidthMBs", Unit: UnitMegabytesPerSecond,
		Description:  "sustained write throughput reported by the host memory stress tool",
		Aggregation:  AggMin,
		ThresholdUse: ThresholdUseAcceptance,
	},

	// --- Latency and duration ----------------------------------------------
	"latencyUs": {
		Name: "latencyUs", Unit: UnitMicroseconds,
		Description:  "round-trip latency",
		Aggregation:  AggMax,
		ThresholdUse: ThresholdUseAcceptance,
	},
	"minLatencyUs": {
		Name: "minLatencyUs", Unit: UnitMicroseconds,
		Description:  "the fastest single round trip observed. It is the link's floor — what the fabric can do with nothing in the way — and it is EVIDENCE rather than acceptance, because a gate on the best sample any run happened to catch says nothing about what the fabric delivers under load",
		Aggregation:  AggMin,
		ThresholdUse: ThresholdUseEvidence,
	},
	"maxLatencyUs": {
		Name: "maxLatencyUs", Unit: UnitMicroseconds,
		Description:  "the slowest single round trip observed. A single outlier is normal on a shared fabric — a scheduler tick, an interrupt — so this is evidence that qualifies p99LatencyUs rather than a gate of its own; a fleet gating on one worst sample fails healthy links on noise",
		Aggregation:  AggMax,
		ThresholdUse: ThresholdUseEvidence,
	},
	"p99LatencyUs": {
		Name: "p99LatencyUs", Unit: UnitMicroseconds,
		Description:  "99th-percentile round-trip latency: the number a collective actually runs at. A collective proceeds at the speed of its slowest participant on EVERY iteration, so a link with a healthy mean and a p99 an order of magnitude worse degrades every job on the fleet while passing a bandwidth gate outright. This is the acceptance-grade latency figure; latencyUs is the mean and is not",
		Aggregation:  AggMax,
		ThresholdUse: ThresholdUseAcceptance,
	},
	"tcpRttUs": {
		Name: "tcpRttUs", Unit: UnitMicroseconds,
		Description:  "mean smoothed round-trip time the sender's TCP stack observed during the test. A path property rather than a fabric one — it includes host stack and scheduling — so it is a useful ceiling gate and a poor floor",
		Aggregation:  AggMax,
		ThresholdUse: ThresholdUseAcceptance,
	},
	"messageRateMpps": {
		Name: "messageRateMpps", Unit: UnitNone,
		Description:  "millions of messages per second the link sustained — a SEPARATE finding from bandwidth rather than a restatement of it. A link can carry its rated bytes while delivering far fewer messages per second than its class, which bandwidth cannot see and a collective feels immediately, because a collective is many small messages rather than one large one",
		Aggregation:  AggMin,
		ThresholdUse: ThresholdUseAcceptance,
	},
	"ioLatencyUs": {
		Name: "ioLatencyUs", Unit: UnitMicroseconds,
		Description:  "mean per-operation latency of the test's read/write loop; distinct from latencyUs, which is a network round trip",
		Aggregation:  AggMax,
		ThresholdUse: ThresholdUseAcceptance,
	},
	"elapsedS": {
		Name: "elapsedS", Unit: UnitSeconds,
		Description:  "wall-clock seconds the test body ran, excluding image pull and container start; a soak that exits early has not proven what its duration claims",
		Aggregation:  AggSum,
		ThresholdUse: ThresholdUseAcceptance,
	},
	"durationRequestedS": {
		Name: "durationRequestedS", Unit: UnitSeconds,
		Description:  "the duration the runner was ASKED for (BURNIN_DURATION_SECONDS), echoed back before any clamping; it is an input read back, not a measurement, and elapsedS is the one that says what actually happened",
		Aggregation:  AggLast,
		ThresholdUse: ThresholdUseEvidence,
	},
	"warmupS": {
		Name: "warmupS", Unit: UnitSeconds,
		Description:  "seconds of load applied before sampling began, so an unwarmed part's legitimately idle clocks are not sampled; a probe's own schedule, not a property of the part",
		Aggregation:  AggLast,
		ThresholdUse: ThresholdUseEvidence,
	},
	"sampleWindowS": {
		Name: "sampleWindowS", Unit: UnitSeconds,
		Description:  "seconds of the run during which measurements were taken (elapsedS minus warmupS); it bounds how much evidence the other metrics rest on rather than saying anything about the hardware",
		Aggregation:  AggLast,
		ThresholdUse: ThresholdUseEvidence,
	},

	// --- Thermal, power and clocks -----------------------------------------
	"gpuTempC": {
		Name: "gpuTempC", Unit: UnitCelsius,
		Description:  "peak GPU temperature observed during the test",
		Aggregation:  AggMax,
		ThresholdUse: ThresholdUseAcceptance,
	},
	"meanTempUnderLoadC": {
		Name: "meanTempUnderLoadC", Unit: UnitCelsius,
		Description:  "GPU temperature averaged across the load window; distinct from gpuTempC, which is the peak — a part that runs hot throughout and one that spikes once are different findings and must not share a name",
		Aggregation:  AggMax,
		ThresholdUse: ThresholdUseAcceptance,
	},
	"tempAtMinClockC": {
		Name: "tempAtMinClockC", Unit: UnitCelsius,
		Description:  "the temperature recorded at the sample where the SM clock was lowest; it exists to attribute a clock shortfall to heat or to a power-delivery wedge, and its value depends on which single sample happened to be the minimum — so it explains a verdict rather than deciding one",
		Aggregation:  AggLast,
		ThresholdUse: ThresholdUseEvidence,
	},
	"thermalTempThresholdC": {
		Name: "thermalTempThresholdC", Unit: UnitCelsius,
		Description:  "the temperature at or above which the runner will attribute a clock shortfall to heat; a configured input echoed back so a verdict can be audited against the threshold that produced it, not a reading",
		Aggregation:  AggLast,
		ThresholdUse: ThresholdUseEvidence,
	},
	"powerDrawW": {
		Name: "powerDrawW", Unit: UnitWatts,
		Description:  "peak board power draw observed during the test",
		Aggregation:  AggMax,
		ThresholdUse: ThresholdUseAcceptance,
	},
	"meanPowerW": {
		Name: "meanPowerW", Unit: UnitWatts,
		Description:  "board power averaged across the load window; the floor form is the useful gate, because a part that draws far LESS than its budget under full load is the power-delivery wedge signature rather than a well-behaved one",
		Aggregation:  AggMax,
		ThresholdUse: ThresholdUseAcceptance,
	},
	"enforcedPowerLimitW": {
		Name: "enforcedPowerLimitW", Unit: UnitWatts,
		Description:  "the power limit the driver is currently enforcing on the board — on GB10 this is the negotiated USB-C Power-Delivery contract made visible, and a degraded supply or cable shows up here well below the board's default; absolute and therefore SKU-specific, so a fleet-wide gate belongs on powerLimitRatioPct",
		Aggregation:  AggMin,
		ThresholdUse: ThresholdUseAcceptance,
	},
	"defaultPowerLimitW": {
		Name: "defaultPowerLimitW", Unit: UnitWatts,
		Description:  "the board's nameplate default power limit, read back from the driver so powerLimitRatioPct can be audited against its denominator; a nameplate constant identifies the SKU rather than its health, so gating on it fails a heterogeneous fleet for no hardware reason",
		Aggregation:  AggLast,
		ThresholdUse: ThresholdUseEvidence,
	},
	"powerLimitRatioPct": {
		Name: "powerLimitRatioPct", Unit: UnitPercent,
		Description:  "enforcedPowerLimitW as a percentage of defaultPowerLimitW: the SKU-portable form of 'is this board allowed its full power budget?', and well under 100 is the power-delivery wedge. NOTE it is omitted entirely on a part whose driver does not expose both limits (GB10 reports power.limit as [N/A], which the runner records in nvmlUnsupported), and an omitted metric fails its threshold closed — so a gate on this must be scoped to parts known to answer",
		Aggregation:  AggMin,
		ThresholdUse: ThresholdUseAcceptance,
	},
	"sustainedClockPct": {
		Name: "sustainedClockPct", Unit: UnitPercent,
		Description:  "achieved SM clock as a percentage of the part's rated boost clock, averaged over the load window; the SKU-portable form of smClockMHz",
		Aggregation:  AggMin,
		ThresholdUse: ThresholdUseAcceptance,
	},
	"smClockMHz": {
		Name: "smClockMHz", Unit: UnitMegahertz,
		Description:  "SM core clock sustained under load; absolute and therefore SKU-specific, which is why a fleet-wide gate belongs on sustainedClockPct",
		Aggregation:  AggMin,
		ThresholdUse: ThresholdUseAcceptance,
	},
	"memClockMHz": {
		Name: "memClockMHz", Unit: UnitMegahertz,
		Description:  "device memory clock sustained under load",
		Aggregation:  AggMin,
		ThresholdUse: ThresholdUseAcceptance,
	},
	"ratedBoostClockMHz": {
		Name: "ratedBoostClockMHz", Unit: UnitMegahertz,
		Description:  "the part's nameplate boost clock, read back from the driver so sustainedClockPct can be audited against its denominator; a nameplate constant identifies the SKU rather than its health, so gating on it fails a heterogeneous fleet for no hardware reason",
		Aggregation:  AggLast,
		ThresholdUse: ThresholdUseEvidence,
	},
	"minSmClockPct": {
		Name: "minSmClockPct", Unit: UnitPercent,
		Description:  "the LOWEST single SM-clock sample of the load window, as a percentage of rated boost; a floor on it asserts the part never dropped out, which sustainedClockPct's average can hide. It is a single sample, so it moves on one transient dip — calibrate it against measured fleet behaviour rather than against a spec sheet",
		Aggregation:  AggMin,
		ThresholdUse: ThresholdUseAcceptance,
	},
	"maxSmClockPct": {
		Name: "maxSmClockPct", Unit: UnitPercent,
		Description:  "the HIGHEST single SM-clock sample of the load window, as a percentage of rated boost, recorded so the spread around sustainedClockPct is visible; it is not an acceptance figure, because a wedged part that boosts for one sample and collapses satisfies a floor on it while failing the sustained behaviour this measurement exists to judge",
		Aggregation:  AggMax,
		ThresholdUse: ThresholdUseEvidence,
	},
	"clockFloorPct": {
		Name: "clockFloorPct", Unit: UnitPercent,
		Description:  "the sustained-clock floor the runner was configured with, echoed back so a verdict can be read against the bar it was judged by; a configured input, not a measurement",
		Aggregation:  AggLast,
		ThresholdUse: ThresholdUseEvidence,
	},
	"thermalClockFloorPct": {
		Name: "thermalClockFloorPct", Unit: UnitPercent,
		Description:  "the more lenient sustained-clock floor the runner applies when a shortfall is positively attributed to heat, echoed back as configured; a configured input, not a measurement",
		Aggregation:  AggLast,
		ThresholdUse: ThresholdUseEvidence,
	},
	"clockFloorAppliedPct": {
		Name: "clockFloorAppliedPct", Unit: UnitPercent,
		Description:  "which of the two configured floors this run actually judged against (see clockFloorBasis); it records the runner's own decision so a pass or fail can be reproduced, and gating on it would gate on the runner's configuration rather than on the part",
		Aggregation:  AggMin,
		ThresholdUse: ThresholdUseEvidence,
	},
	"gpuUtilizationPct": {
		Name: "gpuUtilizationPct", Unit: UnitPercent,
		Description:  "device utilization averaged over the load window. It is the headline SYMPTOM of a clock wedge — it reads perfectly healthy while the clock does not — so it is published beside the clock to make that fault legible, not to be gated on: the runner already refuses to judge a run whose utilization shows the load never landed, and a threshold here would turn that same measurement artifact into a hardware verdict",
		Aggregation:  AggMax,
		ThresholdUse: ThresholdUseEvidence,
	},
	"memUtilizationPct": {
		Name: "memUtilizationPct", Unit: UnitPercent,
		Description:  "device memory-controller utilization averaged over the load window; for a deliberately register-resident load it is expected to be near zero, so its value describes the probe's own load shape rather than the part",
		Aggregation:  AggMax,
		ThresholdUse: ThresholdUseEvidence,
	},

	// --- Compute throughput -------------------------------------------------
	"throughputTflops": {
		Name: "throughputTflops", Unit: UnitTeraflops,
		Description:  "achieved throughput; for compute-smoke this is a single unwarmed launch — a liveness signal, not a benchmark, and not safe to threshold on",
		Aggregation:  AggLast,
		ThresholdUse: ThresholdUseEvidence,
	},
	"sustainedThroughputTflops": {
		Name: "sustainedThroughputTflops", Unit: UnitTeraflops,
		Description:  "throughput averaged across a sustained burn, after the part has reached its steady thermal and clock state; unlike throughputTflops this is a benchmark figure and may be thresholded",
		Aggregation:  AggMin,
		ThresholdUse: ThresholdUseAcceptance,
	},
	"sustainedFmaThroughputTflops": {
		Name: "sustainedFmaThroughputTflops", Unit: UnitTeraflops,
		Description:  "throughput of a register-resident FP32 FMA chain sustained across the load window — no memory traffic, so it moves in lockstep with the achieved clock and is an independent cross-check on it. It is DELIBERATELY not sustainedThroughputTflops: that name belongs to a dense GEMM burn, the two differ by a large factor on the same healthy part, and sharing a name would silently condemn every node a gate calibrated on the other one touched",
		Aggregation:  AggMin,
		ThresholdUse: ThresholdUseAcceptance,
	},
	"peakFmaThroughputTflops": {
		Name: "peakFmaThroughputTflops", Unit: UnitTeraflops,
		Description:  "the best single measurement window of the same FMA load, published so throughputConsistencyPct can be audited against its denominator; as the best window rather than the sustained one it is the figure a collapsing part still satisfies, which is why the consistency ratio and not this is the gate",
		Aggregation:  AggMax,
		ThresholdUse: ThresholdUseEvidence,
	},
	"throughputConsistencyPct": {
		Name: "throughputConsistencyPct", Unit: UnitPercent,
		Description:  "sustainedFmaThroughputTflops as a percentage of peakFmaThroughputTflops: a part that starts fast and collapses mid-window scores low here even when its mean clock still clears every floor, which is the case an average is blind to",
		Aggregation:  AggMin,
		ThresholdUse: ThresholdUseAcceptance,
	},

	// --- Correctness counters -----------------------------------------------
	"nonfiniteCount": {
		Name: "nonfiniteCount", Unit: UnitNone,
		Description:  "count of NaN or Inf values in a kernel's output",
		Aggregation:  AggSum,
		ThresholdUse: ThresholdUseAcceptance,
	},
	"miscompares": {
		Name: "miscompares", Unit: UnitNone,
		Description:  "count of computed values that did not match the reference — a wrong answer from hardware that reported success",
		Aggregation:  AggSum,
		ThresholdUse: ThresholdUseAcceptance,
	},
	"sdcDetections": {
		Name: "sdcDetections", Unit: UnitNone,
		Description:  "count of distinct silent-data-corruption incidents; miscompares counts every mismatched value, so one corrupted region can produce many miscompares and a single detection",
		Aggregation:  AggSum,
		ThresholdUse: ThresholdUseAcceptance,
	},

	// --- Hardware fault counters --------------------------------------------
	"eccErrors": {
		Name: "eccErrors", Unit: UnitNone,
		Description:  "count of ECC errors observed during the test",
		Aggregation:  AggLast,
		ThresholdUse: ThresholdUseAcceptance,
	},
	"memoryErrors": {
		Name: "memoryErrors", Unit: UnitNone,
		Description:  "count of hardware memory errors reported by the host memory stress tool (stressapptest hardware incidents); host memory, not device ECC",
		Aggregation:  AggSum,
		ThresholdUse: ThresholdUseAcceptance,
	},
	"xidEvents": {
		Name: "xidEvents", Unit: UnitNone,
		Description:  "count of NVIDIA Xid errors the driver logged during the test window",
		Aggregation:  AggSum,
		ThresholdUse: ThresholdUseAcceptance,
	},
	"remappedRows": {
		Name: "remappedRows", Unit: UnitNone,
		Description:  "count of device memory rows the driver has remapped; a rising count is a degrading part even while every test still passes",
		Aggregation:  AggLast,
		ThresholdUse: ThresholdUseAcceptance,
	},
	"pcieReplayErrors": {
		Name: "pcieReplayErrors", Unit: UnitNone,
		Description:  "count of PCIe replay events on the device's link — a link-integrity signal that usually precedes a bandwidth shortfall",
		Aggregation:  AggSum,
		ThresholdUse: ThresholdUseAcceptance,
	},
	"tcpRetransmits": {
		Name: "tcpRetransmits", Unit: UnitNone,
		Description:  "count of TCP segment retransmissions during the test. The signal worth gating on: a link that reaches line rate while retransmitting heavily has a problem that throughput alone will not show, and it will surface later as tail latency in a collective",
		Aggregation:  AggSum,
		ThresholdUse: ThresholdUseAcceptance,
	},
	"acceleratorCount": {
		Name: "acceleratorCount", Unit: UnitNone,
		Description:  "how many accelerators the node's PCI bus reports. A legitimate acceptance gate, unlike the identity strings beside it: a node that should hold eight cards and reports seven has lost one, and that is a hardware fault rather than a fact about which vendor made it",
		Aggregation:  AggLast,
		ThresholdUse: ThresholdUseAcceptance,
	},
	"ioErrors": {
		Name: "ioErrors", Unit: UnitNone,
		Description:  "count of I/O errors observed during a storage test. A zero here IS a measurement — the run reached the device and counted none — unlike a bandwidth figure, which is omitted when nothing moved rather than reported as zero",
		Aggregation:  AggSum,
		ThresholdUse: ThresholdUseAcceptance,
	},
	"nicLinkDownEvents": {
		Name: "nicLinkDownEvents", Unit: UnitNone,
		Description:  "count of link-down transitions on the node's fabric NICs during the test window",
		Aggregation:  AggSum,
		ThresholdUse: ThresholdUseAcceptance,
	},
	"throttleEvents": {
		Name: "throttleEvents", Unit: UnitNone,
		Description:  "count of clock-throttle events observed during the test",
		Aggregation:  AggSum,
		ThresholdUse: ThresholdUseAcceptance,
	},
	"throttledSamples": {
		Name: "throttledSamples", Unit: UnitNone,
		Description:  "count of individual samples in which the driver reported a CAPPING throttle reason; throttleEvents counts transitions into that state, so a part throttled once for the whole window scores 1 event and many samples, and the pair separates a brief excursion from a permanent cap",
		Aggregation:  AggSum,
		ThresholdUse: ThresholdUseAcceptance,
	},
	"throttleReasonsMask": {
		Name: "throttleReasonsMask", Unit: UnitNone,
		Description:  "the union of the driver's raw throttle-reason bits over the run, kept so a stored result can be re-classified later against a vendor's own bit definitions. It is not a quantity: it is a bitfield in decimal, arithmetic on it is meaningless, and it includes the idle bit — which is not a throttle at all — so `throttleReasonsMask Equal 0` fails any part that was briefly idle mid-window. Gate on throttledSamples instead",
		Aggregation:  AggMax,
		ThresholdUse: ThresholdUseEvidence,
	},
	// DELIBERATELY NOT REGISTERED: the per-reason sample counters that accompany
	// the two above — gpuIdleSamples, swPowerCapSamples, hwSlowdownSamples,
	// hwThermalSlowdownSamples and the rest, one per bit of throttleReasonsMask.
	//
	// They are the mask's decomposition, published so a stored result can be
	// re-read without decoding a bitfield, and they are diagnostic in the
	// strictest sense: their meaning is the vendor's own enum rather than ours,
	// their magnitude scales with the configured sample interval so they are not
	// comparable across profiles, and everything an acceptance decision needs
	// from them is already in throttledSamples and throttleEvents. Registering a
	// name commits this project to it and to its unit permanently, and these are
	// the ones most likely to change shape when a second vendor's runner reports
	// a different set of reasons.
	//
	// Leaving them out is safe in the way the label metrics below are NOT,
	// because they are ordinary integers: a threshold on one evaluates and means
	// what it says, and is simply narrower than the aggregate. There is no
	// fails-closed trap here.
	//
	// It is not free, though, and the cost is deliberate.
	// verdict.ValidateThresholdsForKind reports an unregistered metric on a
	// kind whose runner this project ships, so an author who does gate on one is
	// told to register it. That is the right moment to ask: writing a threshold
	// is what promotes a name from incidental evidence to acceptance-deciding,
	// and registration is owed at that point rather than in advance of any
	// consumer. Until then these stay out, and the advice stays advice — it does
	// not block a run.
	"unsupportedReads": {
		Name: "unsupportedReads", Unit: UnitNone,
		Description:  "count of driver properties the runner asked for and was refused (the names are in nvmlUnsupported). It records how complete the probe's view was, and a non-zero count is NORMAL on a part that genuinely does not expose everything — GB10 refuses the power-limit reads — so gating on it condemns healthy hardware for its driver's silence, which is the same mistake as gating ECC counters on a part with no ECC",
		Aggregation:  AggMax,
		ThresholdUse: ThresholdUseEvidence,
	},

	// --- Test-execution counters --------------------------------------------
	"diagTestsFailed": {
		Name: "diagTestsFailed", Unit: UnitNone,
		Description:  "count of individual diagnostic subtests that returned a failure; zero alongside a non-zero runner exit means the suite could not run rather than that the hardware failed",
		Aggregation:  AggSum,
		ThresholdUse: ThresholdUseAcceptance,
	},
	"iterationsCompleted": {
		Name: "iterationsCompleted", Unit: UnitNone,
		Description:  "count of stress iterations the runner completed; a soak that completes far fewer than expected did less work than its duration suggests",
		Aggregation:  AggSum,
		ThresholdUse: ThresholdUseAcceptance,
	},
	"samplesTaken": {
		Name: "samplesTaken", Unit: UnitNone,
		Description:  "count of measurement samples the probe collected during its window; it bounds how much evidence the derived statistics rest on, and its magnitude is set by the configured sample interval rather than by the part",
		Aggregation:  AggSum,
		ThresholdUse: ThresholdUseEvidence,
	},
	"loadLaunches": {
		Name: "loadLaunches", Unit: UnitNone,
		Description:  "count of load-kernel launches issued during the measured window; the runner sizes each launch to a fixed wall time on the part in front of it, so this lands near the same figure on a healthy and a wedged part alike and says nothing about either — elapsedS is the metric that says the window ran",
		Aggregation:  AggSum,
		ThresholdUse: ThresholdUseEvidence,
	},
	"loadThreads": {
		Name: "loadThreads", Unit: UnitNone,
		Description:  "total CUDA threads the load was launched with, derived from the part's SM count; it records how the probe sized itself to the device, which makes a throughput figure reproducible and is not a property of the device's health",
		Aggregation:  AggLast,
		ThresholdUse: ThresholdUseEvidence,
	},
	"loadItersPerLaunch": {
		Name: "loadItersPerLaunch", Unit: UnitNone,
		Description:  "inner-loop iterations per load-kernel launch, chosen by calibrating against the live part so each launch takes a fixed wall time; it is an artefact of that calibration — a slower part gets a smaller number by design — so it explains the load rather than judging the part",
		Aggregation:  AggLast,
		ThresholdUse: ThresholdUseEvidence,
	},

	// --- Identity, configuration and classification LABELS -------------------
	//
	// Every metric below is registered because its value is NOT A NUMBER, and a
	// threshold is evaluated by parsing the value as a float64. Unregistered,
	// each of these would sail through an authoring-time lint — valid grammar,
	// no unit suffix, open-world SafeToThresholdOn — and then fail closed on
	// every node in the fleet forever, reading as a hardware verdict on healthy
	// hardware. Registering them as Evidence is what makes the linter refuse the
	// gate instead. See the "Why a metric with no number in it is registered at
	// all" note above.
	//
	// The runner's own exit code is what decides a node on these findings: a
	// wedge is a clockprobe exit 1 with the classification in the message. A
	// profile does not need — and cannot have — a threshold to reach the same
	// verdict.
	"tcpTestInterface": {
		Name: "tcpTestInterface", Unit: UnitNone,
		Description:  "the interface the TCP baseline actually loaded. Recorded because the management-path guard's decision is only auditable if the answer is in the result: a number with no interface beside it cannot be checked against the node's topology later",
		Aggregation:  AggLast,
		ThresholdUse: ThresholdUseEvidence,
	},
	"tcpMgmtInterface": {
		Name: "tcpMgmtInterface", Unit: UnitNone,
		Description:  "the interface carrying this node's default route, which the guard treats as the management path and refuses to load. Reported alongside tcpTestInterface so a reader can see the two were different",
		Aggregation:  AggLast,
		ThresholdUse: ThresholdUseEvidence,
	},
	"pciAddresses": {
		Name: "pciAddresses", Unit: UnitNone,
		Description:  "PCI slots of the node's accelerators, comma-joined in slot order (\"0000:01:00.0,0000:02:00.0\"). What an engineer walking to the rack needs",
		Aggregation:  AggLast,
		ThresholdUse: ThresholdUseEvidence,
	},
	"pciVendorIds": {
		Name: "pciVendorIds", Unit: UnitNone,
		Description:  "PCI vendor IDs as the devices report them (\"0x10de\"). Ground truth in a way a label is not: readable on a node with no device plugin, no NFD and no labels at all",
		Aggregation:  AggLast,
		ThresholdUse: ThresholdUseEvidence,
	},
	"pciDeviceIds": {
		Name: "pciDeviceIds", Unit: UnitNone,
		Description:  "PCI device IDs as the devices report them, in the same order as pciAddresses",
		Aggregation:  AggLast,
		ThresholdUse: ThresholdUseEvidence,
	},
	"acceleratorVendors": {
		Name: "acceleratorVendors", Unit: UnitNone,
		Description:  "distinct accelerator vendors on the node, resolved from PCI vendor IDs where this project has a name for one and left as the raw hex ID where it does not. A LABEL: gating on it fails a node for being the vendor it is, which is a targeting decision belonging in a node selector",
		Aggregation:  AggLast,
		ThresholdUse: ThresholdUseEvidence,
	},
	"acceleratorDrivers": {
		Name: "acceleratorDrivers", Unit: UnitNone,
		Description:  "kernel modules bound to the node's accelerators. OMITTED entirely when nothing is bound, because a device the kernel sees and no driver claims is the shape of a node whose plugin never came up — and an empty value would read as absent information rather than an absent driver",
		Aggregation:  AggLast,
		ThresholdUse: ThresholdUseEvidence,
	},
	"diskIoPath": {
		Name: "diskIoPath", Unit: UnitNone,
		Description:  "the directory a storage test measured. A label: recorded because a throughput figure with no path beside it cannot be checked against the node's topology later — measuring a container overlay and measuring an NVMe produce the same-shaped number",
		Aggregation:  AggLast,
		ThresholdUse: ThresholdUseEvidence,
	},
	"gpuName": {
		Name: "gpuName", Unit: UnitNone,
		Description:  "the accelerator's product name as the driver reports it (\"NVIDIA GB10\"). A label, not a measurement: no comparison against it is arithmetic, and identity is a fleet-inventory question rather than an acceptance one — target a heterogeneous fleet with node selectors, not with a threshold",
		Aggregation:  AggLast,
		ThresholdUse: ThresholdUseEvidence,
	},
	"computeCap": {
		Name: "computeCap", Unit: UnitNone,
		Description:  "the CUDA compute capability the device reports, as major.minor (\"12.1\"). It is a VERSION, not a quantity, and the resemblance is the trap: compared as a decimal, 12.1 and 12.10 are the same value and any ordering that looks right does so by coincidence of notation. A runner that cares about capability decides it in the runner, where the two components are compared separately",
		Aggregation:  AggLast,
		ThresholdUse: ThresholdUseEvidence,
	},
	"pciBusId": {
		Name: "pciBusId", Unit: UnitNone,
		Description:  "the device's PCI bus address, recorded so a verdict can be tied to a physical slot when a node holds several parts; an address, with no ordering a threshold could use",
		Aggregation:  AggLast,
		ThresholdUse: ThresholdUseEvidence,
	},
	"driverVersion": {
		Name: "driverVersion", Unit: UnitNone,
		Description:  "the NVIDIA driver version string (\"580.82.09\"). A three-component version does not parse as a float at all, so a gate on it fails closed on every node; driver-version policy belongs in a fleet's node labels and admission rules, not in a hardware acceptance verdict",
		Aggregation:  AggLast,
		ThresholdUse: ThresholdUseEvidence,
	},
	"builtCudaArch": {
		Name: "builtCudaArch", Unit: UnitNone,
		Description:  "the nvcc arch target this runner IMAGE was compiled for (\"sm_121a\"), printed by the runner so a stored result names the image that produced it. A property of the image that was pinned rather than of the part it landed on, and the runner already reports a mismatch itself as an Error naming the fix",
		Aggregation:  AggLast,
		ThresholdUse: ThresholdUseEvidence,
	},
	"migMode": {
		Name: "migMode", Unit: UnitNone,
		Description:  "whether MIG partitioning is enabled on the device (\"enabled\"/\"disabled\"). A configuration state expressed as a word; a runner whose measurement is unattributable under MIG skips (exit 2) on its own, which is the reported form of that finding",
		Aggregation:  AggLast,
		ThresholdUse: ThresholdUseEvidence,
	},
	"nvmlUnsupported": {
		Name: "nvmlUnsupported", Unit: UnitNone,
		Description:  "comma-separated names of the driver properties the runner asked for and was refused, recorded rather than substituted so an unsupported read can never be mistaken for a measured value. A list; unsupportedReads is its count, and neither is safe to gate on (see that entry)",
		Aggregation:  AggLast,
		ThresholdUse: ThresholdUseEvidence,
	},
	"configWarnings": {
		Name: "configWarnings", Unit: UnitNone,
		Description:  "semicolon-separated notes about configuration the runner adjusted or could not honour, so a result carries the reason it did not run exactly as asked. Free-form prose, and a gate on it can only fail",
		Aggregation:  AggLast,
		ThresholdUse: ThresholdUseEvidence,
	},
	"throttleReasons": {
		Name: "throttleReasons", Unit: UnitNone,
		Description:  "comma-separated names of the throttle reasons the driver latched during the run (\"none\" when it latched nothing) — the human-readable form of throttleReasonsMask. A list of labels; gate on throttledSamples for \"was it throttled at all\"",
		Aggregation:  AggLast,
		ThresholdUse: ThresholdUseEvidence,
	},
	"throttleClassification": {
		Name: "throttleClassification", Unit: UnitNone,
		Description:  "why the runner believes the clock fell short: one of none|thermal|powerCap|applicationClocks|unknown. This is an ATTRIBUTION, and a label — it exists to send an engineer to a cable or to a cooling path rather than to a die, and the runner has already failed the node itself when it is anything but none. Gate the clock (sustainedClockPct) and read this to find out why",
		Aggregation:  AggLast,
		ThresholdUse: ThresholdUseEvidence,
	},
	"pdWedgeSuspected": {
		Name: "pdWedgeSuspected", Unit: UnitNone,
		Description:  "whether the run matched the power-delivery wedge signature — slow, cool, and not thermally throttled: one of true|false|unknown. Deliberately TRI-STATE and therefore a label, not a boolean: without a temperature a wedge cannot be told from a thermal throttle, and \"we could not tell\" must not collapse into \"false\" and read as an all-clear. The runner fails the node on this itself; a threshold cannot express the three states and fails closed on all of them",
		Aggregation:  AggLast,
		ThresholdUse: ThresholdUseEvidence,
	},
	"clockFloorBasis": {
		Name: "clockFloorBasis", Unit: UnitNone,
		Description:  "which floor the run was judged against, general|thermal, naming the reason clockFloorAppliedPct is what it is; a label recording the runner's own decision",
		Aggregation:  AggLast,
		ThresholdUse: ThresholdUseEvidence,
	},

	// host-health's labels. Same class and same duty as the block above: each of
	// these is a WORD, host-health is a kind this project owns the runner for,
	// and an unregistered name is assumed thresholdable — so a gate on any of
	// them passed every authoring-time check and then failed closed on every node
	// forever. hostHealthVersion is the sly one here, exactly as computeCap is
	// above: it is "1", so it parses, and a gate on it silently compares a schema
	// version as a decimal.
	"fabricCountersSaturated": {
		Name: "fabricCountersSaturated", Unit: UnitNone,
		Description:  "how many IB/RoCE error counters were observed PINNED at their maximum. The IB-spec PMA counters do not wrap — they saturate and stay there until reset — so a port erroring long enough to peg a 16-bit counter shows a delta of ZERO across any window and reads as a clean link. This is the only signal that sees it, and `Equal 0` is the gate",
		Aggregation:  AggMax,
		ThresholdUse: ThresholdUseAcceptance,
	},
	"fabricSaturatedCounters": {
		Name: "fabricSaturatedCounters", Unit: UnitNone,
		Description:  "comma-separated names of the pegged counters, so a reader knows whether the link is throwing symbol errors or dropping subnet-management packets. A list of labels; gate on fabricCountersSaturated for the count",
		Aggregation:  AggLast,
		ThresholdUse: ThresholdUseEvidence,
	},
	"fabricCounterStatus": {
		Name: "fabricCounterStatus", Unit: UnitNone,
		Description:  "whether any IB/RoCE port was visible to the probe at all: ok|absent. `absent` is legitimate — /sys/class/infiniband may be invisible inside the pod depending on the RDMA namespace mode — and it is why every fabric counter is OMITTED rather than zeroed there, so a gate on one fails closed instead of certifying a fabric nobody looked at",
		Aggregation:  AggLast,
		ThresholdUse: ThresholdUseEvidence,
	},
	"xidSource": {
		Name: "xidSource", Unit: UnitNone,
		Description:  "where the Xid scan read from: kmsg|kernlog|none. A label naming the PROVENANCE of xidEvents, and the value that matters most is the one a gate cannot express — \"none\" means the scan did not run, and the counters it would have produced are omitted rather than zeroed, so xidEvents already fails closed on its own",
		Aggregation:  AggLast,
		ThresholdUse: ThresholdUseEvidence,
	},
	"nodeReady": {
		Name: "nodeReady", Unit: UnitNone,
		Description:  "the runner's own verdict echoed as true|false, so a stored result carries it next to the evidence. It is a restatement of the exit code, not an independent measurement, and gating on it would ask a threshold to re-derive a decision the operator already has — as a word, which compares as a float64 and fails closed on both of its values",
		Aggregation:  AggLast,
		ThresholdUse: ThresholdUseEvidence,
	},
	"hostHealthVersion": {
		Name: "hostHealthVersion", Unit: UnitNone,
		Description:  "the shape of the metric set this runner emitted, so a consumer charting these counters over months can tell when the set changed. A schema version of the OUTPUT, saying nothing whatever about the hardware; it parses as a number, which is precisely why it has to be registered",
		Aggregation:  AggLast,
		ThresholdUse: ThresholdUseEvidence,
	},
	"hostHealthStage": {
		Name: "hostHealthStage", Unit: UnitNone,
		Description:  "the furthest stage the runner reached, streamed as it goes so a process that is OOM-killed or brought down by a Go runtime fatal error still says where it died. On a run that completed it is always \"done\", which is why it is worthless as a gate and valuable as evidence: it only carries information when everything else is missing",
		Aggregation:  AggLast,
		ThresholdUse: ThresholdUseEvidence,
	},
	"hostHealthPanic": {
		Name: "hostHealthPanic", Unit: UnitNone,
		Description:  "the one-line summary of a panic the runner recovered from before exiting 3, present only on a run that crashed. It is recorded as a metric as well as in the message because a metric cannot be overwritten by a later stack frame, and the stored result outlives the pod whose log holds the trace",
		Aggregation:  AggLast,
		ThresholdUse: ThresholdUseEvidence,
	},
}

// Lookup returns the registered metric, if this is one the project owns.
func Lookup(name string) (Metric, bool) {
	m, ok := registry[name]
	return m, ok
}

// SafeToThresholdOn reports whether acceptance may be decided on this metric.
//
// An UNREGISTERED name answers true, which is deliberate and is not a hole in
// the fails-closed rule. The registry is an open world by design (see above): a
// runner may report a measurement this package has never heard of, and its
// author — not this package — knows what it means. Answering false there would
// forbid every threshold on every third-party runner, which pushes authors back
// toward metrics nothing can gate on.
//
// A REGISTERED name that has not been classified answers false. That is the
// opposite default on purpose: a name the project owns but has not thought
// about is our own gap, and this package should not vouch for it. The registry
// test forbids that state, so it should never be reachable in a release.
//
// The intended use is admission-time rejection of a threshold, not verdict-time
// suppression. A metric that is unsafe to threshold on is still worth
// delivering, so nothing here filters what an envelope reports.
func SafeToThresholdOn(name string) bool {
	m, ok := registry[name]
	if !ok {
		return true
	}
	return m.SafeToThresholdOn()
}

// Registered lists every registered metric name, sorted. Useful for generating
// documentation and for a consumer to discover what it should expect.
func Registered() []string {
	names := make([]string, 0, len(registry))
	for n := range registry {
		names = append(names, n)
	}
	sort.Strings(names)
	return names
}

// UnitOf reports the unit a well-formed metric name declares.
//
// It works on unregistered names too: the unit is carried by the name itself,
// which is the entire point of the convention. A name with no recognised suffix
// is dimensionless.
func UnitOf(name string) Unit {
	for _, u := range UnitSuffixes {
		if strings.HasSuffix(name, string(u)) && len(name) > len(u) {
			return u
		}
	}
	return UnitNone
}

// ValidateMetricName enforces the grammar.
//
// It deliberately does NOT require the name to be registered. Rejecting unknown
// metrics would mean a runner could not report a new measurement without a
// release of this package, which would push runner authors toward stuffing data
// into the message string where nothing can threshold on it.
func ValidateMetricName(name string) error {
	if name == "" {
		return fmt.Errorf("metric name must not be empty")
	}
	if r := rune(name[0]); r < 'a' || r > 'z' {
		return fmt.Errorf("metric name %q must be lowerCamelCase (start with a lowercase letter)", name)
	}
	for _, r := range name {
		isLower := r >= 'a' && r <= 'z'
		isUpper := r >= 'A' && r <= 'Z'
		isDigit := r >= '0' && r <= '9'
		if !isLower && !isUpper && !isDigit {
			return fmt.Errorf("metric name %q must be alphanumeric lowerCamelCase; found %q", name, r)
		}
	}
	// If the project owns this name, the registered unit is authoritative and
	// the caller must not have invented a variant that reads the same but means
	// something else.
	if m, ok := registry[name]; ok {
		if got := UnitOf(name); got != m.Unit {
			return fmt.Errorf("metric %q is registered with unit %q but its name declares %q", name, m.Unit, got)
		}
	}
	return nil
}

// ValidateMetrics checks every key in a runner's metric map, reporting all
// offending names rather than only the first — a runner author fixing names
// should not have to rediscover them one release at a time.
func ValidateMetrics(metrics map[string]string) error {
	var bad []string
	for name := range metrics {
		if err := ValidateMetricName(name); err != nil {
			bad = append(bad, err.Error())
		}
	}
	if len(bad) == 0 {
		return nil
	}
	sort.Strings(bad)
	return fmt.Errorf("%s", strings.Join(bad, "; "))
}
