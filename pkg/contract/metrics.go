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
	// Combination says how the metric combines across the REPORTERS of one
	// test — the ranks of a Group — as distinct from Aggregation, which is
	// across repeated windows of one reporter. See Combination.
	//
	// Unlike ThresholdUse and Aggregation this may be left unset, because its
	// zero value is the SAFE answer rather than a silent guess: an unclassified
	// metric is not merged at all, and a threshold on it fails closed.
	Combination Combination
	// MayBeUnmeasurable marks a metric that is legitimately `n/a` — a POSITIVE
	// claim, never an omission — on EVERY single-device node: the cross-device
	// spreads (nothing to spread across) and the peer-bandwidth matrices (no
	// peer path with one accelerator). The default applicability, Required,
	// fails closed on n/a — so a `Required` gate on one of these fails every
	// single-device node forever, on the one phase never retried, and the
	// failure reads as a hardware verdict. verdict.ValidateThresholdsForKind
	// reads this to advise RequiredIfMeasurable at authoring time; it changes
	// nothing about evaluation, which still fails closed exactly as before.
	//
	// Deliberately NARROWER than "may be n/a on SOME hardware" — eccErrors and
	// remappedRows are also n/a on some SKUs (a part with no ECC subsystem),
	// but only some, and a fleet whose parts all have ECC legitimately writes
	// `eccErrors Equal 0` with the default Required applicability. Flagging
	// every gate on those two would be noise on the common case rather than
	// advice about a real trap (#405) — they stay false here on purpose; see
	// their own registry comments.
	MayBeUnmeasurable bool
}

// Combination says how a metric combines across the reporters of a single
// test — today, the ranks of a Group.
//
// It exists because Aggregation cannot answer this question, and the two
// disagree in exactly the cases that matter. gpuTempC aggregates Max across
// windows and Max is also right across ranks; miscompares is Sum on both. But
// elapsedS is Sum across windows and Max across ranks — eight ranks running 300s
// took 300s, not 2400s — and eccErrors is Last across windows, which across
// ranks means "whatever rank 0 said", the precise defect this type was added to
// fix (#121).
//
// The zero value is unclassified and is deliberately the conservative one. A
// Group merge drops an unclassified key rather than electing a winner for it, so
// a metric nobody has thought about fails a threshold closed instead of
// certifying every node in the group on one rank's evidence.
type Combination string

const (
	// CombineUnclassified is the zero value: nobody has said how this metric
	// combines across ranks, so nothing may claim it describes the group.
	CombineUnclassified Combination = ""
	// CombineCollective means one value describes the whole group and every
	// rank is reporting the same measurand — a collective's bus bandwidth, or a
	// label like gpuName. The lowest rank's answer stands.
	CombineCollective Combination = "Collective"
	// CombineSum, CombineMax and CombineMin combine per-reporter readings the
	// obvious way. The direction is the metric's, not the operator's guess:
	// Max for a temperature keeps the hottest node, Sum for a fault counter
	// keeps every rank's faults, Min for a bandwidth keeps the slowest.
	CombineSum Combination = "Sum"
	CombineMax Combination = "Max"
	CombineMin Combination = "Min"
)

// CombinationFor returns how a metric combines across the reporters of one test.
//
// An UNREGISTERED metric is unclassified, and so is a registered one that has
// not stated a Combination. Both mean the same thing and both fail closed: this
// operator inventing Sum or Max semantics for a measurement nobody declared is
// how a group-wide claim comes to rest on one member, which is the whole of
// #121.
func CombinationFor(name string) Combination {
	if m, ok := registry[name]; ok {
		return m.Combination
	}
	return CombineUnclassified
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
		Combination:  CombineMin,
		ThresholdUse: ThresholdUseAcceptance,
	},
	"tcpThroughputGbps": {
		Name: "tcpThroughputGbps", Unit: UnitGigabitsPerSecond,
		Description:  "TCP throughput between two nodes, as reported by iperf3. Deliberately NOT bandwidthGbps: that is an RDMA verbs measurement and this is a kernel-stack one over a different path, and a profile running both would otherwise have the second silently overwrite the first — which is exactly the comparison the test exists to enable",
		Aggregation:  AggMin,
		Combination:  CombineMin,
		ThresholdUse: ThresholdUseAcceptance,
	},
	"minBandwidthGbps": {
		Name: "minBandwidthGbps", Unit: UnitGigabitsPerSecond,
		Description:  "the WORST window of a fabric soak. The figure a link is accepted on: a soak that averaged fine and spent ninety seconds at a third of line rate has found a marginal optic, and the mean hides it completely",
		Aggregation:  AggMin,
		Combination:  CombineMin,
		ThresholdUse: ThresholdUseAcceptance,
	},
	"meanBandwidthGbps": {
		Name: "meanBandwidthGbps", Unit: UnitGigabitsPerSecond,
		Description:  "mean across a fabric soak's windows. Evidence: its distance from minBandwidthGbps is what says whether a link is steadily slow or intermittently bad, which are different repairs",
		Aggregation:  AggLast,
		ThresholdUse: ThresholdUseEvidence,
	},
	"p1BandwidthGbps": {
		Name: "p1BandwidthGbps", Unit: UnitGigabitsPerSecond,
		Description:  "1st percentile across a fabric soak's windows, nearest-rank. On a long soak a better floor than the single minimum, which one scheduling hiccup can drag down",
		Aggregation:  AggMin,
		ThresholdUse: ThresholdUseAcceptance,
	},
	"bandwidthStdDevGbps": {
		Name: "bandwidthStdDevGbps", Unit: UnitGigabitsPerSecond,
		Description:  "spread across a fabric soak's windows. A flapping link and a steadily slow one both have a low minimum; only the flapping one has a wide spread, and telling them apart decides whether an optic or a cable budget is at fault",
		Aggregation:  AggMax,
		ThresholdUse: ThresholdUseEvidence,
	},
	"peakBandwidthGbps": {
		Name: "peakBandwidthGbps", Unit: UnitGigabitsPerSecond,
		Description:  "best single-iteration RDMA write throughput on a link; the average (bandwidthGbps) is the figure a link is accepted on, since one good iteration does not survive a marginal cable",
		Aggregation:  AggMin,
		ThresholdUse: ThresholdUseAcceptance,
	},
	"busBandwidthGBs": {
		Name: "busBandwidthGBs", Unit: UnitGigabytesPerSecond,
		Description: "NCCL bus bandwidth: algorithm bandwidth scaled by the collective's communication pattern. " +
			"NAMED THE SAME AT EVERY SCOPE AND FOR EVERY COLLECTIVE (Pair's cross-node all-reduce, Group's, and " +
			"Node's intra-node all-reduce/allgather/reducescatter/alltoall), a deliberate decision against the " +
			"registry's own 'two differently-obtained figures under one name' rule: it is warranted here because " +
			"the SAME nccl-tests formula computes it in every case (busBandwidth(algbw, nranks) is one function, " +
			"reused, not reimplemented per scope), Scope on the TestResult already discriminates a link's number " +
			"from a collective's, and a threshold is written per BurnInTest rather than globally on the metric " +
			"name — so nothing ever compares a Node-scope reading against a Pair-scope gate by accident. Contrast " +
			"throughputTflops vs sustainedThroughputTflops, which stayed separate because they are measured by " +
			"different PROTOCOLS on the same part; this is one protocol run in more places",
		Aggregation:  AggMin,
		Combination:  CombineCollective,
		ThresholdUse: ThresholdUseAcceptance,
	},
	"algBandwidthGBs": {
		Name: "algBandwidthGBs", Unit: UnitGigabytesPerSecond,
		Description: "NCCL algorithm bandwidth: message size over collective time, with no scaling for the " +
			"communication pattern; unlike busBandwidthGBs it is not comparable across collectives or rank " +
			"counts on its own — busBandwidthGBs is. Shares its name across scopes and collectives for the same " +
			"reason busBandwidthGBs does; see that entry",
		Aggregation:  AggMin,
		Combination:  CombineCollective,
		ThresholdUse: ThresholdUseAcceptance,
	},
	"ranks": {
		Name: "ranks", Unit: UnitNone,
		Description:  "how many ranks joined this collective. At Node scope this is the device count ncclCommInitAll actually used, printed so a fleet reading busBandwidthGBs from an intra-node run can tell an 8-GPU result from a 2-GPU one without cross-referencing the node's inventory. Evidence, not a gate: whether the count is what a profile expects is a plan-time or topology question, not this metric's job",
		Aggregation:  AggLast,
		ThresholdUse: ThresholdUseEvidence,
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
	"peerReadBandwidthGBs": {
		Name: "peerReadBandwidthGBs", Unit: UnitGigabytesPerSecond,
		Description:       "worst cell of the all-pairs GPU-to-GPU READ bandwidth matrix, over NVLink, xGMI or a PCIe switch. The MINIMUM rather than the mean because a fabric is as good as its worst link, and a mean over the matrix hides the single degraded lane this measurement exists to find. Unmeasurable (n/a) on a node with one accelerator, or with no peer path between them",
		Aggregation:       AggMin,
		ThresholdUse:      ThresholdUseAcceptance,
		MayBeUnmeasurable: true,
	},
	"peerReadBandwidthMaxGBs": {
		Name: "peerReadBandwidthMaxGBs", Unit: UnitGigabytesPerSecond,
		Description:  "best cell of the same matrix. Evidence, not acceptance: its distance from peerReadBandwidthGBs is what makes ONE degraded link visible as a spread rather than as a slightly lower average",
		Aggregation:  AggMax,
		ThresholdUse: ThresholdUseEvidence,
	},
	"peerWriteBandwidthGBs": {
		Name: "peerWriteBandwidthGBs", Unit: UnitGigabytesPerSecond,
		Description:       "worst cell of the all-pairs GPU-to-GPU WRITE bandwidth matrix. Reported separately from the read direction because a link can degrade asymmetrically and averaging the two would hide it",
		Aggregation:       AggMin,
		ThresholdUse:      ThresholdUseAcceptance,
		MayBeUnmeasurable: true,
	},
	"peerWriteBandwidthMaxGBs": {
		Name: "peerWriteBandwidthMaxGBs", Unit: UnitGigabytesPerSecond,
		Description:  "best cell of the write matrix. Evidence, for the same reason as the read direction",
		Aggregation:  AggMax,
		ThresholdUse: ThresholdUseEvidence,
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
		Combination:  CombineMin,
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
		Combination:  CombineMax,
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
		Combination:  CombineMax,
		ThresholdUse: ThresholdUseEvidence,
	},
	"p99LatencyUs": {
		Name: "p99LatencyUs", Unit: UnitMicroseconds,
		Description:  "99th-percentile round-trip latency: the number a collective actually runs at. A collective proceeds at the speed of its slowest participant on EVERY iteration, so a link with a healthy mean and a p99 an order of magnitude worse degrades every job on the fleet while passing a bandwidth gate outright. This is the acceptance-grade latency figure; latencyUs is the mean and is not",
		Aggregation:  AggMax,
		Combination:  CombineMax,
		ThresholdUse: ThresholdUseAcceptance,
	},
	"tcpRttUs": {
		Name: "tcpRttUs", Unit: UnitMicroseconds,
		Description:  "mean smoothed round-trip time the sender's TCP stack observed during the test. A path property rather than a fabric one — it includes host stack and scheduling — so it is a useful ceiling gate and a poor floor",
		Aggregation:  AggMax,
		Combination:  CombineMax,
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
		Combination:  CombineMax,
		ThresholdUse: ThresholdUseAcceptance,
	},
	"elapsedS": {
		Name: "elapsedS", Unit: UnitSeconds,
		Description:  "wall-clock seconds the test body ran, excluding image pull and container start; a soak that exits early has not proven what its duration claims",
		Aggregation:  AggSum,
		Combination:  CombineMax,
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
		Combination:  CombineMax,
		ThresholdUse: ThresholdUseAcceptance,
	},
	"meanTempUnderLoadC": {
		Name: "meanTempUnderLoadC", Unit: UnitCelsius,
		Description:  "GPU temperature averaged across the load window; distinct from gpuTempC, which is the peak — a part that runs hot throughout and one that spikes once are different findings and must not share a name",
		Aggregation:  AggMax,
		Combination:  CombineMax,
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
		Combination:  CombineMax,
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
	"ratedMemClockMHz": {
		Name: "ratedMemClockMHz", Unit: UnitMegahertz,
		Description:  "the part's nameplate memory clock, read back from the driver (nvmlDeviceGetMaxClockInfo, NVML_CLOCK_MEM) so memClockPct can be audited against its denominator — the same role ratedBoostClockMHz plays for sustainedClockPct; a nameplate constant identifies the SKU rather than its health, so gating on it fails a heterogeneous fleet for no hardware reason. Omitted, never fabricated, on a device whose driver does not answer the call",
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
	"memClockPct": {
		Name: "memClockPct", Unit: UnitPercent,
		Description:  "device memory clock sustained under load, as a percentage of the part's rated memory clock — the memory-domain analog of sustainedClockPct, added to measure rather than assume the clockprobe README's claim that a power-delivery wedge caps only the compute clock and leaves the memory path alone (issue #301). Deliberately Evidence, not Acceptance: nobody has yet watched a wedged part's memory clock and confirmed whether it moves, so there is no floor to gate on. n/a when either the achieved or the rated memory clock could not be read from this device",
		Aggregation:  AggMin,
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

	// --- GEMM sweep (gemm-sweep) --------------------------------------------
	//
	// One kind, one execution per precision, selected by the `precision` variant
	// axis. Thresholds are per variant: an FP4 throughput floor and an FP64
	// floor are different numbers about different silicon.
	"achievedTflops": {
		Name: "achievedTflops", Unit: UnitTeraflops,
		Description: "sustained GEMM throughput at the precision this execution ran, in TFLOP/s. " +
			"Compare it only against a floor written for the SAME precision — tensor-core and " +
			"CUDA-core paths differ by more than an order of magnitude and a single floor across " +
			"them means nothing",
		// A FLOOR takes Min: the worst window describes the part. A run that
		// held 700 TFLOP/s for eleven minutes and 90 for one is a part that
		// dropped to 90, and averaging certifies the drop away.
		Aggregation:  AggMin,
		ThresholdUse: ThresholdUseAcceptance,
	},
	"maxRelativeError": {
		Name: "maxRelativeError", Unit: UnitNone,
		Description: "the largest relative deviation between the device result and the host reference " +
			"over the whole output. Dimensionless. What counts as acceptable is PRECISION-SPECIFIC — " +
			"an FP4 tolerance applied to FP64 would accept a broken part, and an FP64 tolerance " +
			"applied to FP4 would condemn a working one",
		// The WORST deviation describes the part, so segments take Max.
		Aggregation:  AggMax,
		ThresholdUse: ThresholdUseAcceptance,
	},
	"gemmPrecision": {
		Name: "gemmPrecision", Unit: UnitNone,
		Description: "which precision this execution actually ran (fp4, fp8, bf16, tf32, fp64). " +
			"A LABEL: it is echoed so a stored result is self-describing, and it is what makes a " +
			"sweep's cells distinguishable long after the profile that produced them was edited",
		// Last: the label of the most recent segment. Summing a word is
		// meaningless, and so is taking its minimum.
		Aggregation: AggLast,
		// Evidence, not Acceptance. Its values are WORDS, and a threshold is
		// compared as a float64 — so a gate on it would fail closed on every
		// node forever while reading as a hardware verdict.
		ThresholdUse: ThresholdUseEvidence,
	},
	"gemmShape": {
		Name: "gemmShape", Unit: UnitNone,
		Description: "the M x N x K the sweep ran, e.g. \"8192x8192x8192\". A LABEL, and the thing " +
			"that makes a throughput number comparable: two TFLOP/s figures from different shapes " +
			"are not the same measurement",
		Aggregation: AggLast,
		// Evidence for the same reason as gemmPrecision: "8192x8192x8192" is
		// not a number, and a gate on it can never be satisfied.
		ThresholdUse: ThresholdUseEvidence,
	},

	// --- Correctness counters -----------------------------------------------
	"maxAbsError": {
		Name: "maxAbsError", Unit: UnitNone,
		Description:  "largest absolute difference between a GEMM cell and its reference. Evidence beside maxRelativeError, which is the gateable form: an absolute error means nothing without the magnitude it is relative to, and maxAbsRef carries that",
		Aggregation:  AggMax,
		ThresholdUse: ThresholdUseEvidence,
	},
	"maxAbsRef": {
		Name: "maxAbsRef", Unit: UnitNone,
		Description:  "largest absolute value in the reference matrix, the denominator maxRelativeError is taken against. Recorded so a stored result can be re-derived rather than trusted",
		Aggregation:  AggMax,
		ThresholdUse: ThresholdUseEvidence,
	},
	"totalKernelMs": {
		Name: "totalKernelMs", Unit: UnitMilliseconds,
		Description:  "summed device time across the measured GEMM launches. The denominator achievedTflops was computed from, kept so the throughput figure can be checked rather than believed",
		Aggregation:  AggSum,
		ThresholdUse: ThresholdUseEvidence,
	},
	"nonfiniteCount": {
		Name: "nonfiniteCount", Unit: UnitNone,
		Description:  "count of NaN or Inf values in a kernel's output",
		Aggregation:  AggSum,
		Combination:  CombineSum,
		ThresholdUse: ThresholdUseAcceptance,
	},
	"miscompares": {
		Name: "miscompares", Unit: UnitNone,
		Description:  "count of computed values that did not match the reference — a wrong answer from hardware that reported success",
		Aggregation:  AggSum,
		Combination:  CombineSum,
		ThresholdUse: ThresholdUseAcceptance,
	},
	"sdcDetections": {
		Name: "sdcDetections", Unit: UnitNone,
		Description:  "count of distinct silent-data-corruption incidents; miscompares counts every mismatched value, so one corrupted region can produce many miscompares and a single detection",
		Aggregation:  AggSum,
		Combination:  CombineSum,
		ThresholdUse: ThresholdUseAcceptance,
	},

	// --- Hardware fault counters --------------------------------------------
	"eccErrors": {
		Name: "eccErrors", Unit: UnitNone,
		Description:  "count of ECC errors observed during the test",
		Aggregation:  AggLast,
		Combination:  CombineSum,
		ThresholdUse: ThresholdUseAcceptance,
		// NOT MayBeUnmeasurable, deliberately: unlike the spreads and the peer
		// bandwidths, this is n/a only on SOME SKUs (GB10's unified LPDDR5X has
		// no ECC subsystem at all — see the runner's n/a sentinel), not on
		// every single-device node. A fleet whose parts all have ECC legitimately
		// writes `eccErrors Equal 0` with the default Required applicability,
		// and that gate is exactly right for them — flagging it here would be
		// noise on the common case rather than advice about a real trap (#405).
	},
	"memoryErrors": {
		Name: "memoryErrors", Unit: UnitNone,
		Description:  "count of hardware memory errors reported by the host memory stress tool (stressapptest hardware incidents); host memory, not device ECC",
		Aggregation:  AggSum,
		Combination:  CombineSum,
		ThresholdUse: ThresholdUseAcceptance,
	},
	// TWO SOURCES, BOTH READING /dev/kmsg, DIFFERENT WINDOWS — a deliberate
	// decision, not the drift #311 fixed. runners/host-health counts Xid lines
	// (and, on any vendor, the amdgpu reset/RAS lines its own broader
	// kernelHwErrors heuristic matches) over ITS OWN, usually short, Node-scope
	// window. runners/thermal-soak and runners/gpu-burn (and their -rocm
	// counterparts) additionally count them over THEIR OWN load window — the
	// specific interval this test held the part under stress — using a
	// deliberately NARROWER amdgpu pattern than host-health's (see
	// runners/thermal-soak/kmsg/kmsg_watch.h), because this reaches an
	// Acceptance gate rather than host-health's Evidence-only kernelHwErrors,
	// and a false positive here is a fabricated hardware verdict rather than a
	// merely-noisy diagnostic.
	//
	// This is safe for the SAME reason foldMetrics' Sum aggregation is safe
	// across a segmented soak's windows: each source counts only what it
	// itself observed, in its own scope, and Combination:CombineSum is what a
	// Group-scope collective's multiple ranks already rely on to mean "the
	// total across every reporter", not "one reporter's exclusive domain". A
	// profile running BOTH host-health and a soak test gets two independent
	// TestResults, each honestly scoped to what THAT test's own window covered
	// — not one shared counter two runners race to write.
	//
	// On AMD, the value counts amdgpu reset/uncorrectable-RAS log lines rather
	// than an NVIDIA Xid; the name is kept rather than split, matching the
	// busBandwidthGBs precedent in runners/nccl: the SEVERITY CLASS this name
	// means — a driver-logged, catastrophic, kernel-log-visible GPU fault — is
	// vendor-independent even though the literal word "Xid" is NVIDIA's own.
	"xidEvents": {
		Name: "xidEvents", Unit: UnitNone,
		Description:  "count of driver-logged catastrophic GPU faults during the test window — NVIDIA Xid errors, or on AMD hardware the soak family's own amdgpu reset/uncorrectable-RAS pattern",
		Aggregation:  AggSum,
		Combination:  CombineSum,
		ThresholdUse: ThresholdUseAcceptance,
	},
	// The CODE, which is a different quantity from the COUNT above and must not
	// be confused with it. DCGM's DCGM_FI_DEV_XID_ERRORS holds "the specific
	// XID error" rather than a running total, so it answers "which" and never
	// "how many" — dcgm-diag reported a subtraction of two such codes as
	// xidEvents until #311, and read 0 for a GPU sitting at one code all window.
	//
	// Evidence rather than Acceptance, for a reason that outlives the naming
	// fix: dcgm-diag's OWN reading is LIFETIME-scoped, so a non-zero code may
	// name an Xid from weeks before this run — gating on it would condemn a
	// node for history the test never observed. The window-scoped COUNT is
	// xidEvents, above.
	//
	// runners/thermal-soak and runners/gpu-burn (NVIDIA only — there is no AMD
	// equivalent numeric code, see kmsg/kmsg_watch.h) ALSO emit this name, and
	// their reading is WINDOW-scoped — the specific Xid, if any, seen during
	// THAT TEST'S OWN load window — which is why it stays Evidence rather than
	// gaining Acceptance now that a second, safer-scoped source exists: under a
	// SEGMENTED soak, AggLast keeps only the final segment's window-scoped
	// code, so an earlier segment's Xid can vanish from the aggregate even
	// though xidEvents (Sum) still correctly counts it. That loss is
	// acceptable for a field nothing may gate on and unacceptable for one that
	// could, which is exactly why this field stays Evidence-only.
	"lastXidCode": {
		Name: "lastXidCode", Unit: UnitNone,
		Description:  "code of the most recent NVIDIA Xid error. dcgm-diag reads a lifetime-scoped device field and prints 0 when it reads 0; the soak family extracts the code from the kmsg line within that test's own load window and OMITS the key when no code was extracted — its 0 would be a value nobody measured. Which Xid, never how many — the count is xidEvents",
		Aggregation:  AggLast,
		ThresholdUse: ThresholdUseEvidence,
	},
	"remappedRows": {
		Name: "remappedRows", Unit: UnitNone,
		Description:  "count of device memory rows the driver has remapped; a rising count is a degrading part even while every test still passes",
		Aggregation:  AggLast,
		Combination:  CombineSum,
		ThresholdUse: ThresholdUseAcceptance,
		// NOT MayBeUnmeasurable — same reasoning as eccErrors just above: n/a
		// only on the specific SKUs that have no ECC subsystem, not on every
		// single-device node. See that comment.
	},
	"pcieReplayErrors": {
		Name: "pcieReplayErrors", Unit: UnitNone,
		Description:  "count of PCIe replay events on the device's link — a link-integrity signal that usually precedes a bandwidth shortfall",
		Aggregation:  AggSum,
		Combination:  CombineSum,
		ThresholdUse: ThresholdUseAcceptance,
	},
	"tcpRetransmits": {
		Name: "tcpRetransmits", Unit: UnitNone,
		Description:  "count of TCP segment retransmissions during the test. The signal worth gating on: a link that reaches line rate while retransmitting heavily has a problem that throughput alone will not show, and it will surface later as tail latency in a collective",
		Aggregation:  AggSum,
		Combination:  CombineSum,
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
		Combination:  CombineSum,
		ThresholdUse: ThresholdUseAcceptance,
	},
	"nicLinkDownEvents": {
		Name: "nicLinkDownEvents", Unit: UnitNone,
		Description:  "count of link-down transitions on the node's fabric NICs during the test window",
		Aggregation:  AggSum,
		Combination:  CombineSum,
		ThresholdUse: ThresholdUseAcceptance,
	},
	"soakIterations": {
		Name: "soakIterations", Unit: UnitNone,
		Description:  "windows a fabric soak attempted. Context for soakFailedIterations: two failures in ten is a different link from two in a thousand",
		Aggregation:  AggSum,
		ThresholdUse: ThresholdUseEvidence,
	},
	"soakFailedIterations": {
		Name: "soakFailedIterations", Unit: UnitNone,
		Description:  "windows that failed or timed out during a fabric soak. A counter, so a healthy soak reports exactly zero and Equal 0 is safe from day one — unlike the bandwidth figures, which need a measured baseline first",
		Aggregation:  AggSum,
		Combination:  CombineSum,
		ThresholdUse: ThresholdUseAcceptance,
	},
	"soakServerRestarts": {
		Name: "soakServerRestarts", Unit: UnitNone,
		Description:  "how many times the server end restarted its listener during a soak. Evidence from the non-deciding end: perftest's server exits when its client disconnects, so this should track the client's iteration count and a large gap means windows never reached it",
		Aggregation:  AggSum,
		ThresholdUse: ThresholdUseEvidence,
	},
	"linkErrorEvents": {
		Name: "linkErrorEvents", Unit: UnitNone,
		Description:  "sum of the port error counters that MOVED during the soak — symbol errors, link recoveries, link-downs, receive errors. A DELTA, never a lifetime total: a NIC up for two hundred days carries a large count that says nothing about the last four hours. Unmeasurable (n/a) when no sysfs counter could be read, or when a counter went backwards because the port was reset mid-soak",
		Aggregation:  AggSum,
		Combination:  CombineSum,
		ThresholdUse: ThresholdUseAcceptance,
	},
	"throttleEvents": {
		Name: "throttleEvents", Unit: UnitNone,
		Description:  "count of clock-throttle events observed during the test",
		Aggregation:  AggSum,
		Combination:  CombineSum,
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

	// --- Multi-device iteration ---------------------------------------------
	//
	// A Node verdict describes EVERY accelerator on the node, gated on the
	// worst (docs/dev/multi-device.md). The gated metric keeps its existing
	// name — sustainedClockPct is the worst device's — and these entries are
	// what say how many devices were behind it, which one it was, and how far
	// the best was from the worst. Nothing here adds a device dimension to a
	// name: a per-device table is an artifact, never a suffix.
	"deviceCount": {
		Name: "deviceCount", Unit: UnitNone,
		Description:  "how many accelerator devices the runner MEASURED — iterated over and folded into the gated figures. An acceptance gate: a fleet writes deviceCount Equal 8 so a pod handed one card of eight FAILS instead of certifying that card as the node. Under MIG it counts instances, not parts. Distinct from acceleratorCount, which is what the node's PCI bus HAS, and from devicesVisible, which is what the runtime showed the pod; supersedes the unregistered gpuCount, which is not aliased to it",
		Aggregation:  AggLast,
		ThresholdUse: ThresholdUseAcceptance,
	},
	"devicesVisible": {
		Name: "devicesVisible", Unit: UnitNone,
		Description:  "how many accelerator devices the container runtime showed the pod. A multi-device runner prints it beside deviceCount, before applying its allocation budget from BURNIN_RESOURCE_LIMITS (a value above deviceCount is the leak: devices visible that were never allocated, which the runner reports as an Error rather than iterating past its budget). A runner that only SAW devices — a node-wide read-only probe such as host-health or dcgm-diag, or one that has not yet been converted to fold them — prints this and does not claim deviceCount at all. Evidence; what gpu_count became",
		Aggregation:  AggLast,
		ThresholdUse: ThresholdUseEvidence,
	},
	"deviceWindowS": {
		Name: "deviceWindowS", Unit: UnitSeconds,
		Description:  "the load window ONE device actually got. Under sequential iteration it is durationRequestedS divided by deviceCount, so a 15-second clock window on an eight-GPU node is distinguishable from a 120-second one on a single-GPU part; under concurrent iteration it equals durationRequestedS. Evidence, not a gate: how long each device was loaded is context for the figures beside it",
		Aggregation:  AggLast,
		ThresholdUse: ThresholdUseEvidence,
	},
	"worstDeviceIndex": {
		Name: "worstDeviceIndex", Unit: UnitNone,
		Description:  "the runtime index of the device behind the gated figure of THIS window — the one whose reading the fold kept. Under Last across a segmented soak, or across an error-retry that re-runs a window, it names the LAST window's worst device, not the device behind a multi-window aggregate; per-device.json is the only sound attribution across windows. No ArgMin aggregation exists to change that and none should be added. Evidence: an index has no ordering a threshold could use",
		Aggregation:  AggLast,
		ThresholdUse: ThresholdUseEvidence,
	},
	"worstDevicePciBusId": {
		Name: "worstDevicePciBusId", Unit: UnitNone,
		Description:  "the PCI bus address of the device worstDeviceIndex names, so a consumer reading only metrics can dispatch an engineer to a slot. Same window caveat as worstDeviceIndex. pciBusId beside it keeps its existing meaning — the first device's — so its wire meaning on a single-device fleet does not move",
		Aggregation:  AggLast,
		ThresholdUse: ThresholdUseEvidence,
	},
	"deviceHomogeneous": {
		Name: "deviceHomogeneous", Unit: UnitNone,
		Description:  "\"true\" when every measured device reported the same product name and compute capability, \"false\" when the runner READ every identity and they differ. A positive establishment: an unreadable identity on one device is an omission and declares nothing. When false, every spread over an absolute figure is n/a, because a mixed board's spread reads as a fault on healthy hardware. A label",
		Aggregation:  AggLast,
		ThresholdUse: ThresholdUseEvidence,
	},
	"deviceConcurrency": {
		Name: "deviceConcurrency", Unit: UnitNone,
		Description:  "the iteration mode the runner resolved: \"sequential\" (one device at a time, each given deviceWindowS) or \"all\" (every device at once for the full duration). Declared per kind by the runner and overridden by BURNIN_DEVICE_CONCURRENCY. A label; the number a consumer wants is deviceWindowS",
		Aggregation:  AggLast,
		ThresholdUse: ThresholdUseEvidence,
	},
	// The spreads: (max − min) / max × 100 across the devices of ONE window,
	// one name per measurand, formed by dropping the base metric's unit token
	// and keeping its identity. Max across windows because the worst window's
	// spread describes the board; a ceiling a fleet gates with LessThanOrEqual.
	// Deliberately the inverse of throughputConsistencyPct (a consistency is a
	// floor within one device across windows; a spread is a ceiling across
	// devices within one window). n/a on a single-device node, under MIG, and
	// on a heterogeneous board — a positive claim in each case — so a gate on
	// one must be RequiredIfMeasurable, and verdict.ValidateThresholdsForKind
	// reports a Required gate as Unsound. SpreadMetrics is the set.
	"sustainedClockSpreadPct": {
		Name: "sustainedClockSpreadPct", Unit: UnitPercent,
		Description:       "spread of sustainedClockPct across the node's devices in one window: (best − worst) / best × 100. One part holding 60% of rated clock beside seven holding 95% is a board with a fault that the worst-device floor may still clear. n/a on a single-device node, under MIG, and on a heterogeneous board — gate it RequiredIfMeasurable",
		Aggregation:       AggMax,
		ThresholdUse:      ThresholdUseAcceptance,
		MayBeUnmeasurable: true,
	},
	"gemmThroughputSpreadPct": {
		Name: "gemmThroughputSpreadPct", Unit: UnitPercent,
		Description:       "spread of achievedTflops across the node's devices in one window of the same GEMM at the same precision: (best − worst) / best × 100. n/a on a single-device node, under MIG, and on a heterogeneous board — gate it RequiredIfMeasurable",
		Aggregation:       AggMax,
		ThresholdUse:      ThresholdUseAcceptance,
		MayBeUnmeasurable: true,
	},
	"fmaThroughputSpreadPct": {
		Name: "fmaThroughputSpreadPct", Unit: UnitPercent,
		Description:       "spread of sustainedFmaThroughputTflops across the node's devices in one window of the same soak: (best − worst) / best × 100. Its own name rather than a shared throughput spread for the reason sustainedFmaThroughputTflops is not sustainedThroughputTflops. n/a on a single-device node, under MIG, and on a heterogeneous board — gate it RequiredIfMeasurable",
		Aggregation:       AggMax,
		ThresholdUse:      ThresholdUseAcceptance,
		MayBeUnmeasurable: true,
	},
	"hostToDeviceBandwidthSpreadPct": {
		Name: "hostToDeviceBandwidthSpreadPct", Unit: UnitPercent,
		Description:       "spread of hostToDeviceBandwidthGBs across the node's devices in one window: (best − worst) / best × 100. A device on a PCIe link that trained narrow reads low here while the worst-device floor may still clear. n/a on a single-device node, under MIG, and on a heterogeneous board — gate it RequiredIfMeasurable",
		Aggregation:       AggMax,
		ThresholdUse:      ThresholdUseAcceptance,
		MayBeUnmeasurable: true,
	},

	// --- Test-execution counters --------------------------------------------
	"diagTestsFailed": {
		Name: "diagTestsFailed", Unit: UnitNone,
		Description:  "count of individual diagnostic subtests that returned a failure; zero alongside a non-zero runner exit means the suite could not run rather than that the hardware failed",
		Aggregation:  AggSum,
		ThresholdUse: ThresholdUseAcceptance,
	},
	// The count DCGM's own suite reports as WARNED — an observation that did
	// not cross DCGM's failure line. Registered as Acceptance because the
	// decision belongs to the profile: the runner deliberately does not promote
	// a warning to a failure, so without a gateable name a warning passes the
	// node by default and nothing anywhere notices (#323). A site that treats
	// warnings as advisory simply does not gate it.
	"testsWarned": {
		Name: "testsWarned", Unit: UnitNone,
		Description:  "count of diagnostic subtests that reported a warning rather than a failure; the runner does not promote these to a failure, so gate this to act on them. diag_warn_findings carries what they said",
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
	"ncclTransport": {
		Name: "ncclTransport", Unit: UnitNone,
		Description: "which transport NCCL actually selected for an intra-node (Node-scope) collective: one of " +
			"nvlink|p2p|shm|net. BEST-EFFORT and self-omitting, not a probe result: it is read from " +
			"NCCL_DEBUG=INFO's own log text rather than from anything this runner measured directly, so an " +
			"NCCL upgrade that changes its wording makes this label simply absent from a future run rather than " +
			"wrong — it can never become a hardware verdict, because it is Evidence and nothing in this project " +
			"gates on it. It answers the question an intra-node run raises that a cross-node one never does: did " +
			"the collective cross NVLink, or silently fall back to something slower within the same box",
		Aggregation:  AggLast,
		ThresholdUse: ThresholdUseEvidence,
	},
	"pdWedgeSuspected": {
		Name: "pdWedgeSuspected", Unit: UnitNone,
		Description:  "whether the run matched the power-delivery wedge signature — slow, cool, and not thermally throttled: one of true|false|unknown. Deliberately TRI-STATE and therefore a label, not a boolean: without a temperature a wedge cannot be told from a thermal throttle, and \"we could not tell\" must not collapse into \"false\" and read as an all-clear. The runner fails the node on this itself; a threshold cannot express the three states and fails closed on all of them",
		Aggregation:  AggLast,
		ThresholdUse: ThresholdUseEvidence,
	},
	"idleClockLockSuspected": {
		Name: "idleClockLockSuspected", Unit: UnitNone,
		Description:  "whether the run matched the amdgpu idle-clock-lock signature — slow, cool, and busy (ROCm issue #5750): one of true|false|unknown. Tri-state for the same reason pdWedgeSuspected is: without a temperature the lock cannot be told from a thermal throttle, and without a utilization reading \"busy while slow\" cannot be established — neither unknown may read as an all-clear. The runner fails the node on the clock floor itself; a threshold cannot express the three states",
		Aggregation:  AggLast,
		ThresholdUse: ThresholdUseEvidence,
	},
	"gfxTarget": {
		Name: "gfxTarget", Unit: UnitNone,
		Description:  "the AMD accelerator's compute target as HIP reports it (\"gfx1151\"). A label with the same standing as gpuName: identity is a fleet-inventory question, never a threshold",
		Aggregation:  AggLast,
		ThresholdUse: ThresholdUseEvidence,
	},
	"clockFloorBasis": {
		Name: "clockFloorBasis", Unit: UnitNone,
		Description:  "which floor the run was judged against, general|thermal, naming the reason clockFloorAppliedPct is what it is; a label recording the runner's own decision",
		Aggregation:  AggLast,
		ThresholdUse: ThresholdUseEvidence,
	},

	// --- Duty cycle (power-swing) -------------------------------------------
	//
	// power-swing shares thermal-soak's and gpu-burn's engine but alternates
	// the load on a period instead of holding it, and reports what happened in
	// the seconds right after each ramp — a VRM or PSU that cannot keep up with
	// a fast load step shows up here, not in a sustained soak's own sampling.
	// Every metric below is EVIDENCE, not Acceptance: nobody has yet watched a
	// real power-delivery transient on this project's own fleet, so there is no
	// calibrated floor or ceiling to gate on. See
	// runners/power-swing/power_swing.cu's header comment.
	"swingTransitions": {
		Name: "swingTransitions", Unit: UnitNone,
		Description:  "count of OFF->ON duty-cycle transitions (ramps) completed during the test — context for the other swing_* figures: two throttle events in ten ramps is a different finding from two in a thousand. A WINDOWED count, like iterationsCompleted: a segmented soak's segments each contribute the transitions THEY saw, and the total is their sum — which is also why the device fold beneath it is Once rather than Sum, since every device on one node follows the identical wall-clock schedule and would otherwise be double-counted",
		Aggregation:  AggSum,
		ThresholdUse: ThresholdUseEvidence,
	},
	"swingWorstPostRampClockPct": {
		Name: "swingWorstPostRampClockPct", Unit: UnitPercent,
		Description:  "the LOWEST instantaneous SM clock, as a percentage of rated boost, seen inside ANY post-ramp window across the whole run — the figure a VRM that cannot slew fast enough shows up in. Omitted when no ramp sample was taken or the rated clock could not be read",
		Aggregation:  AggMin,
		ThresholdUse: ThresholdUseEvidence,
	},
	"swingPeakRampPowerW": {
		Name: "swingPeakRampPowerW", Unit: UnitWatts,
		Description:  "the HIGHEST instantaneous board power seen inside ANY post-ramp window across the whole run — the figure a PSU that sags rather than surges under a fast load step would instead show as a DIP; published alongside swingWorstPostRampClockPct so a reader can tell the two apart. Omitted when no ramp sample with a working power read was taken",
		Aggregation:  AggMax,
		ThresholdUse: ThresholdUseEvidence,
	},
	"swingNewThrottleEvents": {
		Name: "swingNewThrottleEvents", Unit: UnitNone,
		Description:  "count of post-ramp windows during which a throttle-reason bit appeared that was NOT already latched during the steady OFF phase immediately before it — a reason that shows up specifically at the moment of the ramp, rather than one already present beforehand. Omitted on a device whose throttle reasons could never be read at all, exactly as throttleEvents is",
		Aggregation:  AggSum,
		ThresholdUse: ThresholdUseEvidence,
	},
	// A LABEL, registered under the same duty as pdWedgeSuspected and
	// throttleReasons just above it: the comma-joined human-readable names of
	// the bits swingNewThrottleEvents counts, from the SAME kReasonBits table
	// throttleReasons already renders — reused, not reinvented. Unregistered it
	// would still parse the authoring-time lint (valid grammar, no unit
	// suffix), and then fail closed on every node forever the moment somebody
	// gated on it, reading as a hardware verdict on healthy hardware — the
	// exact trap the "first-party label metric" rule exists to prevent.
	"swingNewThrottleReasons": {
		Name: "swingNewThrottleReasons", Unit: UnitNone,
		Description:  "comma-separated names of the throttle reasons that appeared during a ramp and were not already latched beforehand (\"none\" when none did) — the human-readable form of what swingNewThrottleEvents counts. A list of labels; gate on swingNewThrottleEvents for \"did a new reason appear at all\", not that this project recommends gating on either — see the section header",
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
	"eccMode": {
		Name: "eccMode", Unit: UnitNone,
		Description:  "whether ECC is enabled on the device, as the driver reports it (\"Enabled\"/\"Disabled\"). A LABEL: a threshold is compared as a float64, so gating on it fails closed on every node forever and reads as a hardware verdict. It is what tells \"this part has no ECC\" from \"ECC was switched off\", and only the first is declarable as unmeasurable",
		Aggregation:  AggLast,
		ThresholdUse: ThresholdUseEvidence,
	},
	"eccSource": {
		Name: "eccSource", Unit: UnitNone,
		Description:  "where the ECC counters came from (\"amdgpu-sysfs\"). A LABEL, recorded because eccErrors means the same thing across vendors only if a reader can see which subsystem produced it",
		Aggregation:  AggLast,
		ThresholdUse: ThresholdUseEvidence,
	},
	"eccVendorConflict": {
		Name: "eccVendorConflict", Unit: UnitNone,
		Description:  "present only on a node where two vendors' accelerators both have ECC counters to report (\"nvidia+amd\"). One pod gets one image, so this operator does not support such a node; the first reading stands and this records that a second was declined rather than silently overwriting it",
		Aggregation:  AggLast,
		ThresholdUse: ThresholdUseEvidence,
	},
	"amdRasUnreadableFiles": {
		Name: "amdRasUnreadableFiles", Unit: UnitNone,
		Description:  "the amdgpu RAS counter files that could not be read, comma-joined. A LABEL, and the reason eccErrors is absent rather than zero: without it an omitted counter looks identical to one nobody implemented, and the fail-closed verdict has no reason attached",
		Aggregation:  AggLast,
		ThresholdUse: ThresholdUseEvidence,
	},
	"xidSource": {
		Name: "xidSource", Unit: UnitNone,
		Description:  "where the Xid scan read from: kmsg|kernlog|none. A label naming the PROVENANCE of xidEvents, and the value that matters most is the one a gate cannot express — \"none\" means the scan did not run, and the counters it would have produced are omitted rather than zeroed, so xidEvents already fails closed on its own",
		Aggregation:  AggLast,
		ThresholdUse: ThresholdUseEvidence,
	},
	// A LABEL — free prose naming the path that could not be opened and the
	// remedy — registered under the same duty as pdWedgeSuspected and friends:
	// a first-party metric whose value is words must be registered Evidence, so
	// a gate on it is reported Unsound instead of failing closed on every node
	// in the shape of a hardware verdict. Emitted by host-health and by all
	// four soak-family runners when xid_source=none.
	"xidSourceDetail": {
		Name: "xidSourceDetail", Unit: UnitNone,
		Description:  "why the Xid scan could not run: the path(s) tried and, where the cause is the three-part /dev/kmsg grant, the remedy. Free text for a human; never a number, never gateable",
		Aggregation:  AggLast,
		ThresholdUse: ThresholdUseEvidence,
	},
	// Registered Evidence even though it is an integer, because its emission
	// shape makes any gate on it unsatisfiable-or-vacuous: it is printed only
	// as 1 (a drain lost records — the counters it qualifies are then omitted)
	// and never as 0, so `xidLogDropped Equal 0` would fail every healthy node
	// forever. The linter can only say so if the registry knows the name.
	"xidLogDropped": {
		Name: "xidLogDropped", Unit: UnitNone,
		Description:  "1 when a kernel-log drain lost records (ring wrapped, scan failed, or a read bound was hit); absent otherwise. The counters it qualifies are omitted in the same case, so this is the trace of WHY they are missing — not a counter to gate",
		Aggregation:  AggLast,
		ThresholdUse: ThresholdUseEvidence,
	},
	// The coverage self-report for the soak family's window-scoped Xid watch:
	// 1 when a window was watched start to finish, 0 when it was not (no
	// grant, or a drain lost records). BOTH values are positively established
	// facts about the probe — like faultsInjected, this is a self-report and
	// not a hardware measurement, so emitting the 0 breaks no rule.
	//
	// It exists for the SEGMENTED case: foldMetrics keeps a metric if ANY
	// segment reported it, so a 672-segment soak whose segment 300 lost its
	// watch still sums the other segments' honest zeros into an xidEvents a
	// gate would accept — the one segmentation-changes-the-verdict shape the
	// soak rules forbid, reached through a probe rather than a measurement.
	// Sum-aggregated, this key then reads 671, and a profile pairing
	// `xidEvents Equal 0` with `xidWindowsWatched GreaterThanOrEqual <segment
	// count>` certifies only a soak every window of which was observed. On an
	// unsegmented test it is simply 1 or 0, and Equal 1 is the pairing gate.
	"xidWindowsWatched": {
		Name: "xidWindowsWatched", Unit: UnitNone,
		Description:  "how many of this test's execution windows the kernel-log Xid watch covered start to finish: 1 or 0 per window, summed across a segmented soak's segments. Pair a gate on xidEvents with a floor on this to certify only fully-watched soaks",
		Aggregation:  AggSum,
		Combination:  CombineSum,
		ThresholdUse: ThresholdUseAcceptance,
	},
	// host-health's count of Xids already in the kernel log when its window
	// OPENED — this boot's earlier history, as distinct from xidEvents' "during
	// the window". Registered now because host-health's own README has always
	// recommended gating it (`xidPreexisting Equal 0`), and writing a threshold
	// is the moment registration is owed.
	"xidPreexisting": {
		Name: "xidPreexisting", Unit: UnitNone,
		Description:  "count of NVIDIA Xid errors already in the kernel log when the scan's window opened — the node's earlier history this boot, not the test's own window. The windowed count is xidEvents",
		Aggregation:  AggLast,
		Combination:  CombineMax,
		ThresholdUse: ThresholdUseAcceptance,
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

// MayBeUnmeasurable reports whether name is registered with
// Metric.MayBeUnmeasurable — legitimately n/a, as a positive claim, on EVERY
// single-device node. An unregistered name answers false, the same
// fail-closed default SafeToThresholdOn uses: this package can only vouch for
// a trait it has actually classified.
//
// This is the property verdict.ValidateThresholdsForKind reads to advise
// RequiredIfMeasurable at authoring time (#405) — SpreadMetrics below is one
// NAMING-shaped subset of it (used by TestSpreadMetricsAreRegisteredCeilings
// to hold every spread to the same registered shape), not the whole trait:
// peerReadBandwidthGBs and peerWriteBandwidthGBs carry the same
// MayBeUnmeasurable=true without being spreads at all.
func MayBeUnmeasurable(name string) bool {
	m, ok := registry[name]
	return ok && m.MayBeUnmeasurable
}

// SpreadMetrics are the cross-device homogeneity spreads: (best − worst) /
// best × 100 across the devices of one window. Every one of them is n/a — a
// positive claim, "nothing to spread across" — on a single-device node, under
// MIG, and on a heterogeneous board, so a threshold on any of them must be
// RequiredIfMeasurable: the default Required fails closed on n/a and would fail
// every healthy single-GPU node forever, on the one phase that is never
// retried. verdict.ValidateThresholdsForKind reads this set to say so at
// authoring time. TestSpreadMetricsAreRegisteredCeilings holds the set to the
// registry: every member registered, Acceptance, Max, Pct.
var SpreadMetrics = []string{
	"sustainedClockSpreadPct",
	"gemmThroughputSpreadPct",
	"fmaThroughputSpreadPct",
	"hostToDeviceBandwidthSpreadPct",
}

// IsSpreadMetric reports whether name is one of SpreadMetrics.
func IsSpreadMetric(name string) bool {
	for _, s := range SpreadMetrics {
		if s == name {
			return true
		}
	}
	return false
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
