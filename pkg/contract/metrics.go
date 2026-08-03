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
		ThresholdUse: ThresholdUseAcceptance,
	},
	"peakBandwidthGbps": {
		Name: "peakBandwidthGbps", Unit: UnitGigabitsPerSecond,
		Description:  "best single-iteration RDMA write throughput on a link; the average (bandwidthGbps) is the figure a link is accepted on, since one good iteration does not survive a marginal cable",
		ThresholdUse: ThresholdUseAcceptance,
	},
	"busBandwidthGBs": {
		Name: "busBandwidthGBs", Unit: UnitGigabytesPerSecond,
		Description:  "NCCL bus bandwidth: algorithm bandwidth scaled by the collective's communication pattern",
		ThresholdUse: ThresholdUseAcceptance,
	},
	"algBandwidthGBs": {
		Name: "algBandwidthGBs", Unit: UnitGigabytesPerSecond,
		Description:  "NCCL algorithm bandwidth: message size over collective time, with no scaling for the communication pattern; unlike busBandwidthGBs it is not comparable across collectives or rank counts",
		ThresholdUse: ThresholdUseAcceptance,
	},

	// --- Memory copy and stress bandwidth ----------------------------------
	"hostToDeviceBandwidthGBs": {
		Name: "hostToDeviceBandwidthGBs", Unit: UnitGigabytesPerSecond,
		Description:  "measured copy bandwidth from host memory into device memory; bounded by the host link (PCIe, or C2C on a coherent part)",
		ThresholdUse: ThresholdUseAcceptance,
	},
	"deviceToHostBandwidthGBs": {
		Name: "deviceToHostBandwidthGBs", Unit: UnitGigabytesPerSecond,
		Description:  "measured copy bandwidth from device memory back to host memory; asymmetry with hostToDeviceBandwidthGBs is normal and is why the two directions are separate metrics",
		ThresholdUse: ThresholdUseAcceptance,
	},
	"deviceToDeviceBandwidthGBs": {
		Name: "deviceToDeviceBandwidthGBs", Unit: UnitGigabytesPerSecond,
		Description:  "on-device copy bandwidth, device memory to device memory; bounded by the memory subsystem rather than by the host link",
		ThresholdUse: ThresholdUseAcceptance,
	},
	"memoryBandwidthGBs": {
		Name: "memoryBandwidthGBs", Unit: UnitGigabytesPerSecond,
		Description:  "sustained device memory bandwidth achieved over the whole measurement window, not a peak sample",
		ThresholdUse: ThresholdUseAcceptance,
	},
	"readBandwidthMBs": {
		Name: "readBandwidthMBs", Unit: UnitMegabytesPerSecond,
		Description:  "sustained read throughput reported by the host memory stress tool",
		ThresholdUse: ThresholdUseAcceptance,
	},
	"writeBandwidthMBs": {
		Name: "writeBandwidthMBs", Unit: UnitMegabytesPerSecond,
		Description:  "sustained write throughput reported by the host memory stress tool",
		ThresholdUse: ThresholdUseAcceptance,
	},

	// --- Latency and duration ----------------------------------------------
	"latencyUs": {
		Name: "latencyUs", Unit: UnitMicroseconds,
		Description:  "round-trip latency",
		ThresholdUse: ThresholdUseAcceptance,
	},
	"ioLatencyUs": {
		Name: "ioLatencyUs", Unit: UnitMicroseconds,
		Description:  "mean per-operation latency of the test's read/write loop; distinct from latencyUs, which is a network round trip",
		ThresholdUse: ThresholdUseAcceptance,
	},
	"elapsedS": {
		Name: "elapsedS", Unit: UnitSeconds,
		Description:  "wall-clock seconds the test body ran, excluding image pull and container start; a soak that exits early has not proven what its duration claims",
		ThresholdUse: ThresholdUseAcceptance,
	},

	// --- Thermal, power and clocks -----------------------------------------
	"gpuTempC": {
		Name: "gpuTempC", Unit: UnitCelsius,
		Description:  "peak GPU temperature observed during the test",
		ThresholdUse: ThresholdUseAcceptance,
	},
	"powerDrawW": {
		Name: "powerDrawW", Unit: UnitWatts,
		Description:  "peak board power draw observed during the test",
		ThresholdUse: ThresholdUseAcceptance,
	},
	"sustainedClockPct": {
		Name: "sustainedClockPct", Unit: UnitPercent,
		Description:  "achieved SM clock as a percentage of the part's rated boost clock, averaged over the load window; the SKU-portable form of smClockMHz",
		ThresholdUse: ThresholdUseAcceptance,
	},
	"smClockMHz": {
		Name: "smClockMHz", Unit: UnitMegahertz,
		Description:  "SM core clock sustained under load; absolute and therefore SKU-specific, which is why a fleet-wide gate belongs on sustainedClockPct",
		ThresholdUse: ThresholdUseAcceptance,
	},
	"memClockMHz": {
		Name: "memClockMHz", Unit: UnitMegahertz,
		Description:  "device memory clock sustained under load",
		ThresholdUse: ThresholdUseAcceptance,
	},
	"ratedBoostClockMHz": {
		Name: "ratedBoostClockMHz", Unit: UnitMegahertz,
		Description:  "the part's nameplate boost clock, read back from the driver so sustainedClockPct can be audited against its denominator; a nameplate constant identifies the SKU rather than its health, so gating on it fails a heterogeneous fleet for no hardware reason",
		ThresholdUse: ThresholdUseEvidence,
	},

	// --- Compute throughput -------------------------------------------------
	"throughputTflops": {
		Name: "throughputTflops", Unit: UnitTeraflops,
		Description:  "achieved throughput; for compute-smoke this is a single unwarmed launch — a liveness signal, not a benchmark, and not safe to threshold on",
		ThresholdUse: ThresholdUseEvidence,
	},
	"sustainedThroughputTflops": {
		Name: "sustainedThroughputTflops", Unit: UnitTeraflops,
		Description:  "throughput averaged across a sustained burn, after the part has reached its steady thermal and clock state; unlike throughputTflops this is a benchmark figure and may be thresholded",
		ThresholdUse: ThresholdUseAcceptance,
	},

	// --- Correctness counters -----------------------------------------------
	"nonfiniteCount": {
		Name: "nonfiniteCount", Unit: UnitNone,
		Description:  "count of NaN or Inf values in a kernel's output",
		ThresholdUse: ThresholdUseAcceptance,
	},
	"miscompares": {
		Name: "miscompares", Unit: UnitNone,
		Description:  "count of computed values that did not match the reference — a wrong answer from hardware that reported success",
		ThresholdUse: ThresholdUseAcceptance,
	},
	"sdcDetections": {
		Name: "sdcDetections", Unit: UnitNone,
		Description:  "count of distinct silent-data-corruption incidents; miscompares counts every mismatched value, so one corrupted region can produce many miscompares and a single detection",
		ThresholdUse: ThresholdUseAcceptance,
	},

	// --- Hardware fault counters --------------------------------------------
	"eccErrors": {
		Name: "eccErrors", Unit: UnitNone,
		Description:  "count of ECC errors observed during the test",
		ThresholdUse: ThresholdUseAcceptance,
	},
	"memoryErrors": {
		Name: "memoryErrors", Unit: UnitNone,
		Description:  "count of hardware memory errors reported by the host memory stress tool (stressapptest hardware incidents); host memory, not device ECC",
		ThresholdUse: ThresholdUseAcceptance,
	},
	"xidEvents": {
		Name: "xidEvents", Unit: UnitNone,
		Description:  "count of NVIDIA Xid errors the driver logged during the test window",
		ThresholdUse: ThresholdUseAcceptance,
	},
	"remappedRows": {
		Name: "remappedRows", Unit: UnitNone,
		Description:  "count of device memory rows the driver has remapped; a rising count is a degrading part even while every test still passes",
		ThresholdUse: ThresholdUseAcceptance,
	},
	"pcieReplayErrors": {
		Name: "pcieReplayErrors", Unit: UnitNone,
		Description:  "count of PCIe replay events on the device's link — a link-integrity signal that usually precedes a bandwidth shortfall",
		ThresholdUse: ThresholdUseAcceptance,
	},
	"nicLinkDownEvents": {
		Name: "nicLinkDownEvents", Unit: UnitNone,
		Description:  "count of link-down transitions on the node's fabric NICs during the test window",
		ThresholdUse: ThresholdUseAcceptance,
	},
	"throttleEvents": {
		Name: "throttleEvents", Unit: UnitNone,
		Description:  "count of clock-throttle events observed during the test",
		ThresholdUse: ThresholdUseAcceptance,
	},

	// --- Test-execution counters --------------------------------------------
	"diagTestsFailed": {
		Name: "diagTestsFailed", Unit: UnitNone,
		Description:  "count of individual diagnostic subtests that returned a failure; zero alongside a non-zero runner exit means the suite could not run rather than that the hardware failed",
		ThresholdUse: ThresholdUseAcceptance,
	},
	"iterationsCompleted": {
		Name: "iterationsCompleted", Unit: UnitNone,
		Description:  "count of stress iterations the runner completed; a soak that completes far fewer than expected did less work than its duration suggests",
		ThresholdUse: ThresholdUseAcceptance,
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
