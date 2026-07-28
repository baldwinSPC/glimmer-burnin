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
	UnitMicroseconds,
	UnitMilliseconds,
	UnitPercent,
	UnitCelsius,
	UnitWatts,
	UnitSeconds,
}

// Metric is a registered, canonical metric.
type Metric struct {
	Name string
	Unit Unit
	// Description says what the number means, not merely what it is called.
	// Two bandwidth figures in different units are only safe to store side by
	// side if a reader can tell which one they are looking at.
	Description string
}

// registry holds the metrics the project itself emits. It is intentionally not
// a closed world: a runner may emit a metric that is not registered, and that
// is allowed so long as the name obeys the grammar. The registry exists to stop
// the names we DO own from drifting apart, not to forbid new measurements.
var registry = map[string]Metric{
	"bandwidthGbps": {
		Name: "bandwidthGbps", Unit: UnitGigabitsPerSecond,
		Description: "raw RDMA write throughput on a link, as reported by ib_write_bw",
	},
	"busBandwidthGBs": {
		Name: "busBandwidthGBs", Unit: UnitGigabytesPerSecond,
		Description: "NCCL bus bandwidth: algorithm bandwidth scaled by the collective's communication pattern",
	},
	"latencyUs": {
		Name: "latencyUs", Unit: UnitMicroseconds,
		Description: "round-trip latency",
	},
	"gpuTempC": {
		Name: "gpuTempC", Unit: UnitCelsius,
		Description: "peak GPU temperature observed during the test",
	},
	"powerDrawW": {
		Name: "powerDrawW", Unit: UnitWatts,
		Description: "peak board power draw observed during the test",
	},
	"throughputTflops": {
		Name: "throughputTflops", Unit: UnitTeraflops,
		Description: "achieved throughput; for compute-smoke this is a single unwarmed launch — a liveness signal, not a benchmark, and not safe to threshold on",
	},
	"eccErrors": {
		Name: "eccErrors", Unit: UnitNone,
		Description: "count of ECC errors observed during the test",
	},
	"throttleEvents": {
		Name: "throttleEvents", Unit: UnitNone,
		Description: "count of clock-throttle events observed during the test",
	},
	"nonfiniteCount": {
		Name: "nonfiniteCount", Unit: UnitNone,
		Description: "count of NaN or Inf values in a kernel's output",
	},
}

// Lookup returns the registered metric, if this is one the project owns.
func Lookup(name string) (Metric, bool) {
	m, ok := registry[name]
	return m, ok
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
