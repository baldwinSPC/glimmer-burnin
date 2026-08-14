// Package runner parses what a burn-in runner container reports.
//
// A runner's contract is deliberately small: an exit code, and key=value pairs
// on stdout. That is all a runner has to implement in any language, which is
// what keeps the vendor seam at the image boundary rather than in the
// reconciler.
//
// # The unmeasurable sentinel
//
// One value is reserved: "n/a", case-insensitively. A runner emits
// "eccErrors=n/a" to DECLARE that it looked and this hardware cannot produce
// that measurement at all — NVIDIA GB10 exposes no ECC to NVML, so there is no
// counter to read, and reporting 0 would be a lie about a quantity nobody
// measured. Such a key lands in Result.Unmeasurable, never in Result.Metrics,
// so Numeric() still cleanly reports "not a number" for it and no threshold can
// accidentally compare against it.
//
// The declaration is the whole point, and OMISSION IS NOT ONE. A metric that
// simply never appears fails its threshold closed, exactly as before; only an
// explicit "n/a" can relax a RequiredIfMeasurable gate (see pkg/verdict and
// api/v1alpha1.Applicability). A runner that cannot even look — a probe that
// timed out, a driver that did not answer — must omit the key rather than
// declare it unmeasurable, because "we could not look" and "there is nothing
// here to look at" are different claims about the hardware.
//
// This is public and free of Kubernetes types — genuinely, unlike pkg/verdict,
// which depends on api/v1alpha1 (issue #274). Glimmer's
// pre-Kubernetes burn-in path runs the SAME runner images, and if the two
// dispatchers derived different metrics from identical output they would reach
// different verdicts about the same hardware. One brain, two dispatchers.
package runner

import (
	"regexp"
	"strconv"
	"strings"
	"unicode"

	"github.com/baldwinSPC/glimmer-burnin/pkg/contract"
)

// Verdict is what a runner's exit code means.
type Verdict string

const (
	// VerdictPass is exit 0: the test ran and the hardware passed.
	VerdictPass Verdict = "Pass"
	// VerdictFail is exit 1: the test ran and the hardware failed.
	VerdictFail Verdict = "Fail"
	// VerdictSkip is exit 2: the test does not apply to this hardware.
	//
	// Skip must stay distinct from Fail. A node that cannot run a test has not
	// failed it, and collapsing the two is the false-negative that made healthy
	// Sparks look broken.
	VerdictSkip Verdict = "Skip"
	// VerdictError is any other exit code: the runner itself malfunctioned, so
	// the hardware is unjudged. Distinct from Fail, which is a real verdict.
	VerdictError Verdict = "Error"
)

// VerdictFor maps an exit code to its meaning, FROM THE EXIT CODE ALONE.
//
// Exit 2 is the one code this is not sufficient for, and callers deciding a real
// verdict should use Parse instead. An unrecovered Go panic — and every Go
// runtime fatal error, including out-of-memory, concurrent map writes and stack
// exhaustion — terminates the process with status 2. A crashed runner therefore
// arrives here indistinguishable from one that deliberately skipped, and Skip is
// the worst possible place for that to land: it is never retried, it does not
// affect the run's verdict, and it reports the node as one the test did not
// apply to. See DeclaresSkip.
func VerdictFor(exitCode int) Verdict {
	switch exitCode {
	case 0:
		return VerdictPass
	case 1:
		return VerdictFail
	case 2:
		return VerdictSkip
	default:
		return VerdictError
	}
}

// skipMarker matches a runner's positive declaration that it skipped: an
// upper-case token ending in _SKIP at the start of a line. Every runner in this
// repository that can legitimately skip already prints one — FP4_GEMM_SKIP,
// NCCL_SKIP, IB_WRITE_BW_SKIP, GPUDIRECT_RDMA_SKIP, STRESSAPPTEST_SKIP,
// DCGM_DIAG_SKIP, MEMORY_BW_SKIP, CLOCKPROBE_SKIP — so requiring it codifies
// what they already do rather than asking anything new of them.
var skipMarker = regexp.MustCompile(`(?m)^[A-Z][A-Z0-9_]*_SKIP\b`)

// DeclaresSkip reports whether the runner's output positively declares a skip.
//
// This exists because exit 2 by itself cannot be trusted. A skip is a CLAIM
// ABOUT THE HARDWARE — "acceptance does not apply to this node" — and this
// project's rule is that a runner may only declare what it positively
// established. The same rule already governs the `n/a` unmeasurable sentinel,
// where absence is likewise not a declaration.
//
// The camouflage this defeats is close to perfect otherwise. A panicking Go
// runner exits 2 and, because a container log merges stdout and stderr, its
// output is a stack trace with no key=value line at all — byte-for-byte the
// shape of a legitimate skip, whose normal form is exactly "no metrics". The
// stored Message becomes a stack frame like "/src/main.go:132 +0x1d", which
// reads as nothing in particular.
func DeclaresSkip(stdout string) bool { return skipMarker.MatchString(stdout) }

// Unmeasurable is the reserved metric VALUE with which a runner declares that
// this hardware cannot produce a measurement at all. Matched
// case-insensitively, after trimming.
const Unmeasurable = "n/a"

// Result is one runner execution, parsed.
type Result struct {
	Verdict  Verdict
	ExitCode int
	// Metrics are keyed by CANONICAL name (see pkg/contract), not by whatever
	// the runner printed.
	Metrics map[string]string
	// Unmeasurable is the set of canonical names the runner explicitly declared
	// unmeasurable on this hardware (the "n/a" sentinel). It is kept SEPARATE
	// from Metrics, and the two are disjoint: an unmeasurable name has no
	// value, so storing one would invent the measurement the sentinel exists to
	// deny. Only a name in here can relax a RequiredIfMeasurable threshold; a
	// name in neither map fails closed.
	Unmeasurable map[string]bool
	// Message is the last line that was not a key=value pair — usually a
	// marker or an error string. Kept because a failing runner's most useful
	// output is rarely a metric.
	Message string
	// InvalidNames lists canonical names that failed contract validation. They
	// are reported rather than silently dropped: a runner emitting an unusable
	// name should be visible, not invisible.
	InvalidNames []string
	// Artifacts are the non-metric evidence the runner fenced on stdout — a
	// dcgmi JSON document, a captured topology. Never part of a verdict; see
	// artifact.go.
	Artifacts []Artifact
	// UndeclaredSkip records that the runner exited 2 without declaring a skip,
	// so Verdict is Error rather than Skip. Kept as a field rather than left
	// implicit because the two readings send an engineer to opposite places: a
	// real skip means "this node is out of scope", and this means "the runner
	// died and the node is unjudged". A consumer reporting the verdict should
	// say which it is looking at.
	UndeclaredSkip bool
}

// IsUnmeasurable reports whether the runner declared this metric unmeasurable
// on the hardware it ran against.
func (r Result) IsUnmeasurable(name string) bool { return r.Unmeasurable[name] }

// ReportedNothing reports whether the parse found NO key=value line at all: no
// metric, no "n/a" declaration, not even a name the contract rejected. Half of
// the runner contract came back — the exit code — and the other half is simply
// absent.
//
// This is deliberately NOT "the metric a threshold names is missing", and the
// difference is the whole point. A runner that printed forty numbers and
// omitted the forty-first looked at this hardware; that omission is a real
// missing measurement and pkg/verdict fails it closed, as it must. A runner
// that printed nothing measured nothing — it was killed, evicted, or died
// before its first line — and there is no hardware verdict to be drawn from it
// at any exit code. A dispatcher must not let fail-closed evaluation
// manufacture one out of the absence.
//
// It lives here rather than in a dispatcher because both dispatchers parse with
// this package, and a predicate this load-bearing must not be re-derived — two
// callers disagreeing about whether a runner reported anything is two callers
// disagreeing about whether a node is broken.
func (r Result) ReportedNothing() bool {
	return len(r.Metrics) == 0 && len(r.Unmeasurable) == 0 && len(r.InvalidNames) == 0
}

// aliases maps a runner's own key to the canonical metric name, for the cases
// generic normalisation cannot reach. Keyed by TestKind.
//
// Generic snake_case → lowerCamelCase handles nearly everything
// ("nonfinite_count" → "nonfiniteCount", "elapsed_ms" → "elapsedMs"). An entry
// is only needed where the runner's key is not merely a different spelling of
// the canonical name. Three situations need one:
//
//  1. The key names no measurand. "tflops" is a bare unit; nothing can be
//     derived from it, so it can only be mapped.
//  2. The key's unit differs in CASE from the registered suffix. "gbs" folds to
//     "Gbs", which is not the registered "GBs", so "h2d_bandwidth_gbs" would
//     normalise to a name UnitOf() reads as dimensionless — a bandwidth
//     recorded as a bare number. This is the quiet one, and it is why nearly
//     every bandwidth runner needs an entry.
//  3. The tool's own vocabulary differs from ours. gpu_burn's "errors" are
//     miscompares; stressapptest's "hardware_incidents" are memory errors.
//
// Every value here is a name this project mints, so every value must be
// registered in pkg/contract — enforced by TestAliasTargetsAreRegistered.
// Within one kind, two keys must never map to the same canonical name: parsing
// is last-occurrence-wins, so a collision would silently discard one of two
// real measurements.
var aliases = map[string]map[string]string{
	"compute-smoke": {
		"tflops": "throughputTflops",
	},

	// Measured against a real GB10 capture (runners/gemm-sweep/testdata,
	// issue #265): the runner emitted five keys nothing had registered, so a
	// threshold on any of them could never be evaluated. Two of the five are
	// simply a different spelling of a name that already exists and belong
	// here; the other three are genuinely new quantities and were registered in
	// pkg/contract instead.
	"gemm-sweep": {
		// Seconds, and the registry spells that suffix "S" — `windowSeconds`
		// would normalise to a name whose unit the grammar cannot see, so it
		// would be treated as dimensionless and silently unregistered.
		"window_seconds": "sampleWindowS",
		// The same count gpu-burn reports under the same canonical name; a
		// second name for one quantity is a quantity that eventually disagrees
		// with itself.
		"gemm_iterations": "iterationsCompleted",
	},

	"gpu-burn": {
		// gpu_burn calls a mismatched result an "error"; ours is the more
		// specific word, and keeping it distinct from eccErrors matters — a
		// miscompare with no ECC event is the silent-corruption case.
		"errors":     "miscompares",
		"iterations": "iterationsCompleted",
		// Sustained across the burn, so NOT throughputTflops: that name is
		// reserved for the unwarmed single launch and is marked unsafe to
		// threshold on. Mapping here would make a real benchmark ungateable.
		"tflops": "sustainedThroughputTflops",
	},

	"dcgm-diag": {
		"tests_failed": "diagTestsFailed",
		// No tests_warned entry: generic normalisation already produces
		// testsWarned, and TestAliasEntriesAreNecessary rejects an alias that is
		// merely a respelling. What #323 was actually missing was the REGISTRY
		// entry — the name reached the envelope all along, but nothing declared
		// it, so ValidateThresholds warned anyone who gated on it and no
		// consumer could discover it existed.
		// No xid_errors entry, deliberately. This runner used to publish a
		// subtraction of two DCGM_FI_DEV_XID_ERRORS readings under xidEvents,
		// but that field holds the last Xid CODE rather than a count, so the
		// subtraction was not a number of anything (#311). It now emits
		// last_xid_code, which normalises to lastXidCode without an alias
		// because that is merely a different spelling. xidEvents keeps one
		// derivation, in host-health, which counts events from /dev/kmsg.
		"pcie_replay_count": "pcieReplayErrors",
		"rows_remapped":     "remappedRows",
		// DCGM reports correctable (SBE) and uncorrectable (DBE) ECC counts
		// separately and they mean different things: a handful of SBEs is a
		// working part, one DBE is a failing one. Neither is "eccErrors", and
		// aliasing both to it would collide and silently drop one. They are
		// left to generic normalisation as eccSbeTotal / eccDbeTotal — valid,
		// unregistered, and honest about being two numbers.
	},

	"memory-bw": {
		"h2d_bandwidth_gbs": "hostToDeviceBandwidthGBs",
		"d2h_bandwidth_gbs": "deviceToHostBandwidthGBs",
		"d2d_bandwidth_gbs": "deviceToDeviceBandwidthGBs",
		// The peer matrix. Aliased for the same reason as the three above: the
		// runner spells its unit "gbs", and normalisation would produce "Gbs"
		// — gigaBITS — for a figure that is gigaBYTES.
		"peer_read_bandwidth_gbs":  "peerReadBandwidthGBs",
		"peer_write_bandwidth_gbs": "peerWriteBandwidthGBs",
		"memory_bandwidth_gbs":     "memoryBandwidthGBs",
	},

	"host-health": {
		// Fabric error counters. The runner's keys are already lowerCamelCase
		// after normalisation, so only the ones whose spelling would land wrong
		// are aliased here; the rest fall through to snakeToLowerCamel.
		"ib_counters_saturated": "fabricCountersSaturated",
		"ib_saturated_counters": "fabricSaturatedCounters",
		"ib_counter_status":     "fabricCounterStatus",
		"nic_link_down":         "nicLinkDownEvents",
		"pcie_replay_count":     "pcieReplayErrors",
		"xid_count":             "xidEvents",
		"rows_remapped":         "remappedRows",
	},

	"thermal-soak": {
		// The registered temperature and power metrics are already defined as
		// the peak observed during the test, so the runner's "peak_" keys are
		// the same measurand under a different spelling.
		"peak_temp_c":    "gpuTempC",
		"peak_power_w":   "powerDrawW",
		"throttle_count": "throttleEvents",
		// "soak_seconds" would normalise to soakSeconds, whose lowercase "s"
		// is not the Seconds suffix — a duration stored as dimensionless.
		"soak_seconds": "elapsedS",
	},

	"nccl": {
		// nccl-tests' own column names. Both are GB/s there; the unit is
		// implicit in the tool's output and explicit in ours.
		"algbw":       "algBandwidthGBs",
		"busbw":       "busBandwidthGBs",
		"wrong_count": "miscompares",
	},

	"ib-write-bw": {
		// perftest's BW average / BW peak columns. perftest is dual-licensed
		// GPL/BSD and is consumed here under the BSD option (see NOTICE).
		"bw_average": "bandwidthGbps",
		"bw_peak":    "peakBandwidthGbps",
		// The latency tail. "latency_us" stays the MEAN, unchanged, so every
		// existing profile keeps meaning what it meant; the percentile and the
		// extremes are new names rather than a redefinition of that one.
		"latency_min_us": "minLatencyUs",
		"latency_max_us": "maxLatencyUs",
		"latency_p99_us": "p99LatencyUs",
		"msg_rate_mpps":  "messageRateMpps",
	},

	"gpudirect-rdma": {
		// ib_write_bw --use_cuda: the same link measurand as ib-write-bw, so
		// the same canonical name. A separate name would imply a separate
		// quantity and split the fleet's history in two.
		"bw_average": "bandwidthGbps",
		"t_avg_usec": "latencyUs",
	},

	"clockprobe": {
		// "mhz" folds to "Mhz", which is not the registered "MHz" suffix, so
		// every clock would otherwise land as a dimensionless number.
		"sm_clock_mhz":          "smClockMHz",
		"mem_clock_mhz":         "memClockMHz",
		"rated_boost_clock_mhz": "ratedBoostClockMHz",
	},

	"memory-stress": {
		// stressapptest (Apache-2.0) is the sanctioned memory stressor; the
		// GPL tools in this space cannot be shipped here.
		"hardware_incidents":  "memoryErrors",
		"sdc_count":           "sdcDetections",
		"read_bandwidth_mbs":  "readBandwidthMBs",
		"write_bandwidth_mbs": "writeBandwidthMBs",
	},
}

// Parse turns a runner's stdout and exit code into a Result.
//
// kind selects the alias table. An unknown or custom kind still gets the
// generic key=value scan — that scan IS the runner contract, and a custom
// runner honouring it should not be punished — but gets no kind-specific
// mapping, since only the kind's author knows what its keys mean.
func Parse(kind, stdout string, exitCode int) Result {
	res := Result{
		Verdict:      VerdictFor(exitCode),
		ExitCode:     exitCode,
		Metrics:      map[string]string{},
		Unmeasurable: map[string]bool{},
	}
	// A skip must be DECLARED, not merely exited. Exit 2 with nothing claiming a
	// skip is a runner that died on the way — most often a Go panic or runtime
	// fatal error, both of which exit 2 — and that is an Error: the hardware is
	// unjudged, the retry budget is available, and the run does not settle
	// Passed around a node nobody measured.
	//
	// It fails towards Error rather than towards Skip on purpose. Error is the
	// recoverable mistake — a runner that skips honestly but forgets the marker
	// is reported as unjudged and retried, which is visible and cheap. The
	// opposite mistake certifies a fleet nobody looked at.
	if exitCode == 2 && !DeclaresSkip(stdout) {
		res.Verdict = VerdictError
		res.UndeclaredSkip = true
	}
	kindAliases := aliases[kind]

	// Fenced artifacts are lifted out FIRST, and their payload lines never reach
	// the scanner below. A dcgmi JSON document is full of `"key": value` lines
	// and several would parse as metrics — so without this, a runner returning
	// evidence would silently rewrite its own measurements.
	artifacts, lines := extractArtifacts(stdout)
	res.Artifacts = artifacts

	for _, line := range lines {
		line = strings.TrimSpace(line)
		if line == "" {
			continue
		}
		// Split on the FIRST "=" so a value may itself contain one.
		eq := strings.Index(line, "=")
		if eq <= 0 {
			res.Message = line
			continue
		}
		rawKey := strings.TrimSpace(line[:eq])
		value := strings.TrimSpace(line[eq+1:])
		if rawKey == "" || strings.ContainsAny(rawKey, " \t") {
			// "some prose = with an equals" is not a metric.
			res.Message = line
			continue
		}

		name := canonicalName(rawKey, kindAliases)
		if err := contract.ValidateMetricName(name); err != nil {
			res.InvalidNames = append(res.InvalidNames, name)
			continue
		}
		// Last occurrence wins: runners may report progressively, and the final
		// value is the settled one. That applies ACROSS the two maps, which is
		// what keeps them disjoint — a runner that declares a metric
		// unmeasurable and later reports a real number has measured it, and the
		// reverse retracts the number.
		if isUnmeasurable(value) {
			res.Unmeasurable[name] = true
			delete(res.Metrics, name)
			continue
		}
		delete(res.Unmeasurable, name)
		res.Metrics[name] = value
	}
	return res
}

// isUnmeasurable recognises the reserved value. Only this exact spelling counts:
// "unknown", "none" or an empty value are ordinary values a runner might mean
// something else by, and a gate must not relax on a guess.
func isUnmeasurable(value string) bool {
	return strings.EqualFold(strings.TrimSpace(value), Unmeasurable)
}

// canonicalName maps a runner's key to its canonical form.
func canonicalName(rawKey string, kindAliases map[string]string) string {
	if canonical, ok := kindAliases[rawKey]; ok {
		return canonical
	}
	return snakeToLowerCamel(rawKey)
}

// snakeToLowerCamel converts "max_rel_error" to "maxRelError". A key that is
// already lowerCamelCase passes through unchanged, so a runner may emit either
// spelling.
func snakeToLowerCamel(s string) string {
	if !strings.ContainsAny(s, "_-") {
		return s
	}
	parts := strings.FieldsFunc(s, func(r rune) bool { return r == '_' || r == '-' })
	if len(parts) == 0 {
		return s
	}
	var b strings.Builder
	b.WriteString(strings.ToLower(parts[0]))
	for _, p := range parts[1:] {
		if p == "" {
			continue
		}
		r := []rune(strings.ToLower(p))
		r[0] = unicode.ToUpper(r[0])
		b.WriteString(string(r))
	}
	return b.String()
}

// Numeric reports a metric's value as a float, which is what a threshold is
// evaluated against. Non-numeric metrics (a GPU's name, say) are legitimate
// evidence but are not comparable. A metric declared unmeasurable is not in
// Metrics at all, so it reports false here rather than a fabricated zero.
func (r Result) Numeric(name string) (float64, bool) {
	raw, ok := r.Metrics[name]
	if !ok {
		return 0, false
	}
	f, err := strconv.ParseFloat(raw, 64)
	if err != nil {
		return 0, false
	}
	return f, true
}
