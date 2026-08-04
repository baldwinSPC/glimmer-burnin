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
// # The skip declaration
//
// Exit 2 alone does not make a Skip. A runner must also DECLARE the skip on
// stdout, with a marker line matching SkipMarkerPattern — FP4_GEMM_SKIP,
// NCCL_SKIP, IB_WRITE_BW_SKIP, and so on. An exit 2 with no such line is an
// Error, and the hardware is left unjudged.
//
// This is the same rule as the sentinel above, applied to the exit code: a
// runner may only declare what it positively established, and "this test does
// not apply to this hardware" is a claim ABOUT the hardware. It exists because
// exit 2 is not a code a runner has sole control of. An unrecovered Go panic
// exits 2, and so does every Go runtime fatal error — out of memory, concurrent
// map write, stack exhaustion — none of which a deferred recover can intercept.
// Without this rule a crashed runner is recorded as a deliberate skip, which is
// the worst of the phases to land on: Skip is never retried, so the retry
// budget goes unspent, and it counts towards neither pass nor fail, so the run
// still settles Passed. Hardware silently unjudged, with a report that reads
// clean.
//
// The camouflage is what makes it dangerous rather than merely wrong. Zero
// metrics is the NORMAL shape of a skip, so a runner that died before printing
// anything is byte-for-byte indistinguishable from one that looked and found
// the test inapplicable — see Result.ReportedNothing, which cannot help here
// precisely because the empty harvest is legitimate on this path.
//
// The rule fails safe in the one direction that matters. A third-party runner
// that exits 2 without declaring is recorded Error: unjudged and retried, never
// an acceptance. Every runner this project ships already prints its marker.
//
// Like pkg/verdict, this is public and free of Kubernetes types. Glimmer's
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

// exitCodeSkip is the runner contract's skip code. It is named because two
// places have to agree about it: the table below, and the declaration rule in
// Parse that decides whether a runner was entitled to use it.
const exitCodeSkip = 2

// VerdictFor maps an exit code to its meaning.
//
// This is the exit-code table ALONE, and on its own it over-reports Skip: exit
// 2 is also what the Go runtime uses for an unrecovered panic and for every
// runtime fatal error. Parse applies the skip-declaration rule on top, and a
// dispatcher must go through Parse rather than call this directly — see the
// package comment, and SkipDeclared.
func VerdictFor(exitCode int) Verdict {
	switch exitCode {
	case 0:
		return VerdictPass
	case 1:
		return VerdictFail
	case exitCodeSkip:
		return VerdictSkip
	default:
		return VerdictError
	}
}

// SkipMarkerPattern matches a runner's skip DECLARATION: an upper-case token
// ending in _SKIP, at the start of a line, optionally followed by ": reason".
//
// The shape is what every runner in this project already prints —
// FP4_GEMM_SKIP, CLOCKPROBE_SKIP, MEMORY_BW_SKIP, DCGM_DIAG_SKIP, NCCL_SKIP,
// IB_WRITE_BW_SKIP, GPUDIRECT_RDMA_SKIP, STRESSAPPTEST_SKIP — so the rule
// codifies existing practice rather than imposing a new spelling. It is
// anchored so that prose mentioning a marker cannot declare a skip on a
// runner's behalf.
var SkipMarkerPattern = regexp.MustCompile(`^[A-Z][A-Z0-9_]*_SKIP\b`)

// SkipDeclared reports whether a runner's stdout carries the marker line that
// entitles it to exit 2.
//
// It is public and lives here for the reason Result.ReportedNothing does: both
// dispatchers parse with this package, and a predicate this load-bearing must
// not be re-derived. Two callers disagreeing about whether a runner declared a
// skip is two callers disagreeing about whether a node was tested at all.
func SkipDeclared(stdout string) bool {
	for _, line := range strings.Split(stdout, "\n") {
		if SkipMarkerPattern.MatchString(strings.TrimSpace(line)) {
			return true
		}
	}
	return false
}

// UndeclaredSkipMessage is what an exit 2 with no declaration is recorded as.
// It names the mechanism, because the single most likely cause is a crash the
// runner had no chance to report.
const UndeclaredSkipMessage = "runner exited 2 (skip) without declaring a skip on stdout — a skip is a claim about the hardware and must be declared with a *_SKIP marker line; " +
	"an undeclared exit 2 is most often a crashed runner (an unrecovered Go panic and every Go runtime fatal error exit 2), so the hardware is unjudged, not inapplicable"

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
		"tests_failed":      "diagTestsFailed",
		"xid_errors":        "xidEvents",
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
		"h2d_bandwidth_gbs":    "hostToDeviceBandwidthGBs",
		"d2h_bandwidth_gbs":    "deviceToHostBandwidthGBs",
		"d2d_bandwidth_gbs":    "deviceToDeviceBandwidthGBs",
		"memory_bandwidth_gbs": "memoryBandwidthGBs",
	},

	"host-health": {
		"nic_link_down":     "nicLinkDownEvents",
		"pcie_replay_count": "pcieReplayErrors",
		"xid_count":         "xidEvents",
		"rows_remapped":     "remappedRows",
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
//
// The exit code is not taken at face value for exit 2: a Skip must be declared
// on stdout, and an undeclared one is downgraded to Error. See the package
// comment for why, and SkipDeclared for the rule.
func Parse(kind, stdout string, exitCode int) Result {
	res := Result{
		Verdict:      VerdictFor(exitCode),
		ExitCode:     exitCode,
		Metrics:      map[string]string{},
		Unmeasurable: map[string]bool{},
	}
	kindAliases := aliases[kind]
	skipDeclared := false

	for _, line := range strings.Split(stdout, "\n") {
		line = strings.TrimSpace(line)
		if line == "" {
			continue
		}
		// Tested on the whole line, before the key=value split: a marker's
		// reason text may itself contain an "=", which would otherwise send
		// the line down the metric path and lose the declaration.
		if SkipMarkerPattern.MatchString(line) {
			skipDeclared = true
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

	// An exit 2 nobody declared. Error, never Skip: Skip is not retried and
	// counts towards neither pass nor fail, so a crash landing there is
	// hardware silently unjudged under a report that reads clean.
	//
	// Whatever the runner did manage to say is kept as context. For the case
	// this rule exists for that is the tail of a panic trace — a stack frame,
	// not a diagnosis — which is exactly why the explanation has to lead.
	if exitCode == exitCodeSkip && !skipDeclared {
		res.Verdict = VerdictError
		res.Message = UndeclaredSkipMessage
		if last := lastNonMetricLine(stdout); last != "" {
			res.Message += " [runner said: " + last + "]"
		}
	}
	return res
}

// lastNonMetricLine is the Message the scan above would have settled on. It is
// recomputed rather than captured because the undeclared-skip path overwrites
// Message and still owes the caller whatever the runner printed.
func lastNonMetricLine(stdout string) string {
	last := ""
	for _, line := range strings.Split(stdout, "\n") {
		line = strings.TrimSpace(line)
		if line == "" {
			continue
		}
		if eq := strings.Index(line, "="); eq > 0 {
			key := strings.TrimSpace(line[:eq])
			if key != "" && !strings.ContainsAny(key, " \t") {
				continue
			}
		}
		last = line
	}
	return last
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
