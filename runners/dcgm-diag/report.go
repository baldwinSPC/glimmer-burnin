package main

import (
	"fmt"
	"io"
	"math"
	"strconv"
	"strings"
	"unicode/utf8"
)

// The keys this runner prints. They are the RUNNER's spelling, not the
// project's canonical metric names: pkg/runner normalises snake_case to
// lowerCamelCase and applies the "dcgm-diag" alias table on top. Keeping them
// as constants in one place is what lets TestEmittedKeysSurviveTheParser assert
// the whole chain — that every key this runner can print lands on a name
// pkg/contract accepts, and that the four the project owns land on exactly the
// registered names.
const (
	keyDiagLevel       = "diag_level"
	keyDiagLevelSource = "diag_level_source"
	keyDcgmVersion     = "dcgm_version"
	keyDriverVersion   = "driver_version"
	keyGPUCount        = "gpu_count"
	keyTestsRun        = "tests_run"
	keyTestsFailed     = "tests_failed"
	keyTestsWarned     = "tests_warned"
	keyTestsNotRun     = "tests_not_run"
	keyTestsSkipped    = "tests_skipped"
	keyDcgmiExitCode   = "dcgmi_exit_code"
	keySampleCount     = "sample_count"
	keyCounterReset    = "counter_baseline_reset"
	keyPrunedObjects   = "pruned_objects"
	keyElapsedS        = "elapsed_s"
	keyReason          = "reason"

	// Sampled DCGM field values. See fields.go for the field each one comes
	// from and how it is aggregated.
	keyGPUTempC        = "gpu_temp_c"
	keyXIDErrors       = "xid_errors"
	keyPCIeReplayCount = "pcie_replay_count"
	keyECCSbeTotal     = "ecc_sbe_total"
	keyECCDbeTotal     = "ecc_dbe_total"
	keyRowsRemapped    = "rows_remapped"
)

// emittedKeys is every key this runner can print. It exists so a test can
// enumerate them; nothing at runtime iterates it.
var emittedKeys = []string{
	keyDiagLevel, keyDiagLevelSource, keyDcgmVersion, keyDriverVersion,
	keyGPUCount, keyTestsRun, keyTestsFailed, keyTestsWarned, keyTestsNotRun,
	keyTestsSkipped, keyDcgmiExitCode, keySampleCount, keyCounterReset,
	keyPrunedObjects, keyElapsedS, keyReason,
	keyGPUTempC, keyXIDErrors, keyPCIeReplayCount, keyECCSbeTotal,
	keyECCDbeTotal, keyRowsRemapped,
}

// report is the ordered set of key=value metrics the runner prints.
//
// Insertion order, not map order: a node that just failed acceptance is read by
// a human before it is read by a parser, and a stable layout is what makes two
// runs diffable. The parser is last-occurrence-wins, so re-setting a key keeps
// its original position and takes the new value — which is what we want when a
// later, better measurement supersedes an earlier estimate.
type report struct {
	order []string
	vals  map[string]string
}

func newReport() *report {
	return &report{vals: map[string]string{}}
}

func (r *report) set(key, value string) {
	if _, seen := r.vals[key]; !seen {
		r.order = append(r.order, key)
	}
	r.vals[key] = sanitize(value)
}

func (r *report) setInt(key string, v int64) {
	r.set(key, strconv.FormatInt(v, 10))
}

// setNumber prints an integral float without a spurious ".0" so a counter reads
// as a counter, and a fractional one at full precision.
func (r *report) setNumber(key string, v float64) {
	r.set(key, formatNumber(v))
}

func (r *report) writeTo(w io.Writer) {
	for _, k := range r.order {
		fmt.Fprintf(w, "%s=%s\n", k, r.vals[k])
	}
}

func formatNumber(v float64) string {
	if !math.IsInf(v, 0) && !math.IsNaN(v) && v == math.Trunc(v) && math.Abs(v) < 1e15 {
		return strconv.FormatInt(int64(v), 10)
	}
	return strconv.FormatFloat(v, 'f', -1, 64)
}

// sanitize keeps a value on one line. A metric value that contained a newline
// would be read by the parser as two lines, and the second one — a fragment of
// somebody's error string — would become the run's Message.
func sanitize(s string) string {
	s = strings.Map(func(r rune) rune {
		switch r {
		case '\n', '\r', '\t':
			return ' '
		}
		return r
	}, s)
	s = strings.Join(strings.Fields(s), " ")
	const max = 300
	if len(s) > max {
		// Cut on a rune boundary. DCGM's own prose is ASCII, but the reasons
		// this runner builds are not — they join with an em dash — and a byte
		// slice landing mid-sequence would put a replacement character into a
		// metric value, i.e. corrupt the very field a human reads to find out
		// why a node failed.
		cut := max
		for cut > 0 && !utf8.RuneStart(s[cut]) {
			cut--
		}
		s = s[:cut] + "…"
	}
	return s
}

// finish prints every metric gathered so far, then the marker line, then hands
// back the exit code.
//
// Metrics come FIRST on every path, including the failing and erroring ones. A
// run that ends without evidence tells whoever reads the log nothing about the
// node, and the paths where that matters most are exactly the ones where it is
// tempting to bail out early.
//
// The marker cannot be mistaken for a metric even when the reason contains an
// "=": the parser splits on the first "=" and rejects a key containing
// whitespace, and every reason is preceded by "MARKER: ".
func finish(w io.Writer, rep *report, code int, marker, reason string) int {
	rep.writeTo(w)
	if reason == "" {
		fmt.Fprintln(w, marker)
	} else {
		fmt.Fprintf(w, "%s: %s\n", marker, sanitize(reason))
	}
	return code
}
