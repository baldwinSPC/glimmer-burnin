package main

import (
	"fmt"
	"io"
	"math"
	"strconv"
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
	// What was asked of dcgmi beyond the level: the tests named instead of a
	// level, the exact -p string, and the -t it was given. Evidence, so a stored
	// result answers "which plugins did this run enable?" on its own rather than
	// by re-deriving it from whatever the profile is believed to have said
	// (#307).
	keyDiagTests     = "diag_tests"
	keyDiagParams    = "diag_params"
	keyDiagTimeoutS  = "diag_timeout_s"
	keyDcgmVersion   = "dcgm_version"
	keyDriverVersion = "driver_version"
	keyGPUCount      = "device_count"
	keyTestsRun      = "tests_run"
	keyTestsFailed   = "tests_failed"
	keyTestsWarned   = "tests_warned"
	keyTestsNotRun   = "tests_not_run"
	keyTestsSkipped  = "tests_skipped"
	keyDcgmiExitCode = "dcgmi_exit_code"
	keySampleCount   = "sample_count"
	// #304: findings DCGM reported that are not verdicts about the part.
	keyConfigFindings   = "diag_config_findings"
	keyUnreadableFields = "diag_unreadable_fields"
	keyExcusedSubtests  = "diag_excused_subtests"
	// Which subtests DCGM declined to run, and why — evidence for the
	// partial-coverage verdict.
	keySkippedSubtests = "diag_skipped_subtests"
	// What the WARNED subtests said. testsWarned is a count, and a count with
	// no text leaves an operator who gated on it with nothing to act on (#323).
	keyWarnFindings  = "diag_warn_findings"
	keySkipReason    = "diag_skip_reason"
	keyCounterReset  = "counter_baseline_reset"
	keyPrunedObjects = "pruned_objects"
	keyElapsedS      = "elapsed_s"
	keyReason        = "reason"

	// Sampled DCGM field values. See fields.go for the field each one comes
	// from and how it is aggregated.
	keyGPUTempC        = "gpu_temp_c"
	keyLastXidCode     = "last_xid_code"
	keyPCIeReplayCount = "pcie_replay_count"
	keyECCSbeTotal     = "ecc_sbe_total"
	keyECCDbeTotal     = "ecc_dbe_total"
	keyRowsRemapped    = "rows_remapped"
)

// metricUnmeasurable is the project's reserved metric value, recognised
// case-insensitively by pkg/runner and routed to Result.Unmeasurable rather
// than Result.Metrics.
//
// It means "we looked, and this hardware cannot produce that measurement" — a
// positive declaration, never a substitute for a value we simply failed to
// collect. Emitting it where the probe itself broke would turn a crash into an
// acceptance under RequiredIfMeasurable, so the rule is to OMIT the key in that
// case and use this only where the absence was established.
const metricUnmeasurable = "n/a"

// emittedKeys is every key this runner can print. It exists so a test can
// enumerate them; nothing at runtime iterates it.
var emittedKeys = []string{
	keyDiagLevel, keyDiagLevelSource, keyDiagTests, keyDiagParams,
	keyDiagTimeoutS, keyDcgmVersion, keyDriverVersion,
	keyGPUCount, keyTestsRun, keyTestsFailed, keyTestsWarned, keyTestsNotRun,
	keyTestsSkipped, keyDcgmiExitCode, keySampleCount, keyCounterReset,
	keyPrunedObjects, keyElapsedS, keyReason, keyWarnFindings,
	keyGPUTempC, keyLastXidCode, keyPCIeReplayCount, keyECCSbeTotal,
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
