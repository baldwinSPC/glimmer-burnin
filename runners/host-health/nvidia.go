package main

import (
	"bytes"
	"context"
	"encoding/csv"
	"errors"
	"fmt"
	"os/exec"
	"strconv"
	"strings"
)

// NVML probe.
//
// The counters read here are the ones NVML persists: aggregate ECC totals,
// retired pages, the row remapper, and the PCIe replay counter. They are read
// through nvidia-smi rather than by linking NVML, for the same reason the
// compute-smoke runner does not ship a CUDA runtime: the image must contain no
// NVIDIA-licensed redistributable library. nvidia-smi and libnvidia-ml are
// injected into the container at runtime from the host driver by the NVIDIA
// Container Toolkit (NVIDIA_DRIVER_CAPABILITIES must include "utility"), so the
// published image stays free of them and the probe still works.
//
// A node with no NVIDIA driver simply reports nvml_status=absent and
// contributes no metrics. That is the vendor-neutral path, not an error: the
// remaining probes (kernel log, PCIe AER, NIC link state) are vendor-agnostic
// and still produce a verdict.

// nvidia-smi --query-gpu field names.
const (
	fIndex  = "index"
	fName   = "name"
	fDriver = "driver_version"
	// fECCMode is the discriminator between the two very different reasons an
	// ECC counter comes back N/A, and nothing else can tell them apart:
	//
	//   - "Enabled"/"Disabled": the part HAS an ECC subsystem. If its counters
	//     are then unreadable, that is our failure to read them (or ECC was
	//     switched off), and the counters must be OMITTED so the gate fails
	//     closed. Turning ECC off must never be a way to pass an ECC gate.
	//   - N/A: the part has NO ECC subsystem at all — GB10's unified LPDDR5X
	//     has on-die ECC that NVML cannot see. Only here may the counters be
	//     declared unmeasurable.
	fECCMode        = "ecc.mode.current"
	fECCUncAgg      = "ecc.errors.uncorrected.aggregate.total"
	fECCCorAgg      = "ecc.errors.corrected.aggregate.total"
	fECCUncVol      = "ecc.errors.uncorrected.volatile.total"
	fRetiredSBE     = "retired_pages.single_bit_ecc.count"
	fRetiredDBE     = "retired_pages.double_bit.count"
	fRetiredPending = "retired_pages.pending"
	fRemapCorr      = "remapped_rows.correctable"
	fRemapUnc       = "remapped_rows.uncorrectable"
	fRemapPending   = "remapped_rows.pending"
	fRemapFailure   = "remapped_rows.failure"
	fPCIeReplay     = "pcie.replay.counter"
	fTemp           = "temperature.gpu"
	fPower          = "power.draw"
)

// gpuFields is queried in one call, then field by field if that call fails.
//
// The fallback is what makes this work across driver generations without a
// version matrix: the row remapper does not exist before Ampere and retired
// pages are gone after it, so on any given part some of these names are simply
// invalid and nvidia-smi rejects the WHOLE query. Retrying per field turns
// "this driver does not know one name" into "we lost one metric" instead of
// "we lost the entire NVML probe".
var gpuFields = []string{
	fIndex, fName, fDriver,
	fECCMode, fECCUncAgg, fECCCorAgg, fECCUncVol,
	fRetiredSBE, fRetiredDBE, fRetiredPending,
	fRemapCorr, fRemapUnc, fRemapPending, fRemapFailure,
	fPCIeReplay, fTemp, fPower,
}

// errCommandNotFound distinguishes "there is no nvidia-smi on this node" (a
// normal, vendor-neutral outcome) from "nvidia-smi ran and failed" (which is
// worth reporting as a probe error).
var errCommandNotFound = errors.New("command not found")

type gpuSample struct {
	// status is one of: ok, absent, noDevices, timeout, error.
	status string
	tbl    *gpuTable
}

func (s gpuSample) ok() bool { return s.tbl != nil && s.tbl.rows > 0 }

// gpuTable is one nvidia-smi query result: a column per queried field, one row
// per GPU, in index order.
type gpuTable struct {
	rows int
	col  map[string][]string
}

func probeGPU(cfg config) gpuSample {
	if cfg.run == nil {
		return gpuSample{status: "absent"}
	}
	ctx, cancel := context.WithTimeout(context.Background(), cfg.probeTimeout)
	defer cancel()

	tbl, err := queryGPU(ctx, cfg, gpuFields)
	switch {
	case err == nil && tbl.rows == 0:
		return gpuSample{status: "noDevices"}
	case err == nil:
		return gpuSample{status: "ok", tbl: tbl}
	case errors.Is(err, errCommandNotFound):
		return gpuSample{status: "absent"}
	case ctx.Err() != nil:
		// An nvidia-smi that never returns is a real symptom (a GPU that has
		// fallen off the bus wedges the driver), but it is not a MEASUREMENT.
		// It is reported as a probe status and leaves the NVML counters
		// omitted, so any threshold on them fails closed.
		return gpuSample{status: "timeout"}
	default:
		return gpuSample{status: "error"}
	}
}

func queryGPU(ctx context.Context, cfg config, fields []string) (*gpuTable, error) {
	tbl, err := runQuery(ctx, cfg, fields)
	if err == nil {
		return tbl, nil
	}
	if errors.Is(err, errCommandNotFound) || ctx.Err() != nil {
		return nil, err
	}

	merged := &gpuTable{col: map[string][]string{}}
	firstErr := err
	for _, f := range fields {
		if f == fIndex {
			continue
		}
		one, qerr := runQuery(ctx, cfg, []string{fIndex, f})
		if qerr != nil {
			if errors.Is(qerr, errCommandNotFound) || ctx.Err() != nil {
				return nil, qerr
			}
			continue
		}
		if merged.rows == 0 {
			merged.rows = one.rows
			merged.col[fIndex] = one.col[fIndex]
		}
		if one.rows != merged.rows {
			continue // inconsistent view of the node; drop this field rather than misalign rows
		}
		merged.col[f] = one.col[f]
	}
	if len(merged.col) <= 1 {
		return nil, firstErr
	}
	return merged, nil
}

func runQuery(ctx context.Context, cfg config, fields []string) (*gpuTable, error) {
	args := []string{
		"--query-gpu=" + strings.Join(fields, ","),
		"--format=csv,noheader,nounits",
	}
	stdout, err := cfg.run(ctx, cfg.nvidiaSMI, args)
	if err != nil {
		return nil, err
	}
	return parseSMICSV(stdout, fields)
}

func parseSMICSV(stdout string, fields []string) (*gpuTable, error) {
	r := csv.NewReader(strings.NewReader(strings.TrimSpace(stdout)))
	r.TrimLeadingSpace = true
	r.FieldsPerRecord = -1
	recs, err := r.ReadAll()
	if err != nil {
		return nil, fmt.Errorf("parsing nvidia-smi csv: %w", err)
	}
	tbl := &gpuTable{col: map[string][]string{}}
	for _, rec := range recs {
		if len(rec) != len(fields) {
			return nil, fmt.Errorf("nvidia-smi returned %d values for %d fields", len(rec), len(fields))
		}
		for i, f := range fields {
			tbl.col[f] = append(tbl.col[f], strings.TrimSpace(rec[i]))
		}
		tbl.rows++
	}
	return tbl, nil
}

// counterState is what the driver said about a counter, which is three
// different things and not two.
type counterState int

const (
	// counterUnknown: we could not look. The column was never returned, or only
	// some of the GPUs answered. The counter is OMITTED, so any threshold on it
	// fails closed — that is rule 1 of this package.
	counterUnknown counterState = iota
	// counterOK: every GPU answered with a number.
	counterOK
	// counterUnmeasurable: every GPU answered "N/A". The driver was asked and
	// said this part has no such counter, which is a POSITIVE fact about the
	// hardware rather than a gap in our reading. It is emitted as the "n/a"
	// sentinel so a RequiredIfMeasurable threshold can be reported as not
	// evaluated instead of failing a healthy node.
	//
	// This is GB10's case: its unified LPDDR5X has on-die ECC only, so
	// ecc.errors.* and remapped_rows.* return [N/A] on a perfectly healthy part.
	counterUnmeasurable
)

// counter totals a counter across every GPU on the node and says how well it
// was read.
//
// It reports counterOK only when EVERY GPU produced a usable value. A
// node-level total assembled from a subset of its GPUs is not a smaller
// measurement, it is a wrong one: the GPU that did not answer is exactly the one
// most likely to be broken.
//
// The three-way split is deliberate, and the line between the last two states is
// the whole point. "Every GPU said N/A" is the hardware declaring it has no such
// counter. Anything else — a missing column, an unparseable value, or a MIX of
// numbers and N/A across the node's GPUs — is our failure to read it, and must
// not be dressed up as the hardware's declaration: a node where one GPU reports
// ECC and another does not is not a node without ECC.
func (t *gpuTable) counter(field string) (int64, counterState) {
	if t == nil {
		return 0, counterUnknown
	}
	vals, ok := t.col[field]
	if !ok || t.rows == 0 || len(vals) != t.rows {
		return 0, counterUnknown
	}
	var (
		total   int64
		unavail int
	)
	for _, v := range vals {
		n, ok := parseCount(v)
		if !ok {
			if _, isUnavailable := unavailable(v); !isUnavailable {
				return 0, counterUnknown // garbage: not a reading and not a declaration
			}
			unavail++
			continue
		}
		total += n
	}
	switch {
	case unavail == len(vals):
		return 0, counterUnmeasurable
	case unavail > 0:
		return 0, counterUnknown
	}
	return total, counterOK
}

// sum is counter() for the callers that only care whether a real number came
// back. An unmeasurable counter is not a number, so it reports false here.
func (t *gpuTable) sum(field string) (int64, bool) {
	v, state := t.counter(field)
	return v, state == counterOK
}

// eccSupport is whether the part has an ECC subsystem at all — which is a
// different question from whether ECC is on, and a different question again from
// whether its counters could be read.
type eccSupport int

const (
	// eccSupportUnknown: ecc.mode.current itself could not be read.
	eccSupportUnknown eccSupport = iota
	// eccSupportPresent: at least one GPU reports an ECC mode, so this node has
	// ECC hardware. Any ECC counter that then comes back N/A is a gap in our
	// reading — or ECC deliberately switched off — and is omitted, never
	// declared unmeasurable.
	eccSupportPresent
	// eccSupportAbsent: EVERY GPU reports N/A for the mode itself. There is no
	// ECC subsystem here to have counters, and only in this state may the
	// counters be declared unmeasurable.
	eccSupportAbsent
)

// eccMode reports the node's ECC mode as evidence, and what it implies about
// ECC support. The mode string is "enabled", "disabled", "unsupported", or
// "mixed" where the node's GPUs disagree.
func (t *gpuTable) eccMode() (string, eccSupport) {
	if t == nil {
		return "", eccSupportUnknown
	}
	vals, ok := t.col[fECCMode]
	if !ok || t.rows == 0 || len(vals) != t.rows {
		return "", eccSupportUnknown
	}
	mode := ""
	allAbsent := true
	for i, v := range vals {
		s, isUnavailable := unavailable(v)
		if isUnavailable {
			s = "unsupported"
		} else {
			allAbsent = false
		}
		switch {
		case i == 0:
			mode = s
		case s != mode:
			// One GPU with ECC is enough to make an N/A counter suspicious.
			mode = "mixed"
		}
	}
	if allAbsent {
		return mode, eccSupportAbsent
	}
	return mode, eccSupportPresent
}

// asDeclaration decides whether "every GPU answered N/A" may stand as the
// runner's declaration that this hardware cannot measure the counter.
//
// It may only where the part has said it has no ECC subsystem at all. On a part
// that HAS ECC, an N/A counter means ECC is switched off or the read failed —
// both of which leave the node unproven, so the counter is omitted and the
// threshold fails closed. This is the difference between "a GB10 has no ECC to
// count" and "someone turned ECC off on this H100", and collapsing the two
// would hand every fleet a way to pass an ECC gate by disabling ECC.
func asDeclaration(state counterState, support eccSupport) counterState {
	if state == counterUnmeasurable && support != eccSupportAbsent {
		return counterUnknown
	}
	return state
}

// bothOrNeither folds the two halves of one counter into a single state. The
// halves must agree: a total assembled from one measured and one unmeasurable
// column would understate the damage while looking like a complete answer, and
// claiming a part cannot measure row remapping because one of the two columns
// went missing would relax a gate on no evidence.
func bothOrNeither(a, b counterState) counterState {
	if a == b {
		return a
	}
	return counterUnknown
}

// max reports the largest value across the node's GPUs — the peak, which is
// what gpuTempC and powerDrawW are defined as. Unlike sum, a missing value is
// skipped rather than fatal: these are evidence, and the hottest GPU that DID
// answer is still the best available answer.
func (t *gpuTable) max(field string) (float64, bool) {
	if t == nil {
		return 0, false
	}
	var (
		best  float64
		found bool
	)
	for _, v := range t.col[field] {
		f, ok := parseFloat(v)
		if !ok {
			continue
		}
		if !found || f > best {
			best, found = f, true
		}
	}
	return best, found
}

// first returns GPU 0's value for a field.
func (t *gpuTable) first(field string) (string, bool) {
	if t == nil {
		return "", false
	}
	vals := t.col[field]
	if len(vals) == 0 {
		return "", false
	}
	v := vals[0]
	if _, unusable := unavailable(v); unusable {
		return "", false
	}
	return v, true
}

// unavailable recognises the several ways nvidia-smi says "no value".
//
// "N/A" for an ECC field means ECC is disabled or unsupported on this part —
// and that must NOT become eccErrors=0. A GPU with ECC turned off has not
// proven it is error-free; it has proven nothing, and a burn-in gate on
// eccErrors should fail it rather than wave it through.
func unavailable(v string) (string, bool) {
	s := strings.ToLower(strings.TrimSpace(v))
	s = strings.Trim(s, "[]")
	switch {
	case s == "",
		s == "n/a",
		s == "na",
		strings.Contains(s, "not supported"),
		strings.Contains(s, "insufficient permissions"),
		strings.Contains(s, "unknown error"):
		return s, true
	}
	return s, false
}

// parseCount reads a counter value. Yes/No is accepted because nvidia-smi
// reports the "pending" flags that way, and a flag nothing can threshold on is
// a flag nobody will use.
func parseCount(v string) (int64, bool) {
	s, bad := unavailable(v)
	if bad {
		return 0, false
	}
	switch s {
	case "yes", "true", "enabled":
		return 1, true
	case "no", "false", "disabled":
		return 0, true
	}
	n, err := strconv.ParseInt(s, 10, 64)
	if err != nil {
		// Some builds report large counters with thousands separators.
		if f, err2 := strconv.ParseFloat(strings.ReplaceAll(s, ",", ""), 64); err2 == nil {
			return int64(f), true
		}
		return 0, false
	}
	return n, true
}

func parseFloat(v string) (float64, bool) {
	s, bad := unavailable(v)
	if bad {
		return 0, false
	}
	f, err := strconv.ParseFloat(s, 64)
	if err != nil {
		return 0, false
	}
	return f, true
}

// emitGPU writes the NVML-derived metrics.
//
// Which sample a metric comes from is a deliberate choice per counter:
//
//   - ECC and row-remap totals are DAMAGE. They are cumulative properties of the
//     part, not events in this run's window, so the LIFETIME value is reported
//     and gated. A node that took an uncorrected ECC error last week is not a
//     node that passes acceptance today.
//   - the PCIe replay counter is an EVENT counter, so pcie_replay_count is the
//     window delta and the lifetime figure is kept alongside as evidence.
//   - temperature and power are instantaneous, so the peak across the samples is
//     reported.
func emitGPU(out *emitter, before, after gpuSample) {
	out.set("nvml_status", after.status)
	if !after.ok() {
		if !before.ok() {
			return
		}
		// The first sample worked and the second did not: still report what was
		// measured, from the sample that exists.
		after = before
	}
	t0, t1 := before.tbl, after.tbl

	if name, ok := t1.first(fName); ok {
		out.set("gpu_name", name)
	}
	if drv, ok := t1.first(fDriver); ok {
		out.set("driver_version", drv)
	}
	out.setInt("devices_visible", int64(t1.rows))

	// --- ECC (lifetime; gated) ---------------------------------------------
	// Uncorrected only. Correctable ECC events are expected on healthy HBM and
	// gating on them would fail good parts; they are reported separately so a
	// rising correctable rate is still visible.
	//
	// A part with no ECC subsystem at all (GB10: on-die ECC in the LPDDR5X,
	// invisible to NVML) reports the sentinel rather than a zero it did not
	// measure or a silence nobody can explain. A part that HAS ECC and cannot
	// produce the counter — including one with ECC switched off — still omits
	// it and still fails closed.
	mode, eccHW := t1.eccMode()
	if mode != "" {
		out.set("ecc_mode", mode)
	}
	uncAgg, eccState := t1.counter(fECCUncAgg)
	uncOK := eccState == counterOK
	out.setCounter(keyECCErrors, uncAgg, asDeclaration(eccState, eccHW))
	if v, ok := t1.sum(fECCCorAgg); ok {
		out.setInt("ecc_corrected_aggregate", v)
	}
	if v, ok := t1.sum(fECCUncVol); ok {
		out.setInt("ecc_uncorrected_volatile", v)
	}
	if before0, ok0 := t0.sum(fECCUncAgg); ok0 {
		// An uncorrected ECC error that happened WHILE we watched is a
		// different and worse thing than an old one. Reported separately so
		// the two are never confused.
		if d, ok := delta(before0, uncAgg, ok0, uncOK); ok {
			out.setInt("ecc_uncorrected_window", d)
		}
	}

	// --- row remapper (lifetime; gated) -------------------------------------
	corr, corrState := t1.counter(fRemapCorr)
	unc, uncRowState := t1.counter(fRemapUnc)
	corrOK, uncRowOK := corrState == counterOK, uncRowState == counterOK
	// Both halves or neither: a total assembled from one of the two columns
	// would understate the damage while looking like a complete answer. The row
	// remapper is ECC's repair mechanism, so a part with no ECC subsystem has
	// none to report and says so; anywhere else an unreadable column is omitted.
	out.setCounter(keyRowsRemapped, corr+unc, asDeclaration(bothOrNeither(corrState, uncRowState), eccHW))
	out.setIntIf("remapped_rows_correctable", corr, corrOK)
	out.setIntIf("remapped_rows_uncorrectable", unc, uncRowOK)
	if v, ok := t1.sum(fRemapPending); ok {
		out.setInt("remapped_rows_pending", v)
	}
	if v, ok := t1.sum(fRemapFailure); ok {
		out.setInt("remapped_rows_failure", v)
	}

	// --- retired pages (pre-Ampere equivalent of the row remapper) ----------
	sbe, sbeOK := t1.sum(fRetiredSBE)
	dbe, dbeOK := t1.sum(fRetiredDBE)
	out.setIntIf("retired_pages_sbe", sbe, sbeOK)
	out.setIntIf("retired_pages_dbe", dbe, dbeOK)
	out.setIntIf("retired_pages_total", sbe+dbe, sbeOK && dbeOK)
	if v, ok := t1.sum(fRetiredPending); ok {
		out.setInt("retired_pages_pending", v)
	}

	// --- PCIe replay (window; gated by reconcilePCIeReplay) -----------------
	if v, ok := t1.sum(fPCIeReplay); ok {
		out.setInt("pcie_replay_total", v)
	}

	// --- thermal/power evidence ---------------------------------------------
	temp0, ok0 := t0.max(fTemp)
	temp1, ok1 := t1.max(fTemp)
	if peak, ok := peakOf(temp0, ok0, temp1, ok1); ok {
		out.set("gpu_temp_c", strconv.FormatFloat(peak, 'f', -1, 64))
	}
	pwr0, pok0 := t0.max(fPower)
	pwr1, pok1 := t1.max(fPower)
	if peak, ok := peakOf(pwr0, pok0, pwr1, pok1); ok {
		out.set("power_draw_w", strconv.FormatFloat(peak, 'f', -1, 64))
	}
}

func peakOf(a float64, aOK bool, b float64, bOK bool) (float64, bool) {
	switch {
	case aOK && bOK:
		if a > b {
			return a, true
		}
		return b, true
	case aOK:
		return a, true
	case bOK:
		return b, true
	}
	return 0, false
}

// execRunner runs an external probe.
func execRunner(ctx context.Context, name string, args []string) (string, error) {
	path, err := exec.LookPath(name)
	if err != nil {
		return "", errCommandNotFound
	}
	cmd := exec.CommandContext(ctx, path, args...)
	var stdout, stderr bytes.Buffer
	cmd.Stdout = &stdout
	cmd.Stderr = &stderr
	if err := cmd.Run(); err != nil {
		return stdout.String(), fmt.Errorf("%s: %w: %s", name, err, strings.TrimSpace(stderr.String()))
	}
	return stdout.String(), nil
}
