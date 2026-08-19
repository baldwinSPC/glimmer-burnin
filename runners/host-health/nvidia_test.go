package main

import (
	"context"
	"errors"
	"fmt"
	"strings"
	"testing"
	"time"
)

// healthyGPU is one GPU with every queried field present and clean.
func healthyGPU() map[string]string {
	return map[string]string{
		fIndex:          "0",
		fName:           "NVIDIA GB10",
		fDriver:         "580.82.09",
		fECCMode:        "Enabled",
		fECCUncAgg:      "0",
		fECCCorAgg:      "0",
		fECCUncVol:      "0",
		fRetiredSBE:     "0",
		fRetiredDBE:     "0",
		fRetiredPending: "No",
		fRemapCorr:      "0",
		fRemapUnc:       "0",
		fRemapPending:   "No",
		fRemapFailure:   "No",
		fPCIeReplay:     "0",
		fTemp:           "41",
		fPower:          "38.20",
	}
}

// sparkGPU is a real NVIDIA GB10 / DGX Spark, as nvidia-smi describes one.
//
// It is the hardware this operator is built for and the reason the unmeasurable
// sentinel exists: the unified LPDDR5X has on-die ECC only, so NVML sees no ECC
// subsystem at all and answers [N/A] to the mode, every ECC counter and every
// row-remap counter — on a perfectly healthy part. Gating on eccErrors here used
// to fail every node in the fleet.
func sparkGPU() map[string]string {
	g := healthyGPU()
	for _, f := range []string{
		fECCMode, fECCUncAgg, fECCCorAgg, fECCUncVol,
		fRemapCorr, fRemapUnc, fRemapPending, fRemapFailure,
		fRetiredSBE, fRetiredDBE, fRetiredPending,
	} {
		g[f] = "[N/A]"
	}
	return g
}

// fakeSMI answers --query-gpu the way nvidia-smi does: one CSV row per GPU, in
// the order the fields were requested. A field missing from the map is rejected
// the way the real tool rejects a name its driver does not know, which is what
// exercises the per-field fallback.
func fakeSMI(gpus ...map[string]string) commandRunner {
	return func(_ context.Context, _ string, args []string) (string, error) {
		fields, err := requestedFields(args)
		if err != nil {
			return "", err
		}
		var rows []string
		for _, g := range gpus {
			var row []string
			for _, f := range fields {
				v, ok := g[f]
				if !ok {
					return "", fmt.Errorf("Field %q is not a valid field to query", f)
				}
				row = append(row, v)
			}
			rows = append(rows, strings.Join(row, ", "))
		}
		return strings.Join(rows, "\n") + "\n", nil
	}
}

func requestedFields(args []string) ([]string, error) {
	for _, a := range args {
		if q, ok := strings.CutPrefix(a, "--query-gpu="); ok {
			return strings.Split(q, ","), nil
		}
	}
	return nil, errors.New("no --query-gpu argument")
}

func testConfig(run commandRunner) config {
	return config{nvidiaSMI: "nvidia-smi", probeTimeout: probeTimeoutSeconds * time.Second, run: run}
}

func TestParseSMICSV(t *testing.T) {
	fields := []string{fIndex, fName, fECCUncAgg}
	tbl, err := parseSMICSV("0, NVIDIA GB10, 0\n1, NVIDIA GB10, 3\n", fields)
	if err != nil {
		t.Fatalf("parseSMICSV: %v", err)
	}
	if tbl.rows != 2 {
		t.Fatalf("rows = %d, want 2", tbl.rows)
	}
	if got, ok := tbl.first(fName); !ok || got != "NVIDIA GB10" {
		t.Errorf("first(name) = %q, %v", got, ok)
	}
	if got, ok := tbl.sum(fECCUncAgg); !ok || got != 3 {
		t.Errorf("sum(ecc) = %d, %v; want 3, true", got, ok)
	}
}

func TestParseSMICSVRejectsShortRow(t *testing.T) {
	if _, err := parseSMICSV("0, NVIDIA GB10\n", []string{fIndex, fName, fECCUncAgg}); err == nil {
		t.Error("a row with fewer values than fields must be an error, not a silent misalignment")
	}
}

func TestParseCount(t *testing.T) {
	cases := []struct {
		in   string
		want int64
		ok   bool
	}{
		{"0", 0, true},
		{"17", 17, true},
		{"Yes", 1, true},
		{"No", 0, true},
		{"1,024", 1024, true},
		{"N/A", 0, false},
		{"[N/A]", 0, false},
		{"[Not Supported]", 0, false},
		{"[Insufficient Permissions]", 0, false},
		{"", 0, false},
		{"garbage", 0, false},
	}
	for _, tc := range cases {
		got, ok := parseCount(tc.in)
		if got != tc.want || ok != tc.ok {
			t.Errorf("parseCount(%q) = %d, %v; want %d, %v", tc.in, got, ok, tc.want, tc.ok)
		}
	}
}

// A node-level total assembled from a subset of its GPUs is not a smaller
// measurement, it is a wrong one.
func TestSumRequiresEveryGPU(t *testing.T) {
	tbl := &gpuTable{rows: 2, col: map[string][]string{fECCUncAgg: {"0", "N/A"}}}
	if _, ok := tbl.sum(fECCUncAgg); ok {
		t.Error("sum must not report ok when one GPU did not answer")
	}
}

func TestMaxSkipsMissingValues(t *testing.T) {
	tbl := &gpuTable{rows: 3, col: map[string][]string{fTemp: {"41", "N/A", "55"}}}
	got, ok := tbl.max(fTemp)
	if !ok || got != 55 {
		t.Errorf("max = %v, %v; want 55, true", got, ok)
	}
}

func TestProbeGPUAbsentWithoutNvidiaSMI(t *testing.T) {
	cfg := testConfig(func(context.Context, string, []string) (string, error) {
		return "", errCommandNotFound
	})
	s := probeGPU(cfg)
	if s.status != "absent" {
		t.Errorf("status = %q, want absent", s.status)
	}
	out := newEmitter()
	emitGPU(out, s, s)
	if _, ok := out.get(keyECCErrors); ok {
		t.Error("ecc_errors must be omitted on a node with no NVIDIA driver")
	}
	if got, _ := out.get("nvml_status"); got != "absent" {
		t.Errorf("nvml_status = %q, want absent", got)
	}
}

func TestProbeGPUNoDevices(t *testing.T) {
	cfg := testConfig(fakeSMI())
	if s := probeGPU(cfg); s.status != "noDevices" {
		t.Errorf("status = %q, want noDevices", s.status)
	}
}

func TestEmitGPUHealthyNode(t *testing.T) {
	cfg := testConfig(fakeSMI(healthyGPU()))
	s := probeGPU(cfg)
	if s.status != "ok" {
		t.Fatalf("status = %q, want ok", s.status)
	}

	out := newEmitter()
	emitGPU(out, s, s)
	want := map[string]string{
		"device_count":      "1",
		"gpu_name":          "NVIDIA GB10",
		"driver_version":    "580.82.09",
		keyECCErrors:        "0",
		keyRowsRemapped:     "0",
		"retired_pages_dbe": "0",
		"pcie_replay_total": "0",
		"gpu_temp_c":        "41",
		"power_draw_w":      "38.2",
	}
	for k, v := range want {
		if got, ok := out.get(k); !ok || got != v {
			t.Errorf("%s = %q (present=%v), want %q", k, got, ok, v)
		}
	}
}

// A GPU with ECC SWITCHED OFF reports N/A for its counters — and it must not
// get the unmeasurable sentinel, because the part plainly has ECC: it says so in
// ecc.mode.current. The counter is omitted, so the threshold fails closed, and
// disabling ECC stays a way to fail acceptance rather than a way to skip it.
func TestEmitGPUOmitsECCWhenSwitchedOff(t *testing.T) {
	g := healthyGPU()
	g[fECCMode] = "Disabled"
	g[fECCUncAgg] = "N/A"
	g[fECCCorAgg] = "N/A"

	s := probeGPU(testConfig(fakeSMI(g)))
	out := newEmitter()
	emitGPU(out, s, s)
	if got, ok := out.get(keyECCErrors); ok {
		t.Errorf("ecc_errors = %q; a part with ECC turned off must omit the counter, not declare it unmeasurable", got)
	}
	if got, _ := out.get("ecc_mode"); got != "disabled" {
		t.Errorf("ecc_mode = %q, want disabled — the evidence for omitting the counter", got)
	}
	// The rest of the NVML probe still reports.
	if _, ok := out.get(keyRowsRemapped); !ok {
		t.Error("rows_remapped must still be reported when only ECC is unavailable")
	}
}

// A part whose ECC counters are N/A while ECC is ENABLED did not answer us. That
// is a gap in our reading, not a property of the hardware, and it must not be
// dressed up as a declaration.
func TestEmitGPUOmitsUnreadableECCOnAnECCCapablePart(t *testing.T) {
	g := healthyGPU()
	g[fECCUncAgg] = "Insufficient Permissions"

	s := probeGPU(testConfig(fakeSMI(g)))
	out := newEmitter()
	emitGPU(out, s, s)
	if got, ok := out.get(keyECCErrors); ok {
		t.Errorf("ecc_errors = %q, want the key omitted so the gate fails closed", got)
	}
}

// The GB10 case, end to end through the NVML probe. The part has no ECC
// subsystem at all — ecc.mode.current is itself N/A — so the counters are
// DECLARED unmeasurable rather than omitted or zeroed.
func TestEmitGPUDeclaresECCUnmeasurableWhereThePartHasNone(t *testing.T) {
	s := probeGPU(testConfig(fakeSMI(sparkGPU())))
	if s.status != "ok" {
		t.Fatalf("status = %q, want ok", s.status)
	}
	out := newEmitter()
	emitGPU(out, s, s)

	for _, key := range []string{keyECCErrors, keyRowsRemapped} {
		got, ok := out.get(key)
		if !ok {
			t.Errorf("%s was omitted; on hardware that cannot measure it, silence is an unexplained failure", key)
			continue
		}
		if got != unmeasurableValue {
			t.Errorf("%s = %q, want %q — a fabricated number would be a lie about an unmeasured counter", key, got, unmeasurableValue)
		}
	}
	if got, _ := out.get("ecc_mode"); got != "unsupported" {
		t.Errorf("ecc_mode = %q, want unsupported", got)
	}
	// Everything the part CAN report still reports.
	if got, _ := out.get("gpu_temp_c"); got != "41" {
		t.Errorf("gpu_temp_c = %q, want 41", got)
	}
}

// One GPU with ECC is enough to make an N/A counter suspicious: the node is not
// ECC-free, so nothing may be declared unmeasurable on its behalf.
func TestEmitGPUMixedECCSupportIsNotADeclaration(t *testing.T) {
	spark, ecc := sparkGPU(), healthyGPU()
	ecc[fIndex] = "1"

	s := probeGPU(testConfig(fakeSMI(spark, ecc)))
	out := newEmitter()
	emitGPU(out, s, s)
	if got, ok := out.get(keyECCErrors); ok {
		t.Errorf("ecc_errors = %q, want omitted: one GPU answered and one did not", got)
	}
	if got, _ := out.get("ecc_mode"); got != "mixed" {
		t.Errorf("ecc_mode = %q, want mixed", got)
	}
}

// counter() is the three-state read the whole design rests on: measured, "the
// hardware has no such counter", and "we could not read it". Only the middle one
// may ever reach a profile as a declaration.
func TestCounterDistinguishesUnmeasurableFromUnread(t *testing.T) {
	cases := []struct {
		name      string
		vals      []string
		wantTotal int64
		wantState counterState
	}{
		{"every GPU answered", []string{"1", "2"}, 3, counterOK},
		{"every GPU said N/A", []string{"N/A", "[N/A]"}, 0, counterUnmeasurable},
		{"some answered, some did not", []string{"0", "N/A"}, 0, counterUnknown},
		{"not supported on this driver", []string{"Not Supported"}, 0, counterUnmeasurable},
		{"garbage is not a declaration", []string{"banana"}, 0, counterUnknown},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			tbl := &gpuTable{rows: len(c.vals), col: map[string][]string{fECCUncAgg: c.vals}}
			total, state := tbl.counter(fECCUncAgg)
			if state != c.wantState || total != c.wantTotal {
				t.Errorf("counter = %d, %v; want %d, %v", total, state, c.wantTotal, c.wantState)
			}
		})
	}

	t.Run("column never returned", func(t *testing.T) {
		tbl := &gpuTable{rows: 1, col: map[string][]string{fIndex: {"0"}}}
		if _, state := tbl.counter(fECCUncAgg); state != counterUnknown {
			t.Errorf("state = %v, want counterUnknown — a missing column is not a declaration", state)
		}
	})
}

// Pre-Ampere parts have no row remapper, so nvidia-smi rejects the whole query.
// The per-field fallback must keep every other counter.
func TestEmitGPUFallsBackPerField(t *testing.T) {
	g := healthyGPU()
	delete(g, fRemapCorr)
	delete(g, fRemapUnc)
	delete(g, fRemapPending)
	delete(g, fRemapFailure)
	g[fRetiredSBE] = "12"
	g[fRetiredDBE] = "1"

	s := probeGPU(testConfig(fakeSMI(g)))
	if s.status != "ok" {
		t.Fatalf("status = %q, want ok — a rejected field must not lose the whole probe", s.status)
	}
	out := newEmitter()
	emitGPU(out, s, s)
	if _, ok := out.get(keyRowsRemapped); ok {
		t.Error("rows_remapped must be omitted on a part with no row remapper")
	}
	if got, _ := out.get("retired_pages_total"); got != "13" {
		t.Errorf("retired_pages_total = %q, want 13", got)
	}
	if got, _ := out.get(keyECCErrors); got != "0" {
		t.Errorf("ecc_errors = %q, want 0", got)
	}
}

func TestEmitGPUMultiGPUSumsAndPeaks(t *testing.T) {
	a, b := healthyGPU(), healthyGPU()
	b[fIndex] = "1"
	b[fECCUncAgg] = "2"
	b[fRemapCorr] = "3"
	b[fRemapUnc] = "1"
	b[fTemp] = "67"
	b[fPower] = "120.5"

	s := probeGPU(testConfig(fakeSMI(a, b)))
	out := newEmitter()
	emitGPU(out, s, s)

	if got, _ := out.get("device_count"); got != "2" {
		t.Errorf("device_count = %q, want 2", got)
	}
	if got, _ := out.get(keyECCErrors); got != "2" {
		t.Errorf("ecc_errors = %q, want 2 (summed across GPUs)", got)
	}
	if got, _ := out.get(keyRowsRemapped); got != "4" {
		t.Errorf("rows_remapped = %q, want 4 (correctable + uncorrectable)", got)
	}
	if got, _ := out.get("gpu_temp_c"); got != "67" {
		t.Errorf("gpu_temp_c = %q, want the peak 67", got)
	}
	if got, _ := out.get("power_draw_w"); got != "120.5" {
		t.Errorf("power_draw_w = %q, want the peak 120.5", got)
	}
}

// An ECC error that happened WHILE we watched is worse than an old one, and the
// two must not be confused.
func TestEmitGPUWindowedECC(t *testing.T) {
	before := probeGPU(testConfig(fakeSMI(withField(healthyGPU(), fECCUncAgg, "1"))))
	after := probeGPU(testConfig(fakeSMI(withField(healthyGPU(), fECCUncAgg, "4"))))

	out := newEmitter()
	emitGPU(out, before, after)
	if got, _ := out.get(keyECCErrors); got != "4" {
		t.Errorf("ecc_errors = %q, want the lifetime 4", got)
	}
	if got, _ := out.get("ecc_uncorrected_window"); got != "3" {
		t.Errorf("ecc_uncorrected_window = %q, want 3", got)
	}
}

// A counter that goes backwards was reset (driver reload, device rebind), which
// is not evidence of negative faults.
func TestDeltaClampsAReset(t *testing.T) {
	if d, ok := delta(100, 3, true, true); !ok || d != 0 {
		t.Errorf("delta(100, 3) = %d, %v; want 0, true", d, ok)
	}
	if _, ok := delta(1, 2, false, true); ok {
		t.Error("delta must not be ok when the first sample is missing")
	}
}

func withField(g map[string]string, field, value string) map[string]string {
	g[field] = value
	return g
}
