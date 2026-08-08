package nvvs

import (
	"encoding/json"
	"strings"
	"testing"
	"time"

	"github.com/baldwinSPC/glimmer-burnin/pkg/contract"
	"github.com/baldwinSPC/glimmer-burnin/pkg/report"
)

func at(s string) time.Time {
	t, err := time.Parse(time.RFC3339, s)
	if err != nil {
		panic(err)
	}
	return t
}

func envelope(phase string, results ...contract.TestResult) *contract.Envelope {
	return &contract.Envelope{
		Version:    contract.Version,
		DeliveryID: "d1",
		Reason:     contract.ReasonPhaseChanged,
		SentAt:     at("2026-08-06T12:00:00Z"),
		Run:        contract.RunRef{Namespace: "burnin", Name: "acceptance", UID: "uid-1", Profile: "node-acceptance"},
		Phase:      phase,
		Results:    results,
	}
}

func render(t *testing.T, in report.Input) map[string]Document {
	t.Helper()
	outs, err := New().Render(in)
	if err != nil {
		t.Fatalf("Render: %v", err)
	}
	docs := map[string]Document{}
	for _, o := range outs {
		var d Document
		if err := json.Unmarshal(o.Data, &d); err != nil {
			t.Fatalf("re-decoding %s: %v", o.Filename, err)
		}
		docs[o.Filename] = d
	}
	return docs
}

func input(e *contract.Envelope, nodes ...report.NodeInfo) report.Input {
	return report.Input{
		Envelopes: []*contract.Envelope{e},
		Nodes:     nodes,
		Meta:      report.Meta{Generator: "glimmer-burnin", Version: "v0.5.0", GeneratedAt: at("2026-08-06T12:30:00Z")},
	}
}

func TestErrorRendersAsNotRunAndNeverAsFail(t *testing.T) {
	// The single most important mapping in the package. Error means the
	// hardware was never judged; Fail means it was judged and fell short.
	// Collapsing them at the schema boundary would destroy the distinction
	// where nobody downstream could recover it.
	e := envelope(phaseError,
		contract.TestResult{Name: "compute-smoke", Kind: "compute-smoke", Phase: phaseError,
			Nodes: []string{"n1"}, Message: "image pull failed"},
	)
	docs := render(t, input(e))
	d := docs["diag-n1.json"].Diagnostic

	got := d.Categories[0].Tests[0].Results[0].Status
	if got != statusNotRun {
		t.Errorf("status = %q, want %q — an Error must never render as a hardware verdict", got, statusNotRun)
	}
	if got == statusFail {
		t.Fatal("an Error rendered as Fail")
	}
	if d.OverallResult != statusNotRun {
		t.Errorf("Overall Result = %q, want %q", d.OverallResult, statusNotRun)
	}
}

func TestALinkResultAppearsInBothEndpointsAndIsNeverSplit(t *testing.T) {
	// NVVS has no vocabulary for a result about a relationship. The document
	// says so in words, in every copy, and the identical result appears in both
	// endpoints' documents — attributing a fabric measurement to one machine
	// sends an engineer to replace the wrong part.
	e := envelope(phaseFailed,
		contract.TestResult{Name: "ib-write-bw", Kind: "ib-write-bw", Scope: "Pair",
			Phase: phaseFailed, Nodes: []string{"n1", "n2"},
			Metrics: map[string]string{"bandwidthGbps": "42.1"}},
	)
	docs := render(t, input(e))

	if len(docs) != 2 {
		t.Fatalf("got %d documents, want one per endpoint", len(docs))
	}
	for name, d := range docs {
		tst := d.Diagnostic.Categories[0].Tests[0]
		if !strings.Contains(tst.Name, "(link)") {
			t.Errorf("%s: test name %q does not mark itself as a link", name, tst.Name)
		}
		if len(tst.Results) != 1 {
			t.Errorf("%s: got %d results, want exactly one — a link verdict must not be split per endpoint",
				name, len(tst.Results))
		}
		if tst.Results[0].Status != statusFail {
			t.Errorf("%s: both copies must carry the same status, got %q", name, tst.Results[0].Status)
		}
		joined := strings.Join(tst.Results[0].Info, "|")
		if !strings.Contains(joined, "link=n1<->n2") {
			t.Errorf("%s: info does not name the link: %v", name, tst.Results[0].Info)
		}
		if !strings.Contains(joined, "not about this endpoint") {
			t.Errorf("%s: info does not warn that the verdict is about the link: %v", name, tst.Results[0].Info)
		}
	}

	// The peer named must differ between the two documents.
	p1 := strings.Join(docs["diag-n1.json"].Diagnostic.Categories[0].Tests[0].Results[0].Info, "|")
	p2 := strings.Join(docs["diag-n2.json"].Diagnostic.Categories[0].Tests[0].Results[0].Info, "|")
	if !strings.Contains(p1, "peer=n2") || !strings.Contains(p2, "peer=n1") {
		t.Error("each endpoint's document must name the other as the peer")
	}
}

func TestNoNVIDIAPluginNameIsEverEmitted(t *testing.T) {
	// A synthesised entry must not be mistakable for a vendor plugin's own
	// result. Our names stay ours.
	e := envelope(phasePassed,
		contract.TestResult{Name: "gpu-burn", Kind: "gpu-burn", Phase: phasePassed, Nodes: []string{"n1"}},
		contract.TestResult{Name: "thermal-soak", Kind: "thermal-soak", Phase: phasePassed, Nodes: []string{"n1"}},
		contract.TestResult{Name: "memory-bw", Kind: "memory-bw", Phase: phasePassed, Nodes: []string{"n1"}},
	)
	outs, err := New().Render(input(e))
	if err != nil {
		t.Fatalf("Render: %v", err)
	}
	body := string(outs[0].Data)
	for _, plugin := range []string{
		"sm_stress", "targeted_stress", "targeted_power", "memory_bandwidth",
		"memtest", "pulse", "nvbandwidth", "diagnostic", "pcie", "eud",
	} {
		if strings.Contains(body, `"`+plugin+`"`) {
			t.Errorf("document contains the NVIDIA plugin name %q — a synthesised entry must not be mistakable for that plugin's result", plugin)
		}
	}
}

func TestEveryDocumentCarriesProvenance(t *testing.T) {
	// A schema-compatible report with no provenance is indistinguishable from
	// vendor output once it has been forwarded twice.
	e := envelope(phasePassed,
		contract.TestResult{Name: "a", Kind: "compute-smoke", Phase: phasePassed, Nodes: []string{"n1"}},
	)
	docs := render(t, input(e))
	aux := docs["diag-n1.json"].Diagnostic.AuxData

	if aux == nil {
		t.Fatal("no aux_data: the document does not say what produced it")
	}
	if !strings.Contains(aux.Notice, "NOT produced by") || !strings.Contains(aux.Notice, "NVIDIA") {
		t.Errorf("notice does not disclaim NVIDIA authorship: %q", aux.Notice)
	}
	if !strings.Contains(aux.Generator, "glimmer-burnin") {
		t.Errorf("generator = %q, want this project named", aux.Generator)
	}
	if aux.RunUID != "uid-1" {
		t.Errorf("run_uid = %q, want the run traceable from the document", aux.RunUID)
	}
}

func TestAViolationCarriesItsCauseSoAReaderKnowsWhoShouldAct(t *testing.T) {
	// The cause is the difference between replacing a part and fixing a
	// profile, and NVVS has no equivalent field.
	e := envelope(phaseFailed,
		contract.TestResult{
			Name: "clockprobe", Kind: "clockprobe", Phase: phaseFailed, Nodes: []string{"n1"},
			Violations: []contract.Violation{
				{Index: 0, Metric: "sustainedClockPct", Cause: causeMeasurement, Kind: "Unsatisfied", Reason: "sustainedClockPct 61.2 < 70"},
				{Index: 1, Metric: "pdWedgeSuspected", Cause: causeAuthoring, Kind: "Unsound", Reason: "gate on a label-valued metric"},
			},
		},
	)
	docs := render(t, input(e))
	ws := docs["diag-n1.json"].Diagnostic.Categories[0].Tests[0].Results[0].Warnings

	if len(ws) != 2 {
		t.Fatalf("got %d warnings, want one per violation", len(ws))
	}
	if ws[0].ErrorCategory != causeMeasurement || ws[0].ErrorSeverity != "error" {
		t.Errorf("a hardware shortfall should read as an error: %+v", ws[0])
	}
	if ws[1].ErrorCategory != causeAuthoring || ws[1].ErrorSeverity == "error" {
		t.Errorf("a broken threshold must not be ranked as a hardware error: %+v", ws[1])
	}
}

func TestNoErrorIDIsInvented(t *testing.T) {
	// NVVS carries NVIDIA's own DCGM_FR_* codes in error_id. Emitting one would
	// let a consumer look up a vendor diagnosis the vendor never made.
	e := envelope(phaseFailed,
		contract.TestResult{Name: "a", Kind: "compute-smoke", Phase: phaseFailed, Nodes: []string{"n1"},
			Violations: []contract.Violation{{Metric: "m", Cause: causeMeasurement, Reason: "fell short"}}},
	)
	outs, _ := New().Render(input(e))
	if strings.Contains(string(outs[0].Data), "error_id") {
		t.Error("document emits error_id — a synthesised entry must not carry a vendor error code")
	}
	if strings.Contains(string(outs[0].Data), "DCGM_FR_") {
		t.Error("document invents a DCGM_FR_* code")
	}
}

func TestUnmeasuredFieldsAreAbsentNotBlank(t *testing.T) {
	// An empty serial in an RMA is a claim that the part has none.
	e := envelope(phasePassed,
		contract.TestResult{Name: "a", Kind: "compute-smoke", Phase: phasePassed, Nodes: []string{"n1"}},
	)
	in := input(e, report.NodeInfo{
		Name: "n1",
		GPUs: []report.GPUInfo{{Index: 0, Model: "NVIDIA GB10"}}, // no serial, no driver
	})
	outs, _ := New().Render(in)
	body := string(outs[0].Data)

	if strings.Contains(body, `"serial_num"`) {
		t.Error("an uncaptured serial was emitted as a field — it must be absent")
	}
	if strings.Contains(body, `"Driver Version Detected"`) {
		t.Error("an uncaptured driver version was emitted")
	}
	if strings.Contains(body, `"GPU Device Serials"`) {
		t.Error("an empty serial list was emitted")
	}
	if !strings.Contains(body, "NVIDIA GB10") {
		t.Error("a captured model should be present")
	}
}

func TestABaselineRunIsLabelledAsMeasurementNotCertification(t *testing.T) {
	e := envelope(phasePassed,
		contract.TestResult{Name: "a", Kind: "compute-smoke", Phase: phasePassed, Nodes: []string{"n1"}},
	)
	e.Baseline = true
	docs := render(t, input(e))
	d := docs["diag-n1.json"].Diagnostic

	if !strings.Contains(strings.ToUpper(d.Warning), "BASELINE") {
		t.Errorf("a baseline run must be visibly labelled, got Warning=%q", d.Warning)
	}
	if !d.AuxData.Baseline {
		t.Error("aux_data does not record that this was a baseline run")
	}
}

func TestAnUnfinishedRunSaysSoRatherThanReadingAsAVerdict(t *testing.T) {
	e := envelope(phaseRunning,
		contract.TestResult{Name: "thermal-soak", Kind: "thermal-soak", Phase: phaseRunning, Nodes: []string{"n1"}},
	)
	e.Reason = contract.ReasonCheckpoint
	docs := render(t, input(e))
	d := docs["diag-n1.json"].Diagnostic

	if d.RuntimeError == "" {
		t.Error("an unfinished run rendered with no indication that it is not a verdict")
	}
	if !strings.Contains(d.RuntimeError, "not a verdict") {
		t.Errorf("runtime_error should say what it is: %q", d.RuntimeError)
	}
}

func TestNotEvaluatedIsStatedRatherThanOmitted(t *testing.T) {
	// An absent gate reads as a gate that passed.
	e := envelope(phasePassed,
		contract.TestResult{
			Name: "host-health", Kind: "host-health", Phase: phasePassed, Nodes: []string{"n1"},
			NotEvaluated: []contract.NotEvaluated{{Metric: "eccErrors", Reason: "unmeasurable on this hardware"}},
			Unmeasurable: []string{"eccErrors"},
		},
	)
	docs := render(t, input(e))
	info := strings.Join(docs["diag-n1.json"].Diagnostic.Categories[0].Tests[0].Results[0].Info, "|")

	if !strings.Contains(info, "NOT EVALUATED: eccErrors") {
		t.Errorf("an unevaluated gate must be stated, got %q", info)
	}
}

func TestADroppedArtifactIsSurfaced(t *testing.T) {
	e := envelope(phasePassed,
		contract.TestResult{
			Name: "dcgm-diag", Kind: "dcgm-diag", Phase: phasePassed, Nodes: []string{"n1"},
			Artifacts: []contract.ArtifactRef{{Name: "dcgmi.json", Dropped: "exceeded the per-artifact cap"}},
		},
	)
	docs := render(t, input(e))
	d := docs["diag-n1.json"].Diagnostic

	info := strings.Join(d.Categories[0].Tests[0].Results[0].Info, "|")
	if !strings.Contains(info, "evidence not stored") {
		t.Errorf("a dropped artifact must be visible in the document, got %q", info)
	}
	found := false
	for _, w := range d.AuxData.Warnings {
		if strings.Contains(w, "dcgmi.json") {
			found = true
		}
	}
	if !found {
		t.Errorf("aux_data should also record the omission: %v", d.AuxData.Warnings)
	}
}

func TestAHostTestAttachesToTheHostNotToAnAccelerator(t *testing.T) {
	e := envelope(phasePassed,
		contract.TestResult{Name: "memory-stress", Kind: "memory-stress", Phase: phasePassed, Nodes: []string{"n1"}},
	)
	in := input(e, report.NodeInfo{Name: "n1", GPUs: []report.GPUInfo{{Index: 0, Model: "GB10"}}})
	docs := render(t, in)
	res := docs["diag-n1.json"].Diagnostic.Categories[0].Tests[0].Results[0]

	if res.EntityGroup != "CPU" {
		t.Errorf("entity_group = %q, want CPU: host DIMM stress is not a claim about the accelerator", res.EntityGroup)
	}
}

func TestAMultiGPUNodeSaysTheAttributionWasNotMeasured(t *testing.T) {
	// The runner reported one verdict for the node. Broadcasting it to four
	// devices is an over-attribution, and the document must not let a reader
	// assume per-device detail was measured.
	e := envelope(phaseFailed,
		contract.TestResult{Name: "gpu-burn", Kind: "gpu-burn", Phase: phaseFailed, Nodes: []string{"n1"}},
	)
	in := input(e, report.NodeInfo{Name: "n1", GPUs: []report.GPUInfo{
		{Index: 0, Model: "H100"}, {Index: 1, Model: "H100"},
	}})
	docs := render(t, in)
	results := docs["diag-n1.json"].Diagnostic.Categories[0].Tests[0].Results

	if len(results) != 2 {
		t.Fatalf("got %d results, want one per accelerator", len(results))
	}
	for i, r := range results {
		if !strings.Contains(strings.Join(r.Info, "|"), "attribution=node") {
			t.Errorf("result %d does not disclose that the attribution is the node's: %v", i, r.Info)
		}
	}
}

func TestASingleGPUNodeDoesNotCarryTheAttributionCaveat(t *testing.T) {
	// On a one-GPU part the attribution is exact, and a needless caveat is
	// noise that trains readers to ignore caveats.
	e := envelope(phasePassed,
		contract.TestResult{Name: "gpu-burn", Kind: "gpu-burn", Phase: phasePassed, Nodes: []string{"n1"}},
	)
	in := input(e, report.NodeInfo{Name: "n1", GPUs: []report.GPUInfo{{Index: 0, Model: "GB10"}}})
	docs := render(t, in)
	r := docs["diag-n1.json"].Diagnostic.Categories[0].Tests[0].Results[0]

	if strings.Contains(strings.Join(r.Info, "|"), "attribution=node") {
		t.Error("a single-GPU node should not carry the multi-device attribution caveat")
	}
}

func TestCategoriesUseTheVendorTaxonomy(t *testing.T) {
	e := envelope(phasePassed,
		contract.TestResult{Name: "gpu-burn", Kind: "gpu-burn", Phase: phasePassed, Nodes: []string{"n1"}},
		contract.TestResult{Name: "host-health", Kind: "host-health", Phase: phasePassed, Nodes: []string{"n1"}},
		contract.TestResult{Name: "nccl", Kind: "nccl", Phase: phasePassed, Nodes: []string{"n1"}},
	)
	docs := render(t, input(e))
	got := map[string]string{}
	for _, c := range docs["diag-n1.json"].Diagnostic.Categories {
		for _, tst := range c.Tests {
			got[tst.Name] = c.Category
		}
	}
	want := map[string]string{"gpu-burn": catStress, "host-health": catHardware, "nccl": catIntegration}
	for name, cat := range want {
		if got[name] != cat {
			t.Errorf("%s is in %q, want %q", name, got[name], cat)
		}
	}
}

func TestRenderIsDeterministic(t *testing.T) {
	// Golden fixtures are only meaningful if output is stable.
	e := envelope(phasePassed,
		contract.TestResult{Name: "b", Kind: "gpu-burn", Phase: phasePassed, Nodes: []string{"n1"},
			Metrics: map[string]string{"z": "1", "a": "2", "m": "3"}},
		contract.TestResult{Name: "a", Kind: "gpu-burn", Phase: phasePassed, Nodes: []string{"n1"}},
	)
	first, err := New().Render(input(e))
	if err != nil {
		t.Fatalf("Render: %v", err)
	}
	for i := 0; i < 5; i++ {
		again, err := New().Render(input(e))
		if err != nil {
			t.Fatalf("Render: %v", err)
		}
		if string(again[0].Data) != string(first[0].Data) {
			t.Fatal("output is not byte-stable across renders — map iteration is leaking into the document")
		}
	}
}

func TestRenderDoesNotMutateItsInput(t *testing.T) {
	// Callers render one run through several renderers; a renderer that
	// normalised in place would change what the next one sees.
	e := envelope(phaseFailed,
		contract.TestResult{Name: "ib-write-bw", Kind: "ib-write-bw", Phase: phaseFailed, Nodes: []string{"n1", "n2"}},
	)
	in := input(e, report.NodeInfo{Name: "n1", GPUs: []report.GPUInfo{{Index: 0}, {Index: 1}}})
	before, _ := json.Marshal(in.Envelopes)

	if _, err := New().Render(in); err != nil {
		t.Fatalf("Render: %v", err)
	}
	after, _ := json.Marshal(in.Envelopes)
	if string(before) != string(after) {
		t.Error("Render mutated its Input")
	}
}

func TestTheRootKeyIsTheVendorsOwn(t *testing.T) {
	e := envelope(phasePassed,
		contract.TestResult{Name: "a", Kind: "compute-smoke", Phase: phasePassed, Nodes: []string{"n1"}},
	)
	outs, _ := New().Render(input(e))

	var raw map[string]json.RawMessage
	if err := json.Unmarshal(outs[0].Data, &raw); err != nil {
		t.Fatalf("decoding: %v", err)
	}
	if _, ok := raw["DCGM Diagnostic"]; !ok {
		t.Errorf("root key is not %q: %v", "DCGM Diagnostic", keysOf(raw))
	}
}

func keysOf(m map[string]json.RawMessage) []string {
	out := make([]string, 0, len(m))
	for k := range m {
		out = append(out, k)
	}
	return out
}
