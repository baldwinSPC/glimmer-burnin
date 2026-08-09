package nvvs

import (
	"encoding/json"
	"sort"
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
	d := docs["diag-n1.json"]

	got := d.Diagnostic.Categories[0].Tests[0].Results[0].Status
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
	aux := docs["diag-n1.json"].AuxData

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
	d := docs["diag-n1.json"]

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
	d := docs["diag-n1.json"]

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
	d := docs["diag-n1.json"]

	info := strings.Join(d.Diagnostic.Categories[0].Tests[0].Results[0].Info, "|")
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

// A node-scope verdict on a multi-GPU node is emitted ONCE — issue #206.
//
// This test previously asserted the opposite: that the verdict was broadcast to
// every accelerator with an info line disclosing that the per-device attribution
// had not been measured. The disclosure is honest and it does not reach the
// consumer that matters.
//
// NVVS consumers COUNT results[] objects — tallying failures per entity is the
// normal reason to parse this format. Prose in info[] does not participate in
// counting. So one failing gpu-burn on an eight-GPU node was eight objects with
// status "Fail", and every tallying tool reported eight failed GPUs: an engineer
// replacing seven healthy parts, or a fleet manager quarantining eight devices.
//
// It is not recoverable downstream either. Once the document has been forwarded
// or re-serialised, eight objects are eight results and nothing carries the fact
// that they were one.
//
// The assertion is therefore on the COUNT, and its failure message says what a
// counting consumer would conclude.
func TestANodeVerdictIsEmittedOnceNotOncePerGPU(t *testing.T) {
	gpus := make([]report.GPUInfo, 8)
	for i := range gpus {
		gpus[i] = report.GPUInfo{Index: int32(i), Model: "H100"}
	}
	e := envelope(phaseFailed,
		contract.TestResult{Name: "gpu-burn", Kind: "gpu-burn", Phase: phaseFailed, Nodes: []string{"n1"}},
	)
	docs := render(t, input(e, report.NodeInfo{Name: "n1", GPUs: gpus}))
	results := docs["diag-n1.json"].Diagnostic.Categories[0].Tests[0].Results

	if len(results) != 1 {
		t.Fatalf("one failing node-scope test produced %d results[] objects on an "+
			"8-GPU node. A consumer that tallies failures per entity now reports %d "+
			"failed GPUs for one node-level finding.", len(results), len(results))
	}
	r := results[0]
	if r.EntityID != nil {
		t.Errorf("the verdict is bound to entity_id %d. The runner measured one device "+
			"and the document cannot say which; binding it to one names the wrong part.",
			*r.EntityID)
	}
	if r.EntityGroup != "" {
		t.Errorf("entity_group = %q on a verdict that binds to no entity", r.EntityGroup)
	}

	// The devices are still named, for a human. A machine finds no entity; a
	// person still sees what was in the node.
	info := strings.Join(r.Info, "|")
	if !strings.Contains(info, "scope=node") {
		t.Errorf("nothing says the verdict is node-scoped: %v", r.Info)
	}
	if !strings.Contains(info, "gpus=0,1,2,3,4,5,6,7") {
		t.Errorf("the devices on the node are not named: %v", r.Info)
	}

	// And the entity inventory is still published, so the node's hardware is
	// visible even though the verdict binds to none of it.
	if egs := docs["diag-n1.json"].EntityGroups; len(egs) == 0 || len(egs[0].Entities) != 8 {
		t.Errorf("entity_groups = %+v, want the node's 8 GPUs", egs)
	}
}

// Where the runner said which device it measured, the document says so too.
//
// clockprobe, thermal-soak and gpu-burn each run on ONE CUDA device and emit its
// bus address. On a multi-GPU node, where the verdict binds to no entity, that
// line is the only unambiguous statement available about which part was under
// load — an ordinal is not, because MIG and CUDA_VISIBLE_DEVICES both remap it.
//
// It reaches the document through infoFor, which emits every metric as
// key=value. This test exists because that is load-bearing for a case it was not
// written for: an earlier version of the #206 fix added a second, explicit
// pciBusId line here, and the redundancy was only visible because mutating it
// away changed nothing.
func TestTheDeviceActuallyMeasuredIsNamedWhenKnown(t *testing.T) {
	e := envelope(phaseFailed,
		contract.TestResult{
			Name: "gpu-burn", Kind: "gpu-burn", Phase: phaseFailed, Nodes: []string{"n1"},
			Metrics: map[string]string{"pciBusId": "0000:01:00.0"},
		},
	)
	docs := render(t, input(e, report.NodeInfo{Name: "n1", GPUs: []report.GPUInfo{
		{Index: 0, Model: "H100"}, {Index: 1, Model: "H100"},
	}}))
	r := docs["diag-n1.json"].Diagnostic.Categories[0].Tests[0].Results[0]
	if !strings.Contains(strings.Join(r.Info, "|"), "0000:01:00.0") {
		t.Errorf("the runner reported which device it measured and the document does "+
			"not carry it: %v", r.Info)
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

	if strings.Contains(strings.Join(r.Info, "|"), "scope=node") {
		t.Error("a single-GPU node should not carry the multi-device caveat")
	}
	// And the binding IS made, because here it is exact. Losing it would trade
	// one kind of imprecision for another.
	if r.EntityID == nil || *r.EntityID != 0 || r.EntityGroup != "GPU" {
		t.Errorf("a single-GPU node did not bind exactly: entity_group=%q entity_id=%v",
			r.EntityGroup, r.EntityID)
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

// A measured hardware Fail is not erased by a sibling Error.
//
// `overall()` ranked Error above Fail, and justified it as "deliberately
// pessimistic in the same order the engine uses". The engine uses the opposite
// order, and says so: finalize() is documented "Precedence: Failed beats Error
// beats Passed. A required test that FAILED is a hardware verdict and wins
// outright", and it implements that by setting RunError and then OVERWRITING it
// with RunFailed.
//
// So the operator settled such a run `Failed` while this renderer printed the
// node's verdict as "Not Run" — which in real dcgmi means a plugin that was not
// selected, and which every consumer filters as infrastructure noise. A GPU that
// missed a sustained-clock gate got queued for a re-run instead of an RMA,
// because an unrelated image pull failed on the same node.
//
// The Error-beats-Fail precedence that DOES exist in this project is the PAIR
// rule, and it is a different question: it combines the two ends of ONE
// measurement, where a machinery failure at either end means the link was never
// measured at all. Aggregating INDEPENDENT tests on a host is finalize()'s case,
// not that one.
func TestAMeasuredFailSurvivesASiblingError(t *testing.T) {
	e := envelope(phaseFailed,
		contract.TestResult{
			Name: "soak", Kind: "thermal-soak", Phase: phaseFailed, Nodes: []string{"n1"},
			Violations: []contract.Violation{{
				Metric: "sustainedClockPct", Cause: "Measurement", Reason: "61.2 is below 80"}},
		},
		contract.TestResult{
			Name: "fp4", Kind: "compute-smoke", Phase: phaseError, Nodes: []string{"n1"},
			Message: "ImagePullBackOff",
		},
	)
	docs := render(t, input(e))
	got := docs["diag-n1.json"].OverallResult

	if got == statusNotRun {
		t.Fatalf("Overall Result = %q. The hardware WAS judged and fell short; "+
			"\"Not Run\" states that it was never measured, which is a strictly weaker "+
			"claim than the run established. In real dcgmi that value means a plugin "+
			"was not selected, so a triage queue treats it as infrastructure noise and "+
			"the node is re-run rather than replaced.", got)
	}
	if got != statusFail {
		t.Errorf("Overall Result = %q, want %q", got, statusFail)
	}

	// And the operator's own aggregate agrees, which is the point: the two must
	// not disagree about the same run.
	if e.Phase != phaseFailed {
		t.Fatalf("fixture drift: the envelope's own phase is %q", e.Phase)
	}
}

// An Error with no Fail beside it is still "Not Run".
//
// Asserted alongside the above so the fix cannot be "always report Fail", which
// would destroy the distinction in the other direction.
func TestAnErrorAloneIsStillNotRun(t *testing.T) {
	e := envelope(phaseError,
		contract.TestResult{Name: "fp4", Kind: "compute-smoke", Phase: phaseError,
			Nodes: []string{"n1"}, Message: "ImagePullBackOff"},
		contract.TestResult{Name: "smoke2", Kind: "compute-smoke", Phase: phasePassed,
			Nodes: []string{"n1"}},
	)
	docs := render(t, input(e))
	if got := docs["diag-n1.json"].OverallResult; got != statusNotRun {
		t.Errorf("Overall Result = %q, want %q — a passing test beside an errored one "+
			"does not make the node measured", got, statusNotRun)
	}
}

// A baseline run never reports an admission.
//
// The Baseline flag reached only the prose Warning and aux_data, so the document
// asserted "Overall Result": "Pass" and then said in English that it certified
// nothing. Anything reading the verdict field — which is what that field is for
// — could not tell a thresholdless measurement sweep from a certification, which
// is the exact fail-open the flag was added to close.
func TestABaselineRunNeverReportsAnAdmission(t *testing.T) {
	e := envelope(phasePassed,
		contract.TestResult{Name: "soak", Kind: "thermal-soak", Phase: phasePassed,
			Nodes: []string{"n1"}},
	)
	e.Baseline = true
	docs := render(t, input(e))
	d := docs["diag-n1.json"]

	if d.OverallResult == statusPass {
		t.Fatalf("Overall Result = %q on a run that applied NO thresholds. A consumer "+
			"admitting nodes on \"the last diagnostic passed\" certifies a fleet against "+
			"a run that gated nothing.", d.OverallResult)
	}
	// The per-test statuses are honest too: nothing here met a criterion,
	// because there were none to meet.
	st := d.Diagnostic.Categories[0].Tests[0].Results[0].Status
	if st == statusPass {
		t.Errorf("a baseline result rendered %q; no threshold was applied to it", st)
	}
	// And the prose stays, because a machine reads the field and a person reads
	// the sentence.
	if !strings.Contains(strings.ToUpper(d.Warning), "BASELINE") {
		t.Error("the baseline label was lost from the document-level warning")
	}
}

// statusUnder is the single site of the baseline rule, and this pins it there.
//
// The document-level test above cannot distinguish two guards from one: an
// earlier version of this fix also refused a baseline admission inside overall(),
// and mutating that away changed nothing — overall() reaches every phase THROUGH
// statusUnder, so its own baseline case could never fire. Unreachable code in the
// shape of a guard is worse than no code, so it was removed and the rule lives
// in one place, tested directly.
func TestStatusUnderIsWhereTheBaselineRuleLives(t *testing.T) {
	for _, phase := range []string{phasePassed, phaseFailed, phaseSkipped, phaseError} {
		if got := statusUnder(phase, true); got != statusSkip {
			t.Errorf("statusUnder(%q, baseline=true) = %q, want %q — no threshold was "+
				"applied to any of them, so none met a criterion", phase, got, statusSkip)
		}
	}
	// With the flag off it is the plain mapping again, so the rule is about the
	// baseline rather than about refusing everything.
	if got := statusUnder(phasePassed, false); got != statusPass {
		t.Errorf("statusUnder(Passed, baseline=false) = %q, want %q", got, statusPass)
	}
	if got := statusUnder(phaseError, false); got != statusNotRun {
		t.Errorf("statusUnder(Error, baseline=false) = %q, want %q", got, statusNotRun)
	}
}

// A serial is never attributed to the wrong device.
//
// metadata carried "GPU Device Serials" and "GPU Device IDs" as POSITIONAL
// arrays built by appending only non-empty values. With GPU 0's serial
// uncaptured and GPU 1's captured, the array was ["SN-GPU1"] and a positional
// reader attributes it to GPU 0 — naming the wrong part in the one document
// somebody attaches to an RMA.
//
// A positional array cannot express a partial set at all, which is why DCGM's
// live form is per-entity serial_num: it is self-describing, and it can simply
// be absent for the device that has none.
func TestASerialIsNeverAttributedToTheWrongDevice(t *testing.T) {
	e := envelope(phasePassed,
		contract.TestResult{Name: "smoke", Kind: "compute-smoke", Phase: phasePassed,
			Nodes: []string{"n1"}},
	)
	docs := render(t, input(e, report.NodeInfo{Name: "n1", GPUs: []report.GPUInfo{
		{Index: 0, Model: "NVIDIA H100"},                    // serial NOT captured
		{Index: 1, Model: "NVIDIA H100", Serial: "SN-GPU1"}, // captured
	}}))
	d := docs["diag-n1.json"]

	// Per-entity, where position cannot drift.
	if len(d.EntityGroups) != 1 || len(d.EntityGroups[0].Entities) != 2 {
		t.Fatalf("entity_groups = %+v, want both devices", d.EntityGroups)
	}
	ents := d.EntityGroups[0].Entities
	if ents[0].SerialNum != "" {
		t.Errorf("GPU 0's serial was not captured but the document reports %q",
			ents[0].SerialNum)
	}
	if ents[1].SerialNum != "SN-GPU1" {
		t.Errorf("GPU 1's serial = %q, want SN-GPU1", ents[1].SerialNum)
	}

	// And no positional array anywhere that could re-introduce the drift.
	raw := rawOf(t, e, report.NodeInfo{Name: "n1", GPUs: []report.GPUInfo{
		{Index: 0, Model: "NVIDIA H100"},
		{Index: 1, Model: "NVIDIA H100", Serial: "SN-GPU1"},
	}})
	for _, key := range []string{"GPU Device Serials", "GPU Device IDs"} {
		if strings.Contains(raw, key) {
			t.Errorf("%q is a positional array; with a partial set it shifts every "+
				"later element and misattributes a serial to the wrong device", key)
		}
	}
}

// A node that was targeted and reported nothing still gets a document.
//
// The document set came only from nodes NAMED IN RESULTS, so a node the run
// targeted and which produced nothing — the case a reader most needs to see —
// vanished, and the report read as a fleet that was fully measured.
func TestATargetedNodeThatReportedNothingStillGetsADocument(t *testing.T) {
	e := envelope(phasePassed,
		contract.TestResult{Name: "smoke", Kind: "compute-smoke", Phase: phasePassed,
			Nodes: []string{"n1"}},
	)
	docs := render(t, input(e,
		report.NodeInfo{Name: "n1"},
		report.NodeInfo{Name: "n2", Kernel: "6.11.0"}, // targeted, reported nothing
	))
	d, ok := docs["diag-n2.json"]
	if !ok {
		t.Fatalf("no document for a targeted node that produced no result; the report "+
			"reads as a fleet that was fully measured. Got: %v", docKeys(docs))
	}
	if d.OverallResult == statusPass {
		t.Errorf("a node that measured nothing reports %q", d.OverallResult)
	}
}

// A declared skip keeps its reason.
//
// Only Failed and Error results carried their message. A Skipped result's
// message is the runner's own `_SKIP` line — the declaration that the test does
// not apply to this hardware, and the evidence that the skip was DECLARED rather
// than a crash exiting 2. Dropping it leaves a bare "Skip" that a reader cannot
// distinguish from either.
func TestADeclaredSkipKeepsItsReason(t *testing.T) {
	e := envelope(phasePassed,
		contract.TestResult{
			Name: "fp4", Kind: "compute-smoke", Phase: phaseSkipped, Nodes: []string{"n1"},
			Message: "FP4_GEMM_SKIP: NVFP4 requires compute capability 12.0/12.1, and this part reports 9.0",
		},
	)
	docs := render(t, input(e))
	r := docs["diag-n1.json"].Diagnostic.Categories[0].Tests[0].Results[0]

	joined := strings.Join(r.Info, "|")
	for _, w := range r.Warnings {
		joined += "|" + w.Warning
	}
	if !strings.Contains(joined, "FP4_GEMM_SKIP") {
		t.Fatalf("the skip's declared reason is gone; a bare %q cannot be told from a "+
			"crash that exited 2. Result: %+v", r.Status, r)
	}
}

// A result naming no node is reported once, not in every document.
func TestANodelessResultIsNotDuplicatedIntoEveryDocument(t *testing.T) {
	e := envelope(phaseError,
		contract.TestResult{Name: "unresolvable", Kind: "custom", Phase: phaseError,
			Message: "the profile named a test that does not resolve"},
		contract.TestResult{Name: "smoke", Kind: "compute-smoke", Phase: phasePassed,
			Nodes: []string{"n1"}},
		contract.TestResult{Name: "smoke", Kind: "compute-smoke", Phase: phasePassed,
			Nodes: []string{"n2"}},
	)
	docs := render(t, input(e))

	var seen int
	for name, d := range docs {
		for _, c := range d.Diagnostic.Categories {
			for _, tst := range c.Tests {
				if strings.Contains(tst.Name, "unresolvable") {
					seen++
					t.Logf("found in %s", name)
				}
			}
		}
	}
	if seen != 1 {
		t.Errorf("a result naming no node appears in %d documents, want 1. Reported "+
			"once per node it becomes N findings from one, which is the same "+
			"over-counting a broadcast verdict causes.", seen)
	}
}

// A result whose node name is empty is not dropped.
func TestAResultWithAnEmptyNodeNameIsNotDropped(t *testing.T) {
	// A named node must be present too: with the orphan alone, nodesIn returns
	// nothing and the empty-name fallback document picks it up by accident. The
	// loss only happens once there is a real node to enumerate instead.
	e := envelope(phaseFailed,
		contract.TestResult{Name: "orphan", Kind: "custom", Phase: phaseFailed,
			Nodes: []string{""}, Message: "recorded against no node"},
		contract.TestResult{Name: "smoke", Kind: "compute-smoke", Phase: phasePassed,
			Nodes: []string{"n1"}},
	)
	docs := render(t, input(e))

	var seen bool
	for _, d := range docs {
		for _, c := range d.Diagnostic.Categories {
			for _, tst := range c.Tests {
				if tst.Name == "orphan" {
					seen = true
				}
			}
		}
	}
	if !seen {
		t.Fatalf("a result whose node name is the empty string vanished from every "+
			"document. A required acceptance test missing from the document that "+
			"certifies the fleet is the worst failure this package has. Got: %v",
			docKeys(docs))
	}
}

// A partial record does not render confidently.
//
// contract.Summary is an independent tally of executions. When it accounts for
// more than the results present, these documents are a SUBSET — and a confident
// document over an inconsistent record is how a fleet gets signed off against
// the results that happened to be delivered.
func TestAPartialRecordIsNotAConfidentDocument(t *testing.T) {
	e := envelope(phasePassed,
		contract.TestResult{Name: "smoke", Kind: "compute-smoke", Phase: phasePassed,
			Nodes: []string{"n1"}},
	)
	e.Summary = contract.Summary{Passed: 12} // twelve ran; one was delivered
	docs := render(t, input(e))
	d := docs["diag-n1.json"]

	if d.OverallResult == statusPass {
		t.Fatalf("a document covering 1 of 12 executions reports %q", d.OverallResult)
	}
	if !strings.Contains(strings.ToLower(d.Warning), "incomplete") &&
		!strings.Contains(strings.ToLower(d.Warning), "subset") {
		t.Errorf("nothing in the document says the record is partial: Warning=%q", d.Warning)
	}
}

func docKeys(m map[string]Document) []string {
	out := make([]string, 0, len(m))
	for k := range m {
		out = append(out, k)
	}
	sort.Strings(out)
	return out
}

func rawOf(t *testing.T, e *contract.Envelope, nodes ...report.NodeInfo) string {
	t.Helper()
	outs, err := Renderer{}.Render(input(e, nodes...))
	if err != nil {
		t.Fatal(err)
	}
	var b strings.Builder
	for _, o := range outs {
		b.Write(o.Data)
	}
	return b.String()
}
