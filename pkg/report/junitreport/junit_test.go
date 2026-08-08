package junitreport

import (
	"encoding/xml"
	"strings"
	"testing"
	"time"

	"github.com/baldwinSPC/glimmer-burnin/pkg/contract"
	"github.com/baldwinSPC/glimmer-burnin/pkg/report"
)

func render(t *testing.T, results ...contract.TestResult) (string, testsuites) {
	t.Helper()
	in := report.Input{
		Envelopes: []*contract.Envelope{{
			Version: contract.Version, DeliveryID: "d-1", Reason: contract.ReasonPhaseChanged,
			Run:   contract.RunRef{Namespace: "burnin", Name: "run1", UID: "uid-1", Profile: "acceptance"},
			Phase: "Failed", SentAt: time.Unix(1750000000, 0).UTC(), Results: results,
		}},
		Meta: report.Meta{Generator: "glimmer-burnin", Version: "v0.5.0"},
	}
	outs, err := Renderer{}.Render(in)
	if err != nil {
		t.Fatal(err)
	}
	if len(outs) != 1 || outs[0].Filename != "junit.xml" {
		t.Fatalf("unexpected outputs: %+v", outs)
	}
	var doc testsuites
	if err := xml.Unmarshal(outs[0].Data, &doc); err != nil {
		t.Fatalf("the renderer produced XML it cannot parse back: %v", err)
	}
	return string(outs[0].Data), doc
}

func findCase(doc testsuites, suite, substr string) *testcase {
	for _, s := range doc.Suites {
		if s.Name != suite {
			continue
		}
		for i := range s.Cases {
			if strings.Contains(s.Cases[i].Name, substr) {
				return &s.Cases[i]
			}
		}
	}
	return nil
}

// An errored test produces <error>, never <failure> — issue #147.
//
// Collapsing the two erases the distinction this project's entire verdict engine
// exists to preserve. On a dashboard an unpullable image and a degraded GPU
// would look identical, and only one of them is a reason to touch hardware.
func TestErrorIsNeverAFailure(t *testing.T) {
	raw, doc := render(t,
		contract.TestResult{
			Name: "fp4", Kind: "ComputeSmoke", Scope: "Node", Phase: "Error",
			Nodes: []string{"spark-a"}, Message: "ImagePullBackOff: manifest unknown",
		},
		contract.TestResult{
			Name: "soak", Kind: "ThermalSoak", Scope: "Node", Phase: "Failed",
			Nodes: []string{"spark-a"}, Message: "sustainedClockPct 61.2 < 80",
			Violations: []contract.Violation{{
				Index: 0, Metric: "sustainedClockPct", Cause: "Measurement",
				Kind: "BelowMinimum", Reason: "61.2 is below the minimum 80",
			}},
		},
	)

	errored := findCase(doc, "spark-a", "fp4")
	if errored == nil {
		t.Fatal("the errored test vanished from the document")
	}
	if errored.Failure != nil {
		t.Fatal("an Error rendered as <failure> — a dashboard cannot tell an image " +
			"pull from a degraded GPU, and only one is a reason to touch hardware")
	}
	if errored.Error == nil {
		t.Fatal("an Error rendered as neither <error> nor <failure>, so it reads as a pass")
	}
	if !strings.Contains(errored.Error.Body, "NOT judged") {
		t.Errorf("the <error> body does not say the hardware was unjudged: %q", errored.Error.Body)
	}

	failed := findCase(doc, "spark-a", "soak")
	if failed == nil || failed.Failure == nil {
		t.Fatalf("a threshold violation did not render as <failure>: %+v", failed)
	}
	if failed.Error != nil {
		t.Error("a hardware failure also rendered as <error>")
	}
	if !strings.Contains(failed.Failure.Body, "sustainedClockPct") {
		t.Errorf("the failure body does not name the gate: %q", failed.Failure.Body)
	}
	if !strings.Contains(failed.Failure.Body, "hardware fell short") {
		t.Errorf("the failure body does not say who should act: %q", failed.Failure.Body)
	}

	// And the top-level tally keeps them apart, which is what a dashboard reads.
	if doc.Errors != 1 || doc.Failures != 1 {
		t.Errorf("tally = %d errors, %d failures; want 1 and 1", doc.Errors, doc.Failures)
	}
	if strings.Count(raw, "<failure") != 1 {
		t.Errorf("expected exactly one <failure> element in:\n%s", raw)
	}
}

// A Pair result appears ONCE, in the links suite, naming both nodes.
//
// A point-to-point measurement is a property of the link. Attributing it to one
// endpoint sends an engineer to replace the wrong part.
func TestAPairAppearsOnceNamingBothNodes(t *testing.T) {
	_, doc := render(t,
		contract.TestResult{
			Name: "ib-bw", Kind: "IBWriteBW", Scope: "Pair", Phase: "Failed",
			Nodes: []string{"spark-a", "spark-b"}, Message: "94.1 Gbps < 180",
		},
		contract.TestResult{
			Name: "fp4", Kind: "ComputeSmoke", Scope: "Node", Phase: "Passed",
			Nodes: []string{"spark-a"},
		},
	)

	var seen int
	for _, s := range doc.Suites {
		for _, c := range s.Cases {
			if strings.Contains(c.Name, "ib-bw") {
				seen++
				if s.Name != linksSuite {
					t.Errorf("the link result appeared in suite %q, not %q — attributing "+
						"a link measurement to one endpoint sends an engineer to replace "+
						"the wrong part", s.Name, linksSuite)
				}
				for _, node := range []string{"spark-a", "spark-b"} {
					if !strings.Contains(c.Name, node) {
						t.Errorf("case %q does not name %s", c.Name, node)
					}
				}
			}
		}
	}
	if seen != 1 {
		t.Errorf("the link result appeared %d times, want exactly 1", seen)
	}
}

// A Group result is one verdict naming every rank.
func TestAGroupAppearsOnceNamingEveryRank(t *testing.T) {
	_, doc := render(t, contract.TestResult{
		Name: "nccl-all", Kind: "NCCL", Scope: "Group", Phase: "Failed",
		Nodes: []string{"spark-a", "spark-b", "spark-c", "spark-d"},
	})

	c := findCase(doc, linksSuite, "nccl-all")
	if c == nil {
		t.Fatal("the Group result is not in the links suite")
	}
	for _, node := range []string{"spark-a", "spark-b", "spark-c", "spark-d"} {
		if !strings.Contains(c.Name, node) {
			t.Errorf("case %q omits rank on %s — a deadline names the ranks that hung, "+
				"and a report that hides them indicts every healthy rank equally", c.Name, node)
		}
	}
	if len(doc.Suites) != 1 {
		t.Errorf("a Group result created %d suites; it is ONE verdict", len(doc.Suites))
	}
}

// A baseline sweep says so in the suite name, because a CI dashboard shows the
// name and nothing else.
func TestABaselineSweepIsLabelledOnTheDashboard(t *testing.T) {
	in := report.Input{Envelopes: []*contract.Envelope{{
		Version: contract.Version, DeliveryID: "d-1", Reason: contract.ReasonPhaseChanged,
		Run:   contract.RunRef{Namespace: "burnin", Name: "run1", UID: "uid-1"},
		Phase: "Passed", SentAt: time.Unix(1750000000, 0).UTC(), Baseline: true,
		Results: []contract.TestResult{{Name: "sweep", Phase: "Passed", Nodes: []string{"spark-a"}}},
	}}}
	outs, err := Renderer{}.Render(in)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(string(outs[0].Data), "baseline") {
		t.Error("a thresholdless sweep rendered indistinguishably from a certification; " +
			"a consumer gating on 'the last run passed' would certify hardware against " +
			"a run that gated nothing")
	}
}

// A phase this build does not recognise is an <error>, never a silent pass.
func TestAnUnrecognisedPhaseIsNotAPass(t *testing.T) {
	_, doc := render(t, contract.TestResult{
		Name: "future", Phase: "Quarantined", Nodes: []string{"spark-a"},
	})
	c := findCase(doc, "spark-a", "future")
	if c == nil {
		t.Fatal("the result vanished")
	}
	if c.Error == nil {
		t.Fatal("a phase this build cannot classify rendered as a pass — an unrecognised " +
			"verdict has certified nothing, and reporting it green lets unmeasured " +
			"hardware into a fleet")
	}
}

// A gate that did not run is not a gate that passed.
func TestNotEvaluatedThresholdsSurviveIntoTheDocument(t *testing.T) {
	_, doc := render(t, contract.TestResult{
		Name: "health", Phase: "Failed", Nodes: []string{"spark-a"},
		Message:    "xidEvents 3 != 0",
		Violations: []contract.Violation{{Metric: "xidEvents", Cause: "Measurement", Reason: "3 != 0"}},
		NotEvaluated: []contract.NotEvaluated{{
			Metric: "eccErrors", Reason: "the runner declared eccErrors unmeasurable on this part",
		}},
	})
	c := findCase(doc, "spark-a", "health")
	if c == nil || c.Failure == nil {
		t.Fatal("no failure rendered")
	}
	if !strings.Contains(c.Failure.Body, "eccErrors") ||
		!strings.Contains(c.Failure.Body, "NOT EVALUATED") {
		t.Errorf("a gate that never ran is invisible in the document — a reader cannot "+
			"tell it from one that passed: %q", c.Failure.Body)
	}
}

// A duration nobody recorded is omitted, never zero-filled: 0.000 reads as a
// test that ran instantly.
func TestAnUnrecordedDurationIsOmittedNotZeroed(t *testing.T) {
	start := time.Unix(1750000000, 0).UTC()
	end := start.Add(90 * time.Second)
	_, doc := render(t,
		contract.TestResult{Name: "timed", Phase: "Passed", Nodes: []string{"spark-a"},
			StartedAt: &start, FinishedAt: &end},
		contract.TestResult{Name: "untimed", Phase: "Passed", Nodes: []string{"spark-a"}},
	)
	if c := findCase(doc, "spark-a", "timed"); c == nil || c.Time != "90.000" {
		t.Errorf("timed case = %+v, want time 90.000", c)
	}
	if c := findCase(doc, "spark-a", "untimed"); c == nil || c.Time != "" {
		t.Errorf("untimed case carries time=%q; 0.000 reads as a test that ran instantly", c.Time)
	}
}

// A node's captured identity rides along, and only what was captured.
func TestOnlyCapturedIdentityIsEmitted(t *testing.T) {
	in := report.Input{
		Envelopes: []*contract.Envelope{{
			Version: contract.Version, DeliveryID: "d-1", Reason: contract.ReasonPhaseChanged,
			Run:   contract.RunRef{Namespace: "burnin", Name: "run1", UID: "uid-1"},
			Phase: "Passed", SentAt: time.Unix(1750000000, 0).UTC(),
			Results: []contract.TestResult{{Name: "fp4", Phase: "Passed", Nodes: []string{"spark-a"}}},
		}},
		Nodes: []report.NodeInfo{{
			Name: "spark-a", Kernel: "6.11.0-1004-nvidia",
			// Serial deliberately unset: this fleet's probe never captured it.
			GPUs: []report.GPUInfo{{Index: 0, Model: "NVIDIA GB10", Arch: "sm_121"}},
		}},
	}
	outs, err := Renderer{}.Render(in)
	if err != nil {
		t.Fatal(err)
	}
	raw := string(outs[0].Data)
	if !strings.Contains(raw, "6.11.0-1004-nvidia") || !strings.Contains(raw, "NVIDIA GB10") {
		t.Error("captured identity did not reach the document")
	}
	if strings.Contains(raw, "gpu.0.serial") {
		t.Error("a serial nobody captured appears in the document — an empty serial in " +
			"an RMA is a claim that the part has none")
	}
}
