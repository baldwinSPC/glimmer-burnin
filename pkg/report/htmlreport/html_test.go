package htmlreport

import (
	"regexp"
	"strings"
	"testing"
	"time"

	"github.com/baldwinSPC/glimmer-burnin/pkg/contract"
	"github.com/baldwinSPC/glimmer-burnin/pkg/report"
)

func input(mutate func(*contract.Envelope), nodes ...report.NodeInfo) report.Input {
	e := &contract.Envelope{
		Version: contract.Version, DeliveryID: "d-1", Reason: contract.ReasonPhaseChanged,
		Run:   contract.RunRef{Namespace: "burnin", Name: "run1", UID: "uid-1", Profile: "acceptance"},
		Phase: "Failed", SentAt: time.Unix(1750000000, 0).UTC(),
		Summary: contract.Summary{Passed: 1, Failed: 1, Errored: 1},
	}
	if mutate != nil {
		mutate(e)
	}
	return report.Input{
		Envelopes: []*contract.Envelope{e}, Nodes: nodes,
		Meta: report.Meta{Generator: "glimmer-burnin", Version: "v0.5.0"},
	}
}

func renderHTML(t *testing.T, in report.Input) string {
	t.Helper()
	outs, err := Renderer{}.Render(in)
	if err != nil {
		t.Fatal(err)
	}
	if len(outs) != 1 || outs[0].Filename != "burnin-report.html" {
		t.Fatalf("unexpected outputs: %+v", outs)
	}
	return string(outs[0].Data)
}

func renderMD(t *testing.T, in report.Input) string {
	t.Helper()
	outs, err := MarkdownRenderer{}.Render(in)
	if err != nil {
		t.Fatal(err)
	}
	return string(outs[0].Data)
}

// The file opens with no network access — issue #146.
//
// A report that fetches a stylesheet is useless in a secure facility, useless as
// an email attachment, and useless in six months when the CDN path has moved. A
// burn-in report's whole job is to still be readable long after the run, in a
// room that may have no route to the internet at all.
func TestNoExternalRequests(t *testing.T) {
	html := renderHTML(t, input(func(e *contract.Envelope) {
		e.Results = []contract.TestResult{
			{Name: "fp4", Phase: "Passed", Nodes: []string{"spark-a"},
				Metrics: map[string]string{"tflops": "412.1"}},
			{Name: "ib", Phase: "Failed", Scope: "Pair", Nodes: []string{"spark-a", "spark-b"}},
		}
	}, report.NodeInfo{Name: "spark-a", Kernel: "6.11.0"}))

	// Any absolute or protocol-relative reference is a request the browser will
	// make. Checked as raw substrings, because a URL can appear in an attribute,
	// a CSS url(), or an @import.
	for _, bad := range []string{"http://", "https://", "//fonts.", "src=\"//", "href=\"//", "@import"} {
		if strings.Contains(html, bad) {
			t.Errorf("the rendered report contains %q — it will not open in a facility "+
				"with no egress, which is where an acceptance report is read", bad)
		}
	}
	// And nothing that could fetch at runtime.
	for _, tag := range []string{"<script", "<iframe", "<object", "<embed", "<link "} {
		if strings.Contains(strings.ToLower(html), tag) {
			t.Errorf("the report contains %q", tag)
		}
	}
	// Guard the guard: a renderer that emitted nothing would pass every check
	// above.
	if len(html) < 2000 || !strings.Contains(html, "spark-a") {
		t.Fatalf("the report is %d bytes and does not mention the node — this test is "+
			"asserting nothing", len(html))
	}
}

// Error and Fail are distinguishable in TEXT, not only in colour.
//
// A reader who cannot perceive the colour difference, or who printed the page in
// greyscale — which is what happens to a report attached to an RMA — must still
// tell a condemned node from an unmeasured one.
func TestErrorAndFailAreDistinctInText(t *testing.T) {
	// Deliberately NO violations. The per-violation advice explains a cause in
	// the same words, so a fixture carrying one lets the legend be deleted with
	// this test still green — which is exactly what happened the first two times
	// this was mutated. With no violations the legend is the only thing that can
	// say what the symbols mean, which is the claim being made.
	in := input(func(e *contract.Envelope) {
		e.Results = []contract.TestResult{
			{Name: "soak", Phase: "Failed", Nodes: []string{"spark-a"},
				Message: "sustainedClockPct 61.2 < 80"},
			{Name: "fp4", Phase: "Error", Nodes: []string{"spark-b"},
				Message: "ImagePullBackOff"},
		}
	})

	for name, doc := range map[string]string{"html": renderHTML(t, in), "markdown": renderMD(t, in)} {
		t.Run(name, func(t *testing.T) {
			// Strip every colour declaration; whatever remains is what a
			// greyscale reader has.
			plain := regexp.MustCompile(`(?s)<style>.*?</style>`).ReplaceAllString(doc, "")
			if !strings.Contains(plain, "FAIL") || !strings.Contains(plain, "ERR") {
				t.Fatalf("with colour removed, the document does not say FAIL and ERR "+
					"in text:\n%s", truncate(plain))
			}
			// And the LEGEND must say what the difference means, because the
			// symbols alone do not tell an engineer whether to touch a node.
			//
			// The phrases below are pinned because they appear nowhere else. A
			// first version asserted on "unjudged", which also appears in the
			// banner's summary line — so deleting the legend's own explanation
			// left the test green. A symbol with no legend is two glyphs a
			// reader has to guess at.
			for _, want := range []string{
				"measured and fell short",        // what Failed means
				"the measurement did not happen", // what Error means
			} {
				if !strings.Contains(plain, want) {
					t.Errorf("the legend never says %q, so a reader cannot tell a "+
						"condemned node from an unmeasured one:\n%s", want, truncate(plain))
				}
			}
		})
	}
}

// A Pair result appears once, in the Links section, naming both nodes — never
// split across two node rows.
func TestAPairIsInLinksAndNotSplit(t *testing.T) {
	in := input(func(e *contract.Envelope) {
		e.Results = []contract.TestResult{
			{Name: "ib-bw", Phase: "Failed", Scope: "Pair", Nodes: []string{"spark-a", "spark-b"},
				Message: "94.1 Gbps"},
			{Name: "fp4", Phase: "Passed", Scope: "Node", Nodes: []string{"spark-a"}},
			{Name: "fp4", Phase: "Passed", Scope: "Node", Nodes: []string{"spark-b"}},
		}
	})

	for name, doc := range map[string]string{"html": renderHTML(t, in), "markdown": renderMD(t, in)} {
		t.Run(name, func(t *testing.T) {
			if n := strings.Count(doc, "ib-bw"); n != 1 {
				t.Errorf("the link test appears %d times, want 1 — a point-to-point "+
					"measurement attributed to one endpoint sends an engineer to "+
					"replace the wrong part", n)
			}
			// html/template escapes the arrow, which is correct — a browser
			// renders "&lt;-&gt;" as "<->", so what a READER sees is the label
			// the view model produced. Asserting on the raw bytes would be
			// asserting that the renderer does not escape, which is the
			// opposite of what this document needs.
			label := "spark-a <-> spark-b"
			if name == "html" {
				label = "spark-a &lt;-&gt; spark-b"
			}
			if !strings.Contains(doc, label) {
				t.Errorf("the link is not labelled with both endpoints (looked for %q)", label)
			}
			if !strings.Contains(doc, "never about either endpoint") {
				t.Error("the document does not warn that a link verdict is not a claim " +
					"about an endpoint")
			}
		})
	}
}

// A baseline run is labelled in the banner. A thresholdless sweep and a
// certification both end "Passed", and the label is the only thing between them.
func TestABaselineRunIsLabelledInTheBanner(t *testing.T) {
	in := input(func(e *contract.Envelope) {
		e.Phase, e.Baseline = "Passed", true
		e.Results = []contract.TestResult{{Name: "sweep", Phase: "Passed", Nodes: []string{"spark-a"}}}
	})
	for name, doc := range map[string]string{"html": renderHTML(t, in), "markdown": renderMD(t, in)} {
		t.Run(name, func(t *testing.T) {
			if !strings.Contains(doc, "BASELINE") {
				t.Fatal("a thresholdless sweep is not labelled; it reads as a certification")
			}
			if !strings.Contains(doc, "gated nothing") {
				t.Error("the document does not say the run gated nothing")
			}
		})
	}
}

// An unfinished run's phase is progress, never a verdict.
func TestAnUnfinishedRunSaysSo(t *testing.T) {
	in := input(func(e *contract.Envelope) {
		e.Reason, e.Phase, e.CheckpointSequence = contract.ReasonCheckpoint, "Running", 4
		e.Results = []contract.TestResult{{Name: "soak", Phase: "Running", Nodes: []string{"spark-a"}}}
	})
	doc := renderHTML(t, in)
	if !strings.Contains(doc, "INCOMPLETE") {
		t.Error("a mid-flight checkpoint rendered as a verdict of record")
	}
	if !strings.Contains(doc, "progress, not a verdict") {
		t.Error("the document does not say the phase is progress")
	}
}

// A gate that did not run is not a gate that passed, and a value nobody measured
// is not a zero.
func TestNotEvaluatedAndUnmeasurableSurvive(t *testing.T) {
	in := input(func(e *contract.Envelope) {
		e.Results = []contract.TestResult{{
			Name: "health", Phase: "Passed", Nodes: []string{"spark-a"},
			NotEvaluated: []contract.NotEvaluated{{Metric: "eccErrors", Reason: "declared unmeasurable"}},
			Unmeasurable: []string{"eccErrors"},
		}}
	})
	for name, doc := range map[string]string{"html": renderHTML(t, in), "markdown": renderMD(t, in)} {
		t.Run(name, func(t *testing.T) {
			if !strings.Contains(doc, "eccErrors") {
				t.Fatal("a threshold that never ran is invisible in the report")
			}
			if !strings.Contains(doc, "not a gate that passed") {
				t.Error("the report does not say an unevaluated gate is not a pass")
			}
			if !strings.Contains(doc, "not a measurement of zero") {
				t.Error("the report does not distinguish 'this part has no ECC' from " +
					"'this part reported zero ECC errors'")
			}
		})
	}
}

// A violation's cause tells the reader who should act, in plain language.
func TestEveryCauseTranslatesToAnAction(t *testing.T) {
	in := input(func(e *contract.Envelope) {
		e.Results = []contract.TestResult{{
			Name: "mixed", Phase: "Failed", Nodes: []string{"spark-a"},
			Violations: []contract.Violation{
				{Metric: "clockPct", Cause: "Measurement", Reason: "61.2 < 80"},
				{Metric: "busGBs", Cause: "Evidence", Reason: "the runner emitted no value"},
				{Metric: "computeCap", Cause: "Authoring", Reason: "compared a version as a decimal"},
			},
		}}
	})
	doc := renderHTML(t, in)
	for _, want := range []string{
		"verdict about the part",
		"UNJUDGED, not condemned",
		"No node should be touched",
	} {
		if !strings.Contains(doc, want) {
			t.Errorf("the report never says %q — a Failed test can mix a hardware "+
				"shortfall with a broken threshold, and only one is a reason to "+
				"touch a node", want)
		}
	}
}

// A node with no result at all still appears. A report that omitted it would
// read as a fleet that was fully measured.
func TestATargetedNodeThatReportedNothingStillAppears(t *testing.T) {
	in := input(func(e *contract.Envelope) {
		e.Results = []contract.TestResult{{Name: "fp4", Phase: "Passed", Nodes: []string{"spark-a"}}}
	}, report.NodeInfo{Name: "spark-a"}, report.NodeInfo{Name: "spark-z", Kernel: "6.11.0"})

	doc := renderHTML(t, in)
	if !strings.Contains(doc, "spark-z") {
		t.Fatal("a node in the inventory that produced no result vanished from the " +
			"report, which then reads as a fleet that was fully measured")
	}
}

// A print stylesheet exists, so the document yields a readable PDF.
func TestAPrintStylesheetIsPresent(t *testing.T) {
	doc := renderHTML(t, input(func(e *contract.Envelope) {
		e.Results = []contract.TestResult{{Name: "fp4", Phase: "Passed", Nodes: []string{"spark-a"}}}
	}))
	if !strings.Contains(doc, "@media print") {
		t.Fatal("no print stylesheet: an RMA attachment is printed, and a report that " +
			"paginates badly loses the evidence across a page break")
	}
	if !strings.Contains(doc, "page-break-inside") {
		t.Error("nothing prevents a table being split across a page break")
	}
}

// HTML escaping: a runner's message is untrusted text and must not become markup.
func TestARunnerMessageCannotInjectMarkup(t *testing.T) {
	doc := renderHTML(t, input(func(e *contract.Envelope) {
		e.Results = []contract.TestResult{{
			Name: "evil", Phase: "Failed", Nodes: []string{"spark-a"},
			Message: `<script>alert(1)</script><img src=x onerror=alert(2)>`,
		}}
	}))
	// The check is on the MARKUP, not on the payload text. "onerror=alert(2)"
	// appears in the correctly-escaped output too, so matching that would flag
	// working code — which it did, the first time this test was written.
	for _, live := range []string{"<script", "<img"} {
		if strings.Contains(doc, live) {
			t.Fatalf("a runner's stdout produced live %q markup in the report — a "+
				"runner is a third-party image and its output is the least trusted "+
				"input here", live)
		}
	}
	if !strings.Contains(doc, "&lt;script&gt;") {
		t.Error("the message was dropped rather than escaped; the evidence is gone")
	}
}

func truncate(s string) string {
	if len(s) > 1500 {
		return s[:1500] + "…"
	}
	return s
}
