package htmlreport

import (
	"strings"
	"testing"

	"github.com/baldwinSPC/glimmer-burnin/pkg/contract"
	"github.com/baldwinSPC/glimmer-burnin/pkg/report"
)

func md(t *testing.T, i report.Input) string {
	t.Helper()
	outs, err := NewMarkdown().Render(i)
	if err != nil {
		t.Fatalf("Render: %v", err)
	}
	if len(outs) != 1 || outs[0].Filename != "burnin-report.md" {
		t.Fatalf("unexpected outputs: %+v", outs)
	}
	return string(outs[0].Data)
}

// The two formats must never disagree about the same run — issue #146.
//
// This is the reason the markdown renderer shares build() rather than deriving
// anything of its own, and the reason it lives in this package. Two renderers
// deciding independently which node failed, or wording who should act
// differently, is how one engineer reading a PR comment and another reading an
// attached HTML file reach different conclusions about the same hardware. That
// divergence does not appear in review; it appears months later, in one format
// only, in front of somebody deciding whether to accept a delivery.
//
// So this asserts the FACTS, not the layout: every node, every test, every
// status word, and every violated metric present in one must be present in the
// other.
func TestBothFormatsAgreeAboutTheSameRun(t *testing.T) {
	e := envelope("Failed",
		contract.TestResult{Name: "compute-smoke", Kind: "compute-smoke", Phase: "Passed",
			Nodes: []string{"n1"}},
		contract.TestResult{Name: "thermal-soak", Kind: "thermal-soak", Phase: "Failed",
			Nodes: []string{"n2"}, Message: "sustainedClockPct 61.2 below 80",
			Violations: []contract.Violation{{
				Metric: "sustainedClockPct", Cause: "Measurement", Kind: "BelowMinimum",
				Reason: "61.2 is below the minimum 80"}}},
		contract.TestResult{Name: "host-health", Kind: "host-health", Phase: "Error",
			Nodes: []string{"n2"}, Message: "ImagePullBackOff"},
		contract.TestResult{Name: "ib-write-bw", Kind: "ib-write-bw", Phase: "Failed",
			Scope: "Pair", Nodes: []string{"n1", "n2"}},
	)
	i := in(e,
		report.NodeInfo{Name: "n1", Kernel: "6.11.0-1004-nvidia"},
		report.NodeInfo{Name: "n2", Kernel: "6.11.0-1004-nvidia"},
	)
	h, m := html(t, i), md(t, i)

	for _, fact := range []string{
		// every node
		"n1", "n2",
		// every test
		"compute-smoke", "thermal-soak", "host-health", "ib-write-bw",
		// the metric that was violated, and the number
		"sustainedClockPct", "61.2",
		// the error's own reason
		"ImagePullBackOff",
	} {
		if !strings.Contains(m, fact) {
			t.Errorf("markdown omits %q, which the HTML report carries — the two formats "+
				"would tell two engineers different things about the same hardware", fact)
		}
		if !strings.Contains(h, fact) {
			t.Errorf("HTML omits %q; this fixture is not exercising what it claims to", fact)
		}
	}

	// The violation's CAUSE SENTENCE, which is the field that tells a reader
	// whether to touch hardware at all. It is pinned separately from the metric
	// name because the metric name also appears in the message — so dropping
	// the violations block entirely left this test green until it did.
	//
	// The expected sentence is taken from build(), not written here, so the test
	// compares the two renderers against their shared source rather than against
	// a third opinion.
	for _, want := range causeSentencesIn(t, i) {
		if !strings.Contains(m, want) {
			t.Errorf("markdown omits the cause sentence %q. A Failed test can mix a "+
				"hardware shortfall with a broken threshold, and only one of them is a "+
				"reason to touch a node.", want)
		}
		if !strings.Contains(h, want) {
			t.Errorf("HTML omits the cause sentence %q", want)
		}
	}

	// The status VOCABULARY is shared, so a reader moving between the two does
	// not have to translate.
	for _, word := range statusWordsIn(t, i) {
		if !strings.Contains(m, word) {
			t.Errorf("markdown never uses the status word %q that the shared view "+
				"produced; it has invented its own vocabulary", word)
		}
	}
}

// causeSentencesIn returns the cause explanations build() produced for this run.
func causeSentencesIn(t *testing.T, i report.Input) []string {
	t.Helper()
	p := build(report.BuildView(i))
	seen := map[string]bool{}
	var out []string
	collect := func(vs []violation) {
		for _, v := range vs {
			if v.CauseWhy != "" && !seen[v.CauseWhy] {
				seen[v.CauseWhy] = true
				out = append(out, v.CauseWhy)
			}
		}
	}
	for _, n := range p.Nodes {
		for _, tst := range n.Tests {
			collect(tst.Violations)
		}
	}
	for _, l := range p.Links {
		collect(l.Violations)
	}
	if len(out) == 0 {
		t.Fatal("the fixture produced no violations with a cause, so this comparison " +
			"asserts nothing")
	}
	return out
}

// statusWordsIn returns the words build() chose for this run's phases, so the
// test compares against the shared source rather than against a list it made up.
func statusWordsIn(t *testing.T, i report.Input) []string {
	t.Helper()
	p := build(report.BuildView(i))
	seen := map[string]bool{}
	var out []string
	add := func(w string) {
		if w != "" && !seen[w] {
			seen[w] = true
			out = append(out, w)
		}
	}
	for _, n := range p.Nodes {
		for _, tst := range n.Tests {
			add(tst.Status.Word)
		}
	}
	for _, l := range p.Links {
		add(l.Status.Word)
	}
	if len(out) == 0 {
		t.Fatal("the fixture produced no statuses, so this comparison asserts nothing")
	}
	return out
}

// Error and Fail stay distinguishable in text.
//
// Markdown has no colour at all, so a status that was only a colour would be
// invisible here. That makes this format the honest test of a rule the HTML one
// also has to satisfy: a reader must be able to tell a condemned node from an
// unmeasured one.
func TestErrorAndFailReadDifferentlyInPlainText(t *testing.T) {
	e := envelope("Failed",
		contract.TestResult{Name: "soak", Kind: "thermal-soak", Phase: "Failed",
			Nodes: []string{"n1"}, Message: "fell short"},
		contract.TestResult{Name: "smoke", Kind: "compute-smoke", Phase: "Error",
			Nodes: []string{"n2"}, Message: "ImagePullBackOff"},
	)
	m := md(t, in(e))

	p := build(report.BuildView(in(e)))
	var failWord, errWord string
	for _, n := range p.Nodes {
		for _, tst := range n.Tests {
			switch tst.Name {
			case "soak":
				failWord = tst.Status.Word
			case "smoke":
				errWord = tst.Status.Word
			}
		}
	}
	if failWord == "" || errWord == "" {
		t.Fatal("the fixture did not produce both a Failed and an Error result")
	}
	if failWord == errWord {
		t.Fatalf("the shared view gives Failed and Error the same word %q — in a format "+
			"with no colour, a reader cannot tell a condemned node from an unmeasured "+
			"one", failWord)
	}
	if !strings.Contains(m, failWord) || !strings.Contains(m, errWord) {
		t.Errorf("markdown does not carry both status words (%q, %q)", failWord, errWord)
	}
}

// A pipe in runner-supplied text must not split the row it is in.
//
// A test name, a metric name and a threshold reason all originate in a runner's
// stdout, which is the least trusted input here. Every other markdown
// metacharacter degrades to odd emphasis; this one silently shifts a table so a
// reader sees the wrong value against the wrong label.
func TestAPipeInRunnerTextCannotBreakATable(t *testing.T) {
	e := envelope("Failed",
		contract.TestResult{
			Name: "odd|name", Kind: "custom", Phase: "Failed", Nodes: []string{"n1"},
			Message: "a message with | a pipe",
			Metrics: map[string]string{"weird|metric": "1|2"},
			Violations: []contract.Violation{{
				Metric: "weird|metric", Cause: "Measurement",
				Reason: "expected | got"}},
		},
	)
	m := md(t, in(e))

	// Every pipe that came from the runner is escaped; the only bare pipes left
	// are the ones this renderer wrote as table structure.
	for _, unescaped := range []string{"odd|name", "weird|metric", "with | a pipe", "expected | got", "1|2"} {
		if strings.Contains(m, unescaped) {
			t.Errorf("%q appears unescaped — the row it is in is silently split, and a "+
				"reader sees the wrong value against the wrong label", unescaped)
		}
	}
	// And the content is still there, escaped rather than dropped: evidence
	// removed is worse than evidence that reads awkwardly.
	for _, escaped := range []string{`odd\|name`, `weird\|metric`} {
		if !strings.Contains(m, escaped) {
			t.Errorf("%q was dropped rather than escaped; the evidence is gone", escaped)
		}
	}
}

// A run that gated nothing says so on the first line.
func TestABaselineIsLabelledInTheBanner(t *testing.T) {
	e := envelope("Passed",
		contract.TestResult{Name: "sweep", Kind: "compute-smoke", Phase: "Passed", Nodes: []string{"n1"}})
	e.Baseline = true
	m := md(t, in(e))

	first := strings.SplitN(m, "\n\n", 3)
	banner := strings.Join(first[:min(2, len(first))], "\n\n")
	if !strings.Contains(strings.ToUpper(banner), "BASELINE") {
		t.Fatalf("a thresholdless sweep is not labelled in the banner, so a reader who "+
			"takes only the headline takes away a certification the run did not make:\n%s",
			banner)
	}
	if !strings.Contains(m, "certifies nothing") {
		t.Error("the label does not say what a baseline means")
	}
}

// An unfinished run is progress, never a verdict.
func TestAnUnfinishedRunIsLabelledInTheBanner(t *testing.T) {
	e := envelope("Running",
		contract.TestResult{Name: "soak", Kind: "thermal-soak", Phase: "Running", Nodes: []string{"n1"}})
	e.Reason = contract.ReasonCheckpoint
	e.CheckpointSequence = 4
	m := md(t, in(e))
	if !strings.Contains(m, "INCOMPLETE") || !strings.Contains(m, "not a verdict") {
		t.Errorf("a mid-flight checkpoint reads as a verdict of record:\n%s",
			strings.SplitN(m, "\n\n", 3)[1])
	}
}

// "Not recorded" and "recorded as nothing" must not look the same.
func TestAnUnrecordedFieldIsNotAnEmptyOne(t *testing.T) {
	e := envelope("Passed",
		contract.TestResult{Name: "smoke", Kind: "compute-smoke", Phase: "Passed", Nodes: []string{"n1"}})
	// No profile, no cluster, and results with no timestamps.
	e.Run.Profile = ""
	m := md(t, in(e))

	if strings.Contains(m, "| Profile |  |") || strings.Contains(m, "| Started |  |") {
		t.Error("an uncaptured field rendered as an empty cell, which reads as a value " +
			"that was measured and found to be nothing")
	}
	if !strings.Contains(m, "—") {
		t.Error("nothing marks the uncaptured fields as uncaptured")
	}
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}
