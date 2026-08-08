package htmlreport

import (
	"regexp"
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

func in(e *contract.Envelope, nodes ...report.NodeInfo) report.Input {
	return report.Input{
		Envelopes: []*contract.Envelope{e},
		Nodes:     nodes,
		Meta:      report.Meta{Generator: "glimmer-burnin", Version: "v0.5.0", GeneratedAt: at("2026-08-06T12:30:00Z")},
	}
}

func html(t *testing.T, i report.Input) string {
	t.Helper()
	outs, err := New().Render(i)
	if err != nil {
		t.Fatalf("Render: %v", err)
	}
	if len(outs) != 1 {
		t.Fatalf("got %d documents, want one — a human reads this run as a whole", len(outs))
	}
	return string(outs[0].Data)
}

// TestTheDocumentMakesNoExternalRequests is the constraint the whole package
// exists under.
//
// A report that fetches a stylesheet is useless in a secure facility, useless
// as an email attachment, and useless in eighteen months when the path moved.
func TestTheDocumentMakesNoExternalRequests(t *testing.T) {
	e := envelope("Passed",
		contract.TestResult{Name: "compute-smoke", Kind: "compute-smoke", Phase: "Passed", Nodes: []string{"n1"}},
	)
	body := html(t, in(e))

	for _, bad := range []string{"http://", "https://", `src="//`, `href="//`, "@import"} {
		if strings.Contains(body, bad) {
			t.Errorf("document contains %q — it must render with no network access", bad)
		}
	}
	// A <script> would also be a dependency on a JS engine behaving.
	if regexp.MustCompile(`(?i)<script`).MatchString(body) {
		t.Error("document contains a script tag")
	}
}

// TestErrorAndFailAreDistinguishableWithoutColour is the accessibility rule and
// the correctness rule at once: they mean different things and this document
// gets printed in greyscale.
func TestErrorAndFailAreDistinguishableWithoutColour(t *testing.T) {
	e := envelope("Failed",
		contract.TestResult{Name: "a", Kind: "gpu-burn", Phase: "Failed", Nodes: []string{"n1"}},
		contract.TestResult{Name: "b", Kind: "gpu-burn", Phase: "Error", Nodes: []string{"n1"}},
	)
	body := html(t, in(e))

	if !strings.Contains(body, ">FAIL<") {
		t.Error("no FAIL text: status must not depend on colour alone")
	}
	if !strings.Contains(body, ">ERROR<") {
		t.Error("no ERROR text: status must not depend on colour alone")
	}
	// And the two must not share a colour variable.
	if !strings.Contains(body, "--fail:") || !strings.Contains(body, "--error:") {
		t.Error("fail and error do not have separate colours defined")
	}
	failHue := cssVar(t, body, "--fail")
	errHue := cssVar(t, body, "--error")
	if failHue == errHue {
		t.Errorf("fail and error render in the same colour (%s) — they lead to different actions", failHue)
	}
}

func cssVar(t *testing.T, body, name string) string {
	t.Helper()
	m := regexp.MustCompile(regexp.QuoteMeta(name) + `:\s*(#[0-9a-fA-F]{3,8})`).FindStringSubmatch(body)
	if m == nil {
		t.Fatalf("no definition for %s", name)
	}
	return m[1]
}

// TestALinkIsPresentedAsALinkAndNeverPerEndpoint is the misrender that sends an
// engineer to the wrong machine.
func TestALinkIsPresentedAsALinkAndNeverPerEndpoint(t *testing.T) {
	e := envelope("Failed",
		contract.TestResult{Name: "ib-write-bw", Kind: "ib-write-bw", Scope: "Pair",
			Phase: "Failed", Nodes: []string{"n1", "n2"},
			Metrics: map[string]string{"bandwidthGbps": "42.1"}},
		contract.TestResult{Name: "host-health", Kind: "host-health", Scope: "Node",
			Phase: "Passed", Nodes: []string{"n1"}},
	)
	body := html(t, in(e))

	if !strings.Contains(body, "n1 &lt;-&gt; n2") && !strings.Contains(body, "n1 <-> n2") {
		t.Error("the link's endpoints are not shown together")
	}
	if !strings.Contains(body, "not about either end") {
		t.Error("the document does not say a link verdict is about the connection")
	}
	// The link test must not appear as a column in the per-node matrix.
	matrix := between(body, "<thead>", "</thead>")
	if strings.Contains(matrix, "ib-write-bw") {
		t.Error("a link test appears as a per-node column — it belongs to no single node")
	}
	if !strings.Contains(matrix, "host-health") {
		t.Error("a node-scope test is missing from the matrix")
	}
}

func between(s, start, end string) string {
	i := strings.Index(s, start)
	if i < 0 {
		return ""
	}
	j := strings.Index(s[i:], end)
	if j < 0 {
		return s[i:]
	}
	return s[i : i+j]
}

// TestAViolationExplainsWhoShouldAct — the cause is the most decision-relevant
// thing in the document and must be in words, not a bare enum.
func TestAViolationExplainsWhoShouldAct(t *testing.T) {
	e := envelope("Failed",
		contract.TestResult{
			Name: "clockprobe", Kind: "clockprobe", Phase: "Failed", Nodes: []string{"n1"},
			Violations: []contract.Violation{
				{Metric: "sustainedClockPct", Cause: "Measurement", Reason: "sustainedClockPct 61.2 < 70"},
				{Metric: "pdWedge", Cause: "Authoring", Reason: "gate on a label-valued metric"},
				{Metric: "eccErrors", Cause: "Evidence", Reason: "metric was never reported"},
			},
		},
	)
	body := html(t, in(e))

	if !strings.Contains(body, "about the part") {
		t.Error("a Measurement cause does not say the hardware is implicated")
	}
	if !strings.Contains(body, "profile needs fixing") {
		t.Error("an Authoring cause does not say no hardware is implicated")
	}
	if !strings.Contains(body, "unjudged") {
		t.Error("an Evidence cause does not say the node is unjudged rather than condemned")
	}
}

// TestABaselineRunIsNotPresentedAsAnAcceptance closes the fail-open the flag
// exists for.
func TestABaselineRunIsNotPresentedAsAnAcceptance(t *testing.T) {
	e := envelope("Passed",
		contract.TestResult{Name: "a", Kind: "gpu-burn", Phase: "Passed", Nodes: []string{"n1"}},
	)
	e.Baseline = true
	body := html(t, in(e))

	if !strings.Contains(body, "nothing was certified") {
		t.Error("a baseline run is not labelled as measurement rather than certification")
	}
}

// TestAnUnfinishedRunIsNotPresentedAsAVerdict.
func TestAnUnfinishedRunIsNotPresentedAsAVerdict(t *testing.T) {
	e := envelope("Running",
		contract.TestResult{Name: "thermal-soak", Kind: "thermal-soak", Phase: "Running", Nodes: []string{"n1"}},
	)
	e.Reason = contract.ReasonCheckpoint
	body := html(t, in(e))

	if !strings.Contains(body, "Not a verdict") {
		t.Error("an unfinished run does not say it is progress rather than a verdict")
	}
}

// TestAnUnevaluatedGateIsVisible — an invisible gate reads as a gate that
// passed.
func TestAnUnevaluatedGateIsVisible(t *testing.T) {
	e := envelope("Passed",
		contract.TestResult{
			Name: "host-health", Kind: "host-health", Phase: "Passed", Nodes: []string{"n1"},
			NotEvaluated: []contract.NotEvaluated{{Metric: "eccErrors", Reason: "unmeasurable on this hardware"}},
			Unmeasurable: []string{"eccErrors"},
		},
	)
	body := html(t, in(e))

	if !strings.Contains(body, "Not evaluated: eccErrors") {
		t.Error("an unevaluated gate is not shown; it would read as a gate that passed")
	}
}

// TestADroppedArtifactIsSurfaced — a document that omits evidence silently is
// indistinguishable from a clean run.
func TestADroppedArtifactIsSurfaced(t *testing.T) {
	e := envelope("Passed",
		contract.TestResult{
			Name: "dcgm-diag", Kind: "dcgm-diag", Phase: "Passed", Nodes: []string{"n1"},
			Artifacts: []contract.ArtifactRef{{Name: "dcgmi.json", Dropped: "exceeded the per-artifact cap"}},
		},
	)
	body := html(t, in(e))

	if !strings.Contains(body, "dcgmi.json") || !strings.Contains(body, "not stored") {
		t.Error("a dropped artifact is not surfaced in the report")
	}
}

// TestAnAbsentTestIsNotAPass — a node that never ran a test must not read as
// having passed it.
func TestAnAbsentTestIsNotAPass(t *testing.T) {
	e := envelope("Passed",
		contract.TestResult{Name: "a", Kind: "gpu-burn", Phase: "Passed", Nodes: []string{"n1"}},
		contract.TestResult{Name: "b", Kind: "gpu-burn", Phase: "Passed", Nodes: []string{"n2"}},
	)
	body := html(t, in(e))

	if !strings.Contains(body, "not run on this node") {
		t.Error("a test that did not run on a node is not marked as absent")
	}
}

// TestUserContentIsEscaped — messages come from runner stdout, which is not
// trusted input for a document someone will open in a browser.
func TestUserContentIsEscaped(t *testing.T) {
	e := envelope("Failed",
		contract.TestResult{
			Name: "evil", Kind: "custom", Phase: "Failed", Nodes: []string{"n1"},
			Message: `<script>alert('xss')</script>`,
		},
	)
	body := html(t, in(e))

	if strings.Contains(body, "<script>alert") {
		t.Error("runner output was not escaped — a report is opened in a browser")
	}
	if !strings.Contains(body, "&lt;script&gt;") {
		t.Error("the message was dropped instead of escaped; a reader should still see what the runner said")
	}
}

// TestRenderIsDeterministic — the document is evidence; it must not change
// between renders of the same run.
func TestRenderIsDeterministic(t *testing.T) {
	e := envelope("Failed",
		contract.TestResult{Name: "b", Kind: "gpu-burn", Phase: "Passed", Nodes: []string{"n1"},
			Metrics: map[string]string{"z": "1", "a": "2", "m": "3"}},
		contract.TestResult{Name: "a", Kind: "gpu-burn", Phase: "Failed", Nodes: []string{"n1"}},
	)
	first := html(t, in(e))
	for i := 0; i < 5; i++ {
		if again := html(t, in(e)); again != first {
			t.Fatal("output is not byte-stable — map iteration is leaking into the document")
		}
	}
}

// TestTheDocumentCarriesProvenance.
func TestTheDocumentCarriesProvenance(t *testing.T) {
	e := envelope("Passed",
		contract.TestResult{Name: "a", Kind: "gpu-burn", Phase: "Passed", Nodes: []string{"n1"}},
	)
	body := html(t, in(e))

	if !strings.Contains(body, "glimmer-burnin") {
		t.Error("the document does not name what generated it")
	}
	if !strings.Contains(body, "authored by the site") {
		t.Error("the document does not say thresholds are site-authored — a pass is not a warranty")
	}
}

// TestThereIsAPrintStylesheet — this gets printed and attached to claims.
func TestThereIsAPrintStylesheet(t *testing.T) {
	e := envelope("Passed",
		contract.TestResult{Name: "a", Kind: "gpu-burn", Phase: "Passed", Nodes: []string{"n1"}},
	)
	body := html(t, in(e))

	if !strings.Contains(body, "@media print") {
		t.Error("no print stylesheet: the document is meant to be printed")
	}
	if !strings.Contains(body, "break-inside:avoid") {
		t.Error("nothing prevents a card breaking across a page")
	}
}

// TestInventoryDistinguishesUnknownFromEmpty.
func TestInventoryDistinguishesUnknownFromEmpty(t *testing.T) {
	e := envelope("Passed",
		contract.TestResult{Name: "a", Kind: "gpu-burn", Phase: "Passed", Nodes: []string{"n1"}},
	)
	body := html(t, in(e)) // no inventory supplied
	if !strings.Contains(body, "no inventory was collected") {
		t.Error("a node with no inventory should say so rather than render blank")
	}

	withInv := html(t, in(e, report.NodeInfo{
		Name: "n1", OSImage: "Ubuntu 24.04.1 LTS",
		GPUs: []report.GPUInfo{{Index: 0, Model: "NVIDIA GB10"}},
	}))
	if strings.Contains(withInv, "no inventory was collected") {
		t.Error("an inventory was supplied but the document says none was")
	}
	if !strings.Contains(withInv, "NVIDIA GB10") || !strings.Contains(withInv, "Ubuntu 24.04.1 LTS") {
		t.Error("supplied inventory is not shown")
	}
	if strings.Contains(withInv, "serial") {
		t.Error("an uncaptured serial was rendered — absent means absent")
	}
}

// TestATestNamedAfterItsKindIsNotPrintedTwice — most tests are named for their
// kind, and saying the same word twice is noise that trains a reader to skim.
func TestATestNamedAfterItsKindIsNotPrintedTwice(t *testing.T) {
	e := envelope("Passed",
		contract.TestResult{Name: "compute-smoke", Kind: "compute-smoke", Phase: "Passed", Nodes: []string{"n1"}},
		contract.TestResult{Name: "fp8-sweep", Kind: "gemm-sweep", Phase: "Passed", Nodes: []string{"n1"}},
	)
	body := html(t, in(e))

	// The matrix header names each test once, legitimately. The bug was in the
	// per-node detail, where the name and the kind sat next to each other.
	if strings.Contains(body, `<b>compute-smoke</b>`+"\n"+`          <span class="sub">compute-smoke</span>`) ||
		regexp.MustCompile(`<b>compute-smoke</b>\s*<span class="sub">compute-smoke</span>`).MatchString(body) {
		t.Error("a test named after its kind prints the same word twice in the per-node detail")
	}
	if !regexp.MustCompile(`<b>fp8-sweep</b>\s*<span class="sub">gemm-sweep</span>`).MatchString(body) {
		t.Error("when the name and kind differ, both should be shown — the kind is what actually ran")
	}
}
