package report

import (
	"strings"
	"testing"
	"time"

	"github.com/baldwinSPC/glimmer-burnin/pkg/contract"
)

func at(s string) time.Time {
	t, err := time.Parse(time.RFC3339, s)
	if err != nil {
		panic(err)
	}
	return t
}

func ptr(t time.Time) *time.Time { return &t }

// env builds a minimal valid envelope for one run.
func env(reason contract.Reason, phase, deliveryID, sentAt string, results ...contract.TestResult) *contract.Envelope {
	return &contract.Envelope{
		Version:    contract.Version,
		DeliveryID: deliveryID,
		Reason:     reason,
		SentAt:     at(sentAt),
		Run:        contract.RunRef{Namespace: "burnin", Name: "acceptance", UID: "uid-1", Profile: "node-acceptance"},
		Phase:      phase,
		Results:    results,
	}
}

func TestResolveTakesItsVerdictFromTheTerminalDelivery(t *testing.T) {
	// A run delivers per-test completions as it goes, then a terminal phase
	// change. The terminal one is the verdict of record even though an earlier
	// delivery carried a different phase for the same test.
	early := env(contract.ReasonTestCompleted, PhaseRunning, "d1", "2026-08-06T10:00:00Z",
		contract.TestResult{Name: "compute-smoke", Kind: "compute-smoke", Phase: PhaseRunning, Nodes: []string{"n1"}},
	)
	final := env(contract.ReasonPhaseChanged, PhaseFailed, "d2", "2026-08-06T10:05:00Z",
		contract.TestResult{Name: "compute-smoke", Kind: "compute-smoke", Phase: PhaseFailed, Nodes: []string{"n1"}},
	)

	got, err := Resolve(Input{Envelopes: []*contract.Envelope{final, early}})
	if err != nil {
		t.Fatalf("Resolve: %v", err)
	}
	if got.Run.Phase != PhaseFailed {
		t.Errorf("run phase = %q, want %q", got.Run.Phase, PhaseFailed)
	}
	if !got.Run.Terminal {
		t.Error("Terminal = false, want true for a Failed run")
	}
	if len(got.Results) != 1 {
		t.Fatalf("got %d results, want 1 — the earlier delivery describes the same test and must not duplicate it", len(got.Results))
	}
	if got.Results[0].Phase != PhaseFailed {
		t.Errorf("result phase = %q, want the terminal delivery's %q", got.Results[0].Phase, PhaseFailed)
	}
	if got.Results[0].Partial {
		t.Error("Partial = true, want false: this result came from the authoritative delivery")
	}
}

func TestResolveKeepsAResultTheTerminalDeliveryDroppedAndSaysSo(t *testing.T) {
	// A test completed, but the terminal envelope does not carry it. Losing it
	// silently would understate what ran; including it unmarked would overstate
	// how authoritative it is.
	completed := env(contract.ReasonTestCompleted, PhaseRunning, "d1", "2026-08-06T10:00:00Z",
		contract.TestResult{Name: "host-health", Kind: "host-health", Phase: PhasePassed, Nodes: []string{"n1"}},
	)
	final := env(contract.ReasonPhaseChanged, PhasePassed, "d2", "2026-08-06T10:05:00Z",
		contract.TestResult{Name: "compute-smoke", Kind: "compute-smoke", Phase: PhasePassed, Nodes: []string{"n1"}},
	)

	got, err := Resolve(Input{Envelopes: []*contract.Envelope{completed, final}})
	if err != nil {
		t.Fatalf("Resolve: %v", err)
	}
	if len(got.Results) != 2 {
		t.Fatalf("got %d results, want 2", len(got.Results))
	}
	var hh ResultView
	for _, r := range got.Results {
		if r.Name == "host-health" {
			hh = r
		}
	}
	if !hh.Partial {
		t.Error("host-health.Partial = false, want true: it is known only from a non-authoritative delivery")
	}
	if !hasWarningContaining(got.Warnings, "host-health") {
		t.Errorf("warnings do not mention the partial result: %v", got.Warnings)
	}
}

func TestResolveWarnsWhenTheRunNeverReachedAVerdict(t *testing.T) {
	// A checkpoint from a soak in progress is a legitimate thing to render. It
	// is not a verdict, and the document must not read as one.
	cp := env(contract.ReasonCheckpoint, PhaseRunning, "d1", "2026-08-06T10:00:00Z",
		contract.TestResult{Name: "thermal-soak", Kind: "thermal-soak", Phase: PhaseRunning, Nodes: []string{"n1"}},
	)

	got, err := Resolve(Input{Envelopes: []*contract.Envelope{cp}})
	if err != nil {
		t.Fatalf("Resolve: %v", err)
	}
	if got.Run.Terminal {
		t.Error("Terminal = true for a Running run")
	}
	if !hasWarningContaining(got.Warnings, "no terminal delivery") {
		t.Errorf("want a warning that this is progress not a verdict, got %v", got.Warnings)
	}
}

func TestResolveRefusesEnvelopesFromDifferentRuns(t *testing.T) {
	// Rendering two runs as one document would be a confident false record.
	a := env(contract.ReasonPhaseChanged, PhasePassed, "d1", "2026-08-06T10:00:00Z")
	b := env(contract.ReasonPhaseChanged, PhasePassed, "d2", "2026-08-06T11:00:00Z")
	b.Run.UID = "uid-2"

	_, err := Resolve(Input{Envelopes: []*contract.Envelope{a, b}})
	if err == nil {
		t.Fatal("Resolve accepted envelopes from two different runs")
	}
	if !strings.Contains(err.Error(), "different runs") {
		t.Errorf("error should name the problem, got: %v", err)
	}
}

func TestResolveRefusesAnUnknownEnvelopeVersion(t *testing.T) {
	// The contract's own instruction is to reject rather than guess.
	e := env(contract.ReasonPhaseChanged, PhasePassed, "d1", "2026-08-06T10:00:00Z")
	e.Version = "burnin.glimmer.ai/v9"

	if _, err := Resolve(Input{Envelopes: []*contract.Envelope{e}}); err == nil {
		t.Fatal("Resolve accepted an unrecognised envelope version")
	}
}

func TestResolveRejectsAnEmptyInput(t *testing.T) {
	if _, err := Resolve(Input{}); err == nil {
		t.Fatal("Resolve accepted zero envelopes")
	}
}

func TestResolveMarksAMultiNodeResultAsALink(t *testing.T) {
	// A pair verdict is about the link. A renderer that attributes it to one
	// endpoint sends an engineer to replace the wrong part, so the view has to
	// carry the distinction explicitly.
	final := env(contract.ReasonPhaseChanged, PhaseFailed, "d1", "2026-08-06T10:00:00Z",
		contract.TestResult{Name: "ib-write-bw", Kind: "ib-write-bw", Scope: "Pair", Phase: PhaseFailed, Nodes: []string{"n1", "n2"}},
		contract.TestResult{Name: "host-health", Kind: "host-health", Scope: "Node", Phase: PhasePassed, Nodes: []string{"n1"}},
	)

	got, err := Resolve(Input{Envelopes: []*contract.Envelope{final}})
	if err != nil {
		t.Fatalf("Resolve: %v", err)
	}
	for _, r := range got.Results {
		want := r.Name == "ib-write-bw"
		if r.Link != want {
			t.Errorf("%s: Link = %v, want %v", r.Name, r.Link, want)
		}
	}
}

func TestResolveDerivesTheRunSpanFromWhenWorkHappened(t *testing.T) {
	// Not from delivery timestamps: when the tests ran, not when someone was
	// told about them.
	final := env(contract.ReasonPhaseChanged, PhasePassed, "d1", "2026-08-06T12:00:00Z",
		contract.TestResult{
			Name: "a", Phase: PhasePassed, Nodes: []string{"n1"},
			StartedAt: ptr(at("2026-08-06T10:00:00Z")), FinishedAt: ptr(at("2026-08-06T10:30:00Z")),
		},
		contract.TestResult{
			Name: "b", Phase: PhasePassed, Nodes: []string{"n1"},
			StartedAt: ptr(at("2026-08-06T10:30:00Z")), FinishedAt: ptr(at("2026-08-06T11:00:00Z")),
		},
	)

	got, err := Resolve(Input{Envelopes: []*contract.Envelope{final}})
	if err != nil {
		t.Fatalf("Resolve: %v", err)
	}
	if got.Run.StartedAt == nil || !got.Run.StartedAt.Equal(at("2026-08-06T10:00:00Z")) {
		t.Errorf("StartedAt = %v, want the earliest test start", got.Run.StartedAt)
	}
	if got.Run.FinishedAt == nil || !got.Run.FinishedAt.Equal(at("2026-08-06T11:00:00Z")) {
		t.Errorf("FinishedAt = %v, want the latest test finish", got.Run.FinishedAt)
	}
}

func TestResolveSurfacesADroppedArtifactRatherThanSwallowingIt(t *testing.T) {
	// The operator records a truncation instead of discarding silently. A report
	// that hides it is worse than one that says the evidence was too large.
	final := env(contract.ReasonPhaseChanged, PhasePassed, "d1", "2026-08-06T10:00:00Z",
		contract.TestResult{
			Name: "dcgm-diag", Phase: PhasePassed, Nodes: []string{"n1"},
			Artifacts: []contract.ArtifactRef{{Name: "diag.json", Dropped: "exceeded the per-artifact cap"}},
		},
	)

	got, err := Resolve(Input{Envelopes: []*contract.Envelope{final}})
	if err != nil {
		t.Fatalf("Resolve: %v", err)
	}
	if !hasWarningContaining(got.Warnings, "diag.json") {
		t.Errorf("a dropped artifact must be surfaced, got warnings %v", got.Warnings)
	}
}

func TestResolveFallsBackToTheFingerprintAndSaysItDegraded(t *testing.T) {
	final := env(contract.ReasonPhaseChanged, PhasePassed, "d1", "2026-08-06T10:00:00Z",
		contract.TestResult{Name: "a", Phase: PhasePassed, Nodes: []string{"spark-a"}},
	)
	// The real format, from captureFingerprint: space-joined key=value, and the
	// OS value contains spaces. Splitting on whitespace would yield "Ubuntu" —
	// not a shorter answer but a wrong one.
	final.Fingerprint = map[string]string{
		"spark-a": "kernel=6.11.0-1010-nvidia os=Ubuntu 24.04.1 LTS arch=arm64 nvidia.com/gpu.product=GB10",
	}

	got, err := Resolve(Input{Envelopes: []*contract.Envelope{final}})
	if err != nil {
		t.Fatalf("Resolve: %v", err)
	}
	if len(got.Nodes) != 1 || got.Nodes[0].Name != "spark-a" {
		t.Fatalf("nodes = %+v, want one recovered from the fingerprint", got.Nodes)
	}
	n := got.Nodes[0]
	if n.Kernel != "6.11.0-1010-nvidia" {
		t.Errorf("Kernel = %q, want the full version", n.Kernel)
	}
	if n.OSImage != "Ubuntu 24.04.1 LTS" {
		t.Errorf("OSImage = %q, want the whole value — a truncated OS in an acceptance report is a fabricated one", n.OSImage)
	}
	if n.Arch != "arm64" {
		t.Errorf("Arch = %q, want arm64 — the field after a space-containing value must still parse", n.Arch)
	}
	if !hasWarningContaining(got.Warnings, "fingerprint") {
		t.Errorf("degradation must be recorded, got %v", got.Warnings)
	}
	// The accelerator label is deliberately not promoted to a structured field;
	// it stays reachable through the raw descriptor.
	if !strings.Contains(got.Run.Fingerprint["spark-a"], "nvidia.com/gpu.product=GB10") {
		t.Error("the raw descriptor must survive so a renderer can show what was not promoted")
	}
}

func TestResolveSuppliedNodesWinOverTheFingerprint(t *testing.T) {
	final := env(contract.ReasonPhaseChanged, PhasePassed, "d1", "2026-08-06T10:00:00Z")
	final.Fingerprint = map[string]string{"spark-a": "kernel=6.11.0"}

	got, err := Resolve(Input{
		Envelopes: []*contract.Envelope{final},
		Nodes:     []NodeInfo{{Name: "spark-a", Kernel: "6.11.0", GPUs: []GPUInfo{{Model: "GB10", Vendor: "nvidia"}}}},
	})
	if err != nil {
		t.Fatalf("Resolve: %v", err)
	}
	if len(got.Nodes) != 1 || len(got.Nodes[0].GPUs) != 1 {
		t.Fatalf("supplied inventory was not used: %+v", got.Nodes)
	}
	if hasWarningContaining(got.Warnings, "fingerprint") {
		t.Error("warned about degradation when a full inventory was supplied")
	}
}

func TestResolveIsDeterministic(t *testing.T) {
	// Golden-fixture tests downstream are only meaningful if the view is stable
	// regardless of the order deliveries arrived in.
	a := env(contract.ReasonTestCompleted, PhaseRunning, "d1", "2026-08-06T10:00:00Z",
		contract.TestResult{Name: "zebra", Phase: PhasePassed, Nodes: []string{"n2"}},
	)
	b := env(contract.ReasonPhaseChanged, PhasePassed, "d2", "2026-08-06T10:05:00Z",
		contract.TestResult{Name: "alpha", Phase: PhasePassed, Nodes: []string{"n1"}},
		contract.TestResult{Name: "alpha", Phase: PhasePassed, Nodes: []string{"n2"}},
	)

	first, err := Resolve(Input{Envelopes: []*contract.Envelope{a, b}})
	if err != nil {
		t.Fatalf("Resolve: %v", err)
	}
	second, err := Resolve(Input{Envelopes: []*contract.Envelope{b, a}})
	if err != nil {
		t.Fatalf("Resolve: %v", err)
	}
	if len(first.Results) != len(second.Results) {
		t.Fatalf("result counts differ: %d and %d", len(first.Results), len(second.Results))
	}
	for i := range first.Results {
		if first.Results[i].Name != second.Results[i].Name {
			t.Fatalf("ordering depends on input order at %d: %q then %q",
				i, first.Results[i].Name, second.Results[i].Name)
		}
		if strings.Join(first.Results[i].Nodes, ",") != strings.Join(second.Results[i].Nodes, ",") {
			t.Fatalf("node ordering differs at %d", i)
		}
	}
}

func TestResolveCarriesTheBaselineFlagThrough(t *testing.T) {
	// A thresholdless sweep that renders as an acceptance is the fail-open the
	// flag exists to close, so it has to survive into the view renderers read.
	final := env(contract.ReasonPhaseChanged, PhasePassed, "d1", "2026-08-06T10:00:00Z")
	final.Baseline = true

	got, err := Resolve(Input{Envelopes: []*contract.Envelope{final}})
	if err != nil {
		t.Fatalf("Resolve: %v", err)
	}
	if !got.Run.Baseline {
		t.Error("Baseline was dropped: a measurement sweep would render as a certification")
	}
}

func hasWarningContaining(warnings []string, substr string) bool {
	for _, w := range warnings {
		if strings.Contains(w, substr) {
			return true
		}
	}
	return false
}
