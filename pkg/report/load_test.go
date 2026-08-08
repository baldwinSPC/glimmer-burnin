package report

import (
	"encoding/json"
	"strings"
	"testing"
	"testing/fstest"

	"github.com/baldwinSPC/glimmer-burnin/pkg/contract"
)

func mustJSON(t *testing.T, e *contract.Envelope) []byte {
	t.Helper()
	b, err := json.Marshal(e)
	if err != nil {
		t.Fatalf("marshalling fixture: %v", err)
	}
	return b
}

func TestLoadEnvelopesReadsADirectoryTree(t *testing.T) {
	// The CLI writes results/<run>/envelopes/NNN-*.json, and a caller should be
	// able to point at either level.
	a := env(contract.ReasonTestCompleted, PhaseRunning, "d1", "2026-08-06T10:00:00Z")
	b := env(contract.ReasonPhaseChanged, PhasePassed, "d2", "2026-08-06T10:05:00Z")

	fsys := fstest.MapFS{
		"results/run-1/envelopes/001-TestCompleted-a.json": {Data: mustJSON(t, a)},
		"results/run-1/envelopes/002-RunPhaseChanged.json": {Data: mustJSON(t, b)},
		"results/run-1/run.json":                           {Data: []byte(`{"id":"run-1"}`)},
		"results/run-1/raw/compute-smoke.stdout.txt":       {Data: []byte("tflops=101.99\n")},
	}

	got, err := LoadEnvelopes(fsys, "results/run-1/envelopes")
	if err != nil {
		t.Fatalf("LoadEnvelopes: %v", err)
	}
	if len(got) != 2 {
		t.Fatalf("got %d envelopes, want 2", len(got))
	}
	if got[0].DeliveryID != "d1" || got[1].DeliveryID != "d2" {
		t.Errorf("envelopes not in name order: %q then %q", got[0].DeliveryID, got[1].DeliveryID)
	}
}

func TestLoadEnvelopesIgnoresNonJSONButNotBadJSON(t *testing.T) {
	// A stray text file is not an envelope and is not an error. A .json file
	// that does not parse IS an error — a report assembled from "the ones that
	// happened to load" has omissions nobody knows about, and the file most
	// likely to be malformed is the terminal delivery carrying the verdict.
	good := env(contract.ReasonPhaseChanged, PhasePassed, "d1", "2026-08-06T10:00:00Z")

	ok := fstest.MapFS{
		"e/001.json":  {Data: mustJSON(t, good)},
		"e/notes.txt": {Data: []byte("not an envelope")},
	}
	if _, err := LoadEnvelopes(ok, "e"); err != nil {
		t.Fatalf("a non-JSON file should be ignored, got %v", err)
	}

	bad := fstest.MapFS{
		"e/001.json": {Data: mustJSON(t, good)},
		"e/002.json": {Data: []byte(`{"version":"burnin.glimmer.ai/v1alpha1","deliveryId":`)},
	}
	if _, err := LoadEnvelopes(bad, "e"); err == nil {
		t.Fatal("a malformed envelope must be an error, not a silent omission")
	}
}

func TestLoadEnvelopesRejectsAnInvalidEnvelope(t *testing.T) {
	// Validation is the contract's own, so this package cannot drift into
	// accepting a document the rest of the system rejects.
	e := env(contract.ReasonPhaseChanged, PhasePassed, "d1", "2026-08-06T10:00:00Z")
	e.Run.UID = "" // name alone is not stable identity

	fsys := fstest.MapFS{"e/001.json": {Data: mustJSON(t, e)}}
	_, err := LoadEnvelopes(fsys, "e")
	if err == nil {
		t.Fatal("LoadEnvelopes accepted an envelope the contract rejects")
	}
	if !strings.Contains(err.Error(), "run.uid") {
		t.Errorf("error should name the offending field, got: %v", err)
	}
}

func TestLoadEnvelopesReportsAnEmptyDirectory(t *testing.T) {
	fsys := fstest.MapFS{"e/readme.txt": {Data: []byte("nothing here")}}
	_, err := LoadEnvelopes(fsys, "e")
	if err == nil {
		t.Fatal("an empty directory should be an error, not an empty report")
	}
}

func TestParseEnvelopeRoundTrips(t *testing.T) {
	want := env(contract.ReasonPhaseChanged, PhaseFailed, "d1", "2026-08-06T10:00:00Z",
		contract.TestResult{
			Name: "compute-smoke", Kind: "compute-smoke", Phase: PhaseFailed, Nodes: []string{"n1"},
			Metrics:      map[string]string{"throughputTflops": "101.99"},
			Violations:   []contract.Violation{{Index: 0, Metric: "throughputTflops", Cause: CauseMeasurement, Kind: "Unsatisfied"}},
			NotEvaluated: []contract.NotEvaluated{{Metric: "eccErrors", Reason: "unmeasurable on this hardware"}},
			Unmeasurable: []string{"eccErrors"},
		},
	)

	got, err := ParseEnvelope(mustJSON(t, want))
	if err != nil {
		t.Fatalf("ParseEnvelope: %v", err)
	}
	if len(got.Results) != 1 {
		t.Fatalf("got %d results, want 1", len(got.Results))
	}
	r := got.Results[0]
	if len(r.Violations) != 1 || r.Violations[0].Cause != CauseMeasurement {
		t.Errorf("violations did not survive the round trip: %+v", r.Violations)
	}
	if len(r.NotEvaluated) != 1 || len(r.Unmeasurable) != 1 {
		t.Errorf("not-evaluated/unmeasurable did not survive: %+v / %v", r.NotEvaluated, r.Unmeasurable)
	}
}
