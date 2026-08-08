package report

import (
	"encoding/json"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/baldwinSPC/glimmer-burnin/pkg/contract"
)

// pkg/report must not drag Kubernetes into a consumer that only wants to render
// a result — issue #144.
//
// This is asserted against the REAL build graph rather than by reading the
// imports, because the dependency that lands is never the one at the top of the
// file: pkg/verdict looks harmless and pulls in apimachinery, and any helper
// added later to "just reuse the CRD types" would too. A CI job that formats a
// burn-in result should not need client-go to do it.
func TestNoKubernetesDependency(t *testing.T) {
	// "./..." resolves against the test's working directory, which go test sets
	// to the package under test — so this covers pkg/report and every renderer
	// subpackage, and picks up a new one the day it is added.
	out, err := exec.Command("go", "list", "-deps", "./...").Output()
	if err != nil {
		t.Fatalf("go list: %v", err)
	}
	deps := strings.Split(strings.TrimSpace(string(out)), "\n")
	if len(deps) < 10 {
		t.Fatalf("go list returned %d dependencies, which cannot be right — this guard "+
			"is not reading the build graph and would pass no matter what was imported", len(deps))
	}
	// The contract IS a dependency, and seeing it proves the graph being read is
	// this package's rather than an empty one.
	var sawContract bool
	for _, dep := range deps {
		if strings.HasSuffix(dep, "/pkg/contract") {
			sawContract = true
		}
	}
	if !sawContract {
		t.Fatal("pkg/contract is not in the dependency list, so this guard is reading " +
			"the wrong build graph")
	}
	for _, dep := range deps {
		dep = strings.TrimSpace(dep)
		if strings.HasPrefix(dep, "k8s.io/") || strings.HasPrefix(dep, "sigs.k8s.io/") {
			t.Errorf("pkg/report depends on %s — the audience for this package is CI "+
				"jobs, a CLI and third-party ingest, which is exactly the audience that "+
				"must not be pulling in Kubernetes to format a table", dep)
		}
	}
}

func env(mutate func(*contract.Envelope)) *contract.Envelope {
	e := &contract.Envelope{
		Version:    contract.Version,
		DeliveryID: "d-1",
		Reason:     contract.ReasonPhaseChanged,
		Run:        contract.RunRef{Namespace: "burnin", Name: "run1", UID: "uid-1"},
		Phase:      "Passed",
		SentAt:     time.Unix(1750000000, 0).UTC(),
	}
	if mutate != nil {
		mutate(e)
	}
	return e
}

// The terminal delivery is the verdict of record, whatever order the pile
// arrived in.
func TestTerminalDeliveryIsTheVerdictOfRecord(t *testing.T) {
	in, err := Assemble([]*contract.Envelope{
		env(func(e *contract.Envelope) {
			e.DeliveryID, e.Reason, e.Phase, e.CheckpointSequence = "cp-3", contract.ReasonCheckpoint, "Running", 3
		}),
		env(func(e *contract.Envelope) { e.DeliveryID, e.Phase = "terminal", "Failed" }),
		env(func(e *contract.Envelope) {
			e.DeliveryID, e.Reason, e.Phase, e.CheckpointSequence = "cp-1", contract.ReasonCheckpoint, "Running", 1
		}),
		env(func(e *contract.Envelope) { e.DeliveryID, e.Reason = "tc-1", contract.ReasonTestCompleted }),
	})
	if err != nil {
		t.Fatal(err)
	}

	v, final := in.Verdict()
	if v.DeliveryID != "terminal" {
		t.Errorf("verdict = %q, want the terminal phase change — a report that took the "+
			"last-arrived delivery would present a checkpoint as a verdict", v.DeliveryID)
	}
	if !final {
		t.Error("a terminal Failed run was not reported as final")
	}

	cps := in.Checkpoints()
	if len(cps) != 2 || cps[0].CheckpointSequence != 1 || cps[1].CheckpointSequence != 3 {
		t.Errorf("checkpoints out of sequence: %+v — a soak's progress cannot be put "+
			"back together", cps)
	}
}

// An unfinished run still renders, but must never be presented AS a verdict.
func TestAnUnfinishedRunIsNotFinal(t *testing.T) {
	in, err := Assemble([]*contract.Envelope{
		env(func(e *contract.Envelope) {
			e.DeliveryID, e.Reason, e.Phase, e.CheckpointSequence = "cp-1", contract.ReasonCheckpoint, "Running", 1
		}),
	})
	if err != nil {
		t.Fatal(err)
	}
	if _, final := in.Verdict(); final {
		t.Error("a still-running run was reported as a final verdict")
	}
}

// At-least-once delivery means a results directory legitimately holds the same
// delivery twice.
func TestRetriedDeliveriesAreDeduplicated(t *testing.T) {
	same := func() *contract.Envelope {
		return env(func(e *contract.Envelope) {
			e.DeliveryID, e.Reason, e.Phase, e.CheckpointSequence = "cp-2", contract.ReasonCheckpoint, "Running", 2
		})
	}
	in, err := Assemble([]*contract.Envelope{same(), same(), same()})
	if err != nil {
		t.Fatal(err)
	}
	if got := len(in.Checkpoints()); got != 1 {
		t.Errorf("got %d checkpoints from 3 copies of one delivery, want 1 — a soak "+
			"would appear to progress %dx as fast as it did", got, got)
	}
}

// A report covers ONE run. Two runs in one document would be a verdict about
// hardware that was never tested together.
func TestTwoRunsAreRefused(t *testing.T) {
	_, err := Assemble([]*contract.Envelope{
		env(nil),
		env(func(e *contract.Envelope) { e.DeliveryID, e.Run.UID = "d-2", "uid-2" }),
	})
	if err == nil {
		t.Fatal("two runs were assembled into one report")
	}
	if !strings.Contains(err.Error(), "two different runs") {
		t.Errorf("error = %v, want it to name the problem", err)
	}
}

// A JSON file that is not an envelope is an ERROR, not a skip.
//
// Silently ignoring it is how a report ends up describing three of a run's four
// deliveries with nothing saying so, and a verdict assembled from part of the
// record is the one failure this package must not have.
func TestAJSONFileThatIsNotAnEnvelopeIsAnError(t *testing.T) {
	dir := t.TempDir()
	writeJSON(t, filepath.Join(dir, "good.json"), env(nil))
	if err := os.WriteFile(filepath.Join(dir, "other.json"), []byte(`{"hello":"world"}`), 0o600); err != nil {
		t.Fatal(err)
	}
	if _, err := LoadDir(dir); err == nil {
		t.Fatal("a JSON file that is not an envelope was silently skipped — a report " +
			"assembled from part of the record says nothing about the part it dropped")
	}
}

// Non-JSON files are skipped: a results directory legitimately holds artifacts,
// logs and a README beside the envelopes.
func TestNonJSONFilesAreSkipped(t *testing.T) {
	dir := t.TempDir()
	writeJSON(t, filepath.Join(dir, "nested", "terminal.json"), env(nil))
	for name, body := range map[string]string{
		"README.md":       "# results",
		"runner.log":      "busbw=15.9",
		"dcgmi.txt":       "not json",
		"artifact.tar.gz": "\x1f\x8b",
	} {
		if err := os.WriteFile(filepath.Join(dir, name), []byte(body), 0o600); err != nil {
			t.Fatal(err)
		}
	}
	in, err := LoadDir(dir)
	if err != nil {
		t.Fatalf("LoadDir: %v", err)
	}
	if len(in.Envelopes) != 1 {
		t.Errorf("got %d envelopes, want 1 from a nested directory", len(in.Envelopes))
	}
}

// A file may hold one envelope or an array of them; both shapes exist in the
// wild and a consumer should not have to know which it has.
func TestBothStoredShapesLoad(t *testing.T) {
	dir := t.TempDir()
	writeJSON(t, filepath.Join(dir, "list.json"), []*contract.Envelope{
		env(func(e *contract.Envelope) { e.DeliveryID, e.Reason = "a", contract.ReasonTestCompleted }),
		env(func(e *contract.Envelope) { e.DeliveryID = "b" }),
	})
	in, err := LoadDir(dir)
	if err != nil {
		t.Fatalf("LoadDir: %v", err)
	}
	if len(in.Envelopes) != 2 {
		t.Fatalf("got %d envelopes from an array file, want 2", len(in.Envelopes))
	}
	if v, _ := in.Verdict(); v.DeliveryID != "b" {
		t.Errorf("verdict = %q, want the terminal one", v.DeliveryID)
	}
}

// An envelope written by a NEWER operator must be reported as not fully
// understood, never silently narrowed.
//
// A `baseline` flag a reader never learned about would turn a measurement sweep
// into a certification — the exact bug #142 was filed for, one layer up.
func TestAnUnknownFieldIsRefusedRatherThanDropped(t *testing.T) {
	dir := t.TempDir()
	raw := map[string]any{
		"version": contract.Version, "deliveryId": "d-1", "reason": string(contract.ReasonPhaseChanged),
		"run":   map[string]any{"namespace": "burnin", "name": "run1", "uid": "uid-1"},
		"phase": "Passed", "sentAt": "2025-06-15T14:26:40Z",
		"fieldFromAFutureRelease": true,
	}
	writeJSON(t, filepath.Join(dir, "future.json"), raw)
	if _, err := LoadDir(dir); err == nil {
		t.Fatal("an envelope with a field this build does not understand loaded cleanly; " +
			"whatever that field said was dropped without a word")
	}
}

// A report never fabricates: a value that was not captured is ABSENT.
func TestFingerprintValueDistinguishesAbsentFromEmpty(t *testing.T) {
	fp := map[string]string{"gpu.model": "NVIDIA GB10", "gpu.serial": "", "gpu.arch": "   "}

	if v, ok := FingerprintValue(fp, "gpu.model"); !ok || v != "NVIDIA GB10" {
		t.Errorf("a captured value did not read back: %q %v", v, ok)
	}
	for _, key := range []string{"gpu.serial", "gpu.arch", "gpu.uuid"} {
		if _, ok := FingerprintValue(fp, key); ok {
			t.Errorf("%q reported as captured; an empty serial in an RMA is a claim "+
				"that the part has none", key)
		}
	}
}

// A caller that collected no inventory must be distinguishable from hardware
// that reported nothing.
func TestNoInventoryIsNotAnEmptyInventory(t *testing.T) {
	in := Input{Nodes: []NodeInfo{{Name: "spark-a"}}}

	if _, ok := in.NodeByName("spark-a"); !ok {
		t.Error("a node with an entry but no fields was reported as unknown")
	}
	if _, ok := in.NodeByName("spark-b"); ok {
		t.Error("a node with no entry at all was reported as known")
	}
}

func writeJSON(t *testing.T, path string, v any) {
	t.Helper()
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		t.Fatal(err)
	}
	data, err := json.Marshal(v)
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, data, 0o600); err != nil {
		t.Fatal(err)
	}
}
