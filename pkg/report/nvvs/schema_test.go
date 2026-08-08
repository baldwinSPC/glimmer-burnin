package nvvs

import (
	"encoding/json"
	"os"
	"path/filepath"
	"reflect"
	"regexp"
	"sort"
	"strings"
	"testing"

	"github.com/baldwinSPC/glimmer-burnin/pkg/contract"
	"github.com/baldwinSPC/glimmer-burnin/pkg/report"
)

// TestTheKeyNamesAreTheOnesTheVendorDefines makes the pinned constants
// load-bearing.
//
// The package claims its field names come from NVIDIA's own
// nvvs/include/NvvsJsonStrings.h. Without this test that claim is decorative:
// the constants would be dead code and a struct tag could be quietly retyped to
// something plausible-but-wrong, producing documents that look compatible and
// are not.
//
// This repository has a documented case of exactly that shape — a guard that
// passed because it iterated over nothing — so a constant nobody checks is not
// a pin.
func TestTheKeyNamesAreTheOnesTheVendorDefines(t *testing.T) {
	tagsOf := func(v any) map[string]bool {
		out := map[string]bool{}
		rt := reflect.TypeOf(v)
		for i := 0; i < rt.NumField(); i++ {
			tag := rt.Field(i).Tag.Get("json")
			name, _, _ := strings.Cut(tag, ",")
			if name != "" && name != "-" {
				out[name] = true
			}
		}
		return out
	}

	doc := tagsOf(Document{})
	if !doc[keyRoot] {
		t.Errorf("the root key is not %q — the whole document hangs off it", keyRoot)
	}

	// The tree matters as much as the names, and it is asserted separately
	// against a real capture in TestTheDocumentTreeMatchesARealCapture. Here we
	// only check each key is spelled the vendor's way, on whichever struct
	// carries it.
	diag := tagsOf(Diagnostic{})
	for _, want := range []string{keyVersion, keyCategories} {
		if !diag[want] {
			t.Errorf("Diagnostic has no field tagged %q", want)
		}
	}
	for _, want := range []string{
		keyRuntimeError, keyGlobalWarn, keyMetadata,
		keyEntityGroups, keyOverall, keyAuxData,
	} {
		if !doc[want] {
			t.Errorf("Document has no field tagged %q — these are ROOT siblings of %q, "+
				"not children of it", want, keyRoot)
		}
	}

	meta := tagsOf(Metadata{})
	for _, want := range []string{keyDriverVersion, keyGPUDeviceIDs, keyGPUSerials} {
		if !meta[want] {
			t.Errorf("Metadata has no field tagged %q", want)
		}
	}
}

// TestPhaseVocabularyMatchesTheAPI catches the cost of restating the phase
// strings instead of importing them.
//
// A phase the API grows and this package has not been taught renders as
// "Not Run" — the safe direction, but still a document that says a finished run
// did not finish. So the guard reads the API's own constants and requires every
// one to be known here.
func TestPhaseVocabularyMatchesTheAPI(t *testing.T) {
	src, err := os.ReadFile(filepath.Join("..", "..", "..", "api", "v1alpha1", "burninrun_types.go"))
	if err != nil {
		t.Fatalf("reading the API types: %v", err)
	}

	re := regexp.MustCompile(`(?m)^\s*Run[A-Za-z]+\s+RunPhase\s*=\s*"([A-Za-z]+)"`)
	matches := re.FindAllStringSubmatch(string(src), -1)
	if len(matches) == 0 {
		t.Fatal("found no RunPhase constants — the guard's pattern has gone stale, which is itself the bug")
	}

	known := map[string]bool{
		phasePending: true, phaseRunning: true, phasePassed: true,
		phaseFailed: true, phaseError: true, phaseSkipped: true, phaseCancelled: true,
	}
	for _, m := range matches {
		if !known[m[1]] {
			t.Errorf("the API defines phase %q and this renderer does not know it — "+
				"it would render as %q, so a finished run would read as unfinished", m[1], statusNotRun)
		}
	}
}

// TestEveryKnownPhaseMapsDeliberately pins the status mapping, including the
// one that matters most.
func TestEveryKnownPhaseMapsDeliberately(t *testing.T) {
	want := map[string]string{
		phasePassed:    statusPass,
		phaseFailed:    statusFail,
		phaseSkipped:   statusSkip,
		phaseError:     statusNotRun,
		phasePending:   statusNotRun,
		phaseRunning:   statusNotRun,
		phaseCancelled: statusNotRun,
	}
	for phase, expect := range want {
		if got := statusFor(phase); got != expect {
			t.Errorf("statusFor(%q) = %q, want %q", phase, got, expect)
		}
	}
	if statusFor("SomethingNobodyDefined") != statusNotRun {
		t.Error("an unrecognised phase must not be assumed to have passed")
	}
	if statusFor(phaseError) == statusFail {
		t.Fatal("Error maps to Fail — the distinction the verdict engine protects would be destroyed at the schema boundary")
	}
}

// TestTheNoticeSurvivesEncoding checks the disclaimer is actually in the bytes,
// not merely in a struct a renderer might forget to populate.
func TestTheNoticeSurvivesEncoding(t *testing.T) {
	b, err := json.Marshal(Document{AuxData: &AuxData{Notice: notice, Schema: schemaTag}})
	if err != nil {
		t.Fatalf("marshalling: %v", err)
	}
	body := string(b)
	if !strings.Contains(body, "NOT produced by") {
		t.Error("the disclaimer is not in the encoded document")
	}
	if !strings.Contains(body, "NVIDIA") {
		t.Error("the disclaimer does not name NVIDIA")
	}
}

// TestTheDocumentTreeMatchesARealCapture pins the SHAPE, not just the names.
//
// This package's own comment used to say the tree was "inferred and has NOT been
// checked against a real capture", and it was inferred wrong: metadata,
// entity_groups and "Overall Result" were nested INSIDE "DCGM Diagnostic", where
// a real consumer does not look for them.
//
// The capture needed to settle it was already in this repository. It is not a
// new fixture and it was not obtained from hardware for this test: it is
// `dcgm4GB10` in runners/dcgm-diag/diagjson_test.go, a genuine `dcgmi diag -j`
// document from this project's GB10 fleet under DCGM 4.2.3, which the dcgm-diag
// runner's own parser has been tested against since it was captured.
//
// The failure it prevents is silent and total. Every key name was already
// correct, so nothing looked wrong; a consumer reading the real tree simply
// found no metadata and no entity_groups at the root, and got a document with no
// driver version, no serials, and no entity inventory.
//
// It reads the capture out of that file rather than duplicating it, because a
// second copy is a copy that stops matching.
func TestTheDocumentTreeMatchesARealCapture(t *testing.T) {
	capture := realCapture(t)

	// What the vendor's own tool puts at the ROOT.
	var captureRoot map[string]json.RawMessage
	if err := json.Unmarshal(capture, &captureRoot); err != nil {
		t.Fatalf("the captured document does not parse: %v", err)
	}
	if _, ok := captureRoot[keyRoot]; !ok {
		t.Fatalf("the capture has no %q key, so this guard is reading the wrong "+
			"fixture and would pass whatever the renderer emitted", keyRoot)
	}

	// And what nests inside it. In the capture this is test_categories ALONE.
	var captureInner map[string]json.RawMessage
	if err := json.Unmarshal(captureRoot[keyRoot], &captureInner); err != nil {
		t.Fatal(err)
	}
	if len(captureInner) != 1 {
		t.Fatalf("the capture nests %d keys under %q (%v); this guard assumes exactly "+
			"one and needs rewriting rather than relaxing", len(captureInner), keyRoot,
			sortedKeysOf(captureInner))
	}

	// Now the same two questions, of a document WE emit.
	ours := emitSample(t)
	var ourRoot map[string]json.RawMessage
	if err := json.Unmarshal(ours, &ourRoot); err != nil {
		t.Fatal(err)
	}
	var ourInner map[string]json.RawMessage
	if err := json.Unmarshal(ourRoot[keyRoot], &ourInner); err != nil {
		t.Fatalf("our document has no parseable %q object: %v", keyRoot, err)
	}

	// Every key the capture places at the root, we must place at the root.
	for key := range captureRoot {
		if _, ok := ourRoot[key]; !ok {
			t.Errorf("the capture has %q at the ROOT and our document does not. A "+
				"consumer reading the real tree will not find it.", key)
		}
		if _, nested := ourInner[key]; nested {
			t.Errorf("we nest %q inside %q; the capture has it at the ROOT. Every key "+
				"name here is correct, so nothing looks wrong — the consumer simply "+
				"gets a document with no %s.", key, keyRoot, key)
		}
	}
	// And nothing the capture nests may escape to the root.
	for key := range captureInner {
		if _, ok := ourInner[key]; !ok {
			t.Errorf("the capture nests %q under %q and we do not", key, keyRoot)
		}
	}
}

// realCapture extracts the genuine dcgmi document from the dcgm-diag runner's
// own test file.
//
// Reading it out of the source rather than copying it means there is exactly one
// captured document in this repository. A second copy is a copy that stops
// matching, and the whole value of this guard is that it compares against
// something real.
func realCapture(t *testing.T) []byte {
	t.Helper()
	path := filepath.Join("..", "..", "..", "runners", "dcgm-diag", "diagjson_test.go")
	src, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("cannot read the captured document at %s: %v", path, err)
	}
	m := regexp.MustCompile("(?s)dcgm4GB10\\s*=\\s*`(\\{.*?)`").FindSubmatch(src)
	if m == nil {
		t.Fatalf("the dcgm4GB10 capture is no longer in %s under that name. This guard "+
			"depends on it; find where the capture moved rather than deleting the guard.", path)
	}
	return m[1]
}

// emitSample renders one document through the real renderer.
func emitSample(t *testing.T) []byte {
	t.Helper()
	e := envelope(phasePassed,
		contract.TestResult{Name: "a", Kind: "compute-smoke", Phase: phasePassed, Nodes: []string{"n1"}},
	)
	in := input(e)
	in.Nodes = []report.NodeInfo{{Name: "n1", GPUs: []report.GPUInfo{{Index: 0, Model: "NVIDIA GB10"}}}}
	outs, err := Renderer{}.Render(in)
	if err != nil {
		t.Fatal(err)
	}
	for _, o := range outs {
		if strings.HasSuffix(o.Filename, ".json") {
			return o.Data
		}
	}
	t.Fatal("the renderer emitted no JSON document")
	return nil
}

func sortedKeysOf(m map[string]json.RawMessage) []string {
	out := make([]string, 0, len(m))
	for k := range m {
		out = append(out, k)
	}
	sort.Strings(out)
	return out
}
