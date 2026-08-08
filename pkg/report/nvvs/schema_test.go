package nvvs

import (
	"encoding/json"
	"os"
	"path/filepath"
	"reflect"
	"regexp"
	"strings"
	"testing"
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

	diag := tagsOf(Diagnostic{})
	for _, want := range []string{
		keyVersion, keyRuntimeError, keyGlobalWarn, keyMetadata,
		keyEntityGroups, keyCategories, keyOverall, keyAuxData,
	} {
		if !diag[want] {
			t.Errorf("Diagnostic has no field tagged %q", want)
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
	b, err := json.Marshal(Document{Diagnostic: Diagnostic{AuxData: &AuxData{Notice: notice, Schema: schemaTag}}})
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
