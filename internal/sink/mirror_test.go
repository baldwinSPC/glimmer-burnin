package sink

import (
	"reflect"
	"testing"

	burninv1alpha1 "github.com/baldwinSPC/glimmer-burnin/api/v1alpha1"
	"github.com/baldwinSPC/glimmer-burnin/pkg/contract"
	"github.com/baldwinSPC/glimmer-burnin/pkg/verdict"
)

// Violation and NotEvaluated exist THREE times, and this is what keeps them in
// step — issue #139.
//
// The duplication is forced and worth stating so nobody "fixes" it by importing
// across the boundary:
//
//	pkg/verdict     — must not import api/v1alpha1; that would be a cycle,
//	                  because the API package is what verdict evaluates FOR.
//	api/v1alpha1    — the CRD's own shape, with kubebuilder markers.
//	pkg/contract    — must stay free of Kubernetes types, so a consumer can
//	                  decode a verdict without depending on client-go.
//
// So the compiler cannot help, and a field added to one and forgotten in the
// others is silently dropped somewhere between the evaluator and the consumer —
// which is exactly how a violation's Cause could exist inside the cluster and
// never reach the thing gating on it.
func TestMirroredStructsAgree(t *testing.T) {
	t.Run("Violation", func(t *testing.T) {
		assertSameJSONShape(t,
			reflect.TypeOf(burninv1alpha1.Violation{}),
			reflect.TypeOf(contract.Violation{}))

		// pkg/verdict's is the source: it has no JSON tags (it never crosses a
		// wire) so it is compared by FIELD NAME.
		assertSameFieldNames(t,
			reflect.TypeOf(verdict.Violation{}),
			reflect.TypeOf(burninv1alpha1.Violation{}),
			// verdict carries the typed enums; the API and the envelope carry
			// their string spellings, which is the crossing violationsFor makes.
			map[string]bool{})
	})

	// ArtifactRef exists twice, not three times: pkg/verdict never sees one,
	// because an artifact is evidence ABOUT a verdict and never an input to it.
	t.Run("ArtifactRef", func(t *testing.T) {
		assertSameJSONShape(t,
			reflect.TypeOf(burninv1alpha1.ArtifactRef{}),
			reflect.TypeOf(contract.ArtifactRef{}))
	})

	t.Run("NotEvaluated", func(t *testing.T) {
		assertSameJSONShape(t,
			reflect.TypeOf(burninv1alpha1.NotEvaluated{}),
			reflect.TypeOf(contract.NotEvaluated{}))
		assertSameFieldNames(t,
			reflect.TypeOf(verdict.NotEvaluated{}),
			reflect.TypeOf(burninv1alpha1.NotEvaluated{}),
			map[string]bool{})
	})
}

// assertSameJSONShape compares the two types a CONSUMER sees: the CRD status and
// the delivered envelope. Their JSON names must match exactly, or a field
// written to status silently fails to appear in the document.
func assertSameJSONShape(t *testing.T, a, b reflect.Type) {
	t.Helper()
	an, bn := jsonNames(a), jsonNames(b)
	for name := range an {
		if !bn[name] {
			t.Errorf("%s has JSON field %q and %s does not — it is recorded in the cluster "+
				"and never reaches a consumer", a.Name(), name, b.Name())
		}
	}
	for name := range bn {
		if !an[name] {
			t.Errorf("%s has JSON field %q and %s does not — the envelope promises a field "+
				"nothing populates", b.Name(), name, a.Name())
		}
	}
}

func assertSameFieldNames(t *testing.T, a, b reflect.Type, ignore map[string]bool) {
	t.Helper()
	an, bn := fieldNames(a), fieldNames(b)
	for name := range an {
		if !bn[name] && !ignore[name] {
			t.Errorf("%s.%s has no counterpart in %s — the evaluator produces something the "+
				"stored verdict cannot record", a.Name(), name, b.Name())
		}
	}
	for name := range bn {
		if !an[name] && !ignore[name] {
			t.Errorf("%s.%s has no counterpart in %s — the verdict records something no "+
				"evaluator produces", b.Name(), name, a.Name())
		}
	}
}

func jsonNames(t reflect.Type) map[string]bool {
	out := map[string]bool{}
	for i := 0; i < t.NumField(); i++ {
		tag := t.Field(i).Tag.Get("json")
		name := tag
		for j := 0; j < len(tag); j++ {
			if tag[j] == ',' {
				name = tag[:j]
				break
			}
		}
		if name == "" || name == "-" {
			name = t.Field(i).Name
		}
		out[name] = true
	}
	return out
}

func fieldNames(t reflect.Type) map[string]bool {
	out := map[string]bool{}
	for i := 0; i < t.NumField(); i++ {
		out[t.Field(i).Name] = true
	}
	return out
}

// THE WORDS MUST AGREE, not just the shapes (#295).
//
// TestMirroredStructsAgree above proves the three structs carry the same
// FIELDS. This proves the VALUES that travel in them mean the same thing in
// every package — which is the half that decides whether an engineer is
// dispatched.
//
// pkg/contract now exports Cause, ViolationKind and Phase so a consumer can
// switch on them without hard-coding "Measurement" or importing the producer.
// The risk that creates is a new one: three packages that each spell the
// vocabulary, drifting silently. api/v1alpha1's RunPhase is deliberately NOT an
// alias — it is the CRD's own enum with its own kubebuilder markers — so this
// is the only thing holding the two together.
func TestTheVerdictVocabularyAgreesAcrossPackages(t *testing.T) {
	// Every phase the CRD can record must have a contract spelling, or the
	// envelope would carry a phase no consumer's switch has a case for.
	apiPhases := []burninv1alpha1.RunPhase{
		burninv1alpha1.RunPending, burninv1alpha1.RunRunning, burninv1alpha1.RunPassed,
		burninv1alpha1.RunFailed, burninv1alpha1.RunError, burninv1alpha1.RunSkipped,
		burninv1alpha1.RunCancelled,
	}
	inContract := map[string]bool{}
	for _, p := range contract.Phases {
		inContract[string(p)] = true
	}
	for _, p := range apiPhases {
		if !inContract[string(p)] {
			t.Errorf("api/v1alpha1 can record phase %q and pkg/contract has no constant for it — a "+
				"consumer switching on contract.Phase has no case for a phase the operator emits", p)
		}
	}
	if len(contract.Phases) != len(apiPhases) {
		t.Errorf("contract declares %d phases and the API has %d; the lists have drifted, and the "+
			"envelope's vocabulary is what a consumer builds a switch from",
			len(contract.Phases), len(apiPhases))
	}

	// verdict's names are aliases of contract's, so a mismatch here means
	// somebody re-declared one locally rather than aliasing it.
	if verdict.CauseMeasurement != contract.CauseMeasurement ||
		verdict.CauseEvidence != contract.CauseEvidence ||
		verdict.CauseAuthoring != contract.CauseAuthoring {
		t.Error("pkg/verdict's Cause constants are not pkg/contract's — they must be aliases, or the " +
			"evaluator and the document can disagree about who should act")
	}

	// Every kind maps to a cause, and the mapping lives in ONE place.
	for _, k := range contract.ViolationKinds {
		if c := k.Cause(); c != contract.CauseMeasurement &&
			c != contract.CauseEvidence && c != contract.CauseAuthoring {
			t.Errorf("kind %q maps to cause %q, which is not one of contract.Causes", k, c)
		}
	}

	// The open-world rule, asserted rather than assumed: a kind from a newer
	// version must classify as Evidence — unjudged — and never as Measurement,
	// which would dispatch someone to replace a working part over a word this
	// version did not recognise.
	if got := contract.ViolationKind("SomethingAVersionFromNextYearEmits").Cause(); got != contract.CauseEvidence {
		t.Errorf("an unrecognised kind classified as %q, want %q: erring towards a hardware verdict "+
			"on a word we do not understand is the one direction that condemns good hardware",
			got, contract.CauseEvidence)
	}
}
