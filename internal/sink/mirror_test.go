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
