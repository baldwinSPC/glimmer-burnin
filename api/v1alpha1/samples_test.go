package v1alpha1_test

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/serializer/json"

	burninv1alpha1 "github.com/baldwinSPC/glimmer-burnin/api/v1alpha1"
)

// TestSamplesDecodeStrictly decodes every document in config/samples against
// the API types, with unknown and duplicate fields treated as errors.
//
// Samples are the first thing a new user applies, and a sample that the
// apiserver rejects — or worse, one it accepts while silently dropping a
// misspelled field — is a bad first impression that no other test catches. The
// CRDs are generated from these same Go types, so strict decoding here is a
// close proxy for `kubectl apply --validate=strict` without needing a cluster.
//
// It also asserts that every Kind used in a sample is registered in the scheme.
// A type that exists but was never added to SchemeBuilder.Register decodes to
// "no kind registered", which is exactly the failure mode of forgetting to
// register a newly added CRD.
func TestSamplesDecodeStrictly(t *testing.T) {
	scheme := runtime.NewScheme()
	if err := burninv1alpha1.AddToScheme(scheme); err != nil {
		t.Fatalf("AddToScheme: %v", err)
	}
	// Strict: an unknown field is an error rather than a silent drop. That is
	// the whole point — a threshold named "metrics:" instead of "metric:" must
	// fail the build, not quietly gate on nothing.
	decoder := json.NewSerializerWithOptions(
		json.DefaultMetaFactory, scheme, scheme,
		json.SerializerOptions{Yaml: true, Strict: true},
	)

	files, err := filepath.Glob(filepath.Join("..", "..", "config", "samples", "*.yaml"))
	if err != nil {
		t.Fatalf("glob samples: %v", err)
	}
	if len(files) == 0 {
		t.Fatal("no samples found — config/samples is the documented starting point and must not be empty")
	}

	for _, file := range files {
		t.Run(filepath.Base(file), func(t *testing.T) {
			raw, err := os.ReadFile(file)
			if err != nil {
				t.Fatalf("read: %v", err)
			}
			docs := splitYAMLDocuments(string(raw))
			if len(docs) == 0 {
				t.Fatal("sample file contains no documents")
			}
			for i, doc := range docs {
				obj, gvk, err := decoder.Decode([]byte(doc), nil, nil)
				if err != nil {
					t.Errorf("document %d does not decode: %v", i+1, err)
					continue
				}
				if gvk.Group != burninv1alpha1.GroupVersion.Group ||
					gvk.Version != burninv1alpha1.GroupVersion.Version {
					t.Errorf("document %d has unexpected apiVersion %s", i+1, gvk.GroupVersion())
				}
				if obj == nil {
					t.Errorf("document %d decoded to nil", i+1)
				}
			}
		})
	}
}

// splitYAMLDocuments splits a multi-document YAML stream on "---" separators,
// dropping documents that are only comments or whitespace. It is deliberately
// simple: these are the project's own samples, not arbitrary user input, and a
// "---" inside a block scalar would be a sample worth rewriting anyway.
func splitYAMLDocuments(s string) []string {
	var out []string
	for _, doc := range strings.Split(s, "\n---") {
		if hasContent(doc) {
			out = append(out, doc)
		}
	}
	return out
}

// hasContent reports whether a document has any line that is not blank and not
// a comment. Every sample here opens with a comment block, and decoding one of
// those as an object would fail on missing apiVersion rather than being skipped.
func hasContent(doc string) bool {
	for _, line := range strings.Split(doc, "\n") {
		line = strings.TrimSpace(line)
		if line != "" && !strings.HasPrefix(line, "#") {
			return true
		}
	}
	return false
}
