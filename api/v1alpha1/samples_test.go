package v1alpha1_test

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/serializer/json"

	burninv1alpha1 "github.com/baldwinSPC/glimmer-burnin/api/v1alpha1"
	"github.com/baldwinSPC/glimmer-burnin/pkg/runnerimages"
	"github.com/baldwinSPC/glimmer-burnin/pkg/verdict"
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

// TestSamplesThresholdsLintClean runs the threshold linter over every gate the
// shipped samples declare, at the severity the operator itself applies.
//
// This is the cheap half of giving verdict.ValidateThresholds a surface: the
// reconciler warns the operators of a real cluster, and this warns us. Samples
// are the first thing a new user applies and the thing they copy, so a sample
// carrying a gate the operator would refuse — or one it would run while noting
// that the verdict may not mean what it says — ships that mistake to everyone
// who starts here.
//
// It is deliberately zero-tolerance, including for the ADVISORY severity that
// does not block a run: an unsound gate may be a defensible choice in somebody's
// profile, but it is never a defensible thing to hand a newcomer as an example.
//
// What it cannot check is the only thing that has actually gone wrong here: the
// NUMBER. A clockprobe gate at `>= 90` failed a healthy part by seven points and
// lints perfectly clean, because GreaterThanOrEqual is a sound comparison and
// calibration is measured, not linted. This is a floor, not a substitute for
// measure-then-pin.
func TestSamplesThresholdsLintClean(t *testing.T) {
	scheme := runtime.NewScheme()
	if err := burninv1alpha1.AddToScheme(scheme); err != nil {
		t.Fatalf("AddToScheme: %v", err)
	}
	decoder := json.NewSerializerWithOptions(
		json.DefaultMetaFactory, scheme, scheme,
		json.SerializerOptions{Yaml: true, Strict: true},
	)

	files, err := filepath.Glob(filepath.Join("..", "..", "config", "samples", "*.yaml"))
	if err != nil {
		t.Fatalf("glob samples: %v", err)
	}

	gated := 0
	for _, file := range files {
		raw, err := os.ReadFile(file)
		if err != nil {
			t.Fatalf("read %s: %v", file, err)
		}
		for _, doc := range splitYAMLDocuments(string(raw)) {
			obj, _, err := decoder.Decode([]byte(doc), nil, nil)
			if err != nil {
				continue // TestSamplesDecodeStrictly owns that failure.
			}
			bt, ok := obj.(*burninv1alpha1.BurnInTest)
			if !ok || len(bt.Spec.Thresholds) == 0 {
				continue
			}
			gated++
			// The kind-aware form, because these are OUR runners: a sample
			// gating a built-in kind on a metric no runner of that kind emits is
			// exactly the trap this repo has already fallen into once.
			for _, p := range verdict.ValidateThresholdsForKind(bt.Spec.Kind, bt.Spec.Thresholds) {
				t.Errorf("%s: BurnInTest %q (kind %s) %s",
					filepath.Base(file), bt.Name, bt.Spec.Kind, p.Error())
			}
		}
	}
	// A check that silently stops finding anything to check is not a check.
	if gated == 0 {
		t.Error("no sample declares a threshold — either the samples lost their gates or this test stopped finding them")
	}
}

// TestNoSampleRequestsADurationARunnerIgnores is issue #25's acceptance
// criterion, mechanised.
//
// The operator injects BURNIN_DURATION_SECONDS into every runner, and for a long
// time config/samples/node-acceptance.yaml asked compute-smoke for 120 seconds
// and got milliseconds — the runner has no duration loop and never did. Nothing
// was wrong at runtime; the sample was simply describing a burn-in that did not
// happen, and a sample is the first thing a new user copies.
//
// It is the DURATION counterpart to TestSamplesThresholdsLintClean above, and
// the two catch different things: the linter judges the gates a sample declares,
// and this judges a field the runner will silently ignore. Neither subsumes the
// other — a sample can lint perfectly clean while asking a burst kind to burn a
// node in for two minutes.
//
// The rule is narrow on purpose: it does not say a burst kind may never be given
// a duration, because DurationSeconds legitimately bounds the POD through the
// reconciler's deadline. It says a SAMPLE must not, because a sample is
// documentation, and a number a reader will take for a runtime budget must not
// appear next to a kind that does not use it as one.
func TestNoSampleRequestsADurationARunnerIgnores(t *testing.T) {
	forEachSampleTest(t, func(t *testing.T, file string, i int, spec burninv1alpha1.BurnInTestSpec) {
		if spec.Kind.BurstOnly() && spec.DurationSeconds != 0 {
			t.Errorf("%s document %d: kind %q is burst-only and does not use DurationSeconds as a "+
				"runtime budget, but this sample asks for %ds — a reader will take that for a burn-in "+
				"that never happens. Drop it, or use a duration-bearing kind",
				filepath.Base(file), i, spec.Kind, spec.DurationSeconds)
		}
	})
}

// TestNoSampleSegmentsASoakItCannotDivide is the segmentation counterpart of the
// guard above, and it is here for the same reason: a sample is documentation,
// and the reader copies it.
//
// The operator refuses every one of these at plan time, so nothing here protects
// a cluster. What it protects is the example — a shipped sample that the operator
// would refuse teaches a newcomer a shape that does not work, and they find out
// after cordoning a node.
func TestNoSampleSegmentsASoakItCannotDivide(t *testing.T) {
	forEachSampleTest(t, func(t *testing.T, file string, i int, spec burninv1alpha1.BurnInTestSpec) {
		if spec.Soak == nil {
			return
		}
		name := filepath.Base(file)
		if spec.Kind.BurstOnly() {
			t.Errorf("%s document %d: kind %q is burst-only — its runner ignores the duration entirely, "+
				"so segmenting it would run the same short measurement over and over and call it a soak",
				name, i, spec.Kind)
		}
		if spec.DurationSeconds <= 0 {
			t.Errorf("%s document %d: sets spec.soak but no durationSeconds — there is nothing to divide",
				name, i)
		} else if spec.Soak.SegmentSeconds > spec.DurationSeconds {
			t.Errorf("%s document %d: asks for %ds segments of a %ds soak — a segment longer than the "+
				"soak burns the fleet longer than it was asked to",
				name, i, spec.Soak.SegmentSeconds, spec.DurationSeconds)
		}
		if n := spec.RepeatCount; n != nil && *n > 1 {
			t.Errorf("%s document %d: sets both spec.soak and repeatCount %d — a repeat re-runs a whole "+
				"test and ANDs the verdicts, while a segment is one window of a single verdict",
				name, i, *n)
		}
	})
}

// TestSamplesOfAnUndefaultedKindNameAnImage holds every sample test to the
// honesty node-acceptance.yaml's own header describes: a kind in
// pkg/runnerimages.WithoutDefault() has no built-in image, so a BurnInTest
// naming one is a plan-time infrastructure Error — "no default image" —
// unless spec.runner.image or spec.runner.imagesByVendor says where to get
// one. That is a correct refusal on a freshly-applied sample, but a SILENT
// one: nothing before `kubectl apply` + a failed run told the author the
// pod would never schedule.
//
// A REPLACE-ME placeholder is enough to pass this guard — the point is not
// that the image resolves, it is that a profile copied from this repo fails
// VISIBLY at the placeholder (a string an author notices and has to act on)
// rather than at plan time with a message that reads like a hardware fault.
func TestSamplesOfAnUndefaultedKindNameAnImage(t *testing.T) {
	withoutDefault := map[burninv1alpha1.TestKind]bool{}
	for _, k := range runnerimages.WithoutDefault() {
		withoutDefault[k] = true
	}
	forEachSampleTest(t, func(t *testing.T, file string, i int, spec burninv1alpha1.BurnInTestSpec) {
		if !withoutDefault[spec.Kind] {
			return
		}
		named := spec.Runner != nil && (spec.Runner.Image != "" || len(spec.Runner.ImagesByVendor) > 0)
		if !named {
			t.Errorf("%s document %d: kind %q has no default runner image and this test sets neither "+
				"spec.runner.image nor spec.runner.imagesByVendor — it would refuse at plan time with "+
				"no warning before that; name an image (a REPLACE-ME placeholder is fine) so the failure "+
				"is visible before apply",
				filepath.Base(file), i, spec.Kind)
		}
	})
}

// forEachSampleTest decodes every BurnInTest document in config/samples and
// hands its spec to fn. Non-BurnInTest documents are skipped;
// TestSamplesDecodeStrictly is what asserts they all decode at all.
//
// The two tests above each build their own decoder inline. Consolidating them
// onto this helper is worth doing, but not in the change that introduced it.
func forEachSampleTest(t *testing.T, fn func(t *testing.T, file string, doc int, spec burninv1alpha1.BurnInTestSpec)) {
	t.Helper()
	scheme := runtime.NewScheme()
	if err := burninv1alpha1.AddToScheme(scheme); err != nil {
		t.Fatalf("AddToScheme: %v", err)
	}
	decoder := json.NewSerializerWithOptions(
		json.DefaultMetaFactory, scheme, scheme,
		json.SerializerOptions{Yaml: true, Strict: true},
	)

	files, err := filepath.Glob(filepath.Join("..", "..", "config", "samples", "*.yaml"))
	if err != nil {
		t.Fatalf("glob samples: %v", err)
	}
	seen := 0
	for _, file := range files {
		raw, err := os.ReadFile(file)
		if err != nil {
			t.Fatalf("read %s: %v", file, err)
		}
		for i, doc := range splitYAMLDocuments(string(raw)) {
			obj, _, err := decoder.Decode([]byte(doc), nil, nil)
			if err != nil {
				continue // TestSamplesDecodeStrictly reports this.
			}
			test, ok := obj.(*burninv1alpha1.BurnInTest)
			if !ok {
				continue
			}
			seen++
			fn(t, file, i+1, test.Spec)
		}
	}
	// Asserted rather than assumed: a glob or a decode that quietly stopped
	// matching would leave this sweep passing while checking nothing.
	if seen == 0 {
		t.Fatal("no BurnInTest documents found in config/samples; this guard is checking nothing")
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

// THE SAME BAR, FOR THE PLACE THIS RELEASE MOVED THE GATES TO.
//
// TestSamplesThresholdsLintClean above decodes only BurnInTest documents, so it
// never saw a threshold written inside a profile — and variants are both the
// newest and the most error-prone place to write one, because a variant's
// thresholds REPLACE rather than merge (#297).
//
// CLAUDE.md claims config/samples/ is held to the linter's bar in CI. It was
// not, for the newest surface. A sample gating a variant on an Evidence-only
// metric, or on an exact comparison against a continuous one, would ship to
// every newcomer who copied it with CI green — and samples are the thing people
// copy, which is the entire reason this guard exists.
//
// The testRef resolution is not incidental to that. A gate can only be linted
// against the KIND that will run it, so resolving the reference is the
// precondition — and doing it turns a dangling reference into a test failure
// rather than something a user discovers by applying the sample (#300).
func TestSampleProfileThresholdsLintCleanIncludingVariants(t *testing.T) {
	scheme := runtime.NewScheme()
	if err := burninv1alpha1.AddToScheme(scheme); err != nil {
		t.Fatalf("AddToScheme: %v", err)
	}
	decoder := json.NewSerializerWithOptions(
		json.DefaultMetaFactory, scheme, scheme,
		json.SerializerOptions{Yaml: true, Strict: true},
	)

	files, err := filepath.Glob(filepath.Join("..", "..", "config", "samples", "*.yaml"))
	if err != nil {
		t.Fatalf("glob samples: %v", err)
	}

	// Kinds by BurnInTest name, across every sample: a profile may reference a
	// test declared in another file, which is how the samples are organised.
	kinds := map[string]burninv1alpha1.TestKind{}
	type profileDoc struct {
		file string
		p    *burninv1alpha1.BurnInProfile
	}
	var profiles []profileDoc

	for _, file := range files {
		raw, err := os.ReadFile(file)
		if err != nil {
			t.Fatalf("read %s: %v", file, err)
		}
		for _, doc := range splitYAMLDocuments(string(raw)) {
			obj, _, err := decoder.Decode([]byte(doc), nil, nil)
			if err != nil {
				continue // TestSamplesDecodeStrictly owns that failure.
			}
			switch o := obj.(type) {
			case *burninv1alpha1.BurnInTest:
				kinds[o.Name] = o.Spec.Kind
			case *burninv1alpha1.BurnInProfile:
				profiles = append(profiles, profileDoc{file: filepath.Base(file), p: o})
			}
		}
	}

	gated, resolved := 0, 0
	for _, pd := range profiles {
		for i, pt := range pd.p.Spec.Tests {
			where := fmt.Sprintf("%s: BurnInProfile %q test[%d]", pd.file, pd.p.Name, i)

			// Which kind will run this entry?
			var kind burninv1alpha1.TestKind
			switch {
			case pt.Inline != nil:
				kind = pt.Inline.Kind
			case pt.TestRef != "":
				k, ok := kinds[pt.TestRef]
				if !ok {
					// The feature's worked example referenced two BurnInTests
					// that did not exist, so it could not run at all.
					t.Errorf("%s references testRef %q, and no BurnInTest of that name is declared in "+
						"config/samples/. A sample is the thing people copy; a dangling reference makes "+
						"the example unrunnable and is invisible until someone applies it",
						where, pt.TestRef)
					continue
				}
				kind, resolved = k, resolved+1
			default:
				t.Errorf("%s declares neither testRef nor inline", where)
				continue
			}

			// An inline entry carries its own spec, thresholds included. A
			// testRef entry's base thresholds live on the BurnInTest the first
			// linter already covers, so only the variants are new here.
			if pt.Inline != nil && len(pt.Inline.Thresholds) > 0 {
				gated++
				for _, p := range verdict.ValidateThresholdsForKind(kind, pt.Inline.Thresholds) {
					t.Errorf("%s inline (kind %s) %s", where, kind, p.Error())
				}
			}
			for j, v := range pt.Variants {
				if len(v.Thresholds) == 0 {
					continue
				}
				gated++
				for _, p := range verdict.ValidateThresholdsForKind(kind, v.Thresholds) {
					t.Errorf("%s variant[%d] %q (kind %s) %s", where, j, v.Name, kind, p.Error())
				}
			}
		}
	}

	// A check that silently stops finding anything to check is not a check —
	// and this one failed to find the variants for a whole release.
	if resolved == 0 {
		t.Error("no sample profile resolved a testRef; either the samples stopped using them or this " +
			"test stopped finding them")
	}
	if gated == 0 {
		t.Error("no sample profile declares a threshold on an entry or a variant; the gates moved, or " +
			"this test lost sight of them")
	}
}
