package controller

import (
	"bytes"
	"compress/gzip"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"strings"
	"testing"

	burninv1alpha1 "github.com/baldwinSPC/glimmer-burnin/api/v1alpha1"
)

// The pinned plan can hold a matrix — issue #156.
//
// The plan is what makes a run hermetic, and that is not traded away here: it is
// still one annotation, still written in the same write as the finalizer, still
// the thing every later reconcile reads. What changes is only how it is encoded
// when it would not otherwise fit.

// bigPlan builds a plan large enough to need compressing, out of the input this
// is actually for: many nearly-identical variant cells.
func bigPlan(cells int) *plan {
	p := &plan{Version: 1, Targets: []string{"spark-a", "spark-b"}, ProfileEntries: 1}
	for i := 0; i < cells; i++ {
		p.Tests = append(p.Tests, plannedTest{
			Name:     fmt.Sprintf("gemm-cell-%03d", i),
			Required: true,
			Parent:   "gemm",
			Axes:     map[string]string{"precision": fmt.Sprintf("p%03d", i)},
			Spec: burninv1alpha1.BurnInTestSpec{
				Kind:            burninv1alpha1.KindComputeSmoke,
				Scope:           burninv1alpha1.ScopeNode,
				DurationSeconds: 120,
				Runner: &burninv1alpha1.RunnerSpec{
					Image: "ghcr.io/baldwinspc/glimmer-burnin-compute-smoke:v0.5.0",
					Args:  []string{"--shape", "8192x8192x8192", "--iters", "64", "--check"},
				},
				Thresholds: []burninv1alpha1.Threshold{
					{Metric: "nonfiniteCount", Comparison: burninv1alpha1.EQ, Value: "0"},
					{Metric: "maxRelativeError", Comparison: burninv1alpha1.LTE, Value: "0.001"},
					{Metric: "achievedTflops", Comparison: burninv1alpha1.GTE, Value: "700"},
				},
			},
		})
	}
	return p
}

// A plan too large to store raw is compressed, and round-trips exactly.
func TestAnOversizedPlanIsCompressedAndRoundTripsExactly(t *testing.T) {
	p := bigPlan(400)

	raw, err := json.Marshal(p)
	if err != nil {
		t.Fatal(err)
	}
	if len(raw) <= maxPlanBytes {
		t.Fatalf("the fixture is only %d bytes and fits raw, so this test is not "+
			"exercising compression at all", len(raw))
	}

	run := newRun("r", "prof", "spark-a")
	if err := pinPlan(run, p); err != nil {
		t.Fatalf("pinPlan: %v", err)
	}
	stored := run.Annotations[planAnnotation]
	if len(stored) > maxPlanBytes {
		t.Fatalf("stored %d bytes, over the %d annotation budget", len(stored), maxPlanBytes)
	}
	if isRawJSONPlan(stored) {
		t.Error("an oversized plan was stored raw")
	}

	got, ok, err := loadPlan(run)
	if err != nil || !ok {
		t.Fatalf("loadPlan: ok=%v err=%v", ok, err)
	}
	// Exactly: a plan that round-trips approximately is a run executing
	// something nobody wrote.
	back, err := json.Marshal(got)
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(raw, back) {
		t.Errorf("the plan did not survive the round trip intact\n got %d bytes\nwant %d",
			len(back), len(raw))
	}
}

// A plan that fits stays raw, so `kubectl get -o yaml` shows something readable.
func TestASmallPlanStaysReadable(t *testing.T) {
	run := newRun("r", "prof", "spark-a")
	if err := pinPlan(run, bigPlan(2)); err != nil {
		t.Fatal(err)
	}
	stored := run.Annotations[planAnnotation]
	if !isRawJSONPlan(stored) {
		t.Fatal("a small plan was compressed; a human reading the object with kubectl " +
			"now sees base64 for no benefit")
	}
	if !strings.Contains(stored, "gemm-cell-000") {
		t.Error("the raw plan is not readable")
	}
}

// A plan pinned by the PREVIOUS controller version still reads.
//
// Written by hand as raw JSON, the way the old code stored it, because the point
// is that no migration and no flag day is needed: a run already in flight when
// the manager is upgraded must keep executing the plan it was pinned with.
func TestALegacyRawPlanStillReads(t *testing.T) {
	const legacy = `{"version":1,"targets":["spark-a"],"tests":[{"name":"fp4","required":true,` +
		`"spec":{"kind":"ComputeSmoke","scope":"Node","durationSeconds":60}}]}`

	run := newRun("r", "prof", "spark-a")
	run.Annotations = map[string]string{planAnnotation: legacy}

	p, ok, err := loadPlan(run)
	if err != nil || !ok {
		t.Fatalf("a plan written by the previous controller does not load: ok=%v err=%v", ok, err)
	}
	if len(p.Tests) != 1 || p.Tests[0].Name != "fp4" {
		t.Fatalf("legacy plan decoded wrongly: %+v", p.Tests)
	}
	if p.Targets[0] != "spark-a" {
		t.Errorf("targets = %v", p.Targets)
	}
}

// A compressed annotation may not expand without bound.
func TestTheDecompressedSizeIsCapped(t *testing.T) {
	// Highly compressible, and far over the cap: the shape of a hostile or
	// corrupt annotation.
	var buf bytes.Buffer
	zw := gzip.NewWriter(&buf)
	chunk := bytes.Repeat([]byte("A"), 1<<20)
	for written := 0; written < maxDecompressedPlanBytes+(2<<20); written += len(chunk) {
		if _, err := zw.Write(chunk); err != nil {
			t.Fatal(err)
		}
	}
	if err := zw.Close(); err != nil {
		t.Fatal(err)
	}
	bomb := base64.StdEncoding.EncodeToString(buf.Bytes())
	if len(bomb) > maxPlanBytes {
		t.Fatalf("the fixture is %d bytes stored, so the apiserver would have refused "+
			"it and this test is not exercising the cap", len(bomb))
	}

	run := newRun("r", "prof", "spark-a")
	run.Annotations = map[string]string{planAnnotation: bomb}

	_, ok, err := loadPlan(run)
	if !ok {
		t.Fatal("the annotation was reported absent")
	}
	if err == nil {
		t.Fatal("an annotation decompressing past the cap was expanded anyway. A " +
			"malformed or hostile plan takes the manager down for every run in the " +
			"cluster, not only the one carrying it.")
	}
	if !strings.Contains(err.Error(), "refusing to expand") {
		t.Errorf("error = %v, want it to name the cap", err)
	}
}

// Garbage in the annotation is an error, never a silently empty plan.
func TestAnUndecodableAnnotationIsAnError(t *testing.T) {
	for name, value := range map[string]string{
		"not base64":      "!!!! not base64 !!!!",
		"base64 not gzip": base64.StdEncoding.EncodeToString([]byte("hello")),
		"truncated json":  `{"version":1,"tests":[`,
	} {
		t.Run(name, func(t *testing.T) {
			run := newRun("r", "prof", "spark-a")
			run.Annotations = map[string]string{planAnnotation: value}
			if _, _, err := loadPlan(run); err == nil {
				t.Fatal("an undecodable plan loaded cleanly; the run would execute an " +
					"empty plan and report every node as having passed nothing")
			}
		})
	}
}

// A plan too large even compressed is refused, and says both numbers.
func TestAPlanTooLargeEvenCompressedIsRefused(t *testing.T) {
	// Incompressible, so gzip cannot rescue it.
	p := &plan{Version: 1, Targets: []string{"spark-a"}}
	for i := 0; i < 4000; i++ {
		p.Tests = append(p.Tests, plannedTest{
			Name:     fmt.Sprintf("t-%d-%s", i, randomish(i)),
			Required: true,
			Spec: burninv1alpha1.BurnInTestSpec{
				Kind:  burninv1alpha1.KindCustom,
				Scope: burninv1alpha1.ScopeNode,
				Runner: &burninv1alpha1.RunnerSpec{
					Image: "example.com/" + randomish(i*7) + ":" + randomish(i*13),
					Args:  []string{randomish(i * 3), randomish(i * 5), randomish(i * 11)},
				},
			},
		})
	}
	run := newRun("r", "prof", "spark-a")
	err := pinPlan(run, p)
	if err == nil {
		t.Fatal("a plan that does not fit even compressed was pinned anyway")
	}
	if !strings.Contains(err.Error(), "compressed") {
		t.Errorf("error = %v, want it to give both the raw and the compressed size so "+
			"an author can tell whether trimming would help", err)
	}
}

// randomish produces incompressible-ish text without a random source, which
// workflow scripts and deterministic tests both need.
func randomish(seed int) string {
	const alphabet = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
	x := uint64(seed)*6364136223846793005 + 1442695040888963407
	var b strings.Builder
	for i := 0; i < 24; i++ {
		x = x*6364136223846793005 + 1442695040888963407
		b.WriteByte(alphabet[(x>>33)%uint64(len(alphabet))])
	}
	return b.String()
}
