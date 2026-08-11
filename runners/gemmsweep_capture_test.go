// SPDX-License-Identifier: Apache-2.0
// Copyright the Glimmer authors.

package runners

import (
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"testing"

	"github.com/baldwinSPC/glimmer-burnin/pkg/contract"
	"github.com/baldwinSPC/glimmer-burnin/pkg/runner"
)

// The seam between gemm-sweep and the operator, exercised against REAL stdout.
//
// It lives in the seam package rather than beside the runner because
// runners/gemm-sweep holds precision_test.cc, and `go build` refuses a package
// directory containing C++ sources unless cgo is in use — the same reason
// cxxtests_test.go is here. A Go file next to the .cc would break the build of
// the whole module.
//
// testdata/*.stdout was captured on 2026-08-10 from a GB10 DGX Spark
// (spark-85a9, driver 580.82.09, compute capability 12.1, sm_121a), one file
// per precision, from the image built out of this directory. Until then the
// five kernels had compiled and never executed — issue #265.
//
// Capturing the output is the point. A parser test written from a hand-typed
// sample proves the parser agrees with whoever typed the sample; running the
// real thing is what catches a key that is spelled differently from the one the
// registry expects, which is silently unthresholdable and shows up as a node
// nobody can accept for reasons nobody can see.

func loadCapture(t *testing.T, precision string) string {
	t.Helper()
	b, err := os.ReadFile(filepath.Join("gemm-sweep", "testdata", precision+".stdout"))
	if err != nil {
		t.Fatalf("reading the captured stdout: %v", err)
	}
	return string(b)
}

var precisions = []string{"fp4", "fp8", "bf16", "tf32", "fp64"}

func TestRealStdoutParsesToRegisteredMetricNames(t *testing.T) {
	for _, p := range precisions {
		t.Run(p, func(t *testing.T) {
			res := runner.Parse("gemm-sweep", loadCapture(t, p), 0)

			if res.Verdict != runner.VerdictPass {
				t.Fatalf("a clean capture parsed as %v", res.Verdict)
			}
			if len(res.Metrics) == 0 {
				t.Fatal("no metrics parsed from a real run")
			}

			for name := range res.Metrics {
				if _, known := contract.Lookup(name); !known {
					t.Errorf("%q is not in the metric registry — a threshold written against it "+
						"can never be evaluated, and nothing at runtime would say so", name)
				}
				if err := contract.ValidateMetricName(name); err != nil {
					t.Errorf("%q does not obey the name grammar: %v", name, err)
				}
			}

			// The runner emits snake_case; snakeToLowerCamel is what turns those
			// into the canonical names, with no alias-table entry needed. If
			// that ever stops holding, these are the keys a threshold is
			// written against and they would silently stop arriving.
			for _, want := range []string{"achievedTflops", "maxRelativeError", "nonfiniteCount", "gemmPrecision"} {
				if _, ok := res.Metrics[want]; !ok {
					t.Errorf("%s missing; got %v", want, res.Metrics)
				}
			}
			if got := res.Metrics["gemmPrecision"]; got != p {
				t.Errorf("gemmPrecision = %q, want %q — a cell that misreports which precision it "+
					"ran makes a sweep's results unattributable", got, p)
			}
		})
	}
}

func TestEveryPrecisionReproducedTheReferenceOnRealSilicon(t *testing.T) {
	// Correctness is the half of this test that is not about plumbing. The
	// operands are integer grid points chosen so the product is exactly
	// representable, so anything but fp4 must reproduce the reference EXACTLY —
	// fp4's 4-bit mantissa cannot, and its error is bounded instead.
	for _, p := range precisions {
		res := runner.Parse("gemm-sweep", loadCapture(t, p), 0)

		if got := res.Metrics["nonfiniteCount"]; got != "0" {
			t.Errorf("%s: nonfiniteCount = %q — a NaN or Inf in device output is silicon "+
				"producing garbage", p, got)
		}

		relErr, err := strconv.ParseFloat(res.Metrics["maxRelativeError"], 64)
		if err != nil {
			t.Fatalf("%s: maxRelativeError %q: %v", p, res.Metrics["maxRelativeError"], err)
		}
		if p == "fp4" {
			// Measured 0.00195695 on GB10, and compute-smoke reports 0.00196 on
			// the same part — the same block-scaled NVFP4 path, which is the
			// cross-check that this cell runs what it claims to.
			if relErr <= 0 || relErr > 0.01 {
				t.Errorf("fp4 relative error %v is outside the band a 4-bit mantissa produces", relErr)
			}
		} else if relErr != 0 {
			t.Errorf("%s: relative error %v, want exactly 0 — the operands are integer grid "+
				"points and this precision can represent their product exactly", p, relErr)
		}
	}
}

func TestTheThroughputLadderSeparatesTheInstructionPaths(t *testing.T) {
	// The reason a sweep exists at all: each precision is a different unit, and
	// if two cells reported the same throughput one of them did not run what it
	// said. Measured on GB10 — fp4 90.4, fp8 69.4, bf16 46.5, tf32 29.3,
	// fp64 0.321 TFLOP/s.
	tflops := map[string]float64{}
	for _, p := range precisions {
		res := runner.Parse("gemm-sweep", loadCapture(t, p), 0)
		v, err := strconv.ParseFloat(res.Metrics["achievedTflops"], 64)
		if err != nil {
			t.Fatalf("%s: achievedTflops %q: %v", p, res.Metrics["achievedTflops"], err)
		}
		if v <= 0 {
			t.Fatalf("%s: achievedTflops = %v", p, v)
		}
		tflops[p] = v
	}

	// Monotone down the ladder. Asserted as an ORDERING rather than against
	// absolute figures: the numbers are SKU-specific and this project pins
	// thresholds from measured baselines, so a fixture that hard-coded 90.4
	// would fail on the next part rather than finding a defect.
	order := []string{"fp4", "fp8", "bf16", "tf32", "fp64"}
	for i := 1; i < len(order); i++ {
		lo, hi := order[i], order[i-1]
		if tflops[lo] >= tflops[hi] {
			t.Errorf("%s (%.3f) is not slower than %s (%.3f) — two precisions reporting the same "+
				"throughput means one did not run the unit it claims",
				lo, tflops[lo], hi, tflops[hi])
		}
	}

	// The one gap that is a fact about the silicon rather than a preference:
	// fp64 on a consumer-class Blackwell runs on the CUDA cores at a small
	// fraction of the tensor-core rate. Measured ~91x below tf32. A part where
	// that collapsed to single digits would mean fp64 had been dispatched
	// somewhere it should not be, which is exactly what this cell exists to
	// notice.
	if ratio := tflops["tf32"] / tflops["fp64"]; ratio < 10 {
		t.Errorf("tf32/fp64 = %.1fx, want a large gap — fp64 is a CUDA-core path here and a "+
			"small ratio means it was not", ratio)
	}
}

func TestTheCaptureRecordsWhichSiliconItCameFrom(t *testing.T) {
	// A fixture with no provenance is a fixture nobody can re-derive. These
	// are evidence-class metrics and they are why a stored result can still be
	// attributed to a part months later.
	res := runner.Parse("gemm-sweep", loadCapture(t, "fp4"), 0)
	for _, want := range []string{"gpuName", "computeCap", "builtCudaArch", "gemmShape"} {
		if v, ok := res.Metrics[want]; !ok || strings.TrimSpace(v) == "" {
			t.Errorf("%s missing from the capture", want)
		}
	}
	if got := res.Metrics["builtCudaArch"]; got != "sm_121a" {
		t.Errorf("builtCudaArch = %q, want sm_121a — an 'a' target emits a cubin for GB10 only, "+
			"with no PTX fallback, which is what makes a pass proof that the real instruction "+
			"path ran rather than an emulated one", got)
	}
}
