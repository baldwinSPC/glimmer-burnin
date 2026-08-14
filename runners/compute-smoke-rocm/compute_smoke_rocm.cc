// compute_smoke_rocm.cc — arch-correct matrix-core GEMM smoke test for AMD
// accelerators: the AMD runner image for the "compute-smoke" TestKind,
// selected per node via spec.runner.imagesByVendor ({vendor: amd}).
//
// SPDX-License-Identifier: Apache-2.0
// Copyright the Glimmer authors.
//
// This file is original work licensed under Apache-2.0. It links only the ROCm
// HIP runtime (MIT) and uses the compiler's own gfx11 WMMA builtins; see
// wmma_tile.h for why rocWMMA is NOT used (it refuses to compile for gfx1151,
// the hardware this runner exists for).
//
// WHAT THIS PROVES, AND WHAT IT DELIBERATELY DOES NOT
// ---------------------------------------------------
// One claim: THE MATRIX-CORE PATH EXECUTED ON THIS PART AND PRODUCED THE RIGHT
// ANSWER. That is a correctness statement about silicon, and it is worth
// exactly as much as the narrowness of the arch gate around it — which is why
// gfx_gate.h asks three separate questions before a kernel is launched and why
// the image carries no fallback path. HIP has no PTX-equivalent JIT, so a part
// absent from the offload list has NO code in this image at all; if the kernel
// ran, it ran on that part's real matrix instructions.
//
// It is NOT a soak. The kind is declared BurstOnly, so this runner reads no
// duration budget at all, and runners/pins_test.go enforces that in both
// directions — which is why the variable's name appears in this runner's
// README and deliberately nowhere in its source: the guard scans source, and a
// mention in a comment is indistinguishable from a read. Holding the same GEMM
// in a loop would not strengthen a correctness claim, and it would be a worse
// soak than the kinds that exist to be soaks. Pair it with thermal-soak in a
// profile that wants both.
//
// WHY BF16 AND FP16 BOTH
// ----------------------
// They are different data paths through the same units, and a part can pass one
// and fail the other — the same reasoning that makes gemm-sweep a sweep rather
// than one number. bf16 is what inference on this hardware actually runs, and
// fp16 is the older path that more software still reaches for; measuring one
// and reporting "the matrix cores work" would be a claim about the other that
// nobody made.
//
// Both precisions are gated INDEPENDENTLY and both must pass. A partial pass is
// a FAIL with the failing precision named, never a pass with a footnote.
//
// WHY THE INPUTS ARE FIXED AND EXACTLY REPRESENTABLE
// -------------------------------------------------
// Every input is a multiple of 0.25 under magnitude 8 (see exactInputValue), so
// it is exact in bf16's 8 significand bits and fp16's 11. The tolerance then
// gates the HARDWARE rather than the format: a wrong answer here is the part,
// not rounding. Random inputs would put the format's error budget inside the
// gate and force a tolerance loose enough to hide real faults.
//
// OUTPUT CONTRACT
//   metrics as key=value lines, ALWAYS printed before the decision, then one of
//     WMMA_GEMM_PASS               exit 0   the matrix cores computed correctly
//     WMMA_GEMM_FAIL:  <why>       exit 1   we MEASURED the part and it is wrong
//     WMMA_GEMM_SKIP:  <why>       exit 2   this hardware is out of scope
//     WMMA_GEMM_ERROR: <why>       exit 3   we could not measure; UNJUDGED

#include <hip/hip_runtime.h>

#include <chrono>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "gfx_gate.h"
#include "wmma_tile.h"

using burnin::kExitError;
using burnin::kExitFail;
using burnin::kExitPass;
using burnin::kExitSkip;

namespace {

// One 16x16x16 tile: the fragment shape both WMMA (RDNA3+) and MFMA (CDNA)
// implement, which is what lets one source serve both families.
constexpr int kM = 16;
constexpr int kN = 16;
constexpr int kK = 16;

// kBuiltTargets is the offload list baked in by the Dockerfile, and
// kBuiltFamilies its generations. Both are compile-time strings so the binary
// can answer "what was I built for" without being told at runtime — the
// question archMatch and kernelCovers exist to ask.
#ifndef BURNIN_BUILT_TARGETS
#define BURNIN_BUILT_TARGETS ""
#endif
constexpr const char *kBuiltTargets = BURNIN_BUILT_TARGETS;

int emitMarker(const char *marker, const std::string &why, int code) {
	if (why.empty()) {
		std::printf("%s\n", marker);
	} else {
		std::printf("%s: %s\n", marker, why.c_str());
	}
	return code;
}

int pass() { return emitMarker("WMMA_GEMM_PASS", "", kExitPass); }
int fail(const std::string &w) { return emitMarker("WMMA_GEMM_FAIL", w, kExitFail); }
int skip(const std::string &w) { return emitMarker("WMMA_GEMM_SKIP", w, kExitSkip); }
int errored(const std::string &w) { return emitMarker("WMMA_GEMM_ERROR", w, kExitError); }

// ── device code ──────────────────────────────────────────────────────────────
//
// The gfx11 WMMA builtins, wave32 form. Clang declares them (BuiltinsAMDGPU.def)
// for the whole gfx11-insts family, which includes gfx1151 — the family rocWMMA
// declines to serve.
//
//   f16 : v8f32 (v16f16 a, v16f16 b, v8f32 c)
//   bf16: v8f32 (v16i16 a, v16i16 b, v8f32 c)   bf16 passed as raw 16-bit lanes
typedef _Float16 v16h __attribute__((ext_vector_type(16)));
typedef short v16s __attribute__((ext_vector_type(16)));
typedef float v8f __attribute__((ext_vector_type(8)));

// gemmTileF16 computes one 16x16 output tile with one wavefront.
//
// Every index comes from wmma_tile.h, whose mapping is unit-tested by
// simulating the whole wave in plain C++ — the layout is the part that fails
// quietly, so it is the part that is checked without hardware.
__global__ void gemmTileF16(const float *a, const float *b, float *d) {
	const int lane = static_cast<int>(threadIdx.x);
	v16h aFrag, bFrag;
	for (int e = 0; e < burnin::kFragElems; e++) {
		aFrag[e] = static_cast<_Float16>(a[burnin::aIndex(lane, e)]);
		bFrag[e] = static_cast<_Float16>(b[burnin::bIndex(lane, e)]);
	}
	v8f acc = {0, 0, 0, 0, 0, 0, 0, 0};
	acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(aFrag, bFrag, acc);
	for (int e = 0; e < burnin::kAccElems; e++) {
		d[burnin::dIndex(lane, e)] = acc[e];
	}
}

// gemmTileBf16 is the same tile through the bf16 path.
//
// The conversion is a truncation of the top 16 bits, which is EXACT for this
// runner's inputs and only for them — see burnin::bf16Bits.
__global__ void gemmTileBf16(const float *a, const float *b, float *d) {
	const int lane = static_cast<int>(threadIdx.x);
	v16s aFrag, bFrag;
	for (int e = 0; e < burnin::kFragElems; e++) {
		aFrag[e] = static_cast<short>(burnin::bf16Bits(a[burnin::aIndex(lane, e)]));
		bFrag[e] = static_cast<short>(burnin::bf16Bits(b[burnin::bIndex(lane, e)]));
	}
	v8f acc = {0, 0, 0, 0, 0, 0, 0, 0};
	acc = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(aFrag, bFrag, acc);
	for (int e = 0; e < burnin::kAccElems; e++) {
		d[burnin::dIndex(lane, e)] = acc[e];
	}
}

// runPrecision executes one precision end to end and checks it.
//
// It reports through `why` rather than printing a marker itself: which failure
// wins, and whether it is a Fail or an Error, is a decision for main — a
// per-precision helper that emitted its own verdict would let the first
// precision settle the run.
bool runPrecision(const char *label, void (*kernel)(const float *, const float *, float *),
                  const float *hostA, const float *hostB, const float *hostRef,
                  double tolerance, burnin::GemmCheck *out, std::string *why) {
	const int elems = kM * kK;
	const int outElems = kM * kN;

	float *dA = nullptr, *dB = nullptr, *dOut = nullptr;
	auto cleanup = [&]() {
		if (dA) (void)hipFree(dA);
		if (dB) (void)hipFree(dB);
		if (dOut) (void)hipFree(dOut);
	};
	auto hipFail = [&](const char *where, hipError_t e) {
		*why = std::string(label) + ": " + where + ": " + hipGetErrorString(e);
		cleanup();
		return false;
	};

	hipError_t e;
	if ((e = hipMalloc(&dA, elems * sizeof(float))) != hipSuccess) return hipFail("hipMalloc A", e);
	if ((e = hipMalloc(&dB, elems * sizeof(float))) != hipSuccess) return hipFail("hipMalloc B", e);
	if ((e = hipMalloc(&dOut, outElems * sizeof(float))) != hipSuccess)
		return hipFail("hipMalloc D", e);

	if ((e = hipMemcpy(dA, hostA, elems * sizeof(float), hipMemcpyHostToDevice)) != hipSuccess)
		return hipFail("hipMemcpy A", e);
	if ((e = hipMemcpy(dB, hostB, elems * sizeof(float), hipMemcpyHostToDevice)) != hipSuccess)
		return hipFail("hipMemcpy B", e);

	// Fill the output with a value no correct result can take, so a kernel that
	// never wrote is visible as unwritten rather than passing against a
	// reference that happened to be zero. See kUnwrittenSentinel.
	std::vector<float> sentinel(static_cast<std::size_t>(outElems), burnin::kUnwrittenSentinel);
	if ((e = hipMemcpy(dOut, sentinel.data(), outElems * sizeof(float), hipMemcpyHostToDevice)) !=
	    hipSuccess)
		return hipFail("hipMemcpy sentinel", e);

	// One wavefront computes one tile.
	hipLaunchKernelGGL(kernel, dim3(1), dim3(burnin::kWave32), 0, nullptr, dA, dB, dOut);
	if ((e = hipGetLastError()) != hipSuccess) return hipFail("gemm launch", e);
	if ((e = hipDeviceSynchronize()) != hipSuccess) return hipFail("gemm", e);

	std::vector<float> got(static_cast<std::size_t>(outElems));
	if ((e = hipMemcpy(got.data(), dOut, outElems * sizeof(float), hipMemcpyDeviceToHost)) !=
	    hipSuccess)
		return hipFail("hipMemcpy D", e);

	cleanup();
	*out = burnin::checkGemm(got.data(), hostRef, static_cast<std::size_t>(outElems), tolerance);
	return true;
}

void report(const char *prefix, const burnin::GemmCheck &c) {
	std::printf("%s_max_abs_error=%g\n%s_max_abs_ref=%g\n%s_nonfinite_count=%ld\n%s_unwritten_count=%ld\n",
	            prefix, c.maxAbsError, prefix, c.maxAbsRef, prefix, c.nonfinite, prefix, c.unwritten);
}

}  // namespace

int main() {
	std::printf("built_gfx_targets=%s\n", kBuiltTargets);
	std::printf("m=%d\nn=%d\nk=%d\n", kM, kN, kK);

	int devCount = 0;
	hipError_t e = hipGetDeviceCount(&devCount);
	if (e != hipSuccess || devCount == 0) {
		// No usable device is UNJUDGED, never a hardware verdict. This is the
		// exact case compute-smoke v0.1.0 got wrong by exiting 1.
		return errored(std::string("no usable HIP device (") +
		               (e == hipSuccess ? "zero devices" : hipGetErrorString(e)) +
		               "); hardware unjudged");
	}
	if ((e = hipSetDevice(0)) != hipSuccess) {
		return errored(std::string("hipSetDevice: ") + hipGetErrorString(e));
	}

	hipDeviceProp_t props;
	std::memset(&props, 0, sizeof(props));
	if ((e = hipGetDeviceProperties(&props, 0)) != hipSuccess) {
		return errored(std::string("hipGetDeviceProperties: ") + hipGetErrorString(e));
	}
	std::printf("gpu_name=%s\ngfx_target=%s\nwarp_size=%d\n", props.name, props.gcnArchName,
	            props.warpSize);

	// ── the three arch questions, in order, before any matrix kernel runs ─────
	const burnin::GfxTarget dev = burnin::parseGfx(props.gcnArchName);
	if (!dev.valid) {
		return errored(std::string("could not parse the device's gfx target '") + props.gcnArchName +
		               "'; refusing to guess what this part is — hardware unjudged");
	}

	switch (burnin::scopeOf(dev)) {
	case burnin::Scope::OutOfScope:
		// The ONLY path that may skip: the part genuinely has no matrix cores,
		// so acceptance does not apply to it.
		return skip(std::string("this part (") + props.gcnArchName +
		            ") has no matrix cores: RDNA1/2 and pre-CDNA GCN implement neither WMMA nor "
		            "MFMA, so a matrix-core GEMM is out of scope for it");
	case burnin::Scope::Unknown:
		return errored(std::string("cannot tell whether ") + props.gcnArchName +
		               " has matrix cores; hardware unjudged");
	case burnin::Scope::InScope:
		break;
	}

	// The generations this build emitted code for, derived from the same string
	// the image was built with so the two cannot drift.
	std::vector<int> builtFamilies;
	{
		const char *p = kBuiltTargets;
		while (*p != '\0') {
			while (*p == ' ' || *p == ',' || *p == ';') p++;
			if (*p == '\0') break;
			const char *s = p;
			while (*p != '\0' && *p != ' ' && *p != ',' && *p != ';') p++;
			const burnin::GfxTarget t =
			    burnin::parseGfx(std::string(s, static_cast<std::size_t>(p - s)).c_str());
			if (t.valid) builtFamilies.push_back(burnin::family(t));
		}
	}
	if (burnin::kernelCovers(dev, builtFamilies.data(), builtFamilies.size()) !=
	    burnin::KernelCoverage::Covered) {
		return errored(std::string("this image implements the matrix families in '") +
		               kBuiltTargets + "', and " + props.gcnArchName +
		               " belongs to none of them — its matrix path is not in this build. "
		               "Rebuild with its target in GPU_TARGETS. Hardware unjudged");
	}
	if (burnin::archMatch(kBuiltTargets, dev) != burnin::ArchMatch::Match) {
		return errored(std::string("this image carries device code for '") + kBuiltTargets +
		               "' and the part reports " + props.gcnArchName +
		               ". HIP has no JIT fallback, so there is no code here for it at all — "
		               "rebuild with GPU_TARGETS including that target. Hardware unjudged");
	}

	// The kernels are the WAVE32 form of the WMMA instruction and their
	// fragment layout is wave32's. gfx11 runs compute at wave32, so this should
	// never fire — which is exactly why it is checked rather than assumed: if it
	// ever does, every index in wmma_tile.h is wrong and the tile would be
	// silently mis-assembled rather than refused.
	if (props.warpSize != burnin::kWave32) {
		return errored(std::string("this image's WMMA kernels are the wave32 form, and this part "
		                           "reports a wavefront of ") +
		               std::to_string(props.warpSize) +
		               "; the fragment layout would not match. Hardware unjudged");
	}

	// ── inputs and the host reference ────────────────────────────────────────
	std::vector<float> a(kM * kK), b(kN * kK), ref(kM * kN);
	for (int i = 0; i < kM * kK; i++) a[static_cast<std::size_t>(i)] = burnin::exactInputValue(i);
	for (int i = 0; i < kN * kK; i++)
		b[static_cast<std::size_t>(i)] = burnin::exactInputValue(i + 3);
	burnin::referenceGemm(a.data(), b.data(), ref.data(), kM, kN, kK);

	// The inputs are exact in both formats and the accumulator is fp32, so the
	// only error left is the hardware's. The tolerance is a floor against fp32
	// accumulation order, not a precision allowance.
	const double tolerance = 1e-3;
	std::printf("tolerance=%g\n", tolerance);

	const auto started = std::chrono::steady_clock::now();

	burnin::GemmCheck bf16{}, fp16{};
	std::string why;
	if (!runPrecision("bf16", gemmTileBf16, a.data(), b.data(), ref.data(), tolerance, &bf16,
	                  &why)) {
		return errored(why + " — hardware unjudged");
	}
	if (!runPrecision("fp16", gemmTileF16, a.data(), b.data(), ref.data(), tolerance, &fp16,
	                  &why)) {
		return errored(why + " — hardware unjudged");
	}

	const double elapsedMs =
	    std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - started)
	        .count();

	report("bf16", bf16);
	report("fp16", fp16);
	std::printf("elapsed_ms=%.4f\n", elapsedMs);
	// The canonical single-value forms the contract registers, taken as the
	// WORST of the two precisions: a gate on maxAbsError must not be satisfied
	// by whichever path happened to be healthy.
	std::printf("max_abs_error=%g\nmax_abs_ref=%g\nnonfinite_count=%ld\n",
	            bf16.maxAbsError > fp16.maxAbsError ? bf16.maxAbsError : fp16.maxAbsError,
	            bf16.maxAbsRef > fp16.maxAbsRef ? bf16.maxAbsRef : fp16.maxAbsRef,
	            bf16.nonfinite + fp16.nonfinite);

	// Both precisions are gated independently, and a partial pass is a FAIL
	// naming the precision that failed.
	if (!bf16.ok || !fp16.ok) {
		std::string w;
		if (!bf16.ok) w += "bf16";
		if (!bf16.ok && !fp16.ok) w += " and ";
		if (!fp16.ok) w += "fp16";
		w += " matrix-core GEMM did not match the host reference (tolerance " +
		     std::to_string(tolerance) + ")";
		if (bf16.unwritten > 0 || fp16.unwritten > 0) {
			w += "; outputs still held the pre-launch sentinel, so the kernel did not write them";
		}
		return fail(w);
	}
	return pass();
}
