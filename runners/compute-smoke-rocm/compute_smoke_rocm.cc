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
// MULTI-DEVICE
// ------------
// Iterates every device the pod was allocated (docs/dev/multi-device.md).
// Always sequential: this is a BURST, one launch pair (bf16 + fp16) per
// device, milliseconds each — no window to divide or extend, so
// BURNIN_DEVICE_CONCURRENCY is read and reported but never changes the
// iteration itself, the same exception fp4_smoke.cu documents.
//
// What used to be main()'s body, from device 0's identity read onward, is now
// processDevice(), called once per device with hipSetDevice(index) already
// done; every `return fail(...)/errored(...)/skip(...)` inside it still means
// exactly what it always did — this device's own verdict — because those
// helpers now record into that device's own DeviceResult instead of printing
// a marker directly, and main() prints ONE combined marker at the end
// (device_fold.h's combineExitCodes: Fail > Error > Skip > Pass — a device is
// a PART, so a measured miscompare on one does not erase a measured pass on
// another, but a device whose kernel or arch gate failed leaves the WHOLE
// board unjudged for that device, and folding over an unlaunched device is not
// a verdict).
//
// nonfinite_count and max_abs_error are the fold (Sum, Max), taken as the worse
// of the two precisions per device before folding, exactly as the
// single-device engine already reported them. bf16_*/fp16_*/tolerance/
// elapsed_ms/warp_size/m/n/k are device 0's, evidence exactly as before.
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
#include <map>
#include <string>
#include <vector>

#include "device_fold.h"
#include "gfx_gate.h"
#include "wmma_tile.h"

using burnin::kExitError;
using burnin::kExitFail;
using burnin::kExitPass;
using burnin::kExitSkip;

namespace {

namespace devices = burnin::devices;

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

// DeviceResult is one device's whole outcome. gCurrent points at the one
// currently being processed, so pass()/fail()/skip()/errored() below — called
// from deep inside the arch-gate logic and runPrecision, exactly as before —
// record into it instead of printing a marker for a device that is not
// necessarily the one deciding the whole test.
struct DeviceResult {
	int index = 0;
	int exitCode = kExitPass;
	std::string reason;

	bool identityRead = false;
	std::string busId, name, computeCap;  // computeCap holds the gfx target string

	std::map<std::string, double> values;  // nonfinite_count, max_abs_error

	bool ranGemm = false;  // false for a device that never reached both precisions
	burnin::GemmCheck bf16{}, fp16{};
	double tolerance = 0.0;
	double elapsedMs = 0.0;
	int warpSize = 0;
};

DeviceResult *gCurrent = nullptr;

int emitInto(const std::string &why, int code) {
	if (gCurrent != nullptr) {
		gCurrent->exitCode = code;
		gCurrent->reason = why;
	}
	return code;
}

// pass() is the only WMMA_GEMM_PASS path; fail() is ONLY for a measured
// verdict about the silicon; skip() and errored() carry the same meaning as
// every other kind's: the hardware was not judged either way. Which marker
// each corresponds to is decided once, in main(), from the combined exit code
// across every device — not printed here.
int pass() { return emitInto("", kExitPass); }
int fail(const std::string &w) { return emitInto(w, kExitFail); }
int skip(const std::string &w) { return emitInto(w, kExitSkip); }
int errored(const std::string &w) { return emitInto(w, kExitError); }

// ── device code ──────────────────────────────────────────────────────────────
//
// The gfx11 WMMA builtins, wave32 form. Clang declares them (BuiltinsAMDGPU.def)
// for the whole gfx11-insts family, which includes gfx1151 — the family rocWMMA
// declines to serve. UNCHANGED from the single-device engine.
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

// runPrecision executes one precision end to end and checks it. UNCHANGED from
// the single-device engine.
//
// It reports through `why` rather than printing a marker itself: which failure
// wins, and whether it is a Fail or an Error, is a decision for processDevice —
// a per-precision helper that emitted its own verdict would let the first
// precision settle the device.
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

// processDevice is what main() used to be, scoped to one device, from the
// point hipSetDevice(0) used to run. Every `return fail(...)/skip(...)/
// errored(...)` here means "stop processing THIS device" — the helper records
// the outcome into `out` and processDevice returns, and the caller moves on to
// the next device. The arch-gate logic and the two-precision GEMM below are
// UNCHANGED from the single-device engine; only the wrapping is new.
void processDevice(int index, DeviceResult *out) {
	out->index = index;
	gCurrent = out;

	hipError_t e;
	if ((e = hipSetDevice(index)) != hipSuccess) {
		errored(std::string("hipSetDevice: ") + hipGetErrorString(e));
		return;
	}

	hipDeviceProp_t props;
	std::memset(&props, 0, sizeof(props));
	if ((e = hipGetDeviceProperties(&props, index)) != hipSuccess) {
		errored(std::string("hipGetDeviceProperties: ") + hipGetErrorString(e));
		return;
	}
	out->name = props.name;
	out->computeCap = props.gcnArchName;
	out->warpSize = props.warpSize;

	char busId[32] = {0};
	out->identityRead = hipDeviceGetPCIBusId(busId, sizeof(busId), index) == hipSuccess;
	if (out->identityRead) out->busId = busId;

	// Identity keys keep TODAY's meaning — device 0's — per
	// docs/dev/multi-device.md: a bracket-indexed pseudo-key would not match
	// the registered gpu_name/gfx_target names at all, and every device's
	// identity already rides in the per-device.json artifact. gpu_name/
	// gfx_target/warp_size are always readable once hipGetDeviceProperties
	// succeeded for THIS device, exactly as the single-device engine always
	// reported them; pci_bus_id is new for the fold (worst_device_pci_bus_id
	// needs it) and is the one identity field that can fail independently, so
	// it alone is gated.
	if (index == 0) {
		std::printf("gpu_name=%s\ngfx_target=%s\nwarp_size=%d\n", out->name.c_str(),
		            out->computeCap.c_str(), out->warpSize);
		if (out->identityRead) std::printf("pci_bus_id=%s\n", out->busId.c_str());
	}

	// ── the three arch questions, in order, before any matrix kernel runs ─────
	const burnin::GfxTarget dev = burnin::parseGfx(props.gcnArchName);
	if (!dev.valid) {
		errored(std::string("could not parse the device's gfx target '") + props.gcnArchName +
		        "'; refusing to guess what this part is — hardware unjudged");
		return;
	}

	switch (burnin::scopeOf(dev)) {
	case burnin::Scope::OutOfScope:
		// The ONLY path that may skip: the part genuinely has no matrix cores,
		// so acceptance does not apply to it.
		skip(std::string("this part (") + props.gcnArchName +
		     ") has no matrix cores: RDNA1/2 and pre-CDNA GCN implement neither WMMA nor "
		     "MFMA, so a matrix-core GEMM is out of scope for it");
		return;
	case burnin::Scope::Unknown:
		errored(std::string("cannot tell whether ") + props.gcnArchName +
		        " has matrix cores; hardware unjudged");
		return;
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
		errored(std::string("this image implements the matrix families in '") + kBuiltTargets +
		        "', and " + props.gcnArchName +
		        " belongs to none of them — its matrix path is not in this build. "
		        "Rebuild with its target in GPU_TARGETS. Hardware unjudged");
		return;
	}
	if (burnin::archMatch(kBuiltTargets, dev) != burnin::ArchMatch::Match) {
		errored(std::string("this image carries device code for '") + kBuiltTargets +
		        "' and the part reports " + props.gcnArchName +
		        ". HIP has no JIT fallback, so there is no code here for it at all — "
		        "rebuild with GPU_TARGETS including that target. Hardware unjudged");
		return;
	}

	// The kernels are the WAVE32 form of the WMMA instruction and their
	// fragment layout is wave32's. gfx11 runs compute at wave32, so this should
	// never fire — which is exactly why it is checked rather than assumed: if it
	// ever does, every index in wmma_tile.h is wrong and the tile would be
	// silently mis-assembled rather than refused.
	if (props.warpSize != burnin::kWave32) {
		errored(std::string("this image's WMMA kernels are the wave32 form, and this part "
		                     "reports a wavefront of ") +
		        std::to_string(props.warpSize) + "; the fragment layout would not match. "
		        "Hardware unjudged");
		return;
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
	out->tolerance = tolerance;

	const auto started = std::chrono::steady_clock::now();

	burnin::GemmCheck bf16{}, fp16{};
	std::string why;
	if (!runPrecision("bf16", gemmTileBf16, a.data(), b.data(), ref.data(), tolerance, &bf16,
	                  &why)) {
		errored(why + " — hardware unjudged");
		return;
	}
	if (!runPrecision("fp16", gemmTileF16, a.data(), b.data(), ref.data(), tolerance, &fp16,
	                  &why)) {
		errored(why + " — hardware unjudged");
		return;
	}

	out->elapsedMs =
	    std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - started)
	        .count();
	out->bf16 = bf16;
	out->fp16 = fp16;
	out->ranGemm = true;
	// The canonical single-value forms the contract registers, taken as the
	// WORST of the two precisions: a gate on maxAbsError must not be satisfied
	// by whichever path happened to be healthy.
	out->values["max_abs_error"] = bf16.maxAbsError > fp16.maxAbsError ? bf16.maxAbsError : fp16.maxAbsError;
	out->values["nonfinite_count"] = static_cast<double>(bf16.nonfinite + fp16.nonfinite);

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
		fail(w);
		return;
	}
	pass();
}

devices::DeviceReport toDeviceReport(const DeviceResult &r) {
	devices::DeviceReport rep;
	rep.index = r.index;
	rep.busId = r.busId;
	rep.name = r.name;
	rep.computeCap = r.computeCap;
	rep.identityRead = r.identityRead;
	rep.values = r.values;
	return rep;
}

// wholeRunError is for the one decision made BEFORE any device exists to
// attribute it to: how many devices are visible, and whether the plan even
// allows the run to proceed. pass()/fail()/skip()/errored() record into
// gCurrent, which is null at this point in main(), so this prints the marker
// directly instead.
int wholeRunError(const std::string &why) {
	std::printf("WMMA_GEMM_ERROR: %s\n", why.c_str());
	return kExitError;
}
int wholeRunSkip(const std::string &why) {
	std::printf("WMMA_GEMM_SKIP: %s\n", why.c_str());
	return kExitSkip;
}

}  // namespace

int main() {
	std::printf("built_gfx_targets=%s\n", kBuiltTargets);
	std::printf("m=%d\nn=%d\nk=%d\n", kM, kN, kK);

	int visible = 0;
	hipError_t e = hipGetDeviceCount(&visible);
	if (e != hipSuccess && e != hipErrorNoDevice) {
		// No usable device is UNJUDGED, never a hardware verdict. This is the
		// exact case compute-smoke v0.1.0 got wrong by exiting 1.
		return wholeRunError(std::string("no usable HIP device (") + hipGetErrorString(e) +
		                     "); hardware unjudged");
	}
	if (e == hipErrorNoDevice) visible = 0;

	const devices::Budget budget =
	    devices::parseBudget(std::getenv("BURNIN_RESOURCE_LIMITS"), devices::amdResources());
	const devices::Plan plan = devices::planIteration(visible, budget);
	if (plan.outcome == devices::Plan::Skip) return wholeRunSkip(plan.message);
	if (plan.outcome == devices::Plan::Error) return wholeRunError(plan.message);
	const int planCount = plan.count;

	// A burst has no window to divide or extend, so BURNIN_DEVICE_CONCURRENCY
	// changes nothing here — read anyway so an unrecognised value is reported
	// the same way every other kind reports one, rather than being silently
	// ignored without comment.
	const char *concEnv = std::getenv("BURNIN_DEVICE_CONCURRENCY");
	const devices::ConcurrencyChoice conc =
	    devices::resolveConcurrency(concEnv, devices::Concurrency::Sequential);
	if (concEnv != nullptr && *concEnv != '\0' && !conc.recognised) {
		std::printf(
		    "config_warnings=BURNIN_DEVICE_CONCURRENCY=\"%s\" is neither \"all\" nor \"sequential\"; "
		    "compute-smoke-rocm is a burst and iterates sequentially regardless\n",
		    concEnv);
	}

	std::vector<DeviceResult> results(planCount);
	for (int i = 0; i < planCount; ++i) {
		processDevice(i, &results[i]);
	}
	gCurrent = nullptr;

	std::vector<devices::DeviceReport> reports;
	for (auto &r : results) {
		if (!r.ranGemm) continue;  // never reached both precisions: gated out or errored first
		reports.push_back(toDeviceReport(r));
	}
	static const std::vector<devices::FoldRule> kDeviceFold = {
	    {"nonfinite_count", devices::Fold::Sum},
	    {"max_abs_error", devices::Fold::Max},
	};
	// primaryKey is nonfinite_count, not max_abs_error: nonfiniteCount is the
	// registered Acceptance metric (the sample's own threshold gates on it),
	// while maxAbsError is Evidence-only — the "worst device" a verdict names
	// should be the device the GATE actually turns on.
	const devices::Folded folded = devices::fold(reports, kDeviceFold, "nonfinite_count");

	if (auto it = folded.values.find("nonfinite_count"); it != folded.values.end()) {
		std::printf("nonfinite_count=%.0f\n", it->second);
	}
	if (auto it = folded.values.find("max_abs_error"); it != folded.values.end()) {
		std::printf("max_abs_error=%.6g\n", it->second);
	}
	// Evidence: device 0's, exactly as the single-device engine always
	// reported (max_abs_ref is the per-precision worst too, as before).
	for (auto &r : results) {
		if (!r.ranGemm) continue;
		std::printf("tolerance=%g\n", r.tolerance);
		report("bf16", r.bf16);
		report("fp16", r.fp16);
		std::printf("elapsed_ms=%.4f\n", r.elapsedMs);
		std::printf("max_abs_ref=%g\n",
		            r.bf16.maxAbsRef > r.fp16.maxAbsRef ? r.bf16.maxAbsRef : r.fp16.maxAbsRef);
		break;
	}

	devices::printFold(stdout, reports, visible, /*windowS=*/0, conc.mode, folded, /*spreads=*/{},
	                   /*underMig=*/false);
	if (reports.size() > 1) {
		std::fputs(devices::renderPerDeviceArtifact(reports).c_str(), stdout);
	}

	std::vector<int> codes;
	for (auto &r : results) codes.push_back(r.exitCode);
	const int combined = devices::combineExitCodes(codes);

	if (combined == kExitPass) {
		std::printf("WMMA_GEMM_PASS\n");
		return kExitPass;
	}
	std::string reasons;
	for (auto &r : results) {
		if (r.exitCode == combined && !r.reason.empty()) {
			if (!reasons.empty()) reasons += "; ";
			reasons += "device " + std::to_string(r.index) + ": " + r.reason;
		}
	}
	if (reasons.empty()) reasons = "no device could be measured";
	const char *marker = combined == kExitFail   ? "WMMA_GEMM_FAIL"
	                    : combined == kExitSkip ? "WMMA_GEMM_SKIP"
	                                            : "WMMA_GEMM_ERROR";
	std::printf("%s: %s\n", marker, reasons.c_str());
	return combined;
}
