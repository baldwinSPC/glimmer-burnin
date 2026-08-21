// fp4_smoke.cu — NVFP4 block-scaled GEMM smoke test for NVIDIA Blackwell SM120/SM121 (e.g. GB10).
//
// SPDX-License-Identifier: Apache-2.0
// Copyright the Glimmer authors.
//
// This file is original work licensed under Apache-2.0. It depends on NVIDIA CUTLASS
// (BSD-3-Clause) as a header-only library; no CUTLASS source is redistributed here.
//
// Proves the FP4 tensor cores actually compute: runs one block-scaled NVFP4 GEMM
// (FP4 in, BF16 out) through the CUTLASS collective-builder API and checks the result
// against the CUTLASS host block-scaled reference.
//
// Build:
//   nvcc -std=c++17 -O3 -arch=sm_121a -DBURNIN_CUDA_ARCH='"sm_121a"' \
//        -I cutlass/include -I cutlass/tools/util/include \
//        -o fp4_smoke fp4_smoke.cu
//
// OUTPUT CONTRACT
//   metrics as key=value lines, then one of
//     FP4_GEMM_PASS           exit 0   the FP4 units computed, and the answer is right
//     FP4_GEMM_FAIL:  <why>   exit 1   we MEASURED the part and it is wrong
//     FP4_GEMM_SKIP:  <why>   exit 2   this hardware is out of scope for the test
//     FP4_GEMM_ERROR: <why>   exit 3   we could not measure; the part is UNJUDGED
//
// Why there are four and not three. Exit 1 is a HARDWARE VERDICT, and the
// operator never retries one: re-running a measurement until it comes out clean
// would launder a hardware fault into an acceptance, so a Fail settles the test
// where it happened. That makes exit 1 the most expensive code in this file to
// get wrong — it permanently indicts a node. It is therefore reserved for the
// three things this test actually measures about the silicon: NaN/Inf in the
// device output, an all-zero device output, and a result outside tolerance.
//
// Everything that stopped us from taking that measurement — no CUDA device
// visible, an image with no cubin for the part it landed on, a CUDA runtime or
// driver error, a failure to build the reference — is exit 3. Those are real
// problems, but they are problems with the run, not findings about the node,
// and Error is the retryable phase (see retryOnErrorLimit). Exit 2 stays what it
// always was: the part is out of scope, which is neither a fault nor a retry.
//
// MULTI-DEVICE
// ------------
// Iterates every device the pod was allocated (docs/dev/multi-device.md).
// Always sequential, one device fully processed before the next starts, and
// BURNIN_DEVICE_CONCURRENCY is read but never changes that: this is a BURST —
// one launch, milliseconds — with no window to divide or extend, so the
// concurrency axis has nothing to apply to (the design note's own exception).
//
// The three arch gates and the GEMM itself are UNCHANGED in substance from the
// single-device engine: what used to be main()'s body is now processDevice(),
// called once per device with cudaSetDevice(index) already set, and every
// `return fail(...)/errored(...)/skipped(...)` inside it now means "stop
// processing THIS device" rather than "exit the process" — the helper
// functions record the outcome into that device's own DeviceResult instead of
// printing the marker directly, and main() prints ONE combined marker at the
// end. A device whose kernel gate or arch gate fails is an Error for the WHOLE
// test (device_fold.h's combineExitCodes), per the design note: "a fold over a
// board with an unlaunched device is not a verdict" — a board where device 3
// has no cubin is not soundly certified by devices 0-2 passing.
//
// nonfinite_count and max_abs_error are the fold (Sum, Max) — the two things
// this kind actually measures per device from the values README.md documents.
// m/n/k are compile-time constants, identical on every device; max_abs_ref,
// max_rel_error, elapsed_ms and tflops are device 0's, evidence exactly as
// before (tflops was never a benchmark on a single device and folding it
// across devices would not make it one).

#ifndef BURNIN_CUDA_ARCH
// The Dockerfile passes the real value. The fallback keeps a hand-built binary
// compiling, and says plainly that it does not know rather than naming an arch
// it was not built for.
#define BURNIN_CUDA_ARCH "unknown"
#endif

#include <cstdio>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <vector>

// Host-only, CUDA-free, and therefore testable without a GPU: it decides
// whether the arch this binary was built for covers the part it landed on, and
// composes the operator-facing message when it does not. See arch_match.h for
// why that decision cannot be left to the CUDA runtime's error code.
#include "arch_match.h"
#include "device_fold.h"

#include "cutlass/cutlass.h"
#include "cute/tensor.hpp"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/detail/sm100_blockscaled_layout.hpp"

#include "cutlass/util/host_tensor.h"
#include "cutlass/util/packed_stride.hpp"
#include "cutlass/util/device_memory.h"
#include "cutlass/util/reference/host/tensor_fill.h"
#include "cutlass/util/reference/host/gett.hpp"

namespace {

namespace devices = burnin::devices;

constexpr int kM = 1024, kN = 1024, kK = 1024;
// FP4 operands are exact; the residual error comes from BF16 output rounding and
// fp32 accumulation order. Normalised by max|ref|, anything sane sits well under this.
constexpr float kTolerance = 0.01f;

// The exit contract lives in arch_match.h, where a test can reach it and assert
// the property that matters: nothing describing a failed measurement ever
// returns kExitFail.
using burnin::kExitError;
using burnin::kExitFail;
using burnin::kExitPass;
using burnin::kExitSkip;

// The arch this binary was actually compiled for, so a triager reading a stored
// result can see the image/part mismatch without having to go and inspect the
// image that produced it.
constexpr const char *kBuiltArch = BURNIN_CUDA_ARCH;

// The compute capability THIS DEVICE reports, latched at the top of
// processDevice so any later failure in that device's own processing can be
// compared against kBuiltArch. Negative means "not read yet". Deliberately not
// per-thread: this kind is sequential-only (a burst has no window to run
// devices concurrently over), so one device's processing completes before the
// next's begins and these two are never read for two devices at once.
int gDeviceMajor = -1;
int gDeviceMinor = -1;

// DeviceResult is one device's whole outcome. gCurrent points at the one
// currently being processed, so fail()/errored()/skipped() below — called from
// deep inside the arch-gate logic and the GEMM itself, exactly as before —
// record into it instead of printing a marker for a device that is not
// necessarily the one deciding the whole test.
struct DeviceResult {
  int index = 0;
  int exitCode = kExitPass;
  std::string reason;

  bool identityRead = false;
  std::string busId, name, computeCap;

  std::map<std::string, double> values; // nonfinite_count, max_abs_error

  bool ranGemm = false; // false for a device that never reached run()
  double maxAbsRef = 0.0;
  double maxRelError = 0.0;
  double elapsedMs = 0.0;
  double tflops = 0.0;
};

DeviceResult *gCurrent = nullptr;

// fail() is ONLY for a measured verdict about the silicon. If you are reaching
// for it because something went wrong, you want errored().
int fail(const char *why) {
  if (gCurrent != nullptr) {
    gCurrent->exitCode = kExitFail;
    gCurrent->reason = why;
  }
  return kExitFail;
}

// errored() is every path where the measurement did not happen. The hardware is
// unjudged, and saying so is the whole point: this run establishes nothing about
// the node either way.
int errored(const char *why) {
  if (gCurrent != nullptr) {
    gCurrent->exitCode = kExitError;
    gCurrent->reason = why;
  }
  return kExitError;
}

int skipped(const char *why) {
  if (gCurrent != nullptr) {
    gCurrent->exitCode = kExitSkip;
    gCurrent->reason = why;
  }
  return kExitSkip;
}

// A CUDA error is never a hardware verdict here.
//
// This is only the CUDA-to-plain-C++ translation; the decision and the wording
// live in arch_match.h so they can be tested without a GPU. Three codes are
// named, and the rest fall through to the runtime's own string:
//
//   - cudaErrorNoKernelImageForDevice / cudaErrorInvalidDeviceFunction is the
//     loader refusing the cubin outright, and it is the case this file once
//     reported as a hardware failure. The scope gate in processDevice admits
//     the whole 12.0/12.1 family while a build pins a single arch, so an
//     in-scope part can legitimately reach the launch with nothing to run.
//     That is a statement about which image was pinned, not about the part.
//   - cudaErrorAssert is a device-side assert, which is STICKY: it poisons the
//     CONTEXT — this device's own context, not the process — so every later
//     CUDA call THIS device attempts fails too; another device's own context
//     is unaffected. It is named because the loader does NOT always refuse a
//     wrong-arch cubin — an sm_120a image loads on a CC 12.1 GB10 and asserts
//     inside the kernel instead — and because "device-side assert triggered"
//     on its own tells an operator nothing about what to change.
//
// Whichever code arrived, an established image/part mismatch outranks it as the
// explanation; describeCudaFailure applies that precedence.
int cudaErrored(const char *where, cudaError_t err) {
  burnin::CudaFailure kind = burnin::CudaFailure::Other;
  if (err == cudaSuccess) {
    kind = burnin::CudaFailure::NoErrorLatched;
  } else if (err == cudaErrorNoKernelImageForDevice || err == cudaErrorInvalidDeviceFunction) {
    kind = burnin::CudaFailure::NoKernelImage;
  } else if (err == cudaErrorAssert) {
    kind = burnin::CudaFailure::DeviceAssert;
  }

  char buf[768];
  burnin::describeCudaFailure(buf, sizeof(buf), where, kind,
                              err == cudaSuccess ? nullptr : cudaGetErrorString(err), kBuiltArch,
                              gDeviceMajor, gDeviceMinor);
  return errored(buf);
}

// A CUTLASS status failure. Whatever CUTLASS reports, the actionable detail is
// usually the CUDA error underneath it — a wrong-arch image surfaces here as
// cudaErrorNoKernelImageForDevice rather than as anything CUTLASS names.
int cutlassErrored(const char *where) { return cudaErrored(where, cudaGetLastError()); }

} // namespace

// Whether this build has a block-scaled MMA path at all. Decided ONCE, because
// the answer is needed in two places — the kernel definition below and the
// dispatch at the end of processDevice() — and two #if conditions that must
// agree eventually stop agreeing.
//
// BURNIN_FP4_ASSUME_NO_BLOCK_SCALED_MMA forces the fallback, and exists for
// COMPILE COVERAGE ONLY: the #else branch had never been compiled by any build
// (issue #9), so a syntax or logic error in it would have surfaced for the first
// time on a CUTLASS or CUDA downgrade — which is to say, on a fleet.
//
// It is needed because the arch flag cannot reach that branch. Measured against
// CUTLASS v4.6.1 with CUDA 13.0.1: BOTH CUTLASS_ARCH_MMA_SM120_SUPPORTED and
// CUTLASS_ARCH_MMA_SM121_SUPPORTED are defined even for `-arch=sm_80`, so
// building for a pre-Blackwell arch still takes the #if side. Only an older
// CUTLASS leaves them undefined, and cloning a second, deliberately stale
// upstream into every build to compile eight lines is a worse trade than this
// switch. The Dockerfile compiles both sides on every image build.
#if defined(BURNIN_FP4_ASSUME_NO_BLOCK_SCALED_MMA)
#define BURNIN_FP4_HAVE_BLOCK_SCALED_MMA 0
#elif defined(CUTLASS_ARCH_MMA_SM120_SUPPORTED) || defined(CUTLASS_ARCH_MMA_SM121_SUPPORTED)
#define BURNIN_FP4_HAVE_BLOCK_SCALED_MMA 1
#else
#define BURNIN_FP4_HAVE_BLOCK_SCALED_MMA 0
#endif

#if BURNIN_FP4_HAVE_BLOCK_SCALED_MMA

using namespace cute;

namespace {

using ElementA   = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
using ElementB   = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
using ElementC   = cutlass::bfloat16_t;
using ElementD   = cutlass::bfloat16_t;
using ElementAcc = float;

using LayoutATag = cutlass::layout::RowMajor;
using LayoutBTag = cutlass::layout::ColumnMajor;
using LayoutCTag = cutlass::layout::RowMajor;

constexpr int kAlignA = 32;
constexpr int kAlignB = 32;
constexpr int kAlignC = 128 / cutlass::sizeof_bits<ElementC>::value;
constexpr int kAlignD = 128 / cutlass::sizeof_bits<ElementD>::value;

using ArchTag      = cutlass::arch::Sm120;
using OpClass      = cutlass::arch::OpClassBlockScaledTensorOp;
using TileShape    = Shape<_128, _128, _128>;
using ClusterShape = Shape<_1, _1, _1>;

using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    ArchTag, OpClass, TileShape, ClusterShape,
    cutlass::epilogue::collective::EpilogueTileAuto,
    ElementAcc, ElementAcc,
    ElementC, LayoutCTag, kAlignC,
    ElementD, LayoutCTag, kAlignD,
    cutlass::epilogue::collective::EpilogueScheduleAuto>::CollectiveOp;

using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    ArchTag, OpClass,
    ElementA, LayoutATag, kAlignA,
    ElementB, LayoutBTag, kAlignB,
    ElementAcc, TileShape, ClusterShape,
    cutlass::gemm::collective::StageCountAutoCarveout<
        static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
    cutlass::gemm::KernelTmaWarpSpecializedPingpong>::CollectiveOp;

using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
    Shape<int, int, int, int>, CollectiveMainloop, CollectiveEpilogue, void>;
using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

using StrideA   = typename Gemm::GemmKernel::StrideA;
using StrideB   = typename Gemm::GemmKernel::StrideB;
using StrideC   = typename Gemm::GemmKernel::StrideC;
using StrideD   = typename Gemm::GemmKernel::StrideD;
using LayoutSFA = typename Gemm::GemmKernel::CollectiveMainloop::LayoutSFA;
using LayoutSFB = typename Gemm::GemmKernel::CollectiveMainloop::LayoutSFB;
using SFConfig  = typename Gemm::GemmKernel::CollectiveMainloop::Sm1xxBlkScaledConfig;

template <class T> auto iter(T *p) { return cute::recast_ptr<T>(p); }

// run() is UNCHANGED from the single-device engine except at its two ends:
// no marker text or PASS-only stdout line, and the metrics it used to print
// directly now land in *out for processDevice to fold and report. Every
// `return fail(...)/errored(...)/cutlassErrored(...)` mid-function still means
// exactly what it always did — record this device's outcome — because those
// helpers now write into gCurrent (== out) instead of printing.
int run(DeviceResult *out) {
  const auto shape = cute::make_shape(kM, kN, kK, 1);

  auto stride_A = cutlass::make_cute_packed_stride(StrideA{}, {kM, kK, 1});
  auto stride_B = cutlass::make_cute_packed_stride(StrideB{}, {kN, kK, 1});
  auto stride_D = cutlass::make_cute_packed_stride(StrideD{}, {kM, kN, 1});
  auto layout_A = make_layout(make_shape(kM, kK, 1), stride_A);
  auto layout_B = make_layout(make_shape(kN, kK, 1), stride_B);
  auto layout_D = make_layout(make_shape(kM, kN, 1), stride_D);
  LayoutSFA layout_SFA = SFConfig::tile_atom_to_shape_SFA(shape);
  LayoutSFB layout_SFB = SFConfig::tile_atom_to_shape_SFB(shape);

  using PVL = cutlass::layout::PackedVectorLayout;
  cutlass::HostTensor<typename ElementA::DataType, PVL>        A(cutlass::make_Coord(size(layout_A)));
  cutlass::HostTensor<typename ElementB::DataType, PVL>        B(cutlass::make_Coord(size(layout_B)));
  cutlass::HostTensor<typename ElementA::ScaleFactorType, PVL> SFA(cutlass::make_Coord(size(filter_zeros(layout_SFA))));
  cutlass::HostTensor<typename ElementB::ScaleFactorType, PVL> SFB(cutlass::make_Coord(size(filter_zeros(layout_SFB))));
  cutlass::HostTensor<ElementD, PVL>                           D(cutlass::make_Coord(size(layout_D)));
  cutlass::HostTensor<ElementAcc, PVL>                         Ref(cutlass::make_Coord(size(layout_D)));

  // Operands land on exact e2m1 grid points; scale factors stay in a benign e4m3 range.
  cutlass::reference::host::TensorFillRandomUniform(A.host_view(), 2024, 2.0, -2.0, 0);
  cutlass::reference::host::TensorFillRandomUniform(B.host_view(), 2025, 2.0, -2.0, 0);
  cutlass::reference::host::TensorFillRandomUniform(SFA.host_view(), 2026, 2.0, 0.5, 0);
  cutlass::reference::host::TensorFillRandomUniform(SFB.host_view(), 2027, 2.0, 0.5, 0);
  A.sync_device(); B.sync_device(); SFA.sync_device(); SFB.sync_device();

  typename Gemm::Arguments args{
      cutlass::gemm::GemmUniversalMode::kGemm,
      {kM, kN, kK, 1},
      {A.device_data(), stride_A, B.device_data(), stride_B,
       SFA.device_data(), layout_SFA, SFB.device_data(), layout_SFB},
      {{1.0f, 0.0f}, nullptr, stride_D, D.device_data(), stride_D}};

  // Everything from here to the reference comparison is SETUP: allocating,
  // launching, synchronising. None of it measures the part, so none of it may
  // report a hardware verdict.
  Gemm gemm;
  cutlass::device_memory::allocation<uint8_t> workspace(Gemm::get_workspace_size(args));
  if (gemm.can_implement(args) != cutlass::Status::kSuccess) return cutlassErrored("can_implement rejected the problem");
  if (gemm.initialize(args, workspace.get()) != cutlass::Status::kSuccess) return cutlassErrored("gemm.initialize failed");

  if (gemm.run() != cutlass::Status::kSuccess) return cutlassErrored("warmup gemm.run failed");
  if (cudaError_t e = cudaDeviceSynchronize(); e != cudaSuccess) return cudaErrored("warmup cudaDeviceSynchronize", e);

  cudaEvent_t beg, end;
  if (cudaError_t e = cudaEventCreate(&beg); e != cudaSuccess) return cudaErrored("cudaEventCreate", e);
  if (cudaError_t e = cudaEventCreate(&end); e != cudaSuccess) return cudaErrored("cudaEventCreate", e);
  cudaEventRecord(beg);
  if (gemm.run() != cutlass::Status::kSuccess) return cutlassErrored("timed gemm.run failed");
  cudaEventRecord(end);
  if (cudaError_t e = cudaEventSynchronize(end); e != cudaSuccess) return cudaErrored("cudaEventSynchronize", e);
  float elapsed_ms = 0.f;
  cudaEventElapsedTime(&elapsed_ms, beg, end);
  D.sync_host();

  // Independent reference: CUTLASS host block-scaled GETT, fp32 accumulate and fp32 output.
  auto tA = make_tensor(iter(A.host_data()), layout_A);
  auto tB = make_tensor(iter(B.host_data()), layout_B);
  auto tSFA = make_tensor(SFA.host_data(), layout_SFA);
  auto tSFB = make_tensor(SFB.host_data(), layout_SFB);
  auto tRef = make_tensor(iter(Ref.host_data()), layout_D);
  cutlass::reference::host::GettBlockScalingMainloopParams<
      ElementAcc, decltype(tA), decltype(tSFA), decltype(tB), decltype(tSFB)>
      mainloop{tA, tSFA, tB, tSFB};
  cutlass::reference::host::GettBlockScalingEpilogueParams<
      ElementAcc, ElementAcc, ElementAcc, decltype(tRef), decltype(tRef)>
      epilogue{1.0f, 0.0f, tRef, tRef};
  cutlass::reference::host::Gemm3x(mainloop, epilogue);

  double max_abs_ref = 0.0, max_abs_err = 0.0, sum_abs_got = 0.0;
  long long n_bad = 0;
  const long long n = static_cast<long long>(size(layout_D));
  for (long long i = 0; i < n; ++i) {
    const double r = static_cast<double>(Ref.host_data()[i]);
    const double g = static_cast<double>(static_cast<float>(D.host_data()[i]));
    if (std::isnan(g) || std::isinf(g)) ++n_bad;
    max_abs_ref = std::fmax(max_abs_ref, std::fabs(r));
    max_abs_err = std::fmax(max_abs_err, std::fabs(g - r));
    sum_abs_got += std::fabs(g);
  }
  const double max_rel_error = max_abs_ref > 0.0 ? max_abs_err / max_abs_ref : INFINITY;

  // Captured for processDevice to report — device 0's identity/evidence, and
  // this device's own contribution to the fold.
  out->ranGemm = true;
  out->maxAbsRef = max_abs_ref;
  out->maxRelError = max_rel_error;
  out->elapsedMs = elapsed_ms;
  out->tflops = 2.0 * kM * kN * kK / (elapsed_ms * 1e-3) / 1e12;
  out->values["nonfinite_count"] = static_cast<double>(n_bad);
  out->values["max_abs_error"] = max_abs_err;

  // Correctness first, as in thermal-soak: NaN/Inf in the DEVICE output is a
  // self-contained measurement of the part that needs no reference at all, so it
  // is judged before anything that could go wrong with the yardstick.
  if (n_bad != 0) return fail("output contains NaN/Inf");

  // An all-zero REFERENCE, by contrast, is not a finding about the GPU. The
  // reference is computed entirely on the host, from host-generated operands, by
  // CUTLASS's host GETT — the device never touches it. A zero here means our own
  // yardstick came out degenerate, which also makes max_rel_error meaningless
  // (it is INFINITY by construction, so the tolerance check below would fire off
  // the back of it). Condemning a node because the harness could not build a
  // reference is precisely the confusion this exit contract exists to prevent,
  // so it is an Error, and it is checked before the two verdicts that rely on it.
  if (max_abs_ref <= 0.0)
    return errored("the host reference GEMM came out all zeros; the comparison has no yardstick "
                   "and nothing was measured about this part");

  if (sum_abs_got <= 0.0) return fail("device output is all zeros");
  if (!(max_rel_error <= kTolerance)) return fail("numerical mismatch exceeds tolerance");
  return kExitPass;
}

} // namespace
#endif

namespace {

// processDevice is what main() used to be, scoped to one device. Every
// `return skipped(...)/errored(...)` here means "stop processing THIS
// device" — the helper records the outcome into `out` and processDevice
// returns, and the caller moves on to the next device. The arch-gate logic
// below is UNCHANGED from the single-device engine; only the wrapping (which
// device, and what happens to the result) is new.
void processDevice(int index, DeviceResult *out) {
  out->index = index;
  gCurrent = out;
  gDeviceMajor = -1;
  gDeviceMinor = -1;

  if (cudaError_t e = cudaSetDevice(index); e != cudaSuccess) {
    cudaErrored("cudaSetDevice", e);
    return;
  }
  cudaDeviceProp props{};
  if (cudaError_t e = cudaGetDeviceProperties(&props, index); e != cudaSuccess) {
    cudaErrored("no usable CUDA device (cudaGetDeviceProperties)", e);
    return;
  }
  char busId[32] = {0};
  out->identityRead = cudaDeviceGetPCIBusId(busId, sizeof(busId), index) == cudaSuccess;
  if (out->identityRead) out->busId = busId;
  out->name = props.name;

#if defined(BURNIN_FP4_FORCE_CC_MAJOR) && defined(BURNIN_FP4_FORCE_CC_MINOR)
  // TEST-ONLY, and never present in a published image.
  //
  // The Skip path below can only be taken by a part that is not compute
  // capability 12.0/12.1, and no such accelerator is available to this project
  // (issue #9). Exit 2 is the contract's load-bearing distinction — a node that
  // cannot run a test has NOT failed it — so leaving it never once executed on
  // real silicon was the gap this exists to close: defining these two macros
  // builds the SAME source, with the SAME toolchain, and runs it on a REAL GPU
  // while telling it the part reports a capability it does not have.
  //
  // Three things keep it from ever becoming a way to fabricate a Skip in a
  // fleet. It is compile-time, so no environment variable or argument can reach
  // it; no Dockerfile in this repository defines it, which runners/pins_test.go
  // asserts; and a build that does define it says so on stdout, in a key=value
  // line that lands in the stored result and the delivered envelope, so any
  // verdict such a binary produced is self-identifying forever.
  props.major = BURNIN_FP4_FORCE_CC_MAJOR;
  props.minor = BURNIN_FP4_FORCE_CC_MINOR;
  if (index == 0) std::printf("forced_compute_cap=%d.%d\n", props.major, props.minor);
#endif

  gDeviceMajor = props.major;
  gDeviceMinor = props.minor;
  char cc[16];
  std::snprintf(cc, sizeof(cc), "%d.%d", props.major, props.minor);
  out->computeCap = cc;

  // Identity keys keep TODAY's meaning — device 0's — per
  // docs/dev/multi-device.md: a bracket-indexed pseudo-key would not match the
  // registered gpu_name/compute_cap names at all, and every device's identity
  // already rides in the per-device.json artifact. Printed here, gated on
  // index==0 and reached only once cudaGetDeviceProperties has succeeded for
  // THIS device — exactly as the single-device engine only ever reported
  // identity for a device whose properties it could read. pci_bus_id is new
  // for the fold (worst_device_pci_bus_id needs it) and is the one identity
  // field that can fail independently, so it alone is gated a second time.
  if (index == 0) {
    std::printf("gpu_name=%s\n", out->name.c_str());
    std::printf("compute_cap=%s\n", out->computeCap.c_str());
    if (out->identityRead) std::printf("pci_bus_id=%s\n", out->busId.c_str());
  }

  // The SCOPE gate — "is NVFP4 block-scaled GEMM a thing this part can be asked
  // to do at all?", which is a property of the hardware and is answered with a
  // Skip. It is deliberately the whole SM12x family rather than the single arch
  // this binary was built for; the reasoning, and the second question it must
  // not be collapsed with, are in arch_match.h.
  //
  // The decision and its wording live there so a plain C++ test can drive them
  // to every capability without a GPU. That matters here more than anywhere
  // else in this file: this is the only Skip the runner has, and the hardware
  // that would take it does not exist in this fleet.
  if (burnin::scopeOf(props.major, props.minor) == burnin::Scope::OutOfScope) {
    char buf[512];
    burnin::describeOutOfScope(buf, sizeof(buf), props.major, props.minor);
    skipped(buf);
    return;
  }

  // The KERNEL gate, and it runs BEFORE the image gate on purpose.
  //
  // A CC 10.x part fails both: this image has no SM10x cubin, and it has no
  // SM10x kernel either. Only one of those is the actionable finding. Reporting
  // the gencode mismatch first would tell an operator to rebuild with
  // CUDA_ARCH=sm_100a — which compiles, produces an Sm120 kernel emitted for a
  // part that cannot run it, and sends them round a loop with no exit. Saying
  // "the kernel for this path does not exist here yet, see #10" is the truth and
  // is actionable in one step.
  //
  // It is an Error and emphatically not a Skip: NVFP4 acceptance APPLIES to a
  // B200, and a Skip would record a fleet as out of scope for the very
  // capability it has. See the three-questions note in arch_match.h.
  if (burnin::kernelCovers(props.major, props.minor) == burnin::KernelCoverage::WrongFamily) {
    char buf[640];
    burnin::describeWrongKernelFamily(buf, sizeof(buf), props.major, props.minor);
    errored(buf);
    return;
  }

  // The IMAGE gate, and the second of the two questions above: the part is in
  // scope, so does this image actually carry code for it? Answered here, from
  // the arch baked in at build time and the capability the device just
  // reported, rather than left to whatever the runtime latches at the launch.
  //
  // It cannot be settled by a preprocessor macro either: nvcc's
  // __CUDA_ARCH_LIST__ records 1210 for sm_121a and 1200 for both sm_120a and
  // sm_120f, and only the "f" form also covers 12.1 — so the macro cannot
  // distinguish an image that runs here from one that does not. That is why this
  // compares the BURNIN_CUDA_ARCH string, which does carry the distinction,
  // against the capability the device reported.
  //
  // It has to be answered here because the launch does not answer it reliably.
  // The loader refuses an sm_121a cubin on a CC 12.0 part, and that direction
  // was always caught. The other direction is not refused at all: SM121 is
  // binary-compatible with SM120, so an sm_120a image LOADS on a CC 12.1 GB10
  // and then trips a device-side assert inside the CUTLASS kernel. Measured on
  // real hardware, that reached the operator as a bare "warmup
  // cudaDeviceSynchronize: device-side assert triggered" naming neither the
  // arch nor the fix.
  //
  // Refusing BEFORE the launch is what makes this worth doing, rather than only
  // improving the message afterwards. Nothing is launched, so no assert fires,
  // so the CUDA context is not poisoned and CUTLASS emits none of its several
  // hundred lines of template assert text. That spew is not merely ugly: a
  // container log is stdout and stderr MERGED, and pkg/runner.Parse takes
  // Result.Message from the LAST line that is not key=value — so a CUTLASS line
  // landing after our diagnosis becomes the message stored in the TestResult,
  // and the operator loses the one line that told them what to change.
  //
  // Conservative by construction: burnin::archMatch reports Mismatch only where
  // it is positively established, and answers Unknown for an arch string it
  // does not recognise (a hand-built binary carries "unknown"). An Unknown runs
  // on and leaves the runtime error to speak, exactly as before. And this is an
  // Error, never a Skip: reporting it as out-of-scope hardware would tell a
  // whole fleet that acceptance does not apply to it when the truth is that
  // somebody pinned the wrong tag.
  if (burnin::archMatch(kBuiltArch, props.major, props.minor) == burnin::ArchMatch::Mismatch) {
    char buf[768];
    burnin::describeArchMismatch(buf, sizeof(buf), kBuiltArch, props.major, props.minor);
    errored(buf);
    return;
  }

#if BURNIN_FP4_HAVE_BLOCK_SCALED_MMA
  run(out);
#else
  // The part is in scope but this binary has no block-scaled MMA path compiled
  // in at all — a build-flag or toolchain mistake in the image, not a fault in
  // the silicon, so it is UNJUDGED and never a hardware verdict.
  errored("this binary was not compiled with SM120/SM121 block-scaled MMA support "
         "(need CUTLASS >= v4.5.0 and -arch=sm_120a/sm_120f/sm_121a); the part is in "
         "scope and UNJUDGED");
#endif
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

// wholeRunError/wholeRunSkip are for the handful of decisions made BEFORE any
// device exists to attribute them to — how many devices are visible, and
// whether the plan even allows the run to proceed. fail()/errored()/skipped()
// record into gCurrent, which is null at this point in main(), so these two
// print the marker directly instead.
int wholeRunError(const char *why) {
  std::printf("FP4_GEMM_ERROR: %s\n", why);
  return kExitError;
}
int wholeRunSkip(const char *why) {
  std::printf("FP4_GEMM_SKIP: %s\n", why);
  return kExitSkip;
}

} // namespace

int main() {
  // Emitted first so that every exit below — including the ones that never get
  // as far as a kernel — leaves the arch this image was built for on the record.
  std::printf("built_cuda_arch=%s\n", kBuiltArch);

  // ── how many devices, and how ────────────────────────────────────────────
  int visible = 0;
  const cudaError_t countErr = cudaGetDeviceCount(&visible);
  if (countErr != cudaSuccess && countErr != cudaErrorNoDevice) {
    char buf[512];
    burnin::describeCudaFailure(buf, sizeof(buf), "no usable CUDA device (cudaGetDeviceCount)",
                                burnin::CudaFailure::Other, cudaGetErrorString(countErr), kBuiltArch,
                                -1, -1);
    return wholeRunError(buf);
  }
  if (countErr == cudaErrorNoDevice) visible = 0;

  const devices::Budget budget =
      devices::parseBudget(std::getenv("BURNIN_RESOURCE_LIMITS"), devices::nvidiaResources());
  const devices::Plan plan = devices::planIteration(visible, budget);
  if (plan.outcome == devices::Plan::Skip) return wholeRunSkip(plan.message.c_str());
  if (plan.outcome == devices::Plan::Error) return wholeRunError(plan.message.c_str());
  const int planCount = plan.count;

  // A burst has no window to divide or extend, so BURNIN_DEVICE_CONCURRENCY
  // changes nothing here — read anyway so an unrecognised value is reported the
  // same way every other kind reports one, rather than being silently ignored
  // without comment.
  const char *concEnv = std::getenv("BURNIN_DEVICE_CONCURRENCY");
  const devices::ConcurrencyChoice conc =
      devices::resolveConcurrency(concEnv, devices::Concurrency::Sequential);
  if (concEnv != nullptr && *concEnv != '\0' && !conc.recognised) {
    std::printf(
        "config_warnings=BURNIN_DEVICE_CONCURRENCY=\"%s\" is neither \"all\" nor \"sequential\"; "
        "compute-smoke is a burst and iterates sequentially regardless\n",
        concEnv);
  }
  // device_window_s and device_concurrency are printed once, below, by
  // printFold — printing them here too would emit each key twice, and
  // pkg/runner.Parse is last-occurrence-wins, so the duplicate would just be
  // silently discarded, but there is no reason to emit it at all.

  std::vector<DeviceResult> results(planCount);
  for (int i = 0; i < planCount; ++i) {
    processDevice(i, &results[i]);
  }
  gCurrent = nullptr;

  std::vector<devices::DeviceReport> reports;
  for (auto &r : results) {
    if (!r.ranGemm) continue; // never reached the GEMM: gated out or errored first
    reports.push_back(toDeviceReport(r));
  }
  static const std::vector<devices::FoldRule> kDeviceFold = {
      {"nonfinite_count", devices::Fold::Sum},
      {"max_abs_error", devices::Fold::Max},
  };
  // primaryKey is nonfinite_count, not max_abs_error: nonfiniteCount is the
  // registered Acceptance metric (the sample's own threshold gates on it),
  // while maxAbsError is Evidence-only (registered ThresholdUseEvidence,
  // beside the gateable maxRelativeError) — the "worst device" a verdict
  // names should be the device the GATE actually turns on.
  const devices::Folded folded = devices::fold(reports, kDeviceFold, "nonfinite_count");

  if (auto it = folded.values.find("nonfinite_count"); it != folded.values.end()) {
    std::printf("nonfinite_count=%.0f\n", it->second);
  }
  if (auto it = folded.values.find("max_abs_error"); it != folded.values.end()) {
    std::printf("max_abs_error=%.6g\n", it->second);
  }
  // Evidence: device 0's, exactly as the single-device engine always reported
  // (a liveness figure, never a benchmark — see the README). m/n/k are
  // compile-time constants and identical for every device, but printed only
  // once a device actually reached the GEMM — matching the single-device
  // engine, which printed them from inside run() and never for a device that
  // never got that far.
  for (auto &r : results) {
    if (!r.ranGemm) continue;
    std::printf("m=%d\nn=%d\nk=%d\n", kM, kN, kK);
    std::printf("max_abs_ref=%.6g\n", r.maxAbsRef);
    std::printf("max_rel_error=%.6g\n", r.maxRelError);
    std::printf("elapsed_ms=%.4f\n", r.elapsedMs);
    std::printf("tflops=%.2f\n", r.tflops);
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
    std::printf("FP4_GEMM_PASS\n");
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
  const char *marker = combined == kExitFail   ? "FP4_GEMM_FAIL"
                       : combined == kExitSkip ? "FP4_GEMM_SKIP"
                                               : "FP4_GEMM_ERROR";
  std::printf("%s: %s\n", marker, reasons.c_str());
  return combined;
}
