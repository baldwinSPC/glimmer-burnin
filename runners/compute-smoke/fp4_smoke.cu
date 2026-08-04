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

#ifndef BURNIN_CUDA_ARCH
// The Dockerfile passes the real value. The fallback keeps a hand-built binary
// compiling, and says plainly that it does not know rather than naming an arch
// it was not built for.
#define BURNIN_CUDA_ARCH "unknown"
#endif

#include <cstdio>
#include <cmath>
#include <cstdlib>

// Host-only, CUDA-free, and therefore testable without a GPU: it decides
// whether the arch this binary was built for covers the part it landed on, and
// composes the operator-facing message when it does not. See arch_match.h for
// why that decision cannot be left to the CUDA runtime's error code.
#include "arch_match.h"

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

// The compute capability this run actually found, latched in main() the moment
// cudaGetDeviceProperties answers, so that any later failure can be compared
// against kBuiltArch. Negative means "not read yet", which is a real state: the
// first two CUDA calls in main() can fail before there is anything to latch,
// and burnin::archMatch answers Unknown for it rather than guessing.
int gDeviceMajor = -1;
int gDeviceMinor = -1;

// fail() is ONLY for a measured verdict about the silicon. If you are reaching
// for it because something went wrong, you want errored().
int fail(const char *why) {
  std::printf("FP4_GEMM_FAIL: %s\n", why);
  return kExitFail;
}

// errored() is every path where the measurement did not happen. The hardware is
// unjudged, and saying so is the whole point: this run establishes nothing about
// the node either way.
int errored(const char *why) {
  std::printf("FP4_GEMM_ERROR: %s\n", why);
  return kExitError;
}

int skipped(const char *why) {
  std::printf("FP4_GEMM_SKIP: %s\n", why);
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
//     reported as a hardware failure. The scope gate in main() admits the whole
//     12.0/12.1 family while a build pins a single arch, so an in-scope part can
//     legitimately reach the launch with nothing to run. That is a statement
//     about which image was pinned, not about the part.
//   - cudaErrorAssert is a device-side assert, which is STICKY: it poisons the
//     context, so every later CUDA call in this process fails too. It is named
//     because the loader does NOT always refuse a wrong-arch cubin — an sm_120a
//     image loads on a CC 12.1 GB10 and asserts inside the kernel instead — and
//     because "device-side assert triggered" on its own tells an operator
//     nothing about what to change.
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

#if defined(CUTLASS_ARCH_MMA_SM120_SUPPORTED) || defined(CUTLASS_ARCH_MMA_SM121_SUPPORTED)

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

int run() {
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

  // Printed before the decision, so a failing, skipping or erroring run still
  // leaves its full evidence behind. Identity is already on stdout from main().
  std::printf("m=%d\nn=%d\nk=%d\n", kM, kN, kK);
  std::printf("max_abs_ref=%g\nmax_abs_error=%g\nmax_rel_error=%g\nnonfinite_count=%lld\nelapsed_ms=%.4f\ntflops=%.2f\n",
              max_abs_ref, max_abs_err, max_rel_error, n_bad,
              elapsed_ms, 2.0 * kM * kN * kK / (elapsed_ms * 1e-3) / 1e12);

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
  std::printf("FP4_GEMM_PASS\n");
  return kExitPass;
}

} // namespace
#endif

int main() {
  // Emitted first so that every exit below — including the ones that never get
  // as far as a kernel — leaves the arch this image was built for on the record.
  std::printf("built_cuda_arch=%s\n", kBuiltArch);

  int dev = 0;
  cudaDeviceProp props{};
  // No device is an ERROR, not a failure. A pod that got no GPU (no device
  // request, a mispresented device plugin, a driver that did not answer) has
  // told us nothing at all about the node's tensor cores. This used to exit 1,
  // which recorded a permanent hardware verdict against a node the test never
  // touched — and, because a Fail is never retried, spent no retry budget
  // proving it.
  if (cudaError_t e = cudaGetDevice(&dev); e != cudaSuccess)
    return cudaErrored("no usable CUDA device (cudaGetDevice)", e);
  if (cudaError_t e = cudaGetDeviceProperties(&props, dev); e != cudaSuccess)
    return cudaErrored("no usable CUDA device (cudaGetDeviceProperties)", e);
  gDeviceMajor = props.major;
  gDeviceMinor = props.minor;
  std::printf("gpu_name=%s\ncompute_cap=%d.%d\n", props.name, props.major, props.minor);

  // The SCOPE gate, and it is deliberately the whole SM12x family rather than
  // the single arch this binary was built for. The two questions are different:
  // "is NVFP4 block-scaled GEMM a thing this part can be asked to do?" is a
  // property of the hardware and belongs in Skip, while "does this image happen
  // to carry a cubin for it?" is a property of the image that was pinned.
  //
  // Collapsing them would misreport one of the two. Narrowing this gate to the
  // built arch would make a wrong-arch image look like out-of-scope hardware,
  // and a whole fleet would report Skip — reading as "acceptance not applicable"
  // — when in truth the operator pinned the wrong tag. So an in-scope part with
  // no cubin passes THIS gate and is stopped by the image gate below as an
  // Error, which is the retryable, hardware-unjudged phase and names the fix.
  //
  // Note the image question cannot be settled by a preprocessor macro: nvcc's
  // __CUDA_ARCH_LIST__ records 1210 for sm_121a and 1200 for both sm_120a and
  // sm_120f, and only the "f" form also covers 12.1 — so the macro cannot
  // distinguish an image that runs here from one that does not. That is why the
  // gate below compares the BURNIN_CUDA_ARCH string, which does carry the
  // distinction, against the capability the device reported. Where even that
  // cannot answer, it says so and runs on rather than guessing.
  if (!(props.major == 12 && (props.minor == 0 || props.minor == 1)))
    return skipped("NVFP4 block-scaled GEMM requires compute capability 12.0/12.1");

  // The IMAGE gate, and the second of the two questions above: the part is in
  // scope, so does this image actually carry code for it? Answered here, from
  // the arch baked in at build time and the capability the device just
  // reported, rather than left to whatever the runtime latches at the launch.
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
    return errored(buf);
  }

#if defined(CUTLASS_ARCH_MMA_SM120_SUPPORTED) || defined(CUTLASS_ARCH_MMA_SM121_SUPPORTED)
  return run();
#else
  // The part is in scope but this binary has no block-scaled MMA path compiled
  // in at all — a build-flag mistake in the image, not a fault in the silicon.
  return errored("this binary was not compiled with SM120/SM121 block-scaled MMA support "
                 "(need -arch=sm_120a/sm_120f/sm_121a); the part is in scope and UNJUDGED");
#endif
}
