// gemm_sweep.cu — one GEMM, one precision per execution, for NVIDIA Blackwell
// SM120/SM121 (GB10 / DGX Spark on arm64, RTX PRO 6000 Blackwell and the other
// x86 SM120 parts on amd64).
//
// SPDX-License-Identifier: Apache-2.0
// Copyright the Glimmer authors.
//
// This file is original work licensed under Apache-2.0. It depends on NVIDIA
// CUTLASS (BSD-3-Clause) as a header-only library; no CUTLASS source is
// redistributed here.
//
// compute-smoke proves one thing precisely: that the NVFP4 block-scaled path
// executes and produces correct numbers. This runner asks the same question of
// the REST of the compute surface, one precision per execution, selected by the
// `precision` variant axis (BURNIN_VARIANT_PRECISION). A part can pass FP4 and
// fail FP64; tensor-core paths and CUDA-core paths are different silicon and
// fail independently.
//
// Which silicon each cell exercises, and why:
//
//   fp4   CUTLASS Sm120 block-scaled NVFP4 tensor-core path (as compute-smoke)
//   fp8   CUTLASS Sm120 dense e4m3 tensor-core path
//   bf16  mma.sync bf16 tensor-core path (CUTLASS 2.x Sm80 kernels; the
//         instruction is native on every part this image can land on)
//   tf32  mma.sync tf32 tensor-core path (same vintage, different unit)
//   fp64  the CUDA cores. The SM12x consumer parts this image targets have no
//         FP64 tensor pipe, so SIMT IS the native double path here — using a
//         tensor-op double kernel would measure an emulation, not the part.
//
// OUTPUT CONTRACT
//   metrics as key=value lines, then one of
//     GEMM_SWEEP_PASS           exit 0   measured, and the answer is right
//     GEMM_SWEEP_FAIL:  <why>   exit 1   we MEASURED the part and it fell short
//     GEMM_SWEEP_SKIP:  <why>   exit 2   this hardware is out of scope
//     GEMM_SWEEP_ERROR: <why>   exit 3   we could not measure; UNJUDGED
//
// The full exit table, and the reasoning that must survive every future edit,
// lives beside the constants in precision.h. The short form: exit 1 is a
// hardware verdict the operator never retries, so it is reserved for the three
// things this test measures about the silicon — NaN/Inf in the device output,
// an all-zero device output, a result outside tolerance. A sweep has strictly
// more error surface than compute-smoke (per precision, per kernel family, per
// tile config), and compute-smoke v0.1.0 shipped reporting three
// failures-to-measure as exit 1. Every such path here is exit 3, or a DECLARED
// exit 2 where the shortfall is a positively established property of the part.
//
// DURATION. This kind is NOT burst-only: BURNIN_DURATION_SECONDS bounds a
// window in which the same GEMM is launched repeatedly, every iteration checked
// against the host reference. The window buys repeated correctness trials and a
// stable throughput figure — it is NOT a thermal soak and does not pretend to
// be one (the per-iteration readback keeps GPU duty well under saturation;
// thermal-soak and gpu-burn own sustained load). One kind, one job.

#ifndef BURNIN_CUDA_ARCH
// The Dockerfile passes the real value. The fallback keeps a hand-built binary
// compiling, and says plainly that it does not know rather than naming an arch
// it was not built for — precision.h's archCovers answers Unknown for it, so
// the image gate runs on and leaves the runtime to speak.
#define BURNIN_CUDA_ARCH "unknown"
#endif

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

// Host-only and CUDA-free: every decision that can produce a wrong VERDICT —
// which precision was asked for, whether this part supports it, whether this
// image covers this part, and what each refusal says — lives in precision.h so
// a plain C++ test can drive it exhaustively without a GPU.
#include "precision.h"

#include "cutlass/cutlass.h"
#include "cute/tensor.hpp"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/detail/sm100_blockscaled_layout.hpp"

// The 2.x device-level API, for the cells whose native instruction predates the
// collective builders (bf16 and tf32 arrived with Ampere; SIMT doubles predate
// everything). These kernels are guarded upstream by __CUDA_ARCH__ >= 800, so
// an sm_120a / sm_121a build compiles the real tensor-core paths.
#include "cutlass/gemm/device/gemm.h"

#include "cutlass/util/host_tensor.h"
#include "cutlass/util/packed_stride.hpp"
#include "cutlass/util/device_memory.h"
#include "cutlass/util/reference/host/tensor_fill.h"
#include "cutlass/util/reference/host/gett.hpp"

namespace {

// One shape for every precision, echoed as gemm_shape= so two throughput
// figures are only ever compared when they measured the same problem. The same
// size as compute-smoke: big enough that the tensor cores do real work, small
// enough that the host reference is seconds rather than minutes.
constexpr int kM = 1024, kN = 1024, kK = 1024;

// Operands are drawn as INTEGERS in [-2, 2] (TensorFillRandomUniform with
// bits=0), which every element type here represents exactly. Every product and
// every partial sum is then exactly representable in the fp32/fp64 accumulator
// (magnitudes stay far under 2^24), so a healthy device result matches the
// double host reference EXACTLY, in any accumulation order. The bound below is
// nonzero only to be a bound: any real defect perturbs an integer product by at
// least 1, a relative error around 1/4096 here — five orders of magnitude
// above it. This is what lets the tolerance be a fact rather than a guess.
constexpr double kDenseTolerance = 1e-6;

// The fp4 cell is the exception: its output element is BF16, whose 8-bit
// mantissa rounds values of this magnitude to ~0.4% steps. Same bound as
// compute-smoke, for the same measured reason.
constexpr double kFp4Tolerance = 0.01;

using burnin::kExitError;
using burnin::kExitFail;
using burnin::kExitPass;
using burnin::kExitSkip;
using burnin::Precision;

constexpr const char *kBuiltArch = BURNIN_CUDA_ARCH;

// fail() is ONLY for a measured verdict about the silicon. If you are reaching
// for it because something went wrong, you want errored(). The table in
// precision.h is the contract; this comment is the tap on the shoulder.
int fail(const char *why) {
  std::printf("GEMM_SWEEP_FAIL: %s\n", why);
  return kExitFail;
}

// errored() is every path where the measurement did not happen. The hardware is
// unjudged, and saying so is the whole point.
int errored(const char *why) {
  std::printf("GEMM_SWEEP_ERROR: %s\n", why);
  return kExitError;
}

// A CUDA error is never a hardware verdict here: it stopped the measurement, it
// is not the measurement. The arch gate in main() has already refused the one
// case (a wrong-arch image) where the runtime's own message would mislead, so
// this is only the translation to the contract.
int cudaErrored(const char *where, cudaError_t err) {
  char buf[512];
  std::snprintf(buf, sizeof(buf),
                "CUDA error at %s: %s — the measurement did not happen and the part is UNJUDGED "
                "(this is a problem with the run, not a finding about the node)",
                where, err == cudaSuccess ? "no error latched" : cudaGetErrorString(err));
  return errored(buf);
}

// A CUTLASS status failure: the actionable detail is usually the CUDA error
// underneath it.
int cutlassErrored(const char *where) { return cudaErrored(where, cudaGetLastError()); }

// toDouble widens a device element for comparison WITHOUT losing what it had:
// double passes through untouched (a static_cast via float would silently round
// the fp64 cell's own output), float widens, and the narrow types go through
// their float conversion.
inline double toDouble(double v) { return v; }
inline double toDouble(float v) { return static_cast<double>(v); }
template <class T> inline double toDouble(T v) {
  return static_cast<double>(static_cast<float>(v));
}

// ── the measurement window ───────────────────────────────────────────────────

struct WindowStats {
  long iterations = 0;
  double totalKernelMs = 0.0;  // event-timed kernel time only, readback excluded
  double maxAbsErr = 0.0;      // worst deviation across ALL iterations
  long long nonfinite = 0;     // NaN/Inf across ALL iterations
  double sumAbsFirst = 0.0;    // first iteration's magnitude, for the all-zeros check
};

// compareIteration folds one device output into the window's statistics. Every
// iteration is checked — a transient miscompute at minute nine must not be
// overwritten by a clean minute ten.
template <class TD>
void compareIteration(const TD *got, const std::vector<double> &ref, WindowStats &s, bool first) {
  double sumAbs = 0.0;
  for (std::size_t i = 0; i < ref.size(); ++i) {
    const double g = toDouble(got[i]);
    if (std::isnan(g) || std::isinf(g)) {
      ++s.nonfinite;
      continue;
    }
    sumAbs += std::fabs(g);
    const double e = std::fabs(g - ref[i]);
    if (e > s.maxAbsErr) s.maxAbsErr = e;
  }
  if (first) s.sumAbsFirst = sumAbs;
}

// measureWindow runs launch() until the wall-clock window closes (at least
// once), check()ing every iteration, then emits the evidence and judges.
//
// launch() and check() return -1 to continue, or an exit code they have already
// reported — which is how a CUDA failure mid-window stays an Error rather than
// being laundered into a comparison.
//
// The metrics are printed BEFORE the decision, so a failing or erroring run
// still leaves its full evidence behind.
template <class LaunchFn, class CheckFn>
int measureWindow(long windowSeconds, double flopsPerIteration, double tolerance,
                  double maxAbsRef, LaunchFn &&launch, CheckFn &&check) {
  cudaEvent_t beg, end;
  if (cudaError_t e = cudaEventCreate(&beg); e != cudaSuccess) return cudaErrored("cudaEventCreate", e);
  if (cudaError_t e = cudaEventCreate(&end); e != cudaSuccess) return cudaErrored("cudaEventCreate", e);

  WindowStats s;
  const auto wallStart = std::chrono::steady_clock::now();
  for (;;) {
    cudaEventRecord(beg);
    if (int rc = launch(); rc >= 0) return rc;
    cudaEventRecord(end);
    if (cudaError_t e = cudaEventSynchronize(end); e != cudaSuccess)
      return cudaErrored("cudaEventSynchronize", e);
    float ms = 0.f;
    cudaEventElapsedTime(&ms, beg, end);
    s.totalKernelMs += ms;
    if (int rc = check(s, s.iterations == 0); rc >= 0) return rc;
    ++s.iterations;

    const double elapsed =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - wallStart).count();
    if (windowSeconds <= 0 || elapsed >= static_cast<double>(windowSeconds)) break;
  }

  const double maxRelError = maxAbsRef > 0.0 ? s.maxAbsErr / maxAbsRef : INFINITY;
  std::printf("gemm_shape=%dx%dx%d\n", kM, kN, kK);
  std::printf("gemm_iterations=%ld\ntotal_kernel_ms=%.3f\n", s.iterations, s.totalKernelMs);
  std::printf("max_abs_ref=%g\nmax_abs_error=%g\nmax_relative_error=%g\nnonfinite_count=%lld\n",
              maxAbsRef, s.maxAbsErr, maxRelError, s.nonfinite);
  // A rate needs a nonzero denominator; if the clock measured nothing, the
  // honest move is to OMIT the key, never to print a number nobody measured.
  if (s.totalKernelMs > 0.0) {
    std::printf("achieved_tflops=%.3f\n",
                flopsPerIteration * static_cast<double>(s.iterations) /
                    (s.totalKernelMs * 1e-3) / 1e12);
  }

  // Correctness first: NaN/Inf in the DEVICE output is a self-contained
  // measurement of the part that needs no reference at all.
  if (s.nonfinite != 0) return fail("output contains NaN/Inf");

  // An all-zero REFERENCE is not a finding about the GPU: the yardstick came
  // out degenerate, so nothing was measured. Checked before the two verdicts
  // that rely on it — condemning a node because the harness could not build a
  // reference is precisely the confusion the exit table exists to prevent.
  if (maxAbsRef <= 0.0)
    return errored("the host reference GEMM came out all zeros; the comparison has no yardstick "
                   "and nothing was measured about this part");

  if (s.sumAbsFirst <= 0.0) return fail("device output is all zeros");
  if (!(maxRelError <= tolerance)) return fail("numerical mismatch exceeds tolerance");
  std::printf("GEMM_SWEEP_PASS\n");
  return kExitPass;
}

// hostReference computes D = A×B on the host in double, from the EXACT operand
// values the device saw, and returns max|ref|. Independent of every device
// path: a broken tensor core cannot vouch for itself.
template <class TensorA, class TensorB>
double hostReference(TensorA &A, TensorB &B, std::vector<double> &ref) {
  std::vector<double> a(static_cast<std::size_t>(kM) * kK);
  std::vector<double> b(static_cast<std::size_t>(kK) * kN);
  for (int i = 0; i < kM; ++i)
    for (int k = 0; k < kK; ++k)
      a[static_cast<std::size_t>(i) * kK + k] = toDouble(A.host_ref().at({i, k}));
  for (int k = 0; k < kK; ++k)
    for (int j = 0; j < kN; ++j)
      b[static_cast<std::size_t>(k) * kN + j] = toDouble(B.host_ref().at({k, j}));

  ref.assign(static_cast<std::size_t>(kM) * kN, 0.0);
  for (int i = 0; i < kM; ++i) {
    double *r = &ref[static_cast<std::size_t>(i) * kN];
    for (int k = 0; k < kK; ++k) {
      const double av = a[static_cast<std::size_t>(i) * kK + k];
      const double *bp = &b[static_cast<std::size_t>(k) * kN];
      for (int j = 0; j < kN; ++j) r[j] += av * bp[j];
    }
  }
  double maxAbsRef = 0.0;
  for (double v : ref) maxAbsRef = std::fmax(maxAbsRef, std::fabs(v));
  return maxAbsRef;
}

// ── the 2.x cells: bf16, tf32, fp64 ──────────────────────────────────────────
//
// One template, three instantiations; the differences are exactly the element
// types and the op class, which is the whole claim of a sweep — same question,
// different silicon.

template <class ElementAB, class ElementCD, class ElementAcc, class OpClass, class Arch>
int runDense2x(long windowSeconds) {
  using LayoutA = cutlass::layout::RowMajor;
  using LayoutB = cutlass::layout::ColumnMajor;
  using LayoutC = cutlass::layout::RowMajor;
  using Gemm = cutlass::gemm::device::Gemm<ElementAB, LayoutA, ElementAB, LayoutB, ElementCD,
                                           LayoutC, ElementAcc, OpClass, Arch>;

  cutlass::gemm::GemmCoord problem(kM, kN, kK);
  cutlass::HostTensor<ElementAB, LayoutA> A(problem.mk());
  cutlass::HostTensor<ElementAB, LayoutB> B(problem.kn());
  cutlass::HostTensor<ElementCD, LayoutC> C(problem.mn());
  cutlass::HostTensor<ElementCD, LayoutC> D(problem.mn());

  // Integer grid points; see kDenseTolerance for why this makes the comparison
  // exact rather than tolerance-shaped.
  cutlass::reference::host::TensorFillRandomUniform(A.host_view(), 2024, 2.0, -2.0, 0);
  cutlass::reference::host::TensorFillRandomUniform(B.host_view(), 2025, 2.0, -2.0, 0);
  cutlass::reference::host::TensorFill(C.host_view());
  A.sync_device();
  B.sync_device();
  C.sync_device();
  D.sync_device();

  typename Gemm::Arguments args(problem, A.device_ref(), B.device_ref(), C.device_ref(),
                                D.device_ref(), {ElementAcc(1), ElementAcc(0)});

  // Everything from here to the comparison is SETUP: none of it measures the
  // part, so none of it may report a hardware verdict.
  Gemm gemm;
  if (Gemm::can_implement(args) != cutlass::Status::kSuccess)
    return cutlassErrored("can_implement rejected the problem");
  cutlass::device_memory::allocation<uint8_t> workspace(Gemm::get_workspace_size(args));
  if (gemm.initialize(args, workspace.get()) != cutlass::Status::kSuccess)
    return cutlassErrored("gemm.initialize failed");
  if (gemm() != cutlass::Status::kSuccess) return cutlassErrored("warmup gemm.run failed");
  if (cudaError_t e = cudaDeviceSynchronize(); e != cudaSuccess)
    return cudaErrored("warmup cudaDeviceSynchronize", e);

  std::vector<double> ref;
  const double maxAbsRef = hostReference(A, B, ref);

  auto launch = [&]() -> int {
    if (gemm() != cutlass::Status::kSuccess) return cutlassErrored("gemm.run failed");
    return -1;
  };
  auto check = [&](WindowStats &s, bool first) -> int {
    D.sync_host();
    compareIteration(D.host_data(), ref, s, first);
    return -1;
  };
  return measureWindow(windowSeconds, 2.0 * kM * kN * kK, kDenseTolerance, maxAbsRef, launch,
                       check);
}

}  // namespace

// ── the SM120 cells: fp4 (block-scaled) and fp8 (dense) ──────────────────────
//
// Whether this build has the SM120/SM121 MMA paths at all. Decided ONCE for
// both cells; two #if conditions that must agree eventually stop agreeing.
//
// BURNIN_GEMM_ASSUME_NO_SM120_MMA forces the fallback and exists for COMPILE
// COVERAGE ONLY: the #else below would otherwise never be compiled by any build
// (measured on CUTLASS v4.6.1 / CUDA 13.0.1, the SM120/SM121 macros are defined
// even for -arch=sm_80, so no arch flag reaches it — only an older CUTLASS
// would). The Dockerfile compiles both sides of this switch on every image
// build and refuses to ship a binary that took the #else.
#if defined(BURNIN_GEMM_ASSUME_NO_SM120_MMA)
#define BURNIN_GEMM_HAVE_SM120_MMA 0
#elif defined(CUTLASS_ARCH_MMA_SM120_SUPPORTED) || defined(CUTLASS_ARCH_MMA_SM121_SUPPORTED)
#define BURNIN_GEMM_HAVE_SM120_MMA 1
#else
#define BURNIN_GEMM_HAVE_SM120_MMA 0
#endif

#if BURNIN_GEMM_HAVE_SM120_MMA

using namespace cute;

// The fp4 cell: block-scaled NVFP4, the same configuration compute-smoke has
// verified on real SM121 silicon. Duplicated deliberately rather than imported:
// each runner is its own Docker build context, and this cell must keep working
// even when compute-smoke's evolves.
namespace fp4cell {

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
using StrideD   = typename Gemm::GemmKernel::StrideD;
using LayoutSFA = typename Gemm::GemmKernel::CollectiveMainloop::LayoutSFA;
using LayoutSFB = typename Gemm::GemmKernel::CollectiveMainloop::LayoutSFB;
using SFConfig  = typename Gemm::GemmKernel::CollectiveMainloop::Sm1xxBlkScaledConfig;

template <class T> auto iter(T *p) { return cute::recast_ptr<T>(p); }

int run(long windowSeconds) {
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

  // Operands land on exact e2m1 grid points; scale factors stay in a benign
  // e4m3 range. Same seeds as compute-smoke so a divergence between the two
  // runners on the same part is a real finding, not noise.
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

  Gemm gemm;
  cutlass::device_memory::allocation<uint8_t> workspace(Gemm::get_workspace_size(args));
  if (gemm.can_implement(args) != cutlass::Status::kSuccess)
    return cutlassErrored("can_implement rejected the problem");
  if (gemm.initialize(args, workspace.get()) != cutlass::Status::kSuccess)
    return cutlassErrored("gemm.initialize failed");
  if (gemm.run() != cutlass::Status::kSuccess) return cutlassErrored("warmup gemm.run failed");
  if (cudaError_t e = cudaDeviceSynchronize(); e != cudaSuccess)
    return cudaErrored("warmup cudaDeviceSynchronize", e);

  // Independent reference: CUTLASS host block-scaled GETT, fp32 accumulate.
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

  const std::size_t n = static_cast<std::size_t>(size(layout_D));
  std::vector<double> ref(n);
  double maxAbsRef = 0.0;
  for (std::size_t i = 0; i < n; ++i) {
    ref[i] = static_cast<double>(Ref.host_data()[i]);
    maxAbsRef = std::fmax(maxAbsRef, std::fabs(ref[i]));
  }

  auto launch = [&]() -> int {
    if (gemm.run() != cutlass::Status::kSuccess) return cutlassErrored("gemm.run failed");
    return -1;
  };
  auto check = [&](WindowStats &s, bool first) -> int {
    D.sync_host();
    compareIteration(D.host_data(), ref, s, first);
    return -1;
  };
  return measureWindow(windowSeconds, 2.0 * kM * kN * kK, kFp4Tolerance, maxAbsRef, launch,
                       check);
}

}  // namespace fp4cell

// The fp8 cell: dense e4m3, fp32 accumulate, fp32 out, through the SM120
// non-block-scaled collective builder. Output in fp32 on purpose: it removes
// the output-rounding term from the comparison, so this cell keeps the exact
// integer-operand property the 2.x cells have.
namespace fp8cell {

using ElementA   = cutlass::float_e4m3_t;
using ElementB   = cutlass::float_e4m3_t;
using ElementC   = float;
using ElementD   = float;
using ElementAcc = float;

using LayoutATag = cutlass::layout::RowMajor;
using LayoutBTag = cutlass::layout::ColumnMajor;
using LayoutCTag = cutlass::layout::RowMajor;

constexpr int kAlignA = 128 / cutlass::sizeof_bits<ElementA>::value;  // 16
constexpr int kAlignB = 128 / cutlass::sizeof_bits<ElementB>::value;  // 16
constexpr int kAlignC = 128 / cutlass::sizeof_bits<ElementC>::value;  // 4
constexpr int kAlignD = 128 / cutlass::sizeof_bits<ElementD>::value;  // 4

using ArchTag      = cutlass::arch::Sm120;
using OpClass      = cutlass::arch::OpClassTensorOp;
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

using StrideA = typename Gemm::GemmKernel::StrideA;
using StrideB = typename Gemm::GemmKernel::StrideB;
using StrideC = typename Gemm::GemmKernel::StrideC;
using StrideD = typename Gemm::GemmKernel::StrideD;

int run(long windowSeconds) {
  auto stride_A = cutlass::make_cute_packed_stride(StrideA{}, {kM, kK, 1});
  auto stride_B = cutlass::make_cute_packed_stride(StrideB{}, {kN, kK, 1});
  auto stride_C = cutlass::make_cute_packed_stride(StrideC{}, {kM, kN, 1});
  auto stride_D = cutlass::make_cute_packed_stride(StrideD{}, {kM, kN, 1});

  cutlass::gemm::GemmCoord problem(kM, kN, kK);
  cutlass::HostTensor<ElementA, cutlass::layout::RowMajor>    A(problem.mk());
  cutlass::HostTensor<ElementB, cutlass::layout::ColumnMajor> B(problem.kn());
  cutlass::HostTensor<ElementD, cutlass::layout::RowMajor>    D(problem.mn());

  cutlass::reference::host::TensorFillRandomUniform(A.host_view(), 2024, 2.0, -2.0, 0);
  cutlass::reference::host::TensorFillRandomUniform(B.host_view(), 2025, 2.0, -2.0, 0);
  A.sync_device();
  B.sync_device();
  D.sync_device();

  typename Gemm::Arguments args{
      cutlass::gemm::GemmUniversalMode::kGemm,
      {kM, kN, kK, 1},
      {A.device_data(), stride_A, B.device_data(), stride_B},
      {{1.0f, 0.0f}, nullptr, stride_C, D.device_data(), stride_D}};

  Gemm gemm;
  cutlass::device_memory::allocation<uint8_t> workspace(Gemm::get_workspace_size(args));
  if (gemm.can_implement(args) != cutlass::Status::kSuccess)
    return cutlassErrored("can_implement rejected the problem");
  if (gemm.initialize(args, workspace.get()) != cutlass::Status::kSuccess)
    return cutlassErrored("gemm.initialize failed");
  if (gemm.run() != cutlass::Status::kSuccess) return cutlassErrored("warmup gemm.run failed");
  if (cudaError_t e = cudaDeviceSynchronize(); e != cudaSuccess)
    return cudaErrored("warmup cudaDeviceSynchronize", e);

  std::vector<double> ref;
  const double maxAbsRef = hostReference(A, B, ref);

  auto launch = [&]() -> int {
    if (gemm.run() != cutlass::Status::kSuccess) return cutlassErrored("gemm.run failed");
    return -1;
  };
  auto check = [&](WindowStats &s, bool first) -> int {
    D.sync_host();
    compareIteration(D.host_data(), ref, s, first);
    return -1;
  };
  return measureWindow(windowSeconds, 2.0 * kM * kN * kK, kDenseTolerance, maxAbsRef, launch,
                       check);
}

}  // namespace fp8cell

#endif  // BURNIN_GEMM_HAVE_SM120_MMA

namespace {

long readWindowSeconds() {
  const char *s = std::getenv("BURNIN_DURATION_SECONDS");
  if (s == nullptr || *s == '\0') return 0;
  char *end = nullptr;
  const long v = std::strtol(s, &end, 10);
  if (end == s || v < 0) return 0;
  return v;
}

}  // namespace

int main() {
  // Emitted first so every exit below — including the ones that never reach a
  // kernel — leaves the arch this image was built for on the record.
  std::printf("built_cuda_arch=%s\n", kBuiltArch);

  // The precision is the execution's identity and needs no hardware to read,
  // so it is settled before anything is asked of the device.
  const char *rawEnv = std::getenv("BURNIN_VARIANT_PRECISION");
  const std::string raw = rawEnv == nullptr ? "" : rawEnv;
  if (raw.empty()) {
    return errored(
        "BURNIN_VARIANT_PRECISION is not set. gemm-sweep runs ONE precision per execution; give "
        "the profile entry variants with axes {precision: fp4|fp8|bf16|tf32|fp64}, one per cell. "
        "Guessing a precision here would produce a measurement gated by thresholds written for a "
        "different one");
  }
  const Precision p = burnin::parsePrecision(raw);
  if (p == Precision::Unknown) {
    std::printf("%s\n", burnin::refuseMessage(p, raw, 0).c_str());
    return kExitError;
  }
  std::printf("gemm_precision=%s\n", burnin::precisionName(p));

  const long windowSeconds = readWindowSeconds();
  std::printf("window_seconds=%ld\n", windowSeconds);

  int dev = 0;
  cudaDeviceProp props{};
  // No device is an ERROR, not a failure: a pod that got no GPU has told us
  // nothing at all about the node's compute paths.
  if (cudaError_t e = cudaGetDevice(&dev); e != cudaSuccess)
    return cudaErrored("no usable CUDA device (cudaGetDevice)", e);
  if (cudaError_t e = cudaGetDeviceProperties(&props, dev); e != cudaSuccess)
    return cudaErrored("no usable CUDA device (cudaGetDeviceProperties)", e);

#if defined(BURNIN_GEMM_FORCE_CC_MAJOR) && defined(BURNIN_GEMM_FORCE_CC_MINOR)
  // TEST-ONLY, and never present in a published image. The Skip path below can
  // only be taken by a part that lacks a precision, and the fleet in front of
  // this project supports all five — so defining these two macros builds the
  // SAME source with the SAME toolchain and runs it on a REAL GPU while telling
  // it the part reports a capability it does not have. Compile-time so no
  // environment variable can reach it; no Dockerfile may define it
  // (runners/pins_test.go); and the build announces itself in a key=value line
  // that rides into the stored result, so any verdict it produced is
  // self-identifying forever.
  props.major = BURNIN_GEMM_FORCE_CC_MAJOR;
  props.minor = BURNIN_GEMM_FORCE_CC_MINOR;
  std::printf("forced_compute_cap=%d.%d\n", props.major, props.minor);
#endif

  std::printf("gpu_name=%s\ncompute_cap=%d.%d\n", props.name, props.major, props.minor);
  const int cc = props.major * 10 + props.minor;

  // The gates, in an order that is load-bearing (precision.h documents each):
  //
  //   1. scopeOf — does the PART support this precision at all? A shortfall
  //      here is a positively established property of the silicon, so it
  //      outranks any statement about the image: fp4 on an H100 is a Skip even
  //      when the image is also wrong for the part, because NVFP4 acceptance
  //      genuinely does not apply to Hopper.
  //   2. archCovers — the part is in scope, so does this IMAGE carry code for
  //      it? A mismatch is an Error, never a Skip: a B200 supports every
  //      precision here, and reporting "out of scope" would certify a fleet as
  //      exempt from the very capability it has (issue #10's lesson). Answered
  //      BEFORE any launch because the runtime does not answer it reliably —
  //      an sm_120a image LOADS on a CC 12.1 GB10 and asserts inside the
  //      kernel, stickily, under several hundred lines of CUTLASS spew that
  //      would overwrite this diagnosis in the stored result.
  switch (burnin::scopeOf(p, cc)) {
    case burnin::Scope::Refuse:
      std::printf("%s\n", burnin::refuseMessage(p, raw, cc).c_str());
      return kExitError;
    case burnin::Scope::Skip:
      std::printf("%s\n", burnin::skipMessage(p, cc).c_str());
      return kExitSkip;
    case burnin::Scope::Run:
      break;
  }

  if (burnin::archCovers(kBuiltArch, cc) == burnin::ArchCover::Mismatch) {
    std::printf("%s\n", burnin::mismatchMessage(kBuiltArch, cc).c_str());
    return kExitError;
  }

  switch (p) {
#if BURNIN_GEMM_HAVE_SM120_MMA
    case Precision::FP4:
      return fp4cell::run(windowSeconds);
    case Precision::FP8:
      return fp8cell::run(windowSeconds);
#else
    case Precision::FP4:
    case Precision::FP8:
      // A build-flag or toolchain mistake in the image, not a fault in the
      // silicon: the part is in scope and UNJUDGED. The exact wording is
      // asserted by the Dockerfile, which compiles this branch on every build
      // and refuses to ship a binary that took it.
      return errored(
          "this binary was not compiled with SM120/SM121 tensor-core MMA support (need CUTLASS "
          ">= v4.5.0 and -arch=sm_120a/sm_120f/sm_121a); the part is in scope and UNJUDGED");
#endif
    case Precision::BF16:
      return runDense2x<cutlass::bfloat16_t, float, float, cutlass::arch::OpClassTensorOp,
                        cutlass::arch::Sm80>(windowSeconds);
    case Precision::TF32:
      // ElementA/B float on the Sm80 tensor-op path IS the tf32 kernel: the
      // mainloop truncates to tf32 for the mma and accumulates in fp32, which
      // is exactly the path a training framework calls "tf32".
      return runDense2x<float, float, float, cutlass::arch::OpClassTensorOp,
                        cutlass::arch::Sm80>(windowSeconds);
    case Precision::FP64:
      return runDense2x<double, double, double, cutlass::arch::OpClassSimt,
                        cutlass::arch::Sm80>(windowSeconds);
    case Precision::Unknown:
      break;  // unreachable: refused above, before the device was touched
  }
  return errored("unreachable precision dispatch — this is a bug in the runner, not the part");
}
