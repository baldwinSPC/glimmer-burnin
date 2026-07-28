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
//   nvcc -std=c++17 -O3 -arch=sm_121a \
//        -I cutlass/include -I cutlass/tools/util/include \
//        -o fp4_smoke fp4_smoke.cu
//
// Output contract: "FP4_GEMM_PASS" + exit 0 | "FP4_GEMM_FAIL: ..." + exit 1 |
//                  "FP4_GEMM_SKIP: ..." + exit 2. Metrics are printed as key=value lines.

#include <cstdio>
#include <cmath>
#include <cstdlib>

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

int fail(const char *why) {
  std::printf("FP4_GEMM_FAIL: %s\n", why);
  return 1;
}

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

  Gemm gemm;
  cutlass::device_memory::allocation<uint8_t> workspace(Gemm::get_workspace_size(args));
  if (gemm.can_implement(args) != cutlass::Status::kSuccess) return fail("can_implement rejected the problem");
  if (gemm.initialize(args, workspace.get()) != cutlass::Status::kSuccess) return fail("gemm.initialize failed");

  if (gemm.run() != cutlass::Status::kSuccess) return fail("warmup gemm.run failed");
  if (cudaDeviceSynchronize() != cudaSuccess) return fail(cudaGetErrorString(cudaGetLastError()));

  cudaEvent_t beg, end;
  cudaEventCreate(&beg); cudaEventCreate(&end);
  cudaEventRecord(beg);
  if (gemm.run() != cutlass::Status::kSuccess) return fail("timed gemm.run failed");
  cudaEventRecord(end);
  if (cudaEventSynchronize(end) != cudaSuccess) return fail(cudaGetErrorString(cudaGetLastError()));
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

  cudaDeviceProp props{}; int dev = 0;
  cudaGetDevice(&dev); cudaGetDeviceProperties(&props, dev);
  std::printf("gpu_name=%s\ncompute_cap=%d.%d\nm=%d\nn=%d\nk=%d\n", props.name, props.major, props.minor, kM, kN, kK);
  std::printf("max_abs_ref=%g\nmax_abs_error=%g\nmax_rel_error=%g\nnonfinite_count=%lld\nelapsed_ms=%.4f\ntflops=%.2f\n",
              max_abs_ref, max_abs_err, max_rel_error, n_bad,
              elapsed_ms, 2.0 * kM * kN * kK / (elapsed_ms * 1e-3) / 1e12);

  if (n_bad != 0) return fail("output contains NaN/Inf");
  if (max_abs_ref <= 0.0) return fail("reference output is all zeros");
  if (sum_abs_got <= 0.0) return fail("device output is all zeros");
  if (!(max_rel_error <= kTolerance)) return fail("numerical mismatch exceeds tolerance");
  std::printf("FP4_GEMM_PASS\n");
  return 0;
}

} // namespace
#endif

int main() {
  int dev = 0;
  cudaDeviceProp props{};
  if (cudaGetDevice(&dev) != cudaSuccess || cudaGetDeviceProperties(&props, dev) != cudaSuccess)
    return fail("no usable CUDA device");
  if (!(props.major == 12 && (props.minor == 0 || props.minor == 1))) {
    std::printf("gpu_name=%s\ncompute_cap=%d.%d\n", props.name, props.major, props.minor);
    std::printf("FP4_GEMM_SKIP: NVFP4 block-scaled GEMM requires compute capability 12.0/12.1\n");
    return 2;
  }
#if defined(CUTLASS_ARCH_MMA_SM120_SUPPORTED) || defined(CUTLASS_ARCH_MMA_SM121_SUPPORTED)
  return run();
#else
  return fail("binary was not compiled with SM120/SM121 block-scaled MMA support (need -arch=sm_120a/sm_121a)");
#endif
}
