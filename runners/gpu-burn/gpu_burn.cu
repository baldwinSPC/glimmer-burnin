// gpu_burn.cu — the "gpu-burn" TestKind.
//
// SPDX-License-Identifier: Apache-2.0
// Copyright the Glimmer authors.
//
// This file is original work licensed under Apache-2.0 and uses no third-party
// source. The load generator, the NVML sampler and the reporting engine live in
// soak_core.cuh, shared byte-for-byte with runners/thermal-soak; this file is
// only the kind's stdout vocabulary and its assertions.
//
// WHY THIS IS NOT wilicc/gpu-burn
// -------------------------------
// The obvious implementation of this TestKind is Ville Timonen's gpu-burn
// (github.com/wilicc/gpu-burn, BSD-2-Clause), and the kind is named after it.
// Its licence is fine. What it links is not: gpu-burn's whole numeric core is
// cuBLAS SGEMM/DGEMM, cuBLAS is covered by the NVIDIA CUDA EULA, and the
// container toolkit does NOT inject it at runtime the way it injects
// libcuda.so.1 and libnvidia-ml.so.1 — those come from the host DRIVER, whereas
// cuBLAS is a toolkit library. So a working gpu-burn image has to carry
// libcublas itself, which is exactly the redistribution this project's runner
// images may not do, and the alternative is the posture runners/dcgm-diag was
// forced into: ship a wrapper and make every site mount the NVIDIA component
// itself. That is a real cost — dcgm-diag reports Error, not a verdict, on any
// node where the site's mount is missing — and it is worth paying only when the
// upstream tool does something we cannot.
//
// Here it does not. gpu-burn's actual method is three lines of idea: run a big
// GEMM over and over, compare each result against the first one, count the
// elements that differ. The value is in the method, not in cuBLAS, and a
// self-contained GEMM makes the image redistributable, removes the site mount,
// and — because it is the same kernel thermal-soak runs — makes the two kinds'
// throughput figures the same measurement rather than two numbers under one
// name. cuBLAS would reach a higher absolute TFLOPS; it would not detect one
// more wrong answer.
//
// (Checked on the hardware rather than assumed: cuBLAS 13 does support sm_121
// and does run on GB10. The blocker is redistribution, not capability.)
//
// WHAT THIS CATCHES THAT NOTHING ELSE DOES
// ----------------------------------------
// A part that returns the WRONG ANSWER and reports success. Not a crash, not an
// Xid, not an ECC event — a bit that is different this time from last time, on
// hardware whose every status register says it is healthy. On a part with ECC
// that fault is usually caught and counted; on a part without visible ECC —
// GB10's unified LPDDR5X has on-die ECC that NVML cannot see — a recompute-and-
// compare burn is the only instrument the fleet has for it.
//
// That is also why this kind's verdict is correctness and NOT temperature or
// clock: those belong to thermal-soak, which runs the identical load. Two kinds,
// two assertion sets, one implementation.

#include "soak_core.cuh"

namespace {

// The stdout vocabulary for this kind. These spellings are not free: the
// operator's alias table (pkg/runner/parse.go, "gpu-burn") already maps them,
// and it maps them to upstream gpu_burn's own words so that a site running the
// upstream tool and a site running this image produce the same canonical
// metrics. "errors" is gpu_burn's name for what we call miscompares; "tflops"
// is its sustained figure, which the table maps to sustainedThroughputTflops
// rather than to throughputTflops — the latter is compute-smoke's single
// unwarmed launch and is marked unsafe to threshold on, so sharing the name
// would silently make this benchmark ungateable.
const soak::Keys kKeys = {
    /*markerPrefix=*/"GPU_BURN",
    /*elapsed=*/"elapsed_s=",
    /*temp=*/"gpu_temp_c=",
    /*power=*/"power_draw_w=",
    /*throttle=*/"throttle_events=",
    /*iterations=*/"iterations=",
    /*miscompares=*/"errors=",
    /*tflops=*/"tflops=",
};

} // namespace

int main() {
  soak::Measurement m;
  const int rc = soak::run(kKeys, &m);
  if (rc != 0) return rc; // Skip or Error; the marker is already printed.

  // Every metric is already on stdout, so each branch from here leaves a
  // complete record behind it.
  char reason[512];

  if (m.miscompares > 0) {
    std::snprintf(reason, sizeof(reason),
                  "%llu wrong element(s) across %ld incident(s) in %ld iterations at up to %.0fC — "
                  "the part returned a different answer to identical arithmetic",
                  m.miscompares, m.sdcDetections, m.iterations, m.tempKnown ? m.peakTempC : 0.0);
    return soak::fail(kKeys, reason);
  }
  if (m.nonfinite > 0) {
    std::snprintf(reason, sizeof(reason),
                  "%llu non-finite value(s) in a result whose operands are all finite",
                  m.nonfinite);
    return soak::fail(kKeys, reason);
  }
  // Only when a real delta was computed. A part that declared it has no ECC
  // subsystem reported the "n/a" sentinel instead, and is judged on the
  // miscompare count alone — which is the whole reason this test exists on such
  // a part.
  if (m.eccKnown && m.eccErrors > 0) {
    std::snprintf(reason, sizeof(reason), "%llu ECC error(s) accumulated during the burn",
                  m.eccErrors);
    return soak::fail(kKeys, reason);
  }

  return soak::pass(kKeys);
}
