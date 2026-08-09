// precision.h — which precision this execution runs, and whether the part in
// front of us can run it.
//
// SPDX-License-Identifier: Apache-2.0
// Copyright the Glimmer authors.
//
// CUDA-FREE ON PURPOSE, and this is the same lesson compute-smoke's arch_match.h
// records: the part of a GPU runner worth unit-testing is never the part that
// needs a GPU. Every decision that can produce a wrong VERDICT lives here —
// which precision was asked for, whether this part supports it, and what to say
// when it does not — so it can be driven exhaustively by a plain C++ test under
// `make test`, on a laptop, with no accelerator anywhere.
//
// The GEMM itself is in gemm_sweep.cu and needs silicon. The judgements do not.

#ifndef BURNIN_GEMM_SWEEP_PRECISION_H
#define BURNIN_GEMM_SWEEP_PRECISION_H

#include <cstddef>
#include <string>
#include <vector>

namespace burnin {

// Precision is what this execution measures. One cell of the sweep.
enum class Precision {
  Unknown,  // the axis value was not recognised — never guessed at
  FP4,      // block-scaled NVFP4, tensor core
  FP8,      // e4m3, tensor core
  BF16,     // tensor core
  TF32,     // tensor core, fed by FP32 inputs
  FP64,     // CUDA core on most parts, and a different unit entirely
};

// Scope is what this binary should do about the part it landed on.
//
// The three outcomes map onto the runner contract exactly, and the mapping is
// the whole reason this header exists as a separate testable unit:
//
//   Run    -> measure, then exit 0 or 1 on what was measured
//   Skip   -> exit 2 WITH a GEMM_SWEEP_SKIP marker: the test does not apply
//   Refuse -> exit 3: we cannot measure, so the part is UNJUDGED
enum class Scope {
  Run,
  Skip,
  Refuse,
};

// parsePrecision reads the `precision` variant axis.
//
// It NEVER guesses. An unrecognised value returns Unknown, which the caller
// turns into exit 3 — because running some other precision would produce a
// measurement gated by thresholds written for the one that was asked for, and
// an FP4 tolerance applied to an FP64 result accepts a broken part while an
// FP64 tolerance applied to FP4 condemns a working one.
//
// Case-insensitive and whitespace-tolerant: how an author capitalises a YAML
// value is not a reason to leave hardware unmeasured.
inline Precision parsePrecision(const std::string& raw) {
  // Trimmed at the ENDS only, and lower-cased. Internal whitespace is not a
  // formatting variation — "f p 4" is a different string from "fp4", and
  // collapsing it would let a typo resolve silently to a real precision. A first
  // version stripped every space and the exhaustive test caught it.
  auto isSpace = [](char c) { return c == ' ' || c == '\t' || c == '\n' || c == '\r'; };
  std::size_t begin = 0, end = raw.size();
  while (begin < end && isSpace(raw[begin])) ++begin;
  while (end > begin && isSpace(raw[end - 1])) --end;

  std::string v;
  v.reserve(end - begin);
  for (std::size_t i = begin; i < end; ++i) {
    const char c = raw[i];
    v += static_cast<char>(c >= 'A' && c <= 'Z' ? c - 'A' + 'a' : c);
  }
  if (v == "fp4" || v == "nvfp4") return Precision::FP4;
  if (v == "fp8" || v == "e4m3") return Precision::FP8;
  if (v == "bf16" || v == "bfloat16") return Precision::BF16;
  if (v == "tf32") return Precision::TF32;
  if (v == "fp64" || v == "double") return Precision::FP64;
  return Precision::Unknown;
}

inline const char* precisionName(Precision p) {
  switch (p) {
    case Precision::FP4: return "fp4";
    case Precision::FP8: return "fp8";
    case Precision::BF16: return "bf16";
    case Precision::TF32: return "tf32";
    case Precision::FP64: return "fp64";
    case Precision::Unknown: return "unknown";
  }
  return "unknown";
}

// everyPrecision is the closed list, for a test that means to be exhaustive.
//
// Unknown is deliberately absent: it is not a precision, it is the absence of
// one, and a test sweeping "every precision" should not have to special-case it.
inline std::vector<Precision> everyPrecision() {
  return {Precision::FP4, Precision::FP8, Precision::BF16,
          Precision::TF32, Precision::FP64};
}

// minimumComputeCapability is the first architecture whose TENSOR CORES
// implement this precision natively, as (major * 10 + minor).
//
// These are the thresholds that decide a Skip, so each is a claim about NVIDIA
// silicon rather than a preference:
//
//   FP4  — block-scaled NVFP4 arrives with Blackwell: SM100 (B200/GB200) and
//          SM120/SM121 (consumer Blackwell, GB10). CC 10.0.
//   FP8  — Hopper, CC 9.0.
//   BF16 — Ampere, CC 8.0.
//   TF32 — Ampere, CC 8.0.
//   FP64 — every CUDA-capable part this project will ever see has doubles. The
//          RATE varies enormously between datacentre and consumer parts, which
//          is a threshold-authoring problem and not a support question: the
//          instruction exists, so the test applies and must not skip.
//
// Returning 0 means "every part supports this", which is why FP64 is 0 rather
// than some low number that could drift.
inline int minimumComputeCapability(Precision p) {
  switch (p) {
    case Precision::FP4: return 100;
    case Precision::FP8: return 90;
    case Precision::BF16: return 80;
    case Precision::TF32: return 80;
    case Precision::FP64: return 0;
    case Precision::Unknown: return 0;
  }
  return 0;
}

// scopeOf decides what to do, from the axis value and the part's compute
// capability.
//
// The ORDER of these checks is load-bearing. An unrecognised precision is
// refused BEFORE the capability is consulted, because "we do not know what you
// asked for" is true whatever silicon is in the box — and reporting Skip for it
// would tell the operator the test does not apply to this hardware, which is a
// claim about the part that nobody established.
inline Scope scopeOf(Precision p, int computeCapability) {
  if (p == Precision::Unknown) return Scope::Refuse;
  // A capability we could not read is not a capability of zero. Refusing is the
  // honest outcome: a zero would compare below every minimum and report a Skip
  // on every part, which reads as "this fleet does not support FP8" when what
  // actually happened is that the runtime query failed.
  if (computeCapability <= 0) return Scope::Refuse;
  if (computeCapability < minimumComputeCapability(p)) return Scope::Skip;
  return Scope::Run;
}

// skipMessage is what the operator stores when this part is out of scope.
//
// It names the precision, what the part reports, and what the precision needs —
// because the reader of a stored Skip months later has none of that context, and
// "not supported" alone cannot be acted on.
inline std::string skipMessage(Precision p, int computeCapability) {
  const int need = minimumComputeCapability(p);
  return std::string("GEMM_SWEEP_SKIP: ") + precisionName(p) +
         " needs compute capability " + std::to_string(need / 10) + "." +
         std::to_string(need % 10) + " or newer for a native tensor-core path, and this part " +
         "reports " + std::to_string(computeCapability / 10) + "." +
         std::to_string(computeCapability % 10) +
         " — the test does not apply to this hardware and the part is NOT failed";
}

// refuseMessage is what the operator stores when we could not measure at all.
inline std::string refuseMessage(Precision p, const std::string& raw, int computeCapability) {
  if (p == Precision::Unknown) {
    return std::string("GEMM_SWEEP_ERROR: BURNIN_VARIANT_PRECISION=\"") + raw +
           "\" is not a precision this runner knows (fp4, fp8, bf16, tf32, fp64). "
           "Refusing rather than measuring something else: a result gated by thresholds "
           "written for a different precision is worse than no result — an fp4 tolerance "
           "accepts a broken part at fp64, and an fp64 tolerance condemns a working one at fp4";
  }
  return std::string("GEMM_SWEEP_ERROR: could not read this device's compute capability (got ") +
         std::to_string(computeCapability) +
         "), so whether " + precisionName(p) +
         " applies to it is unknown — the part is UNJUDGED rather than skipped";
}

}  // namespace burnin

#endif  // BURNIN_GEMM_SWEEP_PRECISION_H
