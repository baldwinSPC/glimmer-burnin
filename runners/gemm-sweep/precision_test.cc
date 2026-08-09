// precision_test.cc — the decisions in precision.h, driven exhaustively.
//
// SPDX-License-Identifier: Apache-2.0
// Copyright the Glimmer authors.
//
// Compiled and run by runners/cxxtests_test.go under `make test`. No GPU, no
// CUDA toolchain, no hardware — which is the point: every branch here can
// produce a wrong VERDICT about somebody's hardware, and none of them needs
// silicon to be wrong.

#include "precision.h"

#include <cstdio>
#include <string>

static int failures = 0;

static void check(bool ok, const std::string& what) {
  if (!ok) {
    std::printf("FAIL: %s\n", what.c_str());
    ++failures;
  }
}

using burnin::Precision;
using burnin::Scope;

// Every spelling an author might reasonably write resolves.
static void testParsing() {
  struct Case { const char* in; Precision want; };
  const Case cases[] = {
      {"fp4", Precision::FP4},     {"nvfp4", Precision::FP4},
      {"FP4", Precision::FP4},     {"  fp4  ", Precision::FP4},
      {"fp8", Precision::FP8},     {"e4m3", Precision::FP8},
      {"bf16", Precision::BF16},   {"bfloat16", Precision::BF16},
      {"BF16", Precision::BF16},
      {"tf32", Precision::TF32},   {"TF32", Precision::TF32},
      {"fp64", Precision::FP64},   {"double", Precision::FP64},
  };
  for (const auto& c : cases) {
    check(burnin::parsePrecision(c.in) == c.want,
          std::string("parsePrecision(\"") + c.in + "\")");
  }
}

// Anything else is Unknown, and Unknown is NEVER a guess at a precision.
//
// Guessing would produce a measurement gated by thresholds written for the
// precision that was actually asked for: an fp4 tolerance accepts a broken part
// at fp64, and an fp64 tolerance condemns a working one at fp4.
static void testUnrecognisedIsNeverGuessed() {
  const char* bad[] = {"", "fp16", "half", "int8", "fp32", "float", "f p 4",
                       "fp4x", "4", "true", "precision"};
  for (const char* in : bad) {
    check(burnin::parsePrecision(in) == Precision::Unknown,
          std::string("parsePrecision(\"") + in + "\") should be Unknown");
  }
}

// An unrecognised precision is REFUSED, not skipped.
//
// "We do not know what you asked for" is true whatever silicon is in the box.
// Reporting Skip would tell the operator the test does not apply to this
// hardware, which is a claim about the part that nobody established.
static void testUnknownRefusesBeforeLookingAtTheHardware() {
  const int caps[] = {0, 60, 80, 90, 100, 121, 999};
  for (int cap : caps) {
    check(burnin::scopeOf(Precision::Unknown, cap) == Scope::Refuse,
          "Unknown precision must Refuse at cc " + std::to_string(cap));
  }
}

// A capability we could not read is not a capability of zero.
//
// Zero compares below every minimum and would report Skip on every part, which
// reads as "this fleet does not support fp8" when what happened is that the
// runtime query failed.
static void testAnUnreadableCapabilityRefusesRatherThanSkips() {
  for (Precision p : burnin::everyPrecision()) {
    for (int cap : {0, -1, -100}) {
      check(burnin::scopeOf(p, cap) == Scope::Refuse,
            std::string("cc ") + std::to_string(cap) + " with " +
                burnin::precisionName(p) + " must Refuse, not Skip");
    }
  }
}

// The support matrix, stated exhaustively. Each row is a claim about NVIDIA
// silicon, so each is written out rather than derived from the same table the
// code uses.
static void testTheSupportMatrix() {
  struct Row { int cc; bool fp4, fp8, bf16, tf32, fp64; };
  const Row rows[] = {
      // Pascal / Volta / Turing: doubles only, from this sweep's point of view.
      {60, false, false, false, false, true},
      {70, false, false, false, false, true},
      {75, false, false, false, false, true},
      // Ampere brings BF16 and TF32.
      {80, false, false, true, true, true},
      {86, false, false, true, true, true},
      // Ada.
      {89, false, false, true, true, true},
      // Hopper brings FP8.
      {90, false, true, true, true, true},
      // Blackwell brings block-scaled NVFP4.
      {100, true, true, true, true, true},
      {120, true, true, true, true, true},
      {121, true, true, true, true, true},
  };
  for (const auto& r : rows) {
    const std::string at = " at cc " + std::to_string(r.cc);
    check((burnin::scopeOf(Precision::FP4, r.cc) == Scope::Run) == r.fp4, "fp4" + at);
    check((burnin::scopeOf(Precision::FP8, r.cc) == Scope::Run) == r.fp8, "fp8" + at);
    check((burnin::scopeOf(Precision::BF16, r.cc) == Scope::Run) == r.bf16, "bf16" + at);
    check((burnin::scopeOf(Precision::TF32, r.cc) == Scope::Run) == r.tf32, "tf32" + at);
    check((burnin::scopeOf(Precision::FP64, r.cc) == Scope::Run) == r.fp64, "fp64" + at);
  }
}

// FP64 applies everywhere, and must never skip on a real part.
//
// Its RATE varies enormously between datacentre and consumer silicon, which is a
// threshold-authoring problem and not a support question. The instruction
// exists, so the test applies — skipping it would report "does not apply" about
// a part that computes doubles perfectly well, just slowly.
static void testFP64NeverSkipsOnARealPart() {
  for (int cc = 50; cc <= 130; ++cc) {
    check(burnin::scopeOf(Precision::FP64, cc) == Scope::Run,
          "fp64 must Run at cc " + std::to_string(cc));
  }
}

// A skip is only ever reported for a part we positively established is out of
// scope — never for one we could not read.
static void testSkipIsOnlyForAnEstablishedShortfall() {
  check(burnin::scopeOf(Precision::FP4, 90) == Scope::Skip, "fp4 skips on Hopper");
  check(burnin::scopeOf(Precision::FP8, 80) == Scope::Skip, "fp8 skips on Ampere");
  check(burnin::scopeOf(Precision::BF16, 70) == Scope::Skip, "bf16 skips on Volta");
  // And the boundary is inclusive: the first supporting arch runs.
  check(burnin::scopeOf(Precision::FP4, 100) == Scope::Run, "fp4 runs on the first Blackwell");
  check(burnin::scopeOf(Precision::FP8, 90) == Scope::Run, "fp8 runs on the first Hopper");
  check(burnin::scopeOf(Precision::BF16, 80) == Scope::Run, "bf16 runs on the first Ampere");
}

// The messages carry what a reader months later cannot otherwise recover.
static void testMessagesAreActionable() {
  const std::string skip = burnin::skipMessage(Precision::FP4, 90);
  check(skip.rfind("GEMM_SWEEP_SKIP:", 0) == 0,
        "a skip must start with the marker, or the operator records it as an Error");
  check(skip.find("fp4") != std::string::npos, "the skip names the precision");
  check(skip.find("9.0") != std::string::npos, "the skip names what the part reports");
  check(skip.find("10.0") != std::string::npos, "the skip names what the precision needs");
  check(skip.find("NOT failed") != std::string::npos,
        "the skip says the part is not being failed");

  const std::string bad = burnin::refuseMessage(Precision::Unknown, "fp16", 121);
  check(bad.rfind("GEMM_SWEEP_ERROR:", 0) == 0, "an error says so");
  check(bad.find("fp16") != std::string::npos, "the refusal quotes what was asked for");
  check(bad.find("fp4") != std::string::npos && bad.find("fp64") != std::string::npos,
        "the refusal lists what IS accepted");

  const std::string unread = burnin::refuseMessage(Precision::FP8, "fp8", 0);
  check(unread.find("UNJUDGED") != std::string::npos,
        "an unreadable capability leaves the part unjudged, and says so");
}

// Every precision has a name, and no two share one.
static void testNamesAreDistinct() {
  for (Precision a : burnin::everyPrecision()) {
    check(std::string(burnin::precisionName(a)) != "unknown",
          "a real precision must not be named \"unknown\"");
    for (Precision b : burnin::everyPrecision()) {
      if (a == b) continue;
      check(std::string(burnin::precisionName(a)) != burnin::precisionName(b),
            "two precisions share a name, so a stored result cannot say which ran");
    }
  }
  // And every name round-trips back through the parser, so the label a result
  // carries is one a profile could have asked for.
  for (Precision p : burnin::everyPrecision()) {
    check(burnin::parsePrecision(burnin::precisionName(p)) == p,
          std::string("precisionName/parsePrecision do not round-trip for ") +
              burnin::precisionName(p));
  }
}

int main() {
  testParsing();
  testUnrecognisedIsNeverGuessed();
  testUnknownRefusesBeforeLookingAtTheHardware();
  testAnUnreadableCapabilityRefusesRatherThanSkips();
  testTheSupportMatrix();
  testFP64NeverSkipsOnARealPart();
  testSkipIsOnlyForAnEstablishedShortfall();
  testMessagesAreActionable();
  testNamesAreDistinct();

  if (failures != 0) {
    std::printf("%d check(s) failed\n", failures);
    return 1;
  }
  std::printf("all precision checks passed\n");
  return 0;
}
