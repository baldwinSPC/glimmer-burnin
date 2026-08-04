// arch_match_test.cc — tests for the two pre-launch decisions in arch_match.h:
// whether the PART is in scope at all (Skip), and whether this IMAGE carries a
// cubin for it (Error).
//
// SPDX-License-Identifier: Apache-2.0
// Copyright the Glimmer authors.
//
// Build and run:
//
//   c++ -std=c++17 -Wall -Wextra -o /tmp/arch_match_test arch_match_test.cc && /tmp/arch_match_test
//
// runners/cxxtests_test.go does exactly that, so this runs under `make test` and
// in CI with no Docker, no CUDA toolchain and no GPU. That is the whole reason
// the logic lives in a CUDA-free header: both decisions were previously
// reachable only by putting the right silicon, or a deliberately mis-built
// image, in front of the runner — and for the Skip, the right silicon is
// hardware this project does not have at all (issue #9).
//
// A plain program with no test framework, following memory-bw/scan_test.cc, so
// it needs nothing beyond a C++ compiler. It is NOT compiled into the runner
// image; the Dockerfile copies only fp4_smoke.cu and arch_match.h.

#include "arch_match.h"

#include <cstdio>
#include <cstring>

namespace {

int failures = 0;

void check(bool ok, const char *what) {
  if (!ok) {
    std::printf("FAIL: %s\n", what);
    ++failures;
  }
}

const char *name(burnin::ArchMatch m) {
  switch (m) {
    case burnin::ArchMatch::Unknown: return "Unknown";
    case burnin::ArchMatch::Covers: return "Covers";
    case burnin::ArchMatch::Mismatch: return "Mismatch";
  }
  return "?";
}

void expectMatch(const char *arch, int major, int minor, burnin::ArchMatch want, const char *why) {
  const burnin::ArchMatch got = burnin::archMatch(arch, major, minor);
  if (got != want) {
    std::printf("FAIL: archMatch(%s, %d.%d) = %s, want %s — %s\n", arch, major, minor, name(got),
                name(want), why);
    ++failures;
  }
}

bool contains(const char *haystack, const char *needle) {
  return std::strstr(haystack, needle) != nullptr;
}

void expectContains(const char *buf, const char *needle, const char *what) {
  if (!contains(buf, needle)) {
    std::printf("FAIL: %s — message does not mention \"%s\".\n  message: %s\n", what, needle, buf);
    ++failures;
  }
}

const char *name(burnin::Scope s) {
  switch (s) {
    case burnin::Scope::Unknown: return "Unknown";
    case burnin::Scope::InScope: return "InScope";
    case burnin::Scope::OutOfScope: return "OutOfScope";
  }
  return "?";
}

void expectScope(int major, int minor, burnin::Scope want, const char *why) {
  const burnin::Scope got = burnin::scopeOf(major, minor);
  if (got != want) {
    std::printf("FAIL: scopeOf(%d.%d) = %s, want %s — %s\n", major, minor, name(got), name(want),
                why);
    ++failures;
  }
}

// ── the SCOPE decision, which is the runner's only Skip ──────────────────────
//
// Issue #9: exit 2 had never executed. It fires only on a part that is not
// compute capability 12.0/12.1, and the only accelerator available to this
// project is a CC 12.1 GB10, which always takes the PASS path. dcgm-diag was
// expected to be the first real-silicon exerciser on the theory that DCGM does
// not support GB10; it does, so that route is gone.
//
// This is why the decision was lifted out of fp4_smoke.cu's main() and into a
// CUDA-free header: the branch that cannot be reached on the hardware in the
// room can still be driven to every corner by a plain C++ program. Exit 2 is the
// contract's load-bearing distinction — a node that cannot run a test has NOT
// failed it — and the cost of getting it wrong is the same false-negative class
// that made healthy Sparks look broken.

void testScopeGate() {
  // In scope: the whole SM12x family, not just the arch a given image is built
  // for. Both are real parts, and the family is deliberate — narrowing this to
  // the built arch would make a wrong-arch IMAGE look like out-of-scope
  // HARDWARE, and a fleet would report Skip when someone had pinned a bad tag.
  expectScope(12, 1, burnin::Scope::InScope, "GB10 / DGX Spark, the part this runner was written for");
  expectScope(12, 0, burnin::Scope::InScope, "the other SM12x member; the gate is the family");

  // Out of scope: real silicon that genuinely cannot do NVFP4 block-scaled
  // GEMM. These are the parts that would take the Skip, and the ones this
  // project has none of — which is the whole reason this test exists.
  expectScope(9, 0, burnin::Scope::OutOfScope, "H100 (sm_90): no block-scaled FP4");
  expectScope(8, 9, burnin::Scope::OutOfScope, "L40S (sm_89)");
  expectScope(8, 0, burnin::Scope::OutOfScope, "A100 (sm_80)");
  expectScope(7, 5, burnin::Scope::OutOfScope, "T4 (sm_75)");
  expectScope(7, 0, burnin::Scope::OutOfScope, "V100 (sm_70)");
  expectScope(6, 1, burnin::Scope::OutOfScope, "Pascal");

  // B200 is SM10x. It is NOT covered by this runner and must Skip here rather
  // than be admitted and then fail somewhere downstream — SM10x needs its own
  // image (issue #10), and a Skip is how this one says so honestly.
  expectScope(10, 0, burnin::Scope::OutOfScope, "B200 (sm_100) needs its own runner image");
  expectScope(10, 3, burnin::Scope::OutOfScope, "any other SM10x member");

  // A newer minor in the same major is NOT admitted. The gate names the two
  // capabilities the kernel is known to run on; a hypothetical 12.2 has not been
  // established, and quietly admitting it would launch an untested path.
  expectScope(12, 2, burnin::Scope::OutOfScope, "a future SM12x minor is not assumed to work");
  expectScope(12, 9, burnin::Scope::OutOfScope, "nor any later one");

  // A future major is likewise out of scope rather than in.
  expectScope(13, 0, burnin::Scope::OutOfScope, "a future architecture");

  // THE RULE THAT MATTERS MOST HERE. A capability that was never read is
  // Unknown, never OutOfScope. Answering OutOfScope would let a node whose
  // properties could not be interrogated record a clean Skip — an acceptance
  // decision taken without looking at anything — which is the same failure as a
  // fabricated measurement. Unknown runs on, and the existing error paths speak.
  expectScope(-1, -1, burnin::Scope::Unknown, "an unread capability is unjudged, not unsupported");
  expectScope(-1, 1, burnin::Scope::Unknown, "a half-read capability is still unread");
  expectScope(12, -1, burnin::Scope::Unknown, "and a negative minor does not become 12.0");

  // The two questions must not be collapsed: a part can be perfectly in scope
  // and still be one this IMAGE has no cubin for. That combination is an Error
  // (wrong tag pinned), never a Skip, and both halves are asserted together here
  // because the bug would be silent if either drifted alone.
  expectScope(12, 0, burnin::Scope::InScope, "an sm_121a image on a CC 12.0 part: the PART is in scope");
  expectMatch("sm_121a", 12, 0, burnin::ArchMatch::Mismatch,
              "...and the IMAGE is the thing that does not cover it");
}

// The Skip message has to name what was found. "requires compute capability
// 12.0/12.1" alone leaves an operator to go and discover what the node actually
// has, and a Skip nobody can attribute is a Skip nobody trusts.
void testOutOfScopeMessage() {
  char buf[512];
  const int code = burnin::describeOutOfScope(buf, sizeof(buf), 9, 0);

  check(code == burnin::kExitSkip, "an out-of-scope part exits Skip");
  check(code != burnin::kExitFail, "an out-of-scope part is NEVER a hardware verdict");
  check(code != burnin::kExitError, "an out-of-scope part is not an infrastructure error either");
  expectContains(buf, "9.0", "the skip message names the capability that was found");
  expectContains(buf, "12.0/12.1", "the skip message names the capability that is required");
  expectContains(buf, "NOT failed", "the skip message says plainly that the part is not failed");

  // Swept: whatever the part reports, a Skip is a Skip and the message is never
  // empty. Nothing here may return the hardware verdict.
  const int caps[][2] = {{9, 0}, {8, 9}, {10, 0}, {13, 7}, {0, 0}, {-1, -1}};
  for (const auto &cap : caps) {
    char b[512];
    if (burnin::describeOutOfScope(b, sizeof(b), cap[0], cap[1]) != burnin::kExitSkip) {
      std::printf("FAIL: describeOutOfScope(%d.%d) did not exit Skip\n", cap[0], cap[1]);
      ++failures;
    }
    if (b[0] == '\0') {
      std::printf("FAIL: describeOutOfScope(%d.%d) produced an empty message\n", cap[0], cap[1]);
      ++failures;
    }
  }

  // A short buffer truncates rather than overruns, as everywhere else here.
  char guard[64];
  std::memset(guard, 'X', sizeof(guard));
  burnin::describeOutOfScope(guard, 32, 9, 0);
  check(std::strlen(guard) < 32, "a short skip buffer truncates and stays NUL-terminated");
  for (std::size_t i = 32; i < sizeof(guard); ++i) {
    if (guard[i] != 'X') {
      std::printf("FAIL: describeOutOfScope wrote past the buffer it was given\n");
      ++failures;
      break;
    }
  }
}

// ── the coverage decision ────────────────────────────────────────────────────

void testArchMatch() {
  // THE REGRESSION. An sm_120a image on a CC 12.1 GB10: the loader accepts the
  // cubin (SM121 is binary-compatible with SM120) and the kernel then trips a
  // device-side assert, so nothing in the CUDA error tells an operator that the
  // image is wrong. Establishing it from the arch string is the fix.
  expectMatch("sm_120a", 12, 1, burnin::ArchMatch::Mismatch,
              "reproduced on real hardware: loads, then asserts inside the kernel");

  // The direction that was always caught, by the loader refusing outright.
  expectMatch("sm_121a", 12, 0, burnin::ArchMatch::Mismatch, "no cubin for a CC 12.0 part");

  // Arch-specific targets cover exactly their own capability.
  expectMatch("sm_121a", 12, 1, burnin::ArchMatch::Covers, "the default build on GB10");
  expectMatch("sm_120a", 12, 0, burnin::ArchMatch::Covers, "exact match");
  expectMatch("sm_90a", 9, 0, burnin::ArchMatch::Covers, "two-digit target, exact match");
  expectMatch("sm_90a", 10, 0, burnin::ArchMatch::Mismatch, "no PTX fallback on an 'a' target");

  // Family-specific targets cover their base and later minors of that major.
  expectMatch("sm_120f", 12, 0, burnin::ArchMatch::Covers, "sm_120f covers CC 12.0");
  expectMatch("sm_120f", 12, 1, burnin::ArchMatch::Covers, "sm_120f covers CC 12.1");
  expectMatch("sm_120f", 11, 8, burnin::ArchMatch::Mismatch, "a different major family");
  expectMatch("sm_121f", 12, 0, burnin::ArchMatch::Mismatch, "a family does not run backwards");

  // Plain targets embed PTX, so the driver can JIT forward but never backwards.
  expectMatch("sm_120", 12, 1, burnin::ArchMatch::Covers, "PTX JITs forward");
  expectMatch("sm_120", 13, 0, burnin::ArchMatch::Covers, "PTX JITs onto a newer major");
  expectMatch("sm_120", 11, 8, burnin::ArchMatch::Mismatch, "PTX cannot JIT backwards");

  // Unknown is a real answer, and it must never become Mismatch: a runner may
  // only declare what it positively established, and an unrecognised arch
  // string establishes nothing. Each of these leaves the runtime error to speak.
  expectMatch("unknown", 12, 1, burnin::ArchMatch::Unknown, "the hand-built fallback");
  expectMatch("", 12, 1, burnin::ArchMatch::Unknown, "empty");
  expectMatch("sm_", 12, 1, burnin::ArchMatch::Unknown, "no digits");
  expectMatch("sm_1", 12, 1, burnin::ArchMatch::Unknown, "one digit is not a capability");
  expectMatch("compute_120", 12, 1, burnin::ArchMatch::Unknown, "a PTX target, not an sm_ target");
  expectMatch("sm_121x", 12, 1, burnin::ArchMatch::Unknown, "a suffix we have not been taught");
  expectMatch("sm_121aa", 12, 1, burnin::ArchMatch::Unknown, "trailing junk");
  expectMatch("sm_121a ", 12, 1, burnin::ArchMatch::Unknown, "trailing space");
  expectMatch("SM_121A", 12, 1, burnin::ArchMatch::Unknown, "nvcc targets are lower case");
  expectMatch("sm_12345", 12, 1, burnin::ArchMatch::Unknown, "too many digits to be a capability");
  check(burnin::archMatch(nullptr, 12, 1) == burnin::ArchMatch::Unknown, "a null arch is Unknown");

  // A capability we never read is Unknown, not a mismatch. This is the state
  // during main()'s first two CUDA calls, which can fail before there is
  // anything to latch.
  expectMatch("sm_121a", -1, -1, burnin::ArchMatch::Unknown, "device properties were never read");
  expectMatch("sm_121a", 12, -1, burnin::ArchMatch::Unknown, "half-read properties");

  // Capability parsing: the LAST digit is the minor version.
  int major = 0, minor = 0;
  char suffix = '\0';
  check(burnin::parseSmTarget("sm_121a", &major, &minor, &suffix) && major == 12 && minor == 1 &&
            suffix == 'a',
        "sm_121a parses as 12.1, arch-specific");
  check(burnin::parseSmTarget("sm_90", &major, &minor, &suffix) && major == 9 && minor == 0 &&
            suffix == '\0',
        "sm_90 parses as 9.0, plain");
  check(burnin::parseSmTarget("sm_100f", &major, &minor, &suffix) && major == 10 && minor == 0 &&
            suffix == 'f',
        "sm_100f parses as 10.0, family-specific");
}

// ── the message, and the exit code ───────────────────────────────────────────

void testPreLaunchMismatchMessage() {
  char buf[768];
  const int code = burnin::describeArchMismatch(buf, sizeof(buf), "sm_120a", 12, 1);

  check(code == burnin::kExitError, "a pre-launch mismatch exits Error");
  check(code != burnin::kExitFail, "a pre-launch mismatch is NEVER a hardware verdict");
  expectContains(buf, "sm_120a", "pre-launch mismatch names the arch it was built for");
  expectContains(buf, "12.1", "pre-launch mismatch names the capability it found");
  expectContains(buf, "UNJUDGED", "pre-launch mismatch says the hardware is unjudged");
  expectContains(buf, "sm_121a", "pre-launch mismatch names the arch that would work");
  expectContains(buf, "CUDA_ARCH", "pre-launch mismatch names the build arg to change");
}

void testWrongImageOutranksTheRuntimeError() {
  // The reproduced case, reached through the post-launch path — which is what
  // an image built for an arch this file cannot pre-judge would still hit.
  char buf[768];
  const int code = burnin::describeCudaFailure(
      buf, sizeof(buf), "warmup cudaDeviceSynchronize", burnin::CudaFailure::DeviceAssert,
      "device-side assert triggered", "sm_120a", 12, 1);

  check(code == burnin::kExitError, "a device-side assert exits Error");
  expectContains(buf, "warmup cudaDeviceSynchronize", "the failure names where it happened");
  expectContains(buf, "device-side assert triggered", "the runtime's own string is preserved");
  expectContains(buf, "sm_120a", "the assert path names the arch it was built for");
  expectContains(buf, "12.1", "the assert path names the capability it found");
  expectContains(buf, "UNJUDGED", "the assert path says the hardware is unjudged");
  expectContains(buf, "sm_121a", "the assert path names the arch that would work");

  // An established mismatch outranks whichever code the runtime happened to
  // latch, so an illegal address on a wrong-arch image reads the same way.
  char other[768];
  burnin::describeCudaFailure(other, sizeof(other), "warmup gemm.run failed",
                              burnin::CudaFailure::Other, "an illegal memory access was encountered",
                              "sm_120a", 12, 1);
  expectContains(other, "no cubin for the part it landed on",
                 "a mismatch explains a generic error too");
}

void testWithoutAnEstablishedMismatch() {
  char buf[768];

  // The loader refused, but the arch string cannot be compared — a hand-built
  // binary. The pre-existing message stands: name the arch, claim no capability.
  burnin::describeCudaFailure(buf, sizeof(buf), "warmup gemm.run failed",
                              burnin::CudaFailure::NoKernelImage,
                              "no kernel image is available for execution on the device", "unknown",
                              12, 1);
  expectContains(buf, "no cubin for the part it landed on", "the loader's refusal is still named");
  expectContains(buf, "unknown", "it names the arch it was built for, such as it is");
  check(!contains(buf, "compute capability 12.1"),
        "an unknown arch must not claim a capability comparison it never made");

  // A device-side assert on an image whose arch DOES cover the part is not the
  // wrong-image case and must not be reported as one.
  burnin::describeCudaFailure(buf, sizeof(buf), "warmup cudaDeviceSynchronize",
                              burnin::CudaFailure::DeviceAssert, "device-side assert triggered",
                              "sm_121a", 12, 1);
  check(!contains(buf, "no cubin for the part it landed on"),
        "a matching arch is never reported as a wrong image");
  expectContains(buf, "poisoned", "a device-side assert says the context is dead");

  // An ordinary error on a matching image stays the plain "where: what".
  burnin::describeCudaFailure(buf, sizeof(buf), "cudaEventCreate", burnin::CudaFailure::Other,
                              "out of memory", "sm_121a", 12, 1);
  check(std::strcmp(buf, "cudaEventCreate: out of memory") == 0,
        "an unremarkable CUDA error is reported unremarkably");

  // No error latched where one was expected.
  burnin::describeCudaFailure(buf, sizeof(buf), "can_implement rejected the problem",
                              burnin::CudaFailure::NoErrorLatched, nullptr, "sm_121a", 12, 1);
  expectContains(buf, "no CUDA error was latched", "an unlatched error says so");
}

// THE property, swept rather than spot-checked: exit 1 is the expensive code —
// it is a hardware verdict, the operator never retries it, and it permanently
// indicts a node. Nothing in this header describes a measurement of the
// silicon, so nothing in it may return anything but Error, whatever the CUDA
// error, whatever the image, whatever the part.
void testNothingHereIsEverAHardwareVerdict() {
  const burnin::CudaFailure kinds[] = {
      burnin::CudaFailure::NoErrorLatched, burnin::CudaFailure::NoKernelImage,
      burnin::CudaFailure::DeviceAssert, burnin::CudaFailure::Other};
  const char *arches[] = {"sm_121a", "sm_120a", "sm_120f", "sm_120", "unknown", "", "sm_121x"};
  const int caps[][2] = {{12, 0}, {12, 1}, {9, 0}, {13, 0}, {-1, -1}};
  const char *strings[] = {"device-side assert triggered", "out of memory", "", nullptr};

  char buf[768];
  for (burnin::CudaFailure kind : kinds) {
    for (const char *arch : arches) {
      for (const auto &cap : caps) {
        for (const char *s : strings) {
          const int code =
              burnin::describeCudaFailure(buf, sizeof(buf), "somewhere", kind, s, arch, cap[0], cap[1]);
          if (code != burnin::kExitError) {
            std::printf("FAIL: describeCudaFailure(arch=%s, cc=%d.%d) returned %d, want %d\n", arch,
                        cap[0], cap[1], code, burnin::kExitError);
            ++failures;
          }
          if (buf[0] == '\0') {
            std::printf("FAIL: describeCudaFailure(arch=%s, cc=%d.%d) produced an empty message\n",
                        arch, cap[0], cap[1]);
            ++failures;
          }
        }
        if (burnin::describeArchMismatch(buf, sizeof(buf), arch, cap[0], cap[1]) !=
            burnin::kExitError) {
          std::printf("FAIL: describeArchMismatch(arch=%s) did not exit Error\n", arch);
          ++failures;
        }
      }
    }
  }
}

// A short buffer must truncate, not overrun, and must stay NUL-terminated: the
// message is built from strings the runtime supplies and a caller could
// reasonably size a buffer smaller than the longest of them.
void testShortBufferTruncatesSafely() {
  char guard[64];
  std::memset(guard, 'X', sizeof(guard));
  burnin::describeCudaFailure(guard, 32, "warmup cudaDeviceSynchronize",
                              burnin::CudaFailure::DeviceAssert, "device-side assert triggered",
                              "sm_120a", 12, 1);
  check(std::strlen(guard) < 32, "a short buffer truncates and stays NUL-terminated");
  for (std::size_t i = 32; i < sizeof(guard); ++i) {
    if (guard[i] != 'X') {
      std::printf("FAIL: describeCudaFailure wrote past the buffer it was given\n");
      ++failures;
      break;
    }
  }
}

}  // namespace

int main() {
  testScopeGate();
  testOutOfScopeMessage();
  testArchMatch();
  testPreLaunchMismatchMessage();
  testWrongImageOutranksTheRuntimeError();
  testWithoutAnEstablishedMismatch();
  testNothingHereIsEverAHardwareVerdict();
  testShortBufferTruncatesSafely();

  if (failures != 0) {
    std::printf("%d check(s) failed\n", failures);
    return 1;
  }
  std::printf("arch_match_test: OK\n");
  return 0;
}
