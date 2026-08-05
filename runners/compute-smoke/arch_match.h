// arch_match.h — the three arch questions compute-smoke answers before it launches.
//
// SPDX-License-Identifier: Apache-2.0
// Copyright the Glimmer authors.
//
// Host-only, and deliberately free of CUDA. These are the decisions an
// operator's verdict hangs off, so they are the part that most needs a test, and
// a header that includes no CUDA can be compiled and exercised by a plain C++
// compiler with no GPU anywhere in sight (see arch_match_test.cc, which
// runners/cxxtests_test.go compiles and runs under `make test`).
//
// WHY THIS EXISTS. fp4_smoke.cu used to learn "this image has no cubin for this
// part" only from the CUDA runtime, via cudaErrorNoKernelImageForDevice. That
// works in the direction the loader refuses outright — an sm_121a image on a CC
// 12.0 part. It does NOT fire in the other direction. Measured on a real GB10
// (CC 12.1, driver 580.82.09, CUDA 13.0.1, CUTLASS v4.6.1), an image built
// `CUDA_ARCH=sm_120a` LOADS — SM121 is binary-compatible with SM120, so the
// loader is satisfied — and then trips a device-side assert inside the CUTLASS
// kernel. What reached the operator was a bare "warmup cudaDeviceSynchronize:
// device-side assert triggered" under several hundred lines of CUTLASS template
// text, and no mention of the arch or the fix.
//
// So the mismatch is established HERE, from the arch string the build baked in
// and the compute capability the device reports, instead of being inferred from
// whichever error the runtime happened to latch. Two properties follow, and
// both matter more than the wording of the message:
//
//   - fp4_smoke.cu can refuse BEFORE the launch. A device-side assert is sticky
//     — it poisons the CUDA context, so nothing further can be measured in the
//     process — and its output is not merely noisy: a container log is stdout
//     and stderr MERGED, and pkg/runner.Parse takes Result.Message from the LAST
//     line that is not key=value. CUTLASS template text arriving after our
//     diagnosis would therefore become the message stored in the TestResult. Not
//     launching is the only fix that removes the spew rather than racing it.
//   - The finding is UNJUDGED, never a verdict. Every path here returns
//     kExitError; see the exit contract below.
//
// THREE QUESTIONS, AND THEY MUST NOT BE COLLAPSED. All three are answered here,
// and keeping them apart is the point of the file:
//
//   scopeOf(major, minor)              "can this part do NVFP4 block-scaled GEMM
//                                      at all?" — a property of the HARDWARE.
//                                      OutOfScope -> Skip, exit 2.
//
//   kernelCovers(major, minor)         "does this image's KERNEL implement the
//                                      MMA path this part uses?" — a property of
//                                      the SOURCE that was compiled.
//                                      WrongFamily -> Error, exit 3.
//
//   archMatch(builtArch, major, minor) "does this image carry a cubin for this
//                                      part?" — a property of the GENCODE the
//                                      image was built with.
//                                      Mismatch -> Error, exit 3.
//
// Collapsing any two misreports one of them. Narrowing the scope gate to the
// family this image happens to cover makes a part that this image simply does
// not serve look like out-of-scope hardware, and a whole fleet reports Skip —
// reading as "acceptance is not applicable to these nodes" — when the truth is
// that acceptance applies perfectly well and nobody ran it. Widening the image
// gates the other way would condemn a genuinely out-of-scope part as a
// configuration error.
//
// THE THIRD QUESTION IS WHY B200 WAS BEING MISREPORTED (issue #10). SM10x
// datacenter Blackwell does NVFP4 block-scaled GEMM in hardware — it is exactly
// the thing this runner exists to verify — but by a different instruction path
// (tcgen05.mma with TMEM accumulation) that needs a different CUTLASS kernel,
// not merely a different gencode of this one. scopeOf used to answer OutOfScope
// for CC 10.x with the comment "needs its own runner image", which is a
// statement about THIS IMAGE wearing the costume of a statement about the
// silicon. A B200 fleet therefore recorded a clean Skip — "the test does not
// apply to this hardware" — for a test that applies and was never run, and the
// run settled Passed around it. That is the exact false negative the two-question
// split was drawn to prevent, reached through the gate that was supposed to
// prevent it.
//
// The failure direction settles the doubt on any part this file is unsure
// about. Being wrong towards InScope costs an Error: hardware unjudged, retried
// under retryOnErrorLimit, and the run does not pass. Being wrong towards
// OutOfScope costs a Skip: acceptance certified as inapplicable to hardware
// nobody looked at, on the one phase the operator never retries. Prefer
// InScope.
//
// The scope gate lives here, rather than inline in fp4_smoke.cu's main(), for
// the reason the rest of this header exists: exit 2 is the contract's
// load-bearing distinction — a node that cannot run a test has NOT failed it —
// and it can only be exercised on hardware this project does not have. Being
// CUDA-free is what lets a plain C++ test drive that decision to every corner
// without a GPU. See issue #9.

#ifndef BURNIN_ARCH_MATCH_H
#define BURNIN_ARCH_MATCH_H

#include <cstddef>
#include <cstdio>

namespace burnin {

// The runner's exit contract. It lives here, rather than in fp4_smoke.cu, so
// that the one property most worth regression-testing about it is reachable
// from a test: NO failure described by this header, on any hardware, for any
// CUDA error, ever returns kExitFail. Exit 1 is a hardware verdict, it is never
// retried by the operator, and it permanently indicts a node — and every case
// this header describes is a reason the measurement did not happen.
constexpr int kExitPass = 0;
constexpr int kExitFail = 1;
constexpr int kExitSkip = 2;
constexpr int kExitError = 3;

// ── the SCOPE gate: is NVFP4 block-scaled GEMM in scope for this part? ───────
//
// This is what produces the runner's only Skip, and Skip is the code that most
// needs to be right: a node that cannot run a test has not failed it, and
// collapsing the two is the false-negative class this runner was written to fix.
// Exit 1 permanently indicts a node and is never retried by the operator.
//
// Deliberately BROADER than what this image can execute — see the three-questions
// note at the top of this file. It answers only "does NVFP4 block-scaled GEMM
// exist on this silicon", so it admits:
//
//   CC 12.0, 12.1   consumer/workstation Blackwell (RTX PRO 6000, RTX 50xx,
//                   GB10). This image's own kernel; runs here.
//   CC 10.x         datacenter Blackwell (B100/B200/GB200 at 10.0, B300/GB300
//                   at 10.3). NVFP4 in hardware by the tcgen05.mma path, which
//                   this image does NOT implement — kernelCovers stops it as an
//                   Error naming issue #10, never as a Skip.
//
// BOTH arms are whole MAJOR families, and the SM12x one became a family after
// #116 made the argument that requires it. A new Blackwell minor is far likelier
// to have NVFP4 than not, and — the part that actually decides it — the cost of
// being wrong runs the safe way in one direction only:
//
//   wrong towards InScope     an Error. Hardware UNJUDGED, retried under
//                             retryOnErrorLimit, and the run does not pass.
//   wrong towards OutOfScope  a Skip. Acceptance certified as inapplicable to
//                             hardware nobody looked at, on the one phase this
//                             operator never retries, and the run settles Passed.
//
// The SM12x arm used to be the exact pair 12.0/12.1, on the reasoning that "a
// hypothetical 12.2 has not been established, and quietly admitting it would
// launch an untested path". The first half is true and the second does not
// follow: admitting a part HERE launches nothing, because kernelCovers and
// archMatch still stand between it and a launch. A CC 12.2 part with this
// image's sm_121a cubin is now stopped by archMatch as an Error naming the
// rebuild, which is the same finding the old comment wanted — reached through
// the gate that says "we did not measure this" instead of the one that says "it
// does not apply to you".
//
// Tri-state, for the same reason ArchMatch is. A Skip is a DECLARATION about the
// hardware — "acceptance does not apply to this part" — and a runner may only
// declare what it positively established. A negative capability means it was
// never read, which is a real state (the first CUDA calls in main() can fail
// before there is anything to latch), and it is emphatically not the same as
// "this part cannot do FP4". Answering OutOfScope there would let an
// uninterrogated node record a clean Skip, which is an acceptance that never
// looked at anything. Unknown runs on and leaves the existing error paths to
// speak, exactly as an Unknown ArchMatch does.
enum class Scope {
  Unknown,
  InScope,
  OutOfScope,
};

inline Scope scopeOf(int devMajor, int devMinor) {
  if (devMajor < 0 || devMinor < 0) return Scope::Unknown;
  if (devMajor == 12) return Scope::InScope;  // consumer/workstation Blackwell
  if (devMajor == 10) return Scope::InScope;  // datacenter Blackwell
  return Scope::OutOfScope;
}

// ── the KERNEL gate: does this image's kernel implement this part's MMA path? ─
//
// The question between the other two, and the one issue #10 is about. A part can
// be squarely in scope for NVFP4 block-scaled GEMM and still be unservable by
// this image, because "NVFP4 block-scaled GEMM" is not one instruction path:
//
//   SM12x  cutlass::arch::Sm120 block-scaled tensor op — what fp4_smoke.cu
//          compiles, and the only path in this image.
//   SM10x  tcgen05.mma with TMEM accumulation — a different CUTLASS kernel with
//          different tile and cluster shapes and a different mainloop schedule.
//          NOT a gencode of the SM12x kernel: building this source with
//          -arch=sm_100a would compile an Sm120 kernel and emit it for a part
//          that cannot run it.
//
// So this is deliberately NOT derived from BURNIN_CUDA_ARCH. The gencode says
// which cubins are in the binary; this says which kernel was written. They fail
// differently and the remedies are different — a gencode mismatch is "rebuild
// with CUDA_ARCH=…", a kernel-family mismatch is "that runner does not exist
// yet", and telling an operator to rebuild for an arch whose kernel this source
// does not contain would send them round a loop that cannot terminate.
enum class KernelCoverage {
  Unknown,
  Covers,
  WrongFamily,
};

// kKernelFamilyMajor is the compute-capability major whose block-scaled MMA path
// fp4_smoke.cu implements. It is a fact about the SOURCE, so it is a constant
// here rather than anything derived from a build flag: an image built with a
// different -arch still contains this kernel and no other.
constexpr int kKernelFamilyMajor = 12;

inline KernelCoverage kernelCovers(int devMajor, int devMinor) {
  if (devMajor < 0 || devMinor < 0) return KernelCoverage::Unknown;
  return devMajor == kKernelFamilyMajor ? KernelCoverage::Covers : KernelCoverage::WrongFamily;
}

// The operator-facing text for an in-scope part this image's kernel does not
// implement, and the exit code — kExitError, because the hardware is UNJUDGED.
//
// It must not suggest a CUDA_ARCH, which is the whole reason it is separate from
// describeArchMismatch: no gencode of this source runs on this part.
inline int describeWrongKernelFamily(char *buf, std::size_t n, int devMajor, int devMinor) {
  std::snprintf(buf, n,
                "this part reports compute capability %d.%d, which DOES do NVFP4 block-scaled "
                "GEMM — but by the tcgen05.mma path, and this image implements only the SM%dx "
                "path. Nothing was launched and the hardware is UNJUDGED: the test applies to "
                "this part and was not run, which is NOT the same as it not applying. There is "
                "no CUDA_ARCH that fixes this, because the kernel itself is different; an SM10x "
                "runner image is tracked as issue #10",
                devMajor, devMinor, kKernelFamilyMajor);
  return kExitError;
}

// The operator-facing text for an out-of-scope part, and the exit code to leave
// with — kExitSkip, which is the whole reason this decision is separated out.
//
// It names the capability it found, because "requires compute capability
// 12.0/12.1" on its own leaves an operator to go and discover what the node
// actually has, and a Skip that nobody can attribute is a Skip nobody trusts.
inline int describeOutOfScope(char *buf, std::size_t n, int devMajor, int devMinor) {
  std::snprintf(buf, n,
                "NVFP4 block-scaled GEMM requires a Blackwell part (compute capability 10.x or "
                "12.x), and this part reports %d.%d — the test does not apply to this hardware and "
                "the part is NOT failed; run a kind whose runner covers this architecture",
                devMajor, devMinor);
  return kExitSkip;
}

// ── the IMAGE gate ──────────────────────────────────────────────────────────

// Whether the built arch covers the part in front of us.
//
// Unknown is a first-class answer and not a failure: a hand-built binary
// carries BURNIN_CUDA_ARCH="unknown", and a target spelled in a form this does
// not recognise is something we have not established. A runner may only declare
// what it positively established, so an Unknown leaves the existing
// runtime-error path to speak and never manufactures a mismatch.
enum class ArchMatch {
  Unknown,
  Covers,
  Mismatch,
};

// Splits an nvcc `sm_` target into its compute capability and its suffix.
//
// The LAST digit is the minor version and the digits before it are the major
// one, which is nvcc's convention throughout: sm_90 is 9.0, sm_100 is 10.0,
// sm_121 is 12.1. Returns false — meaning "not something we recognise" — for
// everything else, including the "unknown" fallback, a `compute_` target, and a
// suffix other than the two that exist.
inline bool parseSmTarget(const char *arch, int *major, int *minor, char *suffix) {
  if (arch == nullptr) return false;
  const char kPrefix[] = "sm_";
  for (std::size_t i = 0; i < sizeof(kPrefix) - 1; ++i) {
    if (arch[i] != kPrefix[i]) return false;
  }

  const char *p = arch + (sizeof(kPrefix) - 1);
  int digits = 0;
  long value = 0;
  while (p[digits] >= '0' && p[digits] <= '9') {
    value = value * 10 + (p[digits] - '0');
    ++digits;
    if (digits > 4) return false;  // not a capability; refuse rather than wrap
  }
  // Two digits is the floor: a capability needs a major and a minor.
  if (digits < 2) return false;

  char suf = '\0';
  if (p[digits] != '\0') {
    suf = p[digits];
    // "a" is architecture-specific, "f" is family-specific. Anything else is a
    // spelling we have not been taught, so we do not claim to understand it.
    if ((suf != 'a' && suf != 'f') || p[digits + 1] != '\0') return false;
  }

  *major = static_cast<int>(value / 10);
  *minor = static_cast<int>(value % 10);
  *suffix = suf;
  return true;
}

// The question in this header's title, answered conservatively: Mismatch only
// where it is positively established.
inline ArchMatch archMatch(const char *builtArch, int devMajor, int devMinor) {
  if (devMajor < 0 || devMinor < 0) return ArchMatch::Unknown;

  int archMajor = 0, archMinor = 0;
  char suffix = '\0';
  if (!parseSmTarget(builtArch, &archMajor, &archMinor, &suffix)) return ArchMatch::Unknown;

  switch (suffix) {
    case 'a':
      // Architecture-specific: a cubin for exactly this capability, with no PTX
      // fallback. That is the deliberate choice for this runner — a pass has to
      // prove the real instruction path ran rather than an emulated one — and it
      // is also precisely what makes the mismatch invisible to the loader on a
      // binary-compatible neighbour, which is the bug this file exists to catch.
      return (devMajor == archMajor && devMinor == archMinor) ? ArchMatch::Covers
                                                              : ArchMatch::Mismatch;
    case 'f':
      // Family-specific: the capability it names and later minors of the same
      // major family. sm_120f covers CC 12.0 and 12.1.
      return (devMajor == archMajor && devMinor >= archMinor) ? ArchMatch::Covers
                                                              : ArchMatch::Mismatch;
    default:
      // Plain sm_XY. `-arch=sm_XY` alone makes nvcc embed the compute_XY PTX
      // alongside the cubin, so the driver can JIT forward onto a newer
      // capability — but never backwards onto an older one.
      return (devMajor > archMajor || (devMajor == archMajor && devMinor >= archMinor))
                 ? ArchMatch::Covers
                 : ArchMatch::Mismatch;
  }
}

// The CUDA failures fp4_smoke.cu tells apart. The caller maps cudaError_t onto
// this, which is what keeps this header CUDA-free and therefore testable.
enum class CudaFailure {
  // No error was latched where one was expected.
  NoErrorLatched,
  // cudaErrorNoKernelImageForDevice / cudaErrorInvalidDeviceFunction: the
  // loader itself refused the cubin.
  NoKernelImage,
  // cudaErrorAssert: a device-side assert fired inside the kernel. STICKY — it
  // poisons the context, so every subsequent CUDA call in this process fails
  // too and nothing more can be measured.
  DeviceAssert,
  // Everything else.
  Other,
};

namespace detail {

// "compute capability 12.1" -> "sm_121a", the arch-specific target for the part
// we are actually standing on. Deliberately the "a" form and not "f": every
// capability has an "a" target, whereas the family targets exist only for
// certain bases, so a generated "f" name could name a target nvcc rejects.
inline void archTargetFor(char *out, std::size_t n, int major, int minor) {
  std::snprintf(out, n, "sm_%d%d%s", major, minor, "a");
}

}  // namespace detail

// The pre-launch form of the finding: no CUDA call has failed, because the
// mismatch was established before the device was asked to do anything at all.
inline int describeArchMismatch(char *buf, std::size_t n, const char *builtArch, int devMajor,
                                int devMinor) {
  char want[16];
  detail::archTargetFor(want, sizeof(want), devMajor, devMinor);
  // The family-target hint is only true of the SM12x pair, where sm_120f really
  // does cover both members. Offered on any other part it is advice that cannot
  // work, and this function is reached with the caller having already
  // established that a rebuild IS the remedy — so the one thing it must not do
  // is name the wrong one. In practice fp4_smoke.cu's kernel gate settles CC
  // 10.x before this is reached; the condition is here so that stays true if the
  // gates are ever reordered.
  const char *familyHint =
      (devMajor == 12) ? ", or sm_120f for one binary covering CC 12.0 and 12.1" : "";
  std::snprintf(buf, n,
                "this image carries no cubin for the part it landed on: it was built for %s, "
                "and this part reports compute capability %d.%d. Nothing was launched, so the "
                "hardware is UNJUDGED; rebuild or pin an image whose arch covers this part "
                "(CUDA_ARCH=%s%s) and re-run",
                builtArch, devMajor, devMinor, want, familyHint);
  return kExitError;
}

// Composes the operator-facing text for a CUDA failure, and returns the exit
// code to leave with — always kExitError, for every input.
//
// errString is cudaGetErrorString(err), passed in rather than looked up here.
inline int describeCudaFailure(char *buf, std::size_t n, const char *where, CudaFailure kind,
                               const char *errString, const char *builtArch, int devMajor,
                               int devMinor) {
  const char *err = (errString != nullptr && errString[0] != '\0') ? errString : "unknown CUDA error";

  // Handled before the arch comparison because there is no runtime error here
  // to explain — and because fp4_smoke.cu's pre-launch gate means an
  // established mismatch never reaches this function in the first place.
  if (kind == CudaFailure::NoErrorLatched) {
    std::snprintf(buf, n, "%s (no CUDA error was latched)", where);
    return kExitError;
  }

  // Checked ahead of every remaining kind, not just the two the loader raises,
  // because the runtime error is not diagnostic on its own: the case that
  // motivated this file surfaces as a device-side assert, and a mismatch could
  // equally surface as an illegal address or a launch failure. Where we can
  // name the real cause we say it, whatever the runtime called it.
  if (archMatch(builtArch, devMajor, devMinor) == ArchMatch::Mismatch) {
    char want[16];
    detail::archTargetFor(want, sizeof(want), devMajor, devMinor);
    std::snprintf(buf, n,
                  "%s: %s — this image carries no cubin for the part it landed on (built for "
                  "%s, and this part reports compute capability %d.%d). The hardware is "
                  "UNJUDGED; rebuild or pin an image whose arch covers this part "
                  "(CUDA_ARCH=%s) and re-run",
                  where, err, builtArch, devMajor, devMinor, want);
    return kExitError;
  }

  if (kind == CudaFailure::NoKernelImage) {
    // The loader refused the cubin, but we could not independently establish
    // the mismatch — a hand-built binary whose BURNIN_CUDA_ARCH is "unknown",
    // or a device whose properties we never read. Report what the runtime said
    // and name the arch, without asserting a capability we did not compare.
    std::snprintf(buf, n,
                  "%s: %s — this image carries no cubin for the part it landed on (built for "
                  "%s). The hardware is UNJUDGED; pin an image built for this compute "
                  "capability and re-run",
                  where, err, builtArch);
    return kExitError;
  }

  if (kind == CudaFailure::DeviceAssert) {
    // Reached when the arch does cover the part, or could not be compared — so
    // this is NOT the wrong-image case and must not be reported as one. Say
    // what is certain: the context is dead, and the wall of text on stderr is
    // the kernel's own assert detail rather than a diagnosis.
    std::snprintf(buf, n,
                  "%s: %s — a device-side assert fired inside the kernel and poisoned the CUDA "
                  "context, so nothing further can be measured in this process. The hardware is "
                  "UNJUDGED. This image was built for %s; stderr carries the kernel's assert "
                  "text, which is template detail and not the diagnosis",
                  where, err, builtArch);
    return kExitError;
  }

  std::snprintf(buf, n, "%s: %s", where, err);
  return kExitError;
}

}  // namespace burnin

#endif  // BURNIN_ARCH_MATCH_H
