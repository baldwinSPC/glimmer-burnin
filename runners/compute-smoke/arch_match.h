// arch_match.h — does the arch this binary was BUILT for cover the part it LANDED on?
//
// SPDX-License-Identifier: Apache-2.0
// Copyright the Glimmer authors.
//
// Host-only, and deliberately free of CUDA. This is the decision an operator's
// error message hangs off, so it is the part that most needs a test, and a
// header that includes no CUDA can be compiled and exercised by a plain C++
// compiler with no GPU anywhere in sight (see arch_match_test.cc, which
// runners/compute-smoke/archmatch_test.go compiles and runs under `make test`).
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
// This answers "does this image carry a cubin for this part?", which is a
// question about the image that was pinned. It is NOT the scope gate in
// fp4_smoke.cu's main(), which asks "can this part do NVFP4 block-scaled GEMM at
// all?" — a question about the hardware, answered with Skip. Collapsing the two
// would report a whole fleet as "acceptance not applicable" when the truth is
// that an operator pinned the wrong tag.

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
  std::snprintf(buf, n,
                "this image carries no cubin for the part it landed on: it was built for %s, "
                "and this part reports compute capability %d.%d. Nothing was launched, so the "
                "hardware is UNJUDGED; rebuild or pin an image whose arch covers this part "
                "(CUDA_ARCH=%s, or sm_120f for one binary covering CC 12.0 and 12.1) and re-run",
                builtArch, devMajor, devMinor, want);
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
