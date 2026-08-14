// gfx_gate.h — the three arch questions compute-smoke-rocm answers before it
// launches, plus the result check it applies after.
//
// SPDX-License-Identifier: Apache-2.0
// Copyright the Glimmer authors.
//
// Host-only, and deliberately free of HIP and of rocWMMA. These are the
// decisions an operator's verdict hangs off, so they are the part that most
// needs a test, and a header that includes no HIP can be compiled and exercised
// by a plain C++ compiler with no GPU anywhere in sight (see gfx_gate_test.cc,
// which runners/cxxtests_test.go compiles and runs under `make test`).
//
// It is the AMD counterpart of compute-smoke/arch_match.h and keeps that file's
// central rule: THREE QUESTIONS, NEVER COLLAPSED. What changes is only how each
// is answered, because AMD names its targets rather than numbering them and has
// two unrelated matrix-instruction families rather than one.
//
//   scopeOf(gfx)          "can this part do matrix-core GEMM at all?" — a
//                         property of the HARDWARE. RDNA1/2 (gfx10xx) and
//                         pre-CDNA GCN have no matrix cores at all.
//                         OutOfScope -> Skip, exit 2.
//
//   kernelCovers(gfx)     "does this image's KERNEL implement the matrix path
//                         this part uses?" — a property of the SOURCE. AMD has
//                         TWO families: WMMA on RDNA3+ (gfx11xx/gfx12xx) and
//                         MFMA on CDNA (gfx908 and later gfx9). rocWMMA emits
//                         whichever the target uses, so coverage follows the
//                         offload list rather than the source — but a part from
//                         a family this build did not target is still a source
//                         question, not a hardware one.
//                         WrongFamily -> Error, exit 3.
//
//   archMatch(built, gfx) "does this image carry device code for this part?" —
//                         a property of the OFFLOAD TARGETS the image was built
//                         with. This gate is HARDER on AMD than on NVIDIA:
//                         there is no PTX-equivalent JIT fallback, so a target
//                         absent from the list has NO code in the image at all.
//                         Mismatch -> Error, exit 3.
//
// Collapsing any two misreports one of them, exactly as arch_match.h records: a
// part this image simply was not built for must never report Skip, because Skip
// reads as "acceptance does not apply to this node" and would certify a whole
// fleet nobody measured.

#pragma once

#include <cctype>
#include <cstddef>
#include <cstdio>
#include <cstring>
#include <string>

namespace burnin {

// Exit codes. Identical to compute-smoke's, because the operator's reading of
// them is identical: 1 is a HARDWARE VERDICT and nothing else may return it.
// A missing kernel, an unusable device, a runtime failure — all of those are
// 3 (Error, unjudged, retryable), never 1.
constexpr int kExitPass = 0;
constexpr int kExitFail = 1;
constexpr int kExitSkip = 2;
constexpr int kExitError = 3;

// GfxTarget is a parsed AMD compute target.
//
// AMD's target names are POSITIONAL, following LLVM's convention: the last
// character is the STEP (a hex digit, so 'a' is 10), the one before it is the
// MINOR, and everything before that is the MAJOR.
//
//   gfx1151 -> major 11, minor 5, step 1
//   gfx1030 -> major 10, minor 3, step 0
//   gfx942  -> major  9, minor 4, step 2
//   gfx908  -> major  9, minor 0, step 8
//   gfx90a  -> major  9, minor 0, step 10
//
// Reading the name as a plain integer instead is a trap this file fell into
// and its tests caught: gfx90a and gfx908 are ADJACENT steps of the same part
// family, but as integers they are 90 and 908, so any ordering between them is
// nonsense — and the ordering is exactly what decides whether an MI200 is
// reported as matrix-capable hardware or as out of scope.
struct GfxTarget {
	int major = 0;
	int minor = 0;
	int step = 0;
	bool valid = false;

	// version is the three fields as one comparable number, so "at least
	// gfx908" is an ordering rather than a special case per target.
	int version() const { return major * 256 + minor * 16 + step; }

	bool sameAs(const GfxTarget &o) const {
		return valid && o.valid && major == o.major && minor == o.minor && step == o.step;
	}
};

// hexValue reads one step/minor character: '0'-'9' then 'a'-'f'.
inline int hexValue(char c, bool *ok) {
	*ok = true;
	if (c >= '0' && c <= '9') return c - '0';
	if (c >= 'a' && c <= 'f') return 10 + (c - 'a');
	*ok = false;
	return 0;
}

// parseGfx reads a target string as HIP reports it in
// hipDeviceProp_t::gcnArchName, which carries feature flags the target name
// itself does not own: "gfx1151:sramecc-:xnack-". Everything from the first
// colon is dropped — those flags describe the RUNTIME configuration of the
// part, and treating them as part of the identity would fail every comparison
// against a plain "gfx1151" built into the image.
inline GfxTarget parseGfx(const char *s) {
	GfxTarget t;
	if (s == nullptr) return t;

	// Skip leading whitespace so a value read from a file or an env var with a
	// stray space is not silently unparseable.
	while (*s != '\0' && std::isspace(static_cast<unsigned char>(*s))) s++;
	if (std::strncmp(s, "gfx", 3) != 0) return t;
	s += 3;

	// The identity run: hex digits, ending at whitespace, a feature-flag colon,
	// or end of string.
	const char *start = s;
	while (std::isdigit(static_cast<unsigned char>(*s)) ||
	       (*s >= 'a' && *s <= 'f')) {
		s++;
	}
	const std::size_t len = static_cast<std::size_t>(s - start);

	while (*s != '\0' && std::isspace(static_cast<unsigned char>(*s))) s++;
	// Anything left must be a feature-flag suffix or nothing at all. A trailing
	// token that is neither means this is not a target string we understand,
	// and guessing at it would compare a fabricated identity.
	if (*s != '\0' && *s != ':') return t;

	// Three characters is the shortest real target (gfx908): major, minor,
	// step. Anything shorter cannot be positioned.
	if (len < 3) return t;

	bool ok = false;
	const int step = hexValue(start[len - 1], &ok);
	if (!ok) return t;
	const int minor = hexValue(start[len - 2], &ok);
	if (!ok) return t;

	int major = 0;
	for (std::size_t i = 0; i + 2 < len; i++) {
		if (!std::isdigit(static_cast<unsigned char>(start[i]))) return t;
		major = major * 10 + (start[i] - '0');
	}

	t.major = major;
	t.minor = minor;
	t.step = step;
	t.valid = true;
	return t;
}

// family returns the architecture generation: 9 for CDNA/GCN (gfx9xx), 10 for
// RDNA1/2, 11 for RDNA3/3.5, 12 for RDNA4.
//
// The generation is the parsed major; see GfxTarget for why the name has to be
// read positionally to get it.
inline int family(const GfxTarget &t) { return t.valid ? t.major : 0; }

// ─────────────────────────── question 1: scope ───────────────────────────────

enum class Scope {
	InScope,     // the part has matrix cores
	OutOfScope,  // it genuinely does not, and acceptance does not apply
	Unknown,     // the target string did not parse; NEVER treated as either
};

// scopeOf answers whether the PART can do matrix-core GEMM at all.
//
//   gfx11xx / gfx12xx   RDNA3, RDNA3.5, RDNA4 — WMMA. In scope.
//   gfx9xx >= 908       CDNA and later — MFMA. In scope.
//   gfx9xx <  908       Vega/GCN — no matrix cores. Out of scope.
//   gfx10xx             RDNA1/2 — no matrix cores. Out of scope.
//
// An unparseable target is Unknown and must never be read as either answer: a
// part we cannot identify has not been shown to be out of scope, and reporting
// Skip for it would certify unmeasured hardware.
inline Scope scopeOf(const GfxTarget &t) {
	if (!t.valid) return Scope::Unknown;
	switch (family(t)) {
	case 11:
	case 12:
		return Scope::InScope;
	case 9: {
		// gfx908 (CDNA1) is where MFMA arrives; gfx900/gfx906 are Vega and have
		// no matrix cores. Compared as an ORDERED version rather than as an
		// integer, so gfx90a (MI200, step 10) sorts ABOVE gfx908 — which is the
		// whole reason GfxTarget is positional.
		const GfxTarget mfmaFloor = {9, 0, 8, true};
		return t.version() >= mfmaFloor.version() ? Scope::InScope : Scope::OutOfScope;
	}
	case 10:
		return Scope::OutOfScope;
	default:
		return Scope::Unknown;
	}
}

// ────────────────────── question 2: kernel coverage ──────────────────────────

enum class KernelCoverage {
	Covered,      // this source implements the matrix path this part uses
	WrongFamily,  // the part uses a matrix family this build did not target
	Unknown,
};

// kernelCovers answers whether THIS IMAGE'S SOURCE implements the matrix path
// the part uses.
//
// rocWMMA compiles one source down to WMMA on RDNA3+ and MFMA on CDNA, so the
// source itself is family-agnostic — but only the families named in the build's
// offload list actually get code generated. builtFamilies is that list reduced
// to its generations, so this stays a question about the SOURCE THAT WAS BUILT
// rather than about the image's exact target set, which archMatch owns.
inline KernelCoverage kernelCovers(const GfxTarget &t, const int *builtFamilies,
                                   std::size_t nBuiltFamilies) {
	if (!t.valid) return KernelCoverage::Unknown;
	const int f = family(t);
	for (std::size_t i = 0; i < nBuiltFamilies; i++) {
		if (builtFamilies[i] == f) return KernelCoverage::Covered;
	}
	return KernelCoverage::WrongFamily;
}

// ─────────────────────── question 3: image coverage ──────────────────────────

enum class ArchMatch {
	Match,     // the image carries device code for this exact target
	Mismatch,  // it does not, and HIP has no JIT to make up the difference
	Unknown,
};

// archMatch answers whether the image carries device code for this part.
//
// builtTargets is the space-separated offload list baked in at build time.
// Comparison is on the PARSED target, so "gfx1151" in the image matches
// "gfx1151:sramecc-:xnack-" from the driver — the feature flags are runtime
// configuration and not identity.
//
// There is no near-miss rule here and there must not be one. CUDA's minor-
// version compatibility lets an sm_90 cubin serve CC 9.x; AMD has no such
// guarantee between targets, and gfx1151 code does not run on gfx1100.
inline ArchMatch archMatch(const char *builtTargets, const GfxTarget &dev) {
	if (!dev.valid) return ArchMatch::Unknown;
	if (builtTargets == nullptr || *builtTargets == '\0') return ArchMatch::Unknown;

	const char *p = builtTargets;
	while (*p != '\0') {
		while (*p == ' ' || *p == ',' || *p == ';') p++;
		if (*p == '\0') break;
		const char *start = p;
		while (*p != '\0' && *p != ' ' && *p != ',' && *p != ';') p++;
		const std::string token(start, static_cast<std::size_t>(p - start));
		const GfxTarget t = parseGfx(token.c_str());
		if (t.sameAs(dev)) return ArchMatch::Match;
	}
	return ArchMatch::Mismatch;
}

// ──────────────────────────── result checking ────────────────────────────────

// GemmCheck is the verdict on one precision's output tile.
struct GemmCheck {
	double maxAbsError = 0;
	double maxAbsRef = 0;
	long nonfinite = 0;
	long unwritten = 0;  // outputs still holding the pre-launch sentinel
	bool ok = false;
};

// kUnwrittenSentinel is what the output buffer is filled with before the
// launch.
//
// It exists because zero is a plausible GEMM result and therefore useless as
// evidence that the kernel ran: a launch that silently did nothing leaves the
// buffer untouched, and against a reference that also happened to be zero the
// comparison would pass and report a matrix path that never executed. A value
// no correct result can take turns "the kernel did not write here" into
// something the check can SEE.
constexpr float kUnwrittenSentinel = -123456.0f;

// checkGemm compares a device result against a host reference.
//
// Every failure mode is counted rather than short-circuited, because the counts
// are the diagnosis: all-unwritten means the kernel never ran, all-nonfinite
// means it ran and produced garbage, and a small maxAbsError on a few elements
// means something far more interesting than either.
inline GemmCheck checkGemm(const float *got, const float *want, std::size_t n, double tolerance) {
	GemmCheck c;
	for (std::size_t i = 0; i < n; i++) {
		const double w = static_cast<double>(want[i]);
		const double absW = w < 0 ? -w : w;
		if (absW > c.maxAbsRef) c.maxAbsRef = absW;

		const float g = got[i];
		if (g == kUnwrittenSentinel) {
			c.unwritten++;
			continue;
		}
		// Neither NaN nor infinity survives a comparison usefully, so they are
		// counted rather than folded into the error.
		if (!(g == g) || g > 3.4e38f || g < -3.4e38f) {
			c.nonfinite++;
			continue;
		}
		double d = static_cast<double>(g) - w;
		if (d < 0) d = -d;
		if (d > c.maxAbsError) c.maxAbsError = d;
	}
	c.ok = c.unwritten == 0 && c.nonfinite == 0 && c.maxAbsError <= tolerance;
	return c;
}

// referenceGemm computes D = A * Bᵀ in double precision on the host.
//
// A is row-major MxK, B is column-major KxN (i.e. stored NxK), D is row-major
// MxN — the exact layout the fragments are loaded with, so the reference and
// the device agree on what was multiplied rather than on a convention that
// happens to match.
inline void referenceGemm(const float *a, const float *b, float *d, int m, int n, int k) {
	for (int i = 0; i < m; i++) {
		for (int j = 0; j < n; j++) {
			double acc = 0;
			for (int x = 0; x < k; x++) {
				acc += static_cast<double>(a[i * k + x]) * static_cast<double>(b[j * k + x]);
			}
			d[i * n + j] = static_cast<float>(acc);
		}
	}
}

// exactInputValue generates test inputs that are EXACTLY representable in both
// bf16 and fp16, so the tolerance gates the hardware rather than the format.
//
// bf16 keeps 8 significand bits and fp16 keeps 11; a multiple of 0.25 with a
// magnitude under 8 is exact in both. Deliberately not random: an acceptance
// test that cannot be reproduced from its source is one nobody can diagnose,
// and a fixed pattern that exercises both signs is worth more here than
// coverage of the input space.
inline float exactInputValue(int index) {
	static const float kValues[8] = {0.25f, -0.5f, 1.0f, -1.25f, 2.0f, -2.5f, 3.0f, -0.75f};
	return kValues[index & 7];
}

}  // namespace burnin
