// gfx_offload.h — "does this image carry device code for this part?", asked
// BEFORE the first kernel launch.
//
// SPDX-License-Identifier: Apache-2.0
// Copyright the Glimmer authors.
//
// Why before: measured on a Strix Halo (gfx1151) under ROCm 7.2.3, an image
// built without gfx1151 does not get an error back from hipLaunchKernel — the
// runtime logs "No compatible code objects found for: gfx1151" and then
// SEGFAULTS inside the launch (#564). No post-launch hipGetLastError can report
// that, so the runner exited 139 with no marker, which the operator records as
// an Error with nothing saying why. Asking the question from the offload list
// the image was built with, against the target the device reports, turns it
// into exit 3 with a sentence naming the rebuild that fixes it.
//
// The parse is compute-smoke-rocm/gfx_gate.h's (positional, feature flags
// dropped, no near-miss rule: AMD has no JIT fallback between targets). That
// file also answers two matrix-core questions these runners never ask, so this
// is the subset they need, byte-identical across clockprobe-rocm, gpu-burn-rocm,
// memory-bw-rocm and thermal-soak-rocm and held there by
// runners/sharedsource_test.go. Host-only, no HIP, tested under `make test`.

#pragma once

#include <cctype>
#include <cstddef>
#include <cstring>
#include <string>

#ifndef BURNIN_BUILT_TARGETS
#define BURNIN_BUILT_TARGETS ""
#endif

namespace gfxoffload {

// kBuiltTargets is the Dockerfile's GPU_TARGETS, baked in with the same string
// the offload flags were generated from, so the two cannot drift.
constexpr const char *kBuiltTargets = BURNIN_BUILT_TARGETS;

struct Target {
	int major = 0, minor = 0, step = 0;
	bool valid = false;
	bool sameAs(const Target &o) const {
		return valid && o.valid && major == o.major && minor == o.minor && step == o.step;
	}
};

inline int hexValue(char c, bool *ok) {
	*ok = true;
	if (c >= '0' && c <= '9') return c - '0';
	if (c >= 'a' && c <= 'f') return 10 + (c - 'a');
	*ok = false;
	return 0;
}

// parse reads "gfx1151" or HIP's "gfx1151:sramecc-:xnack-"; everything from the
// first colon is runtime configuration, not identity. The last character is the
// step and the one before it the minor, both hex (gfx90a is step 10).
inline Target parse(const char *s) {
	Target t;
	if (s == nullptr) return t;
	while (*s != '\0' && std::isspace(static_cast<unsigned char>(*s))) s++;
	if (std::strncmp(s, "gfx", 3) != 0) return t;
	s += 3;
	const char *start = s;
	while (std::isdigit(static_cast<unsigned char>(*s)) || (*s >= 'a' && *s <= 'f')) s++;
	const std::size_t len = static_cast<std::size_t>(s - start);
	while (*s != '\0' && std::isspace(static_cast<unsigned char>(*s))) s++;
	if ((*s != '\0' && *s != ':') || len < 3) return t;
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

enum class Coverage {
	Match,     // the image carries code for this exact target
	Mismatch,  // it does not: launching would crash the runtime, not return an error
	Unknown,   // the device's target, or the build's list, could not be read
};

inline Coverage covers(const char *builtTargets, const char *deviceArch) {
	const Target dev = parse(deviceArch);
	if (!dev.valid || builtTargets == nullptr || *builtTargets == '\0') return Coverage::Unknown;
	const char *p = builtTargets;
	while (*p != '\0') {
		while (*p == ' ' || *p == ',' || *p == ';') p++;
		if (*p == '\0') break;
		const char *start = p;
		while (*p != '\0' && *p != ' ' && *p != ',' && *p != ';') p++;
		if (parse(std::string(start, static_cast<std::size_t>(p - start)).c_str()).sameAs(dev))
			return Coverage::Match;
	}
	return Coverage::Mismatch;
}

// refusal is the error sentence for a non-Match, or "" for a Match. It does
// not say "hardware unjudged": each runner's error path already appends that. An Unknown
// is refused too: launching into a runtime that may segfault is not a way to
// find out, and a runner may only proceed on what it positively established.
inline std::string refusal(const char *builtTargets, const char *deviceArch) {
	switch (covers(builtTargets, deviceArch)) {
	case Coverage::Match:
		return "";
	case Coverage::Mismatch:
		return std::string("this image carries device code for '") + builtTargets +
		       "' and the part reports " + deviceArch +
		       "; HIP has no JIT fallback and the runtime crashes rather than failing the "
		       "launch, so nothing was launched — rebuild with GPU_TARGETS including it";
	case Coverage::Unknown:
		break;
	}
	return std::string("cannot tell whether this image (built for '") +
	       (builtTargets ? builtTargets : "") + "') carries device code for '" +
	       (deviceArch ? deviceArch : "") + "', so nothing was launched";
}

} // namespace gfxoffload
