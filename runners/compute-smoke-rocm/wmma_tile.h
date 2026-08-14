// wmma_tile.h — the 16x16x16 WMMA fragment layout for RDNA3/3.5 (gfx11), and
// the host-side conversions that feed it.
//
// SPDX-License-Identifier: Apache-2.0
// Copyright the Glimmer authors.
//
// WHY THIS FILE EXISTS INSTEAD OF rocWMMA
// ---------------------------------------
// rocWMMA is AMD's portable matrix-fragment API and was this runner's first
// implementation. It does not support gfx1151. ROCm 6.4.4's
// rocwmma/internal/config.hpp ends its architecture dispatch with
//
//     static_assert(0, "Unsupported architecture");
//
// and Strix Halo (gfx1151, RDNA3.5) reaches it: the library's allow-list covers
// gfx908/90a/94x and gfx1100/1101/1102, and the RDNA3.5 APUs are absent. The
// PART is not the problem — gfx1151 implements the same V_WMMA_* instructions
// as gfx1100, and the compiler exposes them as builtins for the whole gfx11
// family. Only the library's list omits it.
//
// So this runner uses the builtins directly. That is a real cost and it is
// worth stating plainly: rocWMMA would have covered CDNA's MFMA from the same
// source, and these builtins cover gfx11 only. The alternative was a runner
// that cannot measure the hardware it was written for.
//
// This header is deliberately free of HIP so the LAYOUT ARITHMETIC — which is
// the part that is easy to get silently wrong — is unit-tested on any machine
// (see wmma_tile_test.cc). The builtin calls themselves live in the .cc.
//
// THE WAVE32 LAYOUT
// -----------------
// One wavefront of 32 lanes computes one 16x16 output tile from a 16x16x16
// multiply. Each lane holds 16 A elements, 16 B elements and 8 D elements.
//
//   lane l in 0..31,  lIdx = l % 16
//   A fragment: a[lIdx * 16 + e]      for e in 0..15   (A row-major MxK)
//   B fragment: b[lIdx * 16 + e]      for e in 0..15   (B col-major, stored NxK)
//   D:  row = e * 2 + (l / 16),  col = lIdx,  for e in 0..7
//
// Lanes 0-15 and 16-31 load the SAME fragments (lIdx repeats); that duplication
// is what the wave32 form of the instruction expects, not an error. The two
// halves differ only on the store, where l / 16 selects the even or odd output
// row — which is why a layout bug here shows up as every other row being wrong
// rather than as garbage, and why it is worth testing rather than eyeballing.
//
// The product is D = A * Bᵀ in this storage convention, which is exactly what
// gfx_gate.h's referenceGemm computes.

#pragma once

#include <cstddef>
#include <cstdint>

// BURNIN_HD marks the functions below as callable from BOTH host and device.
//
// They are called from inside a __global__ kernel, which HIP refuses for a
// plain host function — but this header must ALSO compile under an ordinary C++
// compiler with no HIP anywhere, because that is what lets wmma_tile_test.cc
// check the layout without a GPU. The macro is what keeps both true: it expands
// to the HIP annotations only when a device compiler is what is reading the
// file, and to nothing at all for the unit test.
#if defined(__HIPCC__) || defined(__CUDACC__)
#define BURNIN_HD __host__ __device__
#else
#define BURNIN_HD
#endif

namespace burnin {

constexpr int kWave32 = 32;
constexpr int kFragElems = 16;  // A and B elements per lane
constexpr int kAccElems = 8;    // D elements per lane

// aIndex returns the flat index into A that lane `lane` loads for element `e`.
BURNIN_HD inline int aIndex(int lane, int e) { return (lane % 16) * 16 + e; }

// bIndex returns the flat index into B (stored column-major, NxK) that lane
// `lane` loads for element `e`.
BURNIN_HD inline int bIndex(int lane, int e) { return (lane % 16) * 16 + e; }

// dRow and dCol map an accumulator element back to its place in the output
// tile. Both halves of the wave write the same column and differ by row parity.
BURNIN_HD inline int dRow(int lane, int e) { return e * 2 + (lane / 16); }
BURNIN_HD inline int dCol(int lane) { return lane % 16; }

// dIndex is the flat row-major index into D.
BURNIN_HD inline int dIndex(int lane, int e) { return dRow(lane, e) * 16 + dCol(lane); }

// bf16Bits truncates a float to bfloat16's bit pattern.
//
// Truncation rather than round-to-nearest is EXACT for this runner's inputs and
// only for them: every value exactInputValue produces is a multiple of 0.25
// under magnitude 8, so its significand fits in bf16's 8 bits and the discarded
// low half is zero. That is a property of the test's inputs, not a general
// bf16 conversion — do not lift this into anything that converts arbitrary
// data.
BURNIN_HD inline std::uint16_t bf16Bits(float f) {
	std::uint32_t u = 0;
	// memcpy rather than a union or a pointer cast: the others are undefined
	// behaviour that happens to work until a compiler decides otherwise.
	__builtin_memcpy(&u, &f, sizeof(u));
	return static_cast<std::uint16_t>(u >> 16);
}

// bf16ToFloat reverses bf16Bits, for the host-side check that the conversion
// really is lossless on these inputs.
BURNIN_HD inline float bf16ToFloat(std::uint16_t bits) {
	const std::uint32_t u = static_cast<std::uint32_t>(bits) << 16;
	float f = 0;
	__builtin_memcpy(&f, &u, sizeof(f));
	return f;
}

}  // namespace burnin
