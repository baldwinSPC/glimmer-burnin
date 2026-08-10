# Runner image for the "gemm-sweep" TestKind: one GEMM, one precision per
# execution, selected by BURNIN_VARIANT_PRECISION.  This variant targets NVIDIA
# Blackwell SM120/SM121 (GB10 / DGX Spark on arm64, RTX PRO 6000 Blackwell and
# the other x86 SM120 parts on amd64) and carries five cells: fp4, fp8, bf16,
# tf32, fp64.
#
# The image is built for linux/amd64 AND linux/arm64.  Those are HOST
# architectures and say nothing about which GPU the image can measure; see the
# CUDA_ARCH block below, which is where that decision is made.
#
# Licensing note — this is why the build is multi-stage.  The CUDA toolchain is
# used at BUILD time only; none of its layers are shipped.  The final image
# contains our Apache-2.0 binary on a distroless base and NO NVIDIA-licensed
# redistributable libraries.  libcuda.so.1 is injected at runtime from the host
# driver by the NVIDIA Container Toolkit, so it is never baked in.  cuBLAS would
# make every one of these kernels simpler and is ruled out for exactly this
# reason: it is an NVIDIA redistributable, and the ldd guard below would
# (correctly) refuse an image that linked it.
#   - our source        Apache-2.0
#   - NVIDIA CUTLASS    BSD-3-Clause, header-only (compiled in, not redistributed;
#                       asserted below from the LICENSE.txt of the exact commit
#                       fetched, not from this comment)

ARG CUDA_IMAGE=nvcr.io/nvidia/cuda:13.0.1-devel-ubuntu24.04
FROM ${CUDA_IMAGE} AS build

# BuildKit supplies this; it must still be declared to be visible in the stage.
# The nvcc stage runs EMULATED on the platform it is not native to, which is why
# amd64 (native on GitHub's hosted runners) is the cheap half of this matrix and
# arm64 is the slow QEMU half.
ARG TARGETARCH

# The same pin as compute-smoke, deliberately: the two runners answer related
# questions about the same silicon, and measuring them with two different
# CUTLASS versions would make a divergence between their verdicts a toolchain
# artifact rather than a finding.  v4.5.0 is the floor for SM121 block-scaled
# MMA; see compute-smoke's Dockerfile.
ARG CUTLASS_REF=v4.6.1
# A tag is not an identity: it can be moved.  A published runner image is what a
# node's readiness gate executes and its tag is immutable, so a moved upstream
# tag would change what that image computes with no audit trail.  Bumping
# CUTLASS_REF means bumping CUTLASS_SHA with it; resolve the new value with
#   git ls-remote https://github.com/NVIDIA/cutlass.git \
#       'refs/tags/<ref>' 'refs/tags/<ref>^{}'
# and take the PEELED (^{}) value if one is listed — an annotated tag has two
# SHAs, and a shallow clone leaves the COMMIT at HEAD, which is what the check
# below compares against.  Recording the unpeeled tag-object SHA fails every
# build (see issue #238).
ARG CUTLASS_SHA=e05f953a5b3d38adc240df2ff928e0421c2abba3

# ── CPU architecture and GPU architecture are ORTHOGONAL ─────────────────────
#
# TARGETARCH is the HOST the container runs on (linux/amd64 vs linux/arm64).
# CUDA_ARCH is the GPU gencode.  Conflating them produces images nobody can use:
# an amd64 image full of sm_121a device code is dead weight, because CC 12.1 is
# GB10, GB10 is a Grace part, and there is no x86 host with one.
#
# So CUDA_ARCH defaults PER TARGET PLATFORM, and both defaults are "a" targets:
#
#   arm64 → sm_121a   GB10 / DGX Spark            (CC 12.1)
#   amd64 → sm_120a   RTX PRO 6000 Blackwell,     (CC 12.0)
#                     RTX 5090 and the rest of
#                     the x86 SM120 Blackwells
#
# An "a" target emits cubins for that capability ONLY — no PTX fallback — so a
# pass proves the real instruction path executed rather than something
# JIT-compiled.  The amd64 default is a DIFFERENT arch, not a widened one; it
# keeps the property.  The runtime gate in gemm_sweep.cu (precision.h's
# archCovers) refuses a part this image was not built for BEFORE launching,
# because the loader does not refuse the sm_120a-on-CC-12.1 direction on its
# own — it loads and asserts inside the kernel.
#
# Parts outside the SM12x family are handled by the same gate as an Error
# (wrong image pinned, the part is UNJUDGED) — except where a PRECISION is
# genuinely absent from the part, which is a Skip decided per cell by
# precision.h.  Set CUDA_ARCH explicitly to override BOTH platforms — e.g.
# sm_120f for one binary covering CC 12.0 + 12.1.
ARG CUDA_ARCH=

RUN apt-get update && apt-get install -y --no-install-recommends git ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN git clone --depth 1 -b ${CUTLASS_REF} https://github.com/NVIDIA/cutlass.git /cutlass

RUN got="$(git -C /cutlass rev-parse HEAD)"; \
    if [ "${got}" != "${CUTLASS_SHA}" ]; then \
      echo "refusing to build: ${CUTLASS_REF} is ${got}, expected ${CUTLASS_SHA}" >&2; \
      if [ "$(cd /cutlass && git cat-file -t "${CUTLASS_SHA}" 2>/dev/null)" = "tag" ]; then \
        echo "  ${CUTLASS_SHA} is the ANNOTATED TAG OBJECT, not the commit it points to." >&2; \
        echo "  Use the peeled value: $(cd /cutlass && git rev-parse "${CUTLASS_SHA}^{commit}")" >&2; \
        echo "  Resolve it with: git ls-remote <repo> 'refs/tags/<ref>^{}'" >&2; \
      fi; \
      exit 1; \
    fi; \
    echo "CUTLASS ${CUTLASS_REF} = ${got}: OK"

# The licence is asserted from the exact commit fetched, not from a comment that
# cannot notice the dependency moving.  If CUTLASS ever stops being BSD-3-Clause
# this build fails rather than quietly compiling in something this project may
# not publish.  (CUTLASS's python/CuTeDSL subdirectory is under the NVIDIA EULA
# and is not touched by this build; the headers compiled in are BSD-3-Clause.)
RUN set -e; \
    test -f /cutlass/LICENSE.txt; \
    grep -q "BSD 3-Clause License" /cutlass/LICENSE.txt; \
    grep -q "Redistribution and use in source and binary forms" /cutlass/LICENSE.txt; \
    echo "CUTLASS ${CUTLASS_REF} is BSD-3-Clause: OK"

COPY gemm_sweep.cu /src/gemm_sweep.cu
# Host-only and CUDA-free: every decision that can produce a wrong verdict —
# which precision was asked for, whether this part supports it, whether this
# image covers this part — lives here so it can be unit-tested with a plain C++
# compiler and no GPU (precision_test.cc, run by `make test` via
# runners/cxxtests_test.go).
COPY precision.h /src/precision.h

# -cudart static is nvcc's default on Linux but stated explicitly so the intent
# survives a toolchain change.  -static-libstdc++/-static-libgcc are host-compiler
# flags and MUST go through -Xcompiler; passed bare, nvcc fails outright.
#
# -DBURNIN_CUDA_ARCH bakes the arch string into the binary so it can print
# built_cuda_arch= and refuse a part it was not built for BEFORE launching;
# nvcc gives the runner no other way to know (__CUDA_ARCH_LIST__ records 1200
# for both sm_120a and sm_120f, and only the "f" form also runs on 12.1).
#
# CUDA_ARCH is resolved HERE rather than in the ARG default because the default
# depends on TARGETARCH, and an ARG default cannot branch.  The resolved value
# is written to /out/cuda_arch because the guard steps below need it and a shell
# variable does not survive the end of a RUN — compute-smoke's history shows
# what referring back to the empty ${CUDA_ARCH} does instead.
RUN set -eu; \
    arch="${CUDA_ARCH:-}"; \
    if [ -z "${arch}" ]; then \
      case "${TARGETARCH}" in \
        arm64) arch=sm_121a ;; \
        amd64) arch=sm_120a ;; \
        *) echo "gemm-sweep: no default gencode for TARGETARCH='${TARGETARCH}'." >&2; \
           echo "  Pass --build-arg CUDA_ARCH=sm_1XXa naming the part you are building for." >&2; \
           exit 1 ;; \
      esac; \
    fi; \
    echo "gemm-sweep: TARGETARCH=${TARGETARCH} -> CUDA_ARCH=${arch}"; \
    mkdir -p /out && printf '%s' "${arch}" > /out/cuda_arch; \
    /usr/local/cuda/bin/nvcc \
      -std=c++17 -O3 -arch="${arch}" --expt-relaxed-constexpr \
      -DBURNIN_CUDA_ARCH="\"${arch}\"" \
      -cudart static -Xcompiler -static-libstdc++ -Xcompiler -static-libgcc \
      -I /cutlass/include -I /cutlass/tools/util/include \
      -o /out/gemm_sweep /src/gemm_sweep.cu

# Fail the build rather than ship an image that silently drags in an NVIDIA .so.
RUN ldd /out/gemm_sweep | tee /out/ldd.txt \
    && ! grep -qiE 'libcud(art|a)|libnv' /out/ldd.txt

# ── both sides of the SM120-MMA guard get compiled, every build ───────────────
#
# gemm_sweep.cu's fp4 and fp8 cells sit behind
#   #if defined(CUTLASS_ARCH_MMA_SM120_SUPPORTED) || defined(..._SM121_SUPPORTED)
# with an #else that reports the part UNJUDGED.  Every build takes the #if side
# (both macros are defined even for -arch=sm_80 on this CUTLASS), so without
# these two steps the #else would never be compiled at all and a syntax or logic
# error in it would surface only on a CUTLASS downgrade — on a fleet.  Same
# guard, same reasoning, as compute-smoke's.
#
# 1. The #else compiles, forced with BURNIN_GEMM_ASSUME_NO_SM120_MMA because the
#    arch flag cannot reach it.  Everything else is identical to the release
#    compile above, so this really is the shipped source on the shipped
#    toolchain.  The binary is a compile artefact: written to /tmp, asserted,
#    deleted, never copied into the final image.  grep -a because the marker is
#    a string literal inside an ELF.
ARG GUARD_MARK="was not compiled with SM120/SM121 tensor-core MMA support"
RUN set -eu; \
    arch="$(cat /out/cuda_arch)"; \
    /usr/local/cuda/bin/nvcc \
      -std=c++17 -O3 -arch="${arch}" --expt-relaxed-constexpr \
      -DBURNIN_CUDA_ARCH="\"${arch}\"" -DBURNIN_GEMM_ASSUME_NO_SM120_MMA \
      -cudart static -Xcompiler -static-libstdc++ -Xcompiler -static-libgcc \
      -I /cutlass/include -I /cutlass/tools/util/include \
      -o /tmp/gemm_sweep_no_mma /src/gemm_sweep.cu \
    && grep -aq "${GUARD_MARK}" /tmp/gemm_sweep_no_mma \
    && echo "#else branch compiled and linked: OK" \
    && rm -f /tmp/gemm_sweep_no_mma
#
# 2. The SHIPPED binary took the #if side.  This is the one with teeth: if a
#    CUTLASS bump ever stopped defining those macros for our real arch, the
#    build above would still succeed and we would publish an immutable tag whose
#    fp4 and fp8 cells report "not compiled with tensor-core MMA support" — an
#    Error — on every node in a fleet.  The #else message can only be in the
#    binary if the #else was compiled, so its ABSENCE is the proof the real
#    kernels are in there.
RUN set -eu; \
    arch="$(cat /out/cuda_arch)"; \
    if grep -aq "${GUARD_MARK}" /out/gemm_sweep; then \
      echo "refusing to build: ${arch} did not enable SM120/SM121 MMA, so this image would report every fp4/fp8 cell UNJUDGED" >&2; \
      exit 1; \
    fi; \
    echo "SM120/SM121 MMA kernels compiled in for ${arch}: OK"

# ── final: no NVIDIA layers ───────────────────────────────────────────────────
# cc-debian13 carries a glibc at least as new as the ubuntu24.04 build stage.
# An older base (cc-debian12, glibc 2.36) fails at runtime with GLIBC_2.38 errors.
# A scratch/musl base will not work: the binary dlopen()s the driver and needs glibc.
FROM gcr.io/distroless/cc-debian13

# The NVIDIA-provided CUDA images set these; leaving that base means we must set
# them ourselves.  Without "compute", the toolkit does not mount libcuda.so.1 and
# the binary cannot see the GPU at all.  On the GB10 fleet this is also what
# keeps the pod alive at all: CDI injection is broken there (issue #52) and
# containerd's legacy-runtime fallback injects from exactly these two variables.
ENV NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility

COPY --from=build /out/gemm_sweep /usr/local/bin/gemm_sweep
USER 65532:65532
ENTRYPOINT ["/usr/local/bin/gemm_sweep"]
