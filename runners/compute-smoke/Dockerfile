# Runner image for the "compute-smoke" TestKind: a fast, arch-correct kernel that
# proves the accelerator's tensor cores actually compute.  This variant targets
# NVIDIA Blackwell SM120/SM121 (GB10 / DGX Spark) and exercises NVFP4.
#
# Licensing note — this is why the build is multi-stage.  The CUDA toolchain is
# used at BUILD time only; none of its layers are shipped.  The final image
# contains our Apache-2.0 binary on a distroless base and NO NVIDIA-licensed
# redistributable libraries.  libcuda.so.1 is injected at runtime from the host
# driver by the NVIDIA Container Toolkit, so it is never baked in.
#   - our source        Apache-2.0
#   - NVIDIA CUTLASS    BSD-3-Clause, header-only (compiled in, not redistributed)

ARG CUDA_IMAGE=nvcr.io/nvidia/cuda:13.0.1-devel-ubuntu24.04
FROM ${CUDA_IMAGE} AS build

# v4.5.0 is the floor: its changelog is the first to report block-scaled MMA
# working on Spark (SM121).  Earlier releases mis-handle this exact card.
ARG CUTLASS_REF=v4.6.1
# A tag is not an identity: it can be moved.  A published runner image is what a
# node's readiness gate executes and its tag is immutable, so a moved upstream
# tag would change what that image computes with no audit trail — and CUTLASS is
# the kernel itself here, not a peripheral dependency.  Bumping CUTLASS_REF means
# bumping CUTLASS_SHA with it; resolve the new value with
# `git ls-remote https://github.com/NVIDIA/cutlass.git refs/tags/<ref>`.
ARG CUTLASS_SHA=e05f953a5b3d38adc240df2ff928e0421c2abba3
# sm_121a = GB10 (compute capability 12.1).  An "a" target emits a cubin for that
# capability only — no PTX fallback — which is what we want for a smoke test that
# must prove the real instruction path ran.  Use sm_120f to cover CC 12.0 + 12.1.
ARG CUDA_ARCH=sm_121a

RUN apt-get update && apt-get install -y --no-install-recommends git ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN git clone --depth 1 -b ${CUTLASS_REF} https://github.com/NVIDIA/cutlass.git /cutlass

RUN got="$(git -C /cutlass rev-parse HEAD)"; \
    if [ "${got}" != "${CUTLASS_SHA}" ]; then \
      echo "refusing to build: ${CUTLASS_REF} is ${got}, expected ${CUTLASS_SHA}" >&2; \
      exit 1; \
    fi; \
    echo "CUTLASS ${CUTLASS_REF} = ${got}: OK"

COPY fp4_smoke.cu /src/fp4_smoke.cu
# Host-only and CUDA-free: it decides whether this part is in scope at all and
# whether CUDA_ARCH covers it, which is why both can be unit-tested with a plain
# C++ compiler and no GPU (arch_match_test.cc, run by `make test` via
# runners/cxxtests_test.go).
COPY arch_match.h /src/arch_match.h

# -cudart static is nvcc's default on Linux but stated explicitly so the intent
# survives a toolchain change.  -static-libstdc++/-static-libgcc are host-compiler
# flags and MUST go through -Xcompiler; passed bare, nvcc fails outright.  Linking
# them drops the libstdc++/libgcc_s ABI dependency, leaving only glibc.
#
# -DBURNIN_CUDA_ARCH bakes the arch string into the binary so it can print
# built_cuda_arch= and name itself in a wrong-image error.  nvcc gives the runner
# no other way to report what it was compiled for: __CUDA_ARCH_LIST__ records
# 1200 for both sm_120a and sm_120f, and only the "f" form also runs on 12.1, so
# the macro cannot answer "does this image run on the part in front of me?".
RUN mkdir -p /out && /usr/local/cuda/bin/nvcc \
      -std=c++17 -O3 -arch=${CUDA_ARCH} --expt-relaxed-constexpr \
      -DBURNIN_CUDA_ARCH="\"${CUDA_ARCH}\"" \
      -cudart static -Xcompiler -static-libstdc++ -Xcompiler -static-libgcc \
      -I /cutlass/include -I /cutlass/tools/util/include \
      -o /out/fp4_smoke /src/fp4_smoke.cu

# Fail the build rather than ship an image that silently drags in an NVIDIA .so.
RUN ldd /out/fp4_smoke | tee /out/ldd.txt \
    && ! grep -qiE 'libcud(art|a)|libnv' /out/ldd.txt

# ── both sides of the block-scaled-MMA guard get compiled, every build ────────
#
# fp4_smoke.cu's kernel is wrapped in
#   #if defined(CUTLASS_ARCH_MMA_SM120_SUPPORTED) || defined(..._SM121_SUPPORTED)
# with an #else that reports the part UNJUDGED.  Every build so far took the #if
# side, so the #else had never been compiled at all and a syntax or logic error
# in it would have surfaced only on a CUTLASS or CUDA downgrade — on a fleet
# (issue #9).  These two checks are cheap and they close it from both directions.
#
# 1. The #else compiles.  It is forced with BURNIN_FP4_ASSUME_NO_BLOCK_SCALED_MMA
#    rather than by building for an older arch, because the arch flag does not
#    reach it: measured against CUTLASS v4.6.1 with CUDA 13.0.1, BOTH
#    CUTLASS_ARCH_MMA_SM120_SUPPORTED and ..._SM121_SUPPORTED are defined even
#    for `-arch=sm_80`, so a pre-Blackwell build still takes the #if side.  Only
#    a pre-v4.5.0 CUTLASS leaves them undefined, and cloning a second stale
#    upstream into every build to compile eight lines is the worse trade.
#
#    Everything else is identical to the release compile above, so this really is
#    the shipped source on the shipped toolchain.  The binary is a compile
#    artefact: written to /tmp, asserted, deleted, never copied into the final
#    image.  grep -a because the marker is a string literal inside an ELF.
ARG GUARD_MARK="was not compiled with SM120/SM121 block-scaled MMA support"
RUN /usr/local/cuda/bin/nvcc \
      -std=c++17 -O3 -arch=${CUDA_ARCH} --expt-relaxed-constexpr \
      -DBURNIN_CUDA_ARCH="\"${CUDA_ARCH}\"" -DBURNIN_FP4_ASSUME_NO_BLOCK_SCALED_MMA \
      -cudart static -Xcompiler -static-libstdc++ -Xcompiler -static-libgcc \
      -I /cutlass/include -I /cutlass/tools/util/include \
      -o /tmp/fp4_smoke_no_mma /src/fp4_smoke.cu \
    && grep -aq "${GUARD_MARK}" /tmp/fp4_smoke_no_mma \
    && echo "#else branch compiled and linked: OK" \
    && rm -f /tmp/fp4_smoke_no_mma
#
# 2. The SHIPPED binary took the #if side.  This is the one with teeth: if a
#    CUTLASS bump ever stopped defining those macros for our real arch, the
#    build above would still succeed and we would publish an immutable tag that
#    reports "not compiled with block-scaled MMA support" — an Error — on every
#    node in a fleet.  The #else message can only be in the binary if the #else
#    was compiled, so its ABSENCE is the proof the kernel is really in there.
RUN if grep -aq "${GUARD_MARK}" /out/fp4_smoke; then \
      echo "refusing to build: ${CUDA_ARCH} did not enable block-scaled MMA, so this image would report every node UNJUDGED" >&2; \
      exit 1; \
    fi; \
    echo "block-scaled MMA kernel compiled in for ${CUDA_ARCH}: OK"

# ── final: no NVIDIA layers ───────────────────────────────────────────────────
# cc-debian13 carries a glibc at least as new as the ubuntu24.04 build stage.
# An older base (cc-debian12, glibc 2.36) fails at runtime with GLIBC_2.38 errors.
# A scratch/musl base will not work: the binary dlopen()s the driver and needs glibc.
FROM gcr.io/distroless/cc-debian13

# The NVIDIA-provided CUDA images set these; leaving that base means we must set
# them ourselves.  Without "compute", the toolkit does not mount libcuda.so.1 and
# the binary cannot see the GPU at all.
ENV NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility

COPY --from=build /out/fp4_smoke /usr/local/bin/fp4_smoke
USER 65532:65532
ENTRYPOINT ["/usr/local/bin/fp4_smoke"]
