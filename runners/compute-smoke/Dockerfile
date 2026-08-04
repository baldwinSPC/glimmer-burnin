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
