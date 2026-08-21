# Runner image for the "compute-smoke" TestKind: a fast, arch-correct kernel that
# proves the accelerator's tensor cores actually compute.  This variant targets
# NVIDIA Blackwell SM120/SM121 (GB10 / DGX Spark on arm64, RTX PRO 6000 Blackwell
# and the other x86 SM120 parts on amd64) and exercises NVFP4.
#
# The image is built for linux/amd64 AND linux/arm64.  Those are HOST
# architectures and say nothing about which GPU the image can measure; see the
# CUDA_ARCH block below, which is where that decision is made.
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

# BuildKit supplies this; it must still be declared to be visible in the stage.
# The nvcc stage runs EMULATED on the platform it is not native to, which is why
# amd64 (native on GitHub's hosted runners) is the cheap half of this matrix and
# arm64 is the 14-minute QEMU half.
ARG TARGETARCH

# v4.5.0 is the floor: its changelog is the first to report block-scaled MMA
# working on Spark (SM121).  Earlier releases mis-handle this exact card.
ARG CUTLASS_REF=v4.6.1
# A tag is not an identity: it can be moved.  A published runner image is what a
# node's readiness gate executes and its tag is immutable, so a moved upstream
# tag would change what that image computes with no audit trail — and CUTLASS is
# the kernel itself here, not a peripheral dependency.  Bumping CUTLASS_REF means
# bumping CUTLASS_SHA with it; resolve the new value with
# `git ls-remote https://github.com/NVIDIA/cutlass.git 'refs/tags/<ref>' 'refs/tags/<ref>^{}'`,
# taking the PEELED `^{}` value where the tag is annotated — that is the commit
# a shallow clone leaves at HEAD, and therefore what the assertion compares
# against. CUTLASS's tags happen to be lightweight, so the unpeeled form gives
# the right answer here by luck; it is the wrong command to learn, and the
# build below diagnoses the mistake if you make it.
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
# An "a" target emits a cubin for that capability ONLY — no PTX fallback — which
# is the whole proof this runner offers: a pass means the real block-scaled MMA
# instruction path executed rather than something JIT-compiled or emulated.  The
# amd64 default is a DIFFERENT arch, not a widened one; it keeps the property.
#
# WHAT THIS RUNNER STILL CANNOT DO, and it is a GPU-arch limit, not a CPU-arch
# one: the kernel is CUTLASS `arch::Sm120` block-scaled NVFP4, so it exists for
# CC 12.x and nothing else.  Never a hardware Fail, and the two remaining parts
# are reported differently on purpose:
#
#   H100 (sm_90), A100, L40S …  exit 2, Skip.  These parts have no block-scaled
#                               FP4 at all, so acceptance genuinely does not
#                               apply to them.
#   B200/GB200 (sm_100),        exit 3, Error, UNJUDGED.  These parts DO have
#   B300/GB300 (sm_103)         NVFP4 — by the tcgen05.mma + TMEM path, which
#                               this kernel does not implement — so the test
#                               applies and was not run.  Reporting that as a
#                               Skip certified a B200 fleet as out of scope for
#                               the very capability it has; see issue #10 and
#                               the three-questions note in arch_match.h.
#
# SM10x is a SEPARATE runner, not a CUDA_ARCH of this one: --build-arg
# CUDA_ARCH=sm_100a would compile this `arch::Sm120` kernel and emit it for a
# part that cannot run it.  That is why sm_100a is NOT offered in
# publish-runner.yml's cuda_arch list, and why the runtime gate names the issue
# rather than suggesting a rebuild.  Do not "fix" any of this by adding PTX here.
#
# Set CUDA_ARCH explicitly to override BOTH platforms — e.g. sm_120f for one
# binary covering CC 12.0 + 12.1.  An explicit value is honoured as given.
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

COPY fp4_smoke.cu /src/fp4_smoke.cu
# Host-only and CUDA-free: it decides whether this part is in scope at all and
# whether CUDA_ARCH covers it, which is why both can be unit-tested with a plain
# C++ compiler and no GPU (arch_match_test.cc, run by `make test` via
# runners/cxxtests_test.go).
COPY arch_match.h /src/arch_match.h
# Shared across every runner that iterates devices (docs/dev/multi-device.md);
# duplicated because each runner is its own Docker build context and COPY
# cannot leave one. runners/devicefold_test.go refuses drift between copies.
COPY device_fold.h /src/device_fold.h

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
#
# CUDA_ARCH is resolved HERE rather than in the ARG default because the default
# depends on TARGETARCH, and an ARG default cannot branch.  An empty CUDA_ARCH
# means "this platform's default"; anything else is used verbatim.
#
# The resolved value is written to /out/cuda_arch because THREE later steps need
# it and a shell variable does not survive the end of a RUN.  It was previously
# resolved into a local, and the two guard steps below kept referring to
# ${CUDA_ARCH} — which is empty on every default build, so `nvcc -arch=` had no
# argument at all.  Neither branch caught it alone: the per-platform default and
# the #else guard were written on separate branches, each green, and the break
# appeared only once both were on main.  One resolution, one file, three readers.
RUN set -eu; \
    arch="${CUDA_ARCH:-}"; \
    if [ -z "${arch}" ]; then \
      case "${TARGETARCH}" in \
        arm64) arch=sm_121a ;; \
        amd64) arch=sm_120a ;; \
        *) echo "compute-smoke: no default gencode for TARGETARCH='${TARGETARCH}'." >&2; \
           echo "  Pass --build-arg CUDA_ARCH=sm_1XXa naming the part you are building for." >&2; \
           exit 1 ;; \
      esac; \
    fi; \
    echo "compute-smoke: TARGETARCH=${TARGETARCH} -> CUDA_ARCH=${arch}"; \
    mkdir -p /out && printf '%s' "${arch}" > /out/cuda_arch; \
    /usr/local/cuda/bin/nvcc \
      -std=c++17 -O3 -arch="${arch}" --expt-relaxed-constexpr \
      -DBURNIN_CUDA_ARCH="\"${arch}\"" \
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
RUN set -eu; \
    arch="$(cat /out/cuda_arch)"; \
    /usr/local/cuda/bin/nvcc \
      -std=c++17 -O3 -arch="${arch}" --expt-relaxed-constexpr \
      -DBURNIN_CUDA_ARCH="\"${arch}\"" -DBURNIN_FP4_ASSUME_NO_BLOCK_SCALED_MMA \
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
RUN set -eu; \
    arch="$(cat /out/cuda_arch)"; \
    if grep -aq "${GUARD_MARK}" /out/fp4_smoke; then \
      echo "refusing to build: ${arch} did not enable block-scaled MMA, so this image would report every node UNJUDGED" >&2; \
      exit 1; \
    fi; \
    echo "block-scaled MMA kernel compiled in for ${arch}: OK"

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
