# Runner image for the "compute-smoke" TestKind on AMD accelerators — selected
# per node via spec.runner.imagesByVendor ({vendor: amd}), never as a kind of
# its own.  It runs one 16x16x16 matrix-core GEMM per precision (bf16, fp16)
# against a host reference, in milliseconds, and proves ONE thing: the real
# matrix-core instruction path executed on this part and produced the right
# answer.
#
# PLATFORM: linux/amd64 ONLY, deliberately — same reasoning as clockprobe-rocm.
# AMD publishes the ROCm toolchain and its container images for amd64 alone,
# and every AMD accelerator host is amd64.  ci.yml excludes the arm64 matrix
# leg; publish-runner.yml's two-platform assertion must grow per-runner
# platform awareness before this is ever published (issue #319).
#
# Licensing — the final image ships its accelerator runtime, which the NVIDIA
# runners do not: the ROCm/HIP userspace is MIT, so carrying it is compatible
# with this repository's permissive-only rule.
#   - our source        Apache-2.0   (compute_smoke_rocm.cc, gfx_gate.h)
#   - rocWMMA           MIT          (build-time header library; no runtime dep)
#   - shipped runtime   MIT          (ROCm HIP runtime and its ROCm deps)
#
# Build from AMD's devel image rather than assembling ROCm from apt.  That is
# the convention every NVIDIA runner here follows with nvcr.io/nvidia/cuda:*-devel,
# and clockprobe-rocm's first two CI builds are the record of what the other
# way costs: Ubuntu's universe hipcc shadowed AMD's, and the hand-picked
# package set could not link device code.  See that runner's README.
ARG ROCM_IMAGE=rocm/dev-ubuntu-24.04:6.4.4-complete
FROM ${ROCM_IMAGE} AS build

# The offload targets this image carries device code for.
#
# There is NO JIT FALLBACK in HIP, so this list is exactly the set of parts this
# image can measure — a target absent from it has no code here at all, which the
# runner reports as an Error (hardware unjudged), never a Skip and never a Fail.
#
#   gfx1151  Strix Halo / Ryzen AI Max — WMMA (the hardware this exists for)
#   gfx1100  RDNA3 dGPU (7900 class)   — WMMA, the nearest common desktop part
#   gfx942   Instinct MI300X/MI325X    — MFMA, so one image serves the DC tier
#
# rocWMMA emits WMMA on RDNA3+ and MFMA on CDNA from the same fragment source,
# which is what lets both families share this runner; gfx_gate.h's kernelCovers
# reports a part from any OTHER family as an Error rather than pretending.
ARG GPU_TARGETS="gfx1151 gfx1100 gfx942"

# rocWMMA is in the -complete image, but assert it rather than assume it: a
# missing header would otherwise surface as a wall of template errors in the
# compile step below, and the fix (install rocwmma-dev) is not obvious from
# that output.
RUN test -f /opt/rocm/include/rocwmma/rocwmma.hpp || { \
      echo "ERROR: rocWMMA headers are absent from this base image." >&2; \
      echo "Install rocwmma-dev, or use a -complete ROCm devel tag." >&2; \
      exit 1; \
    }

COPY compute_smoke_rocm.cc gfx_gate.h /src/

# BURNIN_BUILT_TARGETS bakes the offload list into the binary so it can answer
# "what was I built for" at runtime.  That is not a convenience: archMatch
# compares the part in front of it against this string, and a binary that had
# to be TOLD its own targets could be told wrong.
RUN mkdir -p /out && cd /src && /opt/rocm/bin/hipcc \
      -std=c++17 -O3 \
      $(for t in ${GPU_TARGETS}; do printf -- '--offload-arch=%s ' "$t"; done) \
      -DBURNIN_BUILT_TARGETS="\"${GPU_TARGETS}\"" \
      -o /out/compute-smoke-rocm compute_smoke_rocm.cc

RUN ldd /out/compute-smoke-rocm | tee /out/ldd.txt \
    && grep -q libamdhip64 /out/ldd.txt \
    && ! grep -q 'not found' /out/ldd.txt

# ── final: the binary plus the MIT HIP runtime, nothing else ─────────────────
FROM ubuntu:24.04

ARG ROCM_VERSION=6.4.4
RUN apt-get update && apt-get install -y --no-install-recommends \
      ca-certificates wget gnupg \
  && wget -qO- https://repo.radeon.com/rocm/rocm.gpg.key \
      | gpg --dearmor > /usr/share/keyrings/rocm.gpg \
  && echo "deb [arch=amd64 signed-by=/usr/share/keyrings/rocm.gpg] https://repo.radeon.com/rocm/apt/${ROCM_VERSION} noble main" \
      > /etc/apt/sources.list.d/rocm.list \
  # The origin pin, for the reason clockprobe-rocm's first CI build recorded:
  # Ubuntu ships its own builds of ROCm package names, and the runtime the
  # binary loads must be the one it was compiled against.
  && printf 'Package: *\nPin: origin repo.radeon.com\nPin-Priority: 1000\n' \
      > /etc/apt/preferences.d/rocm \
  && apt-get update && apt-get install -y --no-install-recommends \
      rocm-hip-runtime \
  && apt-get purge -y wget gnupg && apt-get autoremove -y \
  && rm -rf /var/lib/apt/lists/*

ENV LD_LIBRARY_PATH=/opt/rocm/lib

COPY --from=build /out/compute-smoke-rocm /usr/local/bin/compute-smoke-rocm

# The runtime stage is assembled separately from the build stage, so assert the
# binary it will execute resolves every library HERE.  Written as `if grep;
# then fail; fi` rather than `grep && { …; exit 1; } || true`, where the
# `|| true` would catch the exit and pass the step with the fault intact.
RUN if ldd /usr/local/bin/compute-smoke-rocm | grep -q 'not found'; then \
      echo "ERROR: unresolved libraries in the runtime stage:" >&2; \
      ldd /usr/local/bin/compute-smoke-rocm >&2; \
      exit 1; \
    fi

USER 65532:65532
ENTRYPOINT ["/usr/local/bin/compute-smoke-rocm"]
