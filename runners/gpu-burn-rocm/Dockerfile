# Runner image for the "gpu-burn" TestKind on AMD accelerators — selected per
# node via spec.runner.imagesByVendor ({vendor: amd}), never as a kind of its
# own.  It holds a heavy, correctness-checked SGEMM on the part for the whole
# duration budget, sampling temperature, power and clocks from sysfs, and asks:
# does it still get the RIGHT ANSWER while it is that hot?
#
# The engine lives in soak_core_rocm.h, shared BYTE-IDENTICALLY with the other
# soak runner; runners/sharedsource_test.go fails if the two copies drift.  It
# is duplicated because each runner is its own Docker build context and COPY
# cannot reach outside it.
#
# PLATFORM: linux/amd64 ONLY, deliberately — same reasoning as the other -rocm
# runners.  ci.yml excludes the arm64 matrix leg; publish-runner.yml needs
# per-runner platform awareness before this is ever published (issue #319).
#
# Licensing:
#   - our source        Apache-2.0   (gpu_burn_rocm.cc, soak_core_rocm.h)
#   - third-party       none linked beyond the HIP runtime
#   - shipped runtime   MIT          (ROCm HIP runtime and its ROCm deps)
#
# Build from AMD's devel image rather than assembling ROCm from apt — the
# convention every NVIDIA runner here follows with nvcr.io/nvidia/cuda:*-devel,
# and the thing clockprobe-rocm's first two CI builds paid to learn.
ARG ROCM_IMAGE=rocm/dev-ubuntu-24.04:6.4.4-complete
FROM ${ROCM_IMAGE} AS build

# The offload targets this image carries device code for.  HIP has no JIT
# fallback, so a part absent from this list has no code here at all.
#
# Unlike compute-smoke-rocm this list INCLUDES CDNA: the load is a plain fp32
# SGEMM with no matrix-instruction dependency, so one image serves both tiers.
#
#   gfx1151  Strix Halo / Ryzen AI Max (the hardware this exists for)
#   gfx1100  RDNA3 dGPU (7900 class)
#   gfx942   Instinct MI300X/MI325X
ARG GPU_TARGETS="gfx1151 gfx1100 gfx942"

RUN set -eu; \
    hipcc=/opt/rocm/bin/hipcc; \
    if [ ! -x "$hipcc" ]; then \
      echo "ERROR: $hipcc is missing; this base image is not the ROCm devel image it claims to be." >&2; \
      exit 1; \
    fi; \
    "$hipcc" --version | grep -qi 'rocm\|AMD clang' || { \
      echo "ERROR: $hipcc does not identify as a ROCm/AMD compiler." >&2; \
      exit 1; \
    }

COPY gpu_burn_rocm.cc soak_core_rocm.h /src/
COPY kmsg/kmsg_watch.h /src/kmsg/kmsg_watch.h
RUN mkdir -p /out && cd /src && /opt/rocm/bin/hipcc \
      -std=c++17 -O3 \
      $(for t in ${GPU_TARGETS}; do printf -- '--offload-arch=%s ' "$t"; done) \
      -o /out/gpu-burn-rocm gpu_burn_rocm.cc

RUN ldd /out/gpu-burn-rocm | tee /out/ldd.txt \
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
  # The origin pin: Ubuntu ships its own builds of ROCm package names, and the
  # runtime the binary loads must be the one it was compiled against.  See
  # clockprobe-rocm's README for the build this rule was learned from.
  && printf 'Package: *\nPin: origin repo.radeon.com\nPin-Priority: 1000\n' \
      > /etc/apt/preferences.d/rocm \
  && apt-get update && apt-get install -y --no-install-recommends \
      rocm-hip-runtime \
  && apt-get purge -y wget gnupg && apt-get autoremove -y \
  && rm -rf /var/lib/apt/lists/*

ENV LD_LIBRARY_PATH=/opt/rocm/lib

COPY --from=build /out/gpu-burn-rocm /usr/local/bin/gpu-burn-rocm

# Written as `if grep; then fail; fi` rather than `grep && { …; exit 1; } || true`,
# where the `|| true` would catch the exit and pass the step with the fault intact.
RUN if ldd /usr/local/bin/gpu-burn-rocm | grep -q 'not found'; then \
      echo "ERROR: unresolved libraries in the runtime stage:" >&2; \
      ldd /usr/local/bin/gpu-burn-rocm >&2; \
      exit 1; \
    fi

USER 65532:65532
ENTRYPOINT ["/usr/local/bin/gpu-burn-rocm"]
