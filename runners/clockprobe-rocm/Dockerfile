# Runner image for the "clockprobe" TestKind on AMD accelerators — selected per
# node via spec.runner.imagesByVendor ({vendor: amd}), never as a kind of its
# own.  It holds a known, steady, clock-bound load and asserts the accelerator
# actually SUSTAINS the top of its own DPM ladder; the fault it exists to catch
# is the amdgpu idle-clock lock (ROCm issue #5750), where a part reports busy
# and healthy while pinned at the bottom of the ladder.
#
# PLATFORM: linux/amd64 ONLY, deliberately.  AMD publishes the ROCm apt
# packages this build needs for amd64 alone — and every AMD accelerator host
# this image could measure (Strix Halo APUs, Instinct) is amd64, so an arm64
# leg would build an image for a machine that does not exist.  ci.yml excludes
# the arm64 matrix leg for this runner; publish-runner.yml's two-platform
# assertion must grow the same awareness before this runner is ever published.
#
# Licensing note — unlike the NVIDIA runners, the final image SHIPS its
# accelerator runtime: the ROCm/HIP userspace is MIT-licensed, so carrying
# libamdhip64 and its dependencies is compatible with this repository's
# permissive-only rule.  What the image does NOT carry is the kernel driver;
# /dev/kfd and /dev/dri arrive from the host via the device plugin.
#   - our source        Apache-2.0   (clockprobe_rocm.cc, sysfs_clocks.h)
#   - shipped runtime   MIT          (ROCm HIP runtime and its ROCm deps)
#   - third-party       none linked beyond the HIP runtime

FROM ubuntu:24.04 AS build

# Pinned by version through AMD's own apt channel; a floating ROCm would change
# the compiler under every existing profile with no audit trail.  6.4.4 is the
# first release train that lists gfx1151 (Strix Halo) as supported.
ARG ROCM_VERSION=6.4.4
RUN apt-get update && apt-get install -y --no-install-recommends \
      ca-certificates wget gnupg \
  && wget -qO- https://repo.radeon.com/rocm/rocm.gpg.key \
      | gpg --dearmor > /usr/share/keyrings/rocm.gpg \
  && echo "deb [arch=amd64 signed-by=/usr/share/keyrings/rocm.gpg] https://repo.radeon.com/rocm/apt/${ROCM_VERSION} noble main" \
      > /etc/apt/sources.list.d/rocm.list \
  && apt-get update && apt-get install -y --no-install-recommends \
      hipcc hip-dev hip-runtime-amd \
  && rm -rf /var/lib/apt/lists/*

# The offload targets this image carries device code for.  HIP has no PTX-style
# JIT fallback: a part whose gfx target is not in this list gets NO code from
# this image and HIP reports it unusable, which the runner reports as an Error
# (hardware unjudged) — never a Skip and never a Fail.
#
#   gfx1151  Strix Halo / Ryzen AI Max (the hardware this variant exists for)
#   gfx1100  RDNA3 dGPUs (7900 class), the nearest common desktop part
#   gfx942   Instinct MI300X/MI325X, so one image serves the datacenter tier
#
# Trim or extend for your fleet; a single --offload-arch=gfx1151 also works.
ARG GPU_TARGETS="gfx1151 gfx1100 gfx942"

COPY clockprobe_rocm.cc sysfs_clocks.h /src/
RUN mkdir -p /out && cd /src && /opt/rocm/bin/hipcc \
      -std=c++17 -O3 \
      $(for t in ${GPU_TARGETS}; do printf -- '--offload-arch=%s ' "$t"; done) \
      -o /out/clockprobe-rocm clockprobe_rocm.cc

# ── final: the binary plus the MIT HIP runtime, nothing else ─────────────────
FROM ubuntu:24.04

ARG ROCM_VERSION=6.4.4
RUN apt-get update && apt-get install -y --no-install-recommends \
      ca-certificates wget gnupg \
  && wget -qO- https://repo.radeon.com/rocm/rocm.gpg.key \
      | gpg --dearmor > /usr/share/keyrings/rocm.gpg \
  && echo "deb [arch=amd64 signed-by=/usr/share/keyrings/rocm.gpg] https://repo.radeon.com/rocm/apt/${ROCM_VERSION} noble main" \
      > /etc/apt/sources.list.d/rocm.list \
  && apt-get update && apt-get install -y --no-install-recommends \
      hip-runtime-amd \
  && apt-get purge -y wget gnupg && apt-get autoremove -y \
  && rm -rf /var/lib/apt/lists/*

# The ROCm debs land their libraries under /opt/rocm/lib; stating the loader
# path here keeps the binary independent of whether the package's ld.so.conf
# drop-in survived the purge above.
ENV LD_LIBRARY_PATH=/opt/rocm/lib

COPY --from=build /out/clockprobe-rocm /usr/local/bin/clockprobe-rocm
USER 65532:65532
ENTRYPOINT ["/usr/local/bin/clockprobe-rocm"]
