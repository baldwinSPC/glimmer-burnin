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
  # AMD's own documented origin pin, and it is load-bearing rather than
  # defensive.  Ubuntu noble/universe ships its OWN `hipcc` package (5.7.1-3,
  # installed to /usr/bin) and without this pin apt is free to prefer it: on
  # the first build of this image every other ROCm package resolved from
  # repo.radeon.com at 6.4.4 while hipcc alone came from archive.ubuntu.com,
  # and the compile below then died with `/opt/rocm/bin/hipcc: not found`.
  # Pinning the ORIGIN closes the whole class rather than that one symptom —
  # a distro-shadowed hsa-rocr or comgr would be far subtler than a missing
  # path, since it would build and then misbehave against a 6.4.4 runtime.
  && printf 'Package: *\nPin: origin repo.radeon.com\nPin-Priority: 1000\n' \
      > /etc/apt/preferences.d/rocm \
  # rocm-hip-runtime-dev is AMD's documented package set for COMPILING HIP
  # applications; the math libraries in rocm-hip-sdk are not needed here.
  && apt-get update && apt-get install -y --no-install-recommends \
      rocm-hip-runtime-dev \
  && rm -rf /var/lib/apt/lists/*

# Assert the compiler is the ROCm one BEFORE compiling with it, so a shadowed
# or relocated toolchain fails here — naming what it found — instead of at the
# compile step with a bare "not found", or worse, succeeding against a distro
# HIP whose runtime does not match the one the final stage ships.
RUN set -eu; \
    hipcc=/opt/rocm/bin/hipcc; \
    if [ ! -x "$hipcc" ]; then \
      echo "ERROR: $hipcc is missing. \`command -v hipcc\` resolves to: $(command -v hipcc || echo none)." >&2; \
      echo "A hipcc outside /opt/rocm is the distro package shadowing AMD's; check the origin pin above." >&2; \
      exit 1; \
    fi; \
    "$hipcc" --version; \
    "$hipcc" --version | grep -qi 'rocm\|AMD clang' || { \
      echo "ERROR: $hipcc does not identify as a ROCm/AMD compiler; refusing to build device code with it." >&2; \
      exit 1; \
    }

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

# Record what the shipped binary actually links, and assert the one dependency
# that decides whether the final stage is complete: libamdhip64.  The runtime
# stage installs rocm-hip-runtime for exactly this, and an image that reached a
# node without it would fail at exec — read as an infrastructure fault on the
# hardware rather than as the packaging mistake it is.
RUN ldd /out/clockprobe-rocm | tee /out/ldd.txt \
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
  # The same origin pin as the build stage, for the same reason: the runtime
  # the binary loads must be the 6.4.4 one it was compiled against, not
  # whatever version the distro happens to ship under the same package name.
  && printf 'Package: *\nPin: origin repo.radeon.com\nPin-Priority: 1000\n' \
      > /etc/apt/preferences.d/rocm \
  && apt-get update && apt-get install -y --no-install-recommends \
      rocm-hip-runtime \
  && apt-get purge -y wget gnupg && apt-get autoremove -y \
  && rm -rf /var/lib/apt/lists/*

# The ROCm debs land their libraries under /opt/rocm/lib; stating the loader
# path here keeps the binary independent of whether the package's ld.so.conf
# drop-in survived the purge above.
ENV LD_LIBRARY_PATH=/opt/rocm/lib

COPY --from=build /out/clockprobe-rocm /usr/local/bin/clockprobe-rocm
USER 65532:65532
ENTRYPOINT ["/usr/local/bin/clockprobe-rocm"]
