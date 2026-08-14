# Runner image for the "clockprobe" TestKind on AMD accelerators — selected per
# node via spec.runner.imagesByVendor ({vendor: amd}), never as a kind of its
# own.  It holds a known, steady, clock-bound load and asserts the accelerator
# actually SUSTAINS the top of its own DPM ladder; the fault it exists to catch
# is the amdgpu idle-clock lock (ROCm issue #5750), where a part reports busy
# and healthy while pinned at the bottom of the ladder.
#
# PLATFORM: linux/amd64 ONLY, deliberately.  AMD publishes the ROCm toolchain
# and its container images for amd64 alone — and every AMD accelerator host
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

# AMD's own devel image, matching how every NVIDIA runner here builds against
# nvcr.io/nvidia/cuda:*-devel rather than assembling a toolchain from apt.  The
# first two builds of this image did assemble it, and both failed in ways that
# are entirely about package archaeology rather than about this runner: Ubuntu
# noble/universe shadowed AMD's hipcc with its own 5.7.1 build, and once that
# was pinned away the hand-picked package set still lacked what amdgcn-link
# needs to link device code.  A vendor devel image is the shape of dependency
# this project already trusts, and it is one pinned artifact instead of a
# resolution that can drift under us.
#
# 6.4.4 is the first release train listing gfx1151 (Strix Halo) as supported.
# Community reports of ROCm 7.x regressions on gfx1151 are why this stays on
# 6.4.x until the hardware pass can measure both.
ARG ROCM_IMAGE=rocm/dev-ubuntu-24.04:6.4.4-complete
FROM ${ROCM_IMAGE} AS build

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

# Assert the compiler is the ROCm one BEFORE compiling with it, so a relocated
# or substituted toolchain fails here — naming what it found — instead of at
# the compile step with a bare "not found", or worse, succeeding against a
# distro HIP whose runtime does not match the one the final stage ships.
RUN set -eu; \
    hipcc=/opt/rocm/bin/hipcc; \
    if [ ! -x "$hipcc" ]; then \
      echo "ERROR: $hipcc is missing. \`command -v hipcc\` resolves to: $(command -v hipcc || echo none)." >&2; \
      echo "A hipcc outside /opt/rocm means this base image is not the ROCm devel image it claims to be." >&2; \
      exit 1; \
    fi; \
    "$hipcc" --version; \
    "$hipcc" --version | grep -qi 'rocm\|AMD clang' || { \
      echo "ERROR: $hipcc does not identify as a ROCm/AMD compiler; refusing to build device code with it." >&2; \
      exit 1; \
    }

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
# NOT the devel image: it carries the whole toolchain and the math libraries,
# none of which a probe needs on a node.
FROM ubuntu:24.04

ARG ROCM_VERSION=6.4.4
RUN apt-get update && apt-get install -y --no-install-recommends \
      ca-certificates wget gnupg \
  && wget -qO- https://repo.radeon.com/rocm/rocm.gpg.key \
      | gpg --dearmor > /usr/share/keyrings/rocm.gpg \
  && echo "deb [arch=amd64 signed-by=/usr/share/keyrings/rocm.gpg] https://repo.radeon.com/rocm/apt/${ROCM_VERSION} noble main" \
      > /etc/apt/sources.list.d/rocm.list \
  # AMD's own documented origin pin, and it is load-bearing rather than
  # defensive.  Ubuntu noble/universe ships its own builds of ROCm package
  # names — its hipcc (5.7.1-3) is what broke this image's first CI build by
  # shadowing AMD's — and the runtime the binary loads must be the 6.4.4 one
  # it was compiled against.  Pinning the ORIGIN closes the whole class: a
  # distro-shadowed hsa-rocr or comgr would install quietly and then misbehave
  # against a 6.4.4-compiled binary, which is far subtler than a missing path.
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

# The runtime stage is assembled separately from the build stage, so assert the
# binary it will actually execute resolves every library HERE.  Without this a
# packaging gap surfaces on a node as an exec failure, which the operator
# records as an infrastructure Error against hardware that was never measured.
#
# Written as `if grep; then fail; fi` rather than `grep && { …; exit 1; } || true`
# on purpose: in the latter, the `|| true` catches the failure of the `&&`
# branch — including the `exit 1` — and the step passes with the fault intact.
# That masked-status shape is a recurring bug class in this project.
RUN if ldd /usr/local/bin/clockprobe-rocm | grep -q 'not found'; then \
      echo "ERROR: unresolved libraries in the runtime stage:" >&2; \
      ldd /usr/local/bin/clockprobe-rocm >&2; \
      exit 1; \
    fi

USER 65532:65532
ENTRYPOINT ["/usr/local/bin/clockprobe-rocm"]
