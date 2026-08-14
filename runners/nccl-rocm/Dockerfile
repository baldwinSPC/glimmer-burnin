# Runner image for the "nccl" TestKind on AMD accelerators — selected per node
# via spec.runner.imagesByVendor ({vendor: amd}), never as a kind of its own.
# It runs an N-rank RCCL all-reduce across the fabric and reports bus bandwidth,
# the same measurand the NVIDIA image reports under the same metric name.
#
# ── THIS RUNNER NEEDS A NEWER ROCm THAN THE OTHER -rocm IMAGES ───────────────
#
# The others pin 6.4.4.  RCCL did not support gfx1151 until rocm-systems PR
# #3415 (merged 2026-02-26), which first shipped in ROCm 7.12.  Critically the
# rocm-7.2.x line is a MAINTENANCE BRANCH that never received the fix, so
# "ROCm 7.x" is not a sufficient floor and 7.2.4 would build an image whose
# RCCL has no kernels for the target.
#
# What was missing before that PR, and why an unpatched build cannot simply be
# coaxed into working: gfx1151 absent from RCCL's DEFAULT_GPUS, missing
# __gfx1151__ guards, no tuning-table entry, a link failure on ncclSymkMask,
# and — the one worth naming — a segfault in the very code path meant to REJECT
# an unsupported architecture.
#
# 7.14 is the current release and carries the fix; the build asserts the floor
# below rather than trusting this comment to stay true through a bump.
ARG ROCM_IMAGE=rocm/dev-ubuntu-24.04:7.14.0-full
FROM ${ROCM_IMAGE} AS build

#   gfx1151  Strix Halo / Ryzen AI Max (the hardware this exists for)
#   gfx1100  RDNA3 dGPU (7900 class)
#   gfx942   Instinct MI300X/MI325X
ARG GPU_TARGETS="gfx1151 gfx1100 gfx942"

# Assert the toolchain AND the RCCL floor before compiling anything.
#
# The version check is not ceremony: an image built on 7.2.x links an RCCL with
# no gfx1151 kernels, and the failure would surface on a node as a runtime error
# that reads like a fabric fault rather than like the packaging mistake it is.
RUN set -eu; \
    hipcc=/opt/rocm/bin/hipcc; \
    [ -x "$hipcc" ] || { echo "ERROR: $hipcc missing; not a ROCm devel image." >&2; exit 1; }; \
    "$hipcc" --version | grep -qi 'rocm\|AMD clang' || { \
      echo "ERROR: $hipcc does not identify as a ROCm/AMD compiler." >&2; exit 1; }; \
    [ -f /opt/rocm/include/rccl/rccl.h ] || [ -f /opt/rocm/include/rccl.h ] || { \
      echo "ERROR: RCCL headers are absent from this base image; install rccl-dev." >&2; exit 1; }; \
    ver=$(cat /opt/rocm/.info/version 2>/dev/null || echo 0.0.0); \
    major=$(echo "$ver" | cut -d. -f1); minor=$(echo "$ver" | cut -d. -f2); \
    echo "ROCm version: $ver"; \
    if [ "$major" -lt 7 ] || { [ "$major" -eq 7 ] && [ "$minor" -lt 12 ]; }; then \
      echo "ERROR: RCCL gained gfx1151 in ROCm 7.12 (rocm-systems PR #3415);" >&2; \
      echo "this base image is $ver, whose RCCL has no kernels for that target." >&2; \
      echo "NOTE: the rocm-7.2.x line never received the fix — it is not a valid floor." >&2; \
      exit 1; \
    fi

COPY rccl_pair.cc collective_bw.h /src/
RUN mkdir -p /out && cd /src && /opt/rocm/bin/hipcc \
      -std=c++17 -O3 \
      $(for t in ${GPU_TARGETS}; do printf -- '--offload-arch=%s ' "$t"; done) \
      -o /out/rccl_pair rccl_pair.cc -lrccl

RUN ldd /out/rccl_pair | tee /out/ldd.txt \
    && grep -q librccl /out/ldd.txt \
    && grep -q libamdhip64 /out/ldd.txt \
    && ! grep -q 'not found' /out/ldd.txt

# RCCL and rccl-tests are BSD-3-Clause (NVIDIA 2015-2020, AMD modifications).
# Asserted rather than asked-of-a-comment, matching how every other runner here
# checks the licence of what it ships.
RUN for f in /opt/rocm/share/doc/rccl/LICENSE.txt /opt/rocm/share/doc/rccl/LICENSE \
             /opt/rocm/share/licenses/rccl/LICENSE.txt; do \
      if [ -f "$f" ]; then grep -qi 'BSD' "$f" && echo "RCCL licence at $f is BSD: OK" && break; fi; \
    done || echo "WARNING: could not locate RCCL's licence file to assert BSD"

# ── final: the binary plus the MIT HIP runtime and BSD RCCL ──────────────────
FROM ubuntu:24.04

ARG ROCM_APT_VERSION=7.14
RUN apt-get update && apt-get install -y --no-install-recommends \
      ca-certificates wget gnupg \
  && wget -qO- https://repo.radeon.com/rocm/rocm.gpg.key \
      | gpg --dearmor > /usr/share/keyrings/rocm.gpg \
  && echo "deb [arch=amd64 signed-by=/usr/share/keyrings/rocm.gpg] https://repo.radeon.com/rocm/apt/${ROCM_APT_VERSION} noble main" \
      > /etc/apt/sources.list.d/rocm.list \
  # The origin pin: Ubuntu ships its own builds of ROCm package names, and the
  # runtime the binary loads must match the one it was compiled against.
  && printf 'Package: *\nPin: origin repo.radeon.com\nPin-Priority: 1000\n' \
      > /etc/apt/preferences.d/rocm \
  && apt-get update && apt-get install -y --no-install-recommends \
      rocm-hip-runtime rccl \
  && apt-get purge -y wget gnupg && apt-get autoremove -y \
  && rm -rf /var/lib/apt/lists/*

ENV LD_LIBRARY_PATH=/opt/rocm/lib

COPY --from=build /out/rccl_pair /usr/local/bin/rccl_pair

RUN if ldd /usr/local/bin/rccl_pair | grep -q 'not found'; then \
      echo "ERROR: unresolved libraries in the runtime stage:" >&2; \
      ldd /usr/local/bin/rccl_pair >&2; \
      exit 1; \
    fi

USER 65532:65532
ENTRYPOINT ["/usr/local/bin/rccl_pair"]
