# Runner image for the "power-swing" TestKind: alternate a heavy,
# correctness-checked load on the accelerator between "on" and "off" on a
# period, and watch the seconds right after each ramp for a clock dip or a
# fresh throttle reason — the signature of a power-delivery TRANSIENT
# (a VRM that cannot slew fast enough, or a PSU that sags when the board
# ramps) that a SUSTAINED soak like thermal-soak is not aimed at catching.
#
# It shares its entire implementation with runners/thermal-soak and
# runners/gpu-burn (soak_core.cuh, nvml_dynamic.h are byte-identical copies;
# see the header comments for why they are copies rather than a shared path).
# This kind differs from thermal-soak only in its duty cycle, its stdout
# vocabulary and in being thresholdless on everything but correctness — see
# power_swing.cu's own header comment.
#
# Licensing note — this is why the build is multi-stage. The CUDA toolchain is
# used at BUILD time only; none of its layers are shipped. The final image
# contains our Apache-2.0 binary on a distroless base and NO NVIDIA-licensed
# redistributable libraries. libcuda.so.1 is injected at runtime from the host
# driver by the NVIDIA Container Toolkit, so it is never baked in.
#   - our source        Apache-2.0   (power_swing.cu, soak_core.cuh, nvml_dynamic.h)
#   - third-party       none. The load is the same hand-written FP32 SGEMM
#                       thermal-soak and gpu-burn use, so there is no NVIDIA
#                       toolkit library to ship and no site mount to arrange.
#
# NVML is resolved at RUNTIME with dlopen, never linked. That is not a style
# choice: libnvidia-ml has no static form, so `-lnvidia-ml` would put a
# DT_NEEDED entry for an NVIDIA redistributable into the shipped binary and trip
# the assertion below. The driver's own libnvidia-ml.so.1 is mounted into the
# container by the toolkit when NVIDIA_DRIVER_CAPABILITIES includes "utility".

ARG CUDA_IMAGE=nvcr.io/nvidia/cuda:13.0.1-devel-ubuntu24.04
FROM ${CUDA_IMAGE} AS build

# Arch policy — identical reasoning to thermal-soak's own comment (this kind's
# claim is the same shape: the part held its clock and kept agreeing with
# itself, this time through a duty cycle rather than a sustained hold), so PTX
# JIT cannot fake any of the three measurands and the same list of real cubins
# plus a PTX fallback applies here too.
#
# NOTE: the publish-runner workflow's CUDA_ARCH input is deliberately NOT
# consumed here, for the same reason thermal-soak's is not — this image must
# run fleet-wide. Override CUDA_ARCH_FLAGS to trim or extend the list for your
# fleet; a single -arch=sm_121 also works.
#
#   sm_80   CC 8.0 – 8.9   A100, A10/A40, L4/L40S
#   sm_90   CC 9.0         H100, H200, GH200
#   sm_100  CC 10.0 – 10.x B200, B300, GB200, GB300
#   sm_120  CC 12.0        RTX PRO 6000 Blackwell, RTX 50xx
#   sm_121  CC 12.1        GB10 / DGX Spark
#   PTX     CC >= 12.1     JIT fallback for anything newer
ARG CUDA_ARCH_FLAGS="-gencode arch=compute_80,code=sm_80 \
-gencode arch=compute_90,code=sm_90 \
-gencode arch=compute_100,code=sm_100 \
-gencode arch=compute_120,code=sm_120 \
-gencode arch=compute_121,code=sm_121 \
-gencode arch=compute_121,code=compute_121"

COPY power_swing.cu /src/power_swing.cu
COPY soak_core.cuh /src/soak_core.cuh
COPY nvml_dynamic.h /src/nvml_dynamic.h
COPY device_fold.h /src/device_fold.h
COPY kmsg/kmsg_watch.h /src/kmsg/kmsg_watch.h

# -cudart static is nvcc's default on Linux but stated explicitly so the intent
# survives a toolchain change. -static-libstdc++/-static-libgcc are host-compiler
# flags and MUST go through -Xcompiler; passed bare, nvcc fails outright. Linking
# them drops the libstdc++/libgcc_s ABI dependency, leaving only glibc.
#
# No -ldl: dlopen/dlsym live in libc on glibc >= 2.34, and this base is well past
# that.
RUN mkdir -p /out && /usr/local/cuda/bin/nvcc \
      -std=c++17 -O3 ${CUDA_ARCH_FLAGS} \
      -cudart static -Xcompiler -static-libstdc++ -Xcompiler -static-libgcc \
      -I /src \
      -o /out/power_swing /src/power_swing.cu

# Fail the build rather than ship an image that silently drags in an NVIDIA .so.
RUN ldd /out/power_swing | tee /out/ldd.txt \
    && ! grep -qiE 'libcud(art|a)|libcublas|libnv' /out/ldd.txt

# ── final: no NVIDIA layers ───────────────────────────────────────────────────
# cc-debian13 carries a glibc at least as new as the ubuntu24.04 build stage.
# An older base (cc-debian12, glibc 2.36) fails at runtime with GLIBC_2.38 errors.
# A scratch/musl base will not work: the binary dlopen()s the driver and needs glibc.
FROM gcr.io/distroless/cc-debian13

# The NVIDIA-provided CUDA images set these; leaving that base means we must set
# them ourselves. "compute" is what mounts libcuda.so.1 so the binary can see the
# GPU at all; "utility" is what mounts libnvidia-ml.so.1, and without it this
# runner has no temperature, clock or throttle telemetry and exits Error rather
# than pretending the hardware is fine.
ENV NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility

COPY --from=build /out/power_swing /usr/local/bin/power_swing
USER 65532:65532
ENTRYPOINT ["/usr/local/bin/power_swing"]
