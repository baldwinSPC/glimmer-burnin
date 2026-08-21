# Runner image for the "thermal-soak" TestKind: hold a heavy, correctness-checked
# load on the accelerator for the whole requested duration and assert that it
# neither throttles to protect itself nor starts returning different answers.
#
# It shares its entire implementation with the gpu-burn runner (soak_core.cuh,
# nvml_dynamic.h are byte-identical copies; see the header comments for why they
# are copies rather than a shared path).  The two kinds differ only in their
# stdout vocabulary and in which question decides the exit code.
#
# Licensing note — this is why the build is multi-stage.  The CUDA toolchain is
# used at BUILD time only; none of its layers are shipped.  The final image
# contains our Apache-2.0 binary on a distroless base and NO NVIDIA-licensed
# redistributable libraries.  libcuda.so.1 is injected at runtime from the host
# driver by the NVIDIA Container Toolkit, so it is never baked in.
#   - our source        Apache-2.0   (thermal_soak.cu, soak_core.cuh, nvml_dynamic.h)
#   - third-party       none.  The load is a hand-written FP32 SGEMM, so unlike a
#                       cuBLAS-based burn there is no NVIDIA toolkit library to
#                       ship and no site mount to arrange.  See runners/gpu-burn
#                       for the full reasoning; it is the same decision.
#
# NVML is resolved at RUNTIME with dlopen, never linked.  That is not a style
# choice: libnvidia-ml has no static form, so `-lnvidia-ml` would put a
# DT_NEEDED entry for an NVIDIA redistributable into the shipped binary and trip
# the assertion below.  The driver's own libnvidia-ml.so.1 is mounted into the
# container by the toolkit when NVIDIA_DRIVER_CAPABILITIES includes "utility".

ARG CUDA_IMAGE=nvcr.io/nvidia/cuda:13.0.1-devel-ubuntu24.04
FROM ${CUDA_IMAGE} AS build

# Arch policy, and why it differs from compute-smoke's single narrow target.
#
# compute-smoke pins sm_121a with no PTX fallback because its claim is "the real
# block-scaled FP4 instruction path executed"; a JIT or emulation fallback could
# produce a passing result without ever touching the FP4 units, so the narrow
# target IS the proof.
#
# A soak claims something else entirely: that the part held its clock, stayed
# within its thermal envelope and kept agreeing with itself, under an ordinary
# FP32 FMA workload.  PTX JIT compiles FFMA one-to-one, so a JIT-compiled GEMM
# runs on the same SMs at the same clock, draws the same power and produces the
# same bit patterns — it cannot fake any of the three measurands.  Meanwhile the
# faults this catches (a dry thermal interface, a blocked intake, marginal
# silicon) are fleet-wide problems, so an image that refuses to run on anything
# but GB10 would be the actual failure.  Hence real cubins for the common
# datacentre parts plus a PTX fallback for anything newer.
#
# NOTE: the publish-runner workflow's CUDA_ARCH input is deliberately NOT
# consumed here, for the reason above — a soak image must run fleet-wide, and a
# single "a"-suffixed target would strip the fallback.  Override
# CUDA_ARCH_FLAGS instead to trim or extend the list for your fleet; a single
# -arch=sm_121 also works.
#
# The list is ONE list for both host architectures, because a gencode says
# nothing about the CPU the container runs on.  What it covers, with CUDA's
# minor-version binary compatibility (a cubin for CC X.Y runs on CC X.Z, Z >= Y):
#
#   sm_80   CC 8.0 – 8.9   A100, A10/A40, L4/L40S
#   sm_90   CC 9.0         H100, H200, GH200
#   sm_100  CC 10.0 – 10.x B200, B300, GB200, GB300
#   sm_120  CC 12.0        RTX PRO 6000 Blackwell, RTX 50xx
#   sm_121  CC 12.1        GB10 / DGX Spark
#   PTX     CC >= 12.1     JIT fallback for anything newer
#
# sm_100 was added because it was the one gap that mattered for x86 adopters:
# compute_121 PTX cannot JIT DOWN onto CC 10.x, so before it a B200 had no code
# in this image at all — and a soak that cannot launch on the part it was sent to
# is an Error on every node of a Blackwell fleet.
ARG CUDA_ARCH_FLAGS="-gencode arch=compute_80,code=sm_80 \
-gencode arch=compute_90,code=sm_90 \
-gencode arch=compute_100,code=sm_100 \
-gencode arch=compute_120,code=sm_120 \
-gencode arch=compute_121,code=sm_121 \
-gencode arch=compute_121,code=compute_121"

COPY thermal_soak.cu /src/thermal_soak.cu
COPY soak_core.cuh /src/soak_core.cuh
COPY nvml_dynamic.h /src/nvml_dynamic.h
COPY device_fold.h /src/device_fold.h
COPY kmsg/kmsg_watch.h /src/kmsg/kmsg_watch.h

# -cudart static is nvcc's default on Linux but stated explicitly so the intent
# survives a toolchain change.  -static-libstdc++/-static-libgcc are host-compiler
# flags and MUST go through -Xcompiler; passed bare, nvcc fails outright.  Linking
# them drops the libstdc++/libgcc_s ABI dependency, leaving only glibc.
#
# No -ldl: dlopen/dlsym live in libc on glibc >= 2.34, and this base is well past
# that.  Linking libdl explicitly would add a DT_NEEDED entry for nothing.
RUN mkdir -p /out && /usr/local/cuda/bin/nvcc \
      -std=c++17 -O3 ${CUDA_ARCH_FLAGS} \
      -cudart static -Xcompiler -static-libstdc++ -Xcompiler -static-libgcc \
      -I /src \
      -o /out/thermal_soak /src/thermal_soak.cu

# Fail the build rather than ship an image that silently drags in an NVIDIA .so.
# For this runner the assertion also proves the dlopen design held: an accidental
# -lnvidia-ml would show up here as libnvidia-ml.so.1 and stop the build.  The
# pattern also catches libcublas, which is what a cuBLAS-based burn would need.
RUN ldd /out/thermal_soak | tee /out/ldd.txt \
    && ! grep -qiE 'libcud(art|a)|libcublas|libnv' /out/ldd.txt

# ── final: no NVIDIA layers ───────────────────────────────────────────────────
# cc-debian13 carries a glibc at least as new as the ubuntu24.04 build stage.
# An older base (cc-debian12, glibc 2.36) fails at runtime with GLIBC_2.38 errors.
# A scratch/musl base will not work: the binary dlopen()s the driver and needs glibc.
FROM gcr.io/distroless/cc-debian13

# The NVIDIA-provided CUDA images set these; leaving that base means we must set
# them ourselves.  "compute" is what mounts libcuda.so.1 so the binary can see the
# GPU at all; "utility" is what mounts libnvidia-ml.so.1, and without it this
# runner has no temperature, clock or throttle telemetry and exits Error rather
# than pretending the hardware is fine.
ENV NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility

COPY --from=build /out/thermal_soak /usr/local/bin/thermal_soak
USER 65532:65532
ENTRYPOINT ["/usr/local/bin/thermal_soak"]
