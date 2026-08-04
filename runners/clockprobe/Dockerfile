# Runner image for the "clockprobe" TestKind: hold a known, steady, clock-bound
# load and assert the accelerator actually SUSTAINS its rated clocks.  It is the
# one test in the suite that asserts speed rather than correctness or health, so
# it is the only one that catches a GPU pinned in a low P-state while reporting
# perfectly healthy utilization — on GB10, the USB-C Power-Delivery wedge.
#
# Licensing note — this is why the build is multi-stage.  The CUDA toolchain is
# used at BUILD time only; none of its layers are shipped.  The final image
# contains our Apache-2.0 binary on a distroless base and NO NVIDIA-licensed
# redistributable libraries.  libcuda.so.1 is injected at runtime from the host
# driver by the NVIDIA Container Toolkit, so it is never baked in.
#   - our source        Apache-2.0   (clockprobe.cu, nvml_dynamic.h)
#   - third-party       none.  Unlike compute-smoke this runner links no
#                       external library at all: the load kernel is plain FP32
#                       FMA and the NVML ABI is declared in nvml_dynamic.h
#                       rather than pulled from a vendor header.
#
# NVML is resolved at RUNTIME with dlopen, never linked.  That is not a style
# choice: libnvidia-ml has no static form, so `-lnvidia-ml` would put a
# DT_NEEDED entry for an NVIDIA redistributable into the shipped binary and trip
# the assertion below.  The driver's own libnvidia-ml.so.1 is mounted into the
# container by the toolkit when NVIDIA_DRIVER_CAPABILITIES includes "utility".

ARG CUDA_IMAGE=nvcr.io/nvidia/cuda:13.0.1-devel-ubuntu24.04
FROM ${CUDA_IMAGE} AS build

# Arch policy differs DELIBERATELY from compute-smoke, and the difference is not
# a relaxation of the "do not widen an arch target" rule — it follows from what
# each test proves.
#
# compute-smoke pins sm_121a with no PTX fallback because its claim is "the real
# block-scaled FP4 instruction path executed"; a JIT or emulation fallback could
# produce a passing result without ever touching the FP4 units, so the narrow
# target IS the proof.
#
# clockprobe makes no claim about which instructions ran.  Its measurand is the
# achieved SM clock and the throughput of an ordinary FP32 FMA chain, and PTX
# JIT compiles FFMA one-to-one — a JIT-compiled loop runs on the same SMs at the
# same clock and cannot fake either number.  Meanwhile the fault it detects is a
# fleet-wide power-delivery problem, so refusing to run on anything but GB10
# would be the actual failure.  Hence real cubins for the common datacentre
# parts plus a PTX fallback for anything newer.
#
# The list is ONE list for both host architectures, because a gencode says
# nothing about the CPU the container runs on: an sm_90 cubin is the same device
# code whether the host is x86 or Grace.  What it covers, with CUDA's
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
# in this image at all.  L40S (CC 8.9) runs the sm_80 cubin rather than an
# sm_89-tuned one; that is fine here, since the measurand is the achieved clock
# and the throughput of an FFMA chain, and a fleet declares its own thresholds.
#
# Trim or extend this list for your fleet; a single -arch=sm_121 also works.
ARG CUDA_ARCH_FLAGS="-gencode arch=compute_80,code=sm_80 \
-gencode arch=compute_90,code=sm_90 \
-gencode arch=compute_100,code=sm_100 \
-gencode arch=compute_120,code=sm_120 \
-gencode arch=compute_121,code=sm_121 \
-gencode arch=compute_121,code=compute_121"

COPY clockprobe.cu /src/clockprobe.cu
COPY nvml_dynamic.h /src/nvml_dynamic.h

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
      -o /out/clockprobe /src/clockprobe.cu

# Fail the build rather than ship an image that silently drags in an NVIDIA .so.
# For this runner the assertion also proves the dlopen design held: an accidental
# -lnvidia-ml would show up here as libnvidia-ml.so.1 and stop the build.
RUN ldd /out/clockprobe | tee /out/ldd.txt \
    && ! grep -qiE 'libcud(art|a)|libnv' /out/ldd.txt

# ── final: no NVIDIA layers ───────────────────────────────────────────────────
# cc-debian13 carries a glibc at least as new as the ubuntu24.04 build stage.
# An older base (cc-debian12, glibc 2.36) fails at runtime with GLIBC_2.38 errors.
# A scratch/musl base will not work: the binary dlopen()s the driver and needs glibc.
FROM gcr.io/distroless/cc-debian13

# The NVIDIA-provided CUDA images set these; leaving that base means we must set
# them ourselves.  "compute" is what mounts libcuda.so.1 so the binary can see the
# GPU at all; "utility" is what mounts libnvidia-ml.so.1, and without it this
# runner has no clock, temperature or throttle telemetry to probe and exits Error
# rather than pretending the hardware is fine.
ENV NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility

COPY --from=build /out/clockprobe /usr/local/bin/clockprobe
USER 65532:65532
ENTRYPOINT ["/usr/local/bin/clockprobe"]
