# Runner image for the "gpu-burn" TestKind: hold a heavy compute load on the
# accelerator and assert that it keeps returning the SAME answer to identical
# arithmetic.  The measurand is correctness under sustained load, not heat.
#
# It shares its entire implementation with the thermal-soak runner
# (soak_core.cuh, nvml_dynamic.h are byte-identical copies; see the header
# comments for why they are copies rather than a shared path).  The two kinds
# differ only in their stdout vocabulary and in which question decides the exit
# code.
#
# ── Why this image does not contain wilicc/gpu-burn ──────────────────────────
#
# The kind is named after Ville Timonen's gpu-burn
# (https://github.com/wilicc/gpu-burn), and its licence is not the problem:
# LICENSE in that repository is BSD-2-Clause, which is on this project's
# permissive allow-list.
#
# What it LINKS is the problem.  gpu-burn's numeric core is cuBLAS
# (cublasSgemm/cublasDgemm), and cuBLAS is covered by the NVIDIA CUDA EULA.
# Critically, cuBLAS is a TOOLKIT library, not a driver library: the NVIDIA
# Container Toolkit injects libcuda.so.1 and libnvidia-ml.so.1 from the host
# driver at runtime, and does NOT inject libcublas.  A working cuBLAS-based
# image therefore has to ship libcublas itself — which is precisely the
# redistribution this project's runner images may not do — or adopt the posture
# runners/dcgm-diag was forced into, where the SITE mounts the NVIDIA component
# and the runner reports Error rather than a verdict on any node where that
# mount is missing.
#
# That posture is a real cost, and it is worth paying only when the upstream
# tool does something we cannot.  Here it does not.  gpu-burn's method is: run a
# big GEMM repeatedly, compare each result against the first, count the elements
# that differ.  The value is in the method, not in cuBLAS.  A self-contained
# hand-written SGEMM keeps the image redistributable, removes the site mount
# entirely, and — because it is the same kernel thermal-soak runs — makes the
# two kinds' throughput figures the same measurement rather than two different
# numbers reported under one metric name.  cuBLAS would reach a higher absolute
# TFLOPS; it would not detect one more wrong answer.
#
# Verified on the hardware rather than assumed: cuBLAS 13 does support sm_121
# and does run on GB10.  The blocker is redistribution, not capability.
#
# Licensing summary for this image:
#   - our source        Apache-2.0   (gpu_burn.cu, soak_core.cuh, nvml_dynamic.h)
#   - third-party       none.  No cuBLAS, no upstream gpu-burn source, no
#                       NVIDIA redistributable of any kind.
#
# NVML is resolved at RUNTIME with dlopen, never linked.  That is not a style
# choice: libnvidia-ml has no static form, so `-lnvidia-ml` would put a
# DT_NEEDED entry for an NVIDIA redistributable into the shipped binary and trip
# the assertion below.

ARG CUDA_IMAGE=nvcr.io/nvidia/cuda:13.0.1-devel-ubuntu24.04
FROM ${CUDA_IMAGE} AS build

# Arch policy, and why it differs from compute-smoke's single narrow target.
#
# compute-smoke pins sm_121a with no PTX fallback because its claim is "the real
# block-scaled FP4 instruction path executed"; a JIT or emulation fallback could
# produce a passing result without ever touching the FP4 units, so the narrow
# target IS the proof.
#
# A burn claims something else: that the part keeps agreeing with itself under
# sustained load.  PTX JIT compiles FFMA one-to-one, so a JIT-compiled GEMM
# produces the same bit patterns on the same SMs and cannot fake a miscompare
# count either way.  Meanwhile silent data corruption is a fleet-wide concern,
# so an image that refuses to run on anything but GB10 would be the actual
# failure.  Hence real cubins for the common datacentre parts plus a PTX
# fallback for anything newer.
#
# NOTE: the publish-runner workflow's CUDA_ARCH input is deliberately NOT
# consumed here, for the reason above.  Override CUDA_ARCH_FLAGS instead to trim
# or extend the list for your fleet; a single -arch=sm_121 also works.
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
# in this image at all — and silent data corruption is precisely what a
# Blackwell fleet wants this runner to look for.
ARG CUDA_ARCH_FLAGS="-gencode arch=compute_80,code=sm_80 \
-gencode arch=compute_90,code=sm_90 \
-gencode arch=compute_100,code=sm_100 \
-gencode arch=compute_120,code=sm_120 \
-gencode arch=compute_121,code=sm_121 \
-gencode arch=compute_121,code=compute_121"

COPY gpu_burn.cu /src/gpu_burn.cu
COPY soak_core.cuh /src/soak_core.cuh
COPY nvml_dynamic.h /src/nvml_dynamic.h
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
      -o /out/gpu_burn /src/gpu_burn.cu

# Fail the build rather than ship an image that silently drags in an NVIDIA .so.
# libcublas is in the pattern on purpose: it is the one library a future edit of
# this runner is most likely to reach for, and this line is what stops that edit
# from quietly turning a redistributable image into one that is not.
RUN ldd /out/gpu_burn | tee /out/ldd.txt \
    && ! grep -qiE 'libcud(art|a)|libcublas|libnv' /out/ldd.txt

# ── final: no NVIDIA layers ───────────────────────────────────────────────────
# cc-debian13 carries a glibc at least as new as the ubuntu24.04 build stage.
# An older base (cc-debian12, glibc 2.36) fails at runtime with GLIBC_2.38 errors.
# A scratch/musl base will not work: the binary dlopen()s the driver and needs glibc.
FROM gcr.io/distroless/cc-debian13

# The NVIDIA-provided CUDA images set these; leaving that base means we must set
# them ourselves.  "compute" is what mounts libcuda.so.1 so the binary can see the
# GPU at all; "utility" is what mounts libnvidia-ml.so.1, and without it this
# runner cannot read the ECC state it must either report or declare unmeasurable,
# so it exits Error rather than pretending the hardware is fine.
ENV NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility

COPY --from=build /out/gpu_burn /usr/local/bin/gpu_burn
USER 65532:65532
ENTRYPOINT ["/usr/local/bin/gpu_burn"]
