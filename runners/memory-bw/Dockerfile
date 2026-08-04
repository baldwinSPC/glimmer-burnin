# Runner image for the "memory-bw" TestKind: measures the host<->device copy
# paths and the on-device copy path, so a node that computes at full rate but
# moves memory at half rate is caught as a bandwidth fault rather than being
# mistaken for a healthy part.
#
# The workload is NVIDIA nvbandwidth (Apache-2.0), built from source, driven by
# a small Apache-2.0 wrapper that turns its human-readable matrix into the
# runner contract (key=value metrics, exit 0/1/2/other).
#
# Licensing note — this is why the build is multi-stage.  The CUDA toolchain is
# used at BUILD time only; none of its layers are shipped.  The final image
# contains two binaries on a distroless base and NO NVIDIA-licensed
# redistributable libraries.  The only NVIDIA objects the shipped binaries
# reference are the host driver libraries libcuda.so.1 and libnvidia-ml.so.1,
# which the NVIDIA Container Toolkit injects at runtime and which are never
# baked in.  The build asserts exactly that, below.
#   - our source            Apache-2.0
#   - NVIDIA nvbandwidth    Apache-2.0     (built from source, binary shipped)
#   - p-ranav/argparse      MIT            (header-only, compiled into nvbandwidth)
#   - JsonCpp               MIT            (vendored in nvbandwidth, compiled in)

# Declared before the first FROM so the build stage and the final stage agree on
# where nvbandwidth lives. The wrapper is compiled against this path, so the two
# cannot drift into an image whose entrypoint looks for a binary that is not there.
ARG NVBANDWIDTH_PATH=/usr/local/bin/nvbandwidth

# Both base images are declared here, in the global scope, because an ARG a FROM
# expands MUST be declared before the first FROM. An ARG declared after a FROM
# belongs to that build stage and is invisible to any later FROM, which fails the
# build with "base name should not be blank" rather than with anything that
# points at the cause. RUNTIME_IMAGE is documented at the final FROM, where it is
# used.
ARG CUDA_IMAGE=nvcr.io/nvidia/cuda:13.0.1-devel-ubuntu24.04
ARG RUNTIME_IMAGE=gcr.io/distroless/cc-debian13

FROM ${CUDA_IMAGE} AS build

# Pinned, and pinned deliberately: a runner image is executed by a node's
# readiness gate, so the tag a gate pins must always mean the same measurement.
ARG NVBANDWIDTH_REF=v0.10.0
# And a tag alone does not deliver that, because a tag can be moved. The runner
# reports NVBANDWIDTH_REF in its own output as the provenance of every number it
# publishes, so a moved tag would make that line a false statement about which
# code produced the measurement. Bumping NVBANDWIDTH_REF means bumping
# NVBANDWIDTH_SHA with it; resolve it with
# `git ls-remote https://github.com/NVIDIA/nvbandwidth.git refs/tags/<ref>`.
ARG NVBANDWIDTH_SHA=82fc4e8c6afa0babb8687793678f615b3b8d793e

# Empty means "use nvbandwidth's own architecture list", which is what its
# CMakeLists picks per CUDA version and which includes PTX for JIT fallback.
# nvbandwidth honours a caller-supplied -DCMAKE_CUDA_ARCHITECTURES, so set this
# (e.g. CUDA_ARCH="121" for GB10) to emit a native cubin instead.
#
# Nothing this runner reports depends on the choice: all three testcases it runs
# are copy-engine testcases, which move data without launching a kernel. The arch
# list only matters if you switch the plan in memory_bw.cc to the _sm variants —
# and then it matters completely, because a JIT-compiled kernel is not the code
# path a gate is trying to accept.
ARG CUDA_ARCH=

# Re-declared so the pre-FROM value is visible inside this stage.
ARG NVBANDWIDTH_PATH

RUN apt-get update && apt-get install -y --no-install-recommends \
        binutils build-essential ca-certificates cmake git \
    && rm -rf /var/lib/apt/lists/*

RUN git clone --depth 1 -b ${NVBANDWIDTH_REF} https://github.com/NVIDIA/nvbandwidth.git /nvbandwidth

RUN got="$(git -C /nvbandwidth rev-parse HEAD)"; \
    if [ "${got}" != "${NVBANDWIDTH_SHA}" ]; then \
      echo "refusing to build: ${NVBANDWIDTH_REF} is ${got}, expected ${NVBANDWIDTH_SHA}" >&2; \
      exit 1; \
    fi; \
    echo "nvbandwidth ${NVBANDWIDTH_REF} = ${got}: OK"

# Three things in this invocation are load-bearing:
#
#   CMAKE_CUDA_RUNTIME_LIBRARY=Static — CMake has defaulted to Static since 3.17,
#   but libcudart IS an NVIDIA redistributable, so the one thing standing between
#   this image and a licence violation is stated explicitly rather than inherited.
#
#   -static-libstdc++ -static-libgcc — the distroless base carries glibc and
#   nothing else. Without these the image starts and immediately dies on a
#   missing libstdc++.so.6. (The linker language for this target is C++, so the
#   flags are passed bare; were nvcc doing the link they would need -Xcompiler.
#   The assertion below is what catches it if that ever changes.)
#
#   -L…/stubs — nvbandwidth links the driver (-lcuda) and NVML (-lnvidia-ml).
#   Neither ships in this image; the toolkit's link-time stubs live off the
#   default search path, and the SONAME they carry is what makes the binary look
#   for the real host driver at runtime.
RUN cmake -S /nvbandwidth -B /nvbandwidth/build \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_CUDA_RUNTIME_LIBRARY=Static \
      -DCMAKE_EXE_LINKER_FLAGS="-static-libstdc++ -static-libgcc -L/usr/local/cuda/lib64/stubs" \
      ${CUDA_ARCH:+-DCMAKE_CUDA_ARCHITECTURES=${CUDA_ARCH}} \
    && cmake --build /nvbandwidth/build --parallel "$(nproc)"

RUN mkdir -p /out && install -m 0755 /nvbandwidth/build/nvbandwidth /out/nvbandwidth

COPY memory_bw.cc /src/memory_bw.cc

# The wrapper's two build-time constants are generated into a header rather than
# passed as quoted -D arguments: the value has to survive the Dockerfile parser
# and the shell intact, and a mangled quote here would silently ship an image
# that reports the wrong nvbandwidth version. memory_bw.cc guards both with
# #ifndef, so it still compiles standalone.
RUN printf '#define NVBANDWIDTH_REF "%s"\n#define NVBANDWIDTH_BIN "%s"\n' \
        "${NVBANDWIDTH_REF}" "${NVBANDWIDTH_PATH}" > /src/nvbandwidth_build.h \
    && cat /src/nvbandwidth_build.h \
    && g++ -std=c++17 -O2 -Wall -Wextra \
        -static-libstdc++ -static-libgcc \
        -include /src/nvbandwidth_build.h \
        -o /out/memory_bw /src/memory_bw.cc

# Fail the build rather than ship an image that silently drags in an NVIDIA
# redistributable, or a C++ runtime the distroless base does not provide.
#
# This reads DT_NEEDED with objdump instead of running ldd, because in this
# stage the driver libraries are only present as link stubs: ldd would report
# them "not found" and the interesting question — which SONAMEs did we record —
# would be buried in a resolution failure.
#
# libcuda.so.1 and libnvidia-ml.so.1 are the allowed exceptions and only those:
# they are host driver libraries injected at runtime by the NVIDIA Container
# Toolkit, never redistributed by us. Everything else matching libcud*/libnv*
# would be a redistributable, and fails the build.
RUN set -eu; \
    : > /out/needed.txt; \
    for bin in /out/nvbandwidth /out/memory_bw; do \
        echo "== $bin"; \
        objdump -p "$bin" | awk '/NEEDED/ {print $2}' | tee -a /out/needed.txt; \
    done; \
    ! grep -qE '^(libstdc\+\+|libgcc_s)' /out/needed.txt; \
    grep -iE 'libcud|libnv' /out/needed.txt | sort -u > /out/nvidia-needed.txt || true; \
    echo "== NVIDIA sonames referenced:"; cat /out/nvidia-needed.txt; \
    ! grep -qvE '^(libcuda\.so\.1|libnvidia-ml\.so\.1)$' /out/nvidia-needed.txt; \
    ! find /out -name '*.so*' -print | grep -q .

# ── final: no NVIDIA layers ───────────────────────────────────────────────────
# cc-debian13 carries a glibc at least as new as the ubuntu24.04 build stage.
# An older base (cc-debian12, glibc 2.36) fails at runtime with GLIBC_2.38 errors.
# A scratch/musl base will not work: the wrapper dlopen()s the driver and
# nvbandwidth links it, and both need glibc.
#
# Release builds should pin RUNTIME_IMAGE (declared above the first FROM) by
# digest — a floating base tag changes the contents of an immutable runner tag,
# which is the audit-trail hole the publish-runner rules exist to close.
FROM ${RUNTIME_IMAGE}
ARG NVBANDWIDTH_PATH

# The NVIDIA-provided CUDA images set these; leaving that base means we must set
# them ourselves.  Without "compute", the toolkit does not mount libcuda.so.1 and
# neither binary can see the GPU at all.  "utility" is what mounts
# libnvidia-ml.so.1, which nvbandwidth links for its NUMA/affinity handling —
# without it the container starts and then dies on a missing NVML.
ENV NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility

COPY --from=build /out/nvbandwidth ${NVBANDWIDTH_PATH}
COPY --from=build /out/memory_bw /usr/local/bin/memory_bw
USER 65532:65532
ENTRYPOINT ["/usr/local/bin/memory_bw"]
