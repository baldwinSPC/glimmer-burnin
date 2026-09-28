# cpu-bench (#542): host CPU FMA throughput and STREAM memory bandwidth.
#
# One C++17 binary over plain threads, built static against libstdc++/libgcc
# and run on distroless. No OpenMP, deliberately: libgomp is GPL with the GCC
# Runtime Library Exception, and plain threads keep the licence question from
# arising at all. No accelerator is touched, so no NVIDIA_* injection is
# declared (runners/pins_test.go records the exemption).
#
# The ISA target is per platform and deliberate, like every runner's arch:
#   amd64  x86-64-v3 (AVX2 + FMA). The binary checks the CPU at start and
#          reports an Error rather than dying on an illegal instruction.
#   arm64  armv8.2-a. FMA is baseline in AArch64, and GB10's Cortex-X925 and
#          Cortex-A725 both implement it; nothing here needs SVE.
# Never -march=native: that would describe the BUILD machine, not the fleet.
ARG BUILD_IMAGE=debian:trixie-slim
ARG RUNTIME_IMAGE=gcr.io/distroless/cc-debian13

FROM ${BUILD_IMAGE} AS build
ARG TARGETARCH

RUN apt-get update && apt-get install -y --no-install-recommends g++ binutils \
    && rm -rf /var/lib/apt/lists/*

COPY cpu_bench.cc cpu_topology.h /src/

# -ffp-contract=fast makes each acc*a+b one fused instruction, which is what
# the FMA figure claims to measure.
RUN set -eu; \
    arch="${TARGETARCH:-$(dpkg --print-architecture)}"; \
    case "${arch}" in \
      amd64) march="-march=x86-64-v3" ;; \
      arm64) march="-march=armv8.2-a" ;; \
      *) echo "cpu-bench: no ISA target defined for ${arch}" >&2; exit 1 ;; \
    esac; \
    mkdir -p /out; \
    g++ -std=c++17 -O3 -ffp-contract=fast ${march} -pthread -Wall -Wextra \
        -static-libstdc++ -static-libgcc \
        -o /out/cpu_bench /src/cpu_bench.cc

# The distroless base carries glibc and nothing else: refuse a binary that
# still needs the C++ runtime, or that links anything NVIDIA's.
RUN set -eu; \
    objdump -p /out/cpu_bench | awk '/NEEDED/ {print $2}' | tee /out/needed.txt; \
    ! grep -qE '^(libstdc\+\+|libgcc_s)' /out/needed.txt; \
    ! grep -qiE 'libcud|libnv' /out/needed.txt

FROM ${RUNTIME_IMAGE}
COPY --from=build /out/cpu_bench /usr/local/bin/cpu_bench
USER 65532:65532
ENTRYPOINT ["/usr/local/bin/cpu_bench"]
