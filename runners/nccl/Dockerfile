# Runner image for the "nccl" TestKind: a two-rank NCCL all-reduce across the
# link between TWO nodes.  Pair scope; see README.md.
#
# ── Licensing of everything in the final image ───────────────────────────────
#   - our wrapper (*.go) and harness (nccl_pair.cu)   Apache-2.0
#   - NVIDIA NCCL                                     BSD-3-Clause
#   - CUDA runtime (libcudart, statically linked)     NVIDIA CUDA EULA, which
#                                                     permits redistribution of
#                                                     the runtime linked into an
#                                                     application; the same
#                                                     arrangement runners/
#                                                     compute-smoke already ships
#                                                     via nvcc's -cudart static
#   - Go standard library                             BSD-3-Clause
#   - rdma-core (libibverbs, librdmacm, providers)    dual GPL-2.0 OR BSD, taken
#                                                     under the BSD option;
#                                                     needed because NCCL
#                                                     dlopen()s libibverbs for
#                                                     its IB/RoCE transport
#   - glibc, libnl                                    LGPL-2.1-or-later, dynamic
#
# NCCL'S LICENCE IS THE THING WORTH CHECKING, AND IT IS GOOD NEWS.  NCCL is often
# assumed to be an NVIDIA-licensed redistributable of the kind this project may
# not ship.  It is not: NVIDIA/nccl is BSD-3-Clause, the same permissive licence
# as CUTLASS, and the build asserts that below by reading the LICENSE.txt of the
# exact commit it fetched rather than trusting this comment.
#
# ── Why NCCL is BUILT FROM SOURCE and linked STATICALLY ──────────────────────
# The prebuilt libnccl from NVIDIA's apt repository is 411 MB, because it embeds
# device code for every architecture NVIDIA ships.  A runner image is executed by
# a node's readiness gate, so its pull cost is paid by every node in the fleet on
# every gate; carrying 400 MB of device code for parts that are not in the fleet
# is not a rounding error.  Building from source with ONE gencode produces a
# fraction of that, and linking it statically leaves no NVIDIA .so in the image
# at all — so the "no NVIDIA library ships here" guard at the end can be an
# unconditional assertion instead of an exception list.

ARG CUDA_IMAGE=nvcr.io/nvidia/cuda:12.9.1-devel-ubuntu24.04
ARG GO_IMAGE=golang:1.26-bookworm
ARG RUNTIME_IMAGE=ubuntu:24.04

# ── NCCL, from source ─────────────────────────────────────────────────────────
FROM ${CUDA_IMAGE} AS nccl-build

# BuildKit supplies this; it must still be declared to be visible in the stage.
ARG TARGETARCH

ARG NCCL_REPO=https://github.com/NVIDIA/nccl.git
ARG NCCL_REF=v2.27.7-1
# A tag is not an identity: it can be moved.  Bumping NCCL_REF means bumping
# NCCL_SHA with it.
ARG NCCL_SHA=593de54e52679b51428571c13271e2ea9f91b1b1

# THE ARCH TARGET IS DELIBERATE, AND IT IS A DIFFERENT AXIS FROM TARGETARCH.
#
# TARGETARCH is the HOST (linux/amd64 vs linux/arm64).  NCCL_GENCODE is the GPU.
# A single gencode is deliberate: each additional architecture multiplies NCCL's
# device-code build time and the size of every node's image pull, and the
# prebuilt libnccl that carries them all is 411 MB.  So the default is ONE part
# per host architecture, chosen as the part that architecture is actually
# deployed with:
#
#   arm64 → compute_121/sm_121   GB10 / DGX Spark, the hardware this runner was
#                                verified on
#   amd64 → compute_90/sm_90     H100 / H200, the dominant x86 multi-node NCCL
#                                part.  Also covers CC 9.x by minor-version
#                                binary compatibility.
#
# THIS IS THE RUNNER WHERE "USABLE ON x86" MEANS "USABLE WITH A GENCODE THAT
# MATCHES YOUR GPU".  An x86 fleet on B200 (sm_100), L40S (sm_89) or A100
# (sm_80) must REBUILD with its own value — there is no PTX fallback in a single
# `code=sm_XX` target, so NCCL will simply have no kernel for the part.  That is
# a one-line --build-arg:
#
#   docker build --build-arg NCCL_GENCODE=-gencode=arch=compute_100,code=sm_100 .
#
# Guessing "everything" on a fleet's behalf is the alternative, and it costs
# every node in every fleet a 400 MB pull to avoid one build argument.
#
# Empty means "this platform's default"; any other value is used verbatim on
# every platform.  Set the harness's CUDA_ARCH to match when you override this.
ARG NCCL_GENCODE=

RUN apt-get update && apt-get install -y --no-install-recommends \
      ca-certificates git make g++ python3 \
    && rm -rf /var/lib/apt/lists/*

RUN git clone --depth 1 -b "${NCCL_REF}" "${NCCL_REPO}" /nccl
WORKDIR /nccl

RUN got="$(git rev-parse HEAD)"; \
    if [ "${got}" != "${NCCL_SHA}" ]; then \
      echo "refusing to build: ${NCCL_REF} is ${got}, expected ${NCCL_SHA}" >&2; \
      if [ "$(cd /nccl && git cat-file -t "${NCCL_SHA}" 2>/dev/null)" = "tag" ]; then \
        echo "  ${NCCL_SHA} is the ANNOTATED TAG OBJECT, not the commit it points to." >&2; \
        echo "  Use the peeled value: $(cd /nccl && git rev-parse "${NCCL_SHA}^{commit}")" >&2; \
        echo "  Resolve it with: git ls-remote <repo> 'refs/tags/<ref>^{}'" >&2; \
      fi; \
      exit 1; \
    fi

# The licence is a build-time fact, not a line in a README nobody re-reads when
# the dependency moves.  If NCCL ever stops being BSD-3-Clause, this build fails
# rather than quietly shipping something this project may not publish.
RUN set -e; \
    test -f LICENSE.txt; \
    grep -q "Redistribution and use in source and binary forms" LICENSE.txt; \
    grep -q "Neither the name of NVIDIA CORPORATION" LICENSE.txt; \
    grep -qi "THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS" LICENSE.txt; \
    echo "NCCL ${NCCL_REF} is BSD-3-Clause: OK"

# The per-platform default is resolved here rather than in the ARG, because an
# ARG default cannot branch on TARGETARCH.
RUN set -eu; \
    gencode="${NCCL_GENCODE:-}"; \
    if [ -z "${gencode}" ]; then \
      case "${TARGETARCH}" in \
        arm64) gencode="-gencode=arch=compute_121,code=sm_121" ;; \
        amd64) gencode="-gencode=arch=compute_90,code=sm_90" ;; \
        *) echo "nccl: no default gencode for TARGETARCH='${TARGETARCH}'." >&2; \
           echo "  Pass --build-arg NCCL_GENCODE=-gencode=arch=compute_XX,code=sm_XX." >&2; \
           exit 1 ;; \
      esac; \
    fi; \
    echo "nccl: TARGETARCH=${TARGETARCH} -> NVCC_GENCODE=${gencode}"; \
    make -j"$(nproc)" src.build NVCC_GENCODE="${gencode}" \
    && ls -l /nccl/build/lib/

# ── the harness: nccl_pair ────────────────────────────────────────────────────
FROM ${CUDA_IMAGE} AS harness-build

ARG TARGETARCH
ARG NCCL_REF=v2.27.7-1
# sm_121a would emit a cubin for GB10 with no PTX fallback.  This harness is NOT
# the place for that: it launches no kernel of its own — every kernel it runs is
# NCCL's — so an "a" target here would constrain the wrapper without proving
# anything about the instruction path.  compute-smoke is where that argument
# belongs.
#
# The default MIRRORS NCCL_GENCODE above (arm64 → sm_121, amd64 → sm_90), because
# a harness compiled for a part NCCL has no kernels for is an image that cannot
# run anywhere.  Override BOTH together or neither.
ARG CUDA_ARCH=

RUN apt-get update && apt-get install -y --no-install-recommends g++ \
    && rm -rf /var/lib/apt/lists/*

COPY --from=nccl-build /nccl/build/include/ /opt/nccl/include/
COPY --from=nccl-build /nccl/build/lib/     /opt/nccl/lib/
COPY --from=nccl-build /nccl/LICENSE.txt    /opt/nccl/LICENSE.txt

COPY nccl_pair.cu /src/nccl_pair.cu

# -cudart static and -lnccl_static: everything NVIDIA travels INSIDE the binary,
# so the final image contains no NVIDIA shared object at all.  libcuda.so.1 (the
# driver) is the sole exception and is injected at runtime by the NVIDIA
# Container Toolkit — never baked in.
#
# -static-libstdc++/-static-libgcc are host-compiler flags and MUST go through
# -Xcompiler; passed bare, nvcc fails outright.
RUN set -eu; \
    arch="${CUDA_ARCH:-}"; \
    if [ -z "${arch}" ]; then \
      case "${TARGETARCH}" in \
        arm64) arch=sm_121 ;; \
        amd64) arch=sm_90 ;; \
        *) echo "nccl: no default CUDA_ARCH for TARGETARCH='${TARGETARCH}'." >&2; exit 1 ;; \
      esac; \
    fi; \
    echo "nccl harness: TARGETARCH=${TARGETARCH} -> CUDA_ARCH=${arch}"; \
    mkdir -p /out && /usr/local/cuda/bin/nvcc \
      -std=c++17 -O3 -arch="${arch}" \
      -I /opt/nccl/include \
      -cudart static \
      -Xcompiler -static-libstdc++ -Xcompiler -static-libgcc \
      -o /out/nccl_pair /src/nccl_pair.cu \
      /opt/nccl/lib/libnccl_static.a \
      -lrt -ldl -lpthread \
    && ls -l /out/nccl_pair

# The binary must actually contain NCCL and must NOT depend on an NVIDIA shared
# object other than the driver.  ldd is the check; libcuda.so.1 is expected.
RUN ldd /out/nccl_pair | tee /out/ldd.txt; \
    ! grep -qi 'libnccl' /out/ldd.txt; \
    ! grep -qi 'libcudart' /out/ldd.txt; \
    ! grep -qi 'libstdc[+][+]' /out/ldd.txt

RUN mkdir -p /out/licenses && cp /opt/nccl/LICENSE.txt /out/licenses/nccl-LICENSE.txt \
    && printf 'NVIDIA NCCL %s, BSD-3-Clause, built from source and statically linked.\n' "${NCCL_REF}" \
       > /out/licenses/nccl-VERSION.txt

# ── the wrapper ───────────────────────────────────────────────────────────────
FROM --platform=$BUILDPLATFORM ${GO_IMAGE} AS wrapper-build
ARG TARGETOS
ARG TARGETARCH
ARG NCCL_REF=v2.27.7-1

ENV CGO_ENABLED=0 GOPROXY=off GOFLAGS=-mod=mod

WORKDIR /src
COPY *.go ./

# The test files import pkg/contract and pkg/runner, which are outside this build
# context; they are stripped here and run by CI from the repository root module.
RUN rm -f ./*_test.go && go mod init glimmer-burnin-nccl

RUN mkdir -p /out && GOOS=${TARGETOS} GOARCH=${TARGETARCH} go build -trimpath \
      -ldflags="-s -w -X main.ncclVersion=${NCCL_REF}" \
      -o /out/burnin-nccl .

RUN ldd /out/burnin-nccl > /out/wrapper-ldd.txt 2>&1 || true; \
    cat /out/wrapper-ldd.txt; \
    ! grep -qiE 'libcud(art|a)|libnv|libnccl' /out/wrapper-ldd.txt

# ── final: no CUDA layers, non-root ───────────────────────────────────────────
FROM ${RUNTIME_IMAGE}

# NCCL dlopen()s libibverbs for its IB/RoCE transport. Without ibverbs-providers
# the devices are invisible to it and NCCL silently falls back to a TCP socket
# transport — a result that is real, reproducible, and about the wrong path.
RUN apt-get update && apt-get install -y --no-install-recommends \
      libibverbs1 librdmacm1 ibverbs-providers \
    && rm -rf /var/lib/apt/lists/* /usr/share/doc /usr/share/man

COPY --from=harness-build /out/nccl_pair  /usr/local/bin/nccl_pair
COPY --from=harness-build /out/licenses/  /licenses/
COPY --from=wrapper-build /out/burnin-nccl /usr/local/bin/burnin-nccl

# THE SHIPPING GUARD: no NVIDIA shared object may be in the image. libcuda.so.1
# arrives from the host driver at runtime and must not be baked in; everything
# else NVIDIA is statically linked into nccl_pair.
RUN set -e; \
    found="$(find / -xdev \( -name 'libcuda*.so*' -o -name 'libnv*.so*' -o -name 'libcudart*' -o -name 'libnccl*' \) 2>/dev/null || true)"; \
    if [ -n "${found}" ]; then \
      echo "refusing to ship an NVIDIA shared library in a runner image:" >&2; echo "${found}" >&2; exit 1; \
    fi; \
    echo "no NVIDIA shared library in the image: OK"

# Without "compute", the NVIDIA Container Toolkit does not mount libcuda.so.1 and
# the harness cannot see the accelerator at all.
ENV NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility

USER 65532:65532
ENTRYPOINT ["/usr/local/bin/burnin-nccl"]
