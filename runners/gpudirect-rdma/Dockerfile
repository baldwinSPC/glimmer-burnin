# Runner image for the "gpudirect-rdma" TestKind: RDMA write bandwidth measured
# straight into ACCELERATOR memory across the link between TWO nodes.  Pair
# scope; see README.md.
#
# This is runners/ib-write-bw's tool built with --enable-cuda.  The two are
# separate images on purpose: ib-write-bw must keep working on a node with no
# accelerator and no CUDA stack at all, and folding the CUDA build into it would
# make every fabric test depend on the GPU being idle and the driver being sane.
#
# ── Licensing of everything in the final image ───────────────────────────────
#   - our wrapper (*.go)            Apache-2.0
#   - linux-rdma/perftest           dual GPL-2.0-only OR BSD-2-Clause, consumed
#                                   and redistributed UNDER THE BSD-2-Clause
#                                   OPTION ONLY (asserted at build time below)
#   - rdma-core (libibverbs,        dual GPL-2.0 OR BSD-2-Clause / OpenIB BSD,
#     librdmacm, ibverbs-providers) consumed under the BSD option; supplied
#                                   unmodified by the base distribution
#   - libnl-3 / libnl-route-3       LGPL-2.1-or-later, dynamically linked
#   - Go standard library           BSD-3-Clause (linked into the wrapper)
#   - glibc (base image)            LGPL-2.1-or-later, dynamically linked
#
# ── The NVIDIA rule, and the one distinction that matters here ───────────────
# The rule is that NO NVIDIA-LICENSED REDISTRIBUTABLE LIBRARY ships in a runner
# image.  libcuda.so.1 is NOT one: it is the DRIVER, it belongs to the host, and
# the NVIDIA Container Toolkit injects it at runtime — which is exactly the
# arrangement runners/compute-smoke already relies on.  So this image may LINK
# against libcuda.so.1 while CONTAINING no NVIDIA library at all.
#
# That distinction is why the guard below is a `find` over the finished
# filesystem and not only an `ldd` over the binary: ldd cannot tell "links a
# library the host will provide" from "ships one", and for this runner the answer
# differs between those two questions.  libcudart (the CUDA runtime, a
# redistributable) is separately forbidden as a link dependency, because nothing
# injects it and shipping it is the mistake the rule exists to prevent.

# The CUDA toolchain is BUILD-TIME ONLY; none of its layers are shipped.
ARG CUDA_IMAGE=nvcr.io/nvidia/cuda:12.9.1-devel-ubuntu24.04
ARG GO_IMAGE=golang:1.26-bookworm
# The runtime base is the same distribution release as the CUDA build image
# (Ubuntu 24.04), so the glibc perftest was linked against is the one it runs on.
ARG RUNTIME_IMAGE=ubuntu:24.04

# ── perftest, built with CUDA ─────────────────────────────────────────────────
FROM ${CUDA_IMAGE} AS perftest-build

ARG PERFTEST_REPO=https://github.com/linux-rdma/perftest.git
# Pinned for the SAME two reasons as runners/ib-write-bw — read that Dockerfile's
# comment before bumping.  In short: the v4.5 series hard-requires pciutils
# (libpci, GPL-2.0-or-later) with no way to turn it off, and does not compile
# against stock rdma-core (50.0 here on Ubuntu 24.04).
ARG PERFTEST_REF=v4.4-0.37
ARG PERFTEST_SHA=5fb4f10a7e7827ed15e53c25810a10be279d6e23

RUN apt-get update && apt-get install -y --no-install-recommends \
      ca-certificates git \
      autoconf automake libtool make gcc libc6-dev \
      libibverbs-dev librdmacm-dev libibumad-dev \
    && rm -rf /var/lib/apt/lists/*

RUN git clone --depth 1 -b "${PERFTEST_REF}" "${PERFTEST_REPO}" /src
WORKDIR /src

RUN got="$(git rev-parse HEAD)"; \
    if [ "${got}" != "${PERFTEST_SHA}" ]; then \
      echo "refusing to build: ${PERFTEST_REF} is ${got}, expected ${PERFTEST_SHA}" >&2; \
      if [ "$(cd /src && git cat-file -t "${PERFTEST_SHA}" 2>/dev/null)" = "tag" ]; then \
        echo "  ${PERFTEST_SHA} is the ANNOTATED TAG OBJECT, not the commit it points to." >&2; \
        echo "  Use the peeled value: $(cd /src && git rev-parse "${PERFTEST_SHA}^{commit}")" >&2; \
        echo "  Resolve it with: git ls-remote <repo> 'refs/tags/<ref>^{}'" >&2; \
      fi; \
      exit 1; \
    fi

# The licence choice, asserted rather than documented.
RUN set -e; \
    lic="$(ls COPYING LICENSE 2>/dev/null | head -1)"; \
    test -n "${lic}"; \
    grep -qi "Redistribution and use in source and binary forms" "${lic}"; \
    grep -qi "BSD" "${lic}" || grep -qi "OpenIB.org BSD license" "${lic}"; \
    echo "perftest licence file ${lic} offers a BSD option: OK"

# The copyleft guard.  DO NOT resolve a failure here by installing pciutils-dev;
# read the script's header first.
COPY assert-no-copyleft.sh /assert-no-copyleft.sh
RUN sh /assert-no-copyleft.sh /src

# CUDA is enabled by SETTING CUDA_H_PATH, not by a --enable-cuda flag: in
# v4.4-0.37 configure.ac declares it with AC_ARG_VAR and keys HAVE_CUDA off
# whether the variable is non-empty.  Passing --enable-cuda is silently accepted
# and does nothing ("configure: WARNING: unrecognized options"), which is exactly
# the sort of quiet no-op the assertion below exists to catch.
#
# LDFLAGS points at the CUDA STUB library.  Linking the stub records a
# DT_NEEDED on libcuda.so.1 without baking anything in; at runtime that entry is
# satisfied by the real driver library the NVIDIA Container Toolkit injects.
RUN ./autogen.sh \
    && ./configure --prefix=/usr/local \
         CUDA_H_PATH=/usr/local/cuda/include/cuda.h \
         LDFLAGS=-L/usr/local/cuda/lib64/stubs \
    && make -j"$(nproc)" \
    && make install DESTDIR=/out

# The binary must actually have the CUDA path compiled in.  A perftest that
# silently configured without CUDA would run, measure HOST memory, and report it
# under a metric name claiming device memory — a wrong answer that looks right.
#
# This is asserted STATICALLY rather than by running `ib_write_bw --help`,
# because the binary cannot start in this build stage: it now needs
# libcuda.so.1, which only exists on a node with a driver.  A check that cannot
# run is not a check.
RUN set -e; \
    grep -q '#define HAVE_CUDA 1' config.h \
      || { echo "refusing to ship: configure did not enable CUDA (HAVE_CUDA absent from config.h)" >&2; exit 1; }; \
    grep -q -- '--use_cuda' /out/usr/local/bin/ib_write_bw \
      || { echo "refusing to ship: this ib_write_bw has no --use_cuda, so it cannot measure device memory" >&2; exit 1; }; \
    echo "ib_write_bw was built with CUDA support: OK"

# libcuda.so.1 (the driver) is EXPECTED here and is injected at runtime.
# libcudart (a redistributable) is not, and must never be linked.
RUN ldd /out/usr/local/bin/ib_write_bw | tee /out/ldd.txt \
    && ldd /out/usr/local/bin/ib_write_lat | tee -a /out/ldd.txt \
    && ! grep -qi 'libcudart' /out/ldd.txt \
    && ! grep -qi 'libnccl' /out/ldd.txt \
    && grep -qi 'libcuda\.so' /out/ldd.txt

RUN mkdir -p /out/licenses \
    && cp "$(ls COPYING LICENSE 2>/dev/null | head -1)" /out/licenses/perftest-COPYING \
    && printf '%s\n' \
        'linux-rdma/perftest is dual-licensed GPL-2.0-only OR BSD-2-Clause.' \
        'glimmer-burnin consumes and redistributes it under the BSD-2-Clause option ONLY.' \
        "Version: ${PERFTEST_REF} (${PERFTEST_SHA}), built with --enable-cuda" \
        > /out/licenses/perftest-LICENSE-CHOICE.txt

# ── the wrapper ───────────────────────────────────────────────────────────────
FROM --platform=$BUILDPLATFORM ${GO_IMAGE} AS wrapper-build
ARG TARGETOS
ARG TARGETARCH
ARG PERFTEST_REF=v4.4-0.37

ENV CGO_ENABLED=0 GOPROXY=off GOFLAGS=-mod=mod

WORKDIR /src
COPY *.go ./

# The test files import pkg/contract and pkg/runner, which are outside this build
# context; they are stripped here and run by CI from the repository root module.
RUN rm -f ./*_test.go && go mod init glimmer-burnin-gpudirect-rdma

RUN mkdir -p /out && GOOS=${TARGETOS} GOARCH=${TARGETARCH} go build -trimpath \
      -ldflags="-s -w -X main.perftestRef=${PERFTEST_REF}" \
      -o /out/burnin-gpudirect-rdma .

RUN ldd /out/burnin-gpudirect-rdma > /out/wrapper-ldd.txt 2>&1 || true; \
    cat /out/wrapper-ldd.txt; \
    ! grep -qiE 'libcud(art|a)|libnv' /out/wrapper-ldd.txt

# ── final: no CUDA layers, non-root ───────────────────────────────────────────
FROM ${RUNTIME_IMAGE}

RUN apt-get update && apt-get install -y --no-install-recommends \
      libibverbs1 librdmacm1 ibverbs-providers \
    && rm -rf /var/lib/apt/lists/* /usr/share/doc /usr/share/man

COPY --from=perftest-build /out/usr/local/bin/ib_write_bw  /usr/local/bin/ib_write_bw
COPY --from=perftest-build /out/usr/local/bin/ib_write_lat /usr/local/bin/ib_write_lat
COPY --from=perftest-build /out/licenses/ /licenses/
COPY --from=wrapper-build  /out/burnin-gpudirect-rdma /usr/local/bin/burnin-gpudirect-rdma

# THE SHIPPING GUARD, as opposed to the linking one above: assert that the
# finished filesystem contains no NVIDIA library.  libcuda.so.1 must arrive from
# the host driver at runtime and must not be baked in.
RUN set -e; \
    found="$(find / -xdev \( -name 'libcuda*.so*' -o -name 'libnv*.so*' -o -name 'libcudart*' -o -name 'libnccl*' \) 2>/dev/null || true)"; \
    if [ -n "${found}" ]; then \
      echo "refusing to ship an NVIDIA library in a runner image:" >&2; echo "${found}" >&2; exit 1; \
    fi; \
    echo "no NVIDIA library in the image: OK"

# Without "compute", the NVIDIA Container Toolkit does not mount libcuda.so.1 and
# ib_write_bw --use_cuda cannot see the accelerator at all.
ENV NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility

USER 65532:65532
ENTRYPOINT ["/usr/local/bin/burnin-gpudirect-rdma"]
