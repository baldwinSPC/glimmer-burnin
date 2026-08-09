# Runner image for the "tcp-baseline" TestKind: plain TCP throughput between two
# nodes, so that an RDMA failure can be told apart from a node whose networking
# is broken in general.
#
# The workload is iperf3, which is BSD-3-Clause (Lawrence Berkeley National
# Laboratory).  It is on this project's permissive allow-list; see the licensing
# rules in CLAUDE.md before substituting anything here — the obvious
# alternatives in this space are not all permissive.
#
# Licensing of everything in the final image:
#   - our wrapper (*.go)        Apache-2.0
#   - iperf3                    BSD-3-Clause (built from source, shipped, with
#                               its LICENSE at /licenses)
#   - Go standard library       BSD-3-Clause (linked into the wrapper)
#   - glibc (distroless base)   LGPL-2.1-or-later, dynamically linked and
#                               unmodified, supplied by the base image
#
# NO ACCELERATOR.  This runner opens no CUDA context, executes no nvidia-smi and
# touches no vendor stack at all — which is the point of it: it runs on any node
# in any fleet, including one with no accelerator and no RDMA hardware.  It
# therefore sets NO NVIDIA_* environment, and runners/pins_test.go asserts that
# absence rather than leaving it to a reader.

ARG BUILD_IMAGE=debian:bookworm-slim
ARG GO_IMAGE=golang:1.24-bookworm
# The runtime base is the SAME Debian release as the iperf3 build stage, so the
# glibc the binary was linked against is the glibc it runs on.  Moving one
# without the other is how a runner starts failing with GLIBC_2.xx errors on
# every node at once.
ARG RUNTIME_IMAGE=gcr.io/distroless/cc-debian12

# ── iperf3: built from source, never vendored ────────────────────────────────
FROM ${BUILD_IMAGE} AS iperf-build

ARG IPERF_REPO=https://github.com/esnet/iperf.git
ARG IPERF_REF=3.19.1
# A tag is not an identity: it can be moved.  A published runner image is what a
# node's readiness gate executes and its tag is immutable, so a moved upstream
# tag would change what a fleet is measured by with no audit trail.  Bumping
# IPERF_REF means bumping IPERF_SHA with it; resolve the new value with
# `git ls-remote https://github.com/esnet/iperf.git refs/tags/<ref>`.
ARG IPERF_SHA=2674a7dea63845abd692ca20f761b2406acf82a4

RUN apt-get update && apt-get install -y --no-install-recommends \
      ca-certificates gcc git make libc6-dev \
    && rm -rf /var/lib/apt/lists/*

RUN git clone --depth 1 -b "${IPERF_REF}" "${IPERF_REPO}" /src
WORKDIR /src

RUN got="$(git rev-parse HEAD)"; \
    if [ "${got}" != "${IPERF_SHA}" ]; then \
      echo "refusing to build: ${IPERF_REF} is ${got}, expected ${IPERF_SHA}" >&2; \
      exit 1; \
    fi

# Refuse to ship a tool whose licence is not the one this project checked.  A
# permissive licence is a build-time fact here, not a line in a README that
# nobody re-reads when the dependency moves.  Both greps matter: the first
# identifies the licence family, the second identifies the licensor, and a
# relicensed fork that kept the BSD text would fail the second.
RUN grep -q "Redistribution and use in source and binary forms" LICENSE \
    && grep -qi "Lawrence Berkeley" LICENSE

# --disable-shared: the library is not shipped and nothing links it but iperf3
# itself, so building it static removes a runtime dependency the distroless base
# would otherwise have to carry.
RUN ./configure --prefix=/usr/local --disable-shared \
    && make -j"$(nproc)" \
    && make install DESTDIR=/out

# Fail the build rather than ship an image that silently drags in an NVIDIA .so
# (this runner has no business linking one) or a library the base cannot supply.
RUN ldd /out/usr/local/bin/iperf3 | tee /out/ldd.txt \
    && ! grep -qiE 'libcud(art|a)|libnv' /out/ldd.txt \
    && ! grep -q 'libiperf' /out/ldd.txt

# BSD-3-Clause requires the licence and the copyright notice to travel with the
# binary.
RUN mkdir -p /out/licenses \
    && cp LICENSE /out/licenses/iperf3-LICENSE

# ── the wrapper: exit codes, metrics, and the management-path guard ──────────
# Pure Go, so it cross-compiles for free: built on the BUILD platform, the arm64
# image costs no QEMU here (unlike the iperf3 stage above).
FROM --platform=$BUILDPLATFORM ${GO_IMAGE} AS wrapper-build
ARG TARGETOS
ARG TARGETARCH
ARG IPERF_REF=3.19.1

# CGO_ENABLED=0 leaves a binary that links nothing at all, so "no NVIDIA library,
# no copyleft library" is a structural property rather than a hope.
#
# GOPROXY=off is the load-bearing part: this wrapper imports NOTHING outside the
# standard library, and turning the proxy off makes that a build-time fact. A
# dependency added later fails the build here rather than quietly appearing in a
# published image whose licence inventory nobody re-derived.
#
# The build CONTEXT is this directory — see the runner-images job in ci.yml — so
# the module is created here rather than inherited from the repository root.
ENV CGO_ENABLED=0 GOPROXY=off GOFLAGS=-mod=mod
WORKDIR /src
COPY *.go ./
RUN rm -f ./*_test.go && go mod init glimmer-burnin-tcp-baseline
RUN mkdir -p /out && GOOS=${TARGETOS} GOARCH=${TARGETARCH} go build -trimpath \
      -ldflags="-s -w -X main.iperfVersion=${IPERF_REF}" \
      -o /out/tcp-baseline .
RUN ldd /out/tcp-baseline > /out/wrapper-ldd.txt 2>&1 || true; \
    cat /out/wrapper-ldd.txt; \
    ! grep -qiE 'libcud(art|a)|libnv' /out/wrapper-ldd.txt

# ── final image ──────────────────────────────────────────────────────────────
FROM ${RUNTIME_IMAGE}

COPY --from=iperf-build /out/usr/local/bin/iperf3 /usr/local/bin/iperf3
COPY --from=iperf-build /out/licenses/ /licenses/
COPY --from=wrapper-build /out/tcp-baseline /usr/local/bin/tcp-baseline

# The wrapper finds iperf3 by PATH; distroless has no shell, so this is stated
# rather than sourced from a profile.
ENV PATH=/usr/local/bin:/usr/bin:/bin

# The guard reads /proc/net/route and enumerates the host's interfaces, both of
# which need the HOST network namespace — which a Pair-scope fabric test wants
# anyway.  It is not requested here: hostNetwork is a pod-level decision the
# BurnInTest makes, and an image cannot grant it to itself.  Without it the
# guard fails closed and the test reports an Error, which is the intended
# degradation: it will not measure a path it cannot classify.

# Unprivileged. The guard reads /proc/net/route, which is world-readable, and
# iperf3 binds a port above 1024 — nothing here needs root, and a runner that
# asks for it anyway is a privilege nobody audited.
USER 65532:65532

ENTRYPOINT ["/usr/local/bin/tcp-baseline"]
