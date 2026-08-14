# Runner image for the "ib-write-bw" TestKind: RDMA write bandwidth and latency
# across the link between TWO nodes.  This is a Pair-scope runner; see README.md.
#
# ── Licensing of everything in the final image ───────────────────────────────
#   - our wrapper (*.go)            Apache-2.0
#   - linux-rdma/perftest           dual GPL-2.0-only OR BSD-2-Clause, and this
#                                   project consumes it UNDER THE BSD-2-Clause
#                                   OPTION ONLY.  The build asserts below that
#                                   the source it fetched still offers that
#                                   option, and copies the licence into
#                                   /licenses so the choice travels with the
#                                   binary.  Recording the option is required:
#                                   under the GPL option this image could not be
#                                   published at all.
#   - rdma-core (libibverbs,        dual GPL-2.0 OR BSD-2-Clause / OpenIB BSD,
#     librdmacm, ibverbs-providers) consumed under the BSD option; supplied
#                                   unmodified by the base distribution
#   - libnl-3 / libnl-route-3       LGPL-2.1-or-later, dynamically linked and
#                                   unmodified — the same arrangement this
#                                   project already accepts for the glibc its
#                                   runner bases carry
#   - Go standard library           BSD-3-Clause (linked into the wrapper)
#   - glibc (base image)            LGPL-2.1-or-later, dynamically linked
#
# NO NVIDIA library is present and none is wanted: this runner never touches the
# accelerator.  The build asserts that below.  The GPUDirect variant of this test
# — perftest built with --use_cuda — is a SEPARATE runner (runners/gpudirect-rdma)
# precisely so that this one stays a pure fabric test with no accelerator
# dependency, and can therefore run on a node whose GPU is busy or absent.
#
# ── Why the runtime base is debian-slim and not distroless ───────────────────
# libibverbs does not link its hardware providers; it dlopen()s them at runtime,
# guided by the .driver files in /etc/libibverbs.d and found on a path compiled
# into the library.  Hand-copying those into a distroless base makes the runner
# depend on two absolute paths that the distribution owns, and the failure mode
# when one moves is "no RDMA device found" — which this runner would report as a
# clean SKIP on hardware that is perfectly healthy, silently excusing a whole
# fleet from its fabric test.  A fabric runner must not be able to fail that way,
# so the packages that own those paths install themselves.

ARG BUILD_IMAGE=debian:trixie-slim
ARG GO_IMAGE=golang:1.26-trixie
# The runtime base is the SAME Debian release as the build stage, so the glibc
# perftest was linked against is the glibc it runs on.
ARG RUNTIME_IMAGE=debian:trixie-slim

# ── perftest: built from source, never vendored ───────────────────────────────
FROM ${BUILD_IMAGE} AS perftest-build

ARG PERFTEST_REPO=https://github.com/linux-rdma/perftest.git
# v4.4-0.37 is pinned by TWO independent constraints, and neither is inertia:
#
#   1. LICENCE.  From the v4.5 series on, perftest's configure hard-requires
#      pciutils (libpci, GPL-2.0-or-later) on every non-FreeBSD platform, with no
#      --without-pciutils.  A GPL library may not ship in an image this project
#      publishes.  v4.4-0.37 predates that requirement and references libpci
#      nowhere; assert-no-copyleft.sh below fails the build if a bump changes
#      that.
#   2. BUILDABILITY.  The v4.5 series does not compile against stock rdma-core
#      (50.0 on Ubuntu 24.04, 56.1 on Debian trixie): it needs the AES-XTS
#      crypto mkey API from mlx5dv that only MLNX_OFED/DOCA ships.
#
# Both were verified by building every v4.5 tag on both distributions.
ARG PERFTEST_REF=v4.4-0.37
# A tag is not an identity: it can be moved.  A published runner image is what a
# node's readiness gate executes, so the exact commit is asserted rather than
# trusted.  Bumping PERFTEST_REF means bumping PERFTEST_SHA with it.
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

# THE LICENCE CHOICE IS A BUILD-TIME ASSERTION, not a line in a README nobody
# re-reads when the dependency moves.  perftest is offered under GPL-2.0 OR
# BSD-2-Clause; this project may only ship the BSD option, so the build fails if
# the BSD grant is no longer present in the source it just fetched.
RUN set -e; \
    lic="$(ls COPYING LICENSE 2>/dev/null | head -1)"; \
    test -n "${lic}"; \
    grep -qi "Redistribution and use in source and binary forms" "${lic}"; \
    grep -qi "BSD" "${lic}" || grep -qi "OpenIB.org BSD license" "${lic}"; \
    echo "perftest licence file ${lic} offers a BSD option: OK"

# The copyleft guard.  A bump of PERFTEST_REF into the v4.5 series would pull in
# pciutils (GPL-2.0-or-later) and make this image unpublishable; this fails the
# build instead.  DO NOT resolve a failure here by installing pciutils-dev — read
# the script's header first.
COPY assert-no-copyleft.sh /assert-no-copyleft.sh
RUN sh /assert-no-copyleft.sh /src

RUN ./autogen.sh \
    && ./configure --prefix=/usr/local \
    && make -j"$(nproc)" \
    && make install DESTDIR=/out

# Fail the build rather than ship an image that silently drags in an NVIDIA .so.
# This runner has no business linking one; the CUDA-enabled build of the same
# tool is runners/gpudirect-rdma.
RUN ldd /out/usr/local/bin/ib_write_bw | tee /out/ldd.txt \
    && ldd /out/usr/local/bin/ib_write_lat | tee -a /out/ldd.txt \
    && ! grep -qiE 'libcud(art|a)|libnv|libnccl' /out/ldd.txt

# BSD-2-Clause requires the notice to travel with the binary.
RUN mkdir -p /out/licenses \
    && cp "$(ls COPYING LICENSE 2>/dev/null | head -1)" /out/licenses/perftest-COPYING \
    && printf '%s\n' \
        'linux-rdma/perftest is dual-licensed GPL-2.0-only OR BSD-2-Clause.' \
        'glimmer-burnin consumes and redistributes it under the BSD-2-Clause option ONLY.' \
        "Version: ${PERFTEST_REF} (${PERFTEST_SHA})" \
        > /out/licenses/perftest-LICENSE-CHOICE.txt

# ── the wrapper: rendezvous, exit codes, metrics ──────────────────────────────
# Pure Go, so it cross-compiles for free: built on the BUILD platform.
FROM --platform=$BUILDPLATFORM ${GO_IMAGE} AS wrapper-build
ARG TARGETOS
ARG TARGETARCH
ARG PERFTEST_REF=v4.4-0.37

# CGO_ENABLED=0 leaves a binary that links nothing at all, so "no NVIDIA library,
# no copyleft library" is a structural property of the wrapper rather than a hope.
#
# GOPROXY=off is an assertion, not an optimisation: this runner's non-test
# sources import only the standard library, and the build must fail loudly rather
# than reach the network if that ever stops being true.
ENV CGO_ENABLED=0 GOPROXY=off GOFLAGS=-mod=mod

WORKDIR /src
COPY *.go ./

# The wrapper's source belongs to the repository's ROOT Go module, so CI's
# `go build ./...`, `go vet ./...` and `go test ./...` cover it like any other
# package.  The publish-runner workflow builds with `context: runners/<name>`,
# which cannot see the repository's go.mod, so the image build synthesises a
# throwaway module here.  Nothing is fetched.
#
# The test files go first: they deliberately DO import pkg/contract and
# pkg/runner (that is how the emitted metric names are proven legal against the
# real registry), and those packages are outside this build context.
RUN rm -f ./*_test.go && go mod init glimmer-burnin-ib-write-bw

RUN mkdir -p /out && GOOS=${TARGETOS} GOARCH=${TARGETARCH} go build -trimpath \
      -ldflags="-s -w -X main.perftestRef=${PERFTEST_REF}" \
      -o /out/burnin-ib-write-bw .

RUN ldd /out/burnin-ib-write-bw > /out/wrapper-ldd.txt 2>&1 || true; \
    cat /out/wrapper-ldd.txt; \
    ! grep -qiE 'libcud(art|a)|libnv' /out/wrapper-ldd.txt

# ── final: minimal, non-root, no toolchain ────────────────────────────────────
FROM ${RUNTIME_IMAGE}

# ibverbs-providers is the package that owns the hardware plugins libibverbs
# dlopen()s.  Without it every device disappears and this runner reports a clean
# SKIP on a healthy fabric, which is the worst outcome available to it.
RUN apt-get update && apt-get install -y --no-install-recommends \
      libibverbs1 librdmacm1 ibverbs-providers \
    && rm -rf /var/lib/apt/lists/* /usr/share/doc /usr/share/man

COPY --from=perftest-build /out/usr/local/bin/ib_write_bw  /usr/local/bin/ib_write_bw
COPY --from=perftest-build /out/usr/local/bin/ib_write_lat /usr/local/bin/ib_write_lat
COPY --from=perftest-build /out/licenses/ /licenses/
COPY --from=wrapper-build  /out/burnin-ib-write-bw /usr/local/bin/burnin-ib-write-bw

# Non-root.  RDMA verbs need no capability beyond access to /dev/infiniband and a
# memlock limit; they do NOT need CAP_SYS_ADMIN or privileged mode.  See the
# README for the pod spec this expects (hostNetwork, an RDMA device, and a
# memlock ulimit), and note that a privileged fabric pod on every node is a far
# larger grant than this test needs.
USER 65532:65532
ENTRYPOINT ["/usr/local/bin/burnin-ib-write-bw"]
