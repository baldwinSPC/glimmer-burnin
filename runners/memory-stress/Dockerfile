# Runner image for the "memory-stress" TestKind: a HOST memory stress test that
# proves the DIMMs return what was written to them.  On a unified-memory part
# (GB10 / DGX Spark, where CPU and GPU share one LPDDR5x pool) this is the same
# physical memory the accelerator computes from, so it doubles as the
# unified-memory test — see README.md.
#
# The workload is stressapptest, which is Apache-2.0.  The obvious alternatives
# in this space (stress-ng, fio, memtester) are GPL and cannot ship in a runner
# image published by this project; see the licensing rules in CLAUDE.md before
# substituting a tool here.
#
# Licensing of everything in the final image:
#   - our wrapper (*.go)        Apache-2.0
#   - stressapptest             Apache-2.0  (built from source, shipped, with
#                               its COPYING and NOTICE at /licenses)
#   - Go standard library       BSD-3-Clause (linked into the wrapper)
#   - libstdc++ / libgcc        GPL-3.0 WITH GCC-exception — statically linked
#                               into stressapptest, which the GCC Runtime
#                               Library Exception exists to permit; the same
#                               arrangement the compute-smoke runner ships
#   - glibc (distroless base)   LGPL-2.1-or-later, dynamically linked and
#                               unmodified, supplied by the base image
# No NVIDIA redistributable library is present, and the build asserts it below.
# This runner never touches the accelerator, so unlike compute-smoke it sets no
# NVIDIA_* environment and needs no GPU.

ARG SAT_BUILD_IMAGE=debian:bookworm-slim
ARG GO_IMAGE=golang:1.24-bookworm
# The runtime base is the SAME Debian release as the stressapptest build stage,
# so the glibc the binary was linked against is the glibc it runs on.  Moving
# one without the other is how a runner starts failing with GLIBC_2.xx errors on
# every node at once.
ARG RUNTIME_IMAGE=gcr.io/distroless/cc-debian12

# ── stressapptest: built from source, never vendored ──────────────────────────
FROM ${SAT_BUILD_IMAGE} AS sat-build

ARG STRESSAPPTEST_REPO=https://github.com/stressapptest/stressapptest.git
ARG STRESSAPPTEST_REF=v1.0.11
# v1.0.11 is the floor for this project: it is the first release with aarch64
# vector-instruction support, and GB10 is arm64.
ARG STRESSAPPTEST_SHA=25e31a2c1fcc66f3a43e91998f0819b016322a70

RUN apt-get update && apt-get install -y --no-install-recommends \
      ca-certificates g++ git make libc6-dev \
    && rm -rf /var/lib/apt/lists/*

RUN git clone --depth 1 -b "${STRESSAPPTEST_REF}" "${STRESSAPPTEST_REPO}" /src
WORKDIR /src

# A tag is not an identity: it can be moved.  A published runner image is what a
# node's readiness gate executes, so the exact commit is asserted rather than
# trusted.  Bumping STRESSAPPTEST_REF means bumping STRESSAPPTEST_SHA with it.
RUN got="$(git rev-parse HEAD)"; \
    if [ "${got}" != "${STRESSAPPTEST_SHA}" ]; then \
      echo "refusing to build: ${STRESSAPPTEST_REF} is ${got}, expected ${STRESSAPPTEST_SHA}" >&2; \
      if [ "$(git cat-file -t "${STRESSAPPTEST_SHA}" 2>/dev/null)" = "tag" ]; then \
        echo "  ${STRESSAPPTEST_SHA} is the ANNOTATED TAG OBJECT, not the commit it points to." >&2; \
        echo "  Use the peeled value: $(git rev-parse "${STRESSAPPTEST_SHA}^{commit}")" >&2; \
        echo "  Resolve it with: git ls-remote <repo> 'refs/tags/<ref>^{}'" >&2; \
      fi; \
      exit 1; \
    fi

# Refuse to ship a tool whose licence is not the one this project checked. A
# permissive licence is a build-time fact here, not a line in a README that
# nobody re-reads when the dependency moves.
RUN grep -q "Apache License" COPYING && grep -q "Version 2.0" COPYING

# -static-libstdc++/-static-libgcc: the distroless runtime base carries glibc and
# little else, so the C++ runtime has to travel inside the binary.  The GCC
# Runtime Library Exception is what makes linking it into a non-GPL image
# permissible, and compute-smoke already ships the same arrangement.
RUN ./configure --prefix=/usr/local LDFLAGS="-static-libstdc++ -static-libgcc" \
    && make -j"$(nproc)" \
    && make install DESTDIR=/out

# Fail the build rather than ship an image that silently drags in an NVIDIA .so
# (this runner has no business linking one) or a libstdc++ the base cannot
# provide (which would fail on the node, per test, as an infrastructure Error).
RUN ldd /out/usr/local/bin/stressapptest | tee /out/ldd.txt \
    && ! grep -qiE 'libcud(art|a)|libnv' /out/ldd.txt \
    && ! grep -q 'libstdc[+][+]' /out/ldd.txt

# Apache-2.0 §4 requires the licence and the notices to travel with the binary.
RUN mkdir -p /out/licenses \
    && cp COPYING /out/licenses/stressapptest-COPYING \
    && cp NOTICE /out/licenses/stressapptest-NOTICE

# ── the wrapper: exit codes, metrics, and the Error/Fail distinction ──────────
# Pure Go, so it cross-compiles for free: built on the BUILD platform, the arm64
# image costs no QEMU here (unlike the stressapptest stage above).
FROM --platform=$BUILDPLATFORM ${GO_IMAGE} AS wrapper-build
ARG TARGETOS
ARG TARGETARCH
ARG STRESSAPPTEST_REF=v1.0.11

# CGO_ENABLED=0 leaves a binary that links nothing at all, so "no NVIDIA library,
# no copyleft library" is a structural property rather than a hope.
#
# GOPROXY=off is an assertion, not an optimisation: this runner's non-test
# sources import only the standard library, and the build must fail loudly
# rather than reach the network if that ever stops being true.
ENV CGO_ENABLED=0 GOPROXY=off GOFLAGS=-mod=mod

WORKDIR /src
COPY *.go ./

# The wrapper's source belongs to the repository's ROOT Go module, so CI's
# `go build ./...`, `go vet ./...` and `go test ./...` cover it like any other
# package.  The publish-runner workflow, however, builds with
# `context: runners/<name>`, which cannot see the repository's go.mod — so the
# image build synthesises a throwaway module here.  Nothing is fetched.
#
# The test files go first: they deliberately DO import pkg/contract and
# pkg/runner (that is how the emitted metric names are proven legal against the
# real registry), and those packages are outside this build context.
RUN rm -f ./*_test.go && go mod init glimmer-burnin-memory-stress

RUN mkdir -p /out && GOOS=${TARGETOS} GOARCH=${TARGETARCH} go build -trimpath \
      -ldflags="-s -w -X main.stressapptestRef=${STRESSAPPTEST_REF}" \
      -o /out/burnin-memory-stress .

# The same assertion as above.  ldd exits non-zero on a static binary, which is
# the expected outcome here — the grep is what matters.
RUN ldd /out/burnin-memory-stress > /out/wrapper-ldd.txt 2>&1 || true; \
    cat /out/wrapper-ldd.txt; \
    ! grep -qiE 'libcud(art|a)|libnv' /out/wrapper-ldd.txt

# ── final: minimal, non-root, no toolchain ────────────────────────────────────
FROM ${RUNTIME_IMAGE}

COPY --from=sat-build /out/usr/local/bin/stressapptest /usr/local/bin/stressapptest
COPY --from=sat-build /out/licenses/ /licenses/
COPY --from=wrapper-build /out/burnin-memory-stress /usr/local/bin/burnin-memory-stress

# Non-root, and it stays that way: stressapptest needs no capability to allocate
# and check anonymous memory.  Running privileged would only buy /proc/self/
# pagemap (physical addresses for a miscompare), which is not worth handing a
# burn-in pod CAP_SYS_ADMIN on every node in a fleet.
USER 65532:65532
ENTRYPOINT ["/usr/local/bin/burnin-memory-stress"]
