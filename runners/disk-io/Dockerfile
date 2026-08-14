# Runner image for the "disk-io" TestKind: sequential throughput and operation
# latency against a site-declared path, with the page cache bypassed.
#
# NO THIRD-PARTY COMPONENT AT ALL, and that is a LICENSING decision rather than
# a minimalist one.  The obvious tools in this space — fio, IOR, elbencho — are
# all GPL, and a copyleft dependency would make this project unpublishable.  So
# the measurement is a standard-library Go program, and this runner's own
# metricnames_test.go asserts the non-test sources import nothing else.
#
# Licensing of everything in the final image:
#   - our runner (*.go)         Apache-2.0
#   - Go standard library       BSD-3-Clause (linked in)
# The distroless STATIC base carries no libc at all, because CGO_ENABLED=0
# leaves a binary that links nothing.
#
# NO ACCELERATOR.  This runner opens no CUDA context and touches no vendor
# stack, so it sets no NVIDIA_* environment; runners/pins_test.go asserts that
# absence rather than leaving it to a reader.

ARG GO_IMAGE=golang:1.26-bookworm
# `static` rather than `cc`: nothing here needs a dynamic loader.
ARG RUNTIME_IMAGE=gcr.io/distroless/static-debian12

FROM --platform=$BUILDPLATFORM ${GO_IMAGE} AS build
ARG TARGETOS
ARG TARGETARCH

# GOPROXY=off is the load-bearing part.  This runner imports nothing outside the
# standard library, and turning the proxy off makes that a build-time fact: a
# dependency added later fails the build here rather than quietly appearing in a
# published image whose licence inventory nobody re-derived.
#
# The build CONTEXT is this directory — see the runner-images job in ci.yml — so
# the module is created here rather than inherited from the repository root.
ENV CGO_ENABLED=0 GOPROXY=off GOFLAGS=-mod=mod
WORKDIR /src
COPY *.go ./
RUN rm -f ./*_test.go && go mod init glimmer-burnin-disk-io
RUN mkdir -p /out && GOOS=${TARGETOS} GOARCH=${TARGETARCH} go build -trimpath \
      -ldflags="-s -w" -o /out/disk-io .

# Fail the build rather than ship an image that links anything at all.
RUN ldd /out/disk-io > /out/ldd.txt 2>&1 || true; \
    cat /out/ldd.txt; \
    ! grep -qiE 'libcud(art|a)|libnv' /out/ldd.txt

FROM ${RUNTIME_IMAGE}
COPY --from=build /out/disk-io /usr/local/bin/disk-io

# Unprivileged.  The runner writes only inside a directory the BurnInTest
# declared through spec.runner.hostPaths, and it reads free space with statfs —
# neither needs root.  It also means the write faces the same permissions any
# other workload on that volume would, rather than root's reserve.
USER 65532:65532

ENTRYPOINT ["/usr/local/bin/disk-io"]
