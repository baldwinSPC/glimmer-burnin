# Runner image for the "fingerprint-probe" TestKind: what the hardware says
# about itself, read from sysfs.
#
# NO THIRD-PARTY COMPONENT.  A standard-library Go program over a read-only
# sysfs mount — no vendor tool, no ioctl, no privileged access.
#
# Licensing of everything in the final image:
#   - our runner (*.go)         Apache-2.0
#   - Go standard library       BSD-3-Clause (linked in)
#
# NO ACCELERATOR ACCESS.  This runner READS ABOUT accelerators; it never opens
# one.  It sets no NVIDIA_* environment and requests no device — which is the
# entire point: it has to work on a node whose device plugin never came up, and
# an image that needed driver injection could not.
# runners/pins_test.go asserts that absence rather than leaving it to a reader.

ARG GO_IMAGE=golang:1.24-bookworm
ARG RUNTIME_IMAGE=gcr.io/distroless/static-debian12

FROM --platform=$BUILDPLATFORM ${GO_IMAGE} AS build
ARG TARGETOS
ARG TARGETARCH

# GOPROXY=off makes "imports nothing outside the standard library" a build-time
# fact rather than an intention — and here it is doing double duty, because this
# runner's pci.go is a deliberate second implementation of
# pkg/hostinfo.accelerators (issue #250).  A future edit that "fixes" the
# duplication by importing the repo package fails HERE, loudly, rather than
# producing an image that cannot be built from this context.
#
# The build CONTEXT is this directory — see the runner-images job in ci.yml.
ENV CGO_ENABLED=0 GOPROXY=off GOFLAGS=-mod=mod
WORKDIR /src
COPY *.go ./
RUN rm -f ./*_test.go && go mod init glimmer-burnin-fingerprint-probe
RUN mkdir -p /out && GOOS=${TARGETOS} GOARCH=${TARGETARCH} go build -trimpath \
      -ldflags="-s -w" -o /out/fingerprint-probe .

RUN ldd /out/fingerprint-probe > /out/ldd.txt 2>&1 || true; \
    cat /out/ldd.txt; \
    ! grep -qiE 'libcud(art|a)|libnv' /out/ldd.txt

FROM ${RUNTIME_IMAGE}
COPY --from=build /out/fingerprint-probe /usr/local/bin/fingerprint-probe

# Unprivileged, and it reads only.  sysfs is world-readable for the attributes
# this needs, so there is nothing here that root would make possible.
USER 65532:65532

ENTRYPOINT ["/usr/local/bin/fingerprint-probe"]
