# Runner image for the "memory-retention" TestKind: writes a pattern into a
# region of host memory, holds it untouched for a duration, then reads it
# back — a memtest86+-shaped bit-fade test, not memtest86+ itself (which
# needs a reboot and cannot run inside either of this project's container
# dispatchers).
#
# Licensing — this is why the build is multi-stage and why the assertions
# below exist. The final image contains ONE file of ours: a statically
# linked, CGO-free Apache-2.0 Go binary. It links nothing from the base image,
# executes no external tool, and touches no vendor library of any kind — this
# runner is entirely vendor-neutral, unlike every other runner in this repo.
#   - our source            Apache-2.0
#   - Go standard library   BSD-3-Clause  (compiled in; no other dependency)
#   - distroless base       Apache-2.0 (the project); the Debian packages it
#                           carries are unmodified upstream and are not
#                           linked into our binary. See README "Licensing".

ARG GO_IMAGE=golang:1.26
ARG BASE_IMAGE=gcr.io/distroless/static-debian13

# ── build ─────────────────────────────────────────────────────────────────────
# BUILDPLATFORM + GOOS/GOARCH: a pure-Go binary with no cgo and no external
# tool, so cross-compiling is free — both the amd64 and the arm64 image build
# natively on whichever runner CI happens to be on, no QEMU either side.
FROM --platform=$BUILDPLATFORM ${GO_IMAGE} AS build
ARG TARGETOS
ARG TARGETARCH

# CGO_ENABLED=0 is what makes the binary static. GOPROXY=off is an assertion,
# not an optimisation: the runner's non-test sources import only the standard
# library, and this build must fail loudly rather than reach the network if
# that ever stops being true.
ENV CGO_ENABLED=0 GOPROXY=off GOFLAGS=-mod=mod

WORKDIR /src
COPY *.go ./

# The runner's source belongs to the repository's ROOT Go module, so CI's
# `go build ./...`, `go vet ./...` and `go test ./...` cover it like any other
# package. The publish-runner workflow, however, builds with
# `context: runners/<name>`, which cannot see the repository's go.mod — so the
# image build synthesises a throwaway module here. Nothing is fetched.
#
# The test files are removed first: they deliberately DO import "testing" and
# rely on t.TempDir()-scoped fixtures, and the shipped binary carries neither.
RUN rm -f ./*_test.go && go mod init glimmer-burnin-memory-retention

RUN mkdir -p /out && GOOS=${TARGETOS} GOARCH=${TARGETARCH} \
    go build -trimpath -ldflags="-s -w" -o /out/memory-retention .

# Fail the build rather than ship a binary that quietly linked something.
RUN set -eu; \
    ldd /out/memory-retention > /out/ldd.txt 2>&1 || true; \
    cat /out/ldd.txt; \
    if ! grep -qE 'not a dynamic executable|statically linked' /out/ldd.txt; then \
      echo "FAIL: binary is not statically linked"; exit 1; \
    fi; \
    echo "OK: static, no dynamic dependency of any kind"

# ── final: no build toolchain, nothing but the binary ─────────────────────────
FROM ${BASE_IMAGE}

COPY --from=build /out/memory-retention /usr/local/bin/memory-retention

USER 65532:65532
ENTRYPOINT ["/usr/local/bin/memory-retention"]
