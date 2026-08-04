# Multi-arch build (linux/arm64 for DGX Spark Grace + linux/amd64).
#
# Keep this major.minor equal to the `go` directive in go.mod. CI asserts it
# ("Dockerfile Go matches go.mod"), because the two drifting is invisible until
# it isn't: everything reading go.mod — setup-go, local builds, `make test` —
# moves to the new toolchain together, while this line stays behind and only
# the image build fails. The image build does not run on pull requests, so the
# break lands on main, already merged and already green everywhere a reviewer
# looked. That is exactly how it happened: a dependency bump raised go.mod to
# 1.26, this line stayed at 1.24, and every subsequent push to main failed on
# `go mod download` with GOTOOLCHAIN=local refusing to fetch a newer toolchain.
FROM --platform=$BUILDPLATFORM golang:1.26 AS build
ARG TARGETOS
ARG TARGETARCH
WORKDIR /workspace

COPY go.mod go.sum ./
RUN go mod download

COPY cmd/ cmd/
COPY api/ api/
COPY internal/ internal/
# pkg/ is NOT optional: internal/controller imports pkg/runner and pkg/verdict,
# and internal/sink imports pkg/contract. Omitting it fails the build with
# "no matching versions for query latest", because the missing local package
# looks to the module resolver like an unresolvable remote one.
COPY pkg/ pkg/

RUN CGO_ENABLED=0 GOOS=${TARGETOS} GOARCH=${TARGETARCH} \
    go build -trimpath -ldflags="-s -w" -o manager ./cmd

FROM gcr.io/distroless/static:nonroot
WORKDIR /
COPY --from=build /workspace/manager /manager
USER 65532:65532
ENTRYPOINT ["/manager"]
