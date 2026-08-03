# Multi-arch build (linux/arm64 for DGX Spark Grace + linux/amd64).
FROM --platform=$BUILDPLATFORM golang:1.24 AS build
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
