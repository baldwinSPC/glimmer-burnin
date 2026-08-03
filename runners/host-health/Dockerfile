# Runner image for the "host-health" TestKind: a passive read of the host- and
# driver-side fault counters (Xid, ECC, row remap / retired pages, PCIe AER,
# fabric NIC link state) that move while every performance test still passes.
#
# Licensing — this is why the build is multi-stage and why the assertions below
# exist.  The final image contains ONE file of ours: a statically linked,
# CGO-free Apache-2.0 Go binary.  It links nothing from the base image and
# contains NO NVIDIA-licensed redistributable library.  The NVML counters are
# read by executing nvidia-smi, which the NVIDIA Container Toolkit injects at
# RUNTIME from the host driver (NVIDIA_DRIVER_CAPABILITIES must include
# "utility"), so nothing NVIDIA-licensed is ever baked into a published layer.
#   - our source            Apache-2.0
#   - Go standard library   BSD-3-Clause  (compiled in; no other dependency)
#   - distroless base       Apache-2.0 (the project); the Debian packages it
#                           carries are unmodified upstream and are not linked
#                           into our binary.  See README "Licensing".

ARG GO_IMAGE=golang:1.24
# cc-debian13 rather than static-debian13 for ONE reason: the nvidia-smi the
# toolkit injects is a DYNAMICALLY linked binary and needs a loader and a libc
# from the container's own rootfs — the toolkit supplies the driver libraries,
# not glibc.  On a static base the NVML probe would silently report
# nvml_status=absent on a perfectly healthy GPU node.  A deployment that does
# not want the NVML probe at all can build with
# BASE_IMAGE=gcr.io/distroless/static-debian13 and carry no glibc.
ARG BASE_IMAGE=gcr.io/distroless/cc-debian13

# ── build ─────────────────────────────────────────────────────────────────────
# BUILDPLATFORM + GOOS/GOARCH: this is a pure-Go binary, so cross-compiling is
# free and the arm64 image builds natively on an amd64 runner.  No QEMU, unlike
# the compute-smoke runner.
FROM --platform=$BUILDPLATFORM ${GO_IMAGE} AS build
ARG TARGETOS
ARG TARGETARCH

# CGO_ENABLED=0 is what makes the binary static, which is what makes "links no
# NVIDIA library" a structural property rather than a hope.
#
# GOPROXY=off is an assertion, not an optimisation: the runner's non-test
# sources import only the standard library, and this build must fail loudly
# rather than reach the network if that ever stops being true.
ENV CGO_ENABLED=0 GOPROXY=off GOFLAGS=-mod=mod

WORKDIR /src
COPY *.go ./

# The runner's source belongs to the repository's ROOT Go module, so CI's
# `go build ./...`, `go vet ./...` and `go test ./...` cover it like any other
# package.  The publish-runner workflow, however, builds with
# `context: runners/<name>`, which cannot see the repository's go.mod — so the
# image build synthesises a throwaway module here.  Nothing is fetched.
#
# The test files are removed first: they deliberately DO import the repository's
# pkg/contract and pkg/runner (that is how the emitted metric names are proven
# legal), and those packages are outside this build context.
RUN rm -f ./*_test.go && go mod init glimmer-burnin-host-health

RUN mkdir -p /out && GOOS=${TARGETOS} GOARCH=${TARGETARCH} \
    go build -trimpath -ldflags="-s -w" -o /out/host-health .

# Fail the build rather than ship an image that drags in an NVIDIA .so.
#
# ldd exits non-zero on a static binary, so its status is discarded and its
# OUTPUT is asserted instead — both that the binary is static, and that no
# NVIDIA library appears.  The binary itself is then scanned for the same
# names, which is the check that would catch a dlopen() added later: a runtime
# dlopen leaves the soname in the binary as a string while leaving ldd clean.
RUN set -eu; \
    ldd /out/host-health > /out/ldd.txt 2>&1 || true; \
    cat /out/ldd.txt; \
    if ! grep -qE 'not a dynamic executable|statically linked' /out/ldd.txt; then \
      echo "FAIL: binary is not statically linked"; exit 1; \
    fi; \
    if grep -qiE 'libcud(art|a)|libnv' /out/ldd.txt; then \
      echo "FAIL: NVIDIA library linked into the binary"; exit 1; \
    fi; \
    if grep -qaiE 'libcud(art|a)|libnv' /out/host-health; then \
      echo "FAIL: NVIDIA library soname embedded in the binary"; exit 1; \
    fi; \
    echo "OK: static, and no NVIDIA redistributable is linked or dlopen-able"

# ── final: no NVIDIA layers, no build toolchain ───────────────────────────────
FROM ${BASE_IMAGE}

# Without "utility" the toolkit does not mount nvidia-smi or libnvidia-ml, and
# the NVML probe degrades to nvml_status=absent.  "compute" is deliberately NOT
# requested: this runner never touches a CUDA context.
ENV NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=utility

COPY --from=build /out/host-health /usr/local/bin/host-health

# Non-root by default.  The sysfs probes (PCIe AER, NIC link state) work fine
# as uid 65532; the /dev/kmsg Xid scan does NOT, on any host with
# kernel.dmesg_restrict=1.  That is a deliberate least-privilege default: the
# runner degrades to xid_source=none and omits xidEvents rather than the image
# demanding root of every node it runs on.  See README "What the pod needs".
USER 65532:65532
ENTRYPOINT ["/usr/local/bin/host-health"]
