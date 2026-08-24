# Runner image for the "xpu-diag" TestKind — UNVERIFIED, see README.md and
# docs/dev/spike-intel-xpu-manager.md (#172). Nobody on this project has an
# Intel Data Center GPU. This Dockerfile is deliverable 4 of that spike: proof
# that CONSTRUCTION is not what stands between this kind and a real image —
# VERIFICATION on real hardware is. Do not publish a tag from this Dockerfile,
# and do not point pkg/runnerimages at it; neither happens automatically, but
# say so here too because a Dockerfile that builds is easy to mistake for one
# that has been proven correct.
#
# What this image contains and where each piece's licence comes from:
#   - our source                              Apache-2.0
#   - Intel XPU Manager (xpu-smi, DAEMONLESS)  MIT     (built from source, binary shipped)
#   - LevelZero Loader (libze_loader)          MIT     (prebuilt .deb, binary shipped)
#   - grpc, metee, igsc                        Apache-2.0 (built-from-source BUILD-TIME
#                                               dependencies of the xpumanager CMake
#                                               build; not known to be linked into the
#                                               DAEMONLESS xpu-smi binary specifically —
#                                               see the licence note in the final stage)
# Nothing here is NVIDIA- or AMD- licensed; there is no equivalent of the
# "runner images ship no NVIDIA-licensed redistributable libraries" constraint
# for this vendor, because unlike CUDA/DCGM, Intel's whole stack down to
# XPU Manager itself is permissively licensed. See docs/dev/spike-intel-xpu-manager.md
# point 1 for the verification this claim rests on and what is NOT yet verified
# (whether the shipped GPU-specific Level Zero BACKEND — intel-level-zero-gpu,
# also MIT, but tied to the installed driver version the way libcuda.so.1 is —
# should be baked in or host-supplied; this Dockerfile takes the host-supplied
# position, unverified, by analogy with every other vendor runner in this repo).
#
# The build stage below is adapted from Intel's own OFFICIAL build recipe
# (builder/Dockerfile.builder-ubuntu22.04 and BUILDING.md in intel/xpumanager's
# v1.3.x line — the actively patched CMake/CLI architecture; see the spike
# report for why NOT the repository's default branch, which is a different,
# daemon-only rearchitecture with no bounded diagnostic invocation at all). It
# is adapted, not copied: this project needs one binary out of a build that
# normally also produces a .deb/.rpm installer and a systemd daemon, and needs
# its own final stage rather than Intel's installer packaging.

ARG GO_IMAGE=golang:1.26
# Pinned to the exact base Intel's own builder recipe uses, digest and all —
# deviating from a known-working recipe blind, with no hardware to notice a
# build that silently produced something subtly wrong, is not a place to
# improvise. jammy-20220801 is from 2022; bumping it is fine but should be
# done deliberately, with the build re-verified, not as a drive-by "newer is
# safer" edit.
ARG BUILDER_IMAGE=ubuntu:jammy-20220801@sha256:34fea4f31bf187bc915536831fd0afc9d214755bf700b5cdb1336c82516d154e
ARG RUNTIME_IMAGE=gcr.io/distroless/cc-debian13

FROM ${BUILDER_IMAGE} AS xpumanager-build

# Pinned, and asserted below. A runner image is executed by a node's readiness
# gate, so the tag a gate pins must always mean the same measurement — and
# because this project's published tag is immutable, a moved upstream tag
# would change what a fleet is measured by with nothing here recording it.
# v1.3.8 is a LIGHTWEIGHT tag (git ls-remote shows no peeled ^{} value), so
# this SHA is already the commit — resolve a future bump with
# `git ls-remote https://github.com/intel/xpumanager.git 'refs/tags/<ref>' 'refs/tags/<ref>^{}'`
# and read the ^{} line if one appears.
ARG XPUMANAGER_REF=v1.3.8
ARG XPUMANAGER_SHA=a22960fe0c3479cb4237233c0863e5c4b6464b95

# Mirrors builder/Dockerfile.builder-ubuntu22.04's own dependency list at
# XPUMANAGER_REF: build tooling, plus the sources this project builds from
# source rather than trusting a prebuilt package for (grpc, metee, igsc).
ARG L0_DEV_DEB_URL=https://github.com/oneapi-src/level-zero/releases/download/v1.19.2/level-zero-devel_1.19.2+u22.04_amd64.deb
ARG L0_DEB_URL=https://github.com/oneapi-src/level-zero/releases/download/v1.19.2/level-zero_1.19.2+u22.04_amd64.deb
ARG CMAKE_INSTALLER_URL=https://cmake.org/files/v3.20/cmake-3.20.0-linux-x86_64.sh
ARG METEE_URL=https://github.com/intel/metee/archive/refs/tags/4.0.0.tar.gz
ARG IGSC_URL=https://github.com/intel/igsc/archive/refs/tags/V1.2.0.tar.gz
# grpc is `git clone`d rather than fetched as a tarball (its submodules need
# `git submodule update`), so — like XPUMANAGER below — a tag alone is not a
# pin: GRPC_REF is a lightweight tag (git ls-remote shows no peeled ^{}
# value), so GRPC_SHA is already the commit it names.
ARG GRPC_REF=v1.56.2
ARG GRPC_SHA=c0d1c393d9365664d47df41746e992ae97b651ef

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install --no-install-recommends -y \
        autoconf automake build-essential ca-certificates ccache \
        dpkg-dev git gcc g++ libssl-dev libtool liblua5.2-0 libudev-dev \
        pkg-config python3 python3-dev python3-pip wget && \
    apt-get remove -y libpciaccess-dev libpciaccess0 && \
    rm -rf /var/lib/apt/lists/*

# --prefix requires the target to already exist — the installer only cd's
# into it, it does not create it. Intel's own official recipe omits --prefix
# entirely (extracting into the installer's own directory instead); this
# stays closer to that by pre-creating the directory rather than following
# the default extraction layout, which is the smaller deviation from a
# recipe with no hardware behind it to notice a subtler one going wrong.
RUN mkdir /cmake-install /cmake && cd /cmake-install && \
    wget -qc --no-check-certificate "${CMAKE_INSTALLER_URL}" -O cmake.sh && \
    chmod +x cmake.sh && ./cmake.sh --skip-license --prefix=/cmake && \
    rm -f cmake.sh
ENV PATH="/cmake/bin:${PATH}"

RUN mkdir /work && cd /work && \
    wget -q --no-check-certificate "${L0_DEV_DEB_URL}" && \
    wget -q --no-check-certificate "${L0_DEB_URL}" && \
    dpkg -i *.deb && \
    rm -fr /work

# grpc is a build-time dependency of xpumanager's own CMake project (the
# daemon's telemetry export path uses it); built from source because there is
# no prebuilt package pinned to a commit the way the level-zero .deb is.
RUN git clone --depth 1 -b "${GRPC_REF}" https://github.com/grpc/grpc /grpc

RUN got="$(git -C /grpc rev-parse HEAD)"; \
    if [ "${got}" != "${GRPC_SHA}" ]; then \
      echo "refusing to build: ${GRPC_REF} is ${got}, expected ${GRPC_SHA}" >&2; \
      if [ "$(cd /grpc && git cat-file -t "${GRPC_SHA}" 2>/dev/null)" = "tag" ]; then \
        echo "  ${GRPC_SHA} is the ANNOTATED TAG OBJECT, not the commit it points to." >&2; \
        echo "  Use the peeled value: $(cd /grpc && git rev-parse "${GRPC_SHA}^{commit}")" >&2; \
      fi; \
      exit 1; \
    fi; \
    echo "grpc ${GRPC_REF} = ${got}: OK"

RUN cd /grpc && git submodule update --init --depth 1 && \
    mkdir -p cmake/build && cd cmake/build && \
    cmake -DgRPC_INSTALL=ON -DgRPC_BUILD_TESTS=OFF -DCMAKE_CXX_STANDARD=14 ../.. && \
    make -j"$(nproc)" && make install && \
    rm -rf /grpc

RUN wget -qc --no-check-certificate "${METEE_URL}" -O - | tar -xz && \
    cd /metee-* && mkdir build && cd build && cmake .. && make -j"$(nproc)" && make install

RUN wget -qc --no-check-certificate "${IGSC_URL}" -O - | tar -xz && \
    cd /igsc-* && mkdir build && cd build && cmake .. && make -j"$(nproc)" && make install

RUN ln -sf "$(which python3)" /usr/bin/python && ldconfig

RUN git clone --depth 1 -b "${XPUMANAGER_REF}" https://github.com/intel/xpumanager.git /xpumanager

RUN got="$(git -C /xpumanager rev-parse HEAD)"; \
    if [ "${got}" != "${XPUMANAGER_SHA}" ]; then \
      echo "refusing to build: ${XPUMANAGER_REF} is ${got}, expected ${XPUMANAGER_SHA}" >&2; \
      if [ "$(cd /xpumanager && git cat-file -t "${XPUMANAGER_SHA}" 2>/dev/null)" = "tag" ]; then \
        echo "  ${XPUMANAGER_SHA} is the ANNOTATED TAG OBJECT, not the commit it points to." >&2; \
        echo "  Use the peeled value: $(cd /xpumanager && git rev-parse "${XPUMANAGER_SHA}^{commit}")" >&2; \
      fi; \
      exit 1; \
    fi; \
    echo "xpumanager ${XPUMANAGER_REF} = ${got}: OK"

# DAEMONLESS=ON is the CLI-only build: no systemd unit, no telemetry daemon,
# just xpu-smi. This is the artefact this runner wraps; the daemon half of
# XPU Manager is out of scope for a bounded, exec-once burn-in test the same
# way DCGM's host-engine daemon is handled differently from dcgmi.
RUN cd /xpumanager && ./build.sh -DDAEMONLESS=ON

RUN set -eu; \
    bin="$(find /xpumanager/build -maxdepth 2 -type f -name 'xpu-smi' | head -n1)"; \
    if [ -z "${bin}" ]; then \
      echo "build.sh -DDAEMONLESS=ON did not produce an xpu-smi binary under /xpumanager/build" >&2; \
      find /xpumanager/build -maxdepth 3 >&2 || true; \
      exit 1; \
    fi; \
    mkdir -p /out && install -m 0755 "${bin}" /out/xpu-smi

# ── our wrapper ────────────────────────────────────────────────────────────
FROM ${GO_IMAGE} AS wrapper-build
WORKDIR /src
COPY *.go ./
# Test files exercise pure logic that needs nothing this stage lacks (unlike
# dcgm-diag's, which import this repository's pkg/contract and pkg/runner and
# so cannot even compile from a synthesised module) — but they still run from
# the repository root in CI, which is the right place for a check that
# compares this runner against the shared parser, so they are dropped here
# the same way every other Go runner in this project drops them.
RUN rm -f ./*_test.go

# No go.mod to copy: the wrapper is stdlib-only by design, which is what makes
# a synthesised module correct here and keeps this image free of any
# dependency whose licence would need separate verification.
RUN go mod init glimmer-burnin-xpu-diag

RUN CGO_ENABLED=0 go build -o /out/xpu-diag .

# ── final: minimal, no build toolchain ──────────────────────────────────────
# cc-debian13's glibc/libstdc++ cover xpu-smi's C++ runtime needs; it is the
# same base dcgm-diag and memory-bw use for the same reason.
#
# NOT INCLUDED, and this is the load-bearing unverified assumption the spike
# report names: the GPU-specific Level Zero backend (intel-level-zero-gpu /
# compute-runtime) and the in-tree i915/xe kernel driver. Both are MIT and
# COULD legally be baked in — unlike NVIDIA's CUDA userspace, nothing here is
# licence-blocked — but the backend is tied to the exact installed driver
# version the way libcuda.so.1 is, and the kernel driver can never be baked
# into any container regardless of vendor. This image ships only the
# vendor-neutral level-zero LOADER; the GPU backend is expected to be
# supplied by the host, by analogy with every other accelerator runner in
# this project. That analogy is exactly what needs checking on real hardware
# before this assumption is trusted — see docs/dev/spike-intel-xpu-manager.md.
FROM ${RUNTIME_IMAGE}

COPY --from=xpumanager-build /out/xpu-smi /usr/local/bin/xpu-smi
COPY --from=wrapper-build /out/xpu-diag /usr/local/bin/xpu-diag

USER 65532:65532
ENTRYPOINT ["/usr/local/bin/xpu-diag"]
