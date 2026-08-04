# Runner image for the "dcgm-diag" TestKind: a wrapper that drives NVIDIA DCGM's
# diagnostic suite (`dcgmi diag -r <level> -j`) and reports the result in the
# burn-in runner contract — key=value metrics on stdout and an exit code of
# 0 (pass) / 1 (fail) / 2 (not applicable to this hardware) / 3 (unjudged).
#
# ── What this image contains, and what it deliberately does not ───────────────
#
# It contains one thing: our Apache-2.0 Go wrapper, statically linked, on the
# same distroless base every other runner in this project ships. It contains NO
# DCGM.
#
# That is not an oversight, and it is not a build that gave up. DCGM's SOURCE is
# Apache-2.0, but every DCGM BINARY NVIDIA distributes — including
# datacenter-gpu-manager-4-core, the package whose contents are built from that
# Apache-2.0 source — carries the NVIDIA Data Center GPU Manager License, whose
# section 2 reads, in part:
#
#   b. You may not modify or create derivative works of any portion of the
#      SOFTWARE.
#   c. Except as provided in this license, you may not sell, rent, sublicense,
#      transfer or distribute the SOFTWARE, or make its functionality available
#      to others.
#   e. You may not use the SOFTWARE in any manner that would cause it to become
#      subject to an open source software license. As examples, licenses that
#      require as a condition of use, modification, and/or distribution that the
#      SOFTWARE be (i) disclosed or distributed in source code form; (ii)
#      licensed for the purpose of making derivative works; or (iii)
#      redistributable at no charge.
#
# Sections 1 through 17 grant no redistribution right anywhere. Baking those
# binaries into an image this project pushes to a public registry under
# Apache-2.0 would breach (c) and (e) directly, and would breach this repo's
# permissive-only rule besides. The prebuilt NGC image
# (nvcr.io/nvidia/cloud-native/dcgm) additionally ships cuda-cudart and
# cuda-compat, which are NVIDIA redistributables that runner images may not
# carry at all. Verified against DCGM 4.2.3 on 2026-08-02; see README.md.
#
# Building DCGM from its Apache-2.0 source instead — which WOULD be compliant —
# is not something a Dockerfile can do. DCGM's build.sh re-invokes itself inside
# NVIDIA's own dcgmbuild container (docker-in-docker, unavailable during a
# build), and forcing DCGM_BUILD_INSIDE_DOCKER=1 requires a crosstool-NG
# toolchain at /opt/cross plus eleven third-party dependencies staged into a
# sysroot, none of which exist in a stock CUDA image. See README.md.
#
# So DCGM is supplied by the SITE, at run time, from an installation the site
# already has and has already accepted NVIDIA's licence for — the GPU operator's
# DCGM container, a host installation, or a mount. The wrapper looks for it on
# PATH, refuses to guess, and reports Error (unjudged) rather than a verdict when
# it is absent. See README.md for the three supported ways to provide it.

ARG GO_IMAGE=golang:1.24

# The base of the shipped image. Declared here, in the global scope, because an
# ARG a FROM expands must be declared before the first FROM — one declared after
# it belongs to that stage and is invisible later, which fails the build with
# "base name should not be blank".
#
# Overriding this is the supported way to build a self-contained variant for
# INTERNAL use, on a site that holds an NVIDIA DCGM licence:
#
#   docker build --build-arg RUNTIME_IMAGE=nvcr.io/nvidia/cloud-native/dcgm:4.2.3-1-ubuntu22.04 .
#
# The wrapper is a static, CGO-free binary, so it runs on any base. An image
# built that way MUST NOT be published or otherwise distributed.
#
# Release builds of the default should pin this by digest — a floating base tag
# changes the contents of an immutable runner tag, which is the audit-trail hole
# the publish-runner rules exist to close.
ARG RUNTIME_IMAGE=gcr.io/distroless/cc-debian13

# ── stage 1: the wrapper, and the field-id assertion ──────────────────────────
FROM --platform=$BUILDPLATFORM ${GO_IMAGE} AS gobuild
ARG TARGETOS
ARG TARGETARCH
WORKDIR /src

COPY *.go ./
# The test files import this repository's pkg/contract and pkg/runner to prove
# the emitted keys survive the parser, and the publish workflow pins the build
# context to this directory, so those packages are not here. They run from the
# repository root in CI, which is the right place for them anyway: they check
# the runner against the parser, and a check living in only one of the two would
# not notice them drifting apart.
RUN rm -f ./*_test.go

# There is no go.mod to copy, for the same context reason. The wrapper is
# stdlib-only by design — that is what makes a synthesised module correct here,
# and what keeps this image free of any dependency whose licence would have to
# be verified separately.
RUN go mod init glimmer-burnin-dcgm-diag

# Built for the BUILD platform, not the target, purely so the assertion below
# can execute it. It is never shipped.
RUN CGO_ENABLED=0 go build -o /out/dcgm-diag-check .

# ── the field-id assertion ────────────────────────────────────────────────────
# The wrapper samples DCGM fields by numeric id. A wrong id does not fail at
# runtime — it reports a different measurement under our name, and a plausible
# small integer in a temperature or remapped-rows column is a substitution
# nothing downstream could ever detect. (This is not hypothetical: the first
# version of fields.go had the two remapped-row ids off by one, which made
# remappedRows the sum of the CORRECTABLE row count and a row-remap-FAILURE
# flag. The assertion is what found it.)
#
# So the ids are checked against DCGM's own header. Only the header is needed,
# and only at build time: the source is Apache-2.0 and nothing from this clone
# is shipped, so reading it carries none of the licence problem that shipping
# NVIDIA's compiled binaries would.
ARG DCGM_REPO=https://github.com/NVIDIA/DCGM.git
# A real tag. NVIDIA tags this repo "v4.2.3"; the "-1" suffix seen on package
# and container versions (4.2.3-1) is a packaging revision that exists in NGC
# and apt, never in git — asking git for v4.2.3-1 fails with "Remote branch not
# found". Pinned, never a branch: "whatever main was that morning" is not a
# reproducible acceptance verdict.
ARG DCGM_REF=v4.2.3
# A tag is not an identity: it can be moved, and this one decides whether the
# field-id assertion below is checking against the header the runner was written
# for. NVIDIA tags DCGM with ANNOTATED tags, so this is the COMMIT the tag peels
# to (`git ls-remote … 'refs/tags/v4.2.3^{}'`), which is what a shallow clone
# leaves at HEAD — not the tag object's own sha, which is 0f3607ae.
ARG DCGM_SHA=6e947dcac9b3160d61d98fea4741d51d4bec5c1f
RUN set -eu; \
    GIT_LFS_SKIP_SMUDGE=1 git clone --depth 1 -b "${DCGM_REF}" "${DCGM_REPO}" /dcgm-src; \
    got="$(git -C /dcgm-src rev-parse HEAD)"; \
    if [ "${got}" != "${DCGM_SHA}" ]; then \
      echo "refusing to build: ${DCGM_REF} is ${got}, expected ${DCGM_SHA}" >&2; \
      exit 1; \
    fi; \
    echo "DCGM ${DCGM_REF} = ${got}: OK"; \
    hdr=$(find /dcgm-src -name dcgm_fields.h -print -quit); \
    [ -n "$hdr" ] || { echo "BUILD ASSERTION FAILED: dcgm_fields.h not found in the DCGM source"; exit 1; }; \
    echo "checking field ids against $hdr (DCGM ${DCGM_REF})"; \
    /out/dcgm-diag-check --print-field-ids > /tmp/field-ids.txt; \
    while IFS='=' read -r sym id; do \
      [ -n "$sym" ] || continue; \
      want=$(sed -n "s/^#define[[:space:]]\{1,\}${sym}[[:space:]]\{1,\}\([0-9]\{1,\}\).*/\1/p" "$hdr" | head -n 1); \
      if [ -z "$want" ]; then \
        echo "FIELD ID ASSERTION FAILED: $sym is not defined in $hdr."; \
        echo "  Fix the symbol or the id in fields.go. Do not delete this check."; \
        exit 1; \
      fi; \
      if [ "$want" != "$id" ]; then \
        echo "FIELD ID ASSERTION FAILED: $sym is $want in $hdr, but this runner was compiled with $id"; \
        exit 1; \
      fi; \
      echo "  ok $sym=$id"; \
    done < /tmp/field-ids.txt; \
    rm -rf /dcgm-src

# The binary that ships. CGO_ENABLED=0 is load-bearing twice over: it is what
# lets one binary run on the distroless default AND on a site's own DCGM image,
# and it is what guarantees the wrapper links nothing NVIDIA even when it is
# built into a base that is full of NVIDIA libraries.
RUN CGO_ENABLED=0 GOOS=${TARGETOS} GOARCH=${TARGETARCH} \
    go build -trimpath -ldflags="-s -w" -o /out/dcgm-diag .

# Held to compute-smoke's rule: a static binary that links nothing NVIDIA at
# all. ldd exits non-zero on a static binary, hence the guard.
RUN set -eu; \
    ldd /out/dcgm-diag > /tmp/ldd.txt 2>&1 || true; \
    cat /tmp/ldd.txt; \
    ! grep -qiE 'libcud|libnv' /tmp/ldd.txt

# ── final ─────────────────────────────────────────────────────────────────────
FROM ${RUNTIME_IMAGE}

# NVIDIA_* : the NVIDIA-provided CUDA images set these; the distroless default
# does not, so we set them ourselves. Without "utility" the Container Toolkit
# does not mount libnvidia-ml.so.1 and DCGM cannot see the driver at all;
# "compute" covers libcuda.so.1, which the CUDA-backed diagnostic levels need.
#
# PATH / LD_LIBRARY_PATH : /usr/local/dcgm is the documented mount point for a
# site-supplied DCGM (bin/ and lib/). Prepending costs nothing when it is absent
# and makes the common case a single -v with no other configuration.
#
# Both PREPEND to whatever the base already had rather than replacing it, which
# is what keeps this correct when RUNTIME_IMAGE is overridden with an image that
# has DCGM installed in its own places: replacing LD_LIBRARY_PATH would drop the
# CUDA path such an image sets for its own plugins. The ${VAR:+:${VAR}} form is
# there so an unset LD_LIBRARY_PATH does not leave a trailing colon — which ld.so
# reads as "also search the current directory".
ENV NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    TMPDIR=/tmp
ENV PATH=/usr/local/dcgm/bin:${PATH}
ENV LD_LIBRARY_PATH=/usr/local/dcgm/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}

COPY --from=gobuild /out/dcgm-diag /usr/local/bin/dcgm-diag

# Non-root, matching every other image in this project. DCGM reads the driver
# through /dev/nvidia*, which the NVIDIA Container Toolkit exposes world
# read/write on a stock installation; a site that tightens those permissions, or
# that supplies a DCGM needing a privileged host engine, must run this pod as a
# uid that can open them. See the README.
USER 65532:65532
ENTRYPOINT ["/usr/local/bin/dcgm-diag"]
