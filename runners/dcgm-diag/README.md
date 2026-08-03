# `dcgm-diag` runner — NVIDIA DCGM diagnostics

Runner image for the `dcgm-diag` [`TestKind`](../../api/v1alpha1/burnintest_types.go).
It wraps NVIDIA's Data Center GPU Manager diagnostic suite — `dcgmi diag -r <level> -j`
— and translates its JSON into the burn-in runner contract: `key=value` metrics
on stdout, then one marker line, then an exit code.

DCGM is the vendor's own opinion about whether a part is healthy. This runner
does not second-guess it; it decides only how DCGM's answer maps onto a
*verdict*, and that mapping is where all the care is.

> **This image does not contain DCGM.** It contains our Apache-2.0 wrapper and
> nothing else. DCGM's *source* is Apache-2.0, but every DCGM *binary* NVIDIA
> distributes carries the NVIDIA Data Center GPU Manager License, which grants
> no right to redistribute — so a published image carrying them would be a
> licence violation. DCGM is supplied by the site at run time. See
> [Supplying DCGM](#supplying-dcgm) for the three supported ways, and
> [Licensing](#licensing) for the evidence.

## The workload

| Level | `-r` | DCGM budget | What it does |
|-------|------|-------------|--------------|
| 1 | `short` | ~1 min | Deployment checks through NVML: denylist, driver/NVML versions, persistence mode, page retirement, inforom, permissions. No CUDA. |
| 2 | `medium` | ~2 min | Adds the short CUDA-backed integration and hardware tests. |
| 3 | `long` | ~15 min | The long hardware tests: memory, PCIe bandwidth, targeted stress. |
| 4 | `xlong` | ~1 hr | The extended production suite. |

The level comes from `BURNIN_DCGM_LEVEL` if set (`1`–`4`, or
`short`/`medium`/`long`/`xlong`). Otherwise it is **derived from
`BURNIN_DURATION_SECONDS`**: the deepest level whose whole budget fits in the
time the test was given, floored at level 1.

Two rules about the level are deliberate and worth knowing before changing them:

- **An explicit level is never downgraded to fit.** Quietly substituting a
  cheaper level would turn a level-4 acceptance gate into a level-1 one with
  nothing in the output admitting it, and the node would be certified against a
  test it never ran. The runner bounds itself at the duration instead, so a
  level that does not fit ends as a legible `Error`.
- **The derived level needs its whole budget to fit.** A level chosen with no
  headroom gets killed at the pod's `activeDeadlineSeconds`, which reports
  `Error` — no verdict, and a burn-in slot spent for nothing.

The runner also imposes its own deadline at exactly `BURNIN_DURATION_SECONDS`.
The pod's deadline sits 120s further out, so the runner's timer always fires
first: it exits through the normal path with every metric it gathered, whereas
the kubelet's deadline is a `SIGKILL` that publishes nothing.

## Output contract

| Result | Marker on stdout | Exit |
|--------|------------------|------|
| Pass | `DCGM_DIAG_PASS` | 0 |
| Fail | `DCGM_DIAG_FAIL: <reason>` | 1 |
| Not applicable | `DCGM_DIAG_SKIP: <reason>` | 2 |
| Unjudged | `DCGM_DIAG_ERROR: <reason>` | 3 |

Metrics are printed **before** the marker on every path, including the failing
and erroring ones — a run that ends without evidence says nothing about the node
it just condemned. Prose goes to stderr; stdout carries only `key=value` lines
and the final marker.

### Metrics

The left column is what the runner prints. The right is the canonical name
[`pkg/runner`](../../pkg/runner/parse.go) normalises it to — the name a
threshold is written against.

| Runner key | Canonical | Meaning |
|------------|-----------|---------|
| `tests_failed` | `diagTestsFailed` | Subtests that returned a failure. A **count**, not a verdict: zero alongside a non-zero exit means the suite could not run. |
| `pcie_replay_count` | `pcieReplayErrors` | PCIe replay events **during the test** |
| `gpu_temp_c` | `gpuTempC` | Peak temperature seen across the test window |
| `xid_errors` | `xidEvents` | Xid errors during the test |
| `rows_remapped` | `remappedRows` | Rows remapped during the test (correctable + uncorrectable) |
| `ecc_sbe_total` | `eccSbeTotal` | Correctable ECC errors during the test |
| `ecc_dbe_total` | `eccDbeTotal` | Uncorrectable ECC errors during the test |
| `elapsed_s` | `elapsedS` | Wall-clock seconds the diagnostic ran |
| `tests_run` / `tests_warned` / `tests_not_run` / `tests_skipped` | `testsRun`, … | Subtest tallies |
| `diag_level` / `diag_level_source` | `diagLevel`, `diagLevelSource` | Which level ran, and whether it was asked for or derived |
| `dcgm_version` / `driver_version` / `gpu_count` | `dcgmVersion`, … | Provenance |
| `dcgmi_exit_code` / `sample_count` / `counter_baseline_reset` | `dcgmiExitCode`, … | Evidence about the run itself |
| `pruned_objects` | `prunedObjects` | Only if the site's DCGM tree carries `/usr/share/glimmer-burnin/dcgm-pruned.txt`, listing objects removed for licensing reasons. Absent otherwise |

Three things about these numbers matter more than the list:

**Counters are deltas, not lifetime totals.** DCGM's aggregate counters are
lifetime figures. Reported raw, a part that took one ECC error two years ago
would fail every burn-in it is ever given, and a threshold of `0` would be
untunable. The runner samples before, during and after the diagnostic and
reports the movement. A counter that goes backwards (driver reload, device
reset) reports `0` and sets `counter_baseline_reset=true` rather than a negative
number nothing can threshold.

**A metric that was not measured is not emitted.** Nothing defaults to zero. A
single sample cannot establish that a counter did not move, so with fewer than
two readings the deltas are omitted entirely. Verdict evaluation fails closed on
an absent metric, which is the correct outcome — an unmeasured counter printed
as `0` would convert that fails-closed FAILURE into a silent pass.

**There is no `eccErrors`.** DCGM reports correctable (SBE) and uncorrectable
(DBE) counts separately and they say different things: a handful of SBEs is a
working part, one DBE is a failing one. Collapsing them into the registered
`eccErrors` would either fail a healthy part on its correctable count or hide an
uncorrectable one, so this runner keeps two numbers — matching the decision
already recorded in `pkg/runner/parse.go`. **A profile thresholding `eccErrors`
against this TestKind will fail closed, because the metric is genuinely absent.
Threshold `eccDbeTotal` (and `eccSbeTotal`) instead.**

## How a verdict is reached

In order, because the order *is* the policy:

1. **Timed out** → `Error`. A diagnostic cut short cannot be read as a pass,
   however many subtests had passed by then.
2. **No subtests, and DCGM's output says the part is unsupported** → `Skip`.
3. **No subtests, no explanation** → `Error`. Excusing a node from acceptance
   requires a reason; without one, the honest answer is that we do not know.
4. **Every subtest skipped** → `Skip`. Nothing was established, and a suite that
   checked nothing must never report a green node.
5. **Any subtest failed** → `Fail`. Partial coverage does not erase an observed
   fault, so this outranks case 6.
6. **Any subtest did not run** → `Error`. A half-run suite reported as a pass
   overstates what was verified.
7. **`dcgmi` exited non-zero over an all-pass document** → `Error`. A
   contradiction is not evidence of health.
8. Otherwise → `Pass`. DCGM *warnings* are counted (`tests_warned`) and left to
   the profile's thresholds; the runner measures, the profile decides.

Per-GPU results are aggregated per subtest by severity, and the ordering there
is a policy too: one failing GPU fails the test for the node, and a GPU whose
result never arrived leaves the test **not run** rather than passed.

### Skip conditions (exit 2)

- DCGM discovers no supported GPUs (`0 GPUs found`).
- `dcgmi` refuses to run against the part, matching one of the
  unsupported-part phrases in `diagjson.go`.
- DCGM runs but skips every subtest.

**This is the path GB10 is expected to take**, and it is the reason exit 2 is in
the contract at all. A node that cannot run a test has not failed it; collapsing
the two is the false negative that makes healthy Sparks look broken.

The unsupported-part phrase list is consulted **only for a run that produced no
subtest results**. That guard is load-bearing: the same words appear in per-test
prose on perfectly healthy hardware ("NVLink is not supported on this GPU"), and
matching them there would turn a real pass into a skip and quietly excuse a node
from acceptance.

## Configuration

| Variable | Default | Purpose |
|----------|---------|---------|
| `BURNIN_DURATION_SECONDS` | *(set by the operator)* | Bounds the run; derives the level |
| `BURNIN_DCGM_LEVEL` | *(derived)* | `1`–`4` or `short`/`medium`/`long`/`xlong` |
| `BURNIN_DCGM_HOSTENGINE_ADDRESS` | *(unset)* | Connect to an existing host engine instead of starting one — use this on nodes already running `dcgm-exporter`, so two engines do not compete for the same device watches |
| `BURNIN_DCGM_PORT` | `5555` | Port for the host engine this runner starts |
| `BURNIN_NV_HOSTENGINE_ARGS` | `-n -b 127.0.0.1 -p <port>` | Full argv override. Bound to loopback deliberately: the host engine protocol is unauthenticated and this pod may run with `hostNetwork` |
| `BURNIN_DCGM_SAMPLE_INTERVAL_SECONDS` | `5` | Field-sampling period |
| `BURNIN_DCGM_BIN`, `BURNIN_NV_HOSTENGINE_BIN` | `dcgmi`, `nv-hostengine` | Binary paths |

## Supplying DCGM

The image ships no DCGM, so one of these is required. Without it the runner
prints its metrics, says why, and exits `3` (**unjudged** — never a pass, and
never a `Skip`, because the absence of a tool says nothing about the hardware).

**1. Mount a DCGM installation at `/usr/local/dcgm`** (verified on GB10). The
image's `PATH` and `LD_LIBRARY_PATH` already lead there, so nothing else needs
configuring. Lay it out as `bin/` and `lib/`; DCGM's binaries carry an RPATH of
`$ORIGIN/../lib/<triplet>`, so a `lib/<triplet>` symlink pointing at `lib` is
the whole trick. DCGM 4 also looks for its diagnostic plugins under
`/usr/libexec/datacenter-gpu-manager-4`, which is an absolute path and must be
mounted as one:

```sh
docker run --rm --gpus all \
  -v /opt/dcgm-runtime:/usr/local/dcgm:ro \
  -v /opt/dcgm-runtime/libexec:/usr/libexec/datacenter-gpu-manager-4:ro \
  ghcr.io/baldwinspc/glimmer-burnin-dcgm-diag:<tag>
```

In Kubernetes this is a `hostPath` or an `initContainer` that populates an
`emptyDir` from the site's own DCGM image.

**2. Point at a host engine that is already running.** On a node with the GPU
operator's `dcgm-exporter`, set `BURNIN_DCGM_HOSTENGINE_ADDRESS` so a second
engine does not compete for the same device watches. `dcgmi` itself still has to
be reachable — this removes the need for `nv-hostengine`, not for the client.

**3. Anywhere else on `PATH`.** `BURNIN_DCGM_BIN` and `BURNIN_NV_HOSTENGINE_BIN`
take absolute paths.

## Build

```sh
docker build -t glimmer-burnin-dcgm-diag:dev .
```

Build args: `DCGM_REF` (default `v4.2.3`), `DCGM_REPO`, `GO_IMAGE`,
`RUNTIME_IMAGE`.

`DCGM_REF` must be a tag that exists **in git**. NVIDIA tags the repository
`v4.2.3`; the `-1` suffix on package and container versions (`4.2.3-1`) is a
packaging revision that exists in NGC and apt and has never existed in git.
Asking git for `v4.2.3-1` fails with `Remote branch not found`.

The build carries two assertions. Neither is decoration, and neither should be
removed to make a build go green:

- **Field ids.** The wrapper samples DCGM fields by numeric id, and a wrong id
  does not fail at runtime — it reports a different measurement under our name.
  So the build clones DCGM's Apache-2.0 source (header only, nothing shipped)
  and checks every id in `fields.go` against `dcgm_fields.h`.

  This is not a theoretical safeguard. The first version of `fields.go` had the
  two remapped-row ids off by one, which made `remappedRows` the sum of the
  *correctable* row count and `DCGM_FI_DEV_ROW_REMAP_FAILURE`, a flag. The
  result was a small, plausible, thresholdable integer that was not the quantity
  its name claimed. The assertion is what caught it.
- **The wrapper's own linkage**, held to compute-smoke's rule: `ldd` must show
  nothing NVIDIA at all. `CGO_ENABLED=0` makes this trivially true, and it is
  also what lets the same binary run on a base full of NVIDIA libraries.

### Building a self-contained image for internal use

A site that holds an NVIDIA DCGM licence can bake DCGM in by overriding the
base, since the wrapper is a static binary that runs anywhere:

```sh
docker build \
  --build-arg RUNTIME_IMAGE=nvcr.io/nvidia/cloud-native/dcgm:4.2.3-1-ubuntu22.04 \
  -t glimmer-burnin-dcgm-diag:internal .
```

**An image built this way must not be published or otherwise distributed** — see
below. It is for a site's own registry, under its own licence acceptance.

### Publishing

Publishing goes through the manual `publish-runner` workflow. That workflow's
`runner` input is currently a fixed choice list containing only `compute-smoke`,
so **adding `dcgm-diag` to it is a prerequisite for publishing** — a change
outside this directory. Published tags are immutable; cut a new version rather
than moving one a gate pins.

## Licensing

| Component | Licence | How it is consumed |
|-----------|---------|--------------------|
| This runner (`*.go`) | Apache-2.0 | Compiled into a static binary — the only thing the image contains |
| [NVIDIA DCGM](https://github.com/NVIDIA/DCGM) source | Apache-2.0 | **Build stage only**, and only `dcgm_fields.h`, to assert the field ids. Nothing from the clone is shipped |
| `gcr.io/distroless/cc-debian13` | Apache-2.0 (image) | Final base — the same one `compute-smoke` ships |
| NVIDIA DCGM binaries | NVIDIA DCGM licence | **Not shipped.** Supplied by the site at run time |

The wrapper has no third-party Go dependencies at all — it is stdlib-only, which
is also what lets the Dockerfile synthesise its module from a build context
pinned to this directory.

### Why DCGM is not in the image

This was checked directly against
`nvcr.io/nvidia/cloud-native/dcgm:4.2.3-1-ubuntu22.04` (arm64) on 2026-08-02,
because the Apache-2.0 licence on DCGM's GitHub repository makes it easy to
assume the binaries inherit it. They do not.

That image installs four DCGM packages, and **every one of them**, including
`datacenter-gpu-manager-4-core` — the package built from the Apache-2.0 source —
carries the *NVIDIA Data Center GPU Manager License* in
`/usr/share/doc/<pkg>/copyright`, not Apache-2.0. Its section 2 reads, in part:

> b. You may not modify or create derivative works of any portion of the
> SOFTWARE.
> c. Except as provided in this license, you may not sell, rent, sublicense,
> transfer or distribute the SOFTWARE, or make its functionality available to
> others.
> e. You may not use the SOFTWARE in any manner that would cause it to become
> subject to an open source software license. As examples, licenses that require
> as a condition of use, modification, and/or distribution that the SOFTWARE be
> (i) disclosed or distributed in source code form; (ii) licensed for the purpose
> of making derivative works; or (iii) redistributable at no charge.

Sections 1 through 17 grant a right to install and use. **None of them grants a
right to redistribute.** Pushing an image containing those binaries to a public
registry, under a repository licensed Apache-2.0, breaches (c) and (e) directly.

Two further facts, either of which would be disqualifying on its own:

- The NGC image also installs `cuda-cudart-12-8` and `cuda-compat-12-8`. Those
  are NVIDIA redistributables, which this project's runner images may not carry
  under any circumstances.
- The repository's dependency rule is a strict allowlist — Apache-2.0, MIT,
  BSD-2/3-Clause. The NVIDIA DCGM licence is not on it, and a licence that
  forbids redistribution is precisely the kind the rule exists to keep out of a
  project intended to be published.

### Why it is not built from source either

Building DCGM from its Apache-2.0 source *would* be compliant. It is not
something a Dockerfile can do:

- `build.sh` re-invokes itself inside NVIDIA's own `dcgmbuild` container via
  `intodocker.sh`. That needs a Docker daemon, which a `docker build` does not
  have.
- Setting `DCGM_BUILD_INSIDE_DOCKER=1` to skip that requires
  `cmake/aarch64-linux-gnu-toolchain.cmake`, which hard-codes a crosstool-NG
  toolchain at `/opt/cross/bin/aarch64-linux-gnu-gcc` with a sysroot at
  `/opt/cross/aarch64-linux-gnu/sysroot`, plus eleven third-party dependencies
  (zlib, jsoncpp, libevent, tclap, yaml-cpp, catch2, plog, libfmt, boost,
  libnuma, and CUDA stubs) staged into it by
  `dcgmbuild/dockerfiles/target-toolchain.Dockerfile`.

None of that exists in a stock CUDA image, and reproducing it is a project in
its own right rather than a build stage. If someone takes it on, the payoff is a
self-contained image this project could publish; until then, the site supplies
DCGM.

## Requirements on the node

- NVIDIA Container Toolkit ≥ 1.17 with the `nvidia` runtime registered.
- A DCGM installation reachable by the container — see
  [Supplying DCGM](#supplying-dcgm).
- The runner runs as uid `65532`. DCGM reads the driver through `/dev/nvidia*`,
  which a stock toolkit installation exposes world read/write. Verified: DCGM
  4.2.3 starts and runs level 1 as uid `65532`, printing `nv-hostengine running
  as non-root. Some functionality will be limited.` A site that tightens
  `/dev/nvidia*` permissions must run the pod as a uid that can open them
  (`spec.runner` does not set `runAsUser`, so the image's `USER` applies).
- If the node already runs a DCGM host engine, set
  `BURNIN_DCGM_HOSTENGINE_ADDRESS` rather than letting this runner start a
  second one.

## Verified on hardware

Built and run on a GB10 DGX Spark (aarch64, driver 580.82.09, DCGM 4.2.3) on
2026-08-02, both with DCGM mounted at `/usr/local/dcgm` and with DCGM baked in
via `RUNTIME_IMAGE`. Both produce the same result:

```
diag_level=1
diag_level_source=derived
gpu_count=1
dcgm_version=4.2.3
driver_version=580.82.09
tests_run=1
tests_failed=1
tests_warned=0
tests_not_run=0
tests_skipped=0
gpu_temp_c=43
xid_errors=0
pcie_replay_count=0
ecc_sbe_total=0
ecc_dbe_total=0
sample_count=2
dcgmi_exit_code=226
elapsed_s=1.2
DCGM_DIAG_FAIL: 1 of 1 DCGM subtests failed: software — software: Persistence Mode: Persistence mode for GPU 0 is disabled. …
```

Three things in that output are the whole point of the wrapper:

- **`dcgmi` exited 226; the runner exited 1.** 226 is not a runner-contract code.
  Passing it through would turn a completely determinate diagnostic result —
  DCGM ran, and one subtest failed — into `Error`, leaving a node unjudged that
  DCGM had in fact judged.
- **`rows_remapped` is absent, not zero.** `dcgmi dmon` reports `N/A` for both
  remapped-row fields on GB10. A missing measurement is never a zero: verdict
  evaluation fails closed on an absent metric, and printing `0` would convert
  that into a silent pass.
- **The failing subtest is a *configuration* fault**, not a broken part —
  persistence mode is disabled on these machines. That is deliberately still a
  `Fail`: DCGM's level-1 deployment check exists to catch exactly this, and a
  node that is not fit for service should not clear a readiness gate. It is
  distinguishable from a hardware fault by its message and by
  `tests_failed`/`dcgmi_exit_code`, which is what `Error is not Fail` requires.

## Still unverified

1. **The pass path.** Every run so far fails on persistence mode, so exit 0 has
   not been observed on real hardware. `nvidia-smi -pm 1` (as root, on the host)
   should produce it. Enabling it changes a driver-level system setting and was
   left to the machine's owner.
2. **Levels 2–4.** Only level 1 has been run. The CUDA-backed levels need DCGM's
   diagnostic plugins, and how completely a site's mounted tree provides them is
   a property of that tree, not of this image.
3. **Multi-GPU aggregation.** The GB10 Spark has one GPU. The per-entity
   aggregation in `diagjson.go` (one failing GPU fails the test for the node) is
   covered by unit tests only.
4. **`BURNIN_DCGM_HOSTENGINE_ADDRESS`.** Supported and exercised by the code
   path that skips starting an engine, but not tested against a real external
   `dcgm-exporter` host engine.
7. **The GB10 skip path** — the whole point of this runner's exit 2. The exact
   phrase DCGM emits on an unsupported part is unconfirmed; if it is not in
   `unsupportedSignatures`, GB10 will report `Error` (unjudged) rather than
   `Skip`. That is the safe direction to be wrong in, but it is still wrong, and
   it is the first thing to check on real silicon.
