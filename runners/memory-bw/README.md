# `memory-bw` runner — NVIDIA nvbandwidth

Runner image for the `memory-bw` [`TestKind`](../../api/v1alpha1/burnintest_types.go):
a Node-scope measurement of the paths data actually travels on — host into
device, device back to host, and device memory to device memory.

It exists as its own kind because a bandwidth shortfall and a compute shortfall
have different root causes. A part that computes at full rate but reads memory
at half rate is usually a link, a seating, or a downgraded PCIe/NVLink
negotiation — not a bad die — and every compute test on the node will pass while
it happens.

The workload is [NVIDIA nvbandwidth](https://github.com/NVIDIA/nvbandwidth)
(Apache-2.0), built from source in the build stage. A small wrapper
(`memory_bw.cc`, Apache-2.0) is the image's entrypoint: it probes for a CUDA
device, runs nvbandwidth once per testcase, and translates the result into the
runner contract.

## Output contract

| Result | stdout marker | Exit |
|--------|---------------|------|
| Pass | `MEMORY_BW_PASS` | 0 |
| Fail | `MEMORY_BW_FAIL: <reason>` | 1 |
| Not applicable | `MEMORY_BW_SKIP: <reason>` | 2 |
| Error (unjudged) | `MEMORY_BW_ERROR: <reason>` | 3 |

Every metric this runner obtained is printed **before** the marker, so a run
that ends in Fail or Error still leaves its evidence behind. stdout is
line-buffered for the same reason: a pod killed at its `activeDeadlineSeconds`
must not take the measurements with it.

nvbandwidth's own output is echoed to **stderr**, one line at a time, behind a
`nvbandwidth| ` prefix. The prefix is load-bearing rather than decorative: the
controller parses the pod log, and a line only becomes a metric when the text
before its first `=` contains no whitespace, so the prefix guarantees nothing
the tool prints can be mistaken for one of ours.

## Metrics

The three acceptance metrics, and the nvbandwidth testcase behind each:

| Canonical name (`pkg/contract`) | stdout key | Source |
|---|---|---|
| `hostToDeviceBandwidthGBs` | `h2d_bandwidth_gbs` | `host_to_device_memcpy_ce` |
| `deviceToHostBandwidthGBs` | `d2h_bandwidth_gbs` | `device_to_host_memcpy_ce` |
| `deviceToDeviceBandwidthGBs` | `d2d_bandwidth_gbs` | `device_local_copy` |

Each is the **minimum** cell of that testcase's matrix, because acceptance is
decided by the worst path on the node: on an eight-GPU box, one accelerator
negotiated down to a narrower link is exactly the fault this test exists to
catch, and a mean would dilute it into noise. Cells reported as `0.00` (the
diagonal of a peer matrix — a pair the testcase did not measure) and `N/A` are
excluded rather than counted as zero.

The corresponding maxima are emitted as evidence —
`hostToDeviceBandwidthMaxGBs`, `deviceToHostBandwidthMaxGBs`,
`deviceToDeviceBandwidthMaxGBs`. On a single-GPU node they equal the minimum; on
a multi-GPU node the spread between the two is what distinguishes "this whole
node is slow" from "one device on this node is slow".

Also emitted, as context rather than as gates: `nvbandwidth_ref`, `gpu_name`,
`devices_visible` (registered as `devicesVisible`: how many devices the runtime
showed the pod — it was `gpu_count` before v0.7 — see
docs/dev/multi-device.md), `cuda_driver_version`, `transfer_size_bytes`,
`test_samples`, `elapsed_s`.

> The two spellings on stdout are deliberate. The three acceptance keys use the
> snake spelling the operator's alias table in
> [`pkg/runner/parse.go`](../../pkg/runner/parse.go) already maps. The maxima are
> printed already-canonical because a new snake key ending in `_gbs` normalises
> to `…Gbs`, which is *not* the registered `GBs` suffix — the metric would be
> stored as a dimensionless number and a bandwidth would quietly stop being a
> bandwidth. The parser accepts either spelling.

## Multi-device

nvbandwidth already iterates every device the runtime shows it, so this
runner's multi-device conversion is not per-device iteration — it is the
**allocation budget check** `device_fold.h`'s `parseBudget`/`planIteration`
perform (docs/dev/multi-device.md): what `BURNIN_RESOURCE_LIMITS` says this
pod was allocated must equal `devices_visible`, or the runner refuses (exit 3)
before nvbandwidth ever runs, rather than measuring devices it cannot
establish this pod owns. This runner still does **not** claim `deviceCount` —
only a runner that folds devices individually does, and nvbandwidth's own
peer-matrix output already covers every visible device, so there is no
per-device fold for `device_fold.h` to perform here.

### Why `device_local_copy` backs `deviceToDeviceBandwidthGBs`

nvbandwidth's `device_to_device_*` testcases are GPU-to-**peer**-GPU copies: they
need at least two accelerators and what they measure is the peer link.

The metric this project registered under that name is the on-device figure —
"bounded by the memory subsystem rather than by the host link" — which is
`device_local_copy`. A peer-link figure is a different measurand, so it gets
**names of its own** rather than sharing this column: putting two quantities in
one place is what the separation avoids.

### The peer matrix

`peerReadBandwidthGBs` and `peerWriteBandwidthGBs` are the **worst cell** of the
all-pairs matrix, the two directions measured separately because a link can
degrade asymmetrically and averaging them would hide it.

On a multi-GPU node this is where the interesting failures live. A degraded
NVLink lane, an xGMI link that trained narrow, a switch port that fell back —
none of them move a device-local copy, and all of them halve a training job that
assumed the fabric. The minimum rather than the mean, for the reason this runner
reports minima everywhere: a fabric is as good as its worst link, and an average
over the matrix hides the single bad lane the measurement exists to find. The
`*MaxGBs` companions are evidence — their distance from the minimum is what
makes ONE degraded link visible as a spread rather than as a slightly lower
average.

**On a single-GPU node these are `n/a` — not absent, and not a skip of the whole
test.** There is no peer, so the runner declares the figures unmeasurable.
Omitting the keys would make a gate on them fail closed and condemn every
single-GPU node in a fleet; exiting 2 would throw away the three device-local
figures the run did take, which on a DGX Spark is the entire value of this kind.
Pair such a gate with `applicability: RequiredIfMeasurable` and it is reported
NOT EVALUATED instead. The same declaration is made when two or more devices are
present but no peer path exists between them.

Ship them thresholdless first: peer bandwidth is SKU- and topology-specific, and
this project pins thresholds from measured baselines rather than datasheets.

### Why `memoryBandwidthGBs` is *not* emitted

That metric is registered as "sustained device memory bandwidth achieved over
the whole measurement window, not a peak sample". nvbandwidth reports the median
of a handful of short samples, which is not that number. Emitting it anyway
would let a profile author write a sustained-bandwidth gate and have it
evaluated against a burst measurement — the same trap `throughputTflops` and
`sustainedThroughputTflops` are kept apart to avoid.

## Thresholds

**This runner ships thresholdless**, pending fleet baselines. Absolute bandwidth
is SKU-, topology- and host-specific (PCIe generation and width, NUMA placement,
C2C on a coherent part), so a floor invented before the fleet has been measured
would either pass everything or condemn healthy hardware. Collect
`hostToDeviceBandwidthGBs`/`deviceToHostBandwidthGBs`/`deviceToDeviceBandwidthGBs`
across the fleet first, then set the floors in a `BurnInProfile`.

All three are registered `Acceptance` metrics, so a profile can gate on them as
soon as those numbers exist. Until then a passing run means "the copy paths ran
and were measured", not "the copy paths were fast enough".

## What each verdict means

**Pass (0)** — all three testcases ran, each produced at least one bandwidth
figure, and nvbandwidth's post-copy data verification found no corruption.

**Fail (1)** — nvbandwidth's data verification found bytes that did not survive
a copy. This is the only hardware verdict the runner reaches on its own while it
is thresholdless, and it is a real one: a copy engine that returns success and
wrong data is silent corruption.

**Skip (2)** — the test does not apply to this node:

- `libcuda.so.1` is not present in the container (no NVIDIA accelerator, or the
  NVIDIA Container Toolkit is not wired up);
- the driver reports `CUDA_ERROR_NO_DEVICE`, or zero CUDA devices;
- nvbandwidth waived one of the three testcases on this hardware.

**Error (3)** — the hardware is *unjudged*. `Error` is not `Fail`, and this
runner leans on that distinction rather than guessing:

- nvbandwidth exited non-zero without a data-verification failure. All of its
  failure paths — a CUDA API error, an unknown testcase, an allocation it could
  not make — exit 1 through one `ASSERT` macro, so the exit code alone cannot
  separate a bad part from a bad environment;
- nvbandwidth exited 0 and no bandwidth figure could be read from its output.
  That is a statement about this wrapper's parser or about a changed output
  format, not about the memory subsystem, and it must never become a Fail;
- the run exceeded its `BURNIN_DURATION_SECONDS` budget;
- `libcuda.so.1` is present but `cuInit` failed for any reason other than "no
  device" — a driver mismatch is an infrastructure fault, and reporting Skip
  would quietly excuse a node that was never tested.

An Error leaves the node unaccepted and lets the operator retry it. A wrong Fail
condemns working hardware, which is why every ambiguous case resolves this way.

## Configuration

| Variable | Default | Effect |
|---|---|---|
| `BURNIN_DURATION_SECONDS` | 600 | Wall-clock **budget**; on expiry the child is killed and the run reports Error. `0` disables the deadline. |
| `BURNIN_MEMORY_BW_SAMPLES` | 3 | nvbandwidth `-i`: benchmark iterations per testcase. |
| `BURNIN_MEMORY_BW_BUFFER_MIB` | 512 | nvbandwidth `-b`: copy buffer size, MiB. |
| `BURNIN_MEMORY_BW_DISABLE_AFFINITY` | 0 | Non-zero passes nvbandwidth `-d`, disabling its automatic CPU affinity control. |

`BURNIN_DURATION_SECONDS` is honoured as a budget, not as a workload length:
this workload is **not duration-shaped**. Its work is set by the sample count,
so a larger duration does not buy a longer soak — use `thermal-soak` or
`gpu-burn` for that. What the budget does buy is a deadline the runner enforces
itself, so a copy wedged in the driver is reported as an Error *with* whatever it
had already measured, instead of being SIGKILLed by the kubelet with an empty
log. A malformed value falls back to the default rather than erroring: a typo in
a profile must not read like a hardware problem.

The buffer size and sample count default to nvbandwidth's own defaults so a
verdict here stays comparable with the upstream tool's published numbers.

## Build

```sh
docker build -t glimmer-burnin-memory-bw:dev .
```

Build args:

| Arg | Default | Notes |
|---|---|---|
| `CUDA_IMAGE` | `nvcr.io/nvidia/cuda:13.0.1-devel-ubuntu24.04` | Build stage only; no CUDA layer is shipped. |
| `NVBANDWIDTH_REF` | `v0.10.0` | Git tag built from source, and the value reported as `nvbandwidth_ref`. |
| `NVBANDWIDTH_SHA` | `82fc4e8c…` | The commit that tag must resolve to. The build refuses otherwise: `nvbandwidth_ref` is the provenance of every number this runner publishes, so a moved tag would make that line a false statement about which code produced the measurement. Bump it with `NVBANDWIDTH_REF`, resolving it with `git ls-remote https://github.com/NVIDIA/nvbandwidth.git 'refs/tags/<ref>'`. |
| `CUDA_ARCH` | *(empty)* | Empty uses nvbandwidth's own architecture list, which is correct on both host architectures. Set e.g. `121` for a native GB10 cubin. Accepts an nvcc gencode name (`sm_121a`) or a bare number; the build translates and refuses anything it cannot. |
| `RUNTIME_IMAGE` | `gcr.io/distroless/cc-debian13` | Final base. Pin by digest for a release build. |
| `NVBANDWIDTH_PATH` | `/usr/local/bin/nvbandwidth` | Where the tool is installed; the wrapper is compiled against this value. |

`RUNTIME_IMAGE` and `NVBANDWIDTH_PATH` are declared **before the first `FROM`**,
and must stay there. An `ARG` that a `FROM` expands has to be in the global
scope; one declared inside a preceding stage belongs to that stage and is
invisible to any later `FROM`, which fails the build with the
cause-free `base name (${RUNTIME_IMAGE}) should not be blank`. Stages that also
need the value re-declare a bare `ARG` to pull it in — that is what the
`ARG NVBANDWIDTH_PATH` lines with no default are for.

`CUDA_ARCH` does not affect any number this runner reports: all three testcases
it runs are copy-engine testcases, which move data without launching a kernel.
It matters only if the plan in `memory_bw.cc` is switched to the `_sm` variants
— and then it matters completely, because a JIT-compiled kernel is not the code
path a gate is trying to accept.

`CUDA_ARCH` is the **GPU** axis and is independent of the host architecture. The
image is published as a manifest list for `linux/amd64` and `linux/arm64`, and
the empty default is right on both: nvbandwidth's own list already spans the
parts either host architecture is deployed with. That is why this runner needed
no per-platform default, unlike `compute-smoke` and `nccl`.

The build fails rather than ship a bad image if any of the following is true,
checked against `DT_NEEDED` with `objdump`:

- a shipped binary needs `libstdc++`/`libgcc_s` (the distroless base has
  neither, so the image would start and immediately die);
- a shipped binary needs any `libcud*`/`libnv*` object other than
  `libcuda.so.1` or `libnvidia-ml.so.1` — those two are host driver libraries
  injected at runtime, everything else in that namespace is an NVIDIA
  redistributable;
- any shared library ended up in the staging directory.

## Tests

`scan_test.cc` covers the part of the wrapper that can be wrong without being
obviously wrong: the scan of nvbandwidth's output, and the classification that
decides whether a non-zero exit is corruption (Fail) or anything else (Error).

```sh
c++ -std=c++17 -Wall -Wextra -o /tmp/scan_test scan_test.cc && /tmp/scan_test
```

It needs no test framework and is not compiled into the image. **`make test` and
CI do run it**: `runners/cxxtests_test.go` sweeps the tree for every
`*_test.cc`, compiles it with the line above and runs it, so a native runner's
unit tests are gated like any other. No Docker, no CUDA toolchain, no GPU.

The operator-side parser for this kind — the alias table that maps
`h2d_bandwidth_gbs` and friends onto canonical metric names — lives in
[`pkg/runner/parse.go`](../../pkg/runner/parse.go) and *is* covered by CI, in
`pkg/runner/parse_test.go`.

## Licensing

| Component | License | How it is consumed |
|---|---|---|
| `memory_bw.cc`, `Dockerfile` | Apache-2.0 | This project's own source |
| [NVIDIA nvbandwidth](https://github.com/NVIDIA/nvbandwidth) `v0.10.0` | Apache-2.0 | Built from source; the binary is shipped |
| [p-ranav/argparse](https://github.com/p-ranav/argparse) `v3.2` | MIT | Header-only, fetched by nvbandwidth's CMake, compiled in |
| JsonCpp | MIT | Vendored in nvbandwidth (`json/`), compiled in |
| NVIDIA CUDA toolchain | NVIDIA EULA | **Build time only** — no CUDA layer, and no CUDA redistributable, is shipped |
| NVIDIA driver (`libcuda.so.1`, `libnvidia-ml.so.1`) | NVIDIA driver license | Injected at runtime from the host by the NVIDIA Container Toolkit; never baked in |

Every third-party component compiled into or shipped by this image is
Apache-2.0 or MIT — both on this project's allowlist. nvbandwidth's own
`Licenses.txt` is the source for the argparse and JsonCpp entries; it records
JsonCpp as "public domain and MIT", and this project consumes it under the
**MIT** option.

Two system-library facts, unchanged from the `compute-smoke` runner and noted
here so they are not rediscovered later: the distroless base provides glibc
(LGPL-2.1, dynamically linked and unmodified), and both binaries are linked with
`-static-libstdc++ -static-libgcc`, which statically links the GCC runtime
libraries (GPL-3.0 **with** the GCC Runtime Library Exception, which is what
permits exactly this).

## Requirements on the node

- NVIDIA Container Toolkit **≥ 1.17** with the `nvidia` runtime registered
  (`nvidia-ctk runtime configure --runtime=docker`). Without it the container
  cannot start and the runtime reports a bare `exit status 125`.
- `NVIDIA_DRIVER_CAPABILITIES` must include **`utility`** as well as `compute`
  (the image sets both). nvbandwidth links NVML; with `compute` alone the
  container starts and then dies on a missing `libnvidia-ml.so.1`.
- Enough free device memory for the copy buffer — 512 MiB by default, on both
  ends of a device-to-device copy.

## Verified on hardware

Built and run on a GB10 DGX Spark (aarch64, driver 580.82.09, CUDA 13.0) on
2026-08-02. Full stdout of `docker run --rm --gpus all …`, exit `0`:

```
nvbandwidth_ref=v0.10.0
gpu_name=NVIDIA GB10
gpu_count=1   # captured before v0.7; the runner now prints devices_visible
cuda_driver_version=13000
transfer_size_bytes=536870912
test_samples=3
h2d_bandwidth_gbs=59.11
hostToDeviceBandwidthMaxGBs=59.11
d2h_bandwidth_gbs=59.07
deviceToHostBandwidthMaxGBs=59.07
d2d_bandwidth_gbs=121.87
deviceToDeviceBandwidthMaxGBs=121.87
elapsed_s=3.1
MEMORY_BW_PASS
```

The figures are what a coherent C2C part should give: host↔device is symmetric
(59.11 / 59.07 GB/s) because there is no PCIe asymmetry to create, and the
on-device copy is roughly double it. The `DT_NEEDED` assertion passed with only
`libcuda.so.1` and `libnvidia-ml.so.1` recorded, confirming
`CMAKE_CUDA_RUNTIME_LIBRARY=Static` keeps `libcudart` out of the shipped binary.

This was a single-GPU node, so the `…Max…` metrics equal their averages by
construction. Note the build used the default (empty) `CUDA_ARCH`; as described
above, that changes nothing for these three copy-engine testcases.

## Still unverified

Before this is published by the `publish-runner` workflow, and before an entry
for `memory-bw` is added to `defaultRunnerImages` in
[`internal/controller/pods.go`](../../internal/controller/pods.go):

1. **A multi-GPU node**, where the min and max metrics should diverge and the
   matrix scan has more than one cell to read per testcase.
2. **A node with no accelerator exits 2, not 1.** The skip path is the one that
   matters most and is the one a single healthy Spark cannot exercise.
3. ~~**`scan_test.cc` is still not run by CI**~~ — closed:
   `runners/cxxtests_test.go` compiles and runs it under `make test`.
