# `memory-stress` runner — stressapptest

Runner image for the `memory-stress` [`TestKind`](../../api/v1alpha1/burnintest_types.go):
a host memory stress test that proves the node's DIMMs return what was written
to them, and reports how fast they did it.

The workload is [stressapptest](https://github.com/stressapptest/stressapptest)
(Apache-2.0), the tool this project sanctions for memory stress. The obvious
alternatives — stress-ng, fio, memtester — are GPL and cannot ship in an image
this project publishes.

Host DIMM faults present as accelerator flakiness: a corrupted staging buffer
becomes a wrong answer that looks like a bad GPU. Accepting a node means
accepting its host memory too.

## On GB10 this is also the unified-memory test

On a Grace-Blackwell superchip (GB10 / DGX Spark) the CPU and the GPU share one
physical LPDDR5x pool over NVLink-C2C. **Device memory is host memory.** A
miscompare found here is a miscompare in the same DRAM the accelerator computes
from, which is why `memoryErrors == 0` is an acceptance gate on that platform
rather than a host-side nicety.

Two limits on that claim, both deliberate:

- It exercises the DRAM through the **CPU's** path — cores, caches, memory
  controller. A fault confined to the GPU's C2C access path is not covered here;
  `memory-bw` and `gpu-burn` are what cover that.
- The pool is shared, so the region this test allocates is memory the
  accelerator cannot use while the test runs. Do not schedule it concurrently
  with a GPU memory test on the same node, and give the pod a memory limit —
  the runner sizes its workload from that limit (see *Sizing*).

The runner requests no GPU and links no CUDA. On GB10 the accelerator sits idle
while its own memory is under test.

## Output contract

| Result | stdout | Exit |
|--------|--------|------|
| Pass | `STRESSAPPTEST_PASS` | 0 |
| Fail | `STRESSAPPTEST_FAIL: <reason>` | 1 |
| Not applicable to this node | `STRESSAPPTEST_SKIP: <reason>` | 2 |
| Unjudged (runner or configuration fault) | `STRESSAPPTEST_ERROR: <reason>` | 3 |

Metrics are `key=value` lines on stdout and are **always printed before the
decision**, so a failing, interrupted or erroring run still yields its evidence.

| Runner key | Canonical metric | Meaning |
|------------|------------------|---------|
| `hardware_incidents` | `memoryErrors` | hardware incidents stressapptest attributed to memory. **The acceptance gate.** |
| `miscompares` | `miscompares` | every value that did not match what was written |
| `sdc_count` | `sdcDetections` | distinct `Hardware Error:` incidents logged during the run |
| `read_bandwidth_mbs` | `readBandwidthMBs` | sustained read throughput (see *Bandwidth*) |
| `write_bandwidth_mbs` | `writeBandwidthMBs` | sustained write throughput |
| `elapsed_s` | `elapsedS` | wall clock of the test body, allocation and pattern fill included |
| `sat_run_s` | `satRunS` | stressapptest's own run time, which excludes that setup |
| `duration_requested_s` | `durationRequestedS` | what `BURNIN_DURATION_SECONDS` asked for |
| `memory_covered_pct` | `memoryCoveredPct` | the region tested, as a percentage of the node's **total** RAM |
| `threads` | `threads` | memory copy threads used |
| `tool`, `tool_version` | `tool`, `toolVersion` | `stressapptest` and the exact upstream ref the image was built from |
| `sat_status` | `satStatus` | the tool's own final status: `pass`, `fail-hardware` or `fail-procedural` |

`elapsed_s` is re-emitted every 60 seconds while the test runs. The operator's
parser is last-occurrence-wins, so a soak that is evicted or cancelled at hour
three still reports how far it actually got via `spec.checkpointIntervalSeconds`
rather than reporting nothing at all.

Two metrics that are deliberately **not** emitted:

- **The absolute size of the tested region.** The metric grammar in
  `pkg/contract` registers no capacity unit, so a `...Mb` name would be a
  physical quantity recorded as a dimensionless number — the exact trap the
  grammar exists to prevent. `memoryCoveredPct` carries the portable form, and
  the absolute figure is in the runner's sizing line in the log. Registering a
  capacity unit in `pkg/contract` is what it would take to emit it properly.
- **`iterationsCompleted`**, which the `TestKind`'s doc comment mentions.
  stressapptest has no iteration counter; it reports volume moved and time
  elapsed. Synthesising one would produce a number a profile could threshold on
  that measures nothing.

## What "pass" actually means

Exit 0 requires **all** of: stressapptest printed a completion summary, its own
status was `PASS`, and every corruption counter was zero. A run that was killed,
timed out, or produced no summary reports **Error**, never Pass — and its
counters are omitted rather than reported as zero. A test that stopped early
does not know that it found no errors; it only knows it stopped looking, and
`hardware_incidents=0` would say the stronger thing.

### `Error` is not `Fail`

stressapptest exits `1` for two entirely different outcomes, and its own status
line is the only thing that separates them:

| stressapptest says | Means | This runner |
|--------------------|-------|-------------|
| `Status: FAIL - test discovered HW problems` | the memory returned wrong data | **Fail**, exit 1 |
| `Status: FAIL - test encountered procedural errors` | it could not run the test (allocation failed, bad arguments) | **Error**, exit 3 |

Forwarding the tool's exit code would collapse the two and condemn hardware
nobody successfully tested. The runner classifies instead, in this order:

1. **Positive evidence of corruption wins.** A non-zero incident, miscompare or
   logged `Hardware Error:` is a verdict about the DIMMs even if the run was
   later killed. Memory that returned a wrong value returned a wrong value.
2. **Anything else that is not a clean, completed `PASS` is an Error** —
   unjudged — and never a Fail. A runner that could not measure has not found
   bad hardware; it has found nothing.

A `SIGTERM` (the kubelet's deadline kill, a cancelled run) is an Error too. The
entrypoint deliberately does **not** exit 0 on a signal: an entrypoint that
does turns a deadline kill into a passing verdict for untested hardware.

### Skip is for the hardware, not for the configuration

Exit 2 means "this test does not apply to this node", and the runner is careful
about which side of that line a small memory budget falls on:

- **No cgroup memory limit and the machine itself is too small** to hold the
  minimum stress region → **Skip**. Nobody misconfigured anything; the node is
  what it is.
- **A cgroup memory limit leaves too little** → **Error**, naming the knob
  (`spec.resources.limits.memory`, or `BURNIN_MEMORY_MB`). Somebody configured
  this pod, and the node's memory was never looked at. Reporting it as a Skip
  would quietly excuse a misconfigured profile from ever testing memory.

## Sizing

Unless `BURNIN_MEMORY_MB` says otherwise, the region is 70% of the smaller of
the cgroup memory limit and `MemAvailable`. The remaining 30% is headroom: a
container that allocates its whole limit is OOM-killed, which costs the run its
time and reports an infrastructure Error rather than a verdict.

`memoryCoveredPct` is what tells a reader how much a pass is worth — a clean run
that touched 4% of a node's RAM is a much weaker statement than one that touched
70%, and nothing else in the metric set distinguishes them.

The copy-thread count defaults to the usable CPU count, clamped by the cgroup
CPU quota: threads beyond the quota add scheduling latency rather than memory
traffic, and would depress the bandwidth figures for a reason that has nothing
to do with the DIMMs.

## Bandwidth

stressapptest reports throughput per **thread class**, not per direction, so the
two directions are derived from what each class does to memory:

| Thread class | Reads | Writes |
|--------------|-------|--------|
| memory copy (`-m`) | yes | yes |
| data check (`-c`) | yes | — |
| invert data (`-i`) | yes | yes |

`readBandwidthMBs` = copy + check + invert; `writeBandwidthMBs` = copy + invert.

With the default thread mix (copy threads only) **the two figures are equal by
construction** — a memcpy moves the same number of bytes in each direction.
That is a property of the workload, not a parser bug. They are still reported
separately because the invert and check threads a profile can enable do pull
them apart, and because a consumer charting a fleet wants the same two series
from every node regardless of thread mix.

Both figures are stressapptest's own MB/s, where "MB" is 2²⁰ bytes — the
registry defines these metrics as what the host memory stress tool reports, so
they are passed through unconverted.

## Thresholds

```yaml
thresholds:
  - metric: memoryErrors # the gate this kind exists for
    comparison: Equal
    value: "0"
  - metric: miscompares
    comparison: Equal
    value: "0"
  - metric: sdcDetections
    comparison: Equal
    value: "0"
  - metric: memoryCoveredPct # a pass over 4% of RAM is not an acceptance
    comparison: GreaterThanOrEqual
    value: "50"
```

These thresholds interlock with the fail-closed rule in `pkg/verdict`, which
treats a threshold naming a metric the runner did not emit as a **failure**.
A run that stopped early exits 3 and is an Error, so its thresholds are never
reached; a run that exits 0 has, by construction, emitted every counter above.
There is no path on which `memoryErrors == 0` is satisfied by silence.

Threshold bandwidth only with a **pinned** configuration (fixed
`BURNIN_MEMORY_THREADS`, `BURNIN_MEMORY_MB` and thread mix). The figures move
with thread count, region size and `-W`, so a fleet-wide gate on an auto-sized
run compares nodes that ran different workloads.

`elapsedS` is slightly **under** `BURNIN_DURATION_SECONDS` by design (see
*Duration*), so gate it at `>= 0.9 × duration` rather than at the exact value.

## Duration

`BURNIN_DURATION_SECONDS` is the whole budget for the test body. stressapptest's
`-s` clock starts only after it has allocated and pattern-filled the region,
which on a large region takes tens of seconds, so the runner passes it a
slightly smaller `-s` (10% of the budget, held between 15 and 60 seconds). That
keeps the container inside the pod's `activeDeadlineSeconds`; overrunning it
means a kubelet kill, which the operator records as an Error for every node in
the run.

For the same reason the runner enforces its own cap at
`BURNIN_DURATION_SECONDS + 60s`: ending a wedged run **itself**, inside the
pod's grace period, is what lets the metrics gathered so far reach stdout. A
kubelet deadline kill takes the log's last words with it.

## Configuration

All optional. Set them through `spec.runner.env`.

| Variable | Default | Meaning |
|----------|---------|---------|
| `BURNIN_DURATION_SECONDS` | `600` | total budget; set by the operator for every runner |
| `BURNIN_MEMORY_MB` | auto | explicit region size in MiB; bypasses auto-sizing **and** the minimum-region floor |
| `BURNIN_MEMORY_FRACTION` | `0.70` | share of the usable budget to test when auto-sizing |
| `BURNIN_MEMORY_MIN_MB` | `256` | smallest region worth calling a memory test |
| `BURNIN_MEMORY_THREADS` | usable CPUs | memory copy threads (`-m`) |
| `BURNIN_MEMORY_INVERT_THREADS` | `0` | invert threads (`-i`) |
| `BURNIN_MEMORY_CHECK_THREADS` | `0` | check-only threads (`-c`) |
| `BURNIN_MEMORY_CPU_THREADS` | `0` | CPU-stress threads (`-C`) |
| `BURNIN_MEMORY_STRESSFUL_COPY` | `0` | pass `-W`: a more CPU-stressful copy. Stronger stress, lower bandwidth figures — worth enabling for long soaks, but it changes what the bandwidth metrics mean |
| `BURNIN_MEMORY_VERBOSITY` | `8` | stressapptest `-v` |
| `BURNIN_MEMORY_EXTRA_ARGS` | — | appended verbatim to the command line; unvalidated. `--force_errors` injects fake errors and will produce a **Fail** |
| `BURNIN_STRESSAPPTEST_BIN` | `/usr/local/bin/stressapptest` | tool path; exists for tests |

A malformed value is refused (exit 3) rather than replaced by its default.
`BURNIN_MEMORY_FRACTION=70` meaning "70%" must fail at the first run, not
silently accept a fleet on a workload nobody chose.

## Build

```sh
docker build -t glimmer-burnin-memory-stress:dev .
```

Build args: `STRESSAPPTEST_REF` (default `v1.0.11`), `STRESSAPPTEST_SHA`,
`SAT_BUILD_IMAGE`, `GO_IMAGE`, `RUNTIME_IMAGE`.

The build **asserts the upstream commit**: a tag can be moved, and a moved tag
would silently change every node's memory tester. Bumping `STRESSAPPTEST_REF`
means bumping `STRESSAPPTEST_SHA` with it, or the build refuses.

`v1.0.11` is the floor: it is the first release with aarch64 vector-instruction
support, and GB10 is arm64.

The wrapper's source belongs to the repository's **root Go module**, so CI's
`go build ./...`, `go vet ./...` and `go test ./...` cover it like any other
package — and its tests can import `pkg/contract` and `pkg/runner` to prove the
emitted keys really do normalise to the canonical names a threshold is written
against (`metricnames_test.go`). The `publish-runner` workflow builds with
`context: runners/<name>`, which cannot see the root `go.mod`, so the image
build strips the test files and synthesises a throwaway module. Nothing is
fetched: the non-test sources import the standard library only, and `GOPROXY=off`
in the build stage turns that into an assertion.

## Licensing

| Component | Licence | How it ships |
|-----------|---------|--------------|
| this runner's source | Apache-2.0 | compiled into the wrapper binary |
| [stressapptest](https://github.com/stressapptest/stressapptest) | Apache-2.0 | built from source; binary shipped, with `COPYING` and `NOTICE` at `/licenses` in the image |
| Go standard library | BSD-3-Clause | linked into the wrapper |
| libstdc++ / libgcc | GPL-3.0 **WITH** GCC-exception | statically linked into stressapptest — the arrangement the GCC Runtime Library Exception exists to permit, and the one `compute-smoke` already ships |
| glibc (distroless base) | LGPL-2.1-or-later | dynamically linked, unmodified, supplied by the base image |

No component is under a copyleft licence without an exception that permits this
use, and no GPL **tool** ships: stress-ng, fio, IOR, elbencho and memtester are
all excluded on licence grounds, which is why stressapptest is the sanctioned
choice.

The image contains **no NVIDIA-licensed redistributable libraries**, and the
build asserts it: it fails if `ldd` finds any `libcuda*`/`libcudart*`/`libnv*`
dependency in either binary. This runner never opens the driver at all.

The Apache-2.0 licence check is also a build-time assertion — the build fails if
upstream's `COPYING` stops being Apache-2.0.

## Requirements on the node

- Linux with `/proc/meminfo`. The runner sizes its workload from it and reports
  an Error if it cannot read it, rather than guessing.
- No capabilities, no privileged mode, no GPU. The container runs as
  `65532:65532`.
- A memory limit on the pod is recommended, especially on unified-memory parts:
  it is what bounds the region and keeps the test from competing with the rest
  of the node.
- Without `CAP_SYS_ADMIN` the tool cannot read `/proc/self/pagemap`, so a
  miscompare is reported without its physical address. The incident **counts**
  are unaffected, and the counts are what the verdict is made of; the physical
  address only matters once a human is picking which DIMM to pull.

## Not yet verified

- **No container build has been run against this Dockerfile** — there was no
  container daemon available where it was written. The upstream ref, commit,
  licence and command-line flags were verified against the upstream repository,
  but the compile, the `ldd` assertions and the distroless runtime have not been
  executed.
- **Never run against real hardware.** The output shapes the parser expects were
  taken from stressapptest's own format strings, not from a live capture.
- The base images are pinned by tag, not by digest. Pin digests before the first
  publish; the publish workflow records the resulting image digest, but not the
  bases it was built from.
- `memory-stress` has no entry in the operator's `defaultRunnerImages`
  (`internal/controller/pods.go`) and is not in the `publish-runner` workflow's
  runner list. Both are deliberate: an entry pointing at a tag nobody has pushed
  is an ImagePullBackOff on every node. Add them when this image has been built,
  verified on real hardware, and published.
