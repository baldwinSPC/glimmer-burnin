# cpu-bench

Host CPU compute and host memory bandwidth (#542). Vendor-free: no accelerator
is touched.

| Metric | What |
|---|---|
| `fmaSingleCoreGflops` | double FMA, 8 independent chains, on ONE core: the highest `cpu_capacity` core the pod may use |
| `fmaAllCoreGflops` | the same kernel on every allowed core, one pinned thread each |
| `hostStreamCopyGBs`, `…ScaleGBs`, `…AddGBs`, `…TriadGBs` | McCalpin's STREAM, one pinned thread per core, first-touch placement |
| `cpuBenchThreads`, `cpuBenchPerfCore`, `streamArrayBytes` | evidence: what it ran on and at |

Every figure is the median of its samples. `BURNIN_DURATION_SECONDS` (default
60, minimum 10) is split 20/30/50 between the three phases.

## Why these choices

Ported from an independent GB10 acceptance toolkit, which measured each trap:

- **The single-core figure picks its core by `cpu_capacity`.** On GB10, cpu0 is
  an efficiency core (Cortex-A725, capacity 718–731); the Cortex-X925 cores read
  997–1024. Measuring cpu0 would report the wrong cluster as the node's best.
- **STREAM arrays are 4× the largest single L3 instance**, not 4× the sum. GB10
  has two L3 instances (8 MiB and 16 MiB), so each array is 64 MiB.
- **A RAM headroom guard**: three arrays may use at most three quarters of the
  memory the pod can actually get (the smaller of MemTotal and its cgroup
  limit). Refused, STREAM's metrics are omitted and `streamStatus` says why.
- **CPUs come from the pod's affinity mask**, not the host's list, because a
  pod's cpuset can be smaller than the machine.
- **Plain threads, not OpenMP**, so the image carries no GPL runtime.
- **STREAM uses s = √2 − 1**, not the classic 3. With 3 the values grow every
  pass and overflow after a few hundred iterations; with √2 − 1 a full
  copy/scale/add/triad pass is a fixed point, so the value check stays exact
  however long the phase runs.

## Exit codes

| Code | Marker | Meaning |
|---|---|---|
| 0 | `CPU_BENCH_PASS` | measured |
| 1 | `CPU_BENCH_FAIL` | STREAM produced wrong values: host memory, or the core computing through it, is corrupting data. Metrics are printed before the verdict |
| 3 | `CPU_BENCH_ERROR` | could not measure (no CPU in the mask, an amd64 part without AVX2/FMA, allocation failed); unjudged |

There is no skip: every node has a CPU.

## Measured

spark-043a (GB10), 60 s, 2026-09-28:

```
cpuBenchThreads=20  cpuBenchPerfCore=19  streamArrayBytes=67108864
fmaSingleCoreGflops=15.60  fmaAllCoreGflops=93.45
hostStreamCopyGBs=144.79  hostStreamScaleGBs=137.50  hostStreamAddGBs=132.01  hostStreamTriadGBs=130.77
```

The independent toolkit reported 15.6 GFLOP/s for one X925 core. A throwaway
build that corrupted one array element exited 1 naming the element, which is how
the verification path was exercised on hardware.

**Never overlap this with a GPU test on an APU or a Grace part**: CPU load moves
the accelerator's clock (#534).

**amd64 is built but not yet run on hardware.** The image targets x86-64-v3 and
refuses, as an Error, a CPU without AVX2 and FMA.
