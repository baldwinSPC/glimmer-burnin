# NVIDIA

The reference implementation, and the only vendor whose runners have been
executed on real silicon. Everything else in `docs/vendors/` is measured against
what is written here.

## Coverage

Every built-in `TestKind` appears here. The state vocabulary — shipped,
verified, in tree, planned — is defined once in
[the roadmap](../roadmap.md#the-test-matrix); this table does not invent its own.

| Kind | Measures | State |
|---|---|---|
| `compute-smoke` | FP4 block-scaled GEMM, exact instruction path | shipped (burst-only) |
| `gemm-sweep` | GEMM across FP64→FP4/INT8, one precision per cell | **verified** on GB10; publish [#350] |
| `gpu-burn` | sustained compute, SDC detection | shipped |
| `thermal-soak` | power and thermal behaviour, throttling | shipped |
| `clockprobe` | sustained clock under load | shipped |
| `memory-bw` | host↔device and device-local bandwidth | shipped; p2p cases in tree, verify [#279] |
| `host-health` | Xid, ECC, PCIe AER, NIC counters | shipped |
| `dcgm-diag` | vendor diagnostic suite | shipped, **site-supplied image** |
| `gpudirect-rdma` | GPU↔NIC peer-memory path | shipped |
| `nccl` | collective bandwidth (Pair and Group) | shipped |
| `memory-stress`, `ib-write-bw` | host memory, RDMA wire | shipped (vendor-free) |
| `fabric-soak` | iterated RDMA writes over hours | in tree (vendor-free), verify [#283] |
| `tcp-baseline` | plain TCP throughput and retransmits | in tree (vendor-free), verify [#237] |
| `disk-io` | storage throughput and latency (direct I/O) | in tree (vendor-free), verify [#242] |
| `fingerprint-probe` | what the hardware says about itself | in tree (vendor-free), verify [#354] |

Planned, not written: `power-swing` ([#166]).

[#166]: https://github.com/baldwinSPC/glimmer-burnin/issues/166
[#237]: https://github.com/baldwinSPC/glimmer-burnin/issues/237
[#242]: https://github.com/baldwinSPC/glimmer-burnin/issues/242
[#279]: https://github.com/baldwinSPC/glimmer-burnin/issues/279
[#283]: https://github.com/baldwinSPC/glimmer-burnin/issues/283
[#350]: https://github.com/baldwinSPC/glimmer-burnin/issues/350
[#354]: https://github.com/baldwinSPC/glimmer-burnin/issues/354

## What has to be on the node

- **A driver.** The runner images ship **no NVIDIA redistributable libraries** —
  the builds are multi-stage and fail if `ldd` finds `libcuda`/`libcudart`/`libnv`
  in the final binary. `libcuda.so.1` is injected at runtime from the host by the
  NVIDIA Container Toolkit.
- **Working driver injection.** The images declare both `NVIDIA_VISIBLE_DEVICES`
  and `NVIDIA_DRIVER_CAPABILITIES`, because on hosts where the legacy
  `nvidia-container-runtime` is the injection path, that image environment is the
  only thing that gets the driver in. A CI test asserts every accelerator runner
  declares both.
- **A device plugin**, advertising `nvidia.com/gpu`. Request it in
  `spec.resources` exactly as any workload would.

## Arch targets are deliberate

`compute-smoke` compiles for `sm_121a` on arm64 (GB10 / DGX Spark) and `sm_120a`
on amd64. An `a` target emits a cubin for that capability **only**, with no PTX
fallback — which is the proof the runner offers: a pass means the real
block-scaled MMA instruction path executed, not something JIT-compiled or
emulated.

Do not widen a gencode list to make a build succeed. A widened target turns a
proof into a plausible number.

Compute capabilities outside `12.x` are reported as a **skip**, not a failure —
the kernel does not exist for them, which is a fact about the part and not a
fault in it.

## dcgm-diag: site-supplied

DCGM is not redistributed by this project. The kind exists, and you point it at
your own image containing `dcgmi`:

```yaml
spec:
  kind: dcgm-diag
  runner:
    image: registry.example.com/our-dcgm:4.2.3
```

The same pattern any proprietary suite uses — see the
[overview](README.md#pointing-a-kind-at-your-own-image).

## Not a runner: tests that need the driver unloaded or a reboot

Some vendor tests cannot run inside either dispatcher this project has, at
any scope. Two examples: **NVIDIA Field Diagnostic** and its DGX Spark
packaging, **`dgx-spark-fieldiag`** (publicly installable on GB10; `sudo
./partnerdiag --field`; a PASS/FAIL banner; exit 0 pass, 1 error, 2 retest),
run with the **GPU driver unloaded**. **memtest86+** needs a **reboot**.

Both `internal/controller.podForTest` and `pkg/localrun.ContainerRuntime`
(`runtime.go`) are container executors: they start one image, wait for it to
exit, and read one exit code. A tool that unloads the driver, or that the
kernel itself has to reboot into, cannot produce "one attempt = one exit
code" inside either — there is no container left running to report one.

A `TestKind.HostOnly()` alongside `BurstOnly()`, refused at plan time the way
`validateSoak` already refuses a segmented burst kind, would be a small
change to declare. What it would need behind it is not: a runner directory
(`TestEveryRunnerDirectoryNamesARealKind` in `runners/pins_test.go` requires
one), and a `HostRuntime` implementing `ContainerRuntime` that can drive a
process across a driver reload or a reboot — a different product, closer to
a provisioning tool than a test dispatcher.

**Decision: docs only.** Run these as a **provisioning pre-step**, outside
either dispatcher, and attach the tool's own log as an **artifact** of a
`custom` test — the same fenced-block mechanism any evidence uses
(`docs/runner-contract.md`):

```
-----BEGIN BURNIN ARTIFACT fielddiag.log text/plain-----
<the tool's own output, verbatim>
-----END BURNIN ARTIFACT-----
```

That gets the evidence into the same envelope and the same report as
everything else this operator measures, without pretending a driver-unload
tool is a container workload. It is not a verdict — nothing thresholds an
artifact — but it rides beside the verdicts that ARE measured, on the same
node, in the same run.

Revisit only if the CLI is ever asked to orchestrate a reboot itself; that is
a materially different tool than the one described above, and until it
exists this is the honest answer. Tracked in [#422] so nobody re-derives it.

[#422]: https://github.com/baldwinSPC/glimmer-burnin/issues/422

## GB10 (DGX Spark): platform facts

These were measured on two DGX Spark units by an independent GB10 acceptance
effort, and are recorded here because each one is a trap for a runner or a
threshold. Where this project's own fleet has confirmed one, it says so. The
others are not yet re-measured here, and a number below is a **seed for
[#353], not a pin**: working-set sizes and methods differ from this project's
runners.

**What the part does not expose**

- `nvidia-smi --query-gpu` never populates `clocks.mem`,
  `ecc.errors.corrected.volatile.total` or
  `ecc.errors.uncorrected.volatile.total`; they are null on every sample.
  Consistent with this project's own finding that GB10 exposes no ECC to NVML,
  which is why `host-health` declares `eccErrors=n/a`.
- There is no `nvpmodel`-style discrete power mode to look up. The power and
  clock state a number was measured under can only be READ, at the time, which
  is what [#544] is for.
- Per-rail power, energy counters and the PL1/PL2 caps are visible only through
  a third-party hwmon driver. `nvidia-smi` power is GPU-only. The widely quoted
  240 W is the external supply's rating against a SoC TDP of roughly 140 W.

**Telemetry that lies**

- GPU temperature reads misleadingly **low** at low power states, and a
  firmware/EC-dependent bug pins the driver's own `T.Limit Temp` near a bogus
  50 °C on some units. Never derive a thermal band from driver-reported
  thresholds, and treat an implausibly low reading as a fault mode rather than
  a healthy idle. Idle is about 41 °C.
- nvme-cli `smart-log -o json` reports `temperature` in **Kelvin** (314 for a
  drive at about 41 °C).
- The GHES firmware-first driver prints `[Hardware Error]:` on every line of
  every record, **corrected ones included**. The authoritative field is
  `event severity: fatal|recoverable|corrected`. A scan keyed on the prefix
  counts handled events as faults ([#538]).
- `timedatectl` has no real `SystemClockSynchronized` property; it returns
  empty on healthy nodes. `NTPSynchronized` is the signal.
- The kernel prints an Xid's device as `PCI:0000:01:00`, while `nvidia-smi`
  reports `00000000:01:00.0`. Attributing Xids per device needs both forms.

**CPU and memory topology**

- 10× Cortex-X925 (performance) plus 10× Cortex-A725 (efficiency).
  `cpu_capacity` separates them cleanly: A725 reads 718–731, X925 997–1024,
  with the maximum on cpu19. **cpu0 is not a performance core**, and the two
  clusters are separate cpufreq policies that can carry different governors.
- L3 is two asymmetric instances: 8 MiB for cpus 0–9, 16 MiB for cpus 10–19.
  A STREAM array must exceed the larger one.
- CPU and GPU share one 128 GB LPDDR5X pool. Host and device load on the same
  part interact, the same class of effect [#534] measured on an AMD APU; never
  overlap a CPU measurement with a GPU one.

**Reference numbers (seeds, not pins)**

| Quantity | Reported value | Conditions |
|---|---|---|
| bf16 GEMM | 60–65 TFLOP/s | square 8192, TF32 and reduced-precision paths off |
| device-memory triad | 228–231 GB/s | working set 0.1 of free memory |
| managed-memory triad | 163–165 GB/s | same; `cudaMallocManaged` ([#543]) |
| host STREAM triad | 100–109 GB/s | 64 MiB arrays, all cores |
| CPU FMA roofline | 15.6 GFLOP/s one X925 core; about 103 all 20 | double, 8 chains ([#542]) |
| NVMe (Samsung MZALC4T0HBL1) | seq write 12.1 GB/s, seq read 14.9–15.0, 4 KiB randread 505k–530k IOPS, p99 82–87 µs | direct I/O, 32 in flight ([#545]) |
| load ramp | utilization 0→96 %, power 4.3→80.1 W, SM clock 208→2450 MHz | first to mid-run sample of a training load |

This project's own measured fabric and collective numbers live under
[Thresholds](#thresholds) and in [#353].

[#353]: https://github.com/baldwinSPC/glimmer-burnin/issues/353
[#534]: https://github.com/baldwinSPC/glimmer-burnin/issues/534
[#538]: https://github.com/baldwinSPC/glimmer-burnin/issues/538
[#542]: https://github.com/baldwinSPC/glimmer-burnin/issues/542
[#543]: https://github.com/baldwinSPC/glimmer-burnin/issues/543
[#544]: https://github.com/baldwinSPC/glimmer-burnin/issues/544
[#545]: https://github.com/baldwinSPC/glimmer-burnin/issues/545

## Thresholds

Measured, never from a datasheet. The fleet this project was built against
sustains roughly 99.6 Gb/s on a fabric whose optics are labelled 200; a
threshold from the label fails every healthy node and reads as a hardware
verdict.

Two NVIDIA-specific traps:

- **`eccErrors` may be genuinely unmeasurable.** GB10 exposes no ECC to NVML.
  `host-health` reports `eccErrors=n/a` there — a positive declaration that the
  part has nothing to report, distinct from a probe that failed. Gate it with
  `applicability: RequiredIfMeasurable` and the threshold is reported as NOT
  EVALUATED rather than failing every node in the fleet.
- **`throughputTflops` from `compute-smoke` is evidence, not acceptance.** It is
  a single unwarmed launch — a liveness signal. `pkg/verdict.ValidateThresholds`
  will tell you so at authoring time.
