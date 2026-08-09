# NVIDIA

The reference implementation, and the only vendor whose runners have been
executed on real silicon. Everything else in `docs/vendors/` is measured against
what is written here.

## Coverage

| Kind | Measures | State |
|---|---|---|
| `compute-smoke` | FP4 block-scaled GEMM, exact instruction path | shipped (burst-only) |
| `gpu-burn` | sustained compute, SDC detection | shipped |
| `thermal-soak` | power and thermal behaviour, throttling | shipped |
| `clockprobe` | sustained clock under load | shipped |
| `memory-bw` | host↔device and device-local bandwidth | shipped |
| `host-health` | Xid, ECC, PCIe AER, NIC counters | shipped |
| `dcgm-diag` | vendor diagnostic suite | shipped, **site-supplied image** |
| `gpudirect-rdma` | GPU↔NIC peer-memory path | shipped |
| `nccl` | collective bandwidth (Pair and Group) | shipped |
| `memory-stress`, `ib-write-bw` | host memory, RDMA wire | shipped (vendor-free) |
| `tcp-baseline`, `disk-io` | TCP path, storage | in tree (vendor-free) |

Planned, not written: `gemm-sweep`, `power-swing`, `fabric-soak`, p2p cases.

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
