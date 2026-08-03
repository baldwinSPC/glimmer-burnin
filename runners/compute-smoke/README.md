# `compute-smoke` runner — NVIDIA Blackwell NVFP4

Runner image for the `compute-smoke` [`TestKind`](../../api/v1alpha1/burnintest_types.go):
a fast, architecture-correct kernel that proves the accelerator's tensor cores
actually compute — not merely that a container started.

This variant targets **NVIDIA Blackwell SM120/SM121** (GB10 / DGX Spark, compute
capability 12.0–12.1) and exercises **NVFP4** block-scaled GEMM.

> SM10x (B200) block-scaled kernels use a different instruction path
> (`tcgen05.mma`, TMEM accumulation) and are **not** interchangeable with SM12x
> (warp-level `mma.sync` / `mxf4nvf4`). A separate runner is needed for those cards.

## Output contract

| Result | stdout | Exit |
|--------|--------|------|
| Pass | `FP4_GEMM_PASS` | 0 |
| Fail | `FP4_GEMM_FAIL: <reason>` | 1 |
| Not applicable | `FP4_GEMM_SKIP: <reason>` | 2 |
| Unjudged | `FP4_GEMM_ERROR: <reason>` | 3 |

Metrics are emitted as `key=value` lines for threshold evaluation:
`built_cuda_arch`, `gpu_name`, `compute_cap`, `m`, `n`, `k`, `max_abs_ref`,
`max_abs_error`, `max_rel_error`, `nonfinite_count`, `elapsed_ms`, `tflops`.

`built_cuda_arch`, `gpu_name` and `compute_cap` are printed **first**, before
any gate, so a run that skips or errors still records which image met which
part. `tflops` is a single un-warmed 1024³ launch — it is a liveness signal,
**not** a benchmark. The problem is far too small to saturate a GB10. Do not
threshold on it.

### Fail (exit 1) — the part was measured and it is wrong

Only three things reach exit 1, and all three are properties of the silicon:

- NaN/Inf in the device output (`nonfinite_count > 0`).
- An all-zero device output.
- `max_rel_error` above the tolerance.

Exit 1 is the expensive code. A `Fail` is **never retried** by the operator —
re-running a measurement until it comes out clean would launder a hardware fault
into an acceptance — so it settles the test and permanently indicts the node.
Nothing that merely *prevented* a measurement may use it.

### Error (exit 3) — we could not measure, so the part is unjudged

- **No usable CUDA device.** A pod that got no GPU has established nothing about
  the node's tensor cores.
- **This image carries no cubin for the part it landed on**
  (`cudaErrorNoKernelImageForDevice` / `cudaErrorInvalidDeviceFunction`), or the
  binary was built with no SM120/SM121 block-scaled MMA path at all. Both are
  statements about the image that was pinned, not about the hardware. The error
  message names `built_cuda_arch` and the fix.
- **Any CUDA runtime or driver error**, and any failure to set up the
  measurement — allocation, `can_implement`, `initialize`, a launch, a
  synchronise.
- **The host reference GEMM came out all zeros.** The reference is computed
  entirely on the host by CUTLASS's host GETT; the device never touches it. A
  zero there means our own yardstick is degenerate (and `max_rel_error` is
  `INFINITY` by construction), so the comparison measured nothing.

### The compute-capability gate, and the arch the binary was built for

These are deliberately **two different questions**, and this runner answers them
separately:

| question | about | outcome |
|---|---|---|
| Can this part be asked to do NVFP4 block-scaled GEMM at all? | the hardware | not CC 12.0/12.1 → **Skip** (exit 2) |
| Does *this image* carry a cubin for it? | the image that was pinned | no → **Error** (exit 3) |

The gate admits the whole `12.0`/`12.1` family, while the default build pins
`sm_121a` (CC 12.1 only). So a **CC 12.0 part is in scope, passes the gate, and
then finds no kernel image** — which is why that path is an `Error` and not a
`Fail`. Before this was fixed it exited 1, recording a hardware verdict against
a node whose tensor cores were never exercised.

It is `Error` rather than `Skip` on purpose. Narrowing the gate to the built arch
would report a whole fleet as "not applicable" when the truth is that the
operator pinned the wrong tag; `Error` is the retryable, hardware-unjudged phase
and it names the fix. It also cannot be decided at build time: nvcc's
`__CUDA_ARCH_LIST__` records `1200` for both `sm_120a` and `sm_120f` and only the
`f` form also covers 12.1, so no compile-time macro can answer whether this
binary runs on the part in front of it. Where we cannot know, we say so.

Build with `CUDA_ARCH=sm_120f` for one binary covering CC 12.0 + 12.1.

> **Image tags.** The published `v0.1.0` image predates this fix and reports all
> of the above as exit 1. Published tags are immutable and it is not being
> changed; the corrected contract ships under a **new tag**. A gate still pinning
> `v0.1.0` keeps the old, wrong behaviour until it is repinned.

## What "pass" actually means

The check is deliberately not vacuous — a kernel that launches and returns
garbage must fail:

- The result is compared against CUTLASS's **independent** host block-scaled GETT
  reference computed in fp32, not against itself.
- Tolerance is `max_rel_error <= 0.01`, normalised by `max|ref|`. On real hardware
  the clean floor is ~0.002, so the gate sits ~5× above the noise while still
  catching a single corrupted scale factor (verified: perturbing one `SFA` element
  yields `max_rel_error=0.446` → FAIL).
- NaN/Inf and all-zero outputs are rejected explicitly.

Verified on real hardware (NVIDIA GB10, driver 580.82.09, CUDA 13.0):
`cuobjdump -sass` confirms 128 `OMMA.SF.16864.F32.E2M1.E2M1.UE4M3.4X` instructions
— block-scaled MMA with E2M1 operands and UE4M3 scale factors — and the binary
embeds an `sm_121a` cubin only, so there is no PTX-JIT or emulation fallback that
could produce a passing result without the FP4 units.

## Build

```sh
docker build -t glimmer-burnin-compute-smoke:dev .
```

Build args: `CUDA_IMAGE`, `CUTLASS_REF` (default `v4.6.1`), `CUDA_ARCH`
(default `sm_121a`; use `sm_120f` for one binary covering CC 12.0 + 12.1).

CUTLASS **v4.5.0 is the floor** — it is the first release whose changelog reports
block-scaled MMA working on Spark. Earlier versions mis-handle SM121.

## Licensing

Our source is **Apache-2.0**. [CUTLASS](https://github.com/NVIDIA/cutlass) is
**BSD-3-Clause** and header-only: it compiles into the binary, and no CUTLASS
source is redistributed here.

The build is multi-stage specifically so the CUDA toolchain is used at **build
time only**. The published image contains our binary on a distroless base and
**no NVIDIA-licensed redistributable libraries** — `libcuda.so.1` is injected at
runtime from the host driver by the NVIDIA Container Toolkit. The build asserts
this: it fails if `ldd` finds any `libcuda*`/`libcudart*`/`libnv*` dependency.

## Requirements on the node

- NVIDIA Container Toolkit **≥ 1.17** with the `nvidia` runtime registered
  (`nvidia-ctk runtime configure --runtime=docker`). Without it the container
  cannot start and the runtime reports a bare `exit status 125`.
- A GPU of compute capability 12.0 or 12.1.
