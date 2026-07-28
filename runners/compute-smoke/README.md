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
| Wrong architecture | `FP4_GEMM_SKIP: <reason>` | 2 |

Metrics are emitted as `key=value` lines for threshold evaluation:
`gpu_name`, `compute_cap`, `m`, `n`, `k`, `max_abs_ref`, `max_abs_error`,
`max_rel_error`, `nonfinite_count`, `elapsed_ms`, `tflops`.

`tflops` is a single un-warmed 1024³ launch — it is a liveness signal, **not** a
benchmark. The problem is far too small to saturate a GB10. Do not threshold on it.

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
