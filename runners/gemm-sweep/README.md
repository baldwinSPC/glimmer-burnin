# gemm-sweep

One GEMM, **one precision per execution**, selected by the `precision` variant
axis (`BURNIN_VARIANT_PRECISION`). compute-smoke proves the NVFP4 block-scaled
path executes and computes correctly; this runner asks the same question of the
rest of the compute surface. A part can pass FP4 and fail FP64 — tensor-core
paths and CUDA-core paths are different silicon and fail independently.

The sweep itself is the profile's variant list, not a loop in this binary: each
cell is an ordinary execution with its own verdict, its own retry budget and its
own (per-variant) thresholds. `config/samples/variant-sweep.yaml` is the worked
example.

## What each cell measures

| cell | silicon exercised | kernel |
|---|---|---|
| `fp4`  | block-scaled NVFP4 tensor cores | CUTLASS Sm120 block-scaled (as compute-smoke) |
| `fp8`  | dense e4m3 tensor cores, fp32 accumulate | CUTLASS Sm120 dense |
| `bf16` | `mma.sync` BF16 tensor cores | CUTLASS 2.x Sm80-vintage (native on every part this image targets) |
| `tf32` | `mma.sync` TF32 tensor cores | CUTLASS 2.x Sm80-vintage |
| `fp64` | **the CUDA cores** | CUTLASS SIMT |

`fp64` is SIMT on purpose: the SM12x consumer parts this image targets have no
FP64 tensor pipe, so the CUDA cores *are* the native double path — a tensor-op
double kernel would measure an emulation, not the part.

A **pass** claims: the cell's instruction path executed on this part and its
output matched an independent host reference computed in double precision from
the exact operand values the device saw. Operands are integer grid points, so
for `fp8`/`bf16`/`tf32`/`fp64` a healthy result matches the reference
**exactly** in any accumulation order; the tolerance (1e-6) is a bound, not a
budget. The `fp4` cell's output element is BF16, whose rounding is real, and
carries compute-smoke's measured 1% bound.

## Exit codes, precisely

| exit | meaning here |
|---|---|
| 0 | measured, correct |
| 1 | **measured and wrong**: NaN/Inf in device output, all-zero device output, or mismatch beyond tolerance. Nothing else. Never retried. |
| 2 | the PART lacks this precision (e.g. `fp4` on Hopper), declared with a `GEMM_SWEEP_SKIP:` marker |
| 3 | the measurement did not happen — no usable device, unreadable compute capability, a wrong-arch image, a CUDA/CUTLASS error, an unrecognised precision, `BURNIN_VARIANT_PRECISION` unset. The part is UNJUDGED and the attempt retryable. |

Two refusals worth naming because they are easy to get backwards, and
`precision.h` (unit-tested without a GPU) pins both:

- A **B200** with this image is exit 3, never 2: the part supports every
  precision here; the *image* has no SM10x code. Reporting Skip would certify a
  fleet as out of scope for the very capability it has (issue #10's lesson).
- An **unreadable compute capability** is exit 3, never 2: zero would compare
  below every minimum and read as "this fleet does not support fp8" when the
  truth is that a runtime query failed.

## Duration

Not burst-only. `BURNIN_DURATION_SECONDS` bounds a window in which the same
GEMM is launched repeatedly and **every iteration is checked** against the host
reference — a transient miscompute at minute nine is not overwritten by a clean
minute ten. The window buys repeated correctness trials and a stable TFLOP/s
figure. It is **not** a thermal soak: the per-iteration readback keeps GPU duty
well below saturation, and thermal-soak / gpu-burn own sustained load. One
kind, one job.

## Metrics

| key | canonical name | notes |
|---|---|---|
| `achieved_tflops` | `achievedTflops` | iterations × 2MNK over event-timed kernel seconds; omitted if the clock measured nothing |
| `max_relative_error` | `maxRelativeError` | worst deviation across all iterations, normalised by max\|ref\| |
| `nonfinite_count` | `nonfiniteCount` | NaN/Inf across all iterations |
| `gemm_precision` | `gemmPrecision` | label (Evidence — a word, never gate on it) |
| `gemm_shape` | `gemmShape` | label (Evidence), `1024x1024x1024` |
| `built_cuda_arch`, `gpu_name`, `compute_cap` | identity | as compute-smoke |
| `gemm_iterations`, `total_kernel_ms`, `window_seconds` | — | unregistered diagnostics |

Thresholds are **per variant** and this runner ships with **none**: no measured
baselines exist yet for these precisions on any SKU in the fleet, and this
repository has been bitten by spec-sheet thresholds before. Measure first, then
pin — and an FP4 floor is not an edited FP64 floor.

## Arch targets

Per-platform defaults, both "a" targets with no PTX fallback, exactly as
compute-smoke: arm64 → `sm_121a` (GB10 / DGX Spark), amd64 → `sm_120a` (RTX PRO
6000 Blackwell and the other x86 SM120 parts). `--build-arg CUDA_ARCH=…`
overrides both platforms.

The runtime refuses a part the image was not built for **before launching**
(`precision.h`'s `archCovers`): the loader does not refuse the
`sm_120a`-on-CC-12.1 direction on its own — it loads and trips a sticky
device-side assert under several hundred lines of CUTLASS spew, which would
overwrite the diagnosis in the stored result.

## Host requirements

A GPU, via the NVIDIA Container Toolkit's driver injection
(`NVIDIA_VISIBLE_DEVICES` / `NVIDIA_DRIVER_CAPABILITIES` are set in the image —
on the GB10 fleet this is what keeps the pod alive at all, see issue #52). No
`hostPaths`, no privileges, nothing else.

## Status

Built and unit-tested (the decision layer exhaustively, without a GPU); **not
yet verified on real hardware**, and therefore unpublished — no tag exists and
`pkg/runnerimages` deliberately carries no default image for this kind, so a
BurnInTest must name `spec.runner.image` explicitly. The hardware-verification
checklist, including what to capture for the parser tests, is issue #265.
