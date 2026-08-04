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

Those four are not merely inadvisable to gate on, they are registered as
`Evidence` in [`pkg/contract`](../../pkg/contract/metrics.go) so that
`pkg/verdict.ValidateThresholds` refuses the gate at authoring time. For the
three identity metrics the reason is blunter than "the number is noisy": their
values are **labels**, a threshold is compared as a `float64`, and a gate on one
fails closed on every node forever while reading as a hardware verdict.
`compute_cap` is the sly one — `12.1` parses, so the gate silently compares a
`major.minor` version as a decimal.

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
- **This image carries no code for the part it landed on** — detected *before
  the launch* by comparing `built_cuda_arch` against the compute capability the
  device reports (see [the image gate](#the-image-gate-and-why-it-runs-before-the-launch)) —
  or the binary was built with no SM120/SM121 block-scaled MMA path at all. Both
  are statements about the image that was pinned, not about the hardware. The
  error message names `built_cuda_arch`, the capability it actually found, and
  the `CUDA_ARCH` value that would work.
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

The scope gate admits the whole `12.0`/`12.1` family, while the default build
pins `sm_121a` (CC 12.1 only). So a **CC 12.0 part is in scope, passes the scope
gate, and carries no code this image can run** — which is why that path is an
`Error` and not a `Fail`. Before this was fixed it exited 1, recording a hardware
verdict against a node whose tensor cores were never exercised.

It is `Error` rather than `Skip` on purpose. Narrowing the scope gate to the
built arch would report a whole fleet as "not applicable" when the truth is that
the operator pinned the wrong tag; `Error` is the retryable, hardware-unjudged
phase and it names the fix.

Both decisions live in [`arch_match.h`](arch_match.h) — host-only and free of
CUDA — as `scopeOf()` and `archMatch()`, so a plain C++ program can drive them
to every compute capability with no GPU in the room. See
[how the Skip path is exercised](#how-the-skip-path-is-exercised) for why that
matters more here than anywhere else in this runner.

#### How the Skip path is exercised

Exit 2 is the contract's load-bearing distinction — *a node that cannot run a
test has not failed it* — and until 2026-08-03 it had **never executed**
(issue #9). It fires only on a part that is not CC 12.0/12.1, this project has
no such accelerator, and the route everyone expected is gone: `dcgm-diag` was
supposed to be the first real-silicon exerciser on the theory that DCGM does not
support GB10, and DCGM works on GB10.

It is now covered three ways, in ascending order of how much they prove:

1. **`scopeOf()` is unit-tested exhaustively** in
   [`arch_match_test.cc`](arch_match_test.cc) — H100, L40S, A100, T4, V100,
   B200, a hypothetical CC 12.2, and an unread capability — under `make test`,
   with no Docker, no CUDA and no GPU.
2. **The sentinel is asserted end to end in Go** from captured output, in
   `pkg/runner`'s parser tests and through the reconciler in
   `internal/controller` — the latter with `nonfiniteCount Equal 0` attached, so
   it also proves a skipped run's missing metric does not fail closed into a
   hardware verdict.
3. **The real binary took the path on the real GB10.** A compile-time capability
   injection (`BURNIN_FP4_FORCE_CC_MAJOR` / `_MINOR`, see `fp4_smoke.cu`) builds
   the same source with the same toolchain and tells it the device reports a
   capability it does not have:

   ```console
   $ docker build --target build -t cs-build .
   $ docker run --rm --gpus all cs-build bash -c '
       nvcc -std=c++17 -O3 -arch=sm_121a --expt-relaxed-constexpr \
         -DBURNIN_CUDA_ARCH=\"sm_121a\" \
         -DBURNIN_FP4_FORCE_CC_MAJOR=9 -DBURNIN_FP4_FORCE_CC_MINOR=0 \
         -cudart static -I /cutlass/include -I /cutlass/tools/util/include \
         -o /tmp/skip /src/fp4_smoke.cu && /tmp/skip; echo EXIT=$?'
   built_cuda_arch=sm_121a
   forced_compute_cap=9.0
   gpu_name=NVIDIA GB10
   compute_cap=9.0
   FP4_GEMM_SKIP: NVFP4 block-scaled GEMM requires compute capability 12.0/12.1, and this part reports 9.0 — the test does not apply to this hardware and the part is NOT failed; run a kind whose runner covers this architecture
   EXIT=2
   ```

   The switches are compile-time so no environment variable can reach them, no
   Dockerfile in the repository may define them (`runners/pins_test.go` fails the
   build if one does), and a binary built with them prints
   `forced_compute_cap=` — a `key=value` line, so it lands in the stored result
   and the delivered envelope and keeps such a verdict self-identifying forever.

#### The `#else` guard is compiled on every image build

`fp4_smoke.cu` wraps its kernel in a block-scaled-MMA availability check whose
`#else` reports the part unjudged. That branch had never been compiled by any
build, so a syntax or logic error in it would have surfaced for the first time
on a CUTLASS or CUDA downgrade — which is to say, on a fleet.

**The arch flag cannot reach it.** Measured against CUTLASS v4.6.1 with CUDA
13.0.1: both `CUTLASS_ARCH_MMA_SM120_SUPPORTED` and `..._SM121_SUPPORTED` are
defined *even for `-arch=sm_80`*, so building for a pre-Blackwell arch still
takes the `#if` side. Only a pre-v4.5.0 CUTLASS leaves them undefined, and
cloning a second, deliberately stale upstream into every build to compile eight
lines is the worse trade.

So the Dockerfile compiles **both** sides on every build, forcing the fallback
with `BURNIN_FP4_ASSUME_NO_BLOCK_SCALED_MMA` and everything else identical to
the release compile. The compile artefact is asserted, then deleted, and never
enters the final image. The second half of that guard is the one with teeth: the
**shipped** binary must *not* contain the `#else` message, which is the proof
the real kernel is in there. Without it, a CUTLASS bump that stopped defining
those macros would publish an immutable tag reporting every node in a fleet as
`Error`.

Verified on the GB10 that the branch not only compiles but runs:

```console
FP4_GEMM_ERROR: this binary was not compiled with SM120/SM121 block-scaled MMA support (need CUTLASS >= v4.5.0 and -arch=sm_120a/sm_120f/sm_121a); the part is in scope and UNJUDGED
EXIT=3
```

### The image gate, and why it runs before the launch

The image question is settled by `arch_match.h`, from the `CUDA_ARCH` string
baked in at build time and the capability the device reports — **not** from
whatever the CUDA runtime latches at the launch, and **not** from a preprocessor
macro.

Not a macro, because there isn't one that can answer: nvcc's
`__CUDA_ARCH_LIST__` records `1200` for both `sm_120a` and `sm_120f`, and only
the `f` form also covers 12.1. `BURNIN_CUDA_ARCH` carries the distinction that
the macro loses.

Not the runtime error, because it only fires in one direction. The loader
refuses an `sm_121a` cubin on a CC 12.0 part, and that direction always
surfaced as `cudaErrorNoKernelImageForDevice`. The other direction is not
refused at all — SM121 is binary-compatible with SM120, so an **`sm_120a` image
loads on a CC 12.1 GB10** and then trips a device-side assert inside the CUTLASS
kernel. Measured on real hardware (CC 12.1, driver 580.82.09, CUDA 13.0.1), that
reached the operator as:

```
FP4_GEMM_ERROR: warmup cudaDeviceSynchronize: device-side assert triggered
```

— naming neither the arch nor the fix, under several hundred lines of CUTLASS
template assert text. The exit code was right; the diagnosis was not there.

Refusing **before** the launch is what makes this worth doing, rather than only
rewording the error afterwards. Nothing is launched, so no assert fires, so the
CUDA context is not poisoned (`cudaErrorAssert` is sticky — every later CUDA
call in the process fails too) and CUTLASS emits none of its template spew. That
spew is not merely ugly: a container log is stdout and stderr **merged**, and
`pkg/runner.Parse` takes `Result.Message` from the **last** line that is not
`key=value`, so a CUTLASS line landing after the diagnosis becomes the message
stored in the `TestResult`. Not launching removes the spew instead of racing it.

The comparison is conservative, and reports a mismatch only where one is
positively established:

| `CUDA_ARCH` | covers |
|---|---|
| `sm_121a`, `sm_120a` (arch-specific) | exactly that capability — no PTX fallback, which is [deliberate](#what-pass-actually-means) |
| `sm_120f` (family-specific) | that capability and later minors of the same major — CC 12.0 **and** 12.1 |
| `sm_120` (plain) | that capability and newer, since nvcc embeds PTX the driver can JIT forward |

Anything it does not recognise — an unparseable target, or the `unknown` that a
hand-built binary carries — is **Unknown**, never a mismatch: a runner may only
declare what it positively established. An Unknown runs on and leaves the
runtime error to speak, exactly as before.

Build with `CUDA_ARCH=sm_120f` for one binary covering CC 12.0 + 12.1.

`arch_match.h` is host-only and free of CUDA precisely so this decision is
testable without a GPU. `arch_match_test.cc` covers the classification, the
message, and the property that matters most — that no failure it describes ever
returns exit 1 — and `runners/cxxtests_test.go` compiles and runs it under
`make test`, with no Docker and no CUDA toolchain.

> **Image tags.** The published `v0.1.0` image predates this fix and reports all
> of the above as exit 1. Published tags are immutable and it is not being
> changed; the corrected contract ships under a **new tag**. A gate still pinning
> `v0.1.0` keeps the old, wrong behaviour until it is repinned.

## This runner is a BURST, and it ignores `BURNIN_DURATION_SECONDS`

One warm-up GEMM, one timed GEMM, done — in **milliseconds**, whatever duration
the operator injects. Every other runner in this repository reads
`BURNIN_DURATION_SECONDS`; this one does not, and
[`TestKind.BurstOnly()`](../../api/v1alpha1/burnintest_types.go) says so out
loud.

That was true silently for a long time while `config/samples/node-acceptance.yaml`
asked the kind for `durationSeconds: 120` and got milliseconds — so the shipped
"node acceptance" burned nothing in and nothing anywhere said so (issue #25).

The fix was to **stop pretending**, not to add a loop:

- What this runner proves is that the real block-scaled MMA instruction path
  executed on this part and produced the right answer. That is a *correctness*
  statement, and running the same GEMM for two minutes does not make it a
  stronger one.
- It would make it a **worse soak** than the kinds that exist to be soaks.
  `thermal-soak` and `gpu-burn` both run on the shared duration-honouring load
  wrapper (`soak_core.cuh`), and `clockprobe` holds a load specifically to judge
  sustained clocks. Converging this runner onto that wrapper would duplicate all
  three and leave the fleet with no cheap correctness gate at all.
- One kind, one job. A profile that wants both puts a burst and a soak in it,
  which is what `node-acceptance.yaml` does.

`durationSeconds` is not *meaningless* on this kind: the reconciler still derives
the pod's deadline from it, so an image that hangs before reaching its kernel is
still killed. What it does not do is decide how long the test runs. A profile
author who meant "burn this node in for two minutes" wants a soak kind.

Two guards keep the three statements — behaviour, kind doc, sample — from
drifting apart again, and they run in `make test`:

- `runners/pins_test.go` fails if this runner's source ever *does* read
  `BURNIN_DURATION_SECONDS` while the kind still claims to be burst-only (and,
  in the other direction, if any non-burst runner stops reading it).
- `api/v1alpha1/samples_test.go` fails if any shipped sample sets
  `durationSeconds` on a burst-only kind.

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

Build args: `CUDA_IMAGE`, `CUTLASS_REF` (default `v4.6.1`), `CUTLASS_SHA`,
`CUDA_ARCH` (default `sm_121a`; use `sm_120f` for one binary covering CC 12.0 +
12.1).

The build **asserts the upstream commit**: a tag can be moved, and here CUTLASS
is the kernel itself, so a moved tag would change what every node was accepted
by under an image tag that is meant to be immutable. Bumping `CUTLASS_REF` means
bumping `CUTLASS_SHA` with it, or the build refuses. Resolve the new value with
`git ls-remote https://github.com/NVIDIA/cutlass.git 'refs/tags/<ref>'`.

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
