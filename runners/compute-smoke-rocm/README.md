# compute-smoke-rocm — matrix-core GEMM smoke test for AMD accelerators

The AMD runner image for the **`compute-smoke`** TestKind. It is not a kind of
its own: a profile selects it per node with `imagesByVendor`, so one profile
serves a mixed fleet and the metric names mean the same thing on every vendor.

```yaml
apiVersion: burnin.glimmer.ai/v1alpha1
kind: BurnInTest
metadata:
  name: compute-smoke
spec:
  kind: compute-smoke
  scope: Node
  # No durationSeconds: the kind is burst-only and this runner does not read
  # BURNIN_DURATION_SECONDS. runners/pins_test.go enforces that both ways.
  runner:
    imagesByVendor:
      # No default is registered for this image yet: publication is
      # hardware-gated and it has not run on silicon.
      - vendor: amd
        image: <registry>/glimmer-burnin-compute-smoke-rocm:<version>
  resources:
    limits:
      amd.com/gpu: 1
  thresholds:
    - metric: nonfiniteCount
      comparison: Equal
      value: "0"
```

## What it proves

One claim: **the matrix-core path executed on this part and produced the right
answer.** Two precisions, gated independently:

| Precision | Path | Why it is here |
|---|---|---|
| bf16 | `v_wmma_f32_16x16x16_bf16` | what inference on this hardware actually runs |
| fp16 | `v_wmma_f32_16x16x16_f16` | the older path much software still reaches for |

A part can pass one and fail the other, so both must pass; a partial pass is a
**FAIL** naming the precision that failed, never a pass with a footnote.

The claim is worth exactly as much as the arch gate around it, which is why
there are three separate questions and no fallback path in the image.

## rocWMMA does not support gfx1151

This runner was first written against **rocWMMA**, AMD's portable matrix-fragment
API. It does not compile for Strix Halo. ROCm 6.4.4's
`rocwmma/internal/config.hpp` ends its architecture dispatch with

```
static_assert(0, "Unsupported architecture");
```

and gfx1151 reaches it: the allow-list covers gfx908/90a/94x and
gfx1100/1101/1102, and the RDNA3.5 APUs are absent.

**The part is not the problem.** gfx1151 implements the same `V_WMMA_*`
instructions as gfx1100, and clang exposes them as builtins for the whole
`gfx11-insts` family. Only the library's list omits it — another instance of the
pattern this hardware keeps producing, where AMD's own libraries lag the APU
(compare `amd-smi` reporting N/A on gfx1151 while the kernel has the data).

So the kernels call the wave32 builtins directly. The cost is real and worth
stating: rocWMMA would have covered CDNA's MFMA from the same source, and these
builtins cover gfx11 only. **CDNA parts are not covered by this image** and are
reported as an Error (hardware unjudged), never a Skip. The alternative was a
runner that cannot measure the hardware it was written for.

## The fragment layout is unit-tested

A WMMA layout bug does not crash — it produces a tile that is wrong in a
structured way (every other row, or a transpose), which against a symmetric test
matrix can even look right. So `wmma_tile.h` holds the index arithmetic with no
HIP in it, and `wmma_tile_test.cc` checks it as a permutation without a GPU:

- every one of the 256 outputs is written **exactly once** (a layout that writes
  one twice necessarily leaves another unwritten, and that is invisible against
  a zero-filled buffer);
- every A and B element is loaded by some lane;
- both halves of the wave load the same fragments — the wave32 form, asserted so
  a "fix" that makes them differ fails;
- a **simulated wave** in plain C++ reproduces the reference GEMM end to end.

## The three arch questions

Ported from `compute-smoke/arch_match.h` and kept un-collapsed, because
collapsing any two misreports one of them:

| Question | Property of | Wrong answer means |
|---|---|---|
| `scopeOf` — can this part do matrix GEMM at all? | the **hardware** | **Skip** (exit 2) — acceptance does not apply |
| `kernelCovers` — does this build implement its matrix family? | the **source built** | **Error** (exit 3) — unjudged |
| `archMatch` — does the image carry code for this exact target? | the **offload list** | **Error** (exit 3) — unjudged |

Only a part with genuinely no matrix cores (RDNA1/2, pre-CDNA GCN) may Skip. A
part this image simply was not built for is an Error — reporting Skip for it
would read as "acceptance does not apply to this node" and certify a fleet
nobody measured.

**The gate is harder on AMD than on NVIDIA.** CUDA can JIT from PTX and its
cubins carry minor-version compatibility; HIP has neither, so a target absent
from `GPU_TARGETS` has *no* code in the image at all.

## Reading the target name

AMD target names are **positional**, following LLVM: the last character is the
step (a hex digit, so `a` is 10), the one before it the minor, the rest the
major. `gfx90a` is major 9, minor 0, step 10 — a *later* step than `gfx908`.

Read as plain integers they are 90 and 908, so any ordering between them is
nonsense. That is not hypothetical: the first draft of `gfx_gate.h` did exactly
that and judged an MI200 to be a part with no matrix cores. `gfx_gate_test.cc`
pins the ordering.

## Why the inputs are fixed and exactly representable

Every input is a multiple of 0.25 under magnitude 8, so it is exact in bf16's 8
significand bits and fp16's 11, and the accumulator is fp32. The tolerance then
gates the **hardware** rather than the format — a wrong answer is the part, not
rounding. Random inputs would put the format's error budget inside the gate and
force a tolerance loose enough to hide real faults.

The output buffer is pre-filled with a sentinel no correct result can take.
Zero would be useless as evidence: a launch that silently did nothing leaves
the buffer untouched, and against a zero reference that comparison would pass
and report a matrix path that never executed.

## Multi-device

This runner measures **every device the pod was allocated**, not device 0
(docs/dev/multi-device.md). Request the node's whole board
(`amd.com/gpu: "8"`); iteration is always sequential — this kind is a BURST,
one bf16 + fp16 launch pair per device taking milliseconds, so there is no
window to divide or run concurrently over. `BURNIN_DEVICE_CONCURRENCY` is
read and, if set to something other than `sequential`, reported in
`config_warnings` rather than silently ignored, but it never changes the
iteration.

`nonfiniteCount` (Sum) and `maxAbsError` (Max, taken as the worse of the two
precisions per device) are the fold across devices; `m`/`n`/`k`/`tolerance`/
`bf16_*`/`fp16_*`/`elapsedMs`/`maxAbsRef` stay device 0's, evidence exactly as
before. A device whose scope/kernel/arch gate fails leaves the WHOLE test
unjudged for that device (Error), and the combined verdict across devices is
Fail > Error > Skip > Pass (device_fold.h's `combineExitCodes`) — a device is
a PART, so a measured miscompare on one device does not erase a measured pass
on another, but a device that was never launched cannot be soundly folded
into a Pass. New keys: `deviceCount`, `devicesVisible`, `deviceWindowS`
(always `0`), `deviceConcurrency` (always `sequential`),
`worstDeviceIndex`/`worstDevicePciBusId` (named from `nonfiniteCount`, the
registered Acceptance metric — `maxAbsError` is Evidence-only). When more
than one device reported, a `per-device.json` artifact carries every
device's own reading. `pciBusId` is read directly with
`hipDeviceGetPCIBusId` — unlike clockprobe-rocm and the soak family, this
runner touches no sysfs clock/power file, so it needs no card-correlation
machinery to get it.

**Not verified on multi-GPU AMD hardware, and not verified on hardware at
all** — no Strix Halo unit was available when either the single-device or
the multi-device engine was written.

## Status — read before pinning

- **NOT VERIFIED ON HARDWARE.** No Strix Halo unit was available when this was
  written. The arch gate and the result check are unit-tested without a GPU
  (`gfx_gate_test.cc`, run by CI); the HIP/rocWMMA path is compile-verified
  only. Tracked with the rest of the hardware pass in issue #320.
- **No published image, no registered default** — publication is hardware-gated
  by policy, and `pkg/runnerimages` lists nothing before it is published.
- **linux/amd64 only** (issue #319 covers teaching the publish workflow).
- **CDNA (MFMA) is not covered** — a follow-up, blocked on rocWMMA supporting
  gfx1151 or on a second MFMA kernel path.
- **No fault-injection macros**, unlike the CUDA compute-smoke. Its pair exists
  because its Skip path had never executed anywhere; here that path is covered
  by unit tests over `scopeOf`. Add them if the on-hardware pass finds the
  integration path still unexercised.
- **Runtime prerequisites**: `/dev/kfd` and `/dev/dri` from the device plugin
  (`amd.com/gpu: 1`). Runs as uid 65532; see clockprobe-rocm's README for the
  same open question about group-restricted `/dev/kfd`.
