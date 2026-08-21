# gpu-burn-rocm — sustained-load correctness for AMD accelerators

The AMD runner image for the **`gpu-burn`** TestKind, selected per node with
`imagesByVendor`. It holds a heavy SGEMM for the whole duration budget and asks:
**does the part still get the RIGHT ANSWER while it is that hot?**

Its engine is `soak_core_rocm.h`, shared **byte-identically** with
`thermal-soak-rocm` (`runners/sharedsource_test.go` fails if the copies drift).

## No ECC — which makes the arithmetic check matter *more*

The NVIDIA gpu-burn watches ECC counters alongside the arithmetic. **gfx1151's
LPDDR5X has no ECC at all**, so `eccErrors` is emitted as **`n/a`** rather than
zero. That is exactly the case the contract's `RequiredIfMeasurable`
applicability exists for: the hardware *positively declares* it has no such
counter, which is different from a counter that could not be read. A fabricated
`0` would satisfy an `eccErrors == 0` gate on a part with nothing to count.

The arithmetic half is untouched, and it carries **more** weight here than on a
part with ECC: with no ECC to catch a flipped bit in memory, the bitwise
self-comparison is the only thing in the entire suite that would notice. A
miscompare found on a Halo is something nothing else can see.

## The kernel-log fault watch (`xidEvents`)

The one *other* thing that can corroborate a miscompare here is the kernel log.
This runner reads its own `/dev/kmsg` over exactly the window it is burning and
counts **amdgpu GPU-reset and uncorrectable-RAS lines** as `xid_count` →
`xidEvents` — a miscompare *with* a concurrent driver fault is
driver-acknowledged damage, and a miscompare with `xid_count=0` is the true
silent-corruption case, which on an ECC-less part is worth being able to state
precisely.

Shared implementation ([`kmsg/kmsg_watch.h`](kmsg/kmsg_watch.h), byte-identical
across all four soak-family runners), same opt-in grant (`/dev/kmsg` mount +
`privileged: true` + `runAsUser: 0`), same honest degradation
(`xid_source=none`, `xid_count` **omitted**, never a fabricated zero). The
pattern-narrowness reasoning and the no-`last_xid_code` decision are in
[`../thermal-soak-rocm/README.md`](../thermal-soak-rocm/README.md#the-kernel-log-fault-watch-xidevents);
the pod-grant write-up is in
[`../thermal-soak/README.md`](../thermal-soak/README.md#the-xid-watch-and-what-the-pod-needs-for-it).

## The comparison is bitwise, and a tolerance would be wrong

The first GEMM's output is the reference; every later GEMM is compared against
it bitwise on the device. The kernel's accumulation order is fixed by its
schedule, so a healthy part recomputes the identical bit pattern every time —
any difference is the hardware changing its answer between two runs of the same
arithmetic. That is silent data corruption, and it is invisible to every other
test: the part reports success, the driver logs nothing, the clocks look fine,
and the wrong number flows into a training run.

The reference is computed on the part under test, which bounds the claim: this
detects a part that **stops agreeing with itself**, not one that was wrong from
the first instruction. `compute-smoke-rocm` is the test that asserts the
arithmetic is right at all.

## Verdicts

**Fail (1)** — a bitwise miscompare or a non-finite value under load. Nothing
else.

Temperature and clock are reported as **evidence and deliberately not gated
here** — they are thermal-soak's verdict, and two kinds failing a node for the
same fault would double-count it while sending an engineer to look in two
places.

**Error (3)** — HIP failure, or zero iterations completed. Hardware unjudged.

## Status

**Not verified on hardware** (#320). No published image, no registered default;
amd64-only (#319). Covers gfx1151, gfx1100 and gfx942.
