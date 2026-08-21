# thermal-soak-rocm — thermal soak for AMD accelerators

The AMD runner image for the **`thermal-soak`** TestKind, selected per node with
`imagesByVendor`. It holds a heavy, correctness-checked SGEMM for the whole
duration budget and asks: **did the part stay at clock, and cool enough not to
trip a protective throttle?**

Its engine is `soak_core_rocm.h`, shared **byte-identically** with
`gpu-burn-rocm` (`runners/sharedsource_test.go` fails if the copies drift).
Two kinds, two assertion sets, one implementation — so both kinds'
`sustainedThroughputTflops` is the same quantity, on both vendors, and a fleet
charts them as one series.

## The temperature ceiling is APU-calibrated, and it is not the NVIDIA number

**Strix Halo runs hot by design.** Community measurement puts ordinary sustained
inference at **87–91 °C** edge, and a Framework board has been observed holding
**98.8 °C for hours without throttling**.

The NVIDIA runner's ceiling would therefore fail every healthy Halo in a fleet,
continuously, and every failure would arrive in the shape of a hardware verdict.
The default here is **100 °C** — above normal operation, below the junction
limit where the driver itself intervenes.

That is a **starting point from published behaviour, not a measured threshold.**
Issue #320 is where it gets replaced by a number this fleet actually observed.
Do not tighten it on intuition: the intuitive value is the discrete-GPU one, and
it is wrong here. This is the same calibration question GEP-0753 raises for the
thermal watchdog itself.

## The clock floor is looser than clockprobe's, deliberately

40% of the DPM ladder's top, against clockprobe-rocm's 60%. A soak is long
enough that ordinary thermal management legitimately pulls clocks down — that is
the part working, not failing — so this floor exists only to catch a *collapse*.
clockprobe is the kind that judges sustained clock precisely, on a short window,
before heat is a factor.

## What AMD cannot measure here

`throttleEvents` is emitted as **`n/a`**, not zero. NVML publishes a
throttle-reason mask; amdgpu's sysfs has no equivalent, so throttling can be
inferred from clocks but not enumerated. A fabricated `throttle_count=0` would
satisfy a `throttleEvents == 0` gate on a node whose throttle state was never
read. `eccErrors` is `n/a` for the same discipline — see gpu-burn-rocm's README.

## The kernel-log fault watch (`xidEvents`)

What AMD **can** measure is a driver-logged catastrophic fault. This runner
reads its own `/dev/kmsg` over exactly the window it is soaking — the shared
[`kmsg/kmsg_watch.h`](kmsg/kmsg_watch.h), byte-identical with both NVIDIA soak
runners, positioned at the ring buffer's tail at start so nothing from before
the window is counted — and reports the count as `xid_count` → `xidEvents`.
"Xid" is NVIDIA's word; the *severity class* the metric name means is
vendor-independent, and on this hardware the count is of **amdgpu GPU-reset and
uncorrectable-RAS lines**.

The pattern is deliberately **narrower** than
[`../host-health`](../host-health)'s own amdgpu heuristic (`amdgpu` +
`reset`/`ras`): that one feeds `kernelHwErrors`, which is Evidence-only and
never gates, while this feeds an Acceptance metric a profile is expected to
write `xidEvents Equal 0` against — and a bare `ras` substring matches
informational lines about the feature merely being present, which would turn a
healthy fleet's boot chatter into hardware verdicts. This match requires
`amdgpu` plus `reset` or `uncorrectable`. There is no `last_xid_code` here:
amdgpu's fault lines carry no single canonical numeric field the way
`NVRM: Xid: NN` does, and inventing one would declare a measurement nobody
took.

**Opt-in, honest degradation** — same contract as everywhere else in this
project: the pod needs `/dev/kmsg` mounted (`hostPaths`, `type: CharDevice`)
plus `privileged: true` and `runAsUser: 0`; without the grant the runner
reports `xid_source=none` with an actionable `xid_source_detail` and **omits**
`xid_count`, so a gate on it fails closed rather than certifying a node on a
zero nobody measured. The full three-part-grant write-up is in
[`../thermal-soak/README.md`](../thermal-soak/README.md#the-xid-watch-and-what-the-pod-needs-for-it).

## Verdicts

**Fail (1)** — peak temperature above the ceiling, sustained clock below the
floor, or a bitwise miscompare/non-finite value under load. Correctness failures
fail this kind too: a part that changed its answer during a soak has failed the
soak, and staying silent because another kind owns that question would let a run
pass on evidence of corruption it had in hand.

**Error (3)** — HIP failure, or zero iterations completed. Hardware unjudged.

## Status

**Not verified on hardware** (#320). The temperature ceiling and clock floor are
published-behaviour defaults, not fleet baselines. No published image, no
registered default; amd64-only (#319). Covers gfx1151, gfx1100 and gfx942 — the
load is plain fp32 with no matrix-instruction dependency.

The amdgpu fault patterns are likewise **built from the driver's published
log-message vocabulary, not from a capture**: this project's ROCm fleet has
never produced a GPU reset or an uncorrectable RAS event to record the real
message shape from. The window mechanics are unit-tested against fixtures
(`kmsg/kmsg_watch_test.cc`); the pattern text is what the hardware pass must
confirm.
