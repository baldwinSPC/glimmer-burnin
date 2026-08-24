# `memory-retention` runner — the fault a continuously-accessed test cannot see

Runner image for the `memory-retention` [`TestKind`](../../pkg/contract/vocabulary.go):
write a pattern into a region of host memory, hold it **untouched** for the
bulk of the test's duration, then read it back — a memtest86+-shaped bit-fade
test, not memtest86+ itself.

Entirely vendor-neutral: no accelerator is touched, and this is the only
runner in this repository with **no third-party component and no external
tool of any kind** — a CGO-free Go binary over the standard library alone.

---

## Why this test exists

[`memory-stress`](../memory-stress) (stressapptest) pounds a region of host
RAM with continuous read/write traffic and catches faults that show up
**under access**. A weak DRAM cell that loses its charge over time — because
refresh is marginal, because a neighbouring row disturbs it, because the
memory controller's timing is wrong at the margin — never gets the chance to
fail that way: continuous access refreshes every cell as a side effect of
testing it.

This kind holds a pattern **untouched** instead, which is the one thing
continuous-access testing structurally cannot do. Two patterns are used, in
order: a value and its bitwise complement over the SAME cells (`0xFF` then
`0x00`) — memtest86+'s own "bit fade, 2 patterns" shape. A cell stuck at 1 is
invisible to a pattern that expects 1 there; testing the complement is what
catches it.

**This is not memtest86+.** memtest86+ needs a reboot and cannot run inside
either of this project's container dispatchers — see
[`docs/vendors/nvidia.md`](../../docs/vendors/nvidia.md) for the decision
that keeps it out of scope, and why this kind exists as a different, narrower
answer to the same underlying question.

---

## Output contract

| Result | stdout | Exit |
|---|---|---|
| Pass | `MEMORY_RETENTION_PASS` | 0 |
| Fail | `MEMORY_RETENTION_FAIL: <reason>` | 1 |
| Not applicable | `MEMORY_RETENTION_SKIP: <reason>` | 2 |
| Unjudged | `MEMORY_RETENTION_ERROR: <reason>` | 3 |

Metrics are `key=value` lines on stdout, printed **before** the decision —
`retention_patterns_completed` is even re-emitted mid-run, once per pattern,
so a run killed mid-hold still leaves a record of how far it got.

### Fail (exit 1) — the only gate this kind has

**One check, and it is unambiguous.** This runner wrote every byte of the
tested region itself and held it untouched for the whole hold, so a byte
reading back wrong is memory corruption — full stop, no calibration needed,
unlike a runner reporting evidence about a fault nobody has characterised yet.
`retentionBitFlips Equal 0` is the gate; everything else this kind reports is
context.

A real finding is never lost to an interruption: if a flip is found before
every planned pattern finished, this still reports Fail — the same
record-before-kill principle this project's CUDA soak engine applies to a
miscompare. **Absence of a flip is a different fact from "clean so far"** —
an interruption with nothing found yet is Error (unjudged), not Pass, because
only part of the plan was actually checked.

### Skip (exit 2) / Error (exit 3)

- **Skip**: the node's own usable memory, with no cgroup limit in force, is
  below the minimum test region (`BURNIN_RETENTION_MIN_MB`, default 256 MiB).
  A property of the hardware.
- **Error**: a cgroup memory limit leaves too little room to test (raise
  `spec.resources.limits.memory` or set `BURNIN_RETENTION_MB` explicitly — a
  property of the profile, not the node); `/proc/meminfo` could not be read;
  a malformed environment variable; the run was interrupted (signal or its
  own deadline) before every pattern completed with nothing found yet;
  **`BURNIN_DURATION_SECONDS` too short to complete even two full
  fill/hold/verify cycles** — see below.

### The duration-vs-hold refusal

A run shorter than two full patterns' worth of hold never completes even one
clean fill-hold-verify cycle, so this kind **refuses** rather than silently
running a shorter, less meaningful hold. This is deliberately a refusal, not
a raise-and-warn: two independently configured knobs are involved (the
requested duration, and the hold length), and the gap between what was asked
for and what a real hold needs can be large. Silently running longer than
requested risks overrunning the pod deadline the operator computed from the
*original* request — turning a configuration mismatch into a pod eviction
partway through, rather than an immediate, legible refusal at the very start.

---

## Configuration

| env | default | meaning |
|---|---|---|
| `BURNIN_DURATION_SECONDS` | 3600 | total wall time; must leave at least 30s per pattern after a fill/verify reserve, or the run refuses (see above) |
| `BURNIN_RETENTION_MB` | *(auto-sized)* | explicit test region size; obeyed exactly, never clamped — the pod either has the room or the kernel OOM-kills it, which is a real signal, not a smaller number quietly substituted for what was asked |
| `BURNIN_RETENTION_FRACTION` | 0.50 | fraction of usable memory to allocate when auto-sizing; the remainder is headroom for the page cache and everything else sharing the cgroup |
| `BURNIN_RETENTION_MIN_MB` | 256 | the smallest region worth calling a retention test |
| `BURNIN_RETENTION_HOLD_SECONDS` | *(derived)* | explicit PER-PATTERN hold, overriding the duration-derived one; must be at least 30s or the run refuses |

A value that cannot be parsed is a configuration Error, not a default:
guessing would run a different test than the one asked for while reporting
success.

**Longer holds find more.** A retention fault is a function of how long a
cell sits unrefreshed by access, and this project has no measured data yet on
how long is "long enough" on any of its own hardware — see
[What is not verified](#what-is-not-verified). Prefer the largest
`BURNIN_DURATION_SECONDS` a maintenance window allows.

---

## Metrics

### Correctness (the acceptance gate)

| raw key | canonical | |
|---|---|---|
| `retention_bit_flips` | `retentionBitFlips` | count of bits that read back wrong after being held untouched — a byte with two wrong bits counts as two |
| `retention_bytes_flipped` | `retentionBytesFlipped` | count of BYTES containing at least one wrong bit — the coarser companion, the way `miscompares`/`sdcDetections` answer different questions elsewhere in this project |

### Evidence — never gated

| raw key | canonical | |
|---|---|---|
| `retention_patterns_completed` | `retentionPatternsCompleted` | how many fill/hold/verify cycles finished before the run ended or was interrupted (0–2) |
| `retention_first_flip_offset` | `retentionFirstFlipOffset` | byte offset of the first flip found; omitted when `retentionBitFlips` is 0 |
| `retention_first_flip_pattern` | `retentionFirstFlipPattern` | which pattern (`0xFF` or `0x00`) was being verified when the first flip was found; omitted when `retentionBitFlips` is 0 |
| `retention_memory_locked` | `retentionMemoryLocked` | `"true"` if the region was `mlock`ed for the whole hold (pinned against swap/reclaim), `"false"` on a best-effort failure — see below |
| `region_mb`, `patterns_planned`, `hold_seconds`, `duration_requested_s` | *(unregistered; generic normalisation)* | the resolved configuration, echoed before the run so a pod killed at its deadline still records what it was going to be judged against |

### Why `retentionMemoryLocked` can honestly be `false`

`mlock` needs `CAP_IPC_LOCK` or a raised `RLIMIT_MEMLOCK` that most clusters
do not grant a burn-in pod by default — the same posture this project already
accepts for `nccl`'s own memlock. Locking is attempted so the region cannot
be swapped out or reclaimed mid-hold (a page round-tripped through swap is
not "held untouched in DRAM" any more), but the attempt is **non-fatal**:
refusing to run without it would make this kind unusable on any cluster that
has not arranged the grant. A `false` value does not invalidate a `Pass` or a
`Fail` — it means the OS was structurally free to move the region during the
hold, not that it necessarily did.

---

## Build

```sh
docker build -t memory-retention runners/memory-retention
docker run --rm \
  -e BURNIN_DURATION_SECONDS=90 \
  -e BURNIN_RETENTION_MB=64 \
  -e BURNIN_RETENTION_HOLD_SECONDS=30 \
  memory-retention
```

No CUDA, no cross-compilation cost: a pure-Go, CGO-free binary with `GOPROXY=off`
enforced at build time (the runner's non-test sources import only the standard
library, and the build fails loudly rather than reaching the network if that
ever stops being true). Both `linux/amd64` and `linux/arm64` build natively —
no QEMU either side.

---

## What is not verified

Real, structural verification exists here that most runners in this
repository cannot get: this is pure Go with no accelerator dependency, so it
was built and run for real, repeatedly, in a real Linux container, on the
machine that wrote it — including watching `mlock` genuinely succeed, and
confirming the Pass, Skip, and both Error paths (a malformed env var, and a
cgroup limit too tight) against real container output, not just a passing
test suite. `go build`/`go vet`/`go test` are clean, and `pattern_test.go`
exercises the fill/verify arithmetic with a REAL background goroutine
mutating the buffer mid-hold — not a mock — confirming `runTest` actually
surfaces what `verifyPattern` finds.

What genuinely is **not** verified:

- **Any real retention fault, on any real hardware.** Every test above
  confirms the plumbing is correct on healthy memory; none of it can prove
  the kind catches what it is designed to catch, because no faulty DIMM has
  been available to test against.
- **How long a hold needs to be** to find a real fault on this project's own
  fleet. The default (a ~1785s hold per pattern, from the 3600s default
  duration) is a reasonable starting point, not a measured one — see
  [Longer holds find more](#configuration) above.
- **`mlock` behaviour under real cluster memory pressure.** Confirmed working
  in a local, lightly-loaded container; not confirmed against a node under
  the kind of memory pressure that would actually test whether the lock
  holds.

File the hardware-verification issue this playbook calls for (the shape of
[#237](https://github.com/baldwinSPC/glimmer-burnin/issues/237),
[#242](https://github.com/baldwinSPC/glimmer-burnin/issues/242)) before
publishing: what to run, what to capture, and — ideally — a real weak DIMM or
an injected fault to confirm the detector actually detects.

---

## Licensing

No third-party component of any kind. `main.go`, `config.go`, `sysinfo.go`
and `pattern.go` are Apache-2.0 (this project); the only other thing compiled
in is the Go standard library (BSD-3-Clause). No external tool is executed,
no vendor library is dlopened, and no accelerator driver is touched — this is
the one runner in this repository that needs none of it.

---

## Requirements on the node

None beyond host memory and a Linux kernel. No `hostPaths` mount, no
`privileged`, no `runAsUser` override, no accelerator of any kind — the
default `USER 65532:65532` is sufficient for everything this runner does
except `mlock`, which degrades honestly (see
[`retentionMemoryLocked`](#why-retentionmemorylocked-can-honestly-be-false)
above) rather than requiring a privilege grant.
