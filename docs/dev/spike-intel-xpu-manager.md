# Spike: Intel XPU Manager as a third `dcgm-diag`-shaped runner (#172)

This answers issue #172's five deliverables. It is a spike report, not a
design doc for a committed feature: the code it produced
(`runners/xpu-diag/`) is real and builds, but is marked UNVERIFIED throughout
and is deliberately not wired to a default image (`pkg/runnerimages.WithoutDefault()`
carries `KindXPUDiag`). See `runners/xpu-diag/README.md` for the runner
itself; this document is the investigation behind it.

**Recommendation up front (point 5 in full below): build the scaffolding now,
defer publishing.** That is exactly the state this spike leaves the repository
in — nothing further needs to change to act on this recommendation.

---

## 1. Licence and shipping story

**Intel XPU Manager is MIT-licensed.** Verified two independent ways: the
GitHub API's own licence classification (`GET /repos/intel/xpumanager/license`
→ `spdx_id: "MIT"`) and `LICENSE.md`'s actual text at the `v1.3.8` tag
(a standard MIT permission grant, Copyright Intel Corporation). Both checks
were run against the source this spike actually builds from, not assumed from
the project's front page.

### The repository has two, disconnected, actively-maintained lines — and the issue's premise describes the OLDER one

This is the load-bearing finding of this whole investigation, and it was not
visible from the outside before this spike:

- **`main`** (the repository's real default branch, confirmed via the GitHub
  API — not `master`, which is a *stale* branch last touched 2026-06-01) and
  the **`v2.x`** release line (latest: `v2.1.0`, published 2026-08-12) have
  been **re-architected**: the classic `xpumcli`/`xpu-smi` one-shot diagnostic
  CLI is gone, replaced by `xpumd`, a long-running daemon described in its own
  README as *"a custom OpenTelemetry Collector"* exposing GPU health through
  Prometheus/OTel/gRPC metric streams. There is no bounded "run a suite, get a
  verdict, exit" invocation left in this line at all — `git log` confirms
  `v2.1.0` and `v1.3.8` share **no common ancestor** (`GET
  /repos/intel/xpumanager/compare/v2.1.0...master` returns 404, "No common
  ancestor").
- **`v1.3.x`** is a separate, *also actively maintained* branch carrying the
  classic CMake-based `cli`/`core` architecture — the one whose diagnostic
  command is documented as `xpumcli diag -l 1|2|3 -j`, matching the issue's
  own description exactly. It is not a frozen legacy fork: `v1.3.8` was
  published **2026-08-18, five days before this spike**, newer than
  `v2.1.0`. Intel is patching both lines in parallel.

A wrapper built against `main`/`v2.x` would not fit this project's runner
contract at all — there is nothing in that architecture shaped like "exec
once, get an exit code." Everything downstream of this point (and all of
`runners/xpu-diag/`) targets **`v1.3.x`, pinned at `v1.3.8`**
(`a22960fe0c3479cb4237233c0863e5c4b6464b95`, a lightweight tag — see the
Dockerfile's own pinning comment), which is the line that actually offers the
shape `dcgm-diag`'s pattern needs.

**Anyone revisiting this spike should re-check which branch is current before
trusting anything below it that is version-specific** — a project maintaining
two disconnected release lines in parallel is exactly the kind of fact that
goes stale.

### What the image needs at runtime, and why nothing here is licence-blocked

Unlike NVIDIA CUDA/DCGM — where this project ships **no** vendor binaries
because NVIDIA's own EULA forbids redistribution (`runners/dcgm-diag`'s
Dockerfile documents this in detail) — **nothing in Intel's stack down to
`xpu-smi` itself is licence-blocked from being baked into an image**:

| Component | Licence | Verified |
|---|---|---|
| XPU Manager (`xpu-smi`) | MIT | GitHub API + `LICENSE.md` at `v1.3.8` |
| Level Zero Loader (`oneapi-src/level-zero`) | MIT | GitHub API |
| Level Zero GPU backend (`intel/compute-runtime`) | MIT | GitHub API |
| grpc, metee, igsc (build-time deps of the CMake project) | Apache-2.0 | GitHub API |

The equivalent of `dcgm-diag`'s `ldd`-based "no NVIDIA-licensed
redistributable library" guard therefore does not need to exist for this
runner in the same shape — there is no licence reason to keep anything out.
What replaces it is a narrower, still-real question this spike could **not**
verify without hardware: whether the GPU-specific Level Zero **backend**
(tied to the installed driver version, the way `libcuda.so.1` is) should be
baked in or supplied by the host. `runners/xpu-diag/Dockerfile` takes the
host-supplied position, by analogy with every other vendor runner in this
repository — an analogy, not a measurement. The in-tree `i915`/`xe` kernel
driver can never be baked into any container regardless of vendor, the same
as every other accelerator.

One gap this spike leaves open rather than closing blind: `grpc`/`metee`/`igsc`
are compiled as part of Intel's own CMake build regardless of the
`-DDAEMONLESS=ON` flag this runner uses to exclude the daemon; whether they
end up **linked into** the `xpu-smi` binary specifically, or only into the
daemon target this build excludes, was not traced through the CMake source.
All three are Apache-2.0 regardless, so this is a "which NOTICE entry is
load-bearing" question, not a licence-compliance risk — but it is recorded as
open rather than assumed.

---

## 2. Exit codes, and the trap that bit every wrapper before this one

Issue #172 named the risk up front by analogy: *"`dcgmi` exits 226 on a
warning."* The real trap here is the same class, in the opposite direction,
and could not have been found from documentation alone — exactly as the issue
predicted.

**`xpu-smi diag`'s own process exit code does not encode whether any
diagnostic component passed or failed.** This was established by reading the
CLI's own source at `v1.3.8`, not by inference from a table example:

- `cli/src/comlet_base.h`'s `setExitCodeByJson` only sets a non-zero exit code
  when the JSON response carries a top-level `"errno"` field — i.e. a
  **machinery** failure: a bad `-d`/`-l` argument, a task that never started.
- `cli/src/comlet_diagnostic.cpp`'s success path (`getTableResult` /
  `showDeviceDiagnostic`) never inspects an individual component's `result`
  field, and never calls `setExitCodeByJson` after a diagnostic actually
  completes. It only renders a table.

So a wrapper that trusted `$?` the way a naive DCGM wrapper might trust
`$?==0` would report **`Pass` for hardware that genuinely failed one or more
components**, provided the CLI itself didn't error out first. This is not a
hypothetical: it is the literal behaviour of the code at the tag this project
would build from.

### The mapping this spike derived (and unit-tested against the documented schema)

Component-level results come from `xpum_diag_task_result_t`
(`core/include/xpum_structs.h`), a three-value enum:

```c
XPUM_DIAG_RESULT_UNKNOWN = 0   // never a positive result
XPUM_DIAG_RESULT_PASS    = 1
XPUM_DIAG_RESULT_FAIL    = 2
```

`runners/xpu-diag/decide.go` implements this mapping, and `decide_test.go`
exercises every branch against fixtures built from this schema (not from
captured output — see point 4):

| Condition | Runner verdict | Exit |
|---|---|---|
| Any component `result == FAIL` | Fail | 1 |
| No FAIL, but any component `finished == false` OR `result == UNKNOWN` | Error | 3 |
| Top-level JSON `errno != 0` (machinery failure) | Error | 3 |
| Every component `finished == true` and `result == PASS` | Pass | 0 |
| JSON unparseable, or `component_list` empty | Error | 3 |

The `UNKNOWN`-is-never-a-pass rule is the one most tempting to get wrong,
because `UNKNOWN` is the enum's **zero value** — the same trap Go's own
zero-value semantics create for exactly this kind of code, and worth naming
explicitly for whoever picks this up next.

### No Skip marker exists, and this spike does not invent one

`xpum_diag_task_result_t` has no fourth value for "not applicable to this
hardware" the way DCGM's diagnostic distinguishes a skip from a fail. Issue
#172 asked for "a declared skip with an `XPUM_DIAG_SKIP` marker" as one
possible mapping target; this investigation could not find a vendor-side
signal to hang that marker on. Inventing one — e.g. treating `UNKNOWN` as Skip
— would be this runner declaring a distinction the vendor's own API does not
offer, which is exactly the "a runner may only declare what it positively
established" rule this project holds every other runner to. `runners/xpu-diag`
therefore has no exit-2 path at all today. This is recorded as an open
question for real hardware, not resolved by assumption.

---

## 3. JSON output mapped to canonical metric names

`xpu-smi diag -j`'s component list covers 16 test types
(`xpum_diag_task_type_t`), spanning deployment checks through performance
tests. This spike did **not** register any of them in `pkg/contract/metrics.go`
— deferred deliberately, matching `docs/dev/new-testkind-playbook.md`'s own
rule that a parser test (and by extension, a registry entry) should wait for a
real hardware capture rather than be written against invented output. What
follows is the **written mapping** the issue asked for; turning it into code
is follow-up work gated on real captured output.

| `xpum_diag_task_type_t` | Shape | Recommended canonical treatment |
|---|---|---|
| `SOFTWARE_ENV_VARIABLES`, `SOFTWARE_LIBRARY`, `SOFTWARE_PERMISSION`, `SOFTWARE_EXCLUSIVE` | boolean-shaped deployment checks | carried by the overall exit code + marker, same as `dcgm-diag`'s level-1 deployment checks; not independently registered metrics |
| `HARDWARE_SYSMAN`, `INTEGRATION_PCIE` | boolean-shaped hardware checks | same as above |
| `MEMORY_ERROR` | counter | **shares the existing canonical name** `eccErrors` — same measurand, same `Equal 0` semantics as the NVIDIA/AMD runners |
| `PERFORMANCE_MEMORY_BANDWIDTH`, `PERFORMANCE_MEMORY_ALLOCATION` | dimensional (bandwidth) | likely shares `memoryBandwidthGBs` or a sibling already in the registry — **not confirmed**, because the JSON's actual field name and unit for this measurement were not observed (only the component's PASS/FAIL, not a numeric payload, was confirmed from source) |
| `PERFORMANCE_COMPUTATION`, `LIGHT_COMPUTATION` | dimensional (compute throughput) | likely shares a `Tflops`-suffixed name the way `gemm-sweep` registers one — same caveat as above |
| `PERFORMANCE_POWER` | dimensional (power) | Intel-specific unless a numeric payload and unit are confirmed against real output; would take a `W`-suffixed name |
| `MEDIA_CODEC`, `LIGHT_CODEC` | dimensional (fps) | **Intel-specific** — no NVIDIA/AMD equivalent is registered today (this project's runners have no media-codec test); would need a new registry entry, e.g. `mediaCodecFps` |
| `XE_LINK_THROUGHPUT`, `XE_LINK_ALL_TO_ALL_THROUGHPUT` | dimensional (fabric bandwidth) | **Intel-specific by fabric technology** (Xe Link, not NVLink or Infinity Fabric) even though it is conceptually the same *kind* of measurement as `nccl`'s `busBandwidthGBs`/`algBandwidthGBs`; needs its own name unless a real capture shows numerically comparable semantics, which this spike could not check |

Two rules from `CLAUDE.md` this mapping was written to respect, worth
restating because they are exactly where this class of runner tends to go
wrong: **a first-party metric whose value is a LABEL must be registered as
`Evidence`** the moment code emits it — none of the boolean/label-shaped
checks above should ever become a raw threshold target without that — and
**diagnostic integers may stay unregistered** until somebody actually writes a
threshold against one. `runners/xpu-diag/main.go` currently emits only
diagnostic-integer counts (`tests_run`, `tests_passed`, `tests_failed`,
`tests_incomplete`) for exactly this reason: they need no registry entry, and
they carry real information (a "13 of 16 ran" denominator, in the spirit of
`Applied` — #262) without asserting anything about a measurement this spike
never observed.

---

## 4. A Dockerfile that builds

`runners/xpu-diag/Dockerfile` builds. Verified locally with
`docker build -t xpu-diag:local runners/xpu-diag` (Docker 28.1.1, linux/arm64
build host) as of the commit that added it — not yet verified through this
project's own CI runner-images matrix, which builds it automatically the
moment this directory is touched by a PR, same as every other runner.

It is adapted from **Intel's own official build recipe**
(`builder/Dockerfile.builder-ubuntu22.04` + `BUILDING.md`, both at the
`v1.3.x` line) rather than reverse-engineered, for a concrete reason: this
project has no hardware to notice a build that silently produced something
subtly wrong, so deviating from a known-working recipe blind was avoided
wherever the official one was reachable. The one place it necessarily departs
is scope — Intel's recipe produces installer packages *and* the daemon; this
Dockerfile needs one binary (`xpu-smi`, via `build.sh -DDAEMONLESS=ON`) and
supplies its own minimal final stage rather than an installer.

Every upstream this Dockerfile clones with `git` is pinned to a commit and the
build asserts it, the same discipline every other runner in this repository
uses (`XPUMANAGER_REF`/`_SHA`, `GRPC_REF`/`_SHA`) —
`runners/pins_test.go`'s guards pass against it.

**What "it builds" does and does not prove.** It proves construction is
tractable — the Dockerfile compiles Intel's own build chain (grpc from
source, `metee`, `igsc`, then `xpumanager` itself) end to end and produces an
`xpu-smi` binary this project's own Go wrapper links against. It proves
nothing about whether the resulting container actually **runs** correctly
against a real Intel Data Center GPU: whether the Level Zero backend is found
at the path this image expects, whether the device-plugin-supplied `/dev/dri`
nodes are sufficient, whether any of the linking assumptions in
`runners/xpu-diag/README.md`'s "Still unverified" section hold. Those need
real hardware, which is exactly why this spike does not attempt to answer
them by simulation.

---

## 5. Recommendation

**Build the scaffolding now (done, in this spike); defer publishing until a
specific hardware commitment exists.** Reasoning, weighed against the other
two options the issue offered:

- **Not "build it now"** in the sense of publishing a tag: `runners/xpu-diag`
  has never executed against real silicon, its metric mapping (point 3) is
  unconfirmed against real JSON, and this project's own stated principle —
  quoted in the issue itself — is that "a runner that has never run is a
  liability rather than an asset." Publishing a tag nobody has verified is how
  a fleet gets certified by a runner that silently fails closed, or worse,
  passes everything by accident (the exact class of bug `compute-smoke v0.1.0`
  shipped).
- **Not "wait" with nothing produced**: this spike found real, sourced,
  non-obvious risk (the exit-code trap in point 2, the branch-currency trap in
  point 1) that is now written down and cannot bite a future implementation
  cold. It also found that **construction genuinely was the cheap part** —
  the Dockerfile builds against Intel's own official recipe on the first
  attempt that used it faithfully, and the exit-code/JSON decision logic is
  real, sourced, and unit-tested against the vendor's documented schema. Not
  capturing that work would mean re-deriving all of it, including the
  branch-currency trap, the next time somebody picks this up — possibly after
  `v1.3.x` has moved further from `v2.x`, or after Intel deprecates the line
  this depends on.
- **"Defer"** is therefore the only option that is honest about what remains:
  everything in `runners/xpu-diag/README.md`'s "Still unverified" list needs a
  real Intel Data Center GPU, and nothing about waiting for one changes what
  was already built. The moment hardware exists, the remaining work is
  genuinely **verification** — running the image, capturing real
  `xpu-smi diag -j` output, writing the parser test against that capture per
  `new-testkind-playbook.md`, registering the metrics point 3 identified,
  converting the single-device wrapper to the discovery+loop pattern
  `runners/devicefold_test.go`'s PENDING entry names — **not construction**,
  which is exactly what deliverable 4 asked this spike to leave behind. Filed
  as [#474](https://github.com/baldwinSPC/glimmer-burnin/issues/474), in the
  shape of `compute-smoke`'s own hardware-verification issues (#237/#242/#265).

### What would change this recommendation

A specific hardware commitment: access to an Intel Data Center GPU (Flex or
Max series) for long enough to capture real `xpu-smi diag -j` output at all
three levels, confirm the runtime linking assumption in point 1, and run the
image end to end at least once. Nothing else on this list is worth doing
without that — in particular, do not extend the metric mapping in point 3 or
add a `defaultRunnerImages` entry from documentation alone; both of those
failure modes are exactly what this project's own incident history
(`compute-smoke v0.1.0`, the clockprobe metric audit) warns against.
