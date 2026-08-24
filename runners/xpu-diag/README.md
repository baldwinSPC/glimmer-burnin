# `xpu-diag` runner — Intel XPU Manager diagnostics

> **UNVERIFIED.** This runner has never executed against a real Intel Data
> Center GPU. Nobody on this project has one. It exists as the code
> deliverable of a spike (#172) whose written recommendation is
> [docs/dev/spike-intel-xpu-manager.md](../../docs/dev/spike-intel-xpu-manager.md)
> — read that first. No default image is registered for the `xpu-diag`
> `TestKind` (`pkg/runnerimages` deliberately carries none), no tag has ever
> been published, and nothing in this repository points a sample or a profile
> at it. A `BurnInTest` naming this kind fails at plan time unless
> `spec.runner.image` is set explicitly — the existing, intended behaviour for
> a kind with source but no verified default.

Runner image for the `xpu-diag` [`TestKind`](../../api/v1alpha1/burnintest_types.go).
It wraps Intel XPU Manager's diagnostic suite — `xpu-smi diag -d <id> -l <level> -j`
— and translates its JSON into the burn-in runner contract: `key=value` metrics
on stdout, then one marker line, then an exit code. It fills the same role for
Intel that `dcgm-diag` fills for NVIDIA: the vendor's own opinion about whether
a part is healthy, translated into a verdict.

## The exit-code trap this runner exists to avoid

`xpu-smi diag`'s own **process exit code does not encode whether any
diagnostic component passed or failed.** Verified by reading the CLI's own
source (`cli/src/comlet_base.h`'s `setExitCodeByJson`, `cli/src/comlet_diagnostic.cpp`'s
success path) at the `v1.3.x` line: the exit code reflects only a MACHINERY
failure — a bad `-d`/`-l` argument, a task that never started — carried in the
JSON's own top-level `errno` field. A component reporting FAIL inside a
successfully-run diagnostic changes nothing about the process's own exit
status. Issue #172 named this risk up front by analogy with `dcgmi diag`
exiting 226 on a mere warning; here the shape of the trap is the opposite
direction — trusting `$?` would silently report `Pass` for hardware that
genuinely failed.

This runner therefore never reads `$?`. It parses the JSON itself
(`decide.go`) and decides the verdict from `component_list[].result` and
`component_list[].finished`, which is real, sourced behaviour from the
vendor's own code — not from a hardware capture, since none exists yet.

## Output contract

| Result | Marker on stdout | Exit |
|--------|------------------|------|
| Pass | `XPU_DIAG_PASS: <reason>` | 0 |
| Fail | `XPU_DIAG_FAIL: <reason>` | 1 |
| Unjudged | `XPU_DIAG_ERROR: <reason>` | 3 |

There is **no Skip path**. `xpum_diag_task_result_t` (the vendor's own
component-result enum) has exactly three values — `UNKNOWN`, `PASS`, `FAIL` —
with no "not applicable to this hardware" state the way DCGM's diagnostic has.
Inventing an `XPU_DIAG_SKIP` marker here would mean this runner declaring a
distinction the vendor's own API does not offer, which is exactly the kind of
unearned positive declaration this project's runners are not allowed to make.
A `component_type` the wrapper does not expect, or a component that never
finishes, is reported `Error` (unjudged) rather than guessed at.

A component that reports `finished: false`, or `finished: true` with
`result: UNKNOWN` (the enum's zero value), is **never** read as a pass. Both
are folded into `Error`: the diagnostic asked the question and got no usable
answer, and only a `FAIL` anywhere in the component list makes this a `Fail`
result at all — see `decide.go`'s `decide()` for the full logic and why each
branch is there.

## Configuration

| Variable | Default | Meaning |
|---|---|---|
| `BURNIN_XPU_SMI_BIN` | `xpu-smi` | binary name/path on `PATH` |
| `BURNIN_XPU_DEVICE_ID` | `0` | the `-d` argument |
| `BURNIN_XPU_DIAG_LEVEL` | `1` | the `-l` argument (1/2/3) |
| `BURNIN_DURATION_SECONDS` | 600s | a soft deadline on the whole invocation |

There is **no duration-derived level selection** the way `dcgm-diag` has one.
`dcgm-diag`'s README states measured per-level time budgets from real DCGM
runs on real hardware; this runner has no equivalent measurement to state, and
inventing one would be exactly the "test asserting output you imagined"
mistake `docs/dev/new-testkind-playbook.md` warns against. `BURNIN_DURATION_SECONDS`
here only bounds worst-case runtime; it does not choose the level.

## Multi-device

**Single device only, today.** `xpu-smi diag` targets one `-d <deviceId>` per
invocation — unlike `dcgmi diag`, which enumerates every device on the node
itself — so a genuine Node-scope verdict (every accelerator, gated on the
worst) needs this wrapper to discover every device (`xpu-smi discovery -j`)
and loop, folding to the worst result. That conversion is deferred rather than
guessed at, because the discovery JSON's shape has not been verified against
real output either. See `runners/devicefold_test.go`'s PENDING entry for this
directory.

## Licensing

Intel XPU Manager is **MIT**-licensed (verified against `LICENSE.md` at the
`v1.3.8` tag, and against the GitHub API's own licence classification). The
build dependencies compiled from source for this image — `grpc`, `metee`,
`igsc` — are all **Apache-2.0**. The Level Zero loader shipped in this image
(`libze_loader`, from `oneapi-src/level-zero`'s own release `.deb`) is **MIT**.
None of this is NVIDIA- or AMD-licensed, and none of it is a
non-redistributable vendor binary the way DCGM's binaries are — Intel's stack
is permissively licensed all the way down to `xpu-smi` itself, which is why
this Dockerfile builds the tool from source and ships the resulting binary,
unlike `dcgm-diag`, which cannot.

**Unverified**: whether the GPU-specific Level Zero *backend*
(`intel-level-zero-gpu`, part of Intel's compute-runtime, also MIT but tied to
the installed driver version) belongs baked into this image or supplied by the
host. This image currently ships only the vendor-neutral loader and expects
the backend from the host, by analogy with how `libcuda.so.1` is injected at
runtime by the NVIDIA Container Toolkit for every other runner in this
project — but that analogy has not been checked against a real Intel GPU
node, and the in-tree `i915`/`xe` kernel driver can never be baked into any
container regardless of vendor.

See [docs/dev/spike-intel-xpu-manager.md](../../docs/dev/spike-intel-xpu-manager.md)
point 1 for the full licensing investigation this section summarizes.

## Build

This image is **amd64-only**: Intel Data Center GPU hosts are x86_64, and the
Dockerfile's Level Zero `.deb` URLs and CMake installer are amd64-specific —
same reasoning as this repository's ROCm runners, for a third vendor. Build
for the real target platform explicitly:

```sh
docker build --platform linux/amd64 -t xpu-diag:local runners/xpu-diag
```

See the spike report (point 4) for exactly what has and has not been verified
about this build as of the commit that added this file, and CI's own
runner-images matrix, which builds it natively on an amd64 host the moment
this directory is touched by a PR — the same way every other runner here is
verified.

## Still unverified

Everything past "it builds":

- The actual `xpu-smi diag -j` JSON shape, on a real device — this runner's
  parsing logic is checked against the vendor's documented C API header
  (`xpum_structs.h`), not against captured output.
- Whether the image's runtime linking is correct at all — does `xpu-smi`
  actually find its Level Zero backend the way this README assumes.
- Per-level timing, so duration-derived level selection (like `dcgm-diag` has)
  could be added honestly instead of guessed at.
- Multi-device folding (see above).
- Skip semantics, if the vendor's API turns out to offer a way to distinguish
  "not applicable" from "unknown" that this investigation did not find.

None of these are fixable from a laptop. They need an Intel Data Center GPU —
tracked as [#474](https://github.com/baldwinSPC/glimmer-burnin/issues/474).
