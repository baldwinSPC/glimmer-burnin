# CLAUDE.md

Guidance for AI assistants (Claude and others) working on **glimmer-burnin**.

---

## What this is

A vendor-neutral, Kubernetes-native **burn-in operator**: it runs acceptance
tests against accelerator nodes (compute smoke, thermal soak, NCCL, RDMA
bandwidth, DCGM diagnostics), evaluates the results against declared
thresholds, and exports verdicts to an external consumer.

It is a standalone project, published under **Apache-2.0**. It is designed to be
useful to anyone running GPU fleets, not only to its first consumer.

---

## The standalone invariant (the most important rule here)

**This repository must never import the Glimmer control plane.**

CI fails any PR adding an import of `github.com/baldwinSPC/glimmer/…`. There is
no exception, including "just for a type" or "just in a test".

- Dependency direction is one-way: Glimmer MAY depend on this project; this
  project MUST NEVER depend on Glimmer.
- Integration happens through the `BurnInSink` export contract — a delivery
  envelope over a webhook — not through shared code.
- This is what allows the repo to be open-sourced and adopted independently. If
  you need something from Glimmer, the answer is to widen the contract, not to
  add an import.

If a task appears to require reaching into Glimmer, stop and raise it rather
than working around the guard.

---

## Cross-repo context

The Glimmer solution spans several repositories. The canonical registry and the
cross-repo rules live in the `baldwinSPC/glimmer` repo's own `CLAUDE.md`, which
is **private** — so the rules that matter here are restated below rather than
linked, since public contributors cannot read the original.

| Repo | Visibility | Role |
|------|------------|------|
| `baldwinSPC/glimmer` | private | Control plane, and the planning home for the whole solution |
| `baldwinSPC/glimmer-burnin` | this repo | The burn-in operator |
| `baldwinSPC/glimmer-releases` | public | Release artifacts only; never source |

Rules that apply to work in this repo:

- **Design proposals (GEPs) live in `glimmer`**, at `docs/geps/NNNN-name.md`,
  even when the code lands here. This project's design is GEP-0178. The GEP
  states which repo owns which piece.
- **Implementation issues are filed here**, in the repo where the code lands,
  and linked to the workstream's tracking issue in `glimmer`.
- The dependency and license rules apply to every repo in the solution, and
  most strictly to this one, because it is the one that gets published.

---

## Licensing rules (strict)

- **Permissive only: Apache-2.0, MIT, BSD-2-Clause, BSD-3-Clause.**
- **Never** add a dependency under a copyleft license — GPL, AGPL, LGPL, SSPL,
  MPL, or Sleepycat. This is not a preference; a copyleft dependency would make
  the project unpublishable.
- Verify a new dependency's license **in the PR description**. If you cannot
  verify it, ask before touching `go.mod`.
- Where an upstream project is dual-licensed, state explicitly which option we
  consume it under, in `NOTICE` (e.g. `linux-rdma/perftest` is GPL/BSD
  dual-licensed and must be consumed under the **BSD** option).
- Do not reproduce substantial blocks of third-party source. Header-only
  libraries compiled in (e.g. CUTLASS) are fine and are recorded in `NOTICE`;
  redistributing their source is not.

Runner images have an additional constraint: they must ship **no
NVIDIA-licensed redistributable libraries**. Builds are multi-stage so the CUDA
toolchain is used at build time only, and the Dockerfile fails the build if
`ldd` finds `libcuda`/`libcudart`/`libnv` in the final binary. `libcuda.so.1`
is injected at runtime from the host driver by the NVIDIA Container Toolkit.

---

## Build, test, lint

```sh
make generate    # deepcopy methods (controller-gen)
make manifests   # CRDs + RBAC
make test        # unit tests
make lint        # gofmt + go vet
make build       # binary
make all         # generate manifests fmt vet test build
```

After changing anything in `api/v1alpha1`, run `make generate manifests` and
**commit the results** (`zz_generated.deepcopy.go`, `config/crd/*`,
`config/rbac/*`). CI checks for drift.

CI (`.github/workflows/ci.yml`) runs build, vet, gofmt, test, the
no-glimmer-import guard, and the CRD drift check.

---

## Layout

```
api/v1alpha1/        CRD types: BurnInTest, BurnInProfile, BurnInRun,
                     BurnInSink, NodeFingerprint
internal/controller/ the BurnInRun reconciler
internal/verdict/    pure threshold evaluation (no k8s, no I/O)
runners/             per-TestKind runner images, one directory each
config/crd|rbac|samples/   generated manifests and examples
```

---

## Conventions

- **Vendor neutrality lives in the reconciler.** Accelerator- and
  vendor-specific behaviour belongs in *runner images*, never in controller
  branches. Adding support for a vendor should mean adding a runner, not adding
  an `if nvidia {}`.
- **Every new `TestKind`** ships with a default runner image, a result parser,
  and a unit test for that parser.
- **Verdict logic fails closed.** A threshold naming a metric the runner did not
  emit is a FAILURE, not a pass — a missing measurement must never silently
  satisfy acceptance. Keep it that way; `internal/verdict` is deliberately pure
  so this is testable in isolation.
- **`Error` is not `Fail`.** An infrastructure error (image pull, scheduling)
  must stay distinguishable from a hardware verdict. Do not collapse them.
- Small, focused PRs, with tests. CI must be green.

### Runner images

- A runner's contract is its **exit code plus `key=value` metrics on stdout**.
  Exit 0 = pass, 1 = fail, 2 = skip (not applicable to this hardware). The skip
  path matters: a node that cannot run a test must skip cleanly, not report a
  failure.
- Runner images are published **manually** via the `publish-runner` workflow
  (`workflow_dispatch`), never automatically. A runner image is executed by a
  node's readiness gate, so a bad tag degrades a whole fleet at once. Publish
  only after the kernel has been verified on real hardware.
- Published tags are **immutable**. Never republish a tag a gate pins; cut a new
  version. Silently changing a tag changes every node's verdict with no audit
  trail.
- Arch targets are deliberate. `sm_121a` emits a cubin for GB10 only, with no
  PTX fallback, so a pass proves the real instruction path ran rather than an
  emulated one. Do not "helpfully" widen an arch target to make a build succeed.

---

## Tracking work

Follow Kubernetes community conventions. Do not leave findings only in chat or
as bare TODOs in code:

- Unfinished or deferred work → a GitHub issue **in this repo**, linked to the
  GEP tracking issue in `glimmer`.
- Bugs, security issues, untested paths → a GitHub issue with a minimal
  reproduction, written so someone without the session's context can act on it.
- New workstreams or significant design changes → a GEP in `glimmer`
  (`docs/geps/`), not a design document here.

---

## Updating this file

Keep it current. When a convention, invariant, or tool changes, update the
relevant section so the next session starts with accurate context.
