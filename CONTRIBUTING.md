# Contributing

Thanks for your interest in the Glimmer Burn-In Operator.

## Licensing

This project is **Apache-2.0**. By contributing you agree your contribution is
licensed under the same terms. Do **not** add dependencies under a copyleft
license (GPL / AGPL / LGPL / SSPL / MPL) — only permissive licenses (Apache-2.0,
MIT, BSD-2/3) are accepted. New dependencies must have their license verified in
the PR.

## The standalone invariant (enforced in CI)

The operator must never import its control-plane consumer. CI fails any PR that
adds an import of a control plane's module path. Integration happens through
the `BurnInSink` export contract, not code. Keep it that way — it's what lets
this repo be published and adopted on its own.

## Development

```sh
make generate    # deepcopy (controller-gen)
make manifests   # CRDs + RBAC
make test        # unit tests
make lint        # gofmt + go vet
make docker-build IMG=...   # multi-arch (arm64 + amd64)
```

- Run `make generate manifests` after changing any `api/v1alpha1` type and commit
  the results (`zz_generated.deepcopy.go`, `config/crd/*`).
- Every new `TestKind` should ship with a default runner image and a result parser,
  plus a unit test for the parser.
- Keep the controller vendor-neutral: vendor/accelerator specifics belong in runner
  images, not in the reconciler.

## PRs

Small, focused PRs. Include tests. CI must be green (build, vet, test, the
standalone-import guard, and CRD-drift check).

---

## Reporting a security problem

Not here. See [SECURITY.md](SECURITY.md) — use private reporting. This operator
cordons nodes and runs semi-trusted images on production hardware, so a public
issue is the wrong first move.

## Sign-off

Commits do **not** currently require a `Signed-off-by` line, and there is no CLA.
By contributing you agree your contribution is licensed under Apache-2.0, as
stated above.

A DCO is the likely posture as the project grows, since it is standard in this
ecosystem and lighter than a CLA. It is deliberately not enforced yet: turning
on a blocking check is a decision about provenance, not a build change, and it
would reject every PR already in flight. If you want to sign off anyway,
`git commit -s` is welcome and costs nothing.

## Releases

Releases are cut by a maintainer, and **publishing is manual on purpose**.

- **A runner image is executed by a node's readiness gate**, so a bad tag
  degrades a whole fleet at once. Images are published through the
  `publish-runner` workflow (`workflow_dispatch`), never automatically, and only
  after the kernel has been verified on real hardware.
- **Published tags are immutable.** A tag is never re-published — not for a bug
  fix, not for a security fix. A node's gate pins a tag, so silently changing
  what it contains would change every node's verdict with no audit trail. Cut a
  new version.
- The operator and every runner are pinned to the same version at release, so
  "what was this node measured by" has one answer.

## Labels and triage

| Label | Means |
|---|---|
| `bug` | it does something it should not |
| `enhancement` | it does not do something it should |
| `documentation` | the words are wrong or missing |
| `new-runner` / `new-vendor` | filed from those templates; the template fields are the work order |
| `reporting`, `cli`, `matrix`, `soak`, `oss-launch`, `test-factory` | workstream, matching the milestones |
| `vendor:amd`, `vendor:intel` | vendor-specific |

An issue is not triaged until someone can act on it without the filer present.
For a bug that means a reproduction; for a runner proposal it means the template
answered, especially the FAIL-versus-ERROR question, which is the one that
decides whether a verdict can condemn a node.

## Design proposals

Significant design changes go in a GEP, in this project's private
design-tracking repository under `docs/geps/`, not a design document here. This
project's own design is GEP-0178. Implementation issues are filed **here**,
where the code lands.
