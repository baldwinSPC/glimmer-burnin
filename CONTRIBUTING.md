# Contributing

Thanks for your interest in the Glimmer Burn-In Operator.

## Licensing

This project is **Apache-2.0**. By contributing you agree your contribution is
licensed under the same terms. Do **not** add dependencies under a copyleft
license (GPL / AGPL / LGPL / SSPL / MPL) — only permissive licenses (Apache-2.0,
MIT, BSD-2/3) are accepted. New dependencies must have their license verified in
the PR.

## The standalone invariant (enforced in CI)

The operator must never import the Glimmer control plane. CI fails any PR that
adds an import of `github.com/baldwinSPC/glimmer/…`. Integration happens through
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
no-glimmer-import guard, and CRD-drift check).
