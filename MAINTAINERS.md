# Maintainers

| Name | GitHub | Areas |
|---|---|---|
| Matt Baldwin | [@baldwinSPC](https://github.com/baldwinSPC) | everything |

This project currently has one maintainer. That is worth stating plainly rather
than implying a rota that does not exist: response times are best-effort, and a
PR may wait.

## What a maintainer does here

- Reviews and merges.
- Decides what goes in a release, and cuts it. Published tags are immutable and
  a runner tag is executed by a node's readiness gate, so a bad tag degrades a
  whole fleet at once — publishing is deliberately manual.
- Owns the [security process](SECURITY.md).
- Owns the two rules that are not negotiable in review: the **standalone
  invariant** (this repository must never import the Glimmer control plane) and
  the **licence policy** (permissive only; see `hack/licenses-allowed.txt`).
  Both are enforced by CI, and both would make the project unpublishable if
  they were relaxed.

## Becoming a maintainer

There is no formal ladder yet. Sustained, reviewed contribution is the path, and
the honest prerequisite is judgement about the project's central distinction —
that an `Error` is not a `Fail`, and that a measurement nobody took is not a
measurement that came back clean. Most of the review comments in this repository
are about that one thing.
