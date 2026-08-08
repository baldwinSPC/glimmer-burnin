<!--
Describe the change and WHY. The "why" outlives the diff, and this project's
review comments are mostly about intent rather than mechanics.

If this fixes an issue, say "Closes #NNN".
-->

## What and why

## Checks

<!-- Delete rows that genuinely do not apply. Ticking a box you did not check is
     worse than leaving it blank. -->

- [ ] `make test` passes, and any new test was **watched failing** against the
      unfixed code first. A test that passes against broken code is the
      recurring defect class here.
- [ ] **Dependencies**: no new module, or the new one's licence is named below
      and is permissive (Apache-2.0, MIT, BSD-2/3, ISC). CI enforces this over
      the whole module graph; naming it here is how a human reviews it.
- [ ] **`api/v1alpha1` changed?** `make generate manifests` was run and the
      results committed. CI blocks on drift.
- [ ] **Runner image changed?** State the hardware verification status below.
      A runner image is executed by a node's readiness gate, so a bad tag
      degrades a whole fleet at once.
- [ ] **New `TestKind`?** It has a parser entry where the runner's keys are not
      merely a different spelling of the canonical names, a unit test for that
      parser, and it is in `v1alpha1.BuiltInKinds`.
- [ ] Nothing here imports `github.com/baldwinSPC/glimmer/…`.

### New dependency licence

<!-- module@version — licence — where you verified it. Delete if none. -->

### Hardware verification

<!-- Delete if this touches no runner.

     Which part, what was run, and what the output was. "Builds" is not
     verification: an `sm_120a` image LOADS on a CC 12.1 GB10 and trips a
     device-side assert inside the kernel. If it has NOT been verified on
     hardware, say so — that is a reason not to publish, not a reason not to
     merge. -->

- Verified on:
- Not verified — publishing blocked until:
