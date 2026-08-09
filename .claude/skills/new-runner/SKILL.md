---
name: new-runner
description: Add a new TestKind and its runner image to glimmer-burnin. Use when adding a burn-in test kind, writing a runner image, or wiring a new measurement into the operator's contract — the exit-code rules, metric registration, the BURNIN_* environment contract, Dockerfile pinning, and the guards that enforce each.
---

# Adding a TestKind

**Read `docs/dev/new-testkind-playbook.md` first, in full.** It is the ordered
checklist, and it is the single source — this file deliberately does not restate
its rules, because two copies of a rule become two different rules.

```
Read docs/dev/new-testkind-playbook.md
```

What follows is only the working method: how to use that document, and what to do
when it is wrong.

## The one rule to carry in your head

**A runner may only declare what it positively established.** Every rule in the
playbook is that sentence applied to a place where it was once got wrong. When
you hit a case the playbook does not cover, derive the answer from it:

- Did you *measure* the part and find it short? → exit 1.
- Did something stop you measuring? → exit 3, whatever it was.
- Did you *look* and find nothing to measure? → `n/a`, or exit 2 **with a marker**.
- Did you not look? → omit the key. Never a zero.

## Order of work

Do not write the Dockerfile first. The order in the playbook is the order the
guards will judge you in, and each step is cheap to get right before the next
exists and expensive after.

1. Decide it is a kind at all — a new *cell* of an existing measurement is a
   `variants` axis, not a kind.
2. The runner's decision logic, in a **CUDA-free header** with a plain C++ test
   beside it. This is the part with the most verdict risk and the only part you
   can test without hardware.
3. Exit codes and the skip marker.
4. Metrics: register, alias if needed, parser test.
5. Dockerfile, pinned.
6. Wiring: `BuiltInKinds`, `NOTICE`, README, sample, publish workflow.

## Testing discipline — this repository's defining rule

**Watch every new test fail against the unfixed code before you trust it.** A
test that passes against broken code is the recurring defect class here.

Concretely: after writing a test, mutate the implementation to break the property,
run the test, confirm it **fails**, then revert. If it does not fail, the test is
blind — fix the test, do not move on.

Two traps that have caught previous sessions repeatedly:

- **Passing via a second route.** If the string you assert on also appears
  elsewhere in the output, deleting the code under test will not fail the test.
  Pin something only the code under test can produce.
- **A fixture that never reaches the branch.** A uniform fixture, a payload too
  short to hit a clamp, an all-ASCII string in a UTF-8 test. Check the test
  exercises what it claims.

## Verify

```sh
make generate manifests   # after ANY api/v1alpha1 change; commit the results
make lint
go test ./...             # includes the pins, contract and C++ guards
```

The guards named throughout the playbook run here. If one fails, read what it
says before changing anything: they are written to explain the failure they
prevent, not merely to report a mismatch.

## When the playbook is wrong

It is validated by use. If a step is missing, ambiguous, or turns out to be
wrong, **fix `docs/dev/new-testkind-playbook.md` in the same PR as the runner**.
A playbook that drifts from the guards is worse than none, because it will be
trusted.
