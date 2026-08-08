# Getting help

- **A question about how to use it** — open a
  [Discussion](https://github.com/baldwinSPC/glimmer-burnin/discussions), or an
  issue if Discussions are not enabled.
- **A bug** — open an [issue](https://github.com/baldwinSPC/glimmer-burnin/issues/new/choose).
  The bug template asks for the operator version, the runner image tag, and the
  `BurnInRun` status, because a verdict cannot be diagnosed without knowing
  which image produced it.
- **A security problem** — do not use either. See [SECURITY.md](SECURITY.md).
- **A new test kind, or a new vendor** — there are issue templates for both.
  They ask the questions a runner implementation needs answered, so filing one
  is most of writing the work order.

## What this project can and cannot help with

It can help with the operator, the runner images, and the verdict contract.

It cannot help with your hardware being faulty — that is what a failing verdict
means, and the runner's own output is the evidence. If you believe a verdict is
**wrong**, that is a bug and we want to hear it: attach the runner's stdout and
the `TestResult`, because the two together are what the verdict was derived
from.
