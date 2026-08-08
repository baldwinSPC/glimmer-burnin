# Security policy

## Reporting a vulnerability

**Please do not open a public issue for a security problem.**

Use GitHub's private vulnerability reporting:
[**Report a vulnerability**](https://github.com/baldwinSPC/glimmer-burnin/security/advisories/new).
It is private to the maintainers, needs no email exchange, and gives us a place
to work on a fix and coordinate disclosure with you.

If that form is unavailable to you, open a public issue containing **only** a
request for a private channel — no details, no reproduction — and a maintainer
will arrange one.

### What to expect

| | |
|---|---|
| Acknowledgement | within 3 working days |
| Initial assessment | within 10 working days |
| Fix or mitigation plan | communicated with the assessment |

This is a small project. Those are the targets we hold ourselves to, not a
contractual commitment. If you have not heard back within the acknowledgement
window, please assume the report was missed rather than ignored, and say so on
the advisory thread.

We will credit you in the advisory unless you ask us not to.

## What is in scope

- **The operator** — `cmd/`, `internal/`, `api/`, `pkg/`, and the manifests in
  `config/`.
- **The runner images** published under `ghcr.io/baldwinspc/glimmer-burnin-*`,
  including their build inputs in `runners/`.
- **Published tags.** These are immutable and are what a node's readiness gate
  actually executes, so a vulnerability in a published tag is in scope even
  after `main` has been fixed. Say which tag.

Things worth reporting that may not look like classic vulnerabilities, because
of what this operator does:

- Anything that lets a **runner image influence the operator's verdict** beyond
  its documented contract (exit code and `key=value` on stdout). A runner is a
  semi-trusted third-party image executing on a cordoned node; the operator
  parsing its output is a trust boundary.
- Anything that lets a `BurnInTest` reach **host state it did not declare**.
  `spec.runner.hostPaths` is the only sanctioned route to a node's filesystem,
  and it deliberately cannot ask the kubelet to create a path.
- **Privilege escalation through the operator's own RBAC** — it can cordon
  nodes, create pods, and read logs cluster-wide.
- A path that makes the operator **strand a cordon**, taking fleet capacity out
  of scheduling with nothing in the cluster recording why.
- Anything that causes a node to be **certified without being measured**. A
  fabricated pass is a security problem here even though it looks like a
  correctness one: the output of this operator is what admits hardware to a
  fleet.

## What is out of scope

- Vulnerabilities in NVIDIA drivers, the NVIDIA Container Toolkit, DCGM, CUDA,
  or any other upstream. Report those to their own projects; tell us if a
  version we pin is affected and we will move the pin.
- Findings that require an already-compromised cluster-admin credential.
- Denial of service achieved by running the operator's own tests — burning in
  hardware is what it is for. Load that escapes `maxConcurrentNodes` **is** in
  scope, because the concurrency cap is the promise that only N nodes are
  occupied at once.
- Reports generated solely by a scanner, with no reachable code path shown.
  CI runs `govulncheck`, which reports only vulnerabilities in code paths this
  project actually reaches; if you have found one it missed, please say how it
  is reached.

## Supported versions

Only the latest minor release receives security fixes. This project has not yet
reached 1.0 and does not maintain release branches.

| Version | Supported |
|---|---|
| latest minor | yes |
| anything older | no — upgrade |

A fix ships as a new tag. **We never re-publish an existing tag**, including for
a security fix: published tags are immutable because a node's readiness gate
pins one, and silently changing what a tag contains would change every node's
verdict with no audit trail. Pin the new tag.
