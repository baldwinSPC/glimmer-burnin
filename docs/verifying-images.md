# Verifying published images

Every image this project publishes is signed with [cosign][] keyless signing.
There is no public key to distribute and no private key to leak: the signature
is bound to a short-lived certificate that records **which workflow, in which
repository, at which ref** produced it.

[cosign]: https://github.com/sigstore/cosign

That binding is the point. A signature that only proves "something signed this"
is worth very little. This one lets you gate on "this came from
`glimmer-burnin`'s own publish workflow" — as distinct from "someone with push
access to a registry".

## Verify

```sh
cosign verify \
  --certificate-identity-regexp '^https://github\.com/baldwinSPC/glimmer-burnin/\.github/workflows/publish-.*\.yml@refs/' \
  --certificate-oidc-issuer 'https://token.actions.githubusercontent.com' \
  ghcr.io/baldwinspc/glimmer-burnin:v0.6.4
```

The same command works for a runner image:

```sh
cosign verify \
  --certificate-identity-regexp '^https://github\.com/baldwinSPC/glimmer-burnin/\.github/workflows/publish-.*\.yml@refs/' \
  --certificate-oidc-issuer 'https://token.actions.githubusercontent.com' \
  ghcr.io/baldwinspc/glimmer-burnin-compute-smoke:v0.6.0
```

Success prints the certificate subject and the claims. Failure exits non-zero —
which is what you gate on.

### Pin the identity harder if you want to

The regexp above accepts either publish workflow and any ref. To accept only the
operator workflow, and only from a tag:

```sh
--certificate-identity-regexp '^https://github\.com/baldwinSPC/glimmer-burnin/\.github/workflows/publish-operator\.yml@refs/tags/'
```

Tightening it is safe and is encouraged. Loosening it — dropping
`--certificate-oidc-issuer`, or matching any repository — gives up most of what
the signature is for.

## What is signed

**Digests, never tags.**

Published tags are immutable by policy here, but a signature over a tag would be
a signature over a mutable *pointer*: it would still verify after the tag moved,
which is exactly the situation a signature exists to detect. The signature covers
the content a node actually pulls.

For a runner image there are **three** signatures:

| | |
|---|---|
| the manifest list | what a node's readiness gate references |
| `linux/amd64` child | separately pullable by digest |
| `linux/arm64` child | separately pullable by digest |

A mirror, a registry proxy, or anything that resolves a platform before pulling
will reference a child directly, and a signature on the list alone says nothing
about those. Whichever way the image is reached, the thing that is pulled is
signed.

## Gating a cluster on it

Verification at pull time is a cluster-policy concern rather than something this
operator does. Both [Kyverno][] and [Sigstore Policy Controller][] can enforce it
admission-side using the identity above.

[Kyverno]: https://kyverno.io/docs/writing-policies/verify-images/
[Sigstore Policy Controller]: https://docs.sigstore.dev/policy-controller/overview/

**This is worth doing for this project specifically.** A runner image is
executed by a node's readiness gate, against hardware a site is deciding whether
to accept. An unsigned image in that position decides acceptance.

## Also published

Alongside the signature, both workflows publish **SLSA provenance** and an
**SBOM**, attached to the image and inspectable with:

```sh
cosign download attestation ghcr.io/baldwinspc/glimmer-burnin:v0.6.4
```

Attestations answer "how was this built". The signature answers "did this come
from where it claims to". You want both, and they are not substitutes.

## If verification fails

Do not deploy the image. In order of likelihood:

1. **The tag is older than signing.** Images published before this landed carry
   no signature. Check the release notes for the first signed version; there is
   no way to add a signature retroactively without republishing a tag, and this
   project never republishes a tag.
2. **The identity regexp does not match.** The workflow filename or the ref
   pattern may have changed. Inspect what is actually there with
   `cosign verify --certificate-identity-regexp '.*' --certificate-oidc-issuer 'https://token.actions.githubusercontent.com' <image>`
   and compare — but do not leave it that loose.
3. **The image genuinely is not from this repository.** That is the case this
   document exists for.
