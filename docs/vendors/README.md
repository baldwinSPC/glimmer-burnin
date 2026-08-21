# Running on a mixed fleet

**Vendor support is data, not code.**

That is the one thing worth understanding before anything else here, because it
predicts what will and will not work on hardware nobody in this project has
touched. Adding an accelerator means shipping a **runner image** and, if the
node is labelled unusually, a **fingerprint entry**. It does not mean a
controller change. There is no `if nvidia {}` anywhere in
`internal/controller`, and a test parses every file in that package on every CI
run to keep it that way.

The practical consequence: **this operator will run a test on hardware it has
never heard of**, provided you can point it at an image that speaks the runner
contract. What it will not do is invent one for you.

## Three states, and the difference matters

| State | Meaning |
|---|---|
| **shipped** | An image is published and has been run on real hardware of this vendor. |
| **in tree** | The runner and its image build exist and are green in CI, but no image is published and **nothing has run on hardware**. A `BurnInTest` naming it fails at plan time asking for an explicit `spec.runner.image`. |
| **—** | No image, and none planned in this milestone. Point the kind at your own image, or use a vendor-neutral kind instead. |

Nothing on these pages is described as supported because it *ought* to work.
Where a runner has not been executed on a vendor's silicon, it says so.

## The kinds that work anywhere

Four TestKinds touch no accelerator at all and run on any node in any fleet —
including nodes with no GPU, no RDMA, and no vendor stack:

| Kind | Measures | State |
|---|---|---|
| `memory-stress` | host DIMM stress (stressapptest) | shipped |
| `ib-write-bw` | RDMA write bandwidth over verbs | shipped |
| `tcp-baseline` | plain TCP throughput and retransmits | in tree |
| `disk-io` | storage throughput and latency, direct I/O | in tree |

These are the natural first tests in any profile, and on a fleet whose
accelerator vendor has no runner yet they may be the *only* tests — which is
still a great deal better than nothing. A node that fails `memory-stress` or
shows TCP retransmits does not need a GPU test to be worth pulling from service.

`ib-write-bw` deserves particular note: it measures the wire through RDMA verbs
and never opens a CUDA context, so it works identically on NVIDIA, AMD and Intel
fabric. The CUDA variant is a separate kind, `gpudirect-rdma`.

## One profile, several vendors

`spec.runner.imagesByVendor` selects the image by the accelerator vendor of the
node the test lands on. The vendor comes from the node's `NodeFingerprint`,
derived from the DNS domain of the labels a device plugin wrote — or, on a node
with no labels at all, from the domain of its `status.allocatable` extended
resources.

```yaml
apiVersion: burnin.glimmer.ai/v1alpha1
kind: BurnInTest
metadata:
  name: memory-bw-any-vendor
spec:
  kind: memory-bw
  runner:
    imagesByVendor:
      - vendor: nvidia
        image: ghcr.io/baldwinspc/glimmer-burnin-memory-bw:v0.6.0
      - vendor: amd
        image: registry.example.com/our-rocm-memory-bw:v1
```

Resolution is **explicit `image` > `imagesByVendor` > the kind's built-in
default**. A node whose vendor has none of the three is a **plan-time error**
naming the vendor and the field — never a silent skip, because a node quietly
not tested is how a fleet gets certified without being measured.

It is a list rather than a map keyed by vendor for a reason worth knowing if you
are wondering why the YAML is shaped this way: Kubernetes cannot validate map
keys, so a typo'd `nvidai` would be accepted and then resolve to no image on
every NVIDIA node you have. As a list, `vendor` is an enumerated field the
apiserver rejects at apply time.

### The alternative: one profile per vendor

Perfectly valid, and preferable if your vendors want genuinely different
*tests* rather than the same test with a different image. Target them with
`spec.target.nodeSelector` on the device plugin's own labels. The cost is that
N profiles drift apart; the benefit is that they can differ in more than an
image.

## Pointing a kind at your own image

Every TestKind accepts `spec.runner.image`. Anything that obeys the runner
contract works:

```
exit 0   pass
exit 1   fail — measured, and fell short
exit 2   skip, WITH a `<SOMETHING>_SKIP:` marker on stdout
other    error — machinery, not a hardware verdict
```

plus `key=value` lines on stdout for metrics. That is the whole interface. See
[the runner contract](../runner-contract.md) and any directory under `runners/`
for worked examples.

This is how **proprietary vendor suites** are used. AGFHC (AMD) and `hl_qual`
(Habana/Intel) cannot be redistributed by this project — its licence policy is
permissive-only and those are neither — but a site that has licensed them can
wrap them in an image and point a kind at it. `dcgm-diag` already documents this
pattern for site-supplied DCGM, and it is the same shape for any of them.

## Thresholds do not port

Not across vendors, not across SKUs, and not from a datasheet.

This project pins thresholds from **measured** baselines, and the worked example
is its own fleet: a fabric whose optics are labelled 200 Gb/s, whose host
attachment cannot exceed roughly 100 Gb/s. A threshold taken from the label
would have failed every healthy node — and, worse, would have read as a hardware
verdict rather than as the authoring mistake it was.

Run a profile with no thresholds first, look at what the fleet actually reports,
then gate. Counters (`ioErrors`, `tcpRetransmits`, `eccErrors`) are the
exception: a healthy part reports exactly zero of them, and `Equal 0` is safe on
day one precisely because the target is not a rate.

## Adding a vendor

Widening the CRD's `VendorImage.Vendor` enum — the step that lets a
`BurnInTest` actually select an `imagesByVendor` entry for a new vendor — is
a **name and a PCI ID**, not a controller change. Habana/Gaudi is the worked
example: it needed

1. Its name added to `pkg/contract.AcceleratorVendors`, the single list four
   other places are held to agreement with (that list's own doc comment
   names all four and how each is checked): the CRD's kubebuilder `Enum`
   marker, `pkg/hostinfo`'s PCI-vendor-ID table, and
   `runners/fingerprint-probe`'s own copy of that table (a second
   implementation forced by the image build, issue #250 — a runner's
   non-test sources cannot import this repo).
2. Its PCI vendor ID (`0x1da3`) added to both of those tables, so the name
   can actually be detected from hardware rather than only accepted by the
   apiserver.
3. `make generate manifests`, and the chart's CRD mirror kept in sync
   (`hack/chart` guards that).

Nothing in `internal/controller` needed to change, and that is the point:
`vendorFromDomain` derives a node's vendor from whichever DNS domain
published a device fact (`habana.ai/gaudi` → "habana"), with no lookup table
and no branch on the value, so a vendor this project has never named still
gets **fingerprinted** correctly — it just cannot be the `vendor:` of an
`imagesByVendor` entry until the CRD enum accepts it, which is exactly the
three-step change above.

What adding the name does NOT do: it does not make a `spec.runner.image`
appear for that vendor. That is a separate, larger piece of work — a runner
image, a result parser, `runners/pins_test.go`'s
`vendorVariantSuffixes`/`kindForDir` (already primed with a `-gaudi` suffix
mapping to `habana`, waiting for the first directory), and everything else
[the new-TestKind playbook](../dev/new-testkind-playbook.md) walks through.
Until then, a site with Habana hardware points a kind at its own image the
same way any proprietary suite is used (below), naming `vendor: habana`
explicitly rather than relying on a built-in default that does not exist.

## Per-vendor pages

- [NVIDIA](nvidia.md) — the reference implementation, and the only vendor with
  runners verified on hardware
- [AMD](amd.md) — vendor-neutral kinds today, ROCm images planned
- [Intel](intel.md) — vendor-neutral kinds today, a diagnostic spike planned
