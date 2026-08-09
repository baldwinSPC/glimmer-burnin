# fingerprint-probe

What the hardware says about itself: PCI vendor and device IDs, read from a
read-only sysfs mount.

Every other way this operator learns a node's identity depends on **something
else having described it**. `NodeFingerprint` derives a vendor from the DNS
domain of labels a device plugin wrote, and falls back to the domain of the
node's extended resources. Both are good, and both are silent on a node where
the plugin never came up, NFD was never installed, or the labels are wrong —
where the node simply looks like it has no accelerator at all.

PCI IDs are reported by the device, over a bus, with no software having chosen
to describe it. That is what this runner reads.

## It reports; it does not judge

There is no exit 1. The runner passes when it could look and skips when it could
not, and a verdict — if a profile wants one — comes from a threshold.

That distinction decides what may be gated:

- **`acceleratorCount` may be.** A node that should hold eight cards and reports
  seven has lost one, and that is a hardware fault.
- **The identity strings may not.** `acceleratorVendors Equal nvidia` fails an
  AMD node for being AMD and reads as a hardware verdict.
  `pkg/verdict.ValidateThresholds` reports it at authoring time, because
  `pkg/contract` marks them `ThresholdUse: Evidence`. Fleet-composition rules
  belong in `spec.target.nodeSelector`, where they are targeting rather than
  accusation.

## Contract

```
exit 0                                     the hardware was read
exit 2  FINGERPRINT_PROBE_SKIP: <reason>   the sysfs mount is not there
exit 3  FINGERPRINT_PROBE_ERROR: <reason>  machinery
```

**Burst-only.** It reads sysfs once; it ignores `BURNIN_DURATION_SECONDS`
because there is nothing to hold a node for. Pair it with a soak kind in a
profile when you want both.

### Metrics

| Metric | Meaning |
|---|---|
| `acceleratorCount` | how many accelerators the PCI bus reports — **always emitted, including zero** |
| `pciAddresses` | slots, in slot order — what an engineer walking to the rack needs |
| `pciVendorIds`, `pciDeviceIds` | as the devices report them |
| `acceleratorVendors` | distinct vendors, resolved where a name is known |
| `acceleratorDrivers` | bound kernel modules — **omitted when nothing is bound** |

Two deliberate asymmetries:

`acceleratorCount=0` is a real measurement: the runner reached sysfs and found
no accelerator, which is true and useful about a CPU-only node. "I could not
look" is the **skip** path instead, so the two are never confused.

`acceleratorDrivers` is omitted rather than empty when the kernel has bound
nothing. A card the kernel sees that no driver claims is exactly the shape of a
node whose plugin never came up — one of the cases this runner exists to make
visible — and an empty value would read as absent information rather than an
absent driver.

VGA adapters (PCI class `0x0300`) are **not** counted. On a node with onboard
graphics, counting one would report hardware the fleet does not have, and
`acceleratorCount` is gateable.

An unknown vendor ID keeps its raw hex. The ID is true; "unknown" is not, and
the hex is what lets someone look the part up.

## Environment

| Variable | Default | Meaning |
|---|---|---|
| `FINGERPRINT_PROBE_SYSFS` | `/host/sys` | where the host's sysfs is mounted |

Host access is through `spec.runner.hostPaths` and nothing else, read-only. The
runner requests **no accelerator** and sets no `NVIDIA_*` environment — it has
to work on a node whose device plugin never came up, and an image needing driver
injection could not.

## Two implementations, on purpose

`pci.go` duplicates `pkg/hostinfo.accelerators`. That is forced by the image
build, not chosen: every runner image builds a throwaway module (`COPY *.go`,
`go mod init`, `GOPROXY=off`) from a context of its own directory, so a runner's
non-test sources cannot import a repo package. See issue #250 for the three
options and why this one was taken — the alternative was making one runner build
differently from the other thirteen.

`probe_conformance_test.go` is the price and the justification: both
implementations run over the same fixture tree and must agree on every field. A
test file *may* import the repo, because the image build deletes `*_test.go`
first, so the guard costs the shipped image nothing. If it fails, the two have
drifted, and the fix is to bring `pci.go` back in line rather than relax the
comparison.

## Status

**Not published.** No `pkg/runnerimages` entry, so a BurnInTest naming this kind
fails at plan time asking for an explicit `spec.runner.image` rather than
pull-failing on every targeted node.
