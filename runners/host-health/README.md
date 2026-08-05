# `host-health` runner — host and driver fault counters

Runner image for the `host-health` [`TestKind`](../../api/v1alpha1/burnintest_types.go):
a **passive** read of the host- and driver-side fault counters that move while
every performance test still passes.

It applies no load, measures no throughput, finishes in well under a minute, and
**never skips**. That is what makes it cheap enough to attach to every profile.

Its value is timing. A row being remapped, a PCIe link quietly retrying, an
uncorrected ECC error recorded last week, a single Xid during a job — each of
those is a part on its way out, and each of them is invisible to `compute-smoke`,
`gpu-burn`, `thermal-soak` and `nccl`, all of which will happily pass on the
same node. This runner turns that history into an acceptance signal instead of
something discovered in production.

## What it reads

| Source | Counters | Needs |
|--------|----------|-------|
| NVML (via injected `nvidia-smi`) | aggregate ECC totals, retired pages, row remapper, PCIe replay counter, temperature, power | NVIDIA driver + Container Toolkit (`utility` capability) |
| Kernel log (`/dev/kmsg`, else a text kernel log) | Xid events, generic hardware-error lines | a readable kernel log — see [What the pod needs](#what-the-pod-needs) |
| sysfs PCIe AER (`/sys/bus/pci/devices/*/aer_dev_*`) | correctable / non-fatal / fatal AER counts, replay timer + rollover | nothing (sysfs is already mounted) |
| sysfs NIC link state (`/sys/class/net`, `/sys/class/infiniband`) | link-down transitions, operstate, IB port state | `hostNetwork: true` |

Every probe is independent and optional. A node with no NVIDIA driver reports
`nvml_status=absent` and is still judged on the other three — that is the
vendor-neutral path, not an error.

## Output contract

| Result | Last stdout line | Exit |
|--------|------------------|------|
| Pass — every counter that could be read is zero | `HOST_HEALTH_OK` | 0 |
| Fail — a fault counter is non-zero | `HOST_HEALTH_FAULT: <key>=<value> …` | 1 |
| *(never emitted)* | — | 2 |
| Error — nothing could be read; the node is **unjudged** | `HOST_HEALTH_ERROR: …` | 3 |

**Exit 2 is unreachable, deliberately.** Every node has host health, so "not
applicable" is never the right answer. A node whose counters cannot be read has
not passed and has not failed: it is unjudged, which is exit 3 (Error) — a state
the operator keeps distinct from Fail so an infrastructure problem never gets
recorded as a hardware verdict.

Metrics are printed **before** the verdict line, so a failing run still yields
its evidence.

## Metrics

Runner keys are normalised to canonical names by
[`pkg/runner`](../../pkg/runner/parse.go); thresholds are written against the
canonical name.

### Gated — the runner's own exit code depends on these

| Runner key | Canonical | Window or lifetime | Source |
|------------|-----------|--------------------|--------|
| `xid_count` | `xidEvents` | window | kernel log |
| `ecc_errors` | `eccErrors` | lifetime (aggregate, **uncorrected only**) | NVML |
| `rows_remapped` | `remappedRows` | lifetime (correctable + uncorrectable) | NVML |
| `pcie_replay_count` | `pcieReplayErrors` | window | NVML, else sysfs AER |
| `nic_link_down` | `nicLinkDownEvents` | window | sysfs (ethernet + IB summed) |

These are exactly the five counters the `host-health` kind is defined by, each
gated at `Equal 0`, so the runner's exit code agrees with the profile it is
written for.

### Evidence — reported, not gated

`xidPreexisting`, `kernelHwErrors`, `kernelHwErrorsPreexisting`, `xidLogDropped`,
`eccCorrectedAggregate`, `eccUncorrectedVolatile`, `eccUncorrectedWindow`,
`remappedRowsCorrectable`, `remappedRowsUncorrectable`, `remappedRowsPending`,
`remappedRowsFailure`, `eccMode`,
`retiredPagesSbe`, `retiredPagesDbe`, `retiredPagesTotal`,
`retiredPagesPending`, `pcieReplayTotal`, `pcieAerCorrectable`, `pcieAerNonfatal`,
`pcieAerFatal` (and their `…Total` lifetime forms), `nicCount`, `nicUp`,
`nicLinkDownTotal`, `ethLinkDownTotal`, `ibLinkDownTotal`, `ibPorts`,
`ibPortsActive`, `gpuTempC`, `powerDrawW`, `gpuName`, `gpuCount`, `driverVersion`,
`observationWindowS`, `elapsedS`, `nodeReady`, and the probe statuses
`nvmlStatus`, `xidSource`, `aerStatus`, `nicStatus`, `pcieReplaySource`.

Evidence metrics are real measurements and are all thresholdable — a stricter
profile can and probably should gate on some of them (see below). They are not
gated *by the image* because where that line falls is policy, and policy belongs
in the `BurnInTest`, not baked into a tag a whole fleet pins.

`nodeReady` is a human-readable `true`/`false` summary of the verdict. It is not
numeric, so do not write a threshold against it.

### Two rules that decide what gets emitted

**1. Never a zero that was not measured.** A counter this runner could not read
is *omitted*, never reported as `0`. `pkg/verdict` fails a threshold whose metric
is missing, so an omission fails closed. A fabricated zero would silently satisfy
acceptance, which is the exact false-negative this project forbids. Concretely:

- a GPU with ECC **disabled** reports `N/A`, so `eccErrors` is omitted and a
  threshold on it fails — the part has not proven it is error-free, it has
  proven nothing;
- a node-level total is only emitted if **every** GPU answered, because the GPU
  that did not answer is the one most likely to be broken;
- a pod without `hostNetwork` sees only its own veth, so no physical NIC is
  visible and `nicLinkDownEvents` is omitted rather than reported as a confident
  zero measured from an interface that could never fail;
- an unresponsive driver (`nvml_status=timeout` — itself a real symptom) leaves
  every NVML counter omitted rather than guessed.

**1b. …but "this part has no such counter" is a third thing, and it is stated.**
Where the driver was *asked* and answered `N/A` **for the counter and for
`ecc.mode.current` itself**, the part has no ECC subsystem at all, and the runner
emits the reserved value `n/a` (`ecc_errors=n/a`, `rows_remapped=n/a`) instead of
staying silent. This is NVIDIA GB10 / DGX Spark: its unified LPDDR5X has on-die
ECC that NVML cannot see, so every ECC and row-remap field reads `[N/A]` on a
perfectly healthy part, and a plain `eccErrors Equal 0` gate fails the whole
fleet.

A threshold marked `applicability: RequiredIfMeasurable` is then reported as
**not evaluated** — visibly, in the result message and the delivered envelope —
rather than failed. Nothing else relaxes:

- `eccMode` is emitted alongside as the evidence for the claim (`enabled`,
  `disabled`, `unsupported`, or `mixed`);
- ECC **switched off** on a part that has ECC still reports a mode, so its
  counters are *omitted* and the gate still fails. Disabling ECC is not a way to
  skip an ECC gate;
- a missing column, a mix of numbers and `N/A` across the node's GPUs, an absent
  driver or a timeout are all "we could not look", and are still *omitted*;
- omission is still a failure under every applicability. Only an explicit `n/a`
  from the runner can relax anything.

**2. Damage is lifetime, events are windowed.** `eccErrors` and `remappedRows`
are cumulative properties of the part: a node that took an uncorrected ECC error
last week is not a node that passes acceptance today, so the lifetime figure is
what gates. `xidEvents`, `pcieReplayErrors` and `nicLinkDownEvents` are events,
whose since-boot totals are dominated by history that is not this node's fault (a
switch reboot last month is not a bad NIC) — so the window delta gates and the
lifetime total is emitted beside it as `…Total` / `xidPreexisting` for a profile
that wants to be stricter.

Being explicit about the consequence: on a short passive window, `xidEvents` is
usually 0 even on a node with a damning boot history. **If you want the boot
history held against the node, threshold `xidPreexisting` as well** — the
recommended profile below does.

## Recommended `BurnInTest`

There is no default runner image for this kind yet (see
[Publishing](#publishing)), so `spec.runner.image` is required.

```yaml
apiVersion: burnin.glimmer.ai/v1alpha1
kind: BurnInTest
metadata:
  name: host-health
spec:
  kind: host-health
  scope: Node
  # The observation window. Passive, so a short window is the point.
  durationSeconds: 30
  # /sys/class/net is per-netns: without this the NIC probe sees only the pod's
  # veth and nicLinkDownEvents is omitted (and any threshold on it fails).
  hostNetwork: true
  runner:
    image: ghcr.io/baldwinspc/glimmer-burnin-host-health:v0.2.0
    # Both of these are for the /dev/kmsg Xid scan and nothing else; see
    # "What the pod needs". Drop them and xidEvents is omitted, not zeroed.
    privileged: true
    hostPaths:
      - path: /dev/kmsg
        mountPath: /dev/kmsg
        type: CharDevice # readOnly defaults to true
  thresholds:
    - { metric: xidEvents,         comparison: Equal, value: "0" }
    # RequiredIfMeasurable: passes on a GB10, which has no ECC for NVML to read
    # and says so, while still failing an ECC-capable GPU whose counters have
    # moved OR whose ECC has been switched off. Drop the applicability (it
    # defaults to Required) on a fleet where every part has real ECC.
    - metric: eccErrors
      comparison: Equal
      value: "0"
      applicability: RequiredIfMeasurable
    - metric: remappedRows
      comparison: Equal
      value: "0"
      applicability: RequiredIfMeasurable
    - { metric: pcieReplayErrors,  comparison: Equal, value: "0" }
    - { metric: nicLinkDownEvents, comparison: Equal, value: "0" }
    # Recommended additions. Each one is a real fault that the five above do
    # not cover, and each fails closed if its probe could not run.
    - { metric: xidPreexisting,    comparison: Equal, value: "0" }  # Xids earlier this boot
    - { metric: pcieAerFatal,      comparison: Equal, value: "0" }
    - { metric: pcieAerNonfatal,   comparison: Equal, value: "0" }
    - { metric: retiredPagesTotal, comparison: Equal, value: "0" }  # pre-Ampere parts
```

Run it **last** in a profile, after the load tests: the window then covers the
period in which `gpu-burn` or `thermal-soak` has just stressed the part, and a
Xid or ECC error provoked by that load is still in the log.

## What the pod needs

| Probe | Requirement | If missing |
|-------|-------------|------------|
| PCIe AER | nothing — `/sys` is already mounted read-only in any container | `aer_status=absent` |
| NIC link state | `hostNetwork: true` | `nic_status=absent`, `nicLinkDownEvents` omitted |
| NVML | NVIDIA Container Toolkit; the image requests `NVIDIA_DRIVER_CAPABILITIES=utility` | `nvml_status=absent`, all NVML counters omitted |
| Xid scan | a readable `/dev/kmsg`, mounted with `spec.runner.hostPaths` — see below | `xid_source=none`, `xidEvents` omitted (never a false zero) |

A source that is readable at start and then fails **mid-run** is treated the same
way. `xidEvents` is a difference between two samples, so it is emitted only when
**both** were taken: if the opening scan failed the source was never positioned
and the "window" would be the node's whole log, and if the closing scan failed
the window was never read at all. Either way the counter is **omitted** and
`xid_log_dropped=1` records that something went wrong. Omission fails closed
upstream — a profile gating `xidEvents` fails the node rather than certifying it.

It is deliberately not the `n/a` sentinel: `n/a` declares that the *hardware*
cannot produce the value, and this is a statement about the *probe*.

**The Xid scan is the awkward one.** `/dev/kmsg` is not in a container's default
`/dev`, and reading it needs `CAP_SYSLOG` (or root) on any host with
`kernel.dmesg_restrict=1`, which is the default on most distributions. The image
runs as `USER 65532:65532` — least privilege by default, and it degrades cleanly
rather than demanding root of every node.

**Mount it.** `spec.runner.hostPaths` is the supported way, and it is what
`config/samples/node-acceptance.yaml` does:

```yaml
  runner:
    # The mount supplies the device node; privileged supplies CAP_SYSLOG, which
    # reading /dev/kmsg needs wherever kernel.dmesg_restrict=1 — the default on
    # most distributions. On a host with dmesg_restrict=0 and a world-readable
    # /dev/kmsg the mount alone is enough, even as uid 65532.
    privileged: true
    hostPaths:
      - path: /dev/kmsg
        mountPath: /dev/kmsg
        # readOnly defaults to true, which is right here: this runner only ever
        # reads the ring buffer.
        type: CharDevice
```

Do not rely on `privileged: true` alone to supply the device node. It is
documented as giving the container the host's `/dev`, but that was measured to be
unreliable on this project's own nodes — a privileged pod on these hosts is
missing device nodes the host has (see the `uverbs*` case in
[`../ib-write-bw/README.md`](../ib-write-bw/README.md#pod-requirements)). Name
the path and the kubelet mounts the path.

A text kernel log works too: mount it with `hostPaths` and point the runner at it
with `BURNIN_KERN_LOG_PATHS` — and on a hardened node that is the more robust
route, because it needs no capability at all.

One residual gap, unchanged by this field: if a node needs the reader to be
**root** rather than merely to hold `CAP_SYSLOG`, the reconciler still cannot
express it. It sets `privileged` but no `runAsUser`, and the image's
`USER 65532:65532` wins. Expect `xid_source=none` there, and reach for the text
kernel log instead.

**Without the mount the runner degrades honestly, and that is the whole point.**
It reports `xid_source=none` and **omits** `xidEvents` rather than printing a
`0` it never measured. `pkg/verdict` fails a threshold whose metric is missing,
so a profile that gates `xidEvents Equal 0` on an unmounted node **fails the
node** — it does not pass it. The cost of forgetting the mount is a node
condemned for a measurement nobody took; the cost of the alternative would be a
fleet certified clean by a fabricated zero, and only one of those two is
recoverable.

## Duration

`BURNIN_DURATION_SECONDS` (set by the reconciler) is honoured as the
**observation window** — the interval over which the event counters are
differenced. It is treated as an upper bound the caller asked for, never a floor,
and is clamped to **30 s**: the reconciler defaults an unset duration to 600 s,
and a passive counter read has no business occupying ten minutes of a node's
burn-in. Unset or zero gives a 15 s window.

The runner also enforces its own 55 s deadline and an 8 s timeout per external
probe, so a wedged `nvidia-smi` (a GPU that has fallen off the bus does exactly
that) cannot let the pod be killed at its `activeDeadlineSeconds` — which would
destroy the evidence and be reported as an infrastructure Error.

`BURNIN_ATTEMPT` is ignored; nothing in this runner is seeded or stateful.

## Environment

| Variable | Default | Purpose |
|----------|---------|---------|
| `BURNIN_DURATION_SECONDS` | 15 (max 30) | observation window |
| `BURNIN_KMSG_PATH` | `/dev/kmsg` | kernel log device, or a captured log to scan |
| `BURNIN_KERN_LOG_PATHS` | `/var/log/kern.log:/var/log/messages:/var/log/syslog` | text kernel-log fallback, first readable wins |
| `BURNIN_SYSFS_ROOT` | `/sys` | sysfs root (relocatable for testing) |
| `BURNIN_NVIDIA_SMI` | `nvidia-smi` | path to `nvidia-smi` |

## Example output

A healthy GB10 / DGX Spark. Note `ecc_errors=n/a` and `rows_remapped=n/a`: this
part has no ECC subsystem for NVML to read (`ecc_mode=unsupported`), so the
runner declares those counters unmeasurable rather than inventing a zero — and
the ECC family of evidence metrics is absent for the same reason. On an
ECC-capable GPU those lines carry real numbers, and `ecc_corrected_aggregate`,
`remapped_rows_correctable`, `retired_pages_total` and friends appear alongside.

```
host_health_stage=start
host_health_version=1
observation_window_s=30
host_health_stage=kernlog-baseline
host_health_stage=probe-first
host_health_stage=observation-window
host_health_stage=probe-second
host_health_stage=kernlog-collect
host_health_stage=emit
host_health_stage=done
xid_source=kmsg
xid_count=0
xid_preexisting=0
kernel_hw_errors=0
kernel_hw_errors_preexisting=0
nvml_status=ok
gpu_name=NVIDIA GB10
driver_version=580.82.09
gpu_count=1
ecc_mode=unsupported
ecc_errors=n/a
rows_remapped=n/a
pcie_replay_total=0
gpu_temp_c=41
power_draw_w=38.2
aer_status=ok
pcie_aer_devices=1
pcie_aer_correctable_total=0
pcie_aer_fatal_total=0
pcie_aer_correctable=0
pcie_aer_fatal=0
nic_status=ok
nic_count=1
nic_up=1
ib_ports=1
ib_ports_active=1
eth_link_down_total=0
ib_link_down_total=0
nic_link_down_total=0
nic_link_down=0
pcie_replay_count=0
pcie_replay_source=nvml
elapsed_s=30.2
node_ready=true
HOST_HEALTH_OK
```

`host_health_stage` appears once per stage and is the only line written as the
run proceeds; everything else is buffered and printed in one block at the end.
The parser is last-occurrence-wins, so the value the operator stores is the
**furthest stage reached** — always `done` on a run that finished, and something
else only when the runner did not finish. See *When the runner dies* below.

## When the runner dies

A crashed runner used to leave nothing at all. Every metric was buffered until
the end, so a process brought down mid-run printed no metrics, no verdict line,
and nothing naming the probe it was inside — which is why issue #112 (a
`host-health` test that settled `Skipped` on a real node) could never be
diagnosed. Two mechanisms cover the two ways it can die:

| How it died | What you get |
|---|---|
| **panic** | exit **3**, the evidence gathered so far, `host_health_panic=<one-line summary>`, the stack, and a final `HOST_HEALTH_ERROR:` line naming the stage |
| **runtime fatal error** (out of memory, concurrent map write, stack exhaustion) — unrecoverable, exits **2** | every `host_health_stage=` line up to the stage it died in |
| **SIGKILL** (kernel OOM killer, eviction) — exit **137** | the same breadcrumbs |

A panic is recovered so the runner can report it in its own contract: exit 3 is
`Error`, the node is **unjudged**, and the operator retries it under
`retryOnErrorLimit`. The other two cannot be intercepted by any Go program, so
the only thing that survives them is output already on the wire — which is the
whole reason the stage marker is streamed rather than buffered.

Note the second row is the shape that caused #112. A runtime fatal error exits
2, the skip code, and a runner that skips normally prints no metrics either — so
the crash was perfectly camouflaged as "this test does not apply to this node".
`pkg/runner` now refuses to read an exit 2 as a skip unless the runner also
prints a `_SKIP` declaration, and this runner deliberately prints none.

The other half of #112 was the mechanism. Both kernel-log sources used to read
their whole scan into a slice before counting it, so peak memory tracked the
size of the host's log — on the text path, the entire unread tail of
`/var/log/syslog`, bounded by nothing. Both stream now, and
`TestFileSourceScanRetainsNothing` holds them to it.

## Source layout

The Go source lives in the repository's **root module** (`package main` in this
directory), so `make lint`, `go build ./...`, `go vet ./...` and `go test ./...`
cover it like any other package. `metricnames_test.go` runs the real runner's
real output through `pkg/runner` and `pkg/contract` and asserts that every key
it prints normalises to a legal, non-colliding canonical metric name — the seam
between runner and operator is checked by CI rather than by hope.

The non-test sources import **only the standard library**. The publish workflow
builds with `context: runners/<name>`, which cannot see the repository's
`go.mod`, so the image build synthesises a throwaway module; `GOPROXY=off`
turns any future non-stdlib import into a loud build failure rather than a
silent dependency.

## Build

```sh
docker build -t glimmer-burnin-host-health:dev runners/host-health
```

Build args: `GO_IMAGE` (default `golang:1.24`), `BASE_IMAGE` (default
`gcr.io/distroless/cc-debian13`).

The binary is pure Go with `CGO_ENABLED=0`, cross-compiled from
`$BUILDPLATFORM`, so **both** `linux/amd64` and `linux/arm64` build natively on
whichever runner CI is on — no QEMU on either platform, unlike the compute-smoke
runner. The image is published as a manifest list covering both. This runner
compiles no device code, so it has no GPU-architecture axis at all.

The build **fails** unless the binary is statically linked and neither its `ldd`
output nor its own bytes mention `libcuda*`, `libcudart*` or `libnv*`. Scanning
the binary itself is what would catch a `dlopen()` added later: a runtime
`dlopen` leaves the soname in the binary as a string while leaving `ldd` clean.

Local run against a fake node (no container needed):

```sh
go build -o /tmp/host-health ./runners/host-health
BURNIN_SYSFS_ROOT=/tmp/fake-sys BURNIN_KMSG_PATH=/tmp/fake-kmsg \
BURNIN_NVIDIA_SMI=/tmp/fake-nvidia-smi BURNIN_DURATION_SECONDS=1 /tmp/host-health
```

## Publishing

Not published yet, so there is no entry in `defaultRunnerImages`
(`internal/controller/pods.go`) and `spec.runner.image` is required. Per the
project's rules, an entry is added only once the image has been built, verified
on real hardware, and published by the manual `publish-runner` workflow — a
speculative entry would turn into an `ImagePullBackOff` reported per node as an
infrastructure Error. Publishing also needs `host-health` added to the
workflow's `runner` choice list.

## Licensing

Every component is on the project's permissive allowlist.

| Component | License | How it is consumed |
|-----------|---------|--------------------|
| This runner's source | Apache-2.0 | compiled into the image |
| Go standard library / toolchain | BSD-3-Clause | compiled into the binary; it is the **only** dependency |
| `gcr.io/distroless/cc-debian13` | Apache-2.0 (the distroless project) | unmodified base image |
| `nvidia-smi`, `libnvidia-ml.so.1` | NVIDIA proprietary | **never in the image** — injected at runtime from the host driver by the NVIDIA Container Toolkit, and executed as a subprocess, never linked |

No GPL/AGPL/LGPL/SSPL/MPL component is added by this runner. The binary is
statically linked with CGO disabled, so it links **nothing** from the base image
— the Debian packages the base carries (glibc and friends) are present only so
that the injected, dynamically linked `nvidia-smi` has a loader and a libc to
run against. A deployment that does not want the NVML probe can build with
`BASE_IMAGE=gcr.io/distroless/static-debian13` and carry none of them.

Deliberately **not** used here: `dmidecode`, `edac-utils`, `ipmitool`,
`smartmontools`, `lshw` and `hwinfo` are all GPL and would make the image
unpublishable. Everything above is read straight from sysfs, `/dev/kmsg` and
`nvidia-smi` instead, which is why this runner ships no third-party binary at
all.

## Not yet verified

- **The image has never been built** — there is no container daemon in the
  environment this was written in. The Go binary compiles, vets, and passes its
  tests, and the Dockerfile's assertion logic was exercised directly in a shell,
  but the multi-stage build itself, the base image tags, and the injected
  `nvidia-smi` inside a distroless rootfs are all unverified.
- **`nvidia-smi` field names** are from the documented `--query-gpu` set and are
  not identical across driver generations. The per-field fallback exists exactly
  because of that (one rejected name costs one metric, not the whole probe), but
  which fields a given driver accepts has not been checked on hardware. Verify
  on a real GB10 before publishing.
- **`/dev/kmsg` in a privileged pod** — the read path (raw fd, `O_NONBLOCK`,
  `EAGAIN`/`EPIPE` handling) is unit-tested against a regular file, not against
  a real ring buffer under `dmesg_restrict`.
