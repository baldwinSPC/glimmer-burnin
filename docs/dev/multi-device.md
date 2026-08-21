# A Node verdict describes every accelerator on the node

Design note for device iteration. Read this before touching a runner's device
selection; the invariant it establishes is in
[invariants.md](invariants.md#14-a-node-verdict-covers-every-accelerator-on-the-node).
It was reviewed before any code was written; the objections that shaped it are
recorded inline as "(refused: …)" so the next reader does not re-derive them.

## The gap

Every CUDA/HIP runner measures **one device**: `soak_core.cuh` (gpu-burn,
thermal-soak), `clockprobe.cu`, `fp4_smoke.cu`, `gemm_sweep.cu` and every `-rocm`
source call `cudaSetDevice(0)` / `hipSetDevice(0)` or take whatever
`cudaGetDevice` returns; every shipped sample requests `nvidia.com/gpu: "1"`. On
an HGX or MI300X node one arbitrary GPU is certified, the other seven are not,
and **nothing in the stored result says so** — the verdict reads exactly like a
verdict about the node. That is a confident wrong answer of the kind this
project exists to prevent, and it is the largest gap between this operator and
what an eight-GPU customer means by "node acceptance".

`memory-bw` (CUDA) is the precedent for the **verdict semantics**, not the
mechanism: `kCases` in `memory_bw.cc` gates on the WORST cell of what nvbandwidth
measured and publishes the best as Evidence, and declares the peer cases `n/a`
on a one-GPU node. The iteration itself happens inside nvbandwidth. There is no
in-repo precedent for a runner iterating devices, which is why the fold logic
below lives in one shared, unit-tested header rather than in each kernel.

## Mechanism

**The runner iterates over the devices it was allocated, and the pod requests
the node's whole board.** Coverage is guaranteed by the device plugin allocating
what the pod asked for, measured by the runner, and gated by the profile.

*Requesting the board.* There is no "all" quantity in a Kubernetes
`ResourceList`, so a BurnInTest requests the SKU's count — `nvidia.com/gpu: "8"`
— in `spec.resources.limits`. Limits, not requests: Kubernetes refuses a pod
that requests an extended resource without an equal limit, so limits are the
one place the count is always present, and a requests-only test is a visibly
refused pod rather than a silently empty budget. A BurnInTest has NO node
selection — `nodeSelector` lives on the run's and the schedule's
`TargetSelector` — so a fleet with several SKUs writes **one profile and one
run or schedule per SKU**, which multiplies schedules, not tests, and is
materially heavier than one image per vendor. fingerprint-probe's
`acceleratorCount` tells the author what to write. A wrong count is an
unschedulable pod, and `podOverdue` settles that as an `Error` after the pod's
window; this change makes the overdue message carry the pod's
`PodScheduled=False` reason, so "Insufficient nvidia.com/gpu" reaches the
stored result the way the kubelet's message does (#52) — never a silent
partial certification. (Refused, for now: the operator filling the quantity
per node from `NodeFingerprint.Status.GPUs`. It is possible and vendor-neutral,
and it would remove the per-SKU multiplication; but `Resources` rides in the
pinned plan, and a per-node quantity means the plan is no longer SUFFICIENT
to reproduce the pod — everything the author wrote stays pinned, but not
everything the pod ran with. Named as the extension a real fleet is likely to
want, and left for a decision with that fleet in front of it.)

*Allocated is not visible.* Every accelerator runner image bakes
`NVIDIA_VISIBLE_DEVICES=all`, and `runners/pins_test.go` REQUIRES it, because the
GB10 cluster survives only through the legacy runtime, which injects from that
variable and never consults the plugin's allocation. On such a host a pod
handed one card sees the whole board. A runner that iterated "everything
visible" would then drive load on devices allocated to other pods — becoming
the GPU leak this note refuses to lean on — and `deviceCount Equal 8` would
PASS on a pod handed one card, which is the case that gate exists to fail. So:

- The operator injects the pod's OWN limits, uninterpreted and in sorted key
  order, as `BURNIN_RESOURCE_LIMITS=nvidia.com/gpu=8,rdma/hca=1` in
  `podForTest` (resource names contain `/` and `.` and never `,` or `=`, so
  the format needs no escaping). This copies `spec.resources` verbatim, as
  `variantEnv` copies an axis; the reconciler interprets nothing.
- Each runner — which knows its own vendor's resource names because it is
  that vendor's image — sums a declared ALLOW-SET of count-shaped names under
  its vendor's domain (`nvidia.com/gpu` and `nvidia.com/mig-*`; `amd.com/gpu`;
  an Intel image adds its own the day one exists) into a **budget**. (Refused: summing everything under
  the vendor prefix — HAMi/vGPU stacks expose `nvidia.com/gpumem` and
  `nvidia.com/gpucores` beside `nvidia.com/gpu`, and the sum would be 8192.)
  A resource under the vendor's domain that is NOT in the allow-set is exit 3:
  the runner cannot establish its allocation, and a runner may only declare
  what it positively established. `pkg/localrun/translate.go` already holds a
  resource-name→vendor table; a Go test reads the header's allow-set and
  refuses drift between the two.
- **A budget is a count, not an identity.** Under the legacy runtime nothing
  marks WHICH N of the visible devices were allocated, so capping the
  iteration at N would still iterate device 0, which may be another pod's.
  Therefore `budget < visible` is exit 3 — "the runtime showed N devices and
  this pod was allocated M; the allocation cannot be established" — because a
  fold over devices the pod may not own is not a verdict, and in the intended
  configuration (the whole board) budget EQUALS visible, so the mismatch is
  exactly the misconfiguration. The partial mitigation is real but is not the
  guarantee: the run cordons the node and runner pods are the only pods that
  tolerate that taint, so nothing new lands mid-run; only a pre-existing pod
  is exposed.
- The runner reports both figures: `deviceCount` (measured) and
  `devicesVisible` (what the runtime showed). Absent the variable entirely —
  bare metal, where `--gpus all` genuinely IS the allocation — the runner
  iterates every visible device and says so; that fallback is safe there
  specifically, and it is why the CSV is only ever ABSENT or COMPLETE, never
  partial. Set-but-EMPTY is not a state the operator produces, so the header
  reads it as Malformed (exit 3), not Absent: it is somebody overriding the
  injected value to nothing, and the escape hatch must be visible. The
  variable is on the CLI's reserved list; in-cluster, `spec.runner.env` is
  appended after the operator's own variables and would override it — a
  pre-existing gap for every reserved name (`BURNIN_ROLE` included), tracked
  separately (#404), not widened here.

*Two iteration modes, one duration budget.* `durationSeconds` is the TEST's
budget and does not grow with the board. Four reasons, none of them "the
operator cannot know the count" — it can, from the fingerprint, and must not
use it: (1) the fingerprint says what the node HAS, not what the pod was
HANDED, the very distinction `deviceCount` draws; (2) `vendorFor` returns
nothing on a cluster without the fingerprint controller, so a scaled deadline
would silently revert to 1× depending on an unrelated controller having caught
up; (3) `podForTest` derives the deadline from the SEGMENT window of a
segmented soak, and scaling it would run every segment N× long while
`segmentsRequired` still divided the unscaled figure; (4) `docs/soaks.md`'s
capacity arithmetic and `maxConcurrentNodes` read `durationSeconds` as "how
long this test holds the node". So a runner fits its whole iteration inside
`BURNIN_DURATION_SECONDS`:

- **Sequential** — one device at a time, each given `duration / deviceCount`
  (warm-up per device as today). What today's numbers MEAN for a measurement
  kind — a per-die clock, a single GEMM, a correctness burst — where isolation
  is the point and the window is short. The per-device window is echoed as
  `deviceWindowS`, so a 15-second clock window on an 8-GPU node is
  distinguishable from a 120-second one on a Spark.
- **Concurrent** — every device at once, each for the full duration. The only
  honest mode for a kind whose measurement IS the whole board: gpu-burn,
  thermal-soak and (later) power-swing exist to heat the chassis and hold it
  there; a part that holds clock alone but throttles beside seven busy
  neighbours will throttle in production; and a soak sliced into eight
  consecutive `duration/8` windows is not a soak. Concurrency is what keeps a
  soak's duration meaning what the profile says on a multi-GPU node.

The default is **declared per kind by the runner**: soak-class kinds default
to concurrent, measurement kinds to sequential, and `BURNIN_DEVICE_CONCURRENCY=
all|sequential` — a plain environment variable, set through `spec.runner.env`
or a variant's `env` overlay — overrides either way. (Refused: a variant AXIS.
An axis is part of the execution's identity and a profile listing two values
would run the test twice on the same node; an env var overlays.) The resolved
mode is echoed as the label `deviceConcurrency`. A burst kind (compute-smoke)
has no window to divide and iterates sequentially. This is the one place this
note departs from "sequential by default"; the deadline arithmetic is why.
`nccl` Node scope (one communicator over all visible devices) is a different
measurement with its own note (P0.2) and is out of scope here.

## Verdict semantics, keeping the flat metric grammar

There is no device dimension in a metric name and there will not be one — a
suffix-bearing name must never hold a non-float, and a dimension in
`ValidateMetricName` is a contract-class change every consumer would pay for.

1. **The gated metric keeps its existing name and is the WORST device.** The
   direction is the metric's, read off the registry: `Aggregation: Min` for a
   floor (`sustainedClockPct`, `achievedTflops`, `hostToDeviceBandwidthGBs`),
   `Max` for a ceiling (`gpuTempC`, `powerDrawW`), `Sum` for a windowed counter
   (`miscompares`, `xidEvents`). A `Last` metric is the class where
   `Aggregation` says nothing about direction, and there the shared header
   carries an explicit ALLOWLIST: lifetime counters (`eccErrors`,
   `remappedRows`) sum across devices; wall-clock metrics (`elapsedS`,
   `warmupS`, `durationRequestedS`, `deviceWindowS`) describe the pod or the
   window and are NOT folded — emitted once; identity labels (`gpuName`,
   `computeCap`, `pciBusId`, `driverVersion`) keep today's meaning, the FIRST
   device's, so their wire meaning on a single-device fleet does not move.
   (Refused: reusing `Combination` for devices. It is defined across ranks and
   answers `elapsedS` as `Max`, which is right in concurrent mode and wrong in
   sequential; and classifying the ~10 unclassified floors this touches would
   silently change what a GROUP run certifies, in a PR about devices. This
   note touches no `Combination`.)
2. **Attribution is explicit, not overloaded.** `worstDeviceIndex` (Evidence,
   `Last`) and `worstDevicePciBusId` (label, Evidence) name the device behind
   the gated figure of THIS window. Under `Last` across a segmented soak, and
   equally across an error-retry that re-runs a window, they name the LAST
   window's worst device, not the device behind a multi-window aggregate — the
   descriptions say so, and **no ArgMin aggregation class is added to fix
   that**; refuse the request if it comes. `per-device.json` is the only sound
   attribution across windows. There is no `bestDevice…` name: the spread
   carries the same information and the artifact carries the table. (Refused:
   a `<metric>Max<Unit>` infix. It collides with `maxSmClockPct` and
   `peakFmaThroughputTflops`, which mean best WINDOW, and a device extremum
   made to look like a window extremum is exactly the "two differently-obtained
   figures under similar names" the registry forbids.)
3. **A homogeneity spread per measurand**, `(max − min) / max × 100` across
   devices for the same window, named by dropping the base metric's UNIT
   token and keeping its identity: `sustainedClockSpreadPct` (of
   `sustainedClockPct`), `gemmThroughputSpreadPct` (of `achievedTflops`),
   `fmaThroughputSpreadPct` (of `sustainedFmaThroughputTflops`),
   `hostToDeviceBandwidthSpreadPct` (of `hostToDeviceBandwidthGBs`). One name
   per measurand: `sustainedFmaThroughputTflops`'s own description says why it
   is not `sustainedThroughputTflops`, and a shared `throughputSpreadPct`
   would re-commit exactly that under a new suffix. `Pct`, `Aggregation: Max`
   (the worst window's spread describes the board), `Acceptance`. This is
   deliberately the INVERSE of `throughputConsistencyPct` (a consistency is a
   floor, `Min`; a spread is a ceiling, `Max`) and the project now carries
   both directions of "how uniform": consistency within one device across
   windows, spread across devices within one window. (Refused: the mechanical
   `<metric>SpreadPct` — it yields `sustainedClockPctSpreadPct`, a name nobody
   will write and somebody will rename, and a renamed metric is a contract
   break.) The spread names are exported as `contract.SpreadMetrics`, and a
   registry test asserts every member is registered, `Acceptance`, `Max` and
   `Pct`, so the set and the registry cannot drift.
4. **`deviceCount`** — dimensionless counter, `Last`, **`Acceptance`** — is
   how many devices the runner MEASURED. A fleet gates `deviceCount Equal 8`,
   so a pod handed one of eight FAILS instead of certifying one card. Under
   MIG it counts INSTANCES, not parts (56 on an 8-GPU node sliced `1g.5gb`),
   and a MIG fleet gates the instance count. It is distinct from
   `acceleratorCount` (fingerprint-probe: what the node's PCI bus HAS) and from
   `devicesVisible` (Evidence: what the runtime showed the pod). **Only a
   runner that folded devices claims `deviceCount`.** A runner that merely
   SAW them — host-health and dcgm-diag are node-wide, read-only probes that
   load nothing; memory-bw's nvbandwidth iterates every visible device but
   the wrapper does not yet read its budget; memory-bw-rocm is a HIP loop on
   device 0 — reports `devices_visible` for the count it saw and does NOT
   claim the gate, so `deviceCount Equal 8` on such a kind fails closed
   (metric absent) rather than passing on a pod handed one card. That is what
   the unregistered `gpuCount` those four emit today as `gpu_count` becomes:
   a wire break, made deliberately, `gpu_count` NOT aliased (two keys must
   never alias to one name), landing in the same change as the registration.
   Beside it,
   `deviceWindowS` (S, `Last`, Evidence) is the per-device window the runner
   actually used — under concurrent iteration it EQUALS `durationRequestedS`,
   so a concurrent soak's value is not read as evidence of slicing — and
   `deviceConcurrency` (label, Evidence) is the resolved mode.
5. **A single-device node reports every spread as `n/a`** — a positive claim,
   "nothing to spread across" — and `worstDeviceIndex=0`. **The default
   applicability is `Required` and fails closed on `n/a`**, so a fleet profile
   that gates a spread and forgets `RequiredIfMeasurable` fails every healthy
   single-GPU node, and a Fail is never retried. `verdict.ValidateThresholdsForKind`
   therefore reports a `Required` gate on a `contract.SpreadMetrics` name as
   `Unsound` — the authoring-time surface, not a refusal.
6. **Heterogeneous boards fold only what is comparable.** The runner has
   `gpuName`/`computeCap` per device; when every device's identity was READ
   and they differ, `deviceHomogeneous=false` (label, Evidence — a positive
   establishment: an unreadable name on one device is an omission and
   declares nothing), the Pct-of-rated figures (`sustainedClockPct`) and the
   counters fold as usual, and every spread over an ABSOLUTE figure
   (`achievedTflops`, `powerDrawW`, `gpuTempC`, bandwidths) is declared `n/a`
   — a mixed board's spread reads ~100% on healthy hardware and is not a
   measurement of anything.
   **MIG instances** enumerate as devices, share one power, thermal and clock
   domain, and are folded as devices for correctness counters only; every
   spread is `n/a` under MIG (thermal-soak already skips under MIG for the
   same reason). `migMode` is already a label.
7. **Per-device tables go in a fenced artifact, `per-device.json`**
   (`application/json`), one per pod — so one per segment of a soak, and the
   result's ref list grows with segments; a soak's evidence is not one document.
   One object per device: index, bus id, name, and every value the fold
   consumed. Well under `MaxArtifactBytes` (256 KiB) at 8 devices; an oversize
   one is a dropped ref and never a verdict.
8. **Precedence across devices is `Fail > Error > Skip > Pass`, which inverts
   Pair deliberately.** A measured miscompare on device 3 is a fact about a
   PART, and device 6 failing to enumerate does not erase it; Pair puts Error
   first because a machinery failure at either END means the LINK was never
   measured. memory-bw already orders devices this way ("a verification
   mismatch is a fact about the hardware and outranks everything"). A device
   that errors and no device that fails is an Error for the test — a fold over
   seven of eight is a verdict about a board nobody finished measuring. In
   concurrent mode a sticky context error (`cudaErrorAssert`) is diagnosed
   before any device is judged, as compute-smoke already does. **Two
   precedence orders now exist in the project and both are invariants**: the
   discriminator is what the reporters ARE — devices are parts, Pair endpoints
   are halves of one link — and `CLAUDE.md` / `invariants.md` state both in
   the same change, or the next session "fixes" one to match the other.

## Composition

- **With segmented soaks.** The runner folds DEVICES within a window; the
  controller's `foldMetrics` then folds WINDOWS by `Aggregation`. Min-of-min,
  max-of-max and sum-of-sum compose; a lifetime counter summed across devices
  and taken `Last` across windows is the node's total at the last window. A
  spread is `Max` across windows under an `LTE` ceiling, which is monotone in
  the gated direction, so **`AbortEarly` qualifies for it** through the
  existing `monotoneBreach` rule (`Max` under `LTE`), with no new case; the
  step-2 conversion adds a soak test that a spread ceiling aborts a segmented
  soak the way any `Max`/`LTE` gate does.
- **With Group.** The per-device fold happens INSIDE the rank, before the rank
  reports; `Combination` then merges ranks and nothing in `group.go` learns
  devices exist. `deviceCount` is unclassified there, and an unclassified key
  keeps rank 0's value WHILE THE RANKS AGREE and is dropped on disagreement —
  so on 8 ranks × 8 GPUs, `deviceCount Equal 8` merges and passes, and one
  short rank fails it closed. That is right, and it is stated rather than
  deferred.
- **With repeats and retries.** Each attempt is a complete fold; see item 2
  for what `worstDeviceIndex` means across a retry.

## Where the direction lives, and the guards

The direction of every fold is declared ONCE, beside the name, in
`pkg/contract`. `runners/gpu-burn/device_fold.h` — its home; every converting runner takes a copy — a CUDA-free header, byte-
identical across every runner that carries it (`sharedsource_test.go` picks up
any file duplicated across runner directories automatically and refuses drift),
unit-tested by `device_fold_test.cc` under `make test` — holds the fold and a
table mapping each raw key to `Min`/`Max`/`Sum`/`Once` plus the `Last`
allowlist. `runners/devicefold_test.go` reads that table and asserts every
`Min`/`Max`/`Sum` entry agrees with `contract.AggregationFor(canonical(key))`,
so a runner cannot decide, privately, that a floor is a ceiling.
`runners/devicefold_test.go` carries a TOTAL table of accelerator-touching runner
directories: each either uses the shared iteration helper, or is EXEMPT with
its reason (`memory-bw`: nvbandwidth iterates; `dcgm-diag`: dcgmi enumerates;
`host-health`: NVML per GPU already; `fingerprint-probe`: PCI; the fabric
runners: one endpoint per pod IS the measurement, and nccl Node scope is
P0.2), or is PENDING with the issue that converts it. A new accelerator runner
that is in none of the three fails the build.

## Delivery order

Tracked in #398; each step below is its own issue.

1. This note, the contract names, `BURNIN_RESOURCE_LIMITS` and the overdue
   message, the shared header and its tests, the invariant text and the
   total table (every CUDA/HIP runner PENDING), and `gpu_count` →
   `devices_visible` in memory-bw, memory-bw-rocm, host-health and dcgm-diag
   (the wire break lands with the registration; none of the four claims
   `deviceCount`, because none has folded devices). Nothing changes what any
   published image MEASURES.
2. The soak family (`soak_core.cuh`, `soak_core_rocm.h`): concurrent default (#399).
3. clockprobe, compute-smoke, gemm-sweep and their `-rocm` siblings:
   sequential default (#400).
4. Samples request the board (#401). The CLI already passes `--gpus all` /
   `--device nvidia.com/gpu=all` (`pkg/localrun/runtime.go`), so bare metal
   measured device 0 of N silently until step 2 and measures every device
   after it; `cmd/burnin`'s usage text and the README's known-limitations
   section say so.

Steps 2 and 3 change what a KIND measures under a new tag. Published tags are
immutable, so a fleet re-pins deliberately and the release notes carry it —
a node that passed under the old tag was certified on one device. Steps 2–4
each need a real multi-GPU node to verify and the project has none: the fold
is unit-tested CUDA-free, the single-device path is verified on GB10, and one
HGX/MI300X verification issue names what to capture (#402).

## What this does not decide

Which devices are visible is the runtime's decision and is reported, not
enforced: a `CUDA_VISIBLE_DEVICES` set by an operator narrows the board and
`devicesVisible` says so. Whether the fleet requires the whole board is the
profile's threshold on `deviceCount`.
