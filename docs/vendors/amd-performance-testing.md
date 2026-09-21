# How AMD hardware gets performance-tested, and what this project should adopt

A survey of the methods available for measuring and accepting AMD accelerators,
written to answer three questions this project has to decide:

1. What already exists, what does it measure, and **can we redistribute it**?
2. What does AMD itself say acceptance looks like?
3. **A Halo is not an Instinct.** What differs, and what has to change here to
   serve both?

Evidence in this document is labelled, because the three kinds are not
interchangeable:

- **[MEASURED]** — observed on this project's own gfx1151 hardware. Reproducible
  from the capture committed at `runners/clockprobe-rocm/testdata/gfx1151-amd-halo/`.
- **[AMD]** — AMD's published documentation.
- **[COMMUNITY]** — forums, issue trackers, community guides. Often right, often
  specific to a chassis or a workload, and **not verified here**. Treated as a
  lead, never as a threshold.

---

## 1. The landscape

### 1.1 Vendor acceptance suites

| Tool | Licence | Redistributable | What it is |
|---|---|---|---|
| **AGFHC** (AMD GPU Field Health Check) | proprietary | **No** — authorized access only | AMD's own field health check. PASS/FAIL plus `results.json`. Levelled tests (`gfx_lvl1..4`, `hbm3_lvl3..5`, `pcie_lvl1..3`, `xgmi_lvl1`, `all_lvl5`, `all_perf`), groupable into short/extended recipes |
| **RVS** (ROCm Validation Suite) | **MIT** | **Yes** | The open analog. Modules: `gst` (stress), `iet` (power/EDPp), `babel` (BabelStream bandwidth), `mem`, `pebb`/`pbqt` (PCIe bandwidth), `gpup`, `peqt` |
| **CVS** (Cluster Validation Suite) | open | Yes | Configuration checker and cluster health orchestration |

**[AMD]** The split matters operationally: AMD's GPU Operator ships a Test
Runner whose *public* image "only supports executing RVS test", while the image
containing AGFHC "is NOT publicly accessible and requires special
authorization." So **AGFHC is not an option for this project** — which is what
`docs/vendors/amd.md` already says, and it remains correct. RVS is.

### 1.2 Microbenchmarks

| Tool | Licence | Measures | Maps to |
|---|---|---|---|
| **TransferBench** | MIT | host↔device and peer-to-peer copy bandwidth | `memory-bw` |
| **rocm-bandwidth-test** | NCSA/MIT-ish | copy bandwidth, simpler | `memory-bw` |
| **BabelStream** (`babel`) | BSD-3 | device memory bandwidth (STREAM-style) | `memory-bw` |
| **rccl-tests** | BSD-3 | collective bandwidth/latency | `nccl` |
| **rocHPL** | BSD-3 | sustained FP64 HPL | — (no kind) |
| **rocBLAS / hipBLASLt bench** | MIT | GEMM throughput per precision | `gemm-sweep`, `compute-smoke` |
| **OFED perftest** | GPL-2.0 **or** BSD — consume under **BSD** | RDMA verbs bandwidth/latency | `ib-write-bw` |
| **mixbench** | GPL-2.0 | **unusable here** — copyleft | — |

The licence column is the operative one. Per `CLAUDE.md` a copyleft dependency
makes this project unpublishable, with no exception process. `mixbench` is out;
`perftest` is fine *only* under its BSD option and `NOTICE` must say so, which
it already does.

### 1.3 Telemetry

| Source | Reach on gfx1151 | Note |
|---|---|---|
| **sysfs** (`amdgpu` ABI) | clocks, edge temp, socket power, utilisation, GTT/VRAM | No library, no privilege, no ioctl |
| **amd-smi** | rich `APU_*` vocabulary — GFX temp, **GFX power**, per-core clocks, DRAM traffic | **[MEASURED]** see §3.2 — the blanket "amdsmi is blind" claim is wrong |
| **rocm-smi** | reads the same sysfs | |
| **rocprofiler-sdk** | kernel traces, counters | Profiling, not acceptance |

---

## 2. What AMD says acceptance is

**[AMD]** The *Instinct Customer Acceptance Guide* specifies a two-phase
methodology, and its shape is worth noting because this project arrived at
almost the same decomposition independently.

**Node:** prerequisites (OS, firmware/BIOS, GRUB) → health checks (PCIe/GPU
visibility, host memory, interconnect) → validation: AGFHC levels, CVS config
checker, TransferBench, single-node RCCL, rocHPL, then LLM workloads.

**Cluster:** NIC drivers → network config → topology mapping → RDMA perftest
(`ib_write_bw`, `ib_send_bw`, latency variants) → multi-node RCCL, storage
(IOR/FIO/MDTEST), rocBLAS FP32/BF16/INT8, BabelStream.

Two things stand out.

**AMD does not publish node-level numeric thresholds.** The guide "emphasizes
*baselining* rather than pass/fail criteria." That is the same conclusion
`docs/vendors/amd.md` already reaches from the other direction — *run
thresholdless, look at the fleet, then gate* — and it is worth keeping, because
it means nobody is withholding a number we could have used. There isn't one.

**The prerequisites phase is a test.** AMD puts firmware, BIOS and GRUB
*before* the benchmarks, and CVS is a *configuration checker*. §4 argues this is
the single biggest gap in our AMD coverage.

### Mapping onto this project's kinds

| AMD acceptance step | Kind here | State |
|---|---|---|
| Health checks / GPU visibility | `host-health` | shipped; AMD RAS path exists, **[MEASURED]** no `ras/` on an APU (§3.4) |
| AGFHC `gfx_lvl*` / RVS `gst` | `gpu-burn`, `thermal-soak` | `-rocm` in tree, unpublished |
| RVS `iet` (EDPp / power) | `power-swing` | no AMD image |
| AGFHC `hbm_lvl*` / RVS `mem` | `memory-retention`, `memory-stress` | `memory-stress` vendor-free |
| TransferBench / `babel` | `memory-bw` | `-rocm` in tree, unpublished |
| Single/multi-node RCCL | `nccl` | `-rocm` in tree; multi-node is #345 |
| RDMA perftest | `ib-write-bw` | shipped, vendor-free |
| rocHPL, LLM workloads | — | deliberately absent: a benchmark, not an acceptance |
| **CVS configuration checker** | **— nothing** | **the gap, §4** |
| `pcie_lvl*` / `xgmi_lvl*` | `gpudirect-rdma` (NVIDIA-only) | no AMD analog |

The coverage is better than `docs/vendors/amd.md` claims — that page is stale
and is corrected in this change.

---

## 3. Measured on gfx1151

All from `runners/clockprobe-rocm/testdata/gfx1151-amd-halo/`. Host `amd-halo`:
Ryzen AI MAX+ 395, Radeon 8060S, 40 CU, kernel 6.18.44, ROCm 6.4, 128 GB unified.

### 3.1 The part sustains 100% of its rated clock — until the CPU wants power

**[MEASURED]** Under a register-resident FP32 FMA load, held 140 s:
2900/2900 MHz, flat, 97% busy, 37→44 °C.

Then, the same load with all 16 cores busy, and the control:

| Phase | sclk | % rated | busy | edge | GFX power | socket power |
|---|---|---|---|---|---|---|
| GPU only | 2900 MHz | **100%** | 97% | 45.0 °C | 11.5 W | 35 W |
| GPU + all-core CPU | 1177 MHz | **41%** | 98% | 43.8 °C | 3.3 W | 59 W |
| CPU removed | 2898 MHz | **100%** | 97% | 47.2 °C | 11.5 W | 35 W |

The clock recovers completely. This is socket power arbitration on a shared
package — correct, by-design behaviour — and it is **indistinguishable from the
fault `clockprobe-rocm` exists to catch** (`idle_clock_lock_suspected` fires on
slow + cool + busy; this is 41%, 43.8 °C, 98%). Filed as **#534**.

Note the phase-B temperature is *lower* than A and C. The leniency model assumes
slow-because-hot is forgivable and slow-while-cool is a fault. An APU has a
third case that is cool, slow, busy, and healthy.

**Consequence for any AMD profile: CPU-side and GPU-side tests must be
sequenced, never overlapped.** A profile that runs `memory-stress` alongside
`clockprobe` fails its own healthy nodes.

### 3.2 amd-smi is not blind on this part

`sysfs_clocks.h` and the `clockprobe-rocm` README both assert that amdsmi
"reports essentially every monitoring field as N/A on gfx1151" (ROCm#6035).

**[MEASURED] That is not true of this ROCm build.** amd-smi reports a full
`APU_*` vocabulary: `APU_AVERAGE_GFXCLK_FREQUENCY`, `APU_CURRENT_GFX_MAXFREQ`
(2900 MHz — agreeing exactly with the sysfs ladder top), `APU_AVERAGE_GFX_POWER`,
`APU_TEMPERATURE_GFX`, `APU_TEMPERATURE_SOC`, per-core clocks and power, DRAM
read/write rates, and ECC totals.

What *is* N/A is the **Instinct-shaped** surface: `HBM_STACKS`, `XCD`, `OAM_ID`,
`ASIC_SERIAL`, `VCN`/`JPEG` activity. So the claim is half right and mis-stated:
amd-smi is blind to the *datacenter* fields on an APU, not to APU monitoring.

**Sysfs-first remains the right architecture** — no library in the image, works
without a vendor SDK, and `rocm-smi` reads the same files. But the justification
in the source is wrong and the runner is leaving real signal on the table:
amd-smi is the only source for **GPU power as distinct from package power**, and
that distinction is large (11.5 W vs 35 W).

### 3.3 sysfs surface, exactly

**[MEASURED]** hwmon exposes only: `freq1_input` (GFX clock, Hz),
`temp1_input` (millidegrees), `power1_input` (microwatt), `in0/in1_input`,
and labels. Notably:

- **`power1_average` is absent.** `ReadPowerW`'s fallback to `power1_input` is
  the *only* working path on this part. It works — but it was written as a
  fallback and is in fact the primary.
- **`power1_input` is socket power**, ~3x the GPU's own draw. A `powerW` metric
  taken from it describes the package, not the accelerator.
- **No `power1_cap`**, so there is no readable power ceiling.
- The DPM ladder is three levels — `600 / 1100 / 2900 Mhz`, one starred. The
  `Mhz` spelling and star convention are exactly what `ParseDpmSclk` expects.

### 3.4 Device access, and the `n/a` branch

**[MEASURED]** `/dev/kfd` is `crw-rw----+ root render`, mode 0660, **gid 992**,
ACL `other::---`. The runner image's uid 65532 holds no such group and
**cannot open it**. The README's "hosts *may* need `runAsUser: 0` or a
supplemental group" is now a measured requirement, and the gid is host-specific
— so a pod must take it from the node, not hardcode it.

**[MEASURED]** There is **no `ras/` directory** on this part. That makes the
`eccErrors=n/a` branch — which #259 worried might be unreachable dead reasoning
— demonstrably live. It also means amd-smi reports ECC totals (0/0) where sysfs
has no source at all, so host-health's `n/a` is honest about *its* source while
a richer source exists.

### 3.5 CU versus WGP

**[MEASURED]** HIP reports `multiProcessorCount = 20`; amd-smi reports
`NUM_COMPUTE_UNITS: 40`. RDNA groups two CUs per work-group processor. A runner
that sizes its grid from one and reports `deviceCount`-adjacent counts from the
other will disagree with the vendor tool on the same node.

---

## 4. The gap: nothing checks configuration

AMD puts prerequisites and a configuration checker *before* the benchmarks. This
project has no equivalent, and on this platform that is expensive, because a
misconfigured Halo **measures slow while being perfectly healthy** — and this
project's whole discipline is about not condemning healthy hardware.

**[COMMUNITY]** The tuning surface, none of it verified here:

| Setting | Reported effect | Confidence |
|---|---|---|
| `amdgpu.gttsize` / `ttm.pages_limit=31457280` | Linux caps GPU-accessible RAM at ~50% without it | **[AMD]** confirms TTM limit + "GTT default ≈50% of RAM" |
| BIOS UMA framebuffer **small** (512 MB) + large GTT | counterintuitive; a big carveout starves the kernel | **[AMD]** "keep BIOS VRAM small (e.g. 0.5 GB), raise TTM instead" |
| Kernel **≥ 6.18.4** | KFD queue creation / memory availability | **[AMD]**. Our box: 6.18.44 ✓ |
| `HSA_OVERRIDE_GFX_VERSION=11.0.0` | segfaults libamdhip64 on gfx1151 | [COMMUNITY] |
| ROCm 7.x vs 6.4.4 | up to 3x slower; root-caused to a **compiler loop-unroll threshold** regression, workaround `-mllvm --amdgpu-unroll-threshold-local=600` | [COMMUNITY], ROCm/rocm-systems#2865 |
| `rocWMMA` flash-attention on gfx1151 | −41% at long context | [COMMUNITY], llama.cpp#24437 — corroborates **#331** |
| `ROCBLAS_USE_HIPBLASLT=1` | large prompt-processing win | [COMMUNITY] |
| `amdgpu.lockup_timeout=-1` | avoids watchdog GPU reset on long/deep work | [COMMUNITY] — **relevant to long soaks** |
| Fan curve / 120–140 W PPT | stock firmware allows ~95 °C before ramping; sustained load throttles | [COMMUNITY] |

Two of these are directly dangerous to *this* project:

**The ROCm 7.x regression would read as a hardware verdict.** A runner built on
a toolchain with that compiler regression measures a healthy part as slow. Our
images pin ROCm 6.4.4 — and the README's reason ("community has reported 7.x
regressions") now has a specific root cause and a workaround behind it. Keep the
pin; record why.

**The watchdog matters for segmented soaks.** A GPU reset mid-soak surfaces as
an `Error`, spends retry budget, and re-runs the segment — on a cause that is a
kernel parameter, not the silicon.

**Recommendation: an AMD `config-check` surface.** Not necessarily a new
TestKind — `fingerprint-probe` already exists and `NodeFingerprint` already
carries kernel and driver version. The cheapest honest version is to have the
AMD fingerprint path record GTT size, TTM page limit, kernel version and ROCm
version, so a node that measures slow can be *explained* rather than condemned.
That is evidence, not acceptance, and it belongs in the envelope.

---

## 5. A Halo is not an Instinct

This is the structural finding, and it is not really about AMD.

| | **Halo** (gfx1151, RDNA 3.5 APU) | **Instinct** (gfx942, CDNA 3) |
|---|---|---|
| Memory | 128 GB unified, GTT-managed | HBM3, dedicated |
| Power | ~120 W **shared with 16 CPU cores** | ~750 W, GPU-only |
| Matrix engine | WMMA (wave32 builtins) | MFMA |
| `rocWMMA` | **unsupported** (#331) | supported |
| RAS / `ras/` sysfs | **absent** | present |
| Fabric | none (single device) | XGMI + RoCE |
| ECC | no ECC block to read | ECC + row remapping |
| Thermal envelope | ~95 °C, chassis-limited | datacenter-cooled |
| Clock under load | **contends with the CPU (#534)** | does not |

Nearly every row changes a threshold, an image, or whether a kind applies at
all. `compute-smoke-rocm` already reports a CDNA part as `Error` because its
kernel covers gfx11 only (#331). A single `vendor: amd` image reference cannot
serve both, and a single set of thresholds certainly cannot.

**The mechanism this project has is vendor-level and stops one level too high.**
`spec.runner.imagesByVendor` selects on `nvidia|amd|intel|…`; both of these are
`amd`. `pkg/runnerimages.Resolve(kind, runner, vendor)` takes no finer key.

**The data already exists.** `NodeFingerprint.GPUInfo.Arch` is captured per GPU
(`sm_121` for GB10; `gfx1151` here). Nothing consumes it for selection.

And this is not an AMD problem wearing AMD clothes — **#10** is "compute-smoke:
SM10x (B200) needs its own runner image", which is the identical shape on the
NVIDIA side. The right framing is *fleet heterogeneity below the vendor line*,
which AMD merely makes urgent.

### The proposal

Two layers, deliberately separated, because one is small and obviously correct
and the other is an API design question.

**Layer 1 — image selection by arch (small, additive, backward-compatible).**
Give `VendorImage` an optional opaque `arch`:

```yaml
runner:
  imagesByVendor:
    - {vendor: amd, arch: gfx1151, image: …-clockprobe-rocm-gfx1151:v0.1.0}
    - {vendor: amd, arch: gfx942,  image: …-clockprobe-rocm-cdna:v0.1.0}
    - {vendor: amd,                image: …-clockprobe-rocm:v0.1.0}   # fallback
```

Resolution: exact `(vendor, arch)` → `(vendor, "")` → built-in default. The
controller does **string equality and nothing else** — it never learns what
`gfx1151` means, which is what keeps the vendor seam where `CLAUDE.md` requires
it. Implemented in this change.

**Layer 2 — thresholds per class (a GEP, not a patch).** Thresholds are the
larger half — #346 is exactly "per-class thermal ceilings" — and the options
(per-class threshold blocks, profile-level class selectors, or class-scoped
profiles selected per node) change the CRD's shape and how a `BurnInRun` relates
to a `BurnInProfile`. Per this project's rules that is a design proposal, not a
patch landed from a session, and it is raised as one rather than invented here.

---

## 6. What to adopt, in order

1. **Fix the verdict model before publishing anything** (#534). An immutable tag
   that fails healthy Halo nodes is the one mistake this project's history says
   never to make.
2. **Correct the amdsmi claim** in `sysfs_clocks.h` and the README (§3.2), and
   take GFX power from amd-smi where available rather than reporting package
   power as the GPU's.
3. **Land arch-based image selection** (§5 Layer 1) — it unblocks #331's CDNA
   split and #10's B200 case with the same mechanism.
4. **Answer #318 with RVS on real hardware.** RVS is MIT, it is the only
   redistributable vendor suite, and whether `gst`/`iet`/`babel` run on gfx1151
   is still unanswered — it is not packaged for this box and was not installable
   from the configured repos during this pass.
5. **Record configuration in the fingerprint** (§4), so a slow node can be
   explained.
6. **Raise the per-class threshold GEP** (§5 Layer 2).
7. **Then** publish the `-rocm` images with pinned tags (#531).

---

## Sources

- [AMD Instinct Customer Acceptance Guide](https://instinct.docs.amd.com/projects/system-acceptance/en/latest/)
- [AMD Strix Halo system optimization](https://rocm.docs.amd.com/en/docs-7.2.0/how-to/system-optimization/strixhalo.html)
- [ROCm Validation Suite modules](https://rocm.docs.amd.com/projects/ROCmValidationSuite/en/master/conceptual/rvs-modules.html)
- [AGFHC support, AMD GPU Operator](https://instinct.docs.amd.com/projects/gpu-operator/en/latest/test/agfhc.html)
- [ROCm 7+ llama.cpp performance regression](https://github.com/ROCm/rocm-systems/issues/2865)
- [rocWMMA flash-attention regression on gfx1151](https://github.com/ggml-org/llama.cpp/issues/24437)
- [gfx1151 rocBLAS/hipBLAS regression vs gfx1100](https://github.com/ROCm/ROCm/issues/4748)
