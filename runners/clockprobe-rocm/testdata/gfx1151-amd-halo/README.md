# Captured gfx1151 evidence — AMD Ryzen AI MAX+ 395 (Strix Halo)

Captured by `../../hwverify.sh` on `amd-halo` (kernel 6.18.44, ROCm 6.4,
40 CU / 20 WGP, 2900 MHz ladder top) during the #320 verification pass.

These are **real driver bytes**, not hand-written fixtures. The runner's parsing
was unit-tested against invented files before any hardware existed; this is what
it actually has to read.

| File | What it settles |
|---|---|
| `pp_dpm_sclk` | the real ladder. Three levels, `Mhz` spelling, one starred level — `ParseDpmSclk` reads it correctly, and the top (2900) is the rated clock |
| `vendor` | `0x1002`, so `AllAmdgpuCards` selects this card |
| `temp1_input` | millidegrees, confirming the unit the 90 °C threshold assumes |
| `power1_input` | **the only power file present.** `power1_average` is absent on this part, so `ReadPowerW`'s fallback is the sole working path — and what it reads is SOCKET power, not GPU power |
| `device-access.txt` | `/dev/kfd` is `root:render` mode 0660, gid 992, `other::---`. uid 65532 cannot open it |
| `gpu-only-samples.tsv` | 140 s at 1 Hz under an FP32 FMA load: 2900/2900 MHz flat, 97% busy, 37→44 °C |
| `thermal.csv` | GPU-only vs GPU+all-core-CPU, with amd-smi GFX temp/power alongside sysfs |
| `control.csv` | the discriminating run for #534: load → contend → release, and the clock recovers to 100% |

## What the capture proved

- The parse is right and the ladder top is a usable denominator.
- A healthy part sustains **100%** of rated clock — the shipped 60 floor is far
  below the hardware.
- **#534**: concurrent CPU load drops sclk to 41% while the part stays *cool*
  and *busy*, which is precisely `idle_clock_lock_suspected=true`. It recovers
  fully when the CPU load stops. The verdict model cannot currently tell socket
  power arbitration from the fault it exists to catch.
- The README's "87–91 °C under ordinary sustained load" is not reachable by this
  runner's load: the FMA chain draws 11.5 W GFX on a 120 W package.

Regenerate with `./hwverify.sh idle && ./hwverify.sh load 140 && ./hwverify.sh report`.
