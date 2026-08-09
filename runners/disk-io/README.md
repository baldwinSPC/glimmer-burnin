# disk-io

Sequential throughput and operation latency against a site-declared path, with
the page cache bypassed.

Storage is the subsystem this suite otherwise never touches, and on an AI node it
is not a minor one. Checkpoint writes, dataset streaming and shared-filesystem
reads are where a training run stalls when the NVMe underneath it is degrading —
and a drive with a thermal-throttling controller or rising media errors passes
every other test this operator runs.

## What it writes, and where

This is the one runner in the suite that can destroy the thing it measures, so
the rules are stated before anything else. All four defaults fall towards
"cannot lose data".

1. **No declared path, no test.** There is no default directory — not `/tmp`,
   not `/var`, nothing. `DISK_IO_PATH` is required and the runner **skips**
   without it. A default would measure the container's own overlay filesystem
   and report it as the node's NVMe.

2. **It writes one file, which it created.** `O_EXCL`, so an existing file is
   never opened and never truncated. A leftover from a crashed run becomes a
   refusal, not an overwrite of something that might not be ours. The file is
   removed on every exit path, including the failures.

3. **It leaves headroom.** The run is refused unless the write fits with 2 GiB
   still free afterwards. A write sized to fill the filesystem destroys nothing
   directly and takes the node down anyway — a full volume stops scheduling, and
   a burn-in that causes that has damaged the fleet while measuring it.

4. **It reads back what it wrote.** The read measurement never touches site
   data, so pointing this at a directory holding a dataset measures the device
   without reading a byte of the dataset.

What is deliberately **not** offered is a read-only mode over existing files. It
sounds safer and is worse: the measurement would depend on whatever happens to
be lying in the directory — file sizes, extents, page-cache residency — so two
nodes with identical hardware would report different numbers, and a slow node
could not be told from an unlucky layout.

## O_DIRECT, or nothing

The page cache is bypassed, and there is **no fallback to buffered I/O**.

A 1 GiB buffered write on a node with 128 GiB of RAM measures `memcpy`. The
numbers come back enormous, they look like a very healthy NVMe, and a failing
drive is certified. If `O_DIRECT` is unavailable — a filesystem that refuses it,
or a non-Linux platform — the runner **skips**. That is a fact about the target,
not a fault in it, and refusing to answer is the only honest response.

## Contract

```
exit 0                            the device was measured
exit 1                            an I/O error: the device was reached and it failed
exit 2  DISK_IO_SKIP: <reason>    does not apply to this node
exit 3  DISK_IO_ERROR: <reason>   machinery, including a refused write
```

An I/O error mid-run is a **Fail**, not an Error — the device was reached and it
failed, which is a verdict about the part. A refused write (no space, a leftover
file, a bad block size) is an **Error**: nothing was measured.

### Metrics

| Metric | Meaning |
|---|---|
| `writeBandwidthMBs` | sequential write, decimal MB/s, **fsync inside the window** |
| `readBandwidthMBs` | sequential read of the file just written |
| `ioLatencyUs` | mean per-operation latency across both directions |
| `p99LatencyUs` | 99th percentile, nearest-rank — a value an operation really took |
| `ioErrors` | I/O errors during the run |
| `diskIoPath` | the directory measured (evidence) |

A direction that moved no bytes emits **nothing** for itself rather than a zero:
a zero MB/s is a measurement nobody took, and a floor gate on it would fail a
node for a run that never started. `ioErrors=0` is the opposite case and is
always emitted — the run reached the device and counted none, which is what a
gate of `ioErrors Equal 0` needs to see to pass a healthy node.

`p99LatencyUs` is the metric worth having. A drive whose mean is fine and whose
p99 is 40× the mean is a drive that stalls a training step, and a mean alone
hides exactly that.

## Environment

| Variable | Default | Meaning |
|---|---|---|
| `DISK_IO_PATH` | — | **required**; absolute directory, mounted via `spec.runner.hostPaths` |
| `DISK_IO_SIZE_MB` | 1024 | bytes written, before the free-space check |
| `DISK_IO_BLOCK_KB` | 1024 | must be a multiple of 4 KiB, the `O_DIRECT` alignment |
| `BURNIN_DURATION_SECONDS` | 120 | caps the run; a slow device writes less than `DISK_IO_SIZE_MB` |

Host access is through `spec.runner.hostPaths` and nothing else. A test that
declares no mounts gets a pod with no volumes, and this runner then skips.

## Licensing

**No third-party component at all**, and that is a licensing decision rather
than a minimalist one: fio, IOR and elbencho are all GPL, and a copyleft
dependency would make this project unpublishable. The measurement is a
standard-library Go program.

Enforced twice rather than remembered: `metricnames_test.go` fails `make test`
if any non-test source imports a path outside the standard library, and the
image builds with `GOPROXY=off` so a dependency cannot be fetched at all.

## Not implemented

**SMART / NVMe health counters.** Reading them needs an NVMe admin passthrough
ioctl, which is real work and unverifiable without the hardware. Rather than
read a subset badly, this runner reports none — the same rule the ECC path
follows: report what was positively established, omit what could not be read.

## Status

**Not published.** There is no `disk-io` entry in `pkg/runnerimages`, so a
BurnInTest of this kind fails at plan time asking for an explicit
`spec.runner.image` rather than pull-failing on every targeted node. Publishing
is manual and follows verification on real hardware.
