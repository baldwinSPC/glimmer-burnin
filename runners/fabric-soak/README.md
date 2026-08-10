# fabric-soak

The RDMA link between two nodes, run for hours instead of for a minute.

`ib-write-bw` answers *does this link work right now*: about a minute of
traffic, an average, a verdict. It passes on a link with a marginal optic, a
cable seated badly, or a transceiver that fails once it is warm — and those are
the failures that hurt most in production, because they present as a slow
training job or an occasional collective timeout rather than as a link that is
down.

This runner iterates the same measurement and reports what one run cannot.

## What it reports, and why not the average

| Metric | Meaning |
|---|---|
| `minBandwidthGbps` | **the worst window** — the figure a link is accepted on |
| `p1BandwidthGbps` | 1st percentile; a better floor on a long soak than a single minimum one hiccup can drag down |
| `meanBandwidthGbps` | evidence |
| `bandwidthStdDevGbps` | the spread |
| `soakIterations`, `soakFailedIterations` | windows attempted, and windows that did not complete |
| `linkErrorEvents` | port error counters that **moved during the run** |

A soak that averaged fine and spent ninety seconds at a third of line rate has
found something, and the mean hides it completely. That is why the minimum is
the gated figure.

The **spread** is reported alongside it because a flapping link and a steadily
slow one both produce a low minimum — and only the flapping one produces a wide
spread. Telling them apart is the difference between replacing an optic and
re-checking a cable budget.

**A failed window contributes to the failure count and to nothing else.**
Averaging its zero into the bandwidth would report a fabric slower than it is,
which is the fabricated-measurement failure this project forbids everywhere.

## `linkErrorEvents` is a delta, never a lifetime total

The port counters — symbol errors, link recoveries, link-downs, receive errors —
are lifetime totals. A NIC that has been up for two hundred days shows a large
`symbol_error_counter` that says nothing about the last four hours; reporting it
raw would make every long-lived node look faulty and every freshly booted one
look clean.

So the baseline is taken immediately before the first window, and the figure is
what **moved during the soak**. That is what makes `linkErrorEvents Equal 0` a
gate worth writing.

Two cases produce **`n/a`** rather than a number, and neither is a zero:

- **No counter could be read** — no sysfs mount. An unread counter reported as
  zero is a fabric certified by a file nobody opened.
- **A counter went backwards** — the port was reset mid-soak, so no delta over
  the window exists. Clamping to zero would turn a link that bounced into a
  clean run.

Pair a gate on it with `applicability: RequiredIfMeasurable`.

## Progress survives a killed run

Cumulative figures are re-emitted every 60 seconds. The parser is
last-occurrence-wins, so each emission supersedes the last and a soak evicted at
hour six still reports everything measured up to that point. Without it, a
seven-hour run cut short reports nothing at all.

This is the primary consumer of the segmented-soak engine: segmentation improves
it (a segment boundary is a natural emission point and the counters aggregate
`Sum` for failures, `Min` for the floor), and a single long execution works
today without it.

## Contract

```
exit 0                              the link was soaked
exit 1                              measured and wrong: iterations failed
exit 2  FABRIC_SOAK_SKIP: <reason>  no RDMA device, or not half of a Pair
exit 3  FABRIC_SOAK_ERROR: <reason> machinery; the link is UNJUDGED
```

Ordinary Pair rendezvous — `BURNIN_ROLE`, `BURNIN_PEER_HOST` — and no new
environment beyond two tuning knobs:

| Variable | Default | Meaning |
|---|---|---|
| `FABRIC_SOAK_WINDOW_SECONDS` | 20 | one iteration's traffic |
| `FABRIC_SOAK_SYSFS` | `/sys` | where the host's sysfs is mounted |

A window must be shorter than the soak, and the runner refuses otherwise: a soak
of one window is an `ib-write-bw` run with a longer name and reports no spread.

## Port selection and the memlock limit

The port is chosen by `selectPort` — the device carrying the route to the peer,
not simply the first one enumerated. On a single-HCA node those are the same; on
a multi-HCA node they are not, and each end picking its own first device would
have the two soaking different fabrics, or one talking to a device with no route
to the other at all, which perftest reports as a connection failure that reads
as a dead link.

`RLIMIT_MEMLOCK` is read **before the first window** and the transfer is sized to
fit it. Every RDMA buffer is pinned memory, so `ibv_create_cq` fails with ENOMEM
when the limit is small, and perftest reports that as `Couldn't create CQ` —
which names neither the limit nor the resource and is indistinguishable from a
broken HCA.

That matters more here than in a one-shot test: a one-minute run fails once and
loudly, while a four-hour soak would fail *every window for four hours* and
report a link that carried no traffic, with the real cause named nowhere. A
window that fails that way is reported as an Error naming `RLIMIT_MEMLOCK`, not
as a fabric verdict.

## Status

**Not published**, and no measurement it produces has come from real hardware —
it needs two nodes with an RDMA link. Ship it thresholdless and gather baselines
before pinning anything except `soakFailedIterations Equal 0`.
