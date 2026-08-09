# tcp-baseline

Plain TCP throughput between two nodes, so that "the fabric is broken" can be
told apart from "this node's networking is broken".

When an RDMA test fails, the next question is always the same one. The RDMA
runners need `/dev/infiniband`, memlock headroom and a working verbs stack; if
any of that is misconfigured they report a fabric fault that is really a
configuration fault, and an engineer is sent to look at a cable that is fine.
This runner answers the question in seconds, and it is the only fabric-adjacent
test in the suite that runs on hardware with **no RDMA and no accelerator at
all** — including the management path itself, and including nodes from vendors
this project has no other runner for.

That makes it a good first test in a profile and the obvious thing to reach for
when something else fails.

## The management-path guard

This is the rule that is not about measurement quality.

Every other runner in this suite loads a *device*. This one loads a *network*,
and getting it wrong means losing control of the fleet you are testing: kubelet
heartbeats missed, nodes marked `NotReady`, pods evicted fleet-wide, in the
middle of an acceptance run whose entire purpose was to avoid surprises. A
saturated management path also invalidates the measurement it was taken for —
contention with control-plane traffic is not a property of the link.

So the runner decides, before it starts anything, whether the traffic it is
about to generate would cross the interface carrying this node's **default
route**. That interface is treated as the management path: it is how the node
reaches the apiserver, and naming it is the conservative reading — it can only
ever refuse a test that would have been safe, never permit one that would not.

| Situation | Outcome | Why |
|---|---|---|
| A separate fabric interface reaches the peer | runs | the case this test exists for |
| The only route to the peer is the default-route interface | **Skip** | a one-NIC node cannot be load-tested safely; that is a fact about the hardware, not a fault in it |
| `TCP_BASELINE_INTERFACE` names the default-route interface | **Error** | the author asked for something unsafe, and a silent skip would leave them believing it ran |
| `TCP_BASELINE_INTERFACE` names an interface the peer is *not* routed through | **Error** | the name is an assertion, and it is false — see below |
| `/proc/net/route` unreadable, or no default route | **Error** | fails closed — not knowing which interface is the management one is not permission to find out by saturating it |
| The peer's interface cannot be determined | **Error** | same |

There is **no `--force`**. The only honest reason to want one is "I am confident
this is fine", and the cases where an operator is confident and wrong are
exactly the ones that cost a fleet.

`TCP_BASELINE_INTERFACE` is an **assertion to be checked, not an answer to be
trusted**. Nothing binds iperf3 to it — no `--bind` is passed, deliberately,
because the routing table is what decides where a packet goes and a runner that
overrode it would be measuring a path the node does not use. So wherever a peer
address exists to route towards, the runner looks the route up *anyway* and
refuses a name that disagrees with it. Trusting the name would let the guard
clear a run whose load then went out over the management path — the string
matched nothing, the packets left via the default route regardless, and the
guard reported that it had checked. A refusal that never fires is worse than no
guard, because it reads as a safety property. The server end has no resolvable
peer yet (the operator does not create the client until the server is Ready), so
there is nothing to contradict and the name stands on its own.

The decision is recorded in the result — `tcpTestInterface` and
`tcpMgmtInterface` — on every path including the refusals, because a guard whose
decision is not in the output cannot be audited afterwards.

## Contract

```
exit 0                                the path was measured
exit 1                                iperf3 completed and measured nothing
exit 2  TCP_BASELINE_SKIP: <reason>   does not apply to this node
exit 3  TCP_BASELINE_ERROR: <reason>  machinery, including the guard failing closed
```

An `iperf3` connection error is an **Error**, never a Fail. "Unable to connect"
is a statement about this run, not evidence about a cable, and only an Error is
retried.

### Metrics

| Metric | Meaning |
|---|---|
| `tcpThroughputGbps` | sender-side throughput |
| `tcpRetransmits` | retransmitted segments, **omitted** where the platform has no `TCP_INFO` — see below |
| `tcpRttUs` | mean smoothed RTT, **omitted** where the platform has no `TCP_INFO` |
| `elapsedS` | measured window |
| `tcpTestInterface`, `tcpMgmtInterface` | the guard's decision |

All read from iperf3's **sender** summary. Retransmits and RTT are properties
the sending stack observes and the receiver cannot report; taking throughput
from one end and the rest from the other would produce a result whose numbers
came from two different places.

Both come from `TCP_INFO` and both are **omitted, never zeroed**, where the
platform does not supply them. A zero there is a count nobody took, and
`tcpRetransmits` is the metric the sample gates `Equal 0` — so a fabricated zero
would not merely mislead, it would *pass*, certifying a path whose retransmits
were never counted. An omitted metric fails a `Required` gate closed, which is
the correct direction.

**Retransmits are the interesting signal.** A link that reaches line rate while
retransmitting heavily has a problem that throughput alone will not show, and it
surfaces later as tail latency in a collective. `tcpThroughputGbps` is what
people gate on; `tcpRetransmits` is what finds the marginal cable.

A missing `mean_rtt` is omitted rather than reported as `0` — a zero RTT is a
measurement nobody took, and a gate on it would certify a path this runner never
timed. Zero *retransmits*, by contrast, is a real measurement of a clean link
and is always reported.

## Environment

Standard Pair rendezvous: `BURNIN_ROLE`, `BURNIN_PEER_HOST`, `BURNIN_PEER_NODE`,
`BURNIN_DURATION_SECONDS`. Plus:

| Variable | Meaning |
|---|---|
| `TCP_BASELINE_PORT` | iperf3 port (default 5201) |
| `TCP_BASELINE_INTERFACE` | fabric interface to test. **Required on the server**, which has no peer address to route towards and therefore nothing for the guard to compare |

Declare a `readinessProbe` on the server naming the same port. The operator will
not start the client until the server pod is Ready, but without a probe "Ready"
only means the container started.

`hostNetwork: true` is required: the guard reads `/proc/net/route` and
enumerates the host's interfaces, and a fabric test wants the host namespace
anyway. Without it the guard fails closed and the test reports an Error — the
intended degradation, since it will not measure a path it cannot classify.

## Licensing

iperf3 is BSD-3-Clause (Lawrence Berkeley National Laboratory), on this
project's permissive allow-list. It is built from source at a pinned commit and
the build asserts both the licence text and the licensor; a relicensed fork that
kept the BSD wording would fail the second check. Its `LICENSE` ships at
`/licenses/iperf3-LICENSE`.

No accelerator, no vendor stack, no NVIDIA driver injection —
`runners/pins_test.go` asserts that absence rather than leaving it to a reader.

## Status

**Not published.** There is no `tcp-baseline` entry in `pkg/runnerimages`, so a
BurnInTest of this kind fails at plan time asking for an explicit
`spec.runner.image` rather than pull-failing on every targeted node. Publishing
is manual and follows verification on real hardware, per repository policy.
