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
| `/proc/net/route` unreadable, or no default route | **Error** | fails closed — not knowing which interface is the management one is not permission to find out by saturating it |
| The peer's interface cannot be determined | **Error** | same |

There is **no `--force`**. The only honest reason to want one is "I am confident
this is fine", and the cases where an operator is confident and wrong are
exactly the ones that cost a fleet.

The decision is recorded in the result — `tcpTestInterface` and
`tcpMgmtInterface` — on every path including the refusals, because a guard whose
decision is not in the output cannot be audited afterwards.

### The server's half: accept, then classify

The client can run this decision immediately — `BURNIN_PEER_HOST` resolves the
moment it starts, so `classifyRoute` has a real route to compare. The server
cannot: at Pair scope the operator does not create the client pod until the
server is Ready, so the server's peer is a DNS name with nothing behind it at
the point it would need to classify anything. Naming an interface by hand
(`TCP_BASELINE_INTERFACE`) used to be the only way past that — required on the
server, and refused with an Error otherwise.

Since #482 the server instead waits for its real peer to arrive and reads
which of its **own** interfaces that specific connection landed on — a fact
the kernel can only hand over once a connection exists, and the same move
`nccl`'s server makes for the identical structural problem. A small handshake
on a dedicated port (`TCP_BASELINE_GUARD_PORT`, default 5202) does the
waiting: the client says `HELLO`, the server classifies the connection's own
local address exactly as `classifyRoute` would classify a route lookup, and
answers `OK`, `SKIP:<reason>`, or `ERROR:<reason>` before either side has
carried a single byte of load. A refusal here means iperf3 never starts, and a
client reads its peer's own refusal rather than a listener that quietly never
appeared.

`TCP_BASELINE_INTERFACE` still works exactly as before, if set on the server —
it is judged the same way (naming the management interface is still an Error,
not a Skip) and simply skips learning the interface from the connection,
without skipping the handshake itself.

The guard port tolerates a Kubernetes `tcpSocket` readinessProbe landing on it
repeatedly: only a connection that actually sends `HELLO` is treated as the
real peer, so a bare probe connect-and-close is seen, ignored, and the
listener goes back to waiting.

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
| `tcpRetransmits` | retransmitted segments — see below |
| `tcpRttUs` | mean smoothed RTT, **omitted** where the platform has no `TCP_INFO` |
| `elapsedS` | measured window |
| `tcpTestInterface`, `tcpMgmtInterface` | the guard's decision |

All read from iperf3's **sender** summary. Retransmits and RTT are properties
the sending stack observes and the receiver cannot report; taking throughput
from one end and the rest from the other would produce a result whose numbers
came from two different places.

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
| `TCP_BASELINE_GUARD_PORT` | accept-then-classify handshake port (default 5202). Point the server's `readinessProbe` here, not at `TCP_BASELINE_PORT` — see below |
| `TCP_BASELINE_INTERFACE` | override: skip discovery and classify this interface instead, on whichever end sets it. No longer required on the server — see "The server's half" above |

## Start ordering

The client **waits up to 120s for the server's listener** before measuring
anything, and that wait is in the runner rather than in an iperf3 flag on
purpose: `--connect-timeout` bounds a single attempt, and a connection to a port
nothing is listening on is refused immediately with an RST, so the timeout never
elapses and iperf3 exits at once. Without the wait, a client that starts first
dies instantly with "unable to connect" — which reads to an operator like a
fabric fault, and is exactly what this runner exists to stop people misreading.

Declare a `readinessProbe` on the server naming `TCP_BASELINE_GUARD_PORT`
(5202 by default) — **not** `TCP_BASELINE_PORT`. The operator will not create
the client pod until the server is Ready, but iperf3's own port does not open
until the accept-then-classify handshake has a verdict, and that handshake
needs the client to exist. Probing 5201 would deadlock the pair against
itself: Ready waits on iperf3, iperf3 waits on the client, and the client
waits on Ready. The guard port is bound first, unconditionally, before any
classification happens, which is what breaks the cycle. The client's own wait
above still matters on top of the probe: the probe narrows the window between
"Ready" and "actually listening", and the wait is what survives what is left
of it.

`hostNetwork: true` is required: the guard reads `/proc/net/route` and
enumerates the host's interfaces, and a fabric test wants the host namespace
anyway. Without it the guard fails closed and the test reports an Error — the
intended degradation, since it will not measure a path it cannot classify.

That was the documented intent and not the behaviour, until #285. From inside a
pod namespace the guard saw exactly one non-loopback interface which also
carried the default route — **identical to a single-NIC node** — and returned
the declared Skip meant for that case. A Skip is never retried and does not
count against a run, so a profile that merely omitted `hostNetwork: true` had
every node report this test as not-applicable, with the fabric unmeasured and
the acceptance reading clean.

The guard now establishes the host namespace *positively* — `/proc/self/ns/net`,
corroborated by `/proc/1/ns/net` when the pod also shares the host PID namespace
— and records it as `tcpNetNamespace`. The single-NIC Skip is reachable only
once that is known; otherwise the same observation is an Error, because "only
one interface is visible" is not a declaration that this is a one-NIC node.
Absence is not a declaration.

## Licensing

iperf3 is BSD-3-Clause (Lawrence Berkeley National Laboratory), on this
project's permissive allow-list. It is built from source at a pinned commit and
the build asserts both the licence text and the licensor; a relicensed fork that
kept the BSD wording would fail the second check. Its `LICENSE` ships at
`/licenses/iperf3-LICENSE`.

No accelerator, no vendor stack, no NVIDIA driver injection —
`runners/pins_test.go` asserts that absence rather than leaving it to a reader.

## Status

**Published at v0.1.0**, verified through the operator on two real GB10 Sparks
with separate fabric and management interfaces (#237). With no
`TCP_BASELINE_INTERFACE` override, the client's route lookup correctly picked
the fabric interface over the management one — `ifaceForAddr` reads a real
multi-interface routing table, not just a single-NIC laptop. Explicitly naming
the management interface was refused with exit 3, as documented above. A
healthy pair measured 43.7 Gbps, 0 retransmits, 146us RTT.

Not yet verified on real hardware: a peer reachable only through the
management interface producing a Skip (this fleet's fabric route always
exists, so the Skip path needs a topology this pair doesn't have); the
server outliving a settled client without stranding a retrying one; and the
CLI dispatcher (`burnin run --role server|client`) path. See #237 for the full
checklist.

**The accept-then-classify handshake (#482)** replaces the server's old
`TCP_BASELINE_INTERFACE`-required refusal with the design above. Covered by
unit tests against real loopback sockets (`handshake_test.go`), and the
classification itself — the part that changed — is confirmed against real
data from **spark-043a**: with no `TCP_BASELINE_INTERFACE` set, a real
connection landing via `enp1s0f1np1` (a real collective-rail address) was
correctly accepted (`tcpTestInterface=enp1s0f1np1`), and a real connection
landing via `wlP9s9` (the node's real default route) was correctly refused as
`routeIsManagement`, with the same reason text a fleet operator would see.
Both cases ran the real compiled binary against `/proc/net/route` and real
host interfaces on GB10/Linux-arm64 — not synthetic input, and not something
the macOS dev loop that wrote this can exercise directly.

**The full two-pod Pair path, on Kubernetes, end to end (2026-08-29):** a real
`BurnInRun` through the operator, both Sparks, no `TCP_BASELINE_INTERFACE`
anywhere in the spec. The server's guard port satisfied its `readinessProbe`
immediately — proving the deadlock this design exists to avoid does not
happen — the operator created the client pod on `spark-85a9` only once that
probe passed, real cluster DNS resolved the server's headless-Service name,
the accept-then-classify handshake completed with no retry needed, and
iperf3 measured a real **49.68 Gbps, 0 retransmits, 290us mean RTT** across
the fleet's dual-rail fabric — `tcpTestInterface=enP2p1s0f1np1` on the client
end, discovered exactly as before #482 with no configuration anywhere in the
`BurnInTest`. Both the classification rule and the Kubernetes orchestration
around it are now hardware-confirmed.
