// SPDX-License-Identifier: Apache-2.0
// Copyright the Glimmer authors.

// Command tcp-baseline measures plain TCP throughput between two nodes.
//
// It exists to answer the question that always follows an RDMA failure: is the
// FABRIC broken, or is this node's networking broken in general? The RDMA
// runners need /dev/infiniband, memlock headroom and a working verbs stack, and
// if any of that is misconfigured they report a fabric fault that is really a
// configuration fault. A plain TCP test between the same two nodes separates
// those cases in seconds.
//
// It is deliberately vendor-free and accelerator-free: no CUDA, no ROCm, no
// verbs. It builds and runs on any node in any fleet, including one with no
// accelerator at all, which makes it the natural first test in a profile and
// the natural sanity check when something else fails.
//
// Contract:
//
//	exit 0                                  the path was measured
//	exit 1                                  a threshold-free failure: iperf3 itself reported one
//	exit 2  TCP_BASELINE_SKIP: <reason>     the test does not apply to this node
//	exit 3  TCP_BASELINE_ERROR: <reason>    machinery, including the guard failing closed
//
// Rendezvous is the ordinary Pair contract — BURNIN_ROLE, BURNIN_PEER_HOST — so
// one image is the server on one node and the client on the other, and it never
// learns anything about Kubernetes.
package main

import (
	"encoding/json"
	"fmt"
	"net"
	"os"
	"os/exec"
	"strconv"
	"strings"
	"time"
)

const (
	exitPass  = 0
	exitFail  = 1
	exitSkip  = 2
	exitError = 3

	roleServer = "server"
	roleClient = "client"

	// defaultPort is iperf3's own default. Declared here because the server's
	// readinessProbe has to name the same number, and a probe pointing at a
	// port nothing listens on turns "Ready" back into "the container started".
	defaultPort = 5201
)

// iperfVersion is stamped by the Dockerfile from IPERF_REF, so a result can be
// traced to the tool that produced it without inspecting the image.
var iperfVersion = "unknown"

func main() { os.Exit(run()) }

// fin prints the declared marker for a non-pass outcome and returns the code.
//
// The marker is required, not decorative: an exit 2 WITHOUT a recognised
// *_SKIP marker is an Error by contract, precisely so that a runner which
// crashes into a 2 cannot be mistaken for one that declared itself
// inapplicable.
func fin(code int, format string, args ...any) int {
	msg := fmt.Sprintf(format, args...)
	switch code {
	case exitSkip:
		fmt.Printf("TCP_BASELINE_SKIP: %s\n", sanitize(msg))
	case exitError:
		fmt.Printf("TCP_BASELINE_ERROR: %s\n", sanitize(msg))
	case exitFail:
		fmt.Printf("TCP_BASELINE_FAIL: %s\n", sanitize(msg))
	}
	logf("%s", msg)
	return code
}

func run() int {
	role := strings.TrimSpace(os.Getenv("BURNIN_ROLE"))
	peerHost := strings.TrimSpace(os.Getenv("BURNIN_PEER_HOST"))
	peerNode := strings.TrimSpace(os.Getenv("BURNIN_PEER_NODE"))
	duration := envInt("BURNIN_DURATION_SECONDS", 30)
	port := envInt("TCP_BASELINE_PORT", defaultPort)

	// Group scope sets BURNIN_RANK and deliberately does NOT set BURNIN_ROLE.
	// Skipping rather than erroring: a profile may legitimately run this kind
	// at Pair scope elsewhere, and this pod simply is not that.
	if os.Getenv("BURNIN_RANK") != "" && role == "" {
		return fin(exitSkip, "BURNIN_RANK is set, so this is a Group execution; this runner speaks only "+
			"the Pair rendezvous (BURNIN_ROLE/BURNIN_PEER_HOST)")
	}
	if role == "" {
		return fin(exitSkip, "BURNIN_ROLE is not set, so this pod is not half of a Pair — a link test with "+
			"one end measures nothing")
	}
	if role != roleServer && role != roleClient {
		return fin(exitError, "BURNIN_ROLE=%q is neither %q nor %q", role, roleServer, roleClient)
	}
	if role == roleClient && peerHost == "" {
		return fin(exitError, "BURNIN_ROLE=client but BURNIN_PEER_HOST is empty — there is no peer to dial")
	}

	if _, err := exec.LookPath("iperf3"); err != nil {
		return fin(exitError, "iperf3 is not on PATH in this image: %v", err)
	}
	logf("tcp-baseline: iperf3 %s", iperfVersion)

	// ── The guard, before anything is started ────────────────────────────────
	//
	// On BOTH ends. The server is the one that will actually receive the load,
	// so checking only the client would leave the interface that matters
	// unchecked — and a server bound on the management path is exactly the
	// shape this guard exists to refuse.
	if code, ok := guardPath(peerHost, role); !ok {
		return code
	}

	if role == roleServer {
		return runServer(port, duration, peerNode)
	}
	return runClient(peerHost, peerNode, port, duration)
}

// guardPath runs the management-path guard and turns its decision into a
// contract outcome. Returns ok=false with the exit code when the run must not
// proceed.
func guardPath(peerHost, role string) (int, bool) {
	route, err := os.ReadFile("/proc/net/route")
	if err != nil {
		return fin(exitError, "could not read /proc/net/route (%v), so the management-path guard cannot "+
			"run. This image is Linux-only and needs the host network namespace; a burn-in must never "+
			"saturate the interface the cluster is managed through, so it refuses rather than guesses", err), false
	}
	mgmt := defaultRouteIface(string(route))

	// An explicitly named interface is the author's own choice and is judged
	// more harshly than one the routing table picked — see classifyRoute.
	explicit := strings.TrimSpace(os.Getenv("TCP_BASELINE_INTERFACE"))
	testIface := explicit
	if testIface == "" && peerHost != "" {
		testIface = ifaceForAddr(peerHost, hostIfaces())
	}
	if testIface == "" && role == roleServer && explicit == "" {
		// A server has no peer address to route towards yet. It is told which
		// interface to bind, or the guard cannot say anything about it.
		return fin(exitError, "the server end needs TCP_BASELINE_INTERFACE naming the fabric interface to "+
			"bind: with no peer address to route towards, there is nothing to compare against the "+
			"management interface (%s), and the guard fails closed rather than binding everywhere", mgmt), false
	}

	// Which network namespace this is, established positively where it can be.
	// The routing table above describes whatever namespace we are in, and in a
	// pod that is the pod's — see netnsKind (#285).
	selfNS, _ := os.Readlink("/proc/self/ns/net")
	pid1NS, _ := os.Readlink("/proc/1/ns/net")
	ns := netnsFromLinks(selfNS, pid1NS)
	if selfNS != "" {
		metric("tcpNetNamespace", selfNS)
	}

	result := classifyRoute(testIface, mgmt, explicit != "", ns)

	// Recorded on every path, including the refusals: the guard's decision is
	// only auditable if the interfaces are in the result.
	if result.iface != "" {
		metric("tcpTestInterface", result.iface)
	}
	if result.mgmtIface != "" {
		metric("tcpMgmtInterface", result.mgmtIface)
	}

	switch result.decision {
	case routeIsManagement:
		return fin(exitSkip, "%s", result.reason), false
	case routeUnknown:
		return fin(exitError, "%s", result.reason), false
	}
	return 0, true
}

// runServer starts iperf3 in server mode and waits.
//
// The server never decides. It is torn down when the pair settles, which the
// operator does on the CLIENT's terminating — so this deliberately runs until
// killed rather than exiting on its own after one connection. An iperf3 server
// that exits after the first test would leave a retrying client dialling a
// socket nobody is listening on.
func runServer(port, duration int, peerNode string) int {
	logf("tcp-baseline: server on :%d, peer node %s", port, peerNode)

	// --one-off is NOT used, for the reason above.
	cmd := exec.Command("iperf3", "--server", "--port", strconv.Itoa(port))
	cmd.Stdout = os.Stderr // iperf3's server chatter is not our metrics stream
	cmd.Stderr = os.Stderr

	if err := cmd.Start(); err != nil {
		return fin(exitError, "starting the iperf3 server: %v", err)
	}

	// Outlive the client's window by a margin. If the client is still running
	// when this returns, the operator's own deadline is what ends the pair.
	timer := time.AfterFunc(time.Duration(duration+serverGraceSeconds)*time.Second, func() {
		_ = cmd.Process.Kill()
	})
	defer timer.Stop()

	err := cmd.Wait()
	// A killed server is the expected end of a healthy pair, not a failure.
	if err != nil && !strings.Contains(err.Error(), "signal:") {
		return fin(exitError, "the iperf3 server exited badly: %v", err)
	}
	logf("tcp-baseline: server finished")
	return exitPass
}

// serverGraceSeconds is how long the server outlives the client's declared
// window before killing itself, so a client that started late still finds a
// listener.
const serverGraceSeconds = 60

func runClient(peerHost, peerNode string, port, duration int) int {
	logf("tcp-baseline: client → %s:%d (node %s) for %ds", peerHost, port, peerNode, duration)

	// Wait for the server's listener before measuring anything.
	//
	// THIS is the retry, and it has to be here rather than in iperf3's own
	// flags: --connect-timeout bounds ONE attempt, and a connection to a port
	// nothing is listening on is refused immediately with an RST, so the
	// timeout never elapses and iperf3 exits at once. Without this loop a
	// client that starts before its server dies instantly with "unable to
	// connect" — the exact failure mode this design exists to avoid, and one
	// that reads to an operator like a fabric fault.
	//
	// Start ordering is not guaranteed on either dispatcher: in a cluster the
	// operator gates on the server pod being Ready, which without a
	// readinessProbe only means the container started, and on bare metal the
	// ordering is whatever a human or a script did.
	if err := waitForListener(peerHost, port, time.Duration(connectWaitSeconds)*time.Second); err != nil {
		return fin(exitError, "no listener on %s:%d after %ds (node %s): %v — the server end never came up, "+
			"which is a statement about this run and not about the link",
			peerHost, port, connectWaitSeconds, peerNode, err)
	}

	out, err := runIperf(peerHost, port, duration)
	if len(out) > 0 {
		// Echoed to stderr so a failure is diagnosable from pod logs without
		// re-running anything, and never to stdout, where it would be parsed
		// as metrics.
		logf("---- iperf3 json ----\n%s\n---- end ----", oneLine(out))
	}
	if err != nil {
		return fin(exitError, "iperf3 did not complete against %s (node %s): %v — "+
			"a connection error here is a statement about this run, not about the link", peerHost, peerNode, err)
	}

	res, perr := parseIperf(out)
	if perr != nil {
		return fin(exitError, "could not read iperf3's output: %v", perr)
	}
	if res.Error != "" {
		// iperf3 reports its own errors in the JSON. Machinery, not hardware:
		// "unable to connect" is not evidence about a cable.
		return fin(exitError, "iperf3 reported: %s", res.Error)
	}

	// The metrics, whatever the verdict. A number is worth recording even when
	// the run is about to fail, because the number is what explains it.
	metric("tcpThroughputGbps", trim(res.ThroughputGbps))
	metric("tcpRetransmits", strconv.Itoa(res.Retransmits))
	if res.HasRTT {
		metric("tcpRttUs", trim(res.MeanRttUs))
	}
	metric("elapsedS", trim(res.Seconds))

	if res.ThroughputGbps <= 0 {
		return fin(exitFail, "iperf3 completed but measured no throughput to %s (node %s)", peerHost, peerNode)
	}

	logf("tcp-baseline: %.3f Gbps, %d retransmits", res.ThroughputGbps, res.Retransmits)
	return exitPass
}

func runIperf(peerHost string, port, duration int) (string, error) {
	args := []string{
		"--client", peerHost,
		"--port", strconv.Itoa(port),
		"--time", strconv.Itoa(duration),
		"--json",
		// Bounds the connection attempt itself. The RETRY that lets a client
		// survive starting first is waitForListener, above — see the note
		// there for why a flag cannot do that job.
		"--connect-timeout", strconv.Itoa(connectTimeoutMs),
	}
	cmd := exec.Command("iperf3", args...)
	var sb strings.Builder
	cmd.Stdout = &sb
	cmd.Stderr = os.Stderr

	// Bounded well beyond the declared window; the operator's deadline is the
	// real backstop.
	done := make(chan error, 1)
	if err := cmd.Start(); err != nil {
		return "", err
	}
	go func() { done <- cmd.Wait() }()

	select {
	case err := <-done:
		return sb.String(), err
	case <-time.After(time.Duration(duration+serverGraceSeconds) * time.Second):
		_ = cmd.Process.Kill()
		return sb.String(), fmt.Errorf("iperf3 did not finish within %ds", duration+serverGraceSeconds)
	}
}

// connectTimeoutMs bounds one connection attempt.
const connectTimeoutMs = 30000

// connectWaitSeconds is how long a client waits for its server's listener.
//
// Generous on purpose. The cost of waiting too long is a slower run; the cost
// of waiting too briefly is a healthy link recorded as unreachable, and this
// runner exists precisely to stop people misreading connection failures as
// fabric faults.
const connectWaitSeconds = 120

// waitForListener polls until the peer accepts a TCP connection.
//
// A dial and an immediate close: it proves a listener is bound without sending
// anything iperf3 would have to interpret. The polling interval is short
// because the wait is usually near-zero — the server is normally already up,
// and this loop is here for the case where it is not.
func waitForListener(host string, port int, within time.Duration) error {
	deadline := time.Now().Add(within)
	addr := net.JoinHostPort(host, strconv.Itoa(port))

	var last error
	for attempt := 1; ; attempt++ {
		conn, err := net.DialTimeout("tcp", addr, 5*time.Second)
		if err == nil {
			conn.Close()
			if attempt > 1 {
				logf("tcp-baseline: listener on %s appeared after %d attempt(s)", addr, attempt)
			}
			return nil
		}
		last = err
		if time.Now().After(deadline) {
			return last
		}
		time.Sleep(time.Second)
	}
}

// iperfResult is the part of iperf3's JSON this runner reads.
type iperfResult struct {
	ThroughputGbps float64
	Retransmits    int
	MeanRttUs      float64
	HasRTT         bool
	Seconds        float64
	Error          string
}

// parseIperf reads iperf3's --json output.
//
// The SENDER's summary is the measurement, because retransmits and RTT are
// properties the sending stack observes and the receiver cannot report. Falling
// back to the receiver for throughput alone would produce a result whose three
// numbers came from two different ends.
func parseIperf(raw string) (iperfResult, error) {
	var doc struct {
		Error string `json:"error"`
		End   struct {
			SumSent struct {
				Bits    float64 `json:"bits_per_second"`
				Retrans *int    `json:"retransmits"`
				Seconds float64 `json:"seconds"`
			} `json:"sum_sent"`
			Streams []struct {
				Sender struct {
					MeanRtt *float64 `json:"mean_rtt"`
				} `json:"sender"`
			} `json:"streams"`
		} `json:"end"`
	}
	if err := json.Unmarshal([]byte(strings.TrimSpace(raw)), &doc); err != nil {
		return iperfResult{}, fmt.Errorf("not iperf3 JSON: %w", err)
	}
	if doc.Error != "" {
		return iperfResult{Error: doc.Error}, nil
	}

	res := iperfResult{
		ThroughputGbps: doc.End.SumSent.Bits / 1e9,
		Seconds:        doc.End.SumSent.Seconds,
	}
	if doc.End.SumSent.Retrans != nil {
		res.Retransmits = *doc.End.SumSent.Retrans
	}
	// mean_rtt is absent on platforms without TCP_INFO. Omitted rather than
	// reported as zero: a zero RTT is a measurement nobody took, and a gate on
	// it would certify a path this runner never timed.
	for _, s := range doc.End.Streams {
		if s.Sender.MeanRtt != nil {
			// iperf3 reports mean_rtt in microseconds already.
			res.MeanRttUs = *s.Sender.MeanRtt
			res.HasRTT = true
			break
		}
	}
	return res, nil
}

func trim(v float64) string { return strconv.FormatFloat(v, 'f', -1, 64) }

func oneLine(s string) string {
	return strings.TrimSpace(strings.ReplaceAll(s, "\n", " "))
}
