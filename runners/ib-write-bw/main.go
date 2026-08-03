// Command burnin-ib-write-bw is the entrypoint of the runner image for the
// "ib-write-bw" TestKind: it measures RDMA write bandwidth and latency across
// the link between TWO nodes.
//
// SPDX-License-Identifier: Apache-2.0
// Copyright the Glimmer authors.
//
// This file is original work licensed under Apache-2.0. It executes ib_write_bw
// and ib_write_lat from linux-rdma/perftest, which is dual-licensed
// GPL-2.0-only OR BSD-2-Clause and is built into this image and consumed HERE
// UNDER THE BSD-2-Clause OPTION ONLY. See README.md for the full text this
// project relies on and for what that choice obliges.
//
// # Scope
//
// This is a Pair-scope runner and only a Pair-scope runner. A point-to-point
// bandwidth figure is a property of a LINK, and there is no such thing as the
// RDMA write bandwidth of a single node: run at Node scope it exits 2, because
// "there is no second endpoint" is a statement about the test's applicability,
// not a hardware failure.
//
// Output contract (pkg/runner):
//
//	exit 0  IB_WRITE_BW_PASS            the link was measured; thresholds decide acceptance
//	exit 1  IB_WRITE_BW_FAIL: <reason>  the link was measured and is not usable
//	exit 2  IB_WRITE_BW_SKIP: <reason>  the test does not apply to this node
//	exit 3  IB_WRITE_BW_ERROR: <reason> the runner could not measure the link
//
// Metrics are key=value lines on stdout and are ALWAYS printed BEFORE the
// pass/fail decision, so a verdict is never reported without the numbers it was
// reached from.
//
//	bw_average=<Gb/s>   -> bandwidthGbps  (pkg/runner alias table)
//	latency_us=<usec>   -> latencyUs      (generic snake_case normalisation)
//
// # Why the runner itself almost never returns 1
//
// The acceptance number belongs in the BurnInTest's thresholds, not in this
// binary: a 200G fabric and a 100G fabric are both healthy, and a runner that
// hard-coded a floor would condemn one of them. So a completed measurement exits
// 0 and lets pkg/verdict decide. Exit 1 is reserved for the two states that are
// hardware facts on their own terms — no ACTIVE port on any RDMA device, and a
// link that completed the test while carrying no traffic at all.
package main

import (
	"context"
	"errors"
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
)

const (
	pairRoleServer = "server"
	pairRoleClient = "client"
)

// Defaults, all overridable by environment so a fabric with different framing
// can be measured without a new image.
const (
	// defaultHealthPort is the runner's own control/readiness listener. It is
	// NOT perftest's port; see rendezvous.go for why probing perftest's port
	// would break the test.
	defaultHealthPort = 18510
	// defaultPerftestPort is perftest's out-of-band port for the bandwidth
	// phase; the latency phase uses the next one up.
	defaultPerftestPort = 18515
	// defaultMessageBytes is 1 MiB: large enough that the measurement is a
	// bandwidth figure rather than a message-rate one.
	defaultMessageBytes = 1 << 20
	// defaultQPs is 4. One queue pair does not always saturate a modern HCA;
	// four does, and on the DGX Spark pair this was developed against 4 and 8
	// QPs produce the same number, which is how the plateau was confirmed to be
	// a real ceiling rather than a queue-depth artefact.
	defaultQPs = 4
	// defaultLatencyIterations is the ib_write_lat iteration count. Latency is
	// measured in iteration mode, not duration mode, so the figure is over a
	// fixed sample rather than over however many fitted in the clock.
	defaultLatencyIterations = 5000
	defaultDurationSeconds   = 60
	// latencyBudgetSeconds is carved out of the run's duration for the latency
	// phase and the two rendezvous handshakes, so the whole runner finishes
	// inside the pod's activeDeadlineSeconds instead of being killed at the end
	// and reported as an Error.
	latencyBudgetSeconds = 20
	minBandwidthSeconds  = 10
)

// perftestRef is stamped by the image build (-ldflags -X) with the perftest
// release the image was built from. A fabric number is only comparable across a
// fleet if the tool that produced it is the same, so the version is logged with
// every run rather than left to the image tag.
var perftestRef = "unknown"

func main() { os.Exit(run()) }

func logf(format string, args ...any) {
	fmt.Printf("[burnin-ib-write-bw] "+format+"\n", args...)
}

// metric prints one key=value line. Every metric this runner emits goes through
// here so that "metrics before the decision" is a property of the code rather
// than of the author's memory.
func metric(key, value string) { fmt.Printf("%s=%s\n", key, value) }

func fin(code int, format string, args ...any) int {
	marker := map[int]string{
		exitPass: "IB_WRITE_BW_PASS", exitFail: "IB_WRITE_BW_FAIL",
		exitSkip: "IB_WRITE_BW_SKIP", exitError: "IB_WRITE_BW_ERROR",
	}[code]
	fmt.Printf("%s: %s\n", marker, fmt.Sprintf(format, args...))
	return code
}

func run() int {
	role := strings.TrimSpace(os.Getenv("BURNIN_ROLE"))
	peerHost := strings.TrimSpace(os.Getenv("BURNIN_PEER_HOST"))
	peerNode := strings.TrimSpace(os.Getenv("BURNIN_PEER_NODE"))
	duration := envInt("BURNIN_DURATION_SECONDS", defaultDurationSeconds)

	// Node scope: BURNIN_ROLE is only injected for a Pair. A link test with one
	// endpoint is not a failing test, it is an inapplicable one.
	if role == "" {
		return fin(exitSkip, "BURNIN_ROLE is not set, so this pod is not half of a Pair — "+
			"RDMA write bandwidth is a property of a LINK and cannot be measured from one node; "+
			"run this TestKind at scope: Pair")
	}
	if role != pairRoleServer && role != pairRoleClient {
		return fin(exitError, "BURNIN_ROLE=%q is neither %q nor %q", role, pairRoleServer, pairRoleClient)
	}
	if peerHost == "" {
		return fin(exitError, "BURNIN_ROLE=%s but BURNIN_PEER_HOST is empty — there is no peer to rendezvous with", role)
	}

	ports, err := discoverPorts()
	if errors.Is(err, errNoRDMA) {
		return fin(exitSkip, "this node has no RDMA device (%s is absent or empty) — "+
			"there is no fabric here to measure", sysInfiniband)
	}
	if err != nil {
		return fin(exitError, "enumerating RDMA devices: %v", err)
	}
	logf("perftest %s; role=%s peer=%s (node %s) duration=%ds", perftestRef, role, peerHost, peerNode, duration)
	for _, p := range ports {
		logf("found %s", p)
	}

	cfg := plan{
		healthPort:   envInt("BURNIN_HEALTH_PORT", defaultHealthPort),
		perftestPort: envInt("BURNIN_PERFTEST_PORT", defaultPerftestPort),
		messageBytes: envInt("BURNIN_MESSAGE_BYTES", defaultMessageBytes),
		qps:          envInt("BURNIN_QPS", defaultQPs),
		latIters:     envInt("BURNIN_LATENCY_ITERATIONS", defaultLatencyIterations),
	}
	cfg.bwSeconds = duration - latencyBudgetSeconds
	if cfg.bwSeconds < minBandwidthSeconds {
		cfg.bwSeconds = minBandwidthSeconds
	}
	// The whole rendezvous must fit inside the pod's own deadline, so the wait
	// for a peer is bounded by the run's duration rather than by a constant.
	cfg.peerWait = time.Duration(duration+latencyBudgetSeconds) * time.Second
	if cfg.peerWait < 2*time.Minute {
		cfg.peerWait = 2 * time.Minute
	}

	if role == pairRoleServer {
		return runServer(ports, cfg)
	}
	return runClient(ports, peerHost, cfg)
}

// plan is everything both ends must agree on. The server owns it and sends it
// over the control channel; the client never invents a value, which is what
// makes a parameter mismatch — perftest's most confusing failure mode —
// unreachable.
type plan struct {
	healthPort   int
	perftestPort int
	messageBytes int
	qps          int
	latIters     int
	bwSeconds    int
	peerWait     time.Duration
}

func (p plan) latPort() int { return p.perftestPort + 1 }

// ── server ───────────────────────────────────────────────────────────────────

func runServer(ports []rdmaPort, cfg plan) int {
	ln, err := net.Listen("tcp", fmt.Sprintf(":%d", cfg.healthPort))
	if err != nil {
		return fin(exitError, "binding the rendezvous/readiness port %d: %v", cfg.healthPort, err)
	}
	defer ln.Close()
	// From this moment a tcpSocket readinessProbe on healthPort succeeds, and it
	// means what it says: this end is listening and will accept its peer.
	logf("listening on :%d for the %s endpoint (this is the readinessProbe port)", cfg.healthPort, pairRoleClient)

	ch, err := acceptPeer(ln, cfg.peerWait, logf)
	if err != nil {
		return fin(exitError, "%v — the link was never exercised", err)
	}
	defer ch.Close()

	peerIP := ch.peerIP()
	logf("%s endpoint connected from %s", pairRoleClient, peerIP)

	port, why, err := selectPort(ports, peerIP)
	if err != nil {
		_ = ch.send(message{Kind: kindAbort, Reason: err.Error()})
		// Hardware present, no link up. That is a fact about the fabric.
		return fin(exitFail, "%v — the node has RDMA hardware but no usable link", err)
	}
	logf("using %s; %s", port, why)

	for _, ph := range []struct {
		name string
		msg  message
		args func() []string
	}{
		{phaseBandwidth, message{Kind: kindPhase, Phase: phaseBandwidth, Port: cfg.perftestPort,
			Bytes: cfg.messageBytes, QPs: cfg.qps, Seconds: cfg.bwSeconds},
			func() []string { return bwArgs(port, cfg, "") }},
		{phaseLatency, message{Kind: kindPhase, Phase: phaseLatency, Port: cfg.latPort(), Iters: cfg.latIters},
			func() []string { return latArgs(port, cfg, "") }},
	} {
		listenPort := ph.msg.Port
		out, waitErr, startErr := startAndAnnounce(ph.name, ph.args(), listenPort, ch, ph.msg)
		if startErr != nil {
			_ = ch.send(message{Kind: kindAbort, Reason: startErr.Error()})
			return fin(exitError, "%s phase: %v", ph.name, startErr)
		}
		echo(ph.name, out)
		if waitErr != nil {
			// The client is the deciding side; a server-side non-zero exit is
			// still worth reporting, but it is machinery, not a verdict.
			logf("%s phase: the %s-side process exited with %v", ph.name, pairRoleServer, waitErr)
		}
		// Best effort: the client says how it went. Its verdict is the pair's,
		// so nothing here depends on the answer arriving.
		if m, err := ch.recv(60 * time.Second); err == nil && m.Kind == kindAbort {
			return fin(exitError, "the %s endpoint aborted the %s phase: %s", pairRoleClient, ph.name, m.Reason)
		}
	}

	_ = ch.send(message{Kind: kindEnd})
	// NOTE: this exit 0 is NOT a pass on its own. The operator treats a server
	// that exited cleanly with no client report as an Error, because no traffic
	// crossed the link. See internal/controller/pair.go.
	return fin(exitPass, "the %s endpoint served both phases over %s to %s; "+
		"the %s endpoint's measurement is the pair's verdict",
		pairRoleServer, port.Device, peerIP, pairRoleClient)
}

// startAndAnnounce launches a perftest server, waits for it to be LISTENING,
// and only then tells the client to connect.
//
// The wait is the point. perftest binds its out-of-band socket a moment after
// exec, and a client told to connect before that dies with a connection error
// that is indistinguishable, in a report, from a broken fabric. /proc/net/tcp is
// the cheapest true answer to "is it listening?" and needs no extra tool in the
// image.
func startAndAnnounce(name string, args []string, listenPort int, ch *channel, announce message) (string, error, error) {
	logf("%s phase: starting %s", name, strings.Join(args, " "))
	cmd := exec.Command(args[0], args[1:]...)
	var buf strings.Builder
	cmd.Stdout, cmd.Stderr = &buf, &buf
	if err := cmd.Start(); err != nil {
		return "", nil, fmt.Errorf("starting %s: %w", args[0], err)
	}

	if err := waitForListener(listenPort, 30*time.Second); err != nil {
		_ = cmd.Process.Kill()
		_, _ = cmd.Process.Wait()
		return buf.String(), nil, fmt.Errorf("%s did not begin listening on %d: %w (output: %s)",
			args[0], listenPort, err, oneLine(buf.String()))
	}
	logf("%s phase: listening on %d, releasing the %s endpoint", name, listenPort, pairRoleClient)
	if err := ch.send(announce); err != nil {
		_ = cmd.Process.Kill()
		_, _ = cmd.Process.Wait()
		return buf.String(), nil, fmt.Errorf("announcing the %s phase to the %s endpoint: %w", name, pairRoleClient, err)
	}
	// cmd.Wait() must complete BEFORE the output is read: the buffer is filled by
	// exec's copier goroutine, and reading it first would capture a header and
	// no result table.
	waitErr := cmd.Wait()
	return buf.String(), waitErr, nil
}

// waitForListener polls /proc/net/tcp for a socket in LISTEN state on port.
func waitForListener(port int, within time.Duration) error {
	deadline := time.Now().Add(within)
	for {
		if tcpListening(port) {
			return nil
		}
		if time.Now().After(deadline) {
			return fmt.Errorf("no LISTEN socket on port %d after %s", port, within)
		}
		time.Sleep(100 * time.Millisecond)
	}
}

// tcpListening reads /proc/net/tcp{,6} and reports whether anything is
// listening on port. The local address column is "<hex ip>:<hex port>" and
// state 0A is TCP_LISTEN.
func tcpListening(port int) bool {
	want := fmt.Sprintf(":%04X", port)
	for _, path := range []string{"/proc/net/tcp", "/proc/net/tcp6"} {
		b, err := os.ReadFile(path)
		if err != nil {
			continue
		}
		for _, line := range strings.Split(string(b), "\n") {
			f := strings.Fields(line)
			if len(f) < 4 || f[3] != "0A" {
				continue
			}
			if strings.HasSuffix(f[1], want) {
				return true
			}
		}
	}
	return false
}

// ── client ───────────────────────────────────────────────────────────────────

func runClient(ports []rdmaPort, peerHost string, cfg plan) int {
	ch, err := dialPeer(peerHost, cfg.healthPort, cfg.peerWait, logf)
	if err != nil {
		// The peer is machinery, not fabric: a server that never came up says
		// nothing about the link. Error, not Fail — see pair.go's precedence.
		return fin(exitError, "%v — the link was never exercised", err)
	}
	defer ch.Close()

	peerIP := ch.peerIP()
	port, why, err := selectPort(ports, peerIP)
	if err != nil {
		_ = ch.send(message{Kind: kindAbort, Reason: err.Error()})
		return fin(exitFail, "%v — the node has RDMA hardware but no usable link", err)
	}
	logf("using %s; %s", port, why)
	dest := peerIP.String()

	var bw bwResult
	var lat latResult
	var haveLat bool

phases:
	for {
		m, err := ch.recv(cfg.peerWait)
		if err != nil {
			return fin(exitError, "control channel to the %s endpoint failed: %v", pairRoleServer, err)
		}
		switch m.Kind {
		case kindEnd:
			break phases
		case kindAbort:
			return fin(exitError, "the %s endpoint aborted: %s", pairRoleServer, m.Reason)
		case kindPhase:
			// nothing
		default:
			continue
		}

		switch m.Phase {
		case phaseBandwidth:
			c := cfg
			c.perftestPort, c.messageBytes, c.qps, c.bwSeconds = m.Port, m.Bytes, m.QPs, m.Seconds
			out, err := runPerftest(bwArgs(port, c, dest), time.Duration(m.Seconds+120)*time.Second)
			echo(phaseBandwidth, out)
			if err != nil && !strings.Contains(out, "BW average[") {
				_ = ch.send(message{Kind: kindAbort, Reason: err.Error()})
				return fin(exitError, "ib_write_bw failed against %s over %s: %v (output: %s)",
					dest, port.Device, err, oneLine(out))
			}
			bw, err = parseBandwidth(out)
			if err != nil {
				_ = ch.send(message{Kind: kindAbort, Reason: err.Error()})
				return fin(exitError, "reading ib_write_bw output: %v", err)
			}
			_ = ch.send(message{Kind: kindDone, Phase: phaseBandwidth})

		case phaseLatency:
			c := cfg
			c.perftestPort, c.latIters = m.Port-1, m.Iters
			out, err := runPerftest(latArgs(port, c, dest), 5*time.Minute)
			echo(phaseLatency, out)
			if err != nil && !strings.Contains(out, "t_avg[usec]") {
				// Latency is secondary evidence; losing it must not throw away a
				// good bandwidth measurement, so this is logged, not fatal.
				logf("ib_write_lat did not complete (%v); latencyUs will not be reported", err)
				_ = ch.send(message{Kind: kindDone, Phase: phaseLatency})
				continue
			}
			lat, err = parseLatency(out)
			if err != nil {
				logf("reading ib_write_lat output: %v; latencyUs will not be reported", err)
			} else {
				haveLat = true
			}
			_ = ch.send(message{Kind: kindDone, Phase: phaseLatency})
		}
	}

	// ── metrics, before the decision ──────────────────────────────────────────
	metric("bw_average", strconv.FormatFloat(bw.AverageGbps, 'f', 2, 64))
	// bw_peak is deliberately NOT emitted. perftest measures a peak only in
	// ITERATION mode; in duration mode (-D), which is the mode a burn-in runs
	// in, it prints 0.00 and warns that the peak was not measured. Emitting that
	// zero would put a number nobody measured into peakBandwidthGbps, and
	// "never emit a 0 you did not measure" is the rule. Nor is it "n/a": the
	// sentinel declares that the HARDWARE cannot produce the measurement, and
	// this is a property of the mode this runner chose. A threshold on
	// peakBandwidthGbps therefore fails closed, which is correct — see README.
	if haveLat {
		metric("latency_us", strconv.FormatFloat(lat.AvgUs, 'f', 2, 64))
	}

	logf("%d bytes x %d iterations over %s to %s", bw.Bytes, bw.Iterations, port.Device, dest)
	if haveLat {
		logf("latency min/typical/avg/max = %.2f/%.2f/%.2f/%.2f us over %d iterations",
			lat.MinUs, lat.TypicalUs, lat.AvgUs, lat.MaxUs, lat.Iterations)
	}

	if bw.AverageGbps <= 0 {
		return fin(exitFail, "the link completed the test carrying no traffic (bandwidthGbps=%.2f) over %s to %s",
			bw.AverageGbps, port.Device, dest)
	}
	return fin(exitPass, "measured %.2f Gb/s average over %s to %s with %d QPs at %d-byte messages; "+
		"acceptance is the profile's thresholds to decide",
		bw.AverageGbps, port.Device, dest, cfg.qps, bw.Bytes)
}

// ── perftest invocation ──────────────────────────────────────────────────────

// bwArgs builds an ib_write_bw command line. dest empty = server side.
//
// --report_gbits is not optional and is not a preference: without it perftest
// prints MB/sec into the same column, and the value lands in a metric named
// bandwidthGbps eight times too small. perftest.go asserts the header for the
// same reason, so the flag going missing is caught even if this function is
// edited.
func bwArgs(p rdmaPort, cfg plan, dest string) []string {
	a := []string{"ib_write_bw",
		"-d", p.Device,
		"-i", strconv.Itoa(p.Port),
		"-p", strconv.Itoa(cfg.perftestPort),
		"--report_gbits",
		"-D", strconv.Itoa(cfg.bwSeconds),
		"-s", strconv.Itoa(cfg.messageBytes),
		"-q", strconv.Itoa(cfg.qps),
		// -F: do not abort because cpufreq governors are not pinned. A burn-in
		// node is a production node; refusing to measure it is worse than a
		// slightly noisier number.
		"-F",
	}
	if p.GIDIndex >= 0 {
		a = append(a, "-x", strconv.Itoa(p.GIDIndex))
	}
	if dest != "" {
		a = append(a, dest)
	}
	return a
}

func latArgs(p rdmaPort, cfg plan, dest string) []string {
	a := []string{"ib_write_lat",
		"-d", p.Device,
		"-i", strconv.Itoa(p.Port),
		"-p", strconv.Itoa(cfg.latPort()),
		"-n", strconv.Itoa(cfg.latIters),
		"-s", "2",
		"-F",
	}
	if p.GIDIndex >= 0 {
		a = append(a, "-x", strconv.Itoa(p.GIDIndex))
	}
	if dest != "" {
		a = append(a, dest)
	}
	return a
}

func runPerftest(args []string, timeout time.Duration) (string, error) {
	logf("running %s", strings.Join(args, " "))
	ctx, cancel := context.WithTimeout(context.Background(), timeout)
	defer cancel()
	cmd := exec.CommandContext(ctx, args[0], args[1:]...)
	var buf strings.Builder
	cmd.Stdout, cmd.Stderr = &buf, &buf
	err := cmd.Run()
	if ctx.Err() != nil {
		return buf.String(), fmt.Errorf("%s did not finish within %s", args[0], timeout)
	}
	return buf.String(), err
}

// echo copies perftest's own output into the log, prefixed so that no line of it
// can be mistaken for a key=value metric by pkg/runner's parser.
func echo(phase, out string) {
	for _, line := range strings.Split(strings.TrimRight(out, "\n"), "\n") {
		if strings.TrimSpace(line) == "" {
			continue
		}
		fmt.Printf("| %s | %s\n", phase, line)
	}
}

func oneLine(s string) string {
	s = strings.Join(strings.Fields(s), " ")
	if len(s) > 400 {
		return s[:400] + "…"
	}
	return s
}

func envInt(name string, def int) int {
	v := strings.TrimSpace(os.Getenv(name))
	if v == "" {
		return def
	}
	n, err := strconv.Atoi(v)
	if err != nil || n <= 0 {
		logf("ignoring %s=%q: not a positive integer", name, v)
		return def
	}
	return n
}
