// Command burnin-gpudirect-rdma is the entrypoint of the runner image for the
// "gpudirect-rdma" TestKind: RDMA write bandwidth measured STRAIGHT INTO
// ACCELERATOR MEMORY across the link between TWO nodes.
//
// SPDX-License-Identifier: Apache-2.0
// Copyright the Glimmer authors.
//
// This file is original work licensed under Apache-2.0. It executes ib_write_bw
// and ib_write_lat from linux-rdma/perftest, built with --enable-cuda and
// invoked with --use_cuda. perftest is dual-licensed GPL-2.0-only OR
// BSD-2-Clause and is consumed HERE UNDER THE BSD-2-Clause OPTION ONLY. See
// README.md.
//
// # What this measures, and how it differs from ib-write-bw
//
// ib-write-bw registers a buffer in HOST memory: it measures the fabric. This
// runner registers a buffer in DEVICE memory, so the HCA DMAs to and from the
// accelerator without a bounce through host RAM. The difference between the two
// numbers is the thing worth knowing — it is the cost of the bounce, and on a
// node where GPUDirect is misconfigured rather than absent it is large.
//
// # This runner owns its own applicability decision
//
// GPUDirect RDMA requires a peer-memory provider registered with the RDMA
// subsystem. Whether one exists is a fact about the node's drivers, and the only
// component that can look at the node's drivers is the process running on the
// node. So the decision is made here, in applicability.go, and NOT by a vendor
// branch in the reconciler. See that file for why the check keys off the
// provider rather than off the part number.
//
// On NVIDIA GB10 (DGX Spark) there is no provider and cannot be one: CPU and GPU
// share a single memory pool over NVLink-C2C, so this runner exits 2 there. That
// is the expected outcome on that hardware, and it is a Skip, never a Fail.
//
// Output contract (pkg/runner):
//
//	exit 0  GPUDIRECT_RDMA_PASS            device-to-device RDMA was measured
//	exit 1  GPUDIRECT_RDMA_FAIL: <reason>  it was measured and the link is unusable
//	exit 2  GPUDIRECT_RDMA_SKIP: <reason>  this node cannot do GPUDirect RDMA
//	exit 3  GPUDIRECT_RDMA_ERROR: <reason> the runner could not measure it
//
// Metrics, printed BEFORE the decision:
//
//	bw_average=<Gb/s>   -> bandwidthGbps  (pkg/runner alias table)
//	t_avg_usec=<usec>   -> latencyUs      (pkg/runner alias table)
//
// bandwidthGbps is deliberately the SAME canonical name ib-write-bw reports.
// They are the same measurand on the same link, and minting a second name would
// split a link's history in two the day a fleet switched which test it gated on.
// The two are told apart by the TestKind, which the envelope carries.
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

const (
	defaultHealthPort   = 18520
	defaultPerftestPort = 18525
	defaultMessageBytes = 1 << 20
	defaultQPs          = 4

	defaultLatencyIterations = 5000
	defaultDurationSeconds   = 60
	latencyBudgetSeconds     = 20
	minBandwidthSeconds      = 10
)

// perftestRef is stamped by the image build (-ldflags -X).
var perftestRef = "unknown"

func main() { os.Exit(run()) }

func logf(format string, args ...any) {
	fmt.Printf("[burnin-gpudirect-rdma] "+format+"\n", args...)
}

func metric(key, value string) { fmt.Printf("%s=%s\n", key, value) }

func fin(code int, format string, args ...any) int {
	marker := map[int]string{
		exitPass: "GPUDIRECT_RDMA_PASS", exitFail: "GPUDIRECT_RDMA_FAIL",
		exitSkip: "GPUDIRECT_RDMA_SKIP", exitError: "GPUDIRECT_RDMA_ERROR",
	}[code]
	fmt.Printf("%s: %s\n", marker, fmt.Sprintf(format, args...))
	return code
}

func run() int {
	role := strings.TrimSpace(os.Getenv("BURNIN_ROLE"))
	peerHost := strings.TrimSpace(os.Getenv("BURNIN_PEER_HOST"))
	peerNode := strings.TrimSpace(os.Getenv("BURNIN_PEER_NODE"))
	duration := envInt("BURNIN_DURATION_SECONDS", defaultDurationSeconds)

	logf("perftest %s (--enable-cuda); role=%s peer=%s (node %s) duration=%ds",
		perftestRef, role, peerHost, peerNode, duration)

	if role == "" {
		return fin(exitSkip, "BURNIN_ROLE is not set, so this pod is not half of a Pair — "+
			"GPUDirect RDMA is a property of a LINK and cannot be measured from one node; "+
			"run this TestKind at scope: Pair")
	}
	if role != pairRoleServer && role != pairRoleClient {
		return fin(exitError, "BURNIN_ROLE=%q is neither %q nor %q", role, pairRoleServer, pairRoleClient)
	}
	if peerHost == "" {
		return fin(exitError, "BURNIN_ROLE=%s but BURNIN_PEER_HOST is empty — there is no peer to rendezvous with", role)
	}

	// ── THE APPLICABILITY GATE ────────────────────────────────────────────────
	// Checked BEFORE the rendezvous, on purpose. A node that cannot do this test
	// must not hold its peer's node hostage while it waits for a handshake it
	// will refuse anyway, and the operator settles a pair whose server skipped
	// before the client existed as a Skip (see internal/controller/pair.go).
	app := inspect()
	for _, e := range app.Evidence {
		logf("applicability: %s", e)
	}
	if ok, why := app.Applicable(); !ok {
		return fin(exitSkip, "%s", why)
	}
	logf("applicability: GPUDirect RDMA is available here (providers: %s) — proceeding",
		strings.Join(app.Providers, ", "))

	ports, err := discoverPorts()
	if errors.Is(err, errNoRDMA) {
		return fin(exitSkip, "this node has no RDMA device (%s is absent or empty) — "+
			"there is no fabric here to measure", sysInfiniband)
	}
	if err != nil {
		return fin(exitError, "enumerating RDMA devices: %v", err)
	}
	for _, p := range ports {
		logf("found %s", p)
	}

	soft, raised, mlErr := memlock()
	if mlErr != nil {
		return fin(exitError, "%v", mlErr)
	}
	if raised {
		logf("raised the RLIMIT_MEMLOCK soft limit to the hard limit (%s)", humanBytes(soft))
	}
	logf("RLIMIT_MEMLOCK = %s; this runner will register at most %s of RDMA buffers",
		humanBytes(soft), humanBytes(budgetFor(soft)))

	cfg := plan{
		memlock:      soft,
		healthPort:   envInt("BURNIN_HEALTH_PORT", defaultHealthPort),
		perftestPort: envInt("BURNIN_PERFTEST_PORT", defaultPerftestPort),
		messageBytes: envInt("BURNIN_MESSAGE_BYTES", defaultMessageBytes),
		qps:          envInt("BURNIN_QPS", defaultQPs),
		latIters:     envInt("BURNIN_LATENCY_ITERATIONS", defaultLatencyIterations),
		cudaDevice:   envInt("BURNIN_CUDA_DEVICE", 1) - 1,
	}
	cfg.bwSeconds = duration - latencyBudgetSeconds
	if cfg.bwSeconds < minBandwidthSeconds {
		cfg.bwSeconds = minBandwidthSeconds
	}
	cfg.peerWait = time.Duration(duration+latencyBudgetSeconds) * time.Second
	if cfg.peerWait < 2*time.Minute {
		cfg.peerWait = 2 * time.Minute
	}

	if role == pairRoleServer {
		return runServer(ports, cfg)
	}
	return runClient(ports, peerHost, cfg)
}

type plan struct {
	// memlock bounds how much this runner may register; see memlock.go.
	memlock      uint64
	healthPort   int
	perftestPort int
	messageBytes int
	qps          int
	latIters     int
	bwSeconds    int
	cudaDevice   int
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
		return fin(exitFail, "%v — the node has RDMA hardware but no usable link", err)
	}
	logf("using %s; %s", port, why)

	budget := budgetFor(cfg.memlock)
	if peer := ch.peerHello.Memlock; peer > 0 && budgetFor(peer) < budget {
		logf("the %s endpoint reports RLIMIT_MEMLOCK = %s, smaller than this end's %s; planning for the smaller",
			pairRoleClient, humanBytes(peer), humanBytes(cfg.memlock))
		budget = budgetFor(peer)
	}
	fittedBytes, fittedQPs, notes, ok := fitToMemlock(cfg.messageBytes, cfg.qps, budget)
	if !ok {
		reason := fmt.Sprintf("RLIMIT_MEMLOCK is too small to register even a minimal RDMA test "+
			"(%s available to buffers, %s needed at the %d-byte floor) — %s",
			humanBytes(budget), humanBytes(pinnedBytes(minMessageBytes, 1)), minMessageBytes,
			memlockAdvice(cfg.memlock, pinnedBytes(minMessageBytes, 1)))
		_ = ch.send(message{Kind: kindAbort, Reason: reason})
		return fin(exitError, "%s", reason)
	}
	for _, n := range notes {
		logf("REDUCED TO FIT RLIMIT_MEMLOCK: %s", n)
	}
	cfg.messageBytes, cfg.qps = fittedBytes, fittedQPs
	logf("plan: %d-byte messages across %d queue pairs = %s registered (budget %s)",
		cfg.messageBytes, cfg.qps, humanBytes(pinnedBytes(cfg.messageBytes, cfg.qps)), humanBytes(budget))

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
		out, waitErr, startErr := startAndAnnounce(ph.name, ph.args(), ph.msg.Port, ch, ph.msg)
		if startErr != nil {
			_ = ch.send(message{Kind: kindAbort, Reason: startErr.Error()})
			return fin(exitError, "%s phase: %v", ph.name, startErr)
		}
		echo(ph.name, out)
		if waitErr != nil {
			logf("%s phase: the %s-side process exited with %v", ph.name, pairRoleServer, waitErr)
		}
		if m, err := ch.recv(60 * time.Second); err == nil && m.Kind == kindAbort {
			return fin(exitError, "the %s endpoint aborted the %s phase: %s", pairRoleClient, ph.name, m.Reason)
		}
	}

	_ = ch.send(message{Kind: kindEnd})
	// NOT a pass on its own: the operator treats a server that exited cleanly
	// with no client report as an Error, because no traffic crossed the link.
	return fin(exitPass, "the %s endpoint served both phases from device memory over %s to %s; "+
		"the %s endpoint's measurement is the pair's verdict",
		pairRoleServer, port.Device, peerIP, pairRoleClient)
}

func startAndAnnounce(name string, args []string, listenPort int, ch *channel, announce message) (string, error, error) {
	logf("%s phase: starting %s", name, strings.Join(args, " "))
	cmd := exec.Command(args[0], args[1:]...)
	var buf strings.Builder
	cmd.Stdout, cmd.Stderr = &buf, &buf
	if err := cmd.Start(); err != nil {
		return "", nil, fmt.Errorf("starting %s: %w", args[0], err)
	}
	if err := waitForListener(listenPort, 60*time.Second); err != nil {
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
	// Wait BEFORE reading the buffer: it is filled by exec's copier goroutine.
	waitErr := cmd.Wait()
	return buf.String(), waitErr, nil
}

// waitForListener polls /proc/net/tcp for a LISTEN socket on port. The window is
// longer than ib-write-bw's because a CUDA-enabled perftest initialises a CUDA
// context before it binds, and a cold context on a busy accelerator is not fast.
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
	ch, err := dialPeer(peerHost, cfg.healthPort, message{Memlock: cfg.memlock}, cfg.peerWait, logf)
	if err != nil {
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
		default:
			continue
		}

		switch m.Phase {
		case phaseBandwidth:
			c := cfg
			c.perftestPort, c.messageBytes, c.qps, c.bwSeconds = m.Port, m.Bytes, m.QPs, m.Seconds
			out, err := runPerftest(bwArgs(port, c, dest), time.Duration(m.Seconds+180)*time.Second)
			echo(phaseBandwidth, out)
			if err != nil && !strings.Contains(out, "BW average[") {
				_ = ch.send(message{Kind: kindAbort, Reason: err.Error()})
				if looksLikeMemlockExhaustion(out) {
					return fin(exitError, "ib_write_bw --use_cuda could not allocate its RDMA resources against %s over %s: %s. %s",
						dest, port.Device, oneLine(out),
						memlockAdvice(cfg.memlock, pinnedBytes(c.messageBytes, c.qps)))
				}
				return fin(exitError, "ib_write_bw --use_cuda failed against %s over %s: %v (output: %s)",
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
				logf("ib_write_lat --use_cuda did not complete (%v); latencyUs will not be reported", err)
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
	// bw_peak is not emitted: duration mode does not measure a peak and perftest
	// prints 0.00 there. Never emit a 0 you did not measure.
	if haveLat {
		metric("t_avg_usec", strconv.FormatFloat(lat.AvgUs, 'f', 2, 64))
	}

	logf("%d bytes x %d iterations device-to-device over %s to %s", bw.Bytes, bw.Iterations, port.Device, dest)

	if bw.AverageGbps <= 0 {
		return fin(exitFail, "the link completed the test carrying no traffic (bandwidthGbps=%.2f) over %s to %s",
			bw.AverageGbps, port.Device, dest)
	}
	return fin(exitPass, "measured %.2f Gb/s average into device memory over %s to %s with %d QPs at %d-byte messages; "+
		"acceptance is the profile's thresholds to decide",
		bw.AverageGbps, port.Device, dest, cfg.qps, bw.Bytes)
}

// ── perftest invocation ──────────────────────────────────────────────────────

// cudaFlag is what makes this runner different from ib-write-bw: it tells
// perftest to register its buffer in DEVICE memory. perftest numbers CUDA
// devices from 1 in this flag, which is why cudaDevice is stored zero-based and
// incremented here rather than the other way round — an off-by-one would
// silently measure the wrong accelerator on a multi-GPU node.
func cudaFlag(cfg plan) string { return "--use_cuda=" + strconv.Itoa(cfg.cudaDevice+1) }

func bwArgs(p rdmaPort, cfg plan, dest string) []string {
	a := []string{"ib_write_bw",
		"-d", p.Device,
		"-i", strconv.Itoa(p.Port),
		"-p", strconv.Itoa(cfg.perftestPort),
		// --report_gbits is not optional: without it perftest prints MB/sec into
		// the same column and the value lands in bandwidthGbps eight times too
		// small. perftest.go asserts the header for the same reason.
		"--report_gbits",
		"-D", strconv.Itoa(cfg.bwSeconds),
		"-s", strconv.Itoa(cfg.messageBytes),
		"-q", strconv.Itoa(cfg.qps),
		"-F",
		cudaFlag(cfg),
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
		cudaFlag(cfg),
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
