// Command burnin-nccl is the entrypoint of the runner image for the "nccl"
// TestKind: a two-rank NCCL all-reduce across the link between TWO nodes.
//
// SPDX-License-Identifier: Apache-2.0
// Copyright the Glimmer authors.
//
// This file is original work licensed under Apache-2.0. It drives nccl_pair,
// this repository's own two-rank harness (see nccl_pair.cu for why nccl-tests is
// not used directly), which links NVIDIA NCCL — BSD-3-Clause. See README.md.
//
// # What this measures
//
// ib-write-bw measures the wire. This measures what a COLLECTIVE achieves over
// that wire: NCCL's own transport selection, its protocol and algorithm choices,
// and the copies it needs on a node with no GPUDirect RDMA. The two numbers
// answer different questions and a fleet wants both — a link can be perfect and
// the collective still slow because NCCL fell back to a socket transport.
//
// Output contract (pkg/runner):
//
//	exit 0  NCCL_PASS            the collective ran and its results were correct
//	exit 1  NCCL_FAIL: <reason>  the collective ran and returned wrong data
//	exit 2  NCCL_SKIP: <reason>  the test does not apply to this node
//	exit 3  NCCL_ERROR: <reason> the runner could not measure the collective
//
// Metrics, printed BEFORE the decision (see results.go for why exactly one value
// is emitted per name):
//
//	busbw=<GB/s>       -> busBandwidthGBs  (pkg/runner alias table)
//	algbw=<GB/s>       -> algBandwidthGBs  (pkg/runner alias table)
//	latency_us=<usec>  -> latencyUs        (generic normalisation)
//	wrong_count=<n>    -> miscompares      (pkg/runner alias table)
//
// # Why this runner ships NO default threshold, and why the obvious one is wrong
//
// A 2-rank all-reduce cannot exceed the link. On the DGX Spark pair this was
// developed against, ib_write_bw plateaus at 99.6 Gb/s because the ConnectX-7
// attaches at PCIe Gen5 x4 — the wire is 200G and the host can only feed half of
// it. 99.6 Gb/s is 12.45 GB/s, so a threshold of `busBandwidthGBs >= 20` is not
// merely strict, it is ARITHMETICALLY UNREACHABLE on this hardware and would
// condemn every node in the fleet for a property of the part. Ship the
// measurement, decide the gate from the measurement.
package main

import (
	"context"
	"errors"
	"fmt"
	"math"
	"net"
	"os"
	"os/exec"
	"path/filepath"
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
	// defaultHealthPort is the wrapper's control/readiness listener. nccl_pair's
	// bootstrap port and NCCL's own ports are all separate from it; see
	// rendezvous.go for why the readiness port must be one nothing else uses.
	defaultHealthPort = 18530
	// defaultBootstrapPort is where rank 0 publishes the ncclUniqueId.
	defaultBootstrapPort   = 18535
	defaultWarmupIters     = 5
	defaultTimedIters      = 20
	defaultDurationSeconds = 60
	// minGPUsForNodeScope is why a Pair exists for this kind. An all-reduce
	// needs at least two ranks; at Node scope those ranks must be two GPUs in
	// one box.
	minGPUsForNodeScope = 2
)

// nvidiaGPUDir is where the NVIDIA driver publishes one directory per bound GPU.
const nvidiaGPUDir = "/proc/driver/nvidia/gpus"

// ncclVersion is stamped by the image build (-ldflags -X). A collective's
// bandwidth is only comparable across a fleet if the library that produced it is
// the same, so it is logged with every run rather than left to the image tag.
var ncclVersion = "unknown"

func main() { os.Exit(run()) }

func logf(format string, args ...any) {
	fmt.Printf("[burnin-nccl] "+format+"\n", args...)
}

func metric(key, value string) { fmt.Printf("%s=%s\n", key, value) }

func fin(code int, format string, args ...any) int {
	marker := map[int]string{
		exitPass: "NCCL_PASS", exitFail: "NCCL_FAIL",
		exitSkip: "NCCL_SKIP", exitError: "NCCL_ERROR",
	}[code]
	fmt.Printf("%s: %s\n", marker, fmt.Sprintf(format, args...))
	return code
}

func run() int {
	role := strings.TrimSpace(os.Getenv("BURNIN_ROLE"))
	peerHost := strings.TrimSpace(os.Getenv("BURNIN_PEER_HOST"))
	peerNode := strings.TrimSpace(os.Getenv("BURNIN_PEER_NODE"))
	duration := envInt("BURNIN_DURATION_SECONDS", defaultDurationSeconds)

	gpus := countGPUs()
	logf("NCCL %s; role=%s peer=%s (node %s) duration=%ds gpus=%d",
		ncclVersion, role, peerHost, peerNode, duration, gpus)

	if gpus == 0 {
		return fin(exitSkip, "no NVIDIA accelerator is visible to this pod "+
			"(%s is absent or empty) — a collective needs one", nvidiaGPUDir)
	}

	// ── Node scope ────────────────────────────────────────────────────────────
	// An all-reduce needs at least two ranks. At Node scope both ranks would
	// have to be GPUs in the same box, where the test is a free NVLink check and
	// well worth running. On a ONE-GPU part — DGX Spark among them — there is no
	// second rank, so the test is inapplicable rather than failed.
	if role == "" {
		if gpus < minGPUsForNodeScope {
			return fin(exitSkip, "this node has %d accelerator(s) and an all-reduce needs at least %d ranks; "+
				"BURNIN_ROLE is unset, so this is a Node-scope run and there is no second node to be the other rank. "+
				"Run this TestKind at scope: Pair to measure the collective across the fabric",
				gpus, minGPUsForNodeScope)
		}
		return fin(exitSkip, "this node has %d accelerators, but intra-node (Node scope) collectives are not "+
			"implemented by this runner yet — it is a Pair-scope fabric test. Run at scope: Pair", gpus)
	}
	if role != pairRoleServer && role != pairRoleClient {
		return fin(exitError, "BURNIN_ROLE=%q is neither %q nor %q", role, pairRoleServer, pairRoleClient)
	}
	if peerHost == "" {
		return fin(exitError, "BURNIN_ROLE=%s but BURNIN_PEER_HOST is empty — there is no peer to rendezvous with", role)
	}

	cfg := plan{
		healthPort:    envInt("BURNIN_HEALTH_PORT", defaultHealthPort),
		bootstrapPort: envInt("BURNIN_BOOTSTRAP_PORT", defaultBootstrapPort),
		warmup:        envInt("BURNIN_NCCL_WARMUP_ITERS", defaultWarmupIters),
		iters:         envInt("BURNIN_NCCL_ITERS", defaultTimedIters),
		sizes:         strings.TrimSpace(os.Getenv("BURNIN_NCCL_SIZES")),
	}
	cfg.peerWait = time.Duration(duration) * time.Second
	if cfg.peerWait < 2*time.Minute {
		cfg.peerWait = 2 * time.Minute
	}

	// RDMA discovery is advisory here, not a gate: NCCL can fall back to a
	// socket transport, and a collective over sockets is a slow result rather
	// than an inapplicable test. Discovery exists to PIN the fabric NIC, because
	// left to itself NCCL may pick the wireless interface or a CNI overlay.
	ports, err := discoverPorts()
	if err != nil && !errors.Is(err, errNoRDMA) {
		logf("could not enumerate RDMA devices (%v); NCCL will choose its own transport", err)
	}
	for _, p := range ports {
		logf("found %s", p)
	}

	if role == pairRoleServer {
		return runServer(ports, cfg)
	}
	return runClient(ports, peerHost, cfg)
}

type plan struct {
	healthPort    int
	bootstrapPort int
	warmup        int
	iters         int
	sizes         string
	peerWait      time.Duration
}

// countGPUs asks the driver, not CUDA. This runs before any CUDA context exists
// and must work in a pod whose accelerator is busy.
func countGPUs() int {
	if entries, err := os.ReadDir(nvidiaGPUDir); err == nil {
		return len(entries)
	}
	devs, _ := filepath.Glob("/dev/nvidia[0-9]*")
	return len(devs)
}

// ── server (rank 0) ──────────────────────────────────────────────────────────

func runServer(ports []rdmaPort, cfg plan) int {
	ln, err := net.Listen("tcp", fmt.Sprintf(":%d", cfg.healthPort))
	if err != nil {
		return fin(exitError, "binding the rendezvous/readiness port %d: %v", cfg.healthPort, err)
	}
	defer ln.Close()
	logf("listening on :%d for the %s endpoint (this is the readinessProbe port)", cfg.healthPort, pairRoleClient)

	ch, err := acceptPeer(ln, cfg.peerWait, logf)
	if err != nil {
		return fin(exitError, "%v — the collective was never run", err)
	}
	defer ch.Close()

	peerIP := ch.peerIP()
	logf("%s endpoint connected from %s", pairRoleClient, peerIP)

	env := ncclEnv(ports, peerIP)
	args := harnessArgs(0, "", cfg)

	logf("starting %s", strings.Join(args, " "))
	cmd := exec.Command(args[0], args[1:]...)
	cmd.Env = append(os.Environ(), env...)
	var buf strings.Builder
	cmd.Stdout, cmd.Stderr = &buf, &buf
	if err := cmd.Start(); err != nil {
		_ = ch.send(message{Kind: kindAbort, Reason: err.Error()})
		return fin(exitError, "starting the rank-0 harness: %v", err)
	}

	// Release rank 1 only once rank 0's bootstrap socket EXISTS. A rank 1 that
	// connects first would retry (nccl_pair.cu does), but the operator's own
	// readiness gate is only as good as what it gates on, and "the process
	// started" is not the same claim as "the port is bound".
	if err := waitForListener(cfg.bootstrapPort, 90*time.Second); err != nil {
		_ = cmd.Process.Kill()
		_, _ = cmd.Process.Wait()
		echo(buf.String())
		_ = ch.send(message{Kind: kindAbort, Reason: err.Error()})
		return fin(exitError, "the rank-0 harness never published its bootstrap handle: %v", err)
	}
	logf("rank 0 bootstrap handle is published on %d, releasing the %s endpoint", cfg.bootstrapPort, pairRoleClient)
	if err := ch.send(message{Kind: kindPhase, Phase: phaseCollective, Port: cfg.bootstrapPort}); err != nil {
		_ = cmd.Process.Kill()
		_, _ = cmd.Process.Wait()
		return fin(exitError, "releasing the %s endpoint: %v", pairRoleClient, err)
	}

	waitErr := cmd.Wait()
	echo(buf.String())
	if waitErr != nil {
		logf("the rank-0 harness exited with %v", waitErr)
	}
	if m, err := ch.recv(2 * time.Minute); err == nil && m.Kind == kindAbort {
		return fin(exitError, "the %s endpoint aborted: %s", pairRoleClient, m.Reason)
	}
	_ = ch.send(message{Kind: kindEnd})

	// NOT a pass on its own: the operator treats a server that exited cleanly
	// with no client report as an Error, because no collective completed.
	return fin(exitPass, "rank 0 completed the collective with %s; "+
		"the %s endpoint's measurement is the pair's verdict", peerIP, pairRoleClient)
}

// ── client (rank 1) ──────────────────────────────────────────────────────────

func runClient(ports []rdmaPort, peerHost string, cfg plan) int {
	ch, err := dialPeer(peerHost, cfg.healthPort, cfg.peerWait, logf)
	if err != nil {
		return fin(exitError, "%v — the collective was never run", err)
	}
	defer ch.Close()

	peerIP := ch.peerIP()
	m, err := ch.recv(cfg.peerWait)
	if err != nil {
		return fin(exitError, "control channel to the %s endpoint failed: %v", pairRoleServer, err)
	}
	switch m.Kind {
	case kindAbort:
		return fin(exitError, "the %s endpoint aborted: %s", pairRoleServer, m.Reason)
	case kindPhase:
		if m.Port > 0 {
			cfg.bootstrapPort = m.Port
		}
	default:
		return fin(exitError, "the %s endpoint sent %q instead of a phase", pairRoleServer, m.Kind)
	}

	env := ncclEnv(ports, peerIP)
	args := harnessArgs(1, peerIP.String(), cfg)

	out, runErr := runHarness(args, env, 20*time.Minute)
	echo(out)
	if runErr != nil && !strings.Contains(out, "RESULT ") {
		_ = ch.send(message{Kind: kindAbort, Reason: runErr.Error()})
		return fin(exitError, "the rank-1 harness failed against %s: %v", peerIP, runErr)
	}
	s, err := parseSweep(out)
	if err != nil {
		_ = ch.send(message{Kind: kindAbort, Reason: err.Error()})
		return fin(exitError, "reading the harness output: %v", err)
	}
	_ = ch.send(message{Kind: kindDone, Phase: phaseCollective})

	peak, okPeak := s.Peak()
	small, okSmall := s.Smallest()
	if !okPeak {
		return fin(exitError, "the harness produced no usable measurement")
	}

	// The whole sweep goes to the log as evidence; only one value per metric is
	// emitted. See results.go.
	for _, x := range s.Samples {
		logf("  %10s  %9.1f us  alg %7.2f GB/s  bus %7.2f GB/s", humanBytes(x.Bytes), x.TimeUs, x.AlgGBs, x.BusGBs)
	}
	// Re-derive the harness's own arithmetic from the other side. If these ever
	// disagree, one of the two files changed its formula or its units and the
	// fleet's stored bandwidth silently rescaled; better to say so in the log
	// than to let the number drift.
	if want := peak.AlgGBs * busBandwidthFactor(2); math.Abs(want-peak.BusGBs) > 0.01*math.Max(1, want) {
		logf("WARNING: the harness reported busBandwidth %.4f GB/s where alg %.4f x %.2f = %.4f — "+
			"nccl_pair.cu and results.go disagree about the all-reduce scaling factor",
			peak.BusGBs, peak.AlgGBs, busBandwidthFactor(2), want)
	}

	// ── metrics, before the decision ──────────────────────────────────────────
	metric("busbw", strconv.FormatFloat(peak.BusGBs, 'f', 2, 64))
	metric("algbw", strconv.FormatFloat(peak.AlgGBs, 'f', 2, 64))
	if okSmall {
		metric("latency_us", strconv.FormatFloat(small.TimeUs, 'f', 2, 64))
	}
	if s.WrongReported {
		metric("wrong_count", strconv.FormatInt(s.Wrong, 10))
	}
	// A miscompare count that was never produced is NOT emitted as 0: the run
	// did not establish it, and "never emit a 0 you did not measure" is the rule.
	// It is not "n/a" either — that sentinel is for hardware which cannot produce
	// a measurement, and this hardware plainly can.

	if s.WrongReported && s.Wrong > 0 {
		return fin(exitFail, "the all-reduce returned %d incorrect element(s) across the sweep — "+
			"the collective completed but the data is wrong", s.Wrong)
	}
	if peak.BusGBs <= 0 {
		return fin(exitFail, "the collective completed carrying no traffic (busBandwidthGBs=%.2f)", peak.BusGBs)
	}
	return fin(exitPass, "peak bus bandwidth %.2f GB/s at %s over %d ranks with %s; "+
		"acceptance is the profile's thresholds to decide",
		peak.BusGBs, humanBytes(peak.Bytes), 2, peerIP)
}

// ── harness invocation ───────────────────────────────────────────────────────

func harnessArgs(rank int, peer string, cfg plan) []string {
	a := []string{"/usr/local/bin/nccl_pair",
		"--rank", strconv.Itoa(rank),
		"--nranks", "2",
		"--port", strconv.Itoa(cfg.bootstrapPort),
		"--warmup", strconv.Itoa(cfg.warmup),
		"--iters", strconv.Itoa(cfg.iters),
	}
	if cfg.sizes != "" {
		a = append(a, "--sizes", cfg.sizes)
	}
	if peer != "" {
		a = append(a, "--peer", peer)
	}
	return a
}

// ncclEnv PINS NCCL to the fabric this test is about.
//
// Left to itself NCCL enumerates every interface it can see and picks by its own
// ranking. On a burn-in node that set includes the wireless interface, a CNI
// overlay, and docker0 — and NCCL choosing one of those produces a number that
// is real, reproducible, and about the wrong link entirely. So the device, its
// GID index and the socket interface are all named explicitly, from the SAME
// route-based discovery ib-write-bw uses, so the two tests are measuring the
// same physical path and their numbers can be compared.
//
// An operator's own env wins: these are appended to os.Environ(), which the
// operator has already populated from spec.runner.env.
func ncclEnv(ports []rdmaPort, peerIP net.IP) []string {
	port, why, err := selectPort(ports, peerIP)
	if err != nil {
		logf("no RDMA port selected (%v); leaving NCCL to choose its own transport", err)
		return nil
	}
	logf("pinning NCCL to %s; %s", port, why)

	env := []string{
		// <device>:<port> is the form NCCL_IB_HCA takes; naming the port matters
		// on a dual-port HCA where only one side is cabled.
		fmt.Sprintf("NCCL_IB_HCA=%s:%d", port.Device, port.Port),
		// NCCL's own bootstrap ring is a TCP connection, and it must not go over
		// the wireless interface while the data goes over the fabric.
		"NCCL_SOCKET_IFNAME=" + port.NetDev,
	}
	if port.GIDIndex >= 0 {
		// On RoCE there is no safe default: the low GID indices are the
		// link-local IPv6 ones, and picking one produces a connection failure
		// that reads as a broken fabric.
		env = append(env, "NCCL_IB_GID_INDEX="+strconv.Itoa(port.GIDIndex))
	}
	return env
}

func runHarness(args, env []string, timeout time.Duration) (string, error) {
	logf("running %s", strings.Join(args, " "))
	ctx, cancel := context.WithTimeout(context.Background(), timeout)
	defer cancel()
	cmd := exec.CommandContext(ctx, args[0], args[1:]...)
	cmd.Env = append(os.Environ(), env...)
	var buf strings.Builder
	cmd.Stdout, cmd.Stderr = &buf, &buf
	err := cmd.Run()
	if ctx.Err() != nil {
		return buf.String(), fmt.Errorf("%s did not finish within %s", args[0], timeout)
	}
	return buf.String(), err
}

// waitForListener polls /proc/net/tcp for a LISTEN socket on port.
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

// echo copies the harness's output into the log, prefixed so that no line of it
// can be mistaken for a key=value metric by pkg/runner's parser. NCCL's own
// diagnostics travel this way too, which is the point: they are the first thing
// anyone wants when a collective is slow.
func echo(out string) {
	for _, line := range strings.Split(strings.TrimRight(out, "\n"), "\n") {
		if strings.TrimSpace(line) == "" {
			continue
		}
		fmt.Printf("| %s\n", line)
	}
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
