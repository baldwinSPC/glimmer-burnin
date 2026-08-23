// SPDX-License-Identifier: Apache-2.0
// Copyright the Glimmer authors.

// Command fabric-soak runs the RDMA link for hours instead of for a minute.
//
// ib-write-bw answers "does this link work right now": about a minute of
// traffic, an average, a verdict. It passes on a link with a marginal optic, a
// cable seated badly, or a transceiver that fails once it is warm — and those
// are the failures that hurt most in production, because they present as a slow
// training job or an occasional NCCL timeout rather than as a link that is
// down.
//
// This runner iterates the same measurement over a long window and reports what
// a single run cannot: the WORST window, the spread across windows, how many
// iterations failed, and whether the port's error counters moved WHILE it ran.
// A soak that averaged fine and spent ninety seconds at a third of line rate has
// found something; an average hides it completely.
//
// Contract:
//
//	exit 0                                  the link was soaked
//	exit 1                                  measured and wrong: iterations failed
//	exit 2  FABRIC_SOAK_SKIP: <reason>      no fabric here, or not half of a Pair
//	exit 3  FABRIC_SOAK_ERROR: <reason>     machinery; the link is UNJUDGED
//
// It speaks the ordinary Pair rendezvous and adds no environment of its own
// beyond its tuning knobs — one image is the server on one node and the client
// on the other.
package main

import (
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

	pairRoleServer = "server"
	pairRoleClient = "client"

	// defaultWindowSeconds is one iteration's traffic. Long enough that a
	// transceiver has to actually carry data, short enough that the soak has
	// many windows to compare — the spread across windows is the signal, and
	// four windows cannot show a spread.
	defaultWindowSeconds = 20

	// emitEverySeconds is how often cumulative counters are re-printed.
	//
	// Periodic re-emission is what makes a killed run useful. The parser is
	// last-occurrence-wins, so each emission supersedes the previous one and a
	// run cut short at its deadline still leaves everything measured up to that
	// point. Without it a seven-hour soak that is evicted at hour six reports
	// nothing at all.
	emitEverySeconds = 60

	// startSkewBudgetSeconds is how long the operator may take to get the
	// client pod actually running after the server reports Ready.
	//
	// The Pair rendezvous does not create the client until the server is
	// Ready (correctly — see runClient's contract), which means the server's
	// clock always starts first. With identical durations, the server's
	// deadline therefore always fires before the client's, by however many
	// seconds that startup took — usually a couple, per #459's own
	// measurement, but scheduling and image-pull delay under contention can
	// take longer, and this has to cover the case it does not, not the case
	// it usually is.
	startSkewBudgetSeconds = 60

	// The transfer perftest is asked for, before the memlock budget trims it.
	// Matched to ib-write-bw's own defaults so a soak's windows are comparable
	// with the one-shot measurement of the same link.
	defaultMessageBytes = 1 << 20
	defaultQPs          = 1
)

func main() { os.Exit(run()) }

func fin(code int, format string, args ...any) int {
	msg := fmt.Sprintf(format, args...)
	switch code {
	case exitSkip:
		fmt.Printf("FABRIC_SOAK_SKIP: %s\n", sanitize(msg))
	case exitError:
		fmt.Printf("FABRIC_SOAK_ERROR: %s\n", sanitize(msg))
	case exitFail:
		fmt.Printf("FABRIC_SOAK_FAIL: %s\n", sanitize(msg))
	}
	logf("%s", msg)
	return code
}

func run() int {
	role := strings.TrimSpace(os.Getenv("BURNIN_ROLE"))
	peerHost := strings.TrimSpace(os.Getenv("BURNIN_PEER_HOST"))
	peerNode := strings.TrimSpace(os.Getenv("BURNIN_PEER_NODE"))

	if os.Getenv("BURNIN_RANK") != "" && role == "" {
		return fin(exitSkip, "BURNIN_RANK is set, so this is a Group execution; this runner speaks "+
			"only the Pair rendezvous. Group-scope fabric soak is not wired up yet")
	}
	if role == "" {
		return fin(exitSkip, "BURNIN_ROLE is not set, so this pod is not half of a Pair — "+
			"a link soak with one end measures nothing")
	}
	if role != pairRoleServer && role != pairRoleClient {
		return fin(exitError, "BURNIN_ROLE=%q is neither %q nor %q", role, pairRoleServer, pairRoleClient)
	}
	// ONLY THE CLIENT NEEDS IT. The server learns its peer from the accepted
	// connection (conn.RemoteAddr) — see rendezvous.go, which explains why it
	// MUST: the server starts first, and in-cluster its peer's DNS name does not
	// resolve until the operator creates the client, which the operator will not
	// do until the server is Ready. So the server has never read this value.
	//
	// Requiring it anyway made the variable a presence token, and that broke the
	// bare-metal dispatcher outright: `burnin run --role server` correctly does
	// not set a peer it knows the server cannot use, so every Pair test errored
	// before the server bound its socket (#287). The operator sets it for both
	// roles, so in-cluster hid this completely.
	//
	// tcp-baseline already had the right shape and is the model.
	if role == pairRoleClient && peerHost == "" {
		return fin(exitError, "BURNIN_ROLE=%s is the deciding side but BURNIN_PEER_HOST is empty — there is no peer to dial", role)
	}

	duration := envInt("BURNIN_DURATION_SECONDS", 3600)
	windowSeconds := envInt("FABRIC_SOAK_WINDOW_SECONDS", defaultWindowSeconds)
	if windowSeconds >= duration {
		return fin(exitError, "FABRIC_SOAK_WINDOW_SECONDS=%d is not shorter than the %ds soak — "+
			"a soak of one window is an ib-write-bw run with a longer name, and reports no spread",
			windowSeconds, duration)
	}

	ports, err := discoverPorts()
	if err != nil {
		return fin(exitError, "enumerating RDMA devices: %v", err)
	}
	if len(ports) == 0 {
		return fin(exitSkip, "this node has no RDMA device, so there is no fabric to soak")
	}

	// THE MEMLOCK LIMIT, CHECKED UP FRONT.
	//
	// Every RDMA buffer is pinned memory, so ibv_create_cq fails with ENOMEM
	// when RLIMIT_MEMLOCK is too small — and perftest reports that as
	// "Couldn't create CQ", which names neither the limit nor the resource and
	// is indistinguishable from a broken HCA. That cost this project a long
	// bisect once, which is why memlock.go exists.
	//
	// It matters MORE here than in ib-write-bw. A one-minute run fails once and
	// loudly; a four-hour soak would fail every window for four hours and report
	// a link that carried no traffic, with the real cause named nowhere. So the
	// limit is read before the first window and a hopeless one is an Error
	// naming RLIMIT_MEMLOCK, not a fabric verdict.
	soft, raised, mlErr := memlock()
	if mlErr != nil {
		return fin(exitError, "%v", mlErr)
	}
	if raised {
		logf("fabric-soak: raised the RLIMIT_MEMLOCK soft limit to the hard limit (%s)", humanBytes(soft))
	}
	logf("fabric-soak: RLIMIT_MEMLOCK = %s", humanBytes(soft))
	// The port that can REACH THE PEER, not simply the first one enumerated.
	//
	// On a node with one HCA the two are the same. On a node with several they
	// are not, and picking ports[0] on each end independently would have the two
	// ends soaking different fabrics — or one end talking to a device with no
	// route to the other at all, which perftest reports as a connection failure
	// that reads as a dead link.
	//
	// A peer that does not resolve is not fatal here: selectPort falls back to
	// its own preference order, and the reason is logged either way.
	var peerIP net.IP
	if addrs, rerr := net.LookupIP(peerHost); rerr == nil && len(addrs) > 0 {
		peerIP = addrs[0]
	} else {
		logf("fabric-soak: could not resolve %s (%v); selecting a port without a route hint",
			peerHost, rerr)
	}
	port, why, perr := selectPort(ports, peerIP)
	if perr != nil {
		return fin(exitError, "choosing an RDMA port: %v", perr)
	}
	logf("fabric-soak: using %s (%s)", port, why)

	// Size the transfer to the memlock budget rather than letting perftest
	// discover the limit the hard way, one window at a time.
	messageBytes, qps, notes, fits := fitToMemlock(defaultMessageBytes, defaultQPs, budgetFor(soft))
	for _, n := range notes {
		logf("fabric-soak: %s", n)
	}
	if !fits {
		return fin(exitError, "RLIMIT_MEMLOCK is %s and no message size this runner will accept fits "+
			"inside it. %s", humanBytes(soft), memlockAdvice(soft, minMessageBytes))
	}

	logf("fabric-soak: %s, peer %s (node %s), %ds in %ds windows, %s x %d QP",
		role, peerHost, peerNode, duration, windowSeconds, humanBytes(uint64(messageBytes)), qps)

	if role == pairRoleServer {
		return runServer(port, duration, windowSeconds, messageBytes, qps)
	}
	return runClient(port, peerHost, peerNode, duration, windowSeconds, messageBytes, qps, soft)
}

// serverDeadline is when the server stops restarting ib_write_bw, computed
// from its own start time rather than the client's — the server never learns
// when the client actually started (#459).
//
// It runs past duration by however long the client's own last window can
// still legitimately be running: the server and client each compute
// start+duration independently, the client always starts later (the operator
// waits for the server to be Ready before creating it), so with no grace the
// server's earlier deadline can fire while the client is mid-window — and
// that window then fails to connect because the server ran out of time
// first, not because the link is bad. The grace has to cover both the
// pod-start skew that put the client behind to begin with
// (startSkewBudgetSeconds) and the client's own worst-case last window (one
// full window plus its hang tolerance, windowGraceSeconds).
func serverDeadline(start time.Time, durationSeconds, windowSeconds int) time.Time {
	grace := startSkewBudgetSeconds + windowSeconds + windowGraceSeconds
	return start.Add(time.Duration(durationSeconds+grace) * time.Second)
}

// runServer holds the far end open for the whole soak, plus a grace period —
// see serverDeadline.
//
// The server never decides — the client is the deciding side at Pair scope, as
// everywhere in this project. It restarts ib_write_bw for each window because
// perftest's server exits when its client disconnects, and a soak is many
// connections rather than one long one.
func runServer(port rdmaPort, duration, windowSeconds, messageBytes, qps int) int {
	deadline := serverDeadline(time.Now(), duration, windowSeconds)
	restarts := 0

	for time.Now().Before(deadline) {
		cmd := exec.Command("ib_write_bw", serverArgs(port, messageBytes, qps)...)
		cmd.Stdout, cmd.Stderr = os.Stderr, os.Stderr
		if err := cmd.Start(); err != nil {
			return fin(exitError, "starting the ib_write_bw server: %v", err)
		}
		_ = cmd.Wait() // a client disconnect ends it; that is the normal case
		restarts++
	}

	metric("soakServerRestarts", strconv.Itoa(restarts))
	logf("fabric-soak: server finished after %d window(s)", restarts)
	fmt.Println("\nThis is the SERVER end. The client's result is the link's verdict.")
	return exitPass
}

// runClient iterates the measurement and decides.
func runClient(port rdmaPort, peerHost, peerNode string, duration, windowSeconds, messageBytes, qps int, memlockSoft uint64) int {
	deadline := time.Now().Add(time.Duration(duration) * time.Second)

	// The counter baseline is taken HERE, immediately before the first window,
	// not at process start: everything between is setup, and a delta that
	// included it would attribute the setup's errors to the soak.
	sysfs := strings.TrimSpace(os.Getenv("FABRIC_SOAK_SYSFS"))
	if sysfs == "" {
		sysfs = "/sys"
	}
	before := readPortCounters(sysfs, port.Device, strconv.Itoa(port.Port))

	var s soak
	lastEmit := time.Now()

	for time.Now().Before(deadline) {
		out, err := runWindow(port, peerHost, windowSeconds, messageBytes, qps)
		if err != nil {
			s.record(0, false)
			if looksLikeMemlockExhaustion(out) {
				// Named rather than counted. Every window would fail the same
				// way, and "the link is dead" is the wrong sentence for a
				// limit this process could see from the start.
				return fin(exitError, "window %d failed at ibv_create_cq, which is RLIMIT_MEMLOCK "+
					"exhaustion and not a fabric fault: %s", s.iterations(), memlockAdvice(memlockSoft, uint64(messageBytes)))
			}
			logf("fabric-soak: window %d failed: %v", s.iterations(), err)
		} else if bw, perr := parseBandwidth(out); perr != nil {
			// perftest completed and this runner could not read it. That is a
			// statement about the parser, not the fabric, so it counts as a
			// failed WINDOW rather than a Fail — the decision below only fails
			// on failures, and an unreadable window is one.
			s.record(0, false)
			logf("fabric-soak: window %d unreadable: %v", s.iterations(), perr)
		} else {
			s.record(bw.AverageGbps, true)
		}

		if time.Since(lastEmit) >= emitEverySeconds*time.Second {
			emit(&s, nil, true)
			lastEmit = time.Now()
		}
	}

	after := readPortCounters(sysfs, port.Device, strconv.Itoa(port.Port))
	d, clean := delta(before, after)

	emit(&s, &counterReport{delta: d, clean: clean, read: len(before) > 0}, false)

	st, ok := s.stats()
	if !ok {
		return fin(exitError, "no window completed against %s (node %s) in %ds — the link was not soaked",
			peerHost, peerNode, duration)
	}
	if f := s.failures(); f > 0 {
		return fin(exitFail, "%d of %d windows failed against %s (node %s); the link carried traffic "+
			"and stopped carrying it", f, s.iterations(), peerHost, peerNode)
	}

	logf("fabric-soak: %d windows, min %.2f / mean %.2f / max %.2f Gb/s, sd %.2f",
		st.count, st.min, st.mean, st.max, st.stddev)
	return exitPass
}

// counterReport is the port-counter delta, and whether it is trustworthy.
type counterReport struct {
	delta portCounters
	// clean is false when a counter went BACKWARDS, which means the port was
	// reset mid-soak.
	clean bool
	// read is false when no counter could be read at all — no sysfs mount.
	read bool
}

// emit prints the cumulative figures.
//
// Called periodically during the soak and once at the end. Every emission is
// complete rather than incremental, because the parser is last-occurrence-wins:
// a partial log then still yields a coherent set rather than a mixture of two
// windows.
func emit(s *soak, c *counterReport, partial bool) {
	metric("soakIterations", strconv.Itoa(s.iterations()))
	metric("soakFailedIterations", strconv.Itoa(s.failures()))

	if st, ok := s.stats(); ok {
		metric("minBandwidthGbps", trim(st.min))
		metric("meanBandwidthGbps", trim(st.mean))
		metric("peakBandwidthGbps", trim(st.max))
		metric("bandwidthStdDevGbps", trim(st.stddev))
		metric("p1BandwidthGbps", trim(st.p1))
	}

	if c == nil {
		// A periodic emission takes no counter reading: differencing against a
		// mid-run snapshot would report a delta for a window that has not
		// finished, and the final emission replaces this anyway.
		return
	}
	switch {
	case !c.read:
		// Nothing was read, so nothing is declared. NOT a zero: an unread
		// counter reported as zero is a fabric certified by a file nobody
		// opened. `n/a` says the runner looked and found nothing to read.
		metric("linkErrorEvents", "n/a")
	case !c.clean:
		// A counter went backwards: the port was reset during the soak, so no
		// delta over the window exists to report.
		metric("linkErrorEvents", "n/a")
		logf("fabric-soak: a port counter decreased — the port was reset mid-soak, so the " +
			"error delta over this window cannot be computed")
	default:
		metric("linkErrorEvents", strconv.FormatInt(c.delta.total(), 10))
		for name, v := range c.delta {
			// Evidence, so the single number above can be explained. Keys are
			// the kernel's own counter names, echoed rather than renamed.
			logf("fabric-soak: %s delta %d", name, v)
		}
	}
	_ = partial
}

func trim(v float64) string { return strconv.FormatFloat(v, 'f', 2, 64) }

// runWindow is one iteration of traffic.
func runWindow(p rdmaPort, peerHost string, seconds, messageBytes, qps int) (string, error) {
	args := clientArgs(p, peerHost, seconds, messageBytes, qps)
	cmd := exec.Command(args[0], args[1:]...)
	var sb strings.Builder
	cmd.Stdout, cmd.Stderr = &sb, os.Stderr

	if err := cmd.Start(); err != nil {
		return "", err
	}
	done := make(chan error, 1)
	go func() { done <- cmd.Wait() }()

	select {
	case err := <-done:
		return sb.String(), err
	case <-time.After(time.Duration(seconds+windowGraceSeconds) * time.Second):
		_ = cmd.Process.Kill()
		return sb.String(), fmt.Errorf("window did not finish within %ds", seconds+windowGraceSeconds)
	}
}

// windowGraceSeconds is how long a window may overrun before it is killed and
// counted as failed. A window that hangs is exactly the flap this runner exists
// to catch, so it must be bounded rather than allowed to consume the soak.
const windowGraceSeconds = 30

func serverArgs(p rdmaPort, messageBytes, qps int) []string {
	a := []string{"-d", p.Device, "-i", strconv.Itoa(p.Port), "--report_gbits", "-F",
		"-s", strconv.Itoa(messageBytes), "-q", strconv.Itoa(qps)}
	if p.GIDIndex >= 0 {
		a = append(a, "-x", strconv.Itoa(p.GIDIndex))
	}
	return a
}

func clientArgs(p rdmaPort, peerHost string, seconds, messageBytes, qps int) []string {
	a := []string{"ib_write_bw",
		"-d", p.Device,
		"-i", strconv.Itoa(p.Port),
		"--report_gbits",
		"-D", strconv.Itoa(seconds),
		"-s", strconv.Itoa(messageBytes),
		"-q", strconv.Itoa(qps),
		"-F",
	}
	if p.GIDIndex >= 0 {
		a = append(a, "-x", strconv.Itoa(p.GIDIndex))
	}
	return append(a, peerHost)
}
