// SPDX-License-Identifier: Apache-2.0
// Copyright the Glimmer authors.

package main

import (
	"math"
	"net"
	"strconv"
	"strings"
	"testing"
	"time"
)

// A real iperf3 3.16 --json summary, trimmed to the objects this runner reads.
const goodRun = `{
  "start": {"version": "iperf 3.16"},
  "end": {
    "streams": [
      {"sender": {"socket": 5, "bits_per_second": 94200000000, "retransmits": 12, "mean_rtt": 143, "max_rtt": 502}}
    ],
    "sum_sent": {"seconds": 30.000431, "bytes": 353250000000, "bits_per_second": 94200000000, "retransmits": 12},
    "sum_received": {"seconds": 30.000431, "bits_per_second": 94100000000}
  }
}`

func TestARealRunIsReadFromTheSenderSummary(t *testing.T) {
	// The sender's summary, because retransmits and RTT are properties the
	// SENDING stack observes and the receiver cannot report. Mixing ends would
	// produce a result whose three numbers came from two places.
	res, err := parseIperf(goodRun)
	if err != nil {
		t.Fatalf("parseIperf: %v", err)
	}
	if math.Abs(res.ThroughputGbps-94.2) > 0.001 {
		t.Errorf("throughput = %v Gbps, want 94.2", res.ThroughputGbps)
	}
	if res.Retransmits != 12 {
		t.Errorf("retransmits = %d, want 12", res.Retransmits)
	}
	if !res.HasRTT || res.MeanRttUs != 143 {
		t.Errorf("rtt = %v (present=%v), want 143us", res.MeanRttUs, res.HasRTT)
	}
	if math.Abs(res.Seconds-30.000431) > 0.0001 {
		t.Errorf("seconds = %v", res.Seconds)
	}
}

func TestAMissingRTTIsOmittedAndNotReportedAsZero(t *testing.T) {
	// mean_rtt is absent on platforms with no TCP_INFO. A zero there is a
	// measurement nobody took, and a gate on it would certify a path this
	// runner never timed — the same rule host-health follows for xidEvents.
	noRTT := `{"end": {
		"streams": [{"sender": {"bits_per_second": 1000000000}}],
		"sum_sent": {"seconds": 10, "bits_per_second": 1000000000, "retransmits": 0}
	}}`
	res, err := parseIperf(noRTT)
	if err != nil {
		t.Fatalf("parseIperf: %v", err)
	}
	if res.HasRTT {
		t.Error("an absent mean_rtt was reported as present")
	}
	if res.MeanRttUs != 0 {
		t.Errorf("MeanRttUs = %v with nothing to report", res.MeanRttUs)
	}
}

func TestZeroRetransmitsIsAMeasurementAndNotAnAbsence(t *testing.T) {
	// The other direction, and it matters just as much: a clean link really
	// does retransmit zero times, and dropping the key would make a gate on it
	// fail closed against a healthy node.
	clean := `{"end": {
		"streams": [{"sender": {"mean_rtt": 90}}],
		"sum_sent": {"seconds": 30, "bits_per_second": 99000000000, "retransmits": 0}
	}}`
	res, err := parseIperf(clean)
	if err != nil {
		t.Fatalf("parseIperf: %v", err)
	}
	if res.Retransmits != 0 {
		t.Errorf("retransmits = %d, want 0", res.Retransmits)
	}
	if res.ThroughputGbps <= 0 {
		t.Error("a clean run reported no throughput")
	}
}

func TestIperfsOwnErrorIsNotEvidenceAboutACable(t *testing.T) {
	// iperf3 reports its own failures inside the JSON. These are machinery —
	// "unable to connect" says the run did not happen, not that the link is
	// bad — and the runner turns them into an Error rather than a Fail.
	body := `{"error": "unable to connect to server - server may have stopped running or use a different port number"}`
	res, err := parseIperf(body)
	if err != nil {
		t.Fatalf("parseIperf: %v", err)
	}
	if !strings.Contains(res.Error, "unable to connect") {
		t.Errorf("the error did not survive parsing: %+v", res)
	}
}

func TestOutputThatIsNotIperfJSONIsRefused(t *testing.T) {
	for _, raw := range []string{"", "iperf3: error - the server is busy", "<html>502</html>"} {
		if _, err := parseIperf(raw); err == nil {
			t.Errorf("accepted %q as iperf3 JSON", raw)
		}
	}
}

func TestEnvIntRejectsNonsenseRatherThanRunningForZeroSeconds(t *testing.T) {
	t.Setenv("X", "")
	if got := envInt("X", 30); got != 30 {
		t.Errorf("empty → %d, want the default 30", got)
	}
	for _, bad := range []string{"abc", "-5", "0"} {
		t.Setenv("X", bad)
		if got := envInt("X", 30); got != 30 {
			t.Errorf("%q → %d, want the default 30 — a zero-second test measures nothing", bad, got)
		}
	}
	t.Setenv("X", "900")
	if got := envInt("X", 30); got != 900 {
		t.Errorf("900 → %d", got)
	}
}

func TestAClientThatStartsFirstWaitsInsteadOfDying(t *testing.T) {
	// The failure mode this whole design exists to avoid. iperf3's own
	// --connect-timeout cannot do this job: a connection to a port nothing is
	// listening on is REFUSED immediately with an RST, so the timeout never
	// elapses and the client exits at once with "unable to connect" — which
	// reads to an operator like a fabric fault.
	ln, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	port := ln.Addr().(*net.TCPAddr).Port
	ln.Close() // nothing is listening yet

	// The listener appears while the client is already waiting.
	ready := make(chan struct{})
	go func() {
		time.Sleep(300 * time.Millisecond)
		late, err := net.Listen("tcp", net.JoinHostPort("127.0.0.1", strconv.Itoa(port)))
		if err != nil {
			close(ready)
			return
		}
		close(ready)
		time.Sleep(3 * time.Second)
		late.Close()
	}()

	start := time.Now()
	if err := waitForListener("127.0.0.1", port, 10*time.Second); err != nil {
		t.Fatalf("a listener that appeared after 300ms was not waited for: %v", err)
	}
	if time.Since(start) < 200*time.Millisecond {
		t.Error("returned before the listener could possibly have been up")
	}
	<-ready
}

func TestWaitingGivesUpEventuallyRatherThanHanging(t *testing.T) {
	// A server that never comes up must end the run, not hold a node forever.
	ln, _ := net.Listen("tcp", "127.0.0.1:0")
	port := ln.Addr().(*net.TCPAddr).Port
	ln.Close()

	start := time.Now()
	if err := waitForListener("127.0.0.1", port, 1500*time.Millisecond); err == nil {
		t.Fatal("waiting on a port nobody listens to succeeded")
	}
	if elapsed := time.Since(start); elapsed > 8*time.Second {
		t.Errorf("gave up after %v, far past the deadline it was given", elapsed)
	}
}
