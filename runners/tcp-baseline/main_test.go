// SPDX-License-Identifier: Apache-2.0
// Copyright the Glimmer authors.

package main

import (
	"math"
	"strings"
	"testing"
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

func TestAnAbsentRetransmitCountIsOmittedAndNotReportedAsZero(t *testing.T) {
	// THE SAME RULE AS THE RTT ABOVE, on the metric that is actually gated.
	//
	// retransmits comes from TCP_INFO exactly as mean_rtt does, and iperf3 omits
	// the key on a platform that cannot supply it. Reporting the absence as 0
	// would be a zero nobody measured — and config/samples/tcp-baseline.yaml
	// gates `tcpRetransmits Equal 0`, so it would not merely be inaccurate, it
	// would PASS. A clean-looking certification of a path whose retransmits were
	// never counted is the fail-open this project's whole verdict discipline is
	// built to refuse.
	//
	// Omitted rather than `n/a`: a missing JSON key is an absence, and absence is
	// not a declaration. The runner did not positively establish that this
	// platform has nothing to report — it just never saw the field. Both routes
	// fail closed against a Required gate; only omission is honest about which
	// of the two happened.
	noRetrans := `{"end": {
		"streams": [{"sender": {"mean_rtt": 90}}],
		"sum_sent": {"seconds": 30, "bits_per_second": 99000000000}
	}}`
	res, err := parseIperf(noRetrans)
	if err != nil {
		t.Fatalf("parseIperf: %v", err)
	}
	if res.HasRetransmits {
		t.Error("an absent retransmits field was reported as present")
	}
	if res.Retransmits != 0 {
		t.Errorf("Retransmits = %d with nothing to report", res.Retransmits)
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

func TestAnUnreportedMetricNeverReachesStdout(t *testing.T) {
	// The rule at the boundary that matters. parseIperf getting it right is
	// necessary and not sufficient: an unconditional print downstream
	// reintroduces the fabricated zero with the parser still passing its own
	// tests. This asserts the bytes.
	keys := func(res iperfResult) map[string]string {
		m := map[string]string{}
		for _, kv := range measurements(res) {
			m[kv[0]] = kv[1]
		}
		return m
	}

	got := keys(iperfResult{ThroughputGbps: 94.2, Seconds: 30})
	if v, ok := got["tcpRetransmits"]; ok {
		t.Errorf("tcpRetransmits=%s emitted with nothing measured — the sample gates this Equal 0, so a "+
			"fabricated zero certifies a path whose retransmits were never counted", v)
	}
	if v, ok := got["tcpRttUs"]; ok {
		t.Errorf("tcpRttUs=%s emitted with nothing measured", v)
	}
	// Throughput and elapsed are always measured when iperf3 completed, so
	// their absence would be the opposite bug.
	for _, want := range []string{"tcpThroughputGbps", "elapsedS"} {
		if _, ok := got[want]; !ok {
			t.Errorf("%s is missing from a completed run", want)
		}
	}

	full := keys(iperfResult{
		ThroughputGbps: 94.2, Seconds: 30,
		Retransmits: 0, HasRetransmits: true,
		MeanRttUs: 143, HasRTT: true,
	})
	if full["tcpRetransmits"] != "0" {
		t.Errorf("a measured zero was dropped: %q — that would fail a healthy link closed", full["tcpRetransmits"])
	}
	if full["tcpRttUs"] != "143" {
		t.Errorf("tcpRttUs = %q, want 143", full["tcpRttUs"])
	}
}
