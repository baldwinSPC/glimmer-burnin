// SPDX-License-Identifier: Apache-2.0
// Copyright the Glimmer authors.

package runners

import (
	"strings"
	"testing"

	"github.com/baldwinSPC/glimmer-burnin/pkg/contract"
	"github.com/baldwinSPC/glimmer-burnin/pkg/runner"
)

// The seam between memory-bw's own key names and the canonical ones a threshold
// is written against.
//
// IT LIVES HERE, not in runners/memory-bw/, and not by preference: `go build`
// refuses a package directory holding C++ sources unless cgo is in use, so a
// Go file beside memory_bw.cc breaks the build of the whole module. That is the
// same reason cxxtests_test.go lives here. It is also why memory-bw had no
// parser test at all until the peer matrix landed (issue #161) — which is how
// three of its keys reached main unregistered, see
// TestTheMaxKeysAreRegisteredOrKnownNotToBe.
//
// The runner spells its units "gbs". Normalisation alone would turn that into
// "Gbs" — gigaBITS — for figures that are gigaBYTES, so every bandwidth key
// goes through the alias table. That is the mistake this file exists to keep
// caught: a silently wrong unit is worse than an unknown name, because it
// compares.

func TestPeerMatrixKeysReachTheCanonicalNames(t *testing.T) {
	out := strings.Join([]string{
		"h2d_bandwidth_gbs=52.10",
		"d2h_bandwidth_gbs=51.80",
		"d2d_bandwidth_gbs=1400.00",
		"peer_read_bandwidth_gbs=210.40",
		"peer_write_bandwidth_gbs=208.90",
	}, "\n")

	res := runner.Parse("memory-bw", out, 0)

	for raw, want := range map[string]string{
		"peer_read_bandwidth_gbs":  "peerReadBandwidthGBs",
		"peer_write_bandwidth_gbs": "peerWriteBandwidthGBs",
	} {
		if _, leaked := res.Metrics[raw]; leaked {
			t.Errorf("%q reached the result un-aliased", raw)
		}
		if _, ok := res.Metrics[want]; !ok {
			t.Errorf("%q did not become %q: %v", raw, want, res.Metrics)
		}
	}

	for name := range res.Metrics {
		if _, known := contract.Lookup(name); !known {
			t.Errorf("%q is not in the registry — a threshold on it can never be evaluated", name)
		}
		if err := contract.ValidateMetricName(name); err != nil {
			t.Errorf("%q breaks the name grammar: %v", name, err)
		}
	}
}

// TestASingleGPUNodeDeclaresThePeerMatrixUnmeasurable is the behaviour that
// makes this safe to ship to a fleet of DGX Sparks.
//
// A Spark has one accelerator, so there is no peer and no measurement to take.
// The runner emits `n/a`, which pkg/runner routes to Unmeasurable rather than
// Metrics — so a gate with applicability RequiredIfMeasurable is reported NOT
// EVALUATED instead of failing, and the three device-local figures the run DID
// measure still arrive.
//
// Omitting the keys instead would make such a gate fail closed and condemn
// every single-GPU node in the fleet; exiting 2 for the whole test would throw
// away the device-local measurements, which on a Spark are the entire value of
// the kind.
func TestASingleGPUNodeDeclaresThePeerMatrixUnmeasurable(t *testing.T) {
	out := strings.Join([]string{
		"h2d_bandwidth_gbs=52.10",
		"d2h_bandwidth_gbs=51.80",
		"d2d_bandwidth_gbs=1400.00",
		"peer_read_bandwidth_gbs=n/a",
		"peerReadBandwidthMaxGBs=n/a",
		"peer_write_bandwidth_gbs=n/a",
		"peerWriteBandwidthMaxGBs=n/a",
	}, "\n")

	res := runner.Parse("memory-bw", out, 0)

	for _, name := range []string{"peerReadBandwidthGBs", "peerWriteBandwidthGBs"} {
		if _, measured := res.Metrics[name]; measured {
			t.Errorf("%q arrived as a MEASUREMENT on a single-GPU node; n/a must not become a number", name)
		}
		if !res.Unmeasurable[name] {
			t.Errorf("%q was not declared unmeasurable: %v", name, res.Unmeasurable)
		}
	}

	// And the run still reports what it did measure.
	if _, ok := res.Metrics["hostToDeviceBandwidthGBs"]; !ok {
		t.Error("the device-local figures were lost — a Spark would get no memory-bw coverage at all")
	}
}

// TestTheMaxKeysAreRegisteredOrKnownNotToBe records a pre-existing gap rather
// than quietly widening it.
//
// The three original *MaxGBs evidence keys reached main unregistered and
// unaliased, because this runner had no parser test. The peer matrix's two are
// registered. This asserts the four peer keys are known to the registry and
// names the older three as the outstanding item, so the gap is visible instead
// of being inferred from an absence.
func TestTheMaxKeysAreRegisteredOrKnownNotToBe(t *testing.T) {
	for _, name := range []string{"peerReadBandwidthMaxGBs", "peerWriteBandwidthMaxGBs"} {
		if _, known := contract.Lookup(name); !known {
			t.Errorf("%q should be registered: it is emitted by this runner", name)
		}
	}

	// Not asserted as registered — they are not, and that is issue-tracked
	// rather than fixed here, because renaming or registering them changes what
	// a stored result means for anyone already reading them.
	for _, name := range []string{
		"hostToDeviceBandwidthMaxGBs", "deviceToHostBandwidthMaxGBs", "deviceToDeviceBandwidthMaxGBs",
	} {
		if _, known := contract.Lookup(name); known {
			t.Errorf("%q is now registered — good, but this test still describes it as the "+
				"outstanding gap. Update the comment and drop it from this list.", name)
		}
	}
}
