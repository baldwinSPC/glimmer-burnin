package main

import (
	"strings"
	"testing"
)

// Latency as its own gated execution — issue #163.
//
// ib-write-bw measures bandwidth and latency. Latency is a different property of
// the same link, and it is the one that matters for small-message collectives:
// the all-reduce at the end of every training step is latency-bound long before
// it is bandwidth-bound. Before this it was a number that happened to come along
// with a bandwidth test, so it could not carry its own thresholds.
//
// It is a VARIANT of this kind rather than a new TestKind, which keeps one
// image, one rendezvous and one alias table. The two binaries already ship here.

// An unset axis means both, which is what every execution written before
// variants existed has.
func TestAnUnsetMeasurandMeansBoth(t *testing.T) {
	got, err := measurandFrom("")
	if err != nil {
		t.Fatalf("an execution with no variant axis was refused: %v", err)
	}
	if got != measurandBoth {
		t.Fatalf("measurand = %q, want %q", got, measurandBoth)
	}
	if !got.measuresBandwidth() || !got.measuresLatency() {
		t.Error("the default stopped measuring both, which changes the behaviour of " +
			"every profile written before variants existed")
	}
}

func TestEachMeasurandSelectsItsOwnPhases(t *testing.T) {
	for _, tc := range []struct {
		in      string
		want    measurand
		bw, lat bool
	}{
		{"bandwidth", measurandBandwidth, true, false},
		{"latency", measurandLatency, false, true},
		{"both", measurandBoth, true, true},
		// Case and surrounding whitespace are a YAML author's business, not a
		// reason to refuse a measurement.
		{"  LATENCY  ", measurandLatency, false, true},
		{"Bandwidth", measurandBandwidth, true, false},
	} {
		got, err := measurandFrom(tc.in)
		if err != nil {
			t.Errorf("measurandFrom(%q): %v", tc.in, err)
			continue
		}
		if got != tc.want {
			t.Errorf("measurandFrom(%q) = %q, want %q", tc.in, got, tc.want)
		}
		if got.measuresBandwidth() != tc.bw || got.measuresLatency() != tc.lat {
			t.Errorf("%q selects bw=%v lat=%v, want bw=%v lat=%v",
				tc.in, got.measuresBandwidth(), got.measuresLatency(), tc.bw, tc.lat)
		}
	}
}

// An unrecognised measurand is REFUSED, never defaulted.
//
// Defaulting to "both" would silently give an author the measurement they did
// not ask for, with thresholds written for the one they did — so a latency gate
// would be applied to a bandwidth run, fail every link forever, and read as a
// fabric verdict on healthy hardware.
func TestAnUnrecognisedMeasurandIsRefused(t *testing.T) {
	for _, bad := range []string{"bw", "lat", "throughput", "jitter", "true", "0"} {
		got, err := measurandFrom(bad)
		if err == nil {
			t.Errorf("measurandFrom(%q) = %q with no error. Guessing here applies "+
				"thresholds written for one property to a measurement of another.",
				bad, got)
			continue
		}
		if !strings.Contains(err.Error(), bad) {
			t.Errorf("the refusal does not quote the offending value: %v", err)
		}
		if !strings.Contains(err.Error(), "bandwidth") || !strings.Contains(err.Error(), "latency") {
			t.Errorf("the refusal does not say what the valid values are: %v", err)
		}
	}
}

// The two selectors are exhaustive and never both false.
//
// A measurand that selects no phase at all would run nothing and then report
// something, which is the shape of a runner certifying hardware it never
// touched.
func TestEveryMeasurandSelectsAtLeastOnePhase(t *testing.T) {
	for _, m := range []measurand{measurandBoth, measurandBandwidth, measurandLatency} {
		if !m.measuresBandwidth() && !m.measuresLatency() {
			t.Errorf("%q selects no phase; the runner would measure nothing and still "+
				"return a verdict", m)
		}
	}
}
