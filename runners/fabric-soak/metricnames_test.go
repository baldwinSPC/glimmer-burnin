// SPDX-License-Identifier: Apache-2.0
// Copyright the Glimmer authors.

package main

import (
	"strings"
	"testing"

	"github.com/baldwinSPC/glimmer-burnin/pkg/contract"
	"github.com/baldwinSPC/glimmer-burnin/pkg/runner"
)

// The seam between this runner's key names and the canonical ones a threshold
// is written against. Test-only imports; the image build deletes *_test.go,
// which is what keeps the shipped sources standard-library-only.

func TestEmittedNamesObeyTheContract(t *testing.T) {
	// Exactly what emit() can print on a completed soak.
	out := strings.Join([]string{
		"soakIterations=180",
		"soakFailedIterations=0",
		"minBandwidthGbps=96.40",
		"meanBandwidthGbps=98.90",
		"peakBandwidthGbps=99.60",
		"bandwidthStdDevGbps=0.71",
		"p1BandwidthGbps=96.80",
		"linkErrorEvents=0",
	}, "\n")

	res := runner.Parse("fabric-soak", out, 0)
	if len(res.Metrics) != 8 {
		t.Fatalf("parsed %d metrics from 8 lines: %v", len(res.Metrics), res.Metrics)
	}
	for name := range res.Metrics {
		if _, known := contract.Lookup(name); !known {
			t.Errorf("%q is not registered — a threshold on it can never be evaluated", name)
		}
		if err := contract.ValidateMetricName(name); err != nil {
			t.Errorf("%q breaks the name grammar: %v", name, err)
		}
	}
}

func TestZeroFailuresAndZeroLinkErrorsAreMeasurementsNotAbsences(t *testing.T) {
	// Both are counters a healthy soak reports as exactly zero, which is what
	// makes `Equal 0` safe on day one. If either were omitted instead, the gate
	// would fail closed and condemn a healthy link.
	res := runner.Parse("fabric-soak", "soakFailedIterations=0\nlinkErrorEvents=0", 0)
	for _, name := range []string{"soakFailedIterations", "linkErrorEvents"} {
		if res.Metrics[name] != "0" {
			t.Errorf("%q did not survive as a measured zero: %v", name, res.Metrics)
		}
	}
}

func TestAnUnreadableCounterIsDeclaredRatherThanReportedAsZero(t *testing.T) {
	// The distinction the whole counter path exists to preserve. Without a
	// sysfs mount, or after a port reset, there is no delta over the window —
	// and a zero would certify a fabric from a file nobody opened.
	res := runner.Parse("fabric-soak", "soakFailedIterations=0\nlinkErrorEvents=n/a", 0)

	if _, measured := res.Metrics["linkErrorEvents"]; measured {
		t.Error("n/a became a measurement")
	}
	if !res.Unmeasurable["linkErrorEvents"] {
		t.Errorf("linkErrorEvents was not declared unmeasurable: %v", res.Unmeasurable)
	}
	// And the run still reports what it did measure.
	if res.Metrics["soakFailedIterations"] != "0" {
		t.Error("the failure count was lost alongside the unmeasurable counter")
	}
}
