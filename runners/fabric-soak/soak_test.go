// SPDX-License-Identifier: Apache-2.0
// Copyright the Glimmer authors.

package main

import (
	"math"
	"os"
	"path/filepath"
	"strconv"
	"testing"
)

func TestAFailedWindowIsCountedAndNotAveragedIn(t *testing.T) {
	// The rule with teeth. A failed window contributes to the failure count and
	// to NOTHING else: averaging its zero into the bandwidth would report a
	// fabric slower than it is, and a fabricated zero is exactly what this
	// project forbids everywhere else.
	var s soak
	s.record(190, true)
	s.record(0, false) // failed
	s.record(194, true)

	if s.iterations() != 3 || s.failures() != 1 {
		t.Fatalf("iterations=%d failures=%d, want 3 and 1", s.iterations(), s.failures())
	}
	st, ok := s.stats()
	if !ok {
		t.Fatal("no stats from two good windows")
	}
	if st.count != 2 {
		t.Errorf("stats over %d windows, want the 2 that completed", st.count)
	}
	if st.min < 189 {
		t.Errorf("min = %v — the failed window's zero was averaged in", st.min)
	}
}

func TestTheWorstWindowSurvivesAHealthyAverage(t *testing.T) {
	// The reason this runner exists. A link that spent ninety seconds at a
	// third of line rate has found something, and the mean hides it.
	var s soak
	for i := 0; i < 100; i++ {
		s.record(196, true)
	}
	for i := 0; i < 5; i++ {
		s.record(64, true) // the flap
	}

	st, _ := s.stats()
	if st.mean < 180 {
		t.Errorf("mean = %v — this case is meant to have a healthy-looking average", st.mean)
	}
	if st.min > 70 {
		t.Errorf("min = %v, want the flap at 64 — the worst window is the verdict", st.min)
	}
	if st.stddev < 10 {
		t.Errorf("stddev = %v — a flapping link should show a wide spread", st.stddev)
	}
}

func TestASlowLinkAndAFlappingLinkAreDistinguishable(t *testing.T) {
	// Both have a low minimum. The spread is what separates "replace the optic"
	// from "re-check the cable budget", which is why both are reported.
	var flapping, slow soak
	for i := 0; i < 50; i++ {
		flapping.record(196, true)
		slow.record(120, true)
	}
	for i := 0; i < 50; i++ {
		flapping.record(60, true)
		slow.record(118, true)
	}

	f, _ := flapping.stats()
	sl, _ := slow.stats()

	if f.min > 70 || sl.min > 125 {
		t.Fatalf("preconditions wrong: flapping min %v, slow min %v", f.min, sl.min)
	}
	if f.stddev <= sl.stddev*5 {
		t.Errorf("flapping sd %v is not clearly wider than steady-slow sd %v — the two would "+
			"be indistinguishable in a report", f.stddev, sl.stddev)
	}
}

func TestThePercentileIsAValueAWindowReached(t *testing.T) {
	var s soak
	for i := 1; i <= 100; i++ {
		s.record(float64(i), true)
	}
	st, _ := s.stats()
	if st.p1 != 1 {
		t.Errorf("p1 = %v, want 1 — nearest-rank, a value a window actually reached", st.p1)
	}
	if math.Abs(st.mean-50.5) > 0.001 {
		t.Errorf("mean = %v", st.mean)
	}
}

func TestNoCompletedWindowIsNotZeroBandwidth(t *testing.T) {
	// A soak where every window failed has no bandwidth to report. It must not
	// report zero, which would read as a measured line rate of nothing.
	var s soak
	s.record(0, false)
	s.record(0, false)
	if _, ok := s.stats(); ok {
		t.Error("stats were produced from no completed window")
	}
	if s.failures() != 2 {
		t.Errorf("failures = %d, want 2", s.failures())
	}
}

// ── counters ─────────────────────────────────────────────────────────────────

func countersFixture(t *testing.T, values map[string]int64) string {
	t.Helper()
	root := t.TempDir()
	dir := filepath.Join(root, "class", "infiniband", "mlx5_0", "ports", "1", "counters")
	if err := os.MkdirAll(dir, 0o755); err != nil {
		t.Fatal(err)
	}
	for name, v := range values {
		if err := os.WriteFile(filepath.Join(dir, name), []byte(strconv.FormatInt(v, 10)+"\n"), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	return root
}

func TestCountersAreReadAsADeltaNotALifetimeTotal(t *testing.T) {
	// A NIC up for two hundred days has a large symbol_error_counter that says
	// nothing about the last four hours. Reporting the raw value would make
	// every long-lived node look faulty and every freshly booted one look clean.
	before := readPortCounters(countersFixture(t, map[string]int64{
		"symbol_error_counter": 1_000_000,
		"link_downed":          4,
	}), "mlx5_0", "1")
	after := readPortCounters(countersFixture(t, map[string]int64{
		"symbol_error_counter": 1_000_003,
		"link_downed":          4,
	}), "mlx5_0", "1")

	d, clean := delta(before, after)
	if !clean {
		t.Fatal("a forward-moving counter was reported as a reset")
	}
	if d["symbol_error_counter"] != 3 {
		t.Errorf("symbol delta = %d, want 3", d["symbol_error_counter"])
	}
	if d["link_downed"] != 0 {
		t.Errorf("link_downed delta = %d, want 0 — it did not move", d["link_downed"])
	}
	if d.total() != 3 {
		t.Errorf("total = %d, want 3", d.total())
	}
}

func TestACounterThatWentBackwardsIsAResetNotMinusFour(t *testing.T) {
	// A port bounce or driver reload zeroes these. Clamping to zero would
	// silently turn a link that bounced mid-soak into a clean run.
	before := readPortCounters(countersFixture(t, map[string]int64{"link_downed": 9}), "mlx5_0", "1")
	after := readPortCounters(countersFixture(t, map[string]int64{"link_downed": 0}), "mlx5_0", "1")

	if _, clean := delta(before, after); clean {
		t.Error("a counter that decreased was treated as a valid delta")
	}
}

func TestAnUnreadableCounterIsOmittedNotZero(t *testing.T) {
	// A zero delta says "this counter did not move"; an absence says "this
	// counter was never read". Only the first is evidence a link is healthy.
	got := readPortCounters(filepath.Join(t.TempDir(), "nope"), "mlx5_0", "1")
	if len(got) != 0 {
		t.Errorf("read %v from an absent sysfs", got)
	}

	// And a counter missing at one end is not differenced against nothing.
	before := readPortCounters(countersFixture(t, map[string]int64{
		"symbol_error_counter": 5, "link_downed": 1,
	}), "mlx5_0", "1")
	after := readPortCounters(countersFixture(t, map[string]int64{"symbol_error_counter": 7}), "mlx5_0", "1")

	d, clean := delta(before, after)
	if !clean {
		t.Fatal("unexpected reset")
	}
	if _, present := d["link_downed"]; present {
		t.Error("a counter absent from the final reading was still differenced")
	}
	if d["symbol_error_counter"] != 2 {
		t.Errorf("symbol delta = %d, want 2", d["symbol_error_counter"])
	}
}
