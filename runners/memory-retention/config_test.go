// SPDX-License-Identifier: Apache-2.0
// Copyright the Glimmer authors.

package main

import "testing"

// env builds a getenvFunc over a map, mirroring runners/memory-stress's own
// helper: a test never touches the process environment, and cases stay
// independent of each other.
func env(pairs map[string]string) getenvFunc {
	return func(name string) (string, bool) {
		v, ok := pairs[name]
		return v, ok
	}
}

func TestLoadConfigDefaults(t *testing.T) {
	cfg, err := loadConfig(env(nil))
	if err != nil {
		t.Fatalf("loadConfig with an empty environment: %v", err)
	}
	if cfg.durationSeconds != defaultDurationSeconds {
		t.Errorf("durationSeconds = %d, want the default %d", cfg.durationSeconds, defaultDurationSeconds)
	}
	if cfg.regionMB != 0 {
		t.Errorf("regionMB = %d, want 0 (auto-size)", cfg.regionMB)
	}
	if cfg.fraction != defaultFraction {
		t.Errorf("fraction = %v, want the default %v", cfg.fraction, defaultFraction)
	}
}

func TestLoadConfigRejectsGarbage(t *testing.T) {
	cases := map[string]map[string]string{
		"non-integer duration":       {"BURNIN_DURATION_SECONDS": "soon"},
		"non-integer region":         {"BURNIN_RETENTION_MB": "lots"},
		"fraction out of range":      {"BURNIN_RETENTION_FRACTION": "1.5"},
		"negative hold":              {"BURNIN_RETENTION_HOLD_SECONDS": "-5"},
		"region below allowed range": {"BURNIN_RETENTION_MB": "-1"},
	}
	for name, vars := range cases {
		t.Run(name, func(t *testing.T) {
			if _, err := loadConfig(env(vars)); err == nil {
				t.Errorf("loadConfig accepted %v", vars)
			}
		})
	}
}

func TestPlanRunAutoSizes(t *testing.T) {
	cfg, err := loadConfig(env(nil))
	if err != nil {
		t.Fatal(err)
	}
	sys := sysInfo{memTotalBytes: 128 * 1024 * mib, memAvailableBytes: 100 * 1024 * mib}
	p, err := planRun(cfg, sys)
	if err != nil {
		t.Fatalf("planRun: %v", err)
	}
	wantMB := int64(float64(100*1024) * defaultFraction)
	if p.regionMB != wantMB {
		t.Errorf("regionMB = %d, want %d (%.0f%% of available)", p.regionMB, wantMB, defaultFraction*100)
	}
	if p.explicitRegion {
		t.Error("explicitRegion = true for an auto-sized plan")
	}
}

func TestPlanRunObeysExplicitRegionEvenAboveTheCgroupLimit(t *testing.T) {
	// The same rule memory-stress's own BURNIN_MEMORY_MB follows: an explicit
	// size is obeyed, not clamped, because clamping would silently test less
	// memory than the profile says it does. The pod either has the room or
	// the kernel OOM-kills it, which is a real, visible signal — not a
	// smaller number quietly substituted for what was asked.
	cfg, err := loadConfig(env(map[string]string{"BURNIN_RETENTION_MB": "99999"}))
	if err != nil {
		t.Fatal(err)
	}
	sys := sysInfo{memTotalBytes: 1024 * mib, memAvailableBytes: 512 * mib, limitBytes: 256 * mib}
	p, err := planRun(cfg, sys)
	if err != nil {
		t.Fatalf("planRun: %v", err)
	}
	if p.regionMB != 99999 || !p.explicitRegion {
		t.Errorf("regionMB = %d (explicit=%v), want 99999 (explicit=true), unclamped", p.regionMB, p.explicitRegion)
	}
}

func TestPlanRunRefusesATooSmallRegionUnderACgroupLimit(t *testing.T) {
	cfg, err := loadConfig(env(nil))
	if err != nil {
		t.Fatal(err)
	}
	sys := sysInfo{memTotalBytes: 8 * 1024 * mib, memAvailableBytes: 100 * mib, limitBytes: 100 * mib}
	_, err = planRun(cfg, sys)
	re, ok := err.(*runnerError)
	if !ok || re.code != exitError {
		t.Fatalf("planRun under a tight cgroup limit = %v, want an Error naming the limit", err)
	}
}

func TestPlanRunSkipsATooSmallMachine(t *testing.T) {
	cfg, err := loadConfig(env(nil))
	if err != nil {
		t.Fatal(err)
	}
	// No cgroup limit at all: the machine itself is just small.
	sys := sysInfo{memTotalBytes: 100 * mib, memAvailableBytes: 100 * mib}
	_, err = planRun(cfg, sys)
	re, ok := err.(*runnerError)
	if !ok || re.code != exitSkip {
		t.Fatalf("planRun on a genuinely small machine = %v, want a Skip", err)
	}
}

func TestPlanRunDerivesHoldFromDuration(t *testing.T) {
	cfg, err := loadConfig(env(map[string]string{"BURNIN_DURATION_SECONDS": "1000"}))
	if err != nil {
		t.Fatal(err)
	}
	sys := sysInfo{memTotalBytes: 128 * 1024 * mib, memAvailableBytes: 100 * 1024 * mib}
	p, err := planRun(cfg, sys)
	if err != nil {
		t.Fatalf("planRun: %v", err)
	}
	// reserve = max(minReserveSeconds, min(1000/10, maxReserveSeconds)) = 100
	// (1000-100)/2 patterns = 450 per pattern.
	if p.holdSeconds != 450 {
		t.Errorf("holdSeconds = %d, want 450", p.holdSeconds)
	}
	if p.explicitHold {
		t.Error("explicitHold = true for a duration-derived plan")
	}
}

func TestPlanRunRefusesADurationTooShortForAMeaningfulHold(t *testing.T) {
	cfg, err := loadConfig(env(map[string]string{"BURNIN_DURATION_SECONDS": "40"}))
	if err != nil {
		t.Fatal(err)
	}
	sys := sysInfo{memTotalBytes: 128 * 1024 * mib, memAvailableBytes: 100 * 1024 * mib}
	_, err = planRun(cfg, sys)
	re, ok := err.(*runnerError)
	if !ok || re.code != exitError {
		t.Fatalf("planRun with too little duration = %v, want an Error, not a silently raised floor "+
			"(raising here risks overrunning the pod deadline the operator computed from the original request)", err)
	}
}

func TestPlanRunRefusesAnExplicitHoldBelowTheFloor(t *testing.T) {
	cfg, err := loadConfig(env(map[string]string{"BURNIN_RETENTION_HOLD_SECONDS": "5"}))
	if err != nil {
		t.Fatal(err)
	}
	sys := sysInfo{memTotalBytes: 128 * 1024 * mib, memAvailableBytes: 100 * 1024 * mib}
	_, err = planRun(cfg, sys)
	re, ok := err.(*runnerError)
	if !ok || re.code != exitError {
		t.Fatalf("planRun with BURNIN_RETENTION_HOLD_SECONDS=5 = %v, want an Error below the %ds floor", err, minHoldSeconds)
	}
}

func TestPlanRunObeysAnExplicitHoldAtOrAboveTheFloor(t *testing.T) {
	cfg, err := loadConfig(env(map[string]string{"BURNIN_RETENTION_HOLD_SECONDS": "45"}))
	if err != nil {
		t.Fatal(err)
	}
	sys := sysInfo{memTotalBytes: 128 * 1024 * mib, memAvailableBytes: 100 * 1024 * mib}
	p, err := planRun(cfg, sys)
	if err != nil {
		t.Fatalf("planRun: %v", err)
	}
	if p.holdSeconds != 45 || !p.explicitHold {
		t.Errorf("holdSeconds = %d (explicit=%v), want 45 (explicit=true)", p.holdSeconds, p.explicitHold)
	}
}
