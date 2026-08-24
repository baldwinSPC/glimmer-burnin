// SPDX-License-Identifier: Apache-2.0
// Copyright the Glimmer authors.

package main

import (
	"fmt"
	"strconv"
	"strings"
)

// Defaults.
//
// defaultDurationSeconds is deliberately much larger than memory-stress's own
// 600s default: memtest86+-style bit-fade faults are a function of how long a
// cell sits UNREFRESHED-BY-ACCESS, not of how much traffic crosses it, and a
// short hold proves nothing about a defect that only shows up after minutes.
// This is a fallback for the rare bare-CLI invocation with no
// BURNIN_DURATION_SECONDS at all; the operator always sets one.
const (
	defaultDurationSeconds = 3600
	defaultFraction        = 0.50
	defaultMinMB           = 256

	mib int64 = 1 << 20
)

// Reserve, in seconds: time held back from the hold budget for allocating,
// filling and verifying the region — twice, once per pattern. That work is
// not free on a large region (a full scan at a few GB/s is still seconds on a
// multi-GiB buffer), and charging it against the hold budget would silently
// shrink what the operator asked to hold for.
const (
	minReserveSeconds = 15
	maxReserveSeconds = 120
	// hardCapGraceSeconds mirrors memory-stress's own: how long past
	// BURNIN_DURATION_SECONDS the wrapper lets itself live before ending the
	// run on its own terms, inside the pod's activeDeadlineSeconds grace, so a
	// wedged run's metrics still reach stdout instead of being taken by a
	// kubelet SIGKILL.
	hardCapGraceSeconds = 60
)

// minHoldSeconds is the floor a PER-PATTERN hold must clear, whichever
// surface set it (auto-derived from BURNIN_DURATION_SECONDS, or an explicit
// BURNIN_RETENTION_HOLD_SECONDS). Below it this is not a retention test at
// all — it is an immediate verify, which runners/memory-stress already
// covers a different way — so this refuses rather than silently running a
// test that answers a different question than the one its own name promises.
//
// A variable rather than a constant only so a test can shrink it — the same
// reason memory-stress's own hardCapGrace is a var: the floor's ENFORCEMENT
// is what a test needs to exercise, not 30 real seconds of waiting for it.
var minHoldSeconds = 30

// patternBytes are the two fill values, in order: a value and its bitwise
// complement over the SAME memory cells, the shape memtest86+'s own bit-fade
// test uses. Testing a cell stuck-at-1 needs a pattern with a 0 there; testing
// stuck-at-0 needs the opposite — one pattern alone can only ever find half
// the possible stuck-bit faults.
var patternBytes = [2]byte{0xFF, 0x00}

// config is the runner's tunable surface. Every field has a default that is
// sound on an untuned node; nothing here is required.
type config struct {
	durationSeconds int
	// regionMB, when > 0, is an explicit region size and overrides
	// auto-sizing entirely — obeyed as asked, not clamped, the same rule
	// memory-stress's BURNIN_MEMORY_MB follows and for the same reason:
	// clamping would silently test less memory than the profile says it does.
	regionMB int64
	// fraction of the usable memory budget to allocate when auto-sizing. The
	// remainder is headroom for the page cache, the runtime's own bookkeeping
	// and whatever else shares the cgroup — allocating the whole budget gets
	// the container OOM-killed, which reports as an infrastructure Error and
	// spends the run's time without judging the hardware.
	fraction float64
	minMB    int64
	// holdSeconds, when > 0, is an explicit PER-PATTERN hold and overrides the
	// budget-derived one entirely.
	holdSeconds int
}

type getenvFunc func(string) (string, bool)

func loadConfig(getenv getenvFunc) (config, error) {
	cfg := config{fraction: defaultFraction, minMB: defaultMinMB}

	duration, err := intVar(getenv, "BURNIN_DURATION_SECONDS", defaultDurationSeconds, 1, 1<<31-1)
	if err != nil {
		return config{}, err
	}
	cfg.durationSeconds = int(duration)

	if cfg.regionMB, err = intVar(getenv, "BURNIN_RETENTION_MB", 0, 0, 1<<40); err != nil {
		return config{}, err
	}
	if cfg.fraction, err = floatVar(getenv, "BURNIN_RETENTION_FRACTION", defaultFraction, 0.01, 1.0); err != nil {
		return config{}, err
	}
	if cfg.minMB, err = intVar(getenv, "BURNIN_RETENTION_MIN_MB", defaultMinMB, 1, 1<<40); err != nil {
		return config{}, err
	}
	hold, err := intVar(getenv, "BURNIN_RETENTION_HOLD_SECONDS", 0, 0, 1<<31-1)
	if err != nil {
		return config{}, err
	}
	cfg.holdSeconds = int(hold)

	return cfg, nil
}

// intVar reads an integer environment variable.
//
// A malformed or out-of-range value is an error rather than a silent fall
// back to the default — a typo here must be found at the first run, not
// discovered months later as a fleet accepted on a default nobody chose.
func intVar(getenv getenvFunc, name string, def, minimum, maximum int64) (int64, error) {
	raw, ok := getenv(name)
	if !ok || strings.TrimSpace(raw) == "" {
		return def, nil
	}
	v, err := strconv.ParseInt(strings.TrimSpace(raw), 10, 64)
	if err != nil {
		return 0, fmt.Errorf("%s=%q is not an integer", name, raw)
	}
	if v < minimum || v > maximum {
		return 0, fmt.Errorf("%s=%d is outside the accepted range [%d, %d]", name, v, minimum, maximum)
	}
	return v, nil
}

func floatVar(getenv getenvFunc, name string, def, minimum, maximum float64) (float64, error) {
	raw, ok := getenv(name)
	if !ok || strings.TrimSpace(raw) == "" {
		return def, nil
	}
	v, err := strconv.ParseFloat(strings.TrimSpace(raw), 64)
	if err != nil {
		return 0, fmt.Errorf("%s=%q is not a number", name, raw)
	}
	if v < minimum || v > maximum {
		return 0, fmt.Errorf("%s=%v is outside the accepted range [%v, %v]", name, v, minimum, maximum)
	}
	return v, nil
}

// plan is the concrete test one execution will run.
type plan struct {
	regionMB int64
	// holdSeconds is PER PATTERN — the same value is held for both
	// patternBytes entries.
	holdSeconds int
	// explicitRegion/explicitHold record which surface chose each value, so
	// the log can say why a floor did or did not apply.
	explicitRegion bool
	explicitHold   bool
}

// reserveSeconds is how much of the duration budget is held back from the
// hold for allocating, filling and verifying the region — across BOTH
// patterns, since the operation happens twice.
func reserveSeconds(duration int) int {
	r := duration / 10
	if r < minReserveSeconds {
		r = minReserveSeconds
	}
	if r > maxReserveSeconds {
		r = maxReserveSeconds
	}
	return r
}

// planRun decides the test region and the per-pattern hold, or refuses with a
// verdict-carrying error.
func planRun(cfg config, sys sysInfo) (plan, error) {
	var p plan

	budget := sys.memAvailableBytes
	if sys.limitBytes > 0 && sys.limitBytes < budget {
		budget = sys.limitBytes
	}

	if cfg.regionMB > 0 {
		p.regionMB = cfg.regionMB
		p.explicitRegion = true
	} else {
		p.regionMB = int64(float64(budget)*cfg.fraction) / mib
		if p.regionMB < cfg.minMB {
			return plan{}, tooSmallRegion(cfg, sys, p.regionMB)
		}
	}

	if cfg.holdSeconds > 0 {
		if cfg.holdSeconds < minHoldSeconds {
			return plan{}, &runnerError{code: exitError, msg: fmt.Sprintf(
				"BURNIN_RETENTION_HOLD_SECONDS=%d is below the %ds floor — a shorter hold is an "+
					"immediate verify, not a retention test; runners/memory-stress already covers that",
				cfg.holdSeconds, minHoldSeconds)}
		}
		p.holdSeconds = cfg.holdSeconds
		p.explicitHold = true
	} else {
		reserve := reserveSeconds(cfg.durationSeconds)
		remaining := cfg.durationSeconds - reserve
		perPattern := remaining / len(patternBytes)
		if perPattern < minHoldSeconds {
			return plan{}, &runnerError{code: exitError, msg: fmt.Sprintf(
				"BURNIN_DURATION_SECONDS=%d leaves only %ds per pattern after a %ds allocate/fill/verify "+
					"reserve across %d patterns — below the %ds floor a hold needs to test retention rather "+
					"than merely verify a write; raise BURNIN_DURATION_SECONDS or set "+
					"BURNIN_RETENTION_HOLD_SECONDS explicitly",
				cfg.durationSeconds, perPattern, reserve, len(patternBytes), minHoldSeconds)}
		}
		p.holdSeconds = perPattern
	}

	return p, nil
}

// tooSmallRegion decides what "there is not enough memory to test" means.
//
// The two cases are not the same verdict and must not be collapsed — the
// same split memory-stress's own tooSmall makes, and for the same reason: a
// cgroup limit in force means somebody configured this pod and the node's own
// memory is untouched by the question (an Error, naming the knob to turn); no
// limit and a genuinely small machine means the test does not apply (a Skip).
func tooSmallRegion(cfg config, sys sysInfo, regionMB int64) error {
	if sys.limitBytes > 0 {
		return &runnerError{code: exitError, msg: fmt.Sprintf(
			"the pod's memory limit of %d MiB leaves only %d MiB testable, below the %d MiB minimum "+
				"test region — raise spec.resources.limits.memory or set BURNIN_RETENTION_MB; this "+
				"node's memory was not judged",
			sys.limitBytes/mib, regionMB, cfg.minMB)}
	}
	return &runnerError{code: exitSkip, msg: fmt.Sprintf(
		"node has %d MiB of usable host memory, which leaves %d MiB testable — below the %d MiB "+
			"minimum test region",
		sys.memAvailableBytes/mib, regionMB, cfg.minMB)}
}

// runnerError is an error that already knows which exit code it deserves.
type runnerError struct {
	code int
	msg  string
}

func (e *runnerError) Error() string { return e.msg }
