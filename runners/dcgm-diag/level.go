package main

import (
	"fmt"
	"strconv"
	"strings"
	"time"
)

// levelBudgets is the wall-clock DCGM budgets for each of its four run levels
// (`dcgmi diag -r`). Level 1 is the NVML-only deployment check, 2 adds the
// short CUDA-backed tests, 3 the long ones, 4 the extended production suite.
//
// They are used two ways: to derive a level from the time this test was given,
// and to bound the diagnostic when no duration was supplied at all. Index 0 is
// unused so the index IS the level.
var levelBudgets = [5]time.Duration{
	1: 1 * time.Minute,
	2: 2 * time.Minute,
	3: 15 * time.Minute,
	4: 60 * time.Minute,
}

// namedLevels are the spellings DCGM itself accepts for -r, so an operator who
// knows dcgmi can write what they already know.
var namedLevels = map[string]int{
	"short":  1,
	"quick":  1,
	"medium": 2,
	"long":   3,
	"xlong":  4,
}

func levelBudget(level int) time.Duration {
	if level < 1 || level >= len(levelBudgets) {
		return levelBudgets[1]
	}
	return levelBudgets[level]
}

// resolveLevel decides which DCGM run level to ask for.
//
// An explicit BURNIN_DCGM_LEVEL always wins, including when it does not fit the
// duration: the operator asked for a specific level of scrutiny, and quietly
// substituting a weaker one would turn a level-4 acceptance gate into a
// level-1 one with nothing in the output saying so. The runner bounds itself at
// the duration instead, so a level that does not fit ends as a legible Error
// rather than a fabricated pass.
//
// Otherwise the level is derived from the time the test was given, choosing the
// highest level whose whole budget fits. A level chosen with no headroom would
// be killed at the pod's activeDeadlineSeconds, which reports Error — no
// verdict, and a wasted burn-in slot.
//
// The second return says where the level came from; it is emitted as evidence,
// because "why did this node run a level 1?" is otherwise unanswerable from the
// log alone.
func resolveLevel(explicit string, duration time.Duration) (int, string, error) {
	explicit = strings.ToLower(strings.TrimSpace(explicit))
	if explicit != "" {
		if lvl, ok := namedLevels[explicit]; ok {
			return lvl, "explicit", nil
		}
		lvl, err := strconv.Atoi(explicit)
		if err != nil || lvl < 1 || lvl > 4 {
			return 0, "", fmt.Errorf(
				"BURNIN_DCGM_LEVEL=%q is not a DCGM run level: use 1-4, or short/medium/long/xlong",
				explicit)
		}
		return lvl, "explicit", nil
	}
	return deriveLevel(duration), "derived", nil
}

// deriveLevel picks the deepest level whose budget fits in the time available.
func deriveLevel(duration time.Duration) int {
	for lvl := 4; lvl >= 2; lvl-- {
		if duration >= levelBudgets[lvl] {
			return lvl
		}
	}
	// Level 1 is the floor rather than "run nothing": the deployment checks are
	// cheap, and a node that is misconfigured (denylisted device, wrong driver,
	// no persistence mode) is worth catching even in a minute.
	return 1
}
