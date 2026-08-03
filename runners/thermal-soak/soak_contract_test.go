// SPDX-License-Identifier: Apache-2.0
// Copyright the Glimmer authors.

// Package thermalsoak holds no Go code. It exists so that the seam between the
// two CUDA soak runners and the operator has a test that CI runs.
//
// Both runners are C++; their sources cannot import pkg/runner and so cannot
// check themselves the way runners/host-health and runners/memory-stress do.
// What is checkable without a GPU is the part that actually breaks silently:
// the KEY VOCABULARY. A runner prints its own key names, pkg/runner maps them
// to the canonical names a threshold is written against, and nothing at runtime
// reconciles a disagreement. Two keys that normalise onto one canonical name
// silently discard a measurement — parsing is last-occurrence-wins — and the
// symptom is a node nobody can accept for reasons nobody can see, discovered
// after the hardware is no longer under test.
//
// So these tests read the runners' own sources, extract every `key=` literal
// they can emit, and run the lot through the real parser. That is why every
// metric key in soak_core.cuh and in the two mains is written as a string
// literal ending in '=' and printed with "%s": it makes the emitted vocabulary
// mechanically extractable rather than a thing a reviewer has to keep in their
// head.
package thermalsoak

import (
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"testing"

	"github.com/baldwinSPC/glimmer-burnin/pkg/contract"
	"github.com/baldwinSPC/glimmer-burnin/pkg/runner"
)

const (
	thermalSoakDir = "."
	gpuBurnDir     = "../gpu-burn"
)

// sharedSources are byte-identical in both runner directories. They are copies
// rather than a shared path because the publish-runner workflow builds each
// runner with its OWN directory as the Docker build context, and COPY cannot
// reach outside a build context.
var sharedSources = []string{"soak_core.cuh", "nvml_dynamic.h"}

// TestSharedSoakSourcesAreIdentical is what makes "two kinds, one
// implementation" a fact rather than an intention.
//
// The two runners are only comparable — same load, same sampler, so
// sustainedThroughputTflops means the same thing in both — for as long as the
// copies agree. A fix applied to one copy and not the other would leave two
// tests silently measuring different things under one set of metric names,
// which is precisely the failure pkg/contract's naming rules exist to prevent.
func TestSharedSoakSourcesAreIdentical(t *testing.T) {
	for _, name := range sharedSources {
		a, err := os.ReadFile(filepath.Join(thermalSoakDir, name))
		if err != nil {
			t.Fatalf("reading thermal-soak/%s: %v", name, err)
		}
		b, err := os.ReadFile(filepath.Join(gpuBurnDir, name))
		if err != nil {
			t.Fatalf("reading gpu-burn/%s: %v", name, err)
		}
		if string(a) != string(b) {
			t.Errorf("%s has drifted between runners/thermal-soak and runners/gpu-burn (%d vs %d bytes); "+
				"copy one over the other", name, len(a), len(b))
		}
	}
}

// emittedKeyPattern finds every metric key a runner can print. The '=' is part
// of the literal by construction (see the package comment), so this finds the
// keys from the shared engine, the per-kind Keys tables and the one-off printfs
// alike.
var emittedKeyPattern = regexp.MustCompile(`"([a-z][a-zA-Z0-9_]*)=`)

func emittedKeys(t *testing.T, dir string, files ...string) []string {
	t.Helper()
	seen := map[string]bool{}
	for _, f := range files {
		src, err := os.ReadFile(filepath.Join(dir, f))
		if err != nil {
			t.Fatalf("reading %s/%s: %v", dir, f, err)
		}
		for _, m := range emittedKeyPattern.FindAllStringSubmatch(string(src), -1) {
			seen[m[1]] = true
		}
	}
	keys := make([]string, 0, len(seen))
	for k := range seen {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	if len(keys) == 0 {
		t.Fatalf("no metric keys found in %s; the extraction pattern no longer matches the source", dir)
	}
	return keys
}

// canonicalise maps each raw key to the canonical name the operator will store
// it under, one key at a time so that a collision is visible instead of being
// silently resolved by last-occurrence-wins.
func canonicalise(t *testing.T, kind string, keys []string) map[string]string {
	t.Helper()
	out := map[string]string{}
	for _, raw := range keys {
		res := runner.Parse(kind, raw+"=1", 0)
		if len(res.InvalidNames) != 0 {
			t.Errorf("%s: key %q normalises to a name the contract rejects: %v", kind, raw, res.InvalidNames)
			continue
		}
		if len(res.Metrics) != 1 {
			t.Errorf("%s: key %q parsed to %d metrics, want exactly 1: %v", kind, raw, len(res.Metrics), res.Metrics)
			continue
		}
		for name := range res.Metrics {
			out[raw] = name
		}
	}
	return out
}

func TestEmittedKeysObeyTheContract(t *testing.T) {
	cases := []struct {
		kind  string
		dir   string
		files []string
	}{
		{"thermal-soak", thermalSoakDir, []string{"thermal_soak.cu", "soak_core.cuh", "nvml_dynamic.h"}},
		{"gpu-burn", gpuBurnDir, []string{"gpu_burn.cu", "soak_core.cuh", "nvml_dynamic.h"}},
	}

	for _, c := range cases {
		byRaw := canonicalise(t, c.kind, emittedKeys(t, c.dir, c.files...))

		// Every canonical name must survive the grammar. A name rejected at
		// delivery time fails after the run has finished and the hardware is no
		// longer under test.
		for raw, name := range byRaw {
			if err := contract.ValidateMetricName(name); err != nil {
				t.Errorf("%s: %q -> %q: %v", c.kind, raw, name, err)
			}
		}

		// The collision check, and the reason this test exists. Two of this
		// runner's keys landing on one canonical name would throw one of two
		// real measurements away, quietly.
		owner := map[string]string{}
		for _, raw := range sortedKeys(byRaw) {
			name := byRaw[raw]
			if first, clash := owner[name]; clash {
				t.Errorf("%s: keys %q and %q both normalise to %q; parsing is last-occurrence-wins, "+
					"so one of the two measurements would be silently discarded", c.kind, first, raw, name)
				continue
			}
			owner[name] = raw
		}

		// A dimensional quantity that loses its unit suffix is stored as a bare
		// number: it charts happily and compares against nothing. This is the
		// trap that makes "..._mhz" unusable without an alias, and it is why the
		// clock keys are emitted in lowerCamelCase instead.
		for _, name := range []string{"smClockMHz", "memClockMHz", "ratedBoostClockMHz"} {
			if contract.UnitOf(name) != contract.UnitMegahertz {
				t.Errorf("%s: %q no longer declares MHz", c.kind, name)
			}
			if !containsValue(byRaw, name) {
				t.Errorf("%s: clock metric %q is not emitted; a soak that reports no clock cannot "+
					"support a sustained-clock gate", c.kind, name)
			}
		}
	}
}

// TestBothKindsReportTheSameMeasurands is the "two kinds, two assertion sets,
// one implementation" invariant, stated where it can fail loudly.
//
// The two runners share their entire engine, so every measurand one produces
// the other must produce too — under a DIFFERENT raw key, because the operator's
// alias tables already pin different spellings, but onto the SAME canonical
// name. If that ever stops being true, a fleet charting one metric across both
// kinds is charting two different quantities.
func TestBothKindsReportTheSameMeasurands(t *testing.T) {
	thermal := canonicalise(t, "thermal-soak",
		emittedKeys(t, thermalSoakDir, "thermal_soak.cu", "soak_core.cuh"))
	burn := canonicalise(t, "gpu-burn",
		emittedKeys(t, gpuBurnDir, "gpu_burn.cu", "soak_core.cuh"))

	// The measurands the engine produces for both kinds. Anything specific to
	// one kind's own assertions (thermal-soak's clock floor and temperature
	// ceiling) is deliberately not here.
	shared := []string{
		"elapsedS", "iterationsCompleted", "miscompares", "sdcDetections",
		"nonfiniteCount", "sustainedThroughputTflops", "sustainedClockPct",
		"gpuTempC", "powerDrawW", "throttleEvents",
	}
	for _, name := range shared {
		if !containsValue(thermal, name) {
			t.Errorf("thermal-soak does not produce %q", name)
		}
		if !containsValue(burn, name) {
			t.Errorf("gpu-burn does not produce %q", name)
		}
	}

	// And the spellings really are different, so this is not a vacuous claim:
	// the same measurand arrives under gpu_burn's own vocabulary on one side and
	// the soak vocabulary on the other.
	for _, name := range []string{"miscompares", "iterationsCompleted", "elapsedS", "gpuTempC"} {
		if rawFor(thermal, name) == rawFor(burn, name) {
			t.Errorf("%q is emitted under the same raw key by both kinds (%q); the alias tables in "+
				"pkg/runner exist because the two kinds' vocabularies differ", name, rawFor(thermal, name))
		}
	}
}

// TestEccIsDeclaredUnmeasurableNotZeroed pins the GB10 path.
//
// GB10's unified LPDDR5X has on-die ECC that NVML cannot see, so
// nvmlDeviceGetEccMode answers NOT_SUPPORTED and there is no counter to read.
// The runner emits the reserved "n/a" value to DECLARE that, which lands in
// Result.Unmeasurable and never in Result.Metrics. Emitting 0 instead would be
// a lie about a quantity nobody measured, and omitting the key would fail every
// healthy Spark in the fleet against an `eccErrors Equal 0` threshold.
func TestEccIsDeclaredUnmeasurableNotZeroed(t *testing.T) {
	for _, kind := range []string{"thermal-soak", "gpu-burn"} {
		res := runner.Parse(kind, "ecc_errors=n/a\nGPU_OK\n", 0)
		if !res.IsUnmeasurable("eccErrors") {
			t.Errorf("%s: ecc_errors=n/a was not recorded as unmeasurable: %v", kind, res.Unmeasurable)
		}
		if v, stored := res.Metrics["eccErrors"]; stored {
			t.Errorf("%s: eccErrors=%q leaked into Metrics; the sentinel denies a measurement", kind, v)
		}

		// The other half of the rule: a part that HAS ECC reports a number, and
		// that number must land as an ordinary metric.
		measured := runner.Parse(kind, "ecc_errors=0\n", 0)
		if measured.IsUnmeasurable("eccErrors") {
			t.Errorf("%s: a measured zero was treated as a declaration of unmeasurability", kind)
		}
		if measured.Metrics["eccErrors"] != "0" {
			t.Errorf("%s: eccErrors = %v, want a stored 0", kind, measured.Metrics)
		}
	}
}

// TestAcceptanceMetricsAreSafeToThresholdOn guards the profiles the READMEs
// recommend. A threshold on a metric pkg/contract marks as evidence-only is one
// that passes or fails on noise while looking exactly like a hardware
// judgement.
func TestAcceptanceMetricsAreSafeToThresholdOn(t *testing.T) {
	for _, name := range []string{
		"miscompares", "sdcDetections", "nonfiniteCount", "throttleEvents",
		"sustainedClockPct", "eccErrors", "gpuTempC", "elapsedS",
		"iterationsCompleted", "sustainedThroughputTflops",
	} {
		if !contract.SafeToThresholdOn(name) {
			t.Errorf("%q is recommended as an acceptance threshold but pkg/contract marks it "+
				"unsafe to threshold on", name)
		}
	}
}

func sortedKeys(m map[string]string) []string {
	out := make([]string, 0, len(m))
	for k := range m {
		out = append(out, k)
	}
	sort.Strings(out)
	return out
}

func containsValue(m map[string]string, want string) bool {
	for _, v := range m {
		if v == want {
			return true
		}
	}
	return false
}

func rawFor(m map[string]string, want string) string {
	for k, v := range m {
		if v == want {
			return k
		}
	}
	return ""
}
