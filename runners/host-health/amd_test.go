// SPDX-License-Identifier: Apache-2.0
// Copyright the Glimmer authors.

package main

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// amdCard describes one fixture device.
type amdCard struct {
	name   string
	driver string
	// ras maps a block file name to its contents. A nil map means NO ras/
	// directory at all, which is the unmeasurable case.
	ras map[string]string
}

func amdFixture(t *testing.T, cards []amdCard) string {
	t.Helper()
	root := t.TempDir()
	for _, c := range cards {
		device := filepath.Join(root, "class", "drm", c.name, "device")
		if err := os.MkdirAll(device, 0o755); err != nil {
			t.Fatal(err)
		}
		if c.driver != "" {
			target := filepath.Join(root, "bus", "pci", "drivers", c.driver)
			if err := os.MkdirAll(target, 0o755); err != nil {
				t.Fatal(err)
			}
			if err := os.Symlink(target, filepath.Join(device, "driver")); err != nil {
				t.Fatal(err)
			}
		}
		if c.ras == nil {
			continue
		}
		rasDir := filepath.Join(device, "ras")
		if err := os.MkdirAll(rasDir, 0o755); err != nil {
			t.Fatal(err)
		}
		for name, body := range c.ras {
			if err := os.WriteFile(filepath.Join(rasDir, name), []byte(body), 0o644); err != nil {
				t.Fatal(err)
			}
		}
	}
	return root
}

func clean(ue, ce string) map[string]string {
	return map[string]string{
		"umc_err_count":  "ue: " + ue + "\nce: " + ce + "\n",
		"gfx_err_count":  "ue: 0\nce: 0\n",
		"sdma_err_count": "ue: 0\nce: 0\n",
	}
}

func TestAHealthyAMDNodeReportsCountersThatWereActuallyRead(t *testing.T) {
	root := amdFixture(t, []amdCard{
		{name: "card0", driver: "amdgpu", ras: clean("0", "3")},
		{name: "card1", driver: "amdgpu", ras: clean("0", "1")},
	})

	s := probeAMD(root)
	if !s.isAMDNode() || s.cards != 2 || s.withRAS != 2 {
		t.Fatalf("sample = %+v", s)
	}
	if !s.read {
		t.Fatalf("a complete fixture was not read: %+v", s)
	}
	if s.uncorrected != 0 || s.corrected != 4 {
		t.Errorf("ue=%d ce=%d, want 0 and 4 summed across both cards", s.uncorrected, s.corrected)
	}
	if got := amdState(s); got != counterOK {
		t.Errorf("state = %v, want counterOK", got)
	}

	out := newEmitter()
	emitAMD(out, probeAMD(root), s)
	if v, ok := out.get(keyECCErrors); !ok || v != "0" {
		t.Errorf("%s = %q (present=%v) — a clean node's zero IS a measurement", keyECCErrors, v, ok)
	}
	if v, _ := out.get("ecc_corrected_aggregate"); v != "4" {
		t.Errorf("corrected = %q, want 4 as evidence rather than folded into the gate", v)
	}
	if v, _ := out.get("ecc_source"); v != "amdgpu-sysfs" {
		t.Errorf("ecc_source = %q", v)
	}
}

func TestUncorrectedErrorsReachTheGatedCounter(t *testing.T) {
	// The point of the whole change: on an AMD node this metric was absent, so
	// nothing was gated. `ue` is the same measurand the NVIDIA path puts in
	// eccErrors — uncorrected, aggregate — not merely a similarly-named one.
	root := amdFixture(t, []amdCard{{name: "card0", driver: "amdgpu", ras: clean("7", "12")}})

	out := newEmitter()
	s := probeAMD(root)
	emitAMD(out, s, s)

	if v, _ := out.get(keyECCErrors); v != "7" {
		t.Errorf("%s = %q, want the 7 uncorrected errors", keyECCErrors, v)
	}
	if v, _ := out.get("ecc_corrected_aggregate"); v != "12" {
		t.Errorf("corrected errors should be evidence, not part of the gate: %q", v)
	}
}

func TestAnAMDPartWithNoRASBlockDeclaresItRatherThanReportingZero(t *testing.T) {
	// The narrow `n/a` case: the driver is loaded and exposes no ras/ at all,
	// which is the ASIC saying it has no RAS block. The AMD analogue of GB10
	// answering [N/A] for ecc.errors — a positive fact about the part, so a
	// RequiredIfMeasurable gate is reported not-evaluated instead of failing a
	// healthy node.
	root := amdFixture(t, []amdCard{
		{name: "card0", driver: "amdgpu", ras: nil},
		{name: "card1", driver: "amdgpu", ras: nil},
	})

	s := probeAMD(root)
	if got := amdState(s); got != counterUnmeasurable {
		t.Fatalf("state = %v, want counterUnmeasurable", got)
	}

	out := newEmitter()
	emitAMD(out, s, s)
	v, ok := out.get(keyECCErrors)
	if !ok {
		t.Fatal("nothing was emitted; the sentinel is a claim worth making")
	}
	if v != unmeasurableValue {
		t.Errorf("%s = %q, want %q", keyECCErrors, v, unmeasurableValue)
	}
}

func TestACounterThatCouldNotBeReadIsOmittedAndExplained(t *testing.T) {
	// Absence is not a declaration. A file this runner cannot parse means the
	// node total is unknown, so the gated counter is OMITTED and a gate on it
	// fails the node — never `n/a`, which would claim the hardware has nothing
	// to report, and never 0.
	broken := clean("0", "0")
	broken["umc_err_count"] = "totally not the format\n"
	root := amdFixture(t, []amdCard{{name: "card0", driver: "amdgpu", ras: broken}})

	s := probeAMD(root)
	if s.read {
		t.Error("a sample with an unparseable file claimed to be read")
	}
	if got := amdState(s); got != counterUnknown {
		t.Errorf("state = %v, want counterUnknown", got)
	}

	out := newEmitter()
	emitAMD(out, s, s)
	if v, ok := out.get(keyECCErrors); ok {
		t.Errorf("%s = %q — an unreadable counter must be omitted, not reported", keyECCErrors, v)
	}
	if v, _ := out.get("amd_ras_unreadable_files"); !strings.Contains(v, "umc_err_count") {
		t.Errorf("the omission is unexplained: %q", v)
	}
}

func TestHalfAFileIsNotHalfAReading(t *testing.T) {
	// `ue` present, `ce` missing. Guessing the other half as zero would
	// understate the damage while looking like a complete answer.
	half := clean("0", "0")
	half["gfx_err_count"] = "ue: 4\n"
	root := amdFixture(t, []amdCard{{name: "card0", driver: "amdgpu", ras: half}})

	if s := probeAMD(root); s.read {
		t.Error("a file with only ue: was accepted as a complete reading")
	}
}

func TestOneUnreadableCardOmitsTheWholeNodeTotal(t *testing.T) {
	// A subset is a wrong total, not a partial one — the card that did not
	// answer is exactly the one most likely to be broken. Same rule the NVIDIA
	// path applies across GPUs.
	root := amdFixture(t, []amdCard{
		{name: "card0", driver: "amdgpu", ras: clean("0", "0")},
		{name: "card1", driver: "amdgpu", ras: map[string]string{"umc_err_count": "garbage\n"}},
	})

	out := newEmitter()
	s := probeAMD(root)
	emitAMD(out, s, s)
	if _, ok := out.get(keyECCErrors); ok {
		t.Error("a node total was reported although one card could not be read")
	}
}

func TestSomeCardsWithRASAndSomeWithoutIsNotUnmeasurable(t *testing.T) {
	// It is a node this runner could not read COMPLETELY, which fails closed.
	// Calling it unmeasurable would be a claim about hardware that half the
	// node contradicts.
	root := amdFixture(t, []amdCard{
		{name: "card0", driver: "amdgpu", ras: clean("0", "0")},
		{name: "card1", driver: "amdgpu", ras: nil},
	})

	s := probeAMD(root)
	if got := amdState(s); got != counterUnknown {
		t.Errorf("state = %v, want counterUnknown — not %v", got, counterUnmeasurable)
	}
	out := newEmitter()
	emitAMD(out, s, s)
	if v, ok := out.get(keyECCErrors); ok {
		t.Errorf("%s = %q, want omitted", keyECCErrors, v)
	}
}

func TestANonAMDNodeEmitsNothingAtAll(t *testing.T) {
	// "No AMD RAS counters here" is not a measurement, and an `n/a` would be a
	// claim about hardware that is not present. On an NVIDIA or CPU-only node
	// these keys must simply not appear.
	rows := []struct {
		name  string
		cards []amdCard
	}{
		{"an NVIDIA node", []amdCard{{name: "card0", driver: "nvidia"}}},
		{"a card bound to no driver", []amdCard{{name: "card0"}}},
		{"no cards at all", nil},
	}
	for _, r := range rows {
		s := probeAMD(amdFixture(t, r.cards))
		if s.isAMDNode() {
			t.Errorf("%s: detected as an AMD node", r.name)
		}
		out := newEmitter()
		emitAMD(out, s, s)
		for _, key := range []string{keyECCErrors, "ecc_source", "amd_gpus"} {
			if v, ok := out.get(key); ok {
				t.Errorf("%s: emitted %s=%q", r.name, key, v)
			}
		}
	}
}

func TestAnErrorDuringTheWindowIsReportedSeparately(t *testing.T) {
	// An uncorrected error that happened WHILE we watched is a different and
	// worse thing than an old one, exactly as on the NVIDIA path.
	before := probeAMD(amdFixture(t, []amdCard{{name: "card0", driver: "amdgpu", ras: clean("1", "0")}}))
	after := probeAMD(amdFixture(t, []amdCard{{name: "card0", driver: "amdgpu", ras: clean("4", "0")}}))

	out := newEmitter()
	emitAMD(out, before, after)
	if v, _ := out.get("ecc_uncorrected_window"); v != "3" {
		t.Errorf("window delta = %q, want 3", v)
	}
	if v, _ := out.get(keyECCErrors); v != "4" {
		t.Errorf("lifetime total = %q, want 4", v)
	}
}

func TestTheNVIDIAPathKeepsItsCounterOnAMixedNode(t *testing.T) {
	// Not a case this project supports — one pod gets one image, which is why
	// imagesByVendor keys on a single vendor per node. Overwriting would report
	// one vendor's errors as the node's total; the situation is recorded
	// instead so it is visible rather than silently wrong.
	root := amdFixture(t, []amdCard{{name: "card0", driver: "amdgpu", ras: clean("9", "0")}})

	out := newEmitter()
	out.setInt(keyECCErrors, 2) // as if the NVIDIA path had already spoken
	s := probeAMD(root)
	emitAMD(out, s, s)

	if v, _ := out.get(keyECCErrors); v != "2" {
		t.Errorf("%s = %q — the NVIDIA reading should stand", keyECCErrors, v)
	}
	if v, _ := out.get("ecc_vendor_conflict"); v != "nvidia+amd" {
		t.Errorf("the conflict was not recorded: %q", v)
	}
}
