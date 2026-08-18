package main

import (
	"os"
	"path/filepath"
	"testing"
)

// Synthetic sysfs trees, because the two cases that matter cannot both be
// obtained from one machine: a GB10 supplies the display-class-accelerator, and
// only a server baseboard supplies the display-only VGA that made excluding the
// class look correct in the first place.
//
// The GB10 numbers here are MEASURED (2026-08-17, spark-043a): 000f:01:00.0,
// class 0x030000, vendor 0x10de, device 0x2e12, driver nvidia, drm/{card0,
// renderD128}. The ASPEED entry is constructed from its published IDs and the
// property under test — a display-only DRM driver declares no DRIVER_RENDER and
// therefore gets no renderD node.
func writeDev(t *testing.T, root, addr string, files map[string]string, drm []string, driver string) {
	t.Helper()
	dir := filepath.Join(root, "bus", "pci", "devices", addr)
	if err := os.MkdirAll(dir, 0o755); err != nil {
		t.Fatal(err)
	}
	for name, body := range files {
		if err := os.WriteFile(filepath.Join(dir, name), []byte(body+"\n"), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	if drm != nil {
		if err := os.MkdirAll(filepath.Join(dir, "drm"), 0o755); err != nil {
			t.Fatal(err)
		}
		for _, n := range drm {
			if err := os.MkdirAll(filepath.Join(dir, "drm", n), 0o755); err != nil {
				t.Fatal(err)
			}
		}
	}
	if driver != "" {
		target := filepath.Join(root, "bus", "pci", "drivers", driver)
		if err := os.MkdirAll(target, 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.Symlink(target, filepath.Join(dir, "driver")); err != nil {
			t.Fatal(err)
		}
	}
}

func gb10(t *testing.T, root string) {
	writeDev(t, root, "000f:01:00.0", map[string]string{
		"class": "0x030000", "vendor": "0x10de", "device": "0x2e12",
	}, []string{"card0", "renderD128"}, "nvidia")
}

func aspeedBMC(t *testing.T, root string) {
	// Display-only: card0 and NO render node. This is the false positive the
	// old VGA exclusion existed to prevent, and it must stay prevented.
	writeDev(t, root, "0000:03:00.0", map[string]string{
		"class": "0x030000", "vendor": "0x1a03", "device": "0x2000",
	}, []string{"card0"}, "ast")
}

// TestAGB10CountsDespiteBeingVGAClass is the defect from #380.
//
// The runner reported acceleratorCount=0 on both Sparks while compute-smoke
// measured 104 TFLOPS on the same nodes in the same run, and the sample's own
// `acceleratorCount >= 1` turned that into a Fail — which is never retried.
func TestAGB10CountsDespiteBeingVGAClass(t *testing.T) {
	root := t.TempDir()
	gb10(t, root)

	res := scan(root)
	if len(res.Devices) != 1 {
		t.Fatalf("got %d accelerators, want 1 — a GB10 is class 0x030000 and is still an "+
			"accelerator; it has a DRM render node", len(res.Devices))
	}
	if len(res.Ambiguous) != 0 {
		t.Errorf("Ambiguous = %v, want none: a render node settles the question", res.Ambiguous)
	}
	if got := res.Devices[0].Vendor; got != "nvidia" {
		t.Errorf("Vendor = %q, want nvidia", got)
	}
}

// TestADisplayOnlyVGAIsStillNotAnAccelerator keeps the original protection.
//
// Widening to VGA outright — the obvious one-line fix — would count a server
// baseboard's display adapter on every x86 node in a fleet. A false POSITIVE is
// worse than the false negative it replaces: it certifies capacity that is not
// there, where the false negative at least fails loudly.
func TestADisplayOnlyVGAIsStillNotAnAccelerator(t *testing.T) {
	root := t.TempDir()
	aspeedBMC(t, root)

	res := scan(root)
	if len(res.Devices) != 0 {
		t.Errorf("got %d accelerators, want 0 — an ASPEED display adapter has drm/card0 and "+
			"no render node, so it computes nothing", len(res.Devices))
	}
	// It is NOT ambiguous either: ASPEED is not a vendor this project has a
	// name for, so there is nothing to be uncertain about.
	if len(res.Ambiguous) != 0 {
		t.Errorf("Ambiguous = %v, want none: an unknown-vendor display adapter is simply "+
			"not an accelerator", res.Ambiguous)
	}
}

// TestABMCBesideAGPUCountsExactlyOne is the mixed case a real x86 GPU node has.
func TestABMCBesideAGPUCountsExactlyOne(t *testing.T) {
	root := t.TempDir()
	gb10(t, root)
	aspeedBMC(t, root)

	res := scan(root)
	if len(res.Devices) != 1 {
		t.Fatalf("got %d accelerators, want exactly 1", len(res.Devices))
	}
	if res.Devices[0].PCIAddress != "000f:01:00.0" {
		t.Errorf("counted %s, want the GPU", res.Devices[0].PCIAddress)
	}
}

// TestAnAcceleratorVendorWithNoRenderNodeIsUnmeasurable is the (E) half.
//
// An NVIDIA display-class device with no render node is either a display
// adapter or a GPU whose driver never bound. sysfs cannot say which, and both
// answers are a number — too low if it was a GPU, too high if it was not. So
// the COUNT becomes unmeasurable rather than a guess, which is the same rule
// the `n/a` sentinel exists for everywhere else in this project.
func TestAnAcceleratorVendorWithNoRenderNodeIsUnmeasurable(t *testing.T) {
	root := t.TempDir()
	writeDev(t, root, "0000:65:00.0", map[string]string{
		"class": "0x030000", "vendor": "0x10de", "device": "0x2230",
	}, []string{"card0"}, "")

	res := scan(root)
	if len(res.Ambiguous) != 1 {
		t.Fatalf("Ambiguous = %d, want 1: an NVIDIA display-class device with no render node "+
			"cannot be classified from sysfs", len(res.Ambiguous))
	}
	if len(res.Devices) != 0 {
		t.Errorf("Devices = %d, want 0: an unclassifiable device must not be COUNTED as one, "+
			"which would be the guess in the other direction", len(res.Devices))
	}
}

// TestAnUnambiguousAcceleratorClassNeedsNoRenderNode.
//
// A datacenter part at class 0x0302 or 0x1200 is an accelerator by its own
// declaration. Requiring a render node of it would break every headless
// deployment whose driver exposes no DRM at all, which is a regression the
// render-node rule must not smuggle in.
func TestAnUnambiguousAcceleratorClassNeedsNoRenderNode(t *testing.T) {
	root := t.TempDir()
	writeDev(t, root, "0000:17:00.0", map[string]string{
		"class": "0x030200", "vendor": "0x10de", "device": "0x2330",
	}, nil, "nvidia")
	writeDev(t, root, "0000:18:00.0", map[string]string{
		"class": "0x120000", "vendor": "0x1e52", "device": "0xb140",
	}, nil, "")

	res := scan(root)
	if len(res.Devices) != 2 {
		t.Fatalf("got %d, want 2: 3D-controller and processing-accelerator classes are "+
			"unambiguous and must not depend on DRM", len(res.Devices))
	}
	if len(res.Ambiguous) != 0 {
		t.Errorf("Ambiguous = %v, want none", res.Ambiguous)
	}
}

// TestAbsentSysfsIsNotZeroDevices — unchanged behaviour, asserted because the
// return type moved from a slice to a struct and a nil-vs-empty slip here would
// turn "could not look" into "nothing there".
func TestAbsentSysfsIsNotZeroDevices(t *testing.T) {
	res := scan(filepath.Join(t.TempDir(), "does-not-exist"))
	if len(res.Devices) != 0 || len(res.Ambiguous) != 0 {
		t.Errorf("got %+v, want an empty result", res)
	}
}
