package controller

import (
	"go/ast"
	"go/parser"
	"go/token"
	"os"
	"path/filepath"
	"strings"
	"testing"

	burninv1alpha1 "github.com/baldwinSPC/glimmer-burnin/api/v1alpha1"
)

func TestOneProfileServesAMixedFleet(t *testing.T) {
	// The whole point: the same BurnInTestSpec, two nodes, two images.
	spec := burninv1alpha1.BurnInTestSpec{
		Kind: burninv1alpha1.KindMemoryBW,
		Runner: &burninv1alpha1.RunnerSpec{
			ImagesByVendor: []burninv1alpha1.VendorImage{
				{Vendor: "nvidia", Image: "ghcr.io/x/memory-bw:v1"},
				{Vendor: "amd", Image: "ghcr.io/x/memory-bw-rocm:v1"},
			},
		},
	}

	for vendor, want := range map[string]string{
		"nvidia": "ghcr.io/x/memory-bw:v1",
		"amd":    "ghcr.io/x/memory-bw-rocm:v1",
	} {
		got, err := runnerImage(&spec, vendor)
		if err != nil {
			t.Fatalf("%s: %v", vendor, err)
		}
		if got != want {
			t.Errorf("%s resolved to %q, want %q", vendor, got, want)
		}
	}
}

func TestResolutionOrderIsExplicitThenVendorThenDefault(t *testing.T) {
	explicit := burninv1alpha1.BurnInTestSpec{
		Kind: burninv1alpha1.KindMemoryBW,
		Runner: &burninv1alpha1.RunnerSpec{
			Image:          "ghcr.io/x/pinned:v9",
			ImagesByVendor: []burninv1alpha1.VendorImage{{Vendor: "amd", Image: "ghcr.io/x/rocm:v1"}},
		},
	}
	// An explicit pin means EVERY node, including one whose vendor is listed.
	// Anything else would make `image` mean something different depending on
	// which node read it.
	got, err := runnerImage(&explicit, "amd")
	if err != nil {
		t.Fatal(err)
	}
	if got != "ghcr.io/x/pinned:v9" {
		t.Errorf("explicit image lost to imagesByVendor: %q", got)
	}

	// byVendor beats the kind's built-in default.
	byVendor := burninv1alpha1.BurnInTestSpec{
		Kind:   burninv1alpha1.KindMemoryBW,
		Runner: &burninv1alpha1.RunnerSpec{ImagesByVendor: []burninv1alpha1.VendorImage{{Vendor: "amd", Image: "ghcr.io/x/rocm:v1"}}},
	}
	if got, err = runnerImage(&byVendor, "amd"); err != nil || got != "ghcr.io/x/rocm:v1" {
		t.Errorf("byVendor = %q (%v), want the rocm image", got, err)
	}

	// A vendor the list does not name falls through to the default, because the
	// kind HAS one and it is the honest answer for a fleet that is mostly one
	// vendor with a few of another.
	if got, err = runnerImage(&byVendor, "nvidia"); err != nil {
		t.Errorf("a vendor absent from the map should fall through to the kind default: %v", err)
	} else if got == "ghcr.io/x/rocm:v1" {
		t.Error("an nvidia node got the rocm image")
	}
}

func TestAVendorWithNoImageAndNoDefaultFailsAtPlanTimeNamingBoth(t *testing.T) {
	// An ERROR, never a skip. A node silently not being tested is how a fleet
	// gets certified without being measured.
	spec := burninv1alpha1.BurnInTestSpec{
		Kind:   burninv1alpha1.KindTCPBaseline, // deliberately has no built-in default
		Runner: &burninv1alpha1.RunnerSpec{ImagesByVendor: []burninv1alpha1.VendorImage{{Vendor: "nvidia", Image: "ghcr.io/x/tcp:v1"}}},
	}

	_, err := runnerImage(&spec, "amd")
	if err == nil {
		t.Fatal("an unresolvable vendor was accepted")
	}
	for _, want := range []string{"amd", "tcp-baseline", "imagesByVendor", "nvidia"} {
		if !strings.Contains(err.Error(), want) {
			t.Errorf("the error should name %q so it can be acted on: %v", want, err)
		}
	}
}

func TestAnUnknownVendorSaysSoRatherThanPrintingAnEmptyString(t *testing.T) {
	// "no image for vendor \"\"" sends the reader looking for a typo in their
	// YAML. The real problem is that the node has no fingerprint.
	spec := burninv1alpha1.BurnInTestSpec{
		Kind:   burninv1alpha1.KindTCPBaseline,
		Runner: &burninv1alpha1.RunnerSpec{ImagesByVendor: []burninv1alpha1.VendorImage{{Vendor: "nvidia", Image: "ghcr.io/x/tcp:v1"}}},
	}
	_, err := runnerImage(&spec, "")
	if err == nil {
		t.Fatal("accepted")
	}
	if !strings.Contains(err.Error(), "fingerprint") {
		t.Errorf("an empty vendor should point at the fingerprint, not at the YAML: %v", err)
	}
}

func TestASingleVendorFleetIsUnaffected(t *testing.T) {
	// Every profile written before this field existed must resolve exactly as
	// it did, including on a node with no fingerprint at all.
	spec := burninv1alpha1.BurnInTestSpec{Kind: burninv1alpha1.KindMemoryBW}
	withVendor, err := runnerImage(&spec, "nvidia")
	if err != nil {
		t.Fatal(err)
	}
	withoutVendor, err := runnerImage(&spec, "")
	if err != nil {
		t.Fatal(err)
	}
	if withVendor != withoutVendor {
		t.Errorf("the default answer changed with the vendor: %q vs %q", withVendor, withoutVendor)
	}
}

func TestTheVendorComesFromTheFingerprintAndUnknownMeansNone(t *testing.T) {
	rows := []struct {
		name string
		fp   *burninv1alpha1.NodeFingerprint
		want string
	}{
		{"no fingerprint", nil, ""},
		{"no GPUs", &burninv1alpha1.NodeFingerprint{}, ""},
		{"an nvidia node", fingerprintWith("nvidia"), "nvidia"},
		{"an amd node", fingerprintWith("amd"), "amd"},
		// "unknown" is what the fingerprint records for a device it could not
		// attribute. Normalised away because the CRD's vendor enum rejects it,
		// so treating it as a vendor would produce a lookup that can never hit.
		{"a device it could not attribute", fingerprintWith("unknown"), ""},
	}
	for _, r := range rows {
		if got := vendorOf(r.fp); got != r.want {
			t.Errorf("%s: vendor = %q, want %q", r.name, got, r.want)
		}
	}
}

func fingerprintWith(vendor string) *burninv1alpha1.NodeFingerprint {
	fp := &burninv1alpha1.NodeFingerprint{}
	fp.Status.GPUs = []burninv1alpha1.GPUInfo{{Vendor: vendor}}
	return fp
}

// TestTheReconcilerHasNoVendorBranch is the invariant this feature exists to
// preserve, not merely to avoid breaking.
//
// Vendor-neutrality lives in the reconciler: accelerator-specific behaviour
// belongs in runner IMAGES, never in controller branches, so adding support for
// a vendor means adding a runner rather than an `if nvidia {}`. imagesByVendor
// is what lets a mixed fleet be expressed as DATA — and the moment somebody
// reaches for a comparison against a vendor name here, that has been given up.
//
// So the guard is literal: no source file in this package may compare anything
// to a vendor string. Resolution is a lookup and must stay one.
func TestTheReconcilerHasNoVendorBranch(t *testing.T) {
	vendors := map[string]bool{"nvidia": true, "amd": true, "intel": true, "tenstorrent": true}

	entries, err := os.ReadDir(".")
	if err != nil {
		t.Fatal(err)
	}

	checked := 0
	for _, e := range entries {
		name := e.Name()
		if !strings.HasSuffix(name, ".go") || strings.HasSuffix(name, "_test.go") {
			continue
		}
		checked++

		fset := token.NewFileSet()
		f, err := parser.ParseFile(fset, filepath.Join(".", name), nil, 0)
		if err != nil {
			t.Fatalf("parsing %s: %v", name, err)
		}

		ast.Inspect(f, func(n ast.Node) bool {
			bin, ok := n.(*ast.BinaryExpr)
			if !ok || (bin.Op != token.EQL && bin.Op != token.NEQ) {
				return true
			}
			for _, side := range []ast.Expr{bin.X, bin.Y} {
				lit, ok := side.(*ast.BasicLit)
				if !ok || lit.Kind != token.STRING {
					continue
				}
				if vendors[strings.Trim(lit.Value, `"`)] {
					t.Errorf("%s:%d compares against the vendor %s. Vendor-specific behaviour belongs in "+
						"a runner image, never in a controller branch — supporting an accelerator must stay "+
						"a matter of adding a runner, not adding a case here",
						name, fset.Position(lit.Pos()).Line, lit.Value)
				}
			}
			return true
		})
	}

	// Asserted rather than assumed: a rename that emptied this list would leave
	// the guard passing while inspecting nothing.
	if checked == 0 {
		t.Fatal("found no non-test .go files in this package, which cannot be right")
	}
}
