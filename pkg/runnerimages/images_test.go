package runnerimages

import (
	"strings"
	"testing"

	api "github.com/baldwinSPC/glimmer-burnin/api/v1alpha1"
)

// TestEveryBuiltInKindIsDecided is the guard that makes this table complete
// rather than merely populated.
//
// A kind added to the API and forgotten here would schedule pods with no image
// and produce an ImagePullBackOff that reads as a hardware fault. A kind
// deliberately left out — KindCustom — is a decision, and the point of this test
// is that the two cases cannot be confused: every built-in kind must either have
// an image or be on the explicit exclusion list.
func TestEveryBuiltInKindIsDecided(t *testing.T) {
	excluded := map[api.TestKind]bool{}
	for _, k := range WithoutDefault() {
		excluded[k] = true
	}

	for _, kind := range api.BuiltInKinds {
		_, has := Default(kind)
		switch {
		case has && excluded[kind]:
			t.Errorf("%s has a default image AND is listed as deliberately without one", kind)
		case !has && !excluded[kind]:
			t.Errorf("%s has no default image and is not on the deliberate exclusion list — "+
				"add an image once one is published, or say out loud that it has none", kind)
		}
	}
}

// TestEveryDefaultIsPinnedAndNamedByConvention.
//
// A floating tag would change the test under every existing profile with no
// audit trail, which makes an acceptance verdict unreproducible. The name shape
// is the publish workflow's own construction, so an entry here and a
// workflow_dispatch of the same version agree by construction.
func TestEveryDefaultIsPinnedAndNamedByConvention(t *testing.T) {
	for kind, img := range All() {
		if strings.HasSuffix(img, ":latest") || !strings.Contains(img, ":") {
			t.Errorf("%s: %q is not pinned to a version", kind, img)
		}
		want := "ghcr.io/baldwinspc/glimmer-burnin-" + string(kind) + ":"
		if !strings.HasPrefix(img, want) {
			t.Errorf("%s: %q does not follow the publish workflow's naming (%s<version>)", kind, img, want)
		}
	}
}

// TestAllReturnsACopy — a mutable package-level map shared by two dispatchers is
// a way for one of them to change what the other runs.
func TestAllReturnsACopy(t *testing.T) {
	before, ok := Default(api.KindComputeSmoke)
	if !ok {
		t.Fatal("compute-smoke has no default image")
	}

	All()[api.KindComputeSmoke] = "ghcr.io/attacker/evil:latest"

	after, _ := Default(api.KindComputeSmoke)
	if after != before {
		t.Fatalf("mutating the result of All() changed the table: %q became %q", before, after)
	}
}

// TestAnUnknownKindHasNoDefault — the open-world TestKind means a third-party
// kind is legitimate, and it must resolve to "set spec.runner.image" rather than
// to somebody else's image.
func TestAnUnknownKindHasNoDefault(t *testing.T) {
	if img, ok := Default(api.TestKind("something-nobody-registered")); ok {
		t.Errorf("an unregistered kind resolved to %q", img)
	}
}

// Every default declares WHICH VENDOR'S HARDWARE it can measure.
//
// Same discipline as pkg/contract refusing an Unspecified Aggregation: the
// answer lives beside the name, so "can this image measure this node?" is a
// lookup and never a guess. Without it, "fall through to the kind's default" is
// an assumption — and since every image this project ships but two is an NVIDIA
// image, that assumption is wrong on exactly the fleets imagesByVendor exists
// for.
func TestEveryDefaultDeclaresItsVendor(t *testing.T) {
	for kind := range All() {
		v, ok := VendorOf(kind)
		if !ok || v == "" {
			t.Errorf("kind %q has a default image and no declared vendor: nothing can then tell "+
				"whether it may run on an AMD node", kind)
		}
	}
	// And the vendor-neutral pair is exactly the two runners that touch no
	// accelerator, which runners/pins_test.go already names with its reason —
	// so this column is cross-checked against an existing list rather than being
	// new unbacked state.
	neutral := map[api.TestKind]bool{}
	for kind := range All() {
		if v, _ := VendorOf(kind); v == VendorAny {
			neutral[kind] = true
		}
	}
	for _, want := range []api.TestKind{api.KindIBWriteBW, api.KindMemoryStress} {
		if !neutral[want] {
			t.Errorf("%q touches no accelerator and must be %s", want, VendorAny)
		}
		delete(neutral, want)
	}
	for extra := range neutral {
		t.Errorf("%q is declared vendor-neutral: if that is right, say why here and in "+
			"runners/pins_test.go's driver-exemption list", extra)
	}
}

// The resolution ladder, in one place, for both dispatchers.
func TestResolve(t *testing.T) {
	img := func(s string) *api.RunnerSpec { return &api.RunnerSpec{Image: s} }
	byVendor := func(pairs ...api.VendorImage) *api.RunnerSpec {
		return &api.RunnerSpec{ImagesByVendor: pairs}
	}
	rocm := api.VendorImage{Vendor: "amd", Image: "example.invalid/memory-bw-rocm:v1"}
	nvidiaDefault, _ := Default(api.KindMemoryBW)

	for _, tc := range []struct {
		name    string
		kind    api.TestKind
		runner  *api.RunnerSpec
		vendor  string
		want    string
		wantErr string
	}{
		{name: "explicit image beats everything", kind: api.KindMemoryBW,
			runner: img("example.invalid/pinned:v9"), vendor: "amd", want: "example.invalid/pinned:v9"},

		{name: "the vendor's own entry", kind: api.KindMemoryBW,
			runner: byVendor(rocm), vendor: "amd", want: rocm.Image},

		// THE ERGONOMIC CASE THAT MUST KEEP WORKING. Listing only the vendor
		// that needs an override leaves every other node on the operator's own
		// default; forcing the author to hand-pin the nvidia image would freeze
		// it at whatever tag they typed.
		{name: "an unlisted vendor the default can serve", kind: api.KindMemoryBW,
			runner: byVendor(rocm), vendor: "nvidia", want: nvidiaDefault},

		// THE FAIL-OPEN THIS CLOSES. Every built-in default is an NVIDIA image,
		// so before the vendor column this fell through and pulled nvbandwidth
		// onto an AMD node. What the runner then does is its own business and
		// ranges from wasteful to catastrophic — compute-smoke v0.1.0 exits 1
		// for "no usable CUDA device", a permanent hardware indictment, and a
		// runner exiting 2 with a marker certifies hardware nobody measured.
		{name: "an unlisted vendor the default CANNOT serve", kind: api.KindMemoryBW,
			runner: byVendor(api.VendorImage{Vendor: "intel", Image: "x"}), vendor: "amd",
			wantErr: "amd"},

		// Same refusal with no imagesByVendor at all: an NVIDIA-only image on a
		// node that positively declared itself AMD is wrong however it was asked
		// for.
		{name: "no map, and the default cannot serve this node", kind: api.KindMemoryBW,
			vendor: "amd", wantErr: "amd"},

		// A vendor-neutral image serves anything, which is what the column is
		// FOR: it turns "fall through" from a guess into a checkable claim.
		{name: "a vendor-neutral default serves any node", kind: api.KindIBWriteBW,
			vendor: "amd", want: mustDefault(t, api.KindIBWriteBW)},

		// UNKNOWN VENDOR IS NOT A MISMATCH. A node nothing has fingerprinted
		// declared nothing, and absence is not a declaration — this is every
		// cluster before imagesByVendor existed and must not start failing.
		{name: "an unknown vendor falls through as before", kind: api.KindMemoryBW,
			vendor: "", want: nvidiaDefault},

		{name: "no default at all", kind: api.KindCustom, vendor: "nvidia",
			wantErr: "spec.runner.image"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			got, err := Resolve(tc.kind, tc.runner, tc.vendor)
			if tc.wantErr != "" {
				if err == nil {
					t.Fatalf("resolved to %q, want an error mentioning %q", got, tc.wantErr)
				}
				if !strings.Contains(err.Error(), tc.wantErr) {
					t.Errorf("error %q does not mention %q, so nobody can act on it", err, tc.wantErr)
				}
				return
			}
			if err != nil {
				t.Fatalf("Resolve: %v", err)
			}
			if got != tc.want {
				t.Errorf("resolved %q, want %q", got, tc.want)
			}
		})
	}
}

func mustDefault(t *testing.T, kind api.TestKind) string {
	t.Helper()
	img, ok := Default(kind)
	if !ok {
		t.Fatalf("kind %q has no default", kind)
	}
	return img
}
