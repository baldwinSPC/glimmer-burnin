package runnerimages

import (
	"os"
	"path/filepath"
	"sort"
	"strings"
	"testing"

	"github.com/baldwinSPC/glimmer-burnin/pkg/contract"

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
	excluded := map[contract.TestKind]bool{}
	for _, k := range WithoutDefault() {
		excluded[k] = true
	}

	for _, kind := range contract.BuiltInKinds {
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
	before, ok := Default(contract.KindComputeSmoke)
	if !ok {
		t.Fatal("compute-smoke has no default image")
	}

	All()[contract.KindComputeSmoke] = "ghcr.io/attacker/evil:latest"

	after, _ := Default(contract.KindComputeSmoke)
	if after != before {
		t.Fatalf("mutating the result of All() changed the table: %q became %q", before, after)
	}
}

// TestAnUnknownKindHasNoDefault — the open-world TestKind means a third-party
// kind is legitimate, and it must resolve to "set spec.runner.image" rather than
// to somebody else's image.
func TestAnUnknownKindHasNoDefault(t *testing.T) {
	if img, ok := Default(contract.TestKind("something-nobody-registered")); ok {
		t.Errorf("an unregistered kind resolved to %q", img)
	}
}

// Every default declares WHICH VENDOR'S HARDWARE it can measure.
//
// Same discipline as pkg/contract refusing an Unspecified Aggregation: the
// answer lives beside the name, so "can this image measure this node?" is a
// lookup and never a guess. Without it, "fall through to the kind's default" is
// an assumption — and since every image this project ships but four is an
// NVIDIA image, that assumption is wrong on exactly the fleets imagesByVendor
// exists for.
func TestEveryDefaultDeclaresItsVendor(t *testing.T) {
	for kind := range All() {
		v, ok := VendorOf(kind)
		if !ok || v == "" {
			t.Errorf("kind %q has a default image and no declared vendor: nothing can then tell "+
				"whether it may run on an AMD node", kind)
		}
	}
	// And the vendor-neutral group is exactly the runners that touch no
	// accelerator, which runners/pins_test.go already names with its reason —
	// so this column is cross-checked against an existing list rather than being
	// new unbacked state.
	neutral := map[contract.TestKind]bool{}
	for kind := range All() {
		if v, _ := VendorOf(kind); v == VendorAny {
			neutral[kind] = true
		}
	}
	for _, want := range []contract.TestKind{
		contract.KindIBWriteBW, contract.KindMemoryStress, contract.KindTCPBaseline, contract.KindDiskIO,
	} {
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
	nvidiaDefault, _ := Default(contract.KindMemoryBW)

	for _, tc := range []struct {
		name    string
		kind    contract.TestKind
		runner  *api.RunnerSpec
		vendor  string
		want    string
		wantErr string
	}{
		{name: "explicit image beats everything", kind: contract.KindMemoryBW,
			runner: img("example.invalid/pinned:v9"), vendor: "amd", want: "example.invalid/pinned:v9"},

		{name: "the vendor's own entry", kind: contract.KindMemoryBW,
			runner: byVendor(rocm), vendor: "amd", want: rocm.Image},

		// THE ERGONOMIC CASE THAT MUST KEEP WORKING. Listing only the vendor
		// that needs an override leaves every other node on the operator's own
		// default; forcing the author to hand-pin the nvidia image would freeze
		// it at whatever tag they typed.
		{name: "an unlisted vendor the default can serve", kind: contract.KindMemoryBW,
			runner: byVendor(rocm), vendor: "nvidia", want: nvidiaDefault},

		// THE FAIL-OPEN THIS CLOSES. Every built-in default is an NVIDIA image,
		// so before the vendor column this fell through and pulled nvbandwidth
		// onto an AMD node. What the runner then does is its own business and
		// ranges from wasteful to catastrophic — compute-smoke v0.1.0 exits 1
		// for "no usable CUDA device", a permanent hardware indictment, and a
		// runner exiting 2 with a marker certifies hardware nobody measured.
		{name: "an unlisted vendor the default CANNOT serve", kind: contract.KindMemoryBW,
			runner: byVendor(api.VendorImage{Vendor: "intel", Image: "x"}), vendor: "amd",
			wantErr: "amd"},

		// Same refusal with no imagesByVendor at all: an NVIDIA-only image on a
		// node that positively declared itself AMD is wrong however it was asked
		// for.
		{name: "no map, and the default cannot serve this node", kind: contract.KindMemoryBW,
			vendor: "amd", wantErr: "amd"},

		// A vendor-neutral image serves anything, which is what the column is
		// FOR: it turns "fall through" from a guess into a checkable claim.
		{name: "a vendor-neutral default serves any node", kind: contract.KindIBWriteBW,
			vendor: "amd", want: mustDefault(t, contract.KindIBWriteBW)},

		// UNKNOWN VENDOR IS NOT A MISMATCH. A node nothing has fingerprinted
		// declared nothing, and absence is not a declaration — this is every
		// cluster before imagesByVendor existed and must not start failing.
		{name: "an unknown vendor falls through as before", kind: contract.KindMemoryBW,
			vendor: "", want: nvidiaDefault},

		{name: "no default at all", kind: contract.KindCustom, vendor: "nvidia",
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

func mustDefault(t *testing.T, kind contract.TestKind) string {
	t.Helper()
	img, ok := Default(kind)
	if !ok {
		t.Fatalf("kind %q has no default", kind)
	}
	return img
}

// SAMPLES MUST AGREE WITH THIS TABLE, because a sample is a claim about what
// applying it will do to a fleet.
//
// Two samples told an operator they were inert. pair-network-acceptance.yaml
// said "no nccl or ib-write-bw runner ships in this repo yet, so neither kind
// has a default image", and node-acceptance.yaml said "only compute-smoke has a
// published image". Both were true when written, and a release made them false
// without touching either file. An engineer applying the first believing it a
// documented no-op instead cordoned two named nodes, held BOTH for the whole
// test — a pair costs two interlock slots — and put real RDMA load on them
// (#299).
//
// Nothing produced a wrong verdict. It produced unrequested load on a fleet on
// the strength of a comment the release had invalidated, which prose review
// cannot catch: the sentence still reads perfectly.
//
// So the samples carry a GENERATED line and this compares it to the table. The
// same discipline hack/invariants already applies to CLAUDE.md.
func TestSamplesAgreeWithTheRunnerImageTable(t *testing.T) {
	want := make([]string, 0, len(WithoutDefault()))
	for _, k := range WithoutDefault() {
		want = append(want, string(k))
	}
	sort.Strings(want)
	expected := strings.Join(want, ", ")

	files, err := filepath.Glob(filepath.Join("..", "..", "config", "samples", "*.yaml"))
	if err != nil {
		t.Fatal(err)
	}

	marker := "KINDS THAT STILL NEED AN EXPLICIT spec.runner.image"
	found := 0
	for _, file := range files {
		raw, err := os.ReadFile(file)
		if err != nil {
			t.Fatal(err)
		}
		text := string(raw)
		i := strings.Index(text, marker)
		if i < 0 {
			continue
		}
		found++

		// The generated list is the first comment line after the marker's own
		// two-line preamble that names kinds.
		var got string
		for _, line := range strings.Split(text[i:], "\n") {
			trimmed := strings.TrimSpace(strings.TrimPrefix(strings.TrimSpace(line), "#"))
			if trimmed == "" || strings.Contains(trimmed, marker) ||
				strings.HasPrefix(trimmed, "by hand") {
				continue
			}
			got = trimmed
			break
		}
		if got != expected {
			t.Errorf("%s lists kinds needing an explicit image as:\n  %s\nbut runnerimages.WithoutDefault() says:\n  %s\n"+
				"A sample is a claim about what applying it does to a fleet; this one has gone stale, which is "+
				"exactly how #299 shipped.", filepath.Base(file), got, expected)
		}
	}
	if found == 0 {
		t.Errorf("no sample carries the %q marker, so nothing keeps the samples' image claims honest", marker)
	}
}
