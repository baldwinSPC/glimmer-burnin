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
