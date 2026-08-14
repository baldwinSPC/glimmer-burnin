package invariants

import (
	"os/exec"
	"sort"
	"strings"
	"testing"
)

// WHICH PUBLIC PACKAGES COST A CONSUMER KUBERNETES — as a ledger, not a claim.
//
// The documentation asserted for a while that pkg/verdict was Kubernetes-free.
// It was not, and still is not: `Evaluate` takes []burninv1alpha1.Threshold, so
// the CRD package is in its signature and travels with it, bringing
// apimachinery and controller-runtime to every consumer. #274 measured it (0,
// 0 and 9 modules for contract, runner and verdict) and CLAUDE.md now says so
// plainly.
//
// Documentation was corrected; nothing was ever enforced. pkg/report has
// carried its own TestNoKubernetesDependency since it was written — and that
// test's comment names pkg/verdict as the hazard — but the property was
// guarded for exactly one package while three others held it by luck.
//
// So this is a two-sided ledger, the same shape as cmd/burnin's envelope
// parity guard:
//
//   - a package listed CLEAN that acquires a Kubernetes dependency fails. That
//     is the regression #274 describes, caught the day it happens rather than
//     the day an adopter tries the import.
//   - a package listed COUPLED that becomes clean ALSO fails. When the
//     threshold types move down (#274's option 2, and the GEP-0178 amendment),
//     the entry must be deleted in the same change — so the ledger cannot
//     quietly describe a world that no longer exists, which is precisely how
//     the original wrong claim survived.
//
// The unit here is the DEPENDENCY GRAPH (`go list -deps`), not the module
// requirements #274 counted with `go mod tidy`. Different units, same fact: a
// package is clean or it is not.

// kubernetesFree are the packages a third party may import without linking
// Kubernetes. pkg/report guards itself; it is listed for completeness of the
// statement rather than to duplicate that test.
var kubernetesFree = []string{
	"pkg/contract",
	"pkg/runner",
	"pkg/report",
	"pkg/hostinfo",
}

// kubernetesCoupled are the packages that DO cost a consumer Kubernetes, each
// with the reason. Every one of these is a design decision open in #274 rather
// than an accident, and the amendment to GEP-0178 proposes the fix.
var kubernetesCoupled = map[string]string{
	"pkg/verdict": "Evaluate takes []v1alpha1.Threshold, so the CRD package is in its " +
		"signature — the coupling #274 is about",
	"pkg/localrun": "the bare-metal engine consumes v1alpha1.BurnInTestSpec verbatim, and it " +
		"is the package GEP-0178 has Glimmer's Path A adopt — so this is #274's coupling on " +
		"the path that can use Kubernetes least",
	"pkg/runnerimages": "keyed by v1alpha1.TestKind, which is a CRD type today",
}

func TestPublicPackagesDeclareWhatTheyCostAConsumer(t *testing.T) {
	for _, pkg := range kubernetesFree {
		if deps := kubernetesDepsOf(t, pkg); len(deps) > 0 {
			t.Errorf("%s is listed as Kubernetes-free and now depends on %v.\n"+
				"An adopter reads that claim before depending on this project, and the cost falls "+
				"hardest on the bare-metal dispatcher, which would link controller-runtime it cannot "+
				"use. Either drop the dependency, or move this package into kubernetesCoupled WITH "+
				"its reason and correct CLAUDE.md — the one thing that must not happen is the docs "+
				"asserting a property the code does not have (#274).",
				pkg, truncate(deps))
		}
	}

	for pkg, why := range kubernetesCoupled {
		if deps := kubernetesDepsOf(t, pkg); len(deps) == 0 {
			t.Errorf("%s is listed as Kubernetes-COUPLED (%s) but no longer depends on Kubernetes.\n"+
				"If the threshold types moved down (#274 option 2 / the GEP-0178 amendment), delete "+
				"this entry and add the package to kubernetesFree in the same change. A ledger that "+
				"describes a world which no longer exists is how the original wrong claim survived "+
				"as long as it did.", pkg, why)
		}
	}
}

// kubernetesDepsOf returns the k8s.io and sigs.k8s.io packages in one
// package's transitive dependency graph.
func kubernetesDepsOf(t *testing.T, pkg string) []string {
	t.Helper()
	const mod = "github.com/baldwinSPC/glimmer-burnin/"
	out, err := exec.Command("go", "list", "-deps", mod+pkg).Output()
	if err != nil {
		t.Fatalf("go list -deps %s: %v", pkg, err)
	}
	all := strings.Split(strings.TrimSpace(string(out)), "\n")
	// A guard that reads an empty graph passes no matter what is imported.
	if len(all) < 10 {
		t.Fatalf("go list -deps %s returned %d packages, which cannot be right — this guard is "+
			"not reading the build graph", pkg, len(all))
	}
	var k8s []string
	for _, dep := range all {
		dep = strings.TrimSpace(dep)
		if strings.HasPrefix(dep, "k8s.io/") || strings.HasPrefix(dep, "sigs.k8s.io/") {
			k8s = append(k8s, dep)
		}
	}
	sort.Strings(k8s)
	return k8s
}

func truncate(deps []string) []string {
	if len(deps) <= 4 {
		return deps
	}
	return append(append([]string{}, deps[:4]...), "…")
}
