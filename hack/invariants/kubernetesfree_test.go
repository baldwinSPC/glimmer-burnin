package invariants

import (
	"os/exec"
	"sort"
	"strings"
	"testing"
)

// WHICH PUBLIC PACKAGES COST A CONSUMER KUBERNETES — as a ledger, not a claim.
//
// The documentation asserted for a while that pkg/verdict was Kubernetes-free
// when it was not: `Evaluate` took []v1alpha1.Threshold, so the CRD package sat
// in its signature and travelled with it, bringing apimachinery and
// controller-runtime to every consumer. #274 measured that at 0, 0 and 9
// modules for contract, runner and verdict.
//
// It is true NOW, and by construction rather than by luck: the acceptance
// vocabulary moved to pkg/contract (#342), so verdict measures 0. This comment
// itself asserted the old state for one commit after that landed — which is the
// failure mode this file exists to prevent, occurring inside it. The lists below
// are executable and were correct throughout; the prose was not. That is the
// argument for the ledger in one paragraph.
//
// Documentation was corrected after #274; nothing was ever enforced. pkg/report
// has carried its own TestNoKubernetesDependency since it was written — and
// that test's comment names pkg/verdict as the hazard — but the property was
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
	// Freed by the #274 hoist: the acceptance vocabulary moved to pkg/contract,
	// so the shared verdict brain no longer imports the CRD to get its own
	// words. This is the entry that had to move for the ledger to stay exact.
	"pkg/verdict",
}

// kubernetesCoupled are the packages that DO cost a consumer Kubernetes, each
// with the reason. Every one of these is a design decision open in #274 rather
// than an accident, and the amendment to GEP-0178 proposes the fix.
var kubernetesCoupled = map[string]string{
	// Both remaining entries are coupled through the EXECUTION vocabulary —
	// BurnInTestSpec, RunnerSpec, VendorImage — not the acceptance vocabulary
	// #274 was about. Hoisting Threshold and TestKind freed pkg/verdict and did
	// not touch these, which is the honest boundary of that change.
	//
	// Freeing them is a larger decision (the GEP-0178 amendment's option C):
	// it would make the CRD a thin Kubernetes wrapper over a neutral core, and
	// the right time to design that seam is when Path A is built and can say
	// what it actually needs — not speculatively.
	"pkg/localrun": "PlannedTest holds api.BurnInTestSpec whole, plus RunPhase, Violation, " +
		"TestScope and AttemptTrigger. This is the package GEP-0178 has Glimmer's Path A " +
		"adopt, so the coupling lands on the path that can use Kubernetes least",
	"pkg/runnerimages": "resolution is expressed over api.RunnerSpec and api.VendorImage",
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
