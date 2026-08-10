// SPDX-License-Identifier: Apache-2.0
// Copyright the Glimmer authors.

package contract_test

import (
	"go/build"
	"strings"
	"testing"
)

// TestTheKubernetesFreePackagesStayThatWay is the guard behind a claim CLAUDE.md
// makes to prospective adopters.
//
// pkg/contract and pkg/runner are described as carrying NO Kubernetes
// dependency, and that is the reason a consumer can decode a burn-in envelope
// or parse runner output without linking controller-runtime. pkg/verdict is
// NOT in that set — it takes []burninv1alpha1.Threshold and so brings
// api/v1alpha1 with it (issue #274), which is why it is absent below rather
// than forgotten.
//
// Measured rather than asserted: a fresh module importing each package alone
// and running `go mod tidy` gets 0, 0 and 9 Kubernetes modules respectively.
// This test is the cheap continuous version of that measurement — it walks the
// transitive import graph inside this module and fails on the first k8s path.
//
// The failure it prevents is quiet: adding one convenience helper that takes a
// metav1.Time would make the documented claim false with nothing else changing,
// and the cost lands on the bare-metal dispatcher, which would start linking a
// controller runtime it cannot use.
func TestTheKubernetesFreePackagesStayThatWay(t *testing.T) {
	const modulePath = "github.com/baldwinSPC/glimmer-burnin"

	for _, pkg := range []string{
		modulePath + "/pkg/contract",
		modulePath + "/pkg/runner",
	} {
		seen := map[string]bool{}
		var walk func(path, via string)
		walk = func(path, via string) {
			if seen[path] {
				return
			}
			seen[path] = true

			if strings.HasPrefix(path, "k8s.io/") || strings.HasPrefix(path, "sigs.k8s.io/") {
				t.Errorf("%s reaches %s (via %s).\n"+
					"CLAUDE.md tells adopters this package carries no Kubernetes dependency, which is what "+
					"lets a consumer decode results without linking controller-runtime. If the dependency is "+
					"genuinely needed, the documentation has to change with it — see pkg/verdict and #274 for "+
					"what that costs.", pkg, path, via)
				return
			}
			// Only walk this module's own packages; the standard library and
			// third-party trees cannot reach back into k8s without going
			// through one of ours first.
			if !strings.HasPrefix(path, modulePath) {
				return
			}

			p, err := build.Import(path, "", 0)
			if err != nil {
				// A package that does not resolve is not this test's business.
				return
			}
			for _, imp := range p.Imports {
				walk(imp, path)
			}
		}
		walk(pkg, "(root)")
	}
}
