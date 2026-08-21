// SPDX-License-Identifier: Apache-2.0
// Copyright the Glimmer authors.

package main

import (
	"testing"

	"github.com/baldwinSPC/glimmer-burnin/pkg/contract"
)

// TestPCIVendorsAgreeWithTheContractList holds this runner's OWN COPY of the
// PCI-vendor-ID table (forced by the image build, see pci.go's header
// comment — a runner's non-test sources cannot import the repo, but a
// _test.go file can) to pkg/contract.AcceleratorVendors. See that list's own
// doc comment for the full set of sources required to agree and why each is
// checked the way it is.
func TestPCIVendorsAgreeWithTheContractList(t *testing.T) {
	got := map[string]bool{}
	for _, name := range pciVendors {
		got[name] = true
	}
	want := map[string]bool{}
	for _, name := range contract.AcceleratorVendors {
		want[name] = true
	}
	for name := range got {
		if !want[name] {
			t.Errorf("pciVendors names %q, which is not in contract.AcceleratorVendors", name)
		}
	}
	for name := range want {
		if !got[name] {
			t.Errorf("contract.AcceleratorVendors names %q, which this runner's pciVendors has no PCI "+
				"ID for — add one, or the vendor can never actually be detected from hardware here", name)
		}
	}
}
