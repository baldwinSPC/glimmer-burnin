// SPDX-License-Identifier: Apache-2.0
// Copyright the Glimmer authors.

package v1alpha1_test

import (
	"os"
	"regexp"
	"sort"
	"strings"
	"testing"

	"github.com/baldwinSPC/glimmer-burnin/pkg/contract"
)

// TestVendorEnumAgreesWithTheContractList holds VendorImage.Vendor's
// `+kubebuilder:validation:Enum` marker to pkg/contract.AcceleratorVendors —
// see that list's own doc comment for the full set of sources required to
// agree. The marker is not Go code a test can call, so this reads the
// source file and scrapes it, the same technique
// runners/devicefold_test.go's TestDeviceFoldAllowSetsAgreeWithTheCLI uses
// for a different pair of tables that must not drift.
func TestVendorEnumAgreesWithTheContractList(t *testing.T) {
	raw, err := os.ReadFile("burnintest_types.go")
	if err != nil {
		t.Fatalf("reading burnintest_types.go: %v", err)
	}
	re := regexp.MustCompile(`(?m)^\s*// \+kubebuilder:validation:Enum=([a-z;]+)\s*$\n\s*Vendor string`)
	m := re.FindSubmatch(raw)
	if m == nil {
		t.Fatal("found no +kubebuilder:validation:Enum marker directly above VendorImage.Vendor; " +
			"the pattern this guard reads has moved")
	}
	marker := strings.Split(string(m[1]), ";")
	sort.Strings(marker)

	want := append([]string(nil), contract.AcceleratorVendors...)
	sort.Strings(want)

	if strings.Join(marker, ",") != strings.Join(want, ",") {
		t.Errorf("VendorImage.Vendor's CRD enum is %v but contract.AcceleratorVendors is %v; "+
			"the apiserver would accept or refuse a vendor name this project's own vocabulary "+
			"disagrees with it about", marker, want)
	}
}
