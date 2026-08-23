// SPDX-License-Identifier: Apache-2.0
// Copyright the Glimmer authors.

package v1alpha1

import "github.com/baldwinSPC/glimmer-burnin/pkg/contract"

// THE ACCEPTANCE VOCABULARY NOW LIVES IN pkg/contract, and these aliases are
// what make that a non-event for every caller.
//
// A Go type alias is the same type, not a conversion: v1alpha1.Threshold IS
// contract.Threshold, so existing code compiles unchanged, the CRD wire format
// is byte-identical, and a consumer may adopt the neutral names at whatever
// pace suits it. See pkg/contract/vocabulary.go for why the types moved (#274,
// and the GEP-0178 amendment).
//
// The kubebuilder markers moved WITH the types. controller-gen's crd generator
// already scans the whole module (`paths=./...`), so it reads them where they
// now live; only the deepcopy generator is scoped to ./api/..., and these are
// flat value types whose slices copy without a generated method.
//
// DO NOT re-declare any of these here. A local definition would silently become
// a different type from the one pkg/verdict evaluates, and the two would agree
// on every test and disagree on a real document.
type (
	TestKind      = contract.TestKind
	Comparison    = contract.Comparison
	Applicability = contract.Applicability
	Threshold     = contract.Threshold
)

// The kind vocabulary. Aliased individually because Go has no constant-block
// alias, and spelled out so `grep KindComputeSmoke api/` still finds it.
const (
	KindGPUBurn          = contract.KindGPUBurn
	KindComputeSmoke     = contract.KindComputeSmoke
	KindDCGMDiag         = contract.KindDCGMDiag
	KindThermalSoak      = contract.KindThermalSoak
	KindPowerSwing       = contract.KindPowerSwing
	KindNCCL             = contract.KindNCCL
	KindIBWriteBW        = contract.KindIBWriteBW
	KindGPUDirect        = contract.KindGPUDirect
	KindCustom           = contract.KindCustom
	KindMemoryBW         = contract.KindMemoryBW
	KindMemoryStress     = contract.KindMemoryStress
	KindHostHealth       = contract.KindHostHealth
	KindClockProbe       = contract.KindClockProbe
	KindFingerprintProbe = contract.KindFingerprintProbe
	KindTCPBaseline      = contract.KindTCPBaseline
	KindDiskIO           = contract.KindDiskIO
	KindFabricSoak       = contract.KindFabricSoak
	KindGemmSweep        = contract.KindGemmSweep
)

// Comparisons and applicability.
const (
	GTE = contract.GTE
	LTE = contract.LTE
	EQ  = contract.EQ
	NEQ = contract.NEQ

	Required             = contract.Required
	RequiredIfMeasurable = contract.RequiredIfMeasurable
)

// BuiltInKinds is the hand-kept list of kinds this project ships a runner for.
var BuiltInKinds = contract.BuiltInKinds
