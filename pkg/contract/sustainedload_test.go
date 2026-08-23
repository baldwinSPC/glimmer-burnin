package contract_test

import (
	"testing"

	"github.com/baldwinSPC/glimmer-burnin/pkg/contract"
)

// Which kinds hold the part under load, stated once and exhaustively.
//
// The answer has one consumer and a real cost either way. The burn-in operator
// writes a heat declaration on every node it is loading, and a node-local
// thermal watchdog reads it as permission to skip the drain it would otherwise
// run — so a kind wrongly listed here gives away a safety action on a node
// nothing is heating, and a kind wrongly left out is #280 again for that
// kind: the drain evicts it and the hardware comes back unjudged.
//
// Every row states its reason from the RUNNER, because the property is the
// runner's and not the name's.
func TestDrivesSustainedLoad(t *testing.T) {
	for _, tc := range []struct {
		kind  contract.TestKind
		holds bool
		why   string
	}{
		{contract.KindThermalSoak, true, "the shared duration-honouring load wrapper (soak_core.cuh); it exists to hold the part at temperature"},
		{contract.KindGPUBurn, true, "the same wrapper: sustained FP compute for the whole window"},
		{contract.KindPowerSwing, true, "the same wrapper, alternated on a duty cycle rather than held; the ON phases are real GEMM heat the watchdog must not read as a fault"},
		{contract.KindClockProbe, true, "holds a known, steady, clock-bound load — that is how it judges sustained clocks at all"},
		{contract.KindFabricSoak, true, "the ib-write-bw measurement iterated over hours, to find what fails once warm"},
		{contract.KindMemoryStress, true, "stressapptest for the whole window; host RAM, which the watchdog also judges"},
		{contract.KindDCGMDiag, true, "levels 3 and 4 run targeted_stress and sm_stress for ~15 min, and the level is a runner env var this operator must not read"},

		{contract.KindComputeSmoke, false, "one GEMM in milliseconds; burst-only"},
		{contract.KindFingerprintProbe, false, "reads sysfs once; there is no work at all"},
		{contract.KindHostHealth, false, "passive by construction — it applies no load and measures no throughput"},
		{contract.KindMemoryBW, false, "nvbandwidth takes a bandwidth measurement and stops"},
		{contract.KindGemmSweep, false, "one execution per precision; a correctness and throughput measurement, not a held load"},
		{contract.KindDiskIO, false, "storage throughput and latency; it does not hold the accelerator or the CPU package"},
		{contract.KindNCCL, false, "a collective bandwidth measurement over the fabric"},
		{contract.KindIBWriteBW, false, "answers whether the link works right now; fabric-soak is the one that iterates"},
		{contract.KindGPUDirect, false, "validates the GPUDirect RDMA path"},
		{contract.KindTCPBaseline, false, "a plain-TCP throughput measurement, and one that refuses the management path"},
	} {
		t.Run(string(tc.kind), func(t *testing.T) {
			if got := tc.kind.DrivesSustainedLoad(); got != tc.holds {
				t.Errorf("DrivesSustainedLoad() = %v, want %v — %s", got, tc.holds, tc.why)
			}
		})
	}
}

// EVERY KIND THIS PROJECT SHIPS MUST BE CLASSIFIED DELIBERATELY, because
// forgetting is the failure mode that matters: a new soak kind nobody thought
// about answers false, and the first time anyone finds out is a soak coming back
// as Error (137).
func TestEveryKindDeclaresWhetherItHoldsLoad(t *testing.T) {
	classified := map[contract.TestKind]bool{
		contract.KindThermalSoak: true, contract.KindGPUBurn: true,
		contract.KindPowerSwing: true,
		contract.KindClockProbe: true, contract.KindFabricSoak: true,
		contract.KindMemoryStress: true, contract.KindDCGMDiag: true,
		contract.KindComputeSmoke: true, contract.KindFingerprintProbe: true,
		contract.KindHostHealth: true, contract.KindMemoryBW: true,
		contract.KindGemmSweep: true, contract.KindDiskIO: true,
		contract.KindNCCL: true, contract.KindIBWriteBW: true,
		contract.KindGPUDirect: true, contract.KindTCPBaseline: true,
	}
	for _, k := range contract.BuiltInKinds {
		if !classified[k] {
			t.Errorf("%q is a built-in kind that TestDrivesSustainedLoad does not judge. Decide whether "+
				"its runner HOLDS a load for its window — if it does, add it to sustainedLoadKinds too, "+
				"or its soaks will be drained and come back unjudged", k)
		}
	}
}

// A kind this operator has never heard of is somebody else's runner, and this
// answer would suppress a safety action on their behalf. It is a worse guess to
// get wrong than BurstOnly's, so it is not guessed: a site running a custom soak
// asserts the hold on the node itself.
func TestDrivesSustainedLoad_CustomAndUnknownKindsAreNotGuessedAt(t *testing.T) {
	for _, k := range []contract.TestKind{contract.KindCustom, "somebody-elses-soak", ""} {
		if k.DrivesSustainedLoad() {
			t.Errorf("%q claims to hold a load — this project cannot know that about an image it "+
				"does not ship, and the claim suppresses a thermal drain", k)
		}
	}
}
