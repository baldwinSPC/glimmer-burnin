// SPDX-License-Identifier: Apache-2.0
// Copyright the Glimmer authors.

package main

import "strings"

// Not every DCGM failure is a statement about silicon — issue #304.
//
// Measured on a stock GB10 DGX Spark: a level-1 run returns dcgmi exit 226 with
// its single `software` subtest failed, on a node reporting 47 °C, zero Xid,
// zero ECC and zero PCIe replays. The two findings behind that failure were:
//
//	Persistence Mode: Persistence mode for GPU 0 is disabled. Enable
//	  persistence mode by running "nvidia-smi -i <gpuId> -pm 1" as root.
//	Error getting the SRAM Threshold Count for GPU 0. Status = 0.
//	  Value = 9223372036854775794
//
// The first is a node SETTING. The second is DCGM saying it could not READ a
// counter — the value is a sentinel near 2^63, not a measurement. Neither is
// the hardware falling short, and `Fail` is this project's word for exactly
// that: measured, and wrong, and never retried. Reporting them as Fail
// condemns 100% of a healthy Spark fleet for `nvidia-smi -pm 1`.
//
// So findings are classified, and only the ones that are genuinely about the
// part produce a verdict about the part.
//
// THE CLASSIFICATION FAILS CLOSED, which is the property that makes it safe.
// A finding matching nothing below is HARDWARE, so an unfamiliar message keeps
// its Fail. Widening these lists is a deliberate act with a reason; forgetting
// to widen them costs a false Fail, which is recoverable, rather than a false
// pass, which is not.
//
// DCGM aggregates several checks under one subtest name — level 1's `software`
// covers denylist, driver and NVML versions, persistence mode, page retirement,
// inforom and permissions, and page retirement and inforom REALLY ARE the
// hardware. So classification is per FINDING and never per subtest name, and a
// subtest is excused only when every one of its findings is.

// DCGM's OWN CLASSIFICATION, when it gives us one.
//
// Every object in a `warnings` array carries error_severity and error_category
// alongside the prose, and those are DCGM's enums rather than our reading of a
// sentence. Values below are from dcgm_errors.h AS OF DCGM 4.2.3, which is what
// the fleet runs — not inferred, and not to be extended by guess.
//
// The measured GB10 persistence finding is severity 5 / category 8 / error_id
// 29, which is DCGM stating in its own vocabulary what #304 had to infer from
// the wording. Note 8 is SOFTWARE_OTHER, not SOFTWARE_CONFIG: the category does
// not name it a setting, which is why the excusal turns on the error_id.
//
// THE VERSION PIN MATTERS, because this runner ships no DCGM — it drives an
// engine mounted from the site, of a version it never checks, and a runner
// image pinned into a readiness gate outlives the release it was built against.
// These numbers are not stable across all of DCGM's history: at 3.2.x and
// earlier, dcgmErrorSeverity_enum was MONITOR=0 / ISOLATE=1 / UNKNOWN=2, so
// severity 2 meant UNKNOWN where it now means ISOLATE, and reading a 3.2
// document with this table would invert a verdict.
//
// What makes that safe is not luck, it is the `sevOK && catOK` rule in
// findingsFrom: dcgmErrorCategory_enum did not EXIST before 3.3, so a 3.2
// document has no error_category, HasMeta is false, and the finding goes to the
// text path — which is exactly the behaviour we want against an engine whose
// vocabulary we cannot read. The renumbering and the category's introduction
// happened in the same release, so the presence of a category is a proxy for
// "these values mean what this table says". Do not relax that `&&` to an `||`.
const (
	dcgmSeverityNone    = 0
	dcgmSeverityMonitor = 1 // workload can proceed; watch it
	dcgmSeverityIsolate = 2 // cannot perform workload; isolate the GPU
	dcgmSeverityUnknown = 3 // DCGM does not recognise the code
	dcgmSeverityTriage  = 4
	dcgmSeverityConfig  = 5 // "This error can be configured"
	dcgmSeverityReset   = 6 // drain and reset the GPU
)

// Categories 9..15 are DCGM's HARDWARE_* family: THERMAL, MEMORY, NVLINK,
// NVSWITCH, PCIE, POWER, OTHER.
//
// dcgmCategorySoftwareConfig is recorded for the reader but is deliberately NOT
// consulted by classify(). It looks like it should excuse a finding and it must
// not. Category 3 holds seven entries at 4.2.3, and they do not share a
// severity: five are CONFIG (DENYLISTED_DRIVER, CONCURRENT_GPUS,
// EMPTY_GPU_LIST, FILE_CREATE_PERMISSIONS, and the skipped-test entry) and
// DCGM_FR_GPU_OP_MODE is MONITOR — "can perform workload, but needs to be
// monitored", which is an observation about the part. So the category does not
// track severity, and cannot stand alone as an excuse.
//
// The skipped-test entry (error_id 40) is the sharpest one: excusing it as
// configuration would undo #306, which settled that a subtest DCGM declined to
// run is not a subtest that passed.
const (
	dcgmCategoryHardwareFirst = 9
	dcgmCategoryHardwareLast  = 15

	//lint:ignore U1000 documents the enum; see the note above on why it is not consulted
	dcgmCategorySoftwareConfig = 3
)

// finding is one DCGM failure message with whatever metadata came with it.
//
// hasMeta is load-bearing rather than cosmetic: the two findings measured on
// GB10 arrive by DIFFERENT routes. The persistence one is an object in a
// `warnings` array and carries the enums; the unreadable-SRAM one is a bare
// string in `test_summary.info` and carries nothing. So the numeric path cannot
// replace the text path — it can only take precedence where DCGM actually
// supplied an answer.
type finding struct {
	Text     string
	Severity int
	Category int
	ErrorID  int
	HasMeta  bool
}

// findingClass says what kind of claim a DCGM failure message is making.
type findingClass int

const (
	// findingHardware is a claim about the part. The default, deliberately.
	findingHardware findingClass = iota
	// findingConfiguration is a claim about how the node is set up. Fixable
	// with a command, and says nothing about the silicon.
	findingConfiguration
	// findingUnreadable is DCGM reporting that it could not read something.
	// The absence of a measurement, which this project never reports as a
	// value — it is the same rule that puts `n/a` in Result.Unmeasurable.
	findingUnreadable
)

// configurationSignatures mark a finding as a node setting rather than a fault.
//
// Narrow on purpose. Each entry is a message this project has SEEN, not a
// pattern someone thought plausible — a speculative entry here silently
// converts a real hardware failure into a non-verdict.
var configurationSignatures = []string{
	// Verified on GB10, 2026-08-10. DCGM 4.2.3.
	"persistence mode",
	// The denylist and permission checks in the same Deployment suite. Both
	// describe the environment DCGM was run in.
	"is on the denylist",
	"is on the blacklist",
	"run as root",
	"insufficient permission",
	// Verified on GB10, 2026-08-10: nvvs could not create its working file in
	// "/" when run as uid 65532. The runner now passes --home-dir to prevent
	// it, but the signature stays — if it appears for another reason, it is
	// still a statement about the filesystem and not about the part.
	"permissions and os blocks",
	"no permission to create",
}

// unreadableSignatures mark a finding as a failed READ rather than a bad value.
//
// "Error getting the X" is DCGM's own phrasing when a field query fails, and it
// is usually accompanied by a sentinel value near 2^63 — the shape of a counter
// that was never populated rather than one that counted something bad.
var unreadableSignatures = []string{
	"error getting the",
	"could not read",
	"unable to query",
}

// configurationErrorIDs are the DCGM error_ids VERIFIED to name a node setting.
//
// An error_id is the most stable identifier DCGM has: the prose is rewritten
// between releases and the (severity, category) pair is not unique, but the id
// is the enumerator itself. It is what makes an excusal auditable long after
// the run — the same argument CLAUDE.md makes for TestAttempt.Trigger.
//
// This is an ALLOWLIST, and it is short because it holds only what has been
// checked against NVIDIA's own error table. Adding an entry means reading
// dcgm_errors.c and satisfying yourself that the finding cannot be the part.
var configurationErrorIDs = map[int]string{
	// DCGM_FR_PERSISTENCE_MODE. The finding behind #304, measured on GB10.
	29: "DCGM_FR_PERSISTENCE_MODE",
}

// classify decides what one finding claims, constraining it by DCGM's own answer.
//
// The metadata may only ever make this MORE conservative than the prose would
// have been. That is the invariant, and it is the opposite of what the first
// draft did. Reading DCGM's enums as a licence to excuse — `severity == CONFIG
// || category == SOFTWARE_CONFIG` — excused three entries from NVIDIA's own
// table, checked against dcgm_errors.c at 4.2.3:
//
//   - DCGM_FR_ABORTED (CONFIG, SOFTWARE_OTHER): "test was aborted early", a
//     test that measured NOTHING, carrying the identical (5, 8) pair as the
//     persistence-mode finding this feature was built for. (severity,
//     category) is therefore not a unique key and cannot be the whole rule.
//   - DCGM's skipped-test entry, error_id 40 (CONFIG, SOFTWARE_CONFIG): excused as a node
//     setting, which is precisely the acceptance #306 refused.
//   - DCGM_FR_GPU_OP_MODE (MONITOR, SOFTWARE_CONFIG): excused through the
//     category despite a severity this file says outright is not excusable.
//
// Note what is NOT in that list, because the near miss is instructive: an XID.
// NVIDIA files DCGM_FR_XID_ERROR under TRIAGE and both SXID_ERROR and
// UNCONTAINED_ERROR under ISOLATE, so no XID reaches the excusal branch by
// severity, and none ever did. Do not repeat the plausible-sounding claim that
// it does — the argument for this rule stands on the three entries above, which
// were read out of the table rather than imagined.
//
// ORDER MATTERS AND IS FAIL-CLOSED:
//
//  1. A HARDWARE_* category is hardware, whatever the severity says. DCGM
//     naming the subsystem — memory, PCIe, power, thermal — outranks a severity
//     that might otherwise excuse it: a finding marked CONFIG but categorised
//     HARDWARE_MEMORY must never be waved through. This is the one rule the
//     text path cannot express, and the reason the metadata is worth reading.
//  2. Only CONFIG severity may be excused at all. ISOLATE, RESET, TRIAGE,
//     MONITOR, UNKNOWN and NONE are hardware — UNKNOWN especially, because
//     "DCGM does not recognise this code" is the definition of a finding we
//     must not excuse. Note this is deliberately NOT symmetric with category:
//     a SOFTWARE_CONFIG category does not excuse on its own, because NVIDIA
//     files ISOLATE and RESET findings under it too.
//  3. Within CONFIG severity, a verified error_id excuses. Otherwise the prose
//     gets the last word, and only where it matches a signature this project
//     has actually seen — so a CONFIG-severity finding nobody recognises stays
//     a Fail rather than being excused for having the right severity.
//  4. With no metadata at all, the text signatures decide alone. That is not a
//     shortcut: `test_summary.info` entries are bare strings and DCGM attaches
//     no enums to them, and pre-4.x DCGM has no category enum whatsoever.
func classify(f finding) findingClass {
	if !f.HasMeta {
		return classifyFinding(f.Text)
	}
	if f.Category >= dcgmCategoryHardwareFirst && f.Category <= dcgmCategoryHardwareLast {
		return findingHardware
	}
	if f.Severity != dcgmSeverityConfig {
		return findingHardware
	}
	if _, ok := configurationErrorIDs[f.ErrorID]; ok {
		return findingConfiguration
	}
	return classifyFinding(f.Text)
}

// classifyFinding decides what a single DCGM failure message is claiming.
//
// The TEXT path, used only where DCGM supplied no enums. classify() above
// prefers the numeric metadata wherever it exists.
func classifyFinding(text string) findingClass {
	l := strings.ToLower(text)
	for _, sig := range configurationSignatures {
		if strings.Contains(l, sig) {
			return findingConfiguration
		}
	}
	for _, sig := range unreadableSignatures {
		if strings.Contains(l, sig) {
			return findingUnreadable
		}
	}
	return findingHardware
}

// excusedFindings splits a subtest's failure messages by class.
//
// It reports excused=true only when EVERY finding is non-hardware. One
// unrecognised message keeps the whole subtest a Fail: a subtest that failed
// for two reasons, one of them the hardware, has failed.
func excusedFindings(findings []finding) (config, unreadable []string, blocking string, excused bool) {
	if len(findings) == 0 {
		// A failure DCGM gave no reason for is not a failure anyone can excuse.
		return nil, nil, "DCGM reported no reason for the failure", false
	}
	for _, f := range findings {
		r := f.Text
		switch classify(f) {
		case findingConfiguration:
			config = append(config, r)
		case findingUnreadable:
			unreadable = append(unreadable, r)
		default:
			// NAMED, not just counted. Without this an operator sees a Fail on a
			// node whose other findings were all benign and cannot tell which
			// one actually indicts the hardware — and the displayed reason is
			// truncated (report.go), so the blocking finding may not even be
			// visible. Returning it is what makes the fail-closed rule
			// legible instead of merely safe.
			return nil, nil, r, false
		}
	}
	return config, unreadable, "", true
}

// (reasonsFor lived here. It recovered a subtest's messages from the flattened
// failReasons list by matching the "<subtest>: " prefix appendReasons had put
// there. diagResults.findings is now populated in record(), at the node, while
// the document's own nesting is still intact — so the grouping no longer has to
// survive a round trip through a string prefix in order to be recovered from
// it, and the metadata travels with the message instead of being parsed off.)

// orNone renders an empty explanation as something a reader can act on.
//
// DCGM does not always say why it skipped, and "Reasons: " trailing into
// nothing reads as a truncated message rather than as an absent explanation.
func orNone(s string) string {
	if strings.TrimSpace(s) == "" {
		return "DCGM gave no reason"
	}
	return s
}
