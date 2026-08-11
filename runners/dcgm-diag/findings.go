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

// classifyFinding decides what a single DCGM failure message is claiming.
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
func excusedFindings(reasons []string) (config, unreadable []string, blocking string, excused bool) {
	if len(reasons) == 0 {
		// A failure DCGM gave no reason for is not a failure anyone can excuse.
		return nil, nil, "DCGM reported no reason for the failure", false
	}
	for _, r := range reasons {
		switch classifyFinding(r) {
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

// reasonsFor returns the failure messages belonging to one subtest.
//
// appendReasons prefixes each with "<subtest>: ", which is what makes the
// grouping possible — DCGM's own document nests them, but they are flattened by
// the time a verdict is taken.
func reasonsFor(subtest string, all []string) []string {
	prefix := subtest + ": "
	var out []string
	for _, r := range all {
		if strings.HasPrefix(r, prefix) {
			out = append(out, strings.TrimPrefix(r, prefix))
		}
	}
	return out
}
