// SPDX-License-Identifier: Apache-2.0
// Copyright the Glimmer authors.

// This file holds the part of the runner that needs no Intel GPU to be
// correct: turning `xpu-smi diag -j`'s JSON into a verdict. It is exercised by
// decide_test.go against fixtures built from Intel's PUBLISHED, versioned C
// API header (xpum_structs.h — xpum_diag_task_result_t, xpum_result_t), not
// from captured stdout, because nobody has run this against real hardware yet.
// See README.md and docs/dev/spike-intel-xpu-manager.md (#172) for why that
// distinction matters here and what would upgrade this file's confidence.
package main

import "encoding/json"

// Component result codes, from xpum_diag_task_result_enum in
// core/include/xpum_structs.h (intel/xpumanager, v1.3.x line). The zero value
// is UNKNOWN — a component that never reached a verdict, not a component that
// passed — which is exactly the trap this file exists to avoid: a bug that
// forgets to check "finished" would silently read an incomplete component as
// XPUM_DIAG_RESULT_UNKNOWN == 0 and, if code elsewhere ever treated the zero
// value as "no problem", certify a device nothing actually measured.
const (
	xpumDiagResultUnknown = 0
	xpumDiagResultPass    = 1
	xpumDiagResultFail    = 2
)

// diagComponent is one entry in the JSON's component_list[], matching
// xpum_diag_component_info_t.
type diagComponent struct {
	ComponentType int    `json:"component_type"`
	Finished      bool   `json:"finished"`
	Result        int    `json:"result"`
	Message       string `json:"message"`
}

// diagResponse is xpu-smi/xpumcli's `diag -j` top-level JSON shape, as far as
// this file relies on it. Fields this runner does not use are omitted
// deliberately — encoding/json ignores what it is not told about, and naming
// only what is consumed keeps this struct honest about what was actually
// verified against the header rather than guessed from a CLI table example.
type diagResponse struct {
	// Errno and Error carry a MACHINERY failure — bad arguments, no such
	// device, the task never started — the same way every xpum_result_t
	// non-zero value does across the whole CLI, not a verdict about the part.
	// comlet_base.h's setExitCodeByJson reads exactly this pair and nothing
	// else; the process's own exit code does NOT reflect component results
	// (see decide below), so this struct has to carry that signal itself.
	Errno *int   `json:"errno"`
	Error string `json:"error"`

	Level         int             `json:"level"`
	ComponentList []diagComponent `json:"component_list"`
}

// verdict is this runner's OWN small enum, not xpum's — kept separate so a
// future change to Intel's schema cannot silently change what this runner's
// exit code means.
type verdict int

const (
	verdictError verdict = iota
	verdictPass
	verdictFail
)

// decision is what decide returns: enough to both choose an exit code and
// print metrics that explain it.
type decision struct {
	Verdict verdict
	Reason  string // human text for the marker / Message
	Total   int
	Passed  int
	Failed  int
	// Incomplete counts components that never reached PASS or FAIL — either
	// Finished is false, or Result is still xpumDiagResultUnknown despite
	// Finished being true. Both are "we asked and got no usable answer",
	// which this runner treats identically: see decide's comment on why
	// UNKNOWN cannot be read as a pass.
	Incomplete int
}

// decide is the ENTIRE verdict logic, and the reason it is worth having asked
// for so plainly:
//
// xpu-smi/xpumcli diag's own PROCESS EXIT CODE does not encode whether any
// component passed or failed. Verified by reading cli/src/comlet_base.h
// (setExitCodeByJson only fires when the JSON carries a top-level "errno" —
// i.e. an argument or task-launch failure) and cli/src/comlet_diagnostic.cpp
// (the success path — getTableResult / showDeviceDiagnostic — never inspects
// a component's Result field and never calls setExitCodeByJson after a
// completed run). A wrapper that simply proxied the process exit code would
// report Pass for a genuinely failed diagnostic, provided the process itself
// did not error out first — the exact "dcgmi exits 226 on a warning" trap
// issue #172 named up front, in the opposite direction. This runner never
// trusts that exit code; it parses the JSON itself.
//
// UNKNOWN (xpumDiagResultUnknown, the zero value) is never read as a pass.
// The registry's own rule is "a runner may only declare what it positively
// established" — an UNKNOWN component has established nothing, and there is
// no separate "not applicable to this hardware" value in the vendor's own
// enum for diag components (unlike DCGM's distinguishable skip case), so this
// runner cannot respell UNKNOWN as a declared Skip without inventing a
// distinction the vendor's API does not offer. It is reported as unjudged
// (Error) instead of a false Pass or a false Fail.
func decide(raw []byte) decision {
	var resp diagResponse
	if err := json.Unmarshal(raw, &resp); err != nil {
		return decision{Verdict: verdictError, Reason: "could not parse xpu-smi diag JSON output: " + err.Error()}
	}
	if resp.Errno != nil && *resp.Errno != 0 {
		msg := resp.Error
		if msg == "" {
			msg = "xpu-smi diag reported a machinery error with no message"
		}
		return decision{Verdict: verdictError, Reason: msg}
	}
	if len(resp.ComponentList) == 0 {
		return decision{Verdict: verdictError, Reason: "xpu-smi diag reported no components; the hardware was not measured"}
	}

	d := decision{Total: len(resp.ComponentList)}
	for _, c := range resp.ComponentList {
		switch {
		case !c.Finished:
			d.Incomplete++
		case c.Result == xpumDiagResultFail:
			d.Failed++
		case c.Result == xpumDiagResultPass:
			d.Passed++
		default: // xpumDiagResultUnknown, or any value this file does not recognise
			d.Incomplete++
		}
	}

	switch {
	case d.Failed > 0:
		d.Verdict = verdictFail
		d.Reason = "at least one diagnostic component reported FAIL"
	case d.Incomplete > 0:
		d.Verdict = verdictError
		d.Reason = "at least one diagnostic component never reached a PASS/FAIL result"
	default:
		d.Verdict = verdictPass
		d.Reason = "every diagnostic component reported PASS"
	}
	return d
}
