// SPDX-License-Identifier: Apache-2.0
// Copyright the Glimmer authors.

package main

import (
	"fmt"
	"testing"
)

// These fixtures are hand-built from the DOCUMENTED, versioned schema in
// core/include/xpum_structs.h (xpum_diag_component_info_t, xpum_diag_task_result_t,
// xpum_result_t) at intel/xpumanager's v1.3.x line — see decide.go's header
// comment and docs/dev/spike-intel-xpu-manager.md. They are NOT captured real
// output; nobody has run xpu-smi diag on real hardware yet. That is a real gap,
// not a hidden one: new-testkind-playbook.md is explicit that a parser test
// normally waits for a hardware capture, and the reason these exist anyway is
// that the property under test here is entirely about the SCHEMA's shape (do we
// read "errno" correctly, do we fail closed on "finished: false") rather than
// about any specific measured value — the same category of thing arch_match_test.cc
// tests for compute-smoke without a GPU. Do not extend these fixtures with
// invented FIELD VALUES (a made-up temperature, a made-up bandwidth number) —
// only with shapes traceable to the header.

func TestDecideEveryComponentPasses(t *testing.T) {
	raw := []byte(`{
		"level": 1,
		"component_list": [
			{"component_type": 0, "finished": true, "result": 1, "message": "pass"},
			{"component_type": 1, "finished": true, "result": 1, "message": "pass"}
		]
	}`)
	d := decide(raw)
	if d.Verdict != verdictPass {
		t.Fatalf("Verdict = %v, want verdictPass", d.Verdict)
	}
	if d.Total != 2 || d.Passed != 2 || d.Failed != 0 || d.Incomplete != 0 {
		t.Errorf("counts = %+v, want Total=2 Passed=2", d)
	}
}

func TestDecideOneComponentFails(t *testing.T) {
	raw := []byte(`{
		"level": 2,
		"component_list": [
			{"component_type": 0, "finished": true, "result": 1, "message": "pass"},
			{"component_type": 12, "finished": true, "result": 2, "message": "memory error detected"}
		]
	}`)
	d := decide(raw)
	if d.Verdict != verdictFail {
		t.Fatalf("Verdict = %v, want verdictFail", d.Verdict)
	}
	if d.Failed != 1 || d.Passed != 1 {
		t.Errorf("counts = %+v, want Failed=1 Passed=1", d)
	}
}

// The landmine issue #172 named: a component that never finished must NOT be
// silently read as a pass just because nothing else failed.
func TestDecideUnfinishedComponentIsErrorNotPass(t *testing.T) {
	raw := []byte(`{
		"level": 3,
		"component_list": [
			{"component_type": 0, "finished": true, "result": 1, "message": "pass"},
			{"component_type": 10, "finished": false, "result": 0, "message": ""}
		]
	}`)
	d := decide(raw)
	if d.Verdict != verdictError {
		t.Fatalf("Verdict = %v, want verdictError for an unfinished component", d.Verdict)
	}
	if d.Incomplete != 1 {
		t.Errorf("Incomplete = %d, want 1", d.Incomplete)
	}
}

// xpumDiagResultUnknown (0) is the zero value of the JSON int field, so a
// component that is marked finished=true but never actually recorded a real
// PASS/FAIL result must still be treated as unjudged, not as a pass-by-default.
func TestDecideFinishedButUnknownResultIsError(t *testing.T) {
	raw := []byte(`{
		"level": 1,
		"component_list": [
			{"component_type": 3, "finished": true, "result": 0, "message": "unexpected state"}
		]
	}`)
	d := decide(raw)
	if d.Verdict != verdictError {
		t.Fatalf("Verdict = %v, want verdictError for a finished-but-UNKNOWN component", d.Verdict)
	}
}

// The exit-code landmine itself: comlet_base.h's setExitCodeByJson only fires
// on a top-level "errno", which is exactly what a bad -d/-l argument or a
// task that never started produces. This is what a wrapper that blindly
// trusted the raw process exit code would miss in the OTHER direction from a
// silent pass — this fixture is the case that legitimately IS a machinery
// error, and must be reported as one via the same field DCGM-style wrappers
// already handle by inspecting JSON rather than $?.
func TestDecideErrnoIsMachineryErrorNotAVerdict(t *testing.T) {
	errno := 15 // XPUM_RESULT_DIAGNOSTIC_TASK_NOT_FOUND
	raw := []byte(fmt.Sprintf(`{"errno": %d, "error": "diagnostic task not found"}`, errno))
	d := decide(raw)
	if d.Verdict != verdictError {
		t.Fatalf("Verdict = %v, want verdictError", d.Verdict)
	}
	if d.Reason != "diagnostic task not found" {
		t.Errorf("Reason = %q, want the JSON error string surfaced verbatim", d.Reason)
	}
}

func TestDecideNoComponentsIsError(t *testing.T) {
	d := decide([]byte(`{"level": 1, "component_list": []}`))
	if d.Verdict != verdictError {
		t.Fatalf("Verdict = %v, want verdictError for an empty component_list", d.Verdict)
	}
}

func TestDecideUnparseableJSONIsError(t *testing.T) {
	d := decide([]byte(`not json`))
	if d.Verdict != verdictError {
		t.Fatalf("Verdict = %v, want verdictError for unparseable output", d.Verdict)
	}
}
