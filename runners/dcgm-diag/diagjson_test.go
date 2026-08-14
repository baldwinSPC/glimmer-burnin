package main

import (
	"strings"
	"testing"
)

// The DCGM 3.x document shape: everything under "DCGM GPU Diagnostic", tests
// grouped into categories, one result per GPU.
const dcgm3Pass = `
Successfully ran diagnostic for group.
{
  "DCGM GPU Diagnostic" : {
    "test_categories" : [
      {
        "category" : "Deployment",
        "tests" : [
          {"name" : "Denylist",       "results" : [{"gpu_ids":"0","status":"Pass"}]},
          {"name" : "NVML Library",   "results" : [{"gpu_ids":"0","status":"Pass"}]},
          {"name" : "Persistence Mode","results": [{"gpu_ids":"0","status":"Pass"}]}
        ]
      },
      {
        "category" : "Integration",
        "tests" : [
          {"name" : "PCIe", "results" : [{"gpu_ids":"0","status":"Pass"}]}
        ]
      }
    ],
    "version" : "3.3.9",
    "driver_version_detected" : "580.82.09"
  }
}`

// The DCGM 4.x shape: "tests" hoisted to the top level, results described by
// entity, statuses lowercase.
const dcgm4Fail = `{
  "version" : "4.2.3",
  "driver_version" : "580.82.09",
  "tests" : [
    {"name":"software","results":[{"entity_group":"GPU","entity_id":0,"status":"pass"}]},
    {"name":"memory","results":[
       {"entity_group":"GPU","entity_id":0,"status":"pass"},
       {"entity_group":"GPU","entity_id":1,"status":"fail",
        "warnings":[{"warning":"Uncorrectable ECC error detected on GPU 1"}]}
    ]},
    {"name":"pcie","results":[{"entity_group":"GPU","entity_id":0,"status":"pass"}]}
  ]
}`

// The document `dcgmi diag -r 1 -j` actually emitted on a GB10 DGX Spark
// (DCGM 4.2.3, driver 580.82.09), copied verbatim. dcgmi exited 226.
//
// It is kept as a fixture because it is the shape the hand-written 4.x fixture
// above got wrong in two ways, both of which mattered: the tests are nested
// under "DCGM Diagnostic" > "test_categories" rather than hoisted to the top
// level, and the driver version arrives under the human-readable key "Driver
// Version Detected" rather than "driver_version". A parser that reads only the
// invented shape reports no subtests and no driver version, and "no subtests"
// is indistinguishable from "the diagnostic produced nothing" — a false Error
// on every node in the fleet.
const dcgm4GB10 = `{
	"DCGM Diagnostic" :
	{
		"test_categories" :
		[
			{
				"category" : "Deployment",
				"tests" :
				[
					{
						"name" : "software",
						"results" :
						[
							{
								"entity_group" : "GPU",
								"entity_group_id" : 1,
								"entity_id" : 0,
								"status" : "Fail",
								"warnings" :
								[
									{
										"error_category" : 8,
										"error_id" : 29,
										"error_severity" : 5,
										"warning" : "Persistence Mode: Persistence mode for GPU 0 is disabled. Enable persistence mode by running \"nvidia-smi -i <gpuId> -pm 1 \" as root."
									}
								]
							}
						],
						"test_summary" :
						{
							"info" :
							[
								"Error getting the SRAM Threshold Count for GPU 0. Status = 0. Value = 9223372036854775794. SRAM Threshold checking was skipped."
							],
							"status" : "Fail"
						}
					}
				]
			}
		]
	},
	"entity_groups" :
	[
		{
			"entities" :
			[
				{
					"device_id" : "2e12",
					"entity_id" : 0,
					"serial_num" : "<<<NOT_SUPPORTED>>>"
				}
			],
			"entity_group" : "GPU",
			"entity_group_id" : 1
		}
	],
	"metadata" :
	{
		"Driver Version Detected" : "580.82.09",
		"version" : "4.2.3"
	}
}`

func TestParseDiagJSON_RealGB10Document(t *testing.T) {
	res, err := parseDiagJSON(dcgm4GB10)
	if err != nil {
		t.Fatalf("the document real hardware produced is not parseable: %v", err)
	}
	if res.Version != "4.2.3" {
		t.Errorf("Version = %q, want 4.2.3", res.Version)
	}
	if res.DriverVersion != "580.82.09" {
		t.Errorf(`DriverVersion = %q, want 580.82.09 — DCGM 4.2.3 spells the key "Driver Version Detected"`, res.DriverVersion)
	}
	// CHANGED with issue #304, and the change is the point of that issue.
	//
	// This document is a real level-1 run on a healthy GB10. DCGM failed its
	// `software` subtest for two findings: persistence mode being disabled, and
	// a SRAM threshold field it could not read (Value = 9223372036854775794, a
	// sentinel near 2^63). Neither is the part falling short — one is a node
	// setting and one is an absent measurement — so neither may produce a Fail,
	// which is this project's word for "measured, and wrong", and the one phase
	// that is never retried.
	//
	// The subtest still counts as EXECUTED. It ran; it simply did not judge the
	// hardware. Dropping it out of Executed would take the run to the "DCGM
	// skipped everything" branch and report Skipped — "not applicable to this
	// hardware" — which verifies nothing while looking benign.
	c := res.Counts()
	if c.Total != 1 || c.Executed != 1 {
		t.Fatalf("counts = %+v, want one executed subtest", c)
	}
	if c.Failed != 0 {
		t.Errorf("Failed = %d, want 0 — every finding here is a node setting or an unread field, "+
			"and failing a healthy Spark for `nvidia-smi -pm 1` is issue #304", c.Failed)
	}
	if len(c.ExcusedNames) != 1 || c.ExcusedNames[0] != "software" {
		t.Errorf("ExcusedNames = %v, want [software]", c.ExcusedNames)
	}
	if len(c.ConfigFindings) != 1 || !strings.Contains(c.ConfigFindings[0], "Persistence mode") {
		t.Errorf("the persistence-mode finding must survive as a reportable config finding: %v",
			c.ConfigFindings)
	}
	if len(c.UnreadableFindings) != 1 {
		t.Errorf("the unreadable SRAM field must survive as unmeasurable: %v", c.UnreadableFindings)
	}
	if !strings.Contains(strings.Join(res.failReasons, " "), "Persistence mode") {
		t.Errorf("DCGM's own explanation was dropped: %v", res.failReasons)
	}
}

// "<<<NOT_SUPPORTED>>>" appears in the entity list of a perfectly ordinary GB10
// run (it is how DCGM says the part exposes no serial number). If it ever
// reached the unsupported-part matcher, a node with one failing subtest would
// report Skip — excused from acceptance rather than failed by it.
func TestUnsupportedSignature_IgnoresNotSupportedSerialNumber(t *testing.T) {
	if sig := unsupportedSignature(dcgm4GB10); sig != "" {
		t.Errorf("the GB10 document was read as an unsupported-part refusal: %q", sig)
	}
}

func TestParseDiagJSON_ReadsBothDocumentShapes(t *testing.T) {
	res3, err := parseDiagJSON(dcgm3Pass)
	if err != nil {
		t.Fatalf("DCGM 3.x document: %v", err)
	}
	if res3.Version != "3.3.9" || res3.DriverVersion != "580.82.09" {
		t.Errorf("3.x versions = %q / %q", res3.Version, res3.DriverVersion)
	}
	c3 := res3.Counts()
	if c3.Total != 4 || c3.Executed != 4 || c3.Failed != 0 {
		t.Errorf("3.x counts = %+v, want 4 tests all executed and passing", c3)
	}

	res4, err := parseDiagJSON(dcgm4Fail)
	if err != nil {
		t.Fatalf("DCGM 4.x document: %v", err)
	}
	if res4.Version != "4.2.3" || res4.DriverVersion != "580.82.09" {
		t.Errorf("4.x versions = %q / %q", res4.Version, res4.DriverVersion)
	}
	c4 := res4.Counts()
	if c4.Total != 3 || c4.Failed != 1 {
		t.Errorf("4.x counts = %+v, want 3 tests with 1 failing", c4)
	}
	if len(c4.FailedNames) != 1 || c4.FailedNames[0] != "memory" {
		t.Errorf("FailedNames = %v, want [memory]", c4.FailedNames)
	}
	if !strings.Contains(strings.Join(res4.failReasons, " "), "Uncorrectable ECC") {
		t.Errorf("the failure's explanation was dropped: %v", res4.failReasons)
	}
}

// One GPU failing fails the test for the node. Acceptance is a statement about
// the machine, and a node with one bad part is a bad node.
func TestAggregation_OneFailingEntityFailsTheTest(t *testing.T) {
	res, err := parseDiagJSON(dcgm4Fail)
	if err != nil {
		t.Fatal(err)
	}
	if got := res.tests["memory"]; got != statusFail {
		t.Errorf("memory = %v, want fail", got)
	}
}

// A test where one GPU passed and another never reported is NOT a pass. The
// missing result is a measurement we do not have, and treating absence as
// success is exactly the fails-closed rule this project is built on.
func TestAggregation_MissingResultOutranksPass(t *testing.T) {
	const doc = `{"tests":[{"name":"targeted stress","results":[
	  {"entity_id":0,"status":"pass"},
	  {"entity_id":1,"status":"Not Run"}]}]}`
	res, err := parseDiagJSON(doc)
	if err != nil {
		t.Fatal(err)
	}
	if got := res.tests["targeted stress"]; got != statusNotRun {
		t.Fatalf("status = %v, want not run", got)
	}
	if c := res.Counts(); c.NotRun != 1 || c.Executed != 0 {
		t.Errorf("counts = %+v, want the test counted as not run", c)
	}
}

// A status word this runner has never seen must not be optimistically read as a
// pass. A future DCGM release renaming "Pass" would otherwise turn every node
// green without a single test having run.
func TestParseStatus_UnknownIsNotPass(t *testing.T) {
	for _, raw := range []string{"", "Inconclusive", "aborted", "???"} {
		if got := parseStatus(raw); got == statusPass {
			t.Errorf("parseStatus(%q) = pass; an unrecognised word is not evidence of health", raw)
		}
	}
	if parseStatus("Not Run") != statusNotRun {
		t.Error(`parseStatus("Not Run") should be not-run`)
	}
	if parseStatus("SKIPPED") != statusSkip {
		t.Error(`parseStatus("SKIPPED") should be skip`)
	}
}

func TestParseDiagJSON_RejectsNonJSON(t *testing.T) {
	for _, in := range []string{"", "Error: could not open /dev/nvidiactl", "{not json"} {
		if _, err := parseDiagJSON(in); err == nil {
			t.Errorf("parseDiagJSON(%q) accepted output that is not a result document", in)
		}
	}
}

func TestParseDiagJSON_SkipsBannerBeforeTheDocument(t *testing.T) {
	res, err := parseDiagJSON(dcgm3Pass)
	if err != nil {
		t.Fatalf("the banner dcgmi prints before its JSON defeated the parser: %v", err)
	}
	if res.Counts().Total != 4 {
		t.Errorf("Total = %d, want 4", res.Counts().Total)
	}
}

func TestUnsupportedSignature(t *testing.T) {
	hit := unsupportedSignature("Error: Unable to run diagnostic on unsupported GPU 0.\n")
	if hit == "" {
		t.Fatal("DCGM's unsupported-part refusal was not recognised")
	}
	if !strings.Contains(hit, "unsupported GPU") {
		t.Errorf("reason = %q, want DCGM's own wording", hit)
	}
}

// The signature list is only ever consulted for a run that produced no results
// (see run() and verdict()), and this is why: the same words appear in per-test
// prose on healthy hardware. If that guard were ever removed, this test is the
// record of what it was protecting against.
func TestUnsupportedSignature_MatchesProseOnHealthyParts(t *testing.T) {
	if unsupportedSignature("NvLink is not supported by dcgm on this GPU") == "" {
		t.Skip("wording no longer collides; the guard in verdict() is still required")
	}
}
