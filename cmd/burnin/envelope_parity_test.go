// SPDX-License-Identifier: Apache-2.0
// Copyright the Glimmer authors.

package main

import (
	"reflect"
	"sort"
	"testing"
	"time"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"

	api "github.com/baldwinSPC/glimmer-burnin/api/v1alpha1"
	"github.com/baldwinSPC/glimmer-burnin/internal/sink"
	"github.com/baldwinSPC/glimmer-burnin/pkg/contract"
	"github.com/baldwinSPC/glimmer-burnin/pkg/localrun"
)

// TWO ASSEMBLERS, ONE LEDGER.
//
// TestBothDispatchersDeriveTheSameDeliveryID above proves the two envelope
// assemblers agree about IDENTITY. This proves they agree about CONTENT — or
// rather, that where they disagree, the disagreement is written down and
// exact.
//
// The history that makes this worth a guard: every widening of the envelope
// must be mirrored by hand in internal/sink/envelope.go and here, and nothing
// compared them. Segmented soaks shipped with status the envelope silently
// lacked (#229), #228 was its sibling — and writing this very test found a
// third: pkg/localrun EXECUTED Pair scope while delivering Scope:"", because
// TestResult had no Scope field for toContractResult to copy. A consumer
// switching on scope read a bare-metal link verdict as two single-node claims.
//
// Two rules, both directions exact:
//
//  1. The OPERATOR must be able to populate EVERY field of the contract.
//     There is no operator exception list, on purpose: a field of the
//     envelope that no dispatcher can fill is a promise nobody keeps, and the
//     moment someone widens contract.Envelope without teaching the operator's
//     assembler to fill it, this fails.
//  2. The CLI must populate exactly ALL-minus-the-ledger. When the CLI gains
//     segments, variants or artifacts, the ledger entry must be REMOVED —
//     this fails until it is, so support arriving is as visible as support
//     missing.
var cliDoesNotDeliver = map[string][]string{
	"Envelope": {
		// No apiserver, no cluster: the CLI mints a synthetic run identity in
		// namespace "local", and inventing a cluster ref would claim an
		// apiserver that never existed.
		"Cluster",
		// Baseline mode, cancellation and checkpoints are run-lifecycle
		// features the local engine does not have: a local run is one shot,
		// driven by a person, cancelled by ^C.
		"Baseline",
		"CancelReason",
		"CheckpointSequence",
	},
	"TestResult": {
		// Variants are expanded by the operator's profile resolution, which
		// the CLI does not implement yet.
		"VariantAxes",
		// Segmented soaks are pod-lifecycle machinery.
		"Segments",
		// The artifact channel rides in a run-owned ConfigMap.
		"Artifacts",
	},
}

// environmentDependent are fields the CLI fills from a live probe of the host
// it runs on, so a test cannot assert them either way without inheriting the
// CI machine's hardware. The operator side still proves they are deliverable.
var environmentDependent = map[string]bool{"Fingerprint": true}

func TestTheTwoAssemblersAgreeAboutEveryContractField(t *testing.T) {
	// ── The operator's side: three deliveries whose union must be total. ──
	// One envelope cannot carry everything: CancelReason requires a cancelled
	// run, CheckpointSequence requires a checkpoint delivery. Three legal
	// deliveries together must leave nothing unpopulated.
	opRun := maximalRun()
	cluster := &contract.ClusterRef{Name: "spark-fleet", UID: "cl-uid"}
	opEnvs := []*contract.Envelope{
		sink.EnvelopeFor(opRun, "acceptance", contract.ReasonPhaseChanged,
			sink.PhaseKey(opRun.Status.Phase), time.Now(), cluster, true),
		sink.EnvelopeFor(opRun, "acceptance", contract.ReasonCheckpoint,
			sink.CheckpointKey(3), time.Now(), cluster, false),
	}

	opEnvelope := map[string]bool{}
	opResult := map[string]bool{}
	for _, e := range opEnvs {
		union(opEnvelope, populated(*e))
		if len(e.Results) > 0 {
			union(opResult, populated(e.Results[0]))
		}
	}

	if missing := unpopulated(contract.Envelope{}, opEnvelope, nil); len(missing) > 0 {
		t.Errorf("the operator's assembler cannot populate Envelope fields %v; a contract field no "+
			"dispatcher fills is a promise nobody keeps — teach internal/sink/envelope.go to fill "+
			"them or remove them from the contract", missing)
	}
	if missing := unpopulated(contract.TestResult{}, opResult, nil); len(missing) > 0 {
		t.Errorf("the operator's assembler cannot populate TestResult fields %v (see above)", missing)
	}

	// ── The CLI's side: one delivery, judged against the ledger exactly. ──
	cliEnv, err := EnvelopeFor(maximalReport())
	if err != nil {
		t.Fatalf("EnvelopeFor: %v", err)
	}
	cliEnvelope := populated(cliEnv)
	cliResult := populated(cliEnv.Results[0])

	checkLedger(t, "Envelope", contract.Envelope{}, cliEnvelope, cliDoesNotDeliver["Envelope"])
	checkLedger(t, "TestResult", contract.TestResult{}, cliResult, cliDoesNotDeliver["TestResult"])
}

// checkLedger asserts populated == all-fields − ledger, in BOTH directions.
func checkLedger(t *testing.T, kind string, zero any, got map[string]bool, ledger []string) {
	t.Helper()
	excused := map[string]bool{}
	for _, f := range ledger {
		excused[f] = true
	}
	typ := reflect.TypeOf(zero)
	for i := 0; i < typ.NumField(); i++ {
		name := typ.Field(i).Name
		switch {
		case environmentDependent[name]:
			// Asserted on the operator side only.
		case excused[name] && got[name]:
			t.Errorf("%s.%s is in the cliDoesNotDeliver ledger but the CLI now populates it — "+
				"support arrived; remove the ledger entry so the ledger stays exact", kind, name)
		case !excused[name] && !got[name]:
			t.Errorf("%s.%s is populated by the operator but not by the CLI, and is not in the "+
				"cliDoesNotDeliver ledger. Either copy it in toContractResult/EnvelopeFor or add it "+
				"to the ledger WITH the reason it cannot be delivered locally — an undocumented gap "+
				"is how a bare-metal pair verdict shipped without its scope", kind, name)
		}
	}
}

func populated(v any) map[string]bool {
	out := map[string]bool{}
	val := reflect.ValueOf(v)
	for i := 0; i < val.NumField(); i++ {
		if !val.Field(i).IsZero() {
			out[val.Type().Field(i).Name] = true
		}
	}
	return out
}

func union(dst, src map[string]bool) {
	for k := range src {
		dst[k] = true
	}
}

func unpopulated(zero any, got map[string]bool, skip map[string]bool) []string {
	var out []string
	typ := reflect.TypeOf(zero)
	for i := 0; i < typ.NumField(); i++ {
		if name := typ.Field(i).Name; !got[name] && !skip[name] {
			out = append(out, name)
		}
	}
	sort.Strings(out)
	return out
}

// maximalRun is a BurnInRun whose status exercises every field the operator's
// assembler can copy. Values are arbitrary; PRESENCE is what is under test.
func maximalRun() *api.BurnInRun {
	started := metav1.NewTime(time.Now().Add(-time.Hour))
	finished := metav1.NewTime(time.Now())
	return &api.BurnInRun{
		ObjectMeta: metav1.ObjectMeta{Namespace: "burnin", Name: "max", UID: types.UID("op-uid")},
		Spec:       api.BurnInRunSpec{CancelReason: "operator asked"},
		Status: api.BurnInRunStatus{
			Phase:       api.RunCancelled,
			Fingerprint: map[string]string{"node-a": "kernel=6.8 arch=arm64"},
			Passed:      1, Failed: 1, Errored: 1, Skipped: 1,
			Results: []api.TestResult{{
				Name: "ib-write-bw", Kind: api.KindIBWriteBW, Scope: api.ScopePair,
				Phase: api.RunFailed, Nodes: []string{"node-a", "node-b"},
				Metrics:          map[string]string{"bandwidthGbps": "97.2"},
				Message:          "bandwidthGbps below floor",
				Unmeasurable:     []string{"eccErrors"},
				VariantAxes:      map[string]string{"measurand": "bandwidth"},
				SegmentsRequired: 4, SegmentsCompleted: 4, SegmentSeconds: 900,
				TruncatedAttempts: 1,
				Violations: []api.Violation{{
					Index: 0, Metric: "bandwidthGbps", Cause: "BelowFloor",
					Kind: "ib-write-bw", Reason: "97.2 < 99",
				}},
				Artifacts: []api.ArtifactRef{{
					Name: "raw", MediaType: "application/json", SizeBytes: 512,
					Digest: "sha256:abc", ConfigMap: "run-artifacts", Key: "raw.json",
				}},
				NotEvaluated: []api.NotEvaluated{{Metric: "eccErrors", Reason: "unmeasurable"}},
				StartedAt:    &started, FinishedAt: &finished,
			}},
		},
	}
}

// maximalReport is the CLI's equivalent: every field pkg/localrun can carry.
func maximalReport() localrun.Report {
	start := time.Now().Add(-time.Minute)
	return localrun.Report{
		Node:      "spark-a",
		Phase:     api.RunFailed,
		StartedAt: start, FinishedAt: time.Now(),
		Summary: localrun.Summary{Passed: 1, Failed: 1, Errored: 1, Skipped: 1},
		Results: []localrun.TestResult{{
			Name: "ib-write-bw", Kind: "ib-write-bw", Scope: api.ScopePair,
			Phase: api.RunFailed, Nodes: []string{"spark-a", "spark-b"},
			StartedAt: start, FinishedAt: time.Now(),
			Metrics: map[string]string{"bandwidthGbps": "97.2"},
			Message: "bandwidthGbps below floor",
			Violations: []api.Violation{{
				Index: 0, Metric: "bandwidthGbps", Cause: "BelowFloor",
				Kind: "ib-write-bw", Reason: "97.2 < 99",
			}},
			NotEvaluated: []api.NotEvaluated{{Metric: "eccErrors", Reason: "unmeasurable"}},
			Unmeasurable: []string{"eccErrors"},
		}},
	}
}

// The guard would be satisfied by a ledger that lists everything, so pin its
// size: the ledger may only shrink. Growing it is a decision about the
// CONTRACT — a new field the CLI will never deliver — and should hurt a
// little: update this count in the same commit, with the reason in the ledger.
func TestTheLedgerOnlyShrinks(t *testing.T) {
	if n := len(cliDoesNotDeliver["Envelope"]) + len(cliDoesNotDeliver["TestResult"]); n > 7 {
		t.Errorf("the cliDoesNotDeliver ledger grew to %d entries; it starts at 7 and is meant to "+
			"shrink as the CLI gains features. If a new contract field genuinely cannot be delivered "+
			"locally, raise this bound in the same commit as the ledger entry and its reason", n)
	}
}
