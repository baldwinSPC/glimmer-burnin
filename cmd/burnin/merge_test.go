package main

import (
	"strings"
	"testing"
	"time"

	api "github.com/baldwinSPC/glimmer-burnin/api/v1alpha1"
	"github.com/baldwinSPC/glimmer-burnin/pkg/localrun"
)

func rankRec(rank, nranks int, node string, results ...localrun.TestResult) rankRecord {
	return rankRecord{
		Rank: rank, NRanks: nranks, Node: node,
		StartedAt: time.Now().Add(-time.Minute), FinishedAt: time.Now(),
		Results: results,
	}
}

func passed(name string, metrics map[string]string) localrun.TestResult {
	return localrun.TestResult{
		Name: name, Kind: "nccl", Scope: api.ScopeGroup, Phase: api.RunPassed,
		Metrics:  metrics,
		Attempts: []localrun.Attempt{{Attempt: 1, Phase: api.RunPassed, ExitCode: 0}},
	}
}

// THE REFUSAL, and the reason this command exists at all: a collective is only
// MEASURED if every rank took part. Rendering a verdict from the ranks that
// happened to report would certify machines nobody looked at.
func TestMergeRefusesAPartialCollective(t *testing.T) {
	records := []rankRecord{
		rankRec(0, 4, "spark-a", passed("nccl", map[string]string{"busBandwidthGBs": "97.2"})),
		rankRec(1, 4, "spark-b", passed("nccl", nil)),
	}

	_, err := mergeRanks(records, 4)
	if err == nil {
		t.Fatal("two of four ranks produced a verdict; a partial collective has none")
	}
	// The refusal owes the person a list. "The collective was not measured" is
	// not something anyone can act on.
	for _, want := range []string{"rank 2", "rank 3", "2 of 4"} {
		if !strings.Contains(err.Error(), want) {
			t.Errorf("the refusal does not name %q: %v", want, err)
		}
	}
}

// A full turnout folds into ONE result naming EVERY node — the verdict is about
// the collective, and attributing it to one endpoint sends an engineer to the
// wrong rack.
func TestMergeProducesOneResultNamingEveryNode(t *testing.T) {
	records := []rankRecord{
		rankRec(0, 3, "spark-a", passed("nccl", map[string]string{"busBandwidthGBs": "97.2"})),
		rankRec(1, 3, "spark-b", passed("nccl", map[string]string{"busBandwidthGBs": "40.1"})),
		rankRec(2, 3, "spark-c", passed("nccl", map[string]string{"busBandwidthGBs": "41.0"})),
	}

	rep, err := mergeRanks(records, 3)
	if err != nil {
		t.Fatalf("mergeRanks: %v", err)
	}
	if len(rep.Results) != 1 {
		t.Fatalf("got %d results, want exactly 1 — a collective has one verdict", len(rep.Results))
	}
	got := rep.Results[0]
	if len(got.Nodes) != 3 {
		t.Errorf("Nodes = %v, want all three", got.Nodes)
	}
	if got.Phase != api.RunPassed {
		t.Errorf("phase = %v (%s), want Passed", got.Phase, got.Message)
	}
	// busBandwidthGBs is Collective: rank 0 is its reporting rank, and the
	// other ranks' lower figures are not competing measurements of the group.
	if got.Metrics["busBandwidthGBs"] != "97.2" {
		t.Errorf("busBandwidthGBs = %q, want rank 0's 97.2", got.Metrics["busBandwidthGBs"])
	}
	if !strings.Contains(got.Message, "COLLECTIVE") {
		t.Errorf("the message must state the verdict's subject before its evidence: %q", got.Message)
	}
}

// A rank that ran but has no result for this test contributes a NIL result, not
// an absent entry — the difference the turnout check turns into a refusal.
func TestMergeARankMissingThisTestIsNotAPass(t *testing.T) {
	records := []rankRecord{
		rankRec(0, 2, "spark-a", passed("nccl", map[string]string{"busBandwidthGBs": "97.2"})),
		rankRec(1, 2, "spark-b"), // ran, but produced no result for "nccl"
	}

	rep, err := mergeRanks(records, 2)
	if err != nil {
		t.Fatalf("mergeRanks: %v", err)
	}
	if rep.Results[0].Phase != api.RunError {
		t.Errorf("phase = %v, want Error — rank 1 never measured this collective, and a Pass would "+
			"certify it on rank 0's evidence", rep.Results[0].Phase)
	}
}

// Two files claiming one rank are two machines that both believed they were it,
// or one record copied twice. Either way the collective did not run the way the
// merge would describe it.
func TestMergeRefusesADuplicateRank(t *testing.T) {
	dir := t.TempDir()
	rec := rankRec(0, 2, "spark-a", passed("nccl", nil))
	if err := writeJSON(dir+"/ranks/rank-00.json", rec); err != nil {
		t.Fatal(err)
	}
	rec.Node = "spark-b"
	if err := writeJSON(dir+"/rank-00.json", rec); err != nil {
		t.Fatal(err)
	}

	if _, err := loadRankRecords([]string{dir}); err == nil {
		t.Fatal("two records claiming rank 0 were accepted")
	} else if !strings.Contains(err.Error(), "rank 0 appears twice") {
		t.Errorf("the refusal must say which rank: %v", err)
	}
}

// Records that disagree about the size of their own collective did not run the
// same collective, and one verdict over them would describe a group that never
// existed.
func TestExpectedRanksRefusesDisagreement(t *testing.T) {
	records := []rankRecord{rankRec(0, 4, "a"), rankRec(1, 8, "b")}
	if _, err := expectedRanks(0, records); err == nil {
		t.Fatal("records claiming 4 and 8 ranks were merged")
	}

	// --nranks wins when given: if rank 7's machine never ran, nothing in the
	// directory knows rank 7 was expected.
	if n, err := expectedRanks(8, []rankRecord{rankRec(0, 4, "a")}); err != nil || n != 8 {
		t.Errorf("expectedRanks(8, ...) = %d, %v; the flag must win over the records' claim", n, err)
	}
}

// A rank record's own claim is used when the flag is absent and the records
// agree.
func TestExpectedRanksFallsBackToTheRecords(t *testing.T) {
	n, err := expectedRanks(0, []rankRecord{rankRec(0, 2, "a"), rankRec(1, 2, "b")})
	if err != nil || n != 2 {
		t.Errorf("got %d, %v; want 2", n, err)
	}
}

// A failing rank fails the collective however many ranks reported: a rank that
// positively failed established something, and a silent peer does not erase it.
func TestMergeAFailingRankFailsTheCollective(t *testing.T) {
	failing := localrun.TestResult{
		Name: "nccl", Kind: "nccl", Scope: api.ScopeGroup, Phase: api.RunFailed,
		Attempts: []localrun.Attempt{{Attempt: 1, Phase: api.RunFailed, ExitCode: 1}},
	}
	records := []rankRecord{
		rankRec(0, 2, "spark-a", passed("nccl", map[string]string{"busBandwidthGBs": "97.2"})),
		rankRec(1, 2, "spark-b", failing),
	}

	rep, err := mergeRanks(records, 2)
	if err != nil {
		t.Fatalf("mergeRanks: %v", err)
	}
	if rep.Results[0].Phase != api.RunFailed {
		t.Errorf("phase = %v, want Failed", rep.Results[0].Phase)
	}
	if rep.Phase != api.RunFailed {
		t.Errorf("run phase = %v, want Failed", rep.Phase)
	}
}

// A rank record round-trips through the file the runner actually writes.
func TestRankRecordRoundTrips(t *testing.T) {
	dir := t.TempDir()
	want := rankRec(1, 2, "spark-b", passed("nccl", map[string]string{"busBandwidthGBs": "40.1"}))
	if err := writeJSON(dir+"/ranks/"+rankRecordName(1), want); err != nil {
		t.Fatal(err)
	}

	got, err := loadRankRecords([]string{dir})
	if err != nil {
		t.Fatalf("loadRankRecords: %v", err)
	}
	if len(got) != 1 || got[0].Rank != 1 || got[0].Node != "spark-b" || got[0].NRanks != 2 {
		t.Fatalf("round-trip lost something: %+v", got)
	}
	if got[0].Results[0].Metrics["busBandwidthGBs"] != "40.1" {
		t.Errorf("metrics did not survive: %+v", got[0].Results)
	}
}

// A PASSING COLLECTIVE MUST EXIT 0, and this is the bug that says why the rank
// record carries `required` at all.
//
// codeFor reads which tests decide the run from the PLAN, and a merge has no
// plan — it has records written on other machines. Handing it an empty plan
// made every result optional, so a collective that passed exited 2, "nothing
// was judged". A CI job reads that as hardware nobody measured, which is the
// one thing a green run must never look like.
//
// Found by running the binary, not by a test, which is why there is one now.
func TestMergeAPassingCollectiveExitsZero(t *testing.T) {
	records := []rankRecord{
		rankRec(0, 2, "spark-a", passed("nccl", map[string]string{"busBandwidthGBs": "97.2"})),
		rankRec(1, 2, "spark-b", passed("nccl", map[string]string{"busBandwidthGBs": "40.1"})),
	}
	for i := range records {
		records[i].Required = map[string]bool{"nccl": true}
	}

	rep, err := mergeRanks(records, 2)
	if err != nil {
		t.Fatalf("mergeRanks: %v", err)
	}
	if got := codeFor(rep, requiredPlan(records)); got != exitPass {
		t.Errorf("exit = %d, want %d — a passing collective that exits 'nothing was judged' reads as "+
			"hardware nobody measured", got, exitPass)
	}
}

// An OPTIONAL collective that fails is recorded and does not condemn the run,
// which is the other half of carrying `required`: assuming every merged test is
// required would produce a hardware verdict the profile did not ask for.
func TestMergeAnOptionalCollectiveDoesNotDecideTheRun(t *testing.T) {
	failing := localrun.TestResult{
		Name: "nccl", Kind: "nccl", Scope: api.ScopeGroup, Phase: api.RunFailed,
		Attempts: []localrun.Attempt{{Attempt: 1, Phase: api.RunFailed, ExitCode: 1}},
	}
	records := []rankRecord{
		rankRec(0, 2, "spark-a", failing),
		rankRec(1, 2, "spark-b", failing),
	}
	for i := range records {
		records[i].Required = map[string]bool{"nccl": false}
	}

	rep, err := mergeRanks(records, 2)
	if err != nil {
		t.Fatalf("mergeRanks: %v", err)
	}
	if rep.Results[0].Phase != api.RunFailed {
		t.Errorf("the result is still Failed and recorded: %v", rep.Results[0].Phase)
	}
	if got := codeFor(rep, requiredPlan(records)); got == exitFail {
		t.Error("an optional collective's failure reached the exit code as a hardware verdict")
	}
}
