// SPDX-License-Identifier: Apache-2.0
// Copyright the Glimmer authors.

package group

import (
	"testing"

	"github.com/baldwinSPC/glimmer-burnin/pkg/runner"
)

func res(v runner.Verdict, exit int, metrics map[string]string) *runner.Result {
	return &runner.Result{Verdict: v, ExitCode: exit, Metrics: metrics, Unmeasurable: map[string]bool{}}
}

func ranksOf(results ...*runner.Result) []Rank {
	out := make([]Rank, 0, len(results))
	for i, r := range results {
		out = append(out, Rank{Rank: i, Node: string(rune('a' + i)), Result: r})
	}
	return out
}

// THE RULE WITH THE MOST TEETH: a collective is only measured if every rank
// took part, so a rank that never reported is a rank whose participation nobody
// can vouch for.
func TestVerdict_APartialTurnoutIsNeverAPass(t *testing.T) {
	ranks := ranksOf(res(runner.VerdictPass, 0, nil), nil, res(runner.VerdictPass, 0, nil))

	v, _ := Verdict(ranks)
	if v != runner.VerdictError {
		t.Errorf("verdict = %v, want Error — two ranks passing says nothing about the third, and a "+
			"Pass here would certify a machine nobody looked at", v)
	}
	if got := Missing(ranks); len(got) != 1 || got[0] != 1 {
		t.Errorf("Missing = %v, want [1] — a refusal owes the person the list", got)
	}
}

// Skip needs the full turnout for exactly the same reason Pass does, and it is
// the MORE dangerous of the two to get wrong: a Skip does not fail the run, it
// records "acceptance does not apply to this hardware" and the run settles
// Passed around it.
func TestVerdict_APartialTurnoutIsNeverASkipEither(t *testing.T) {
	ranks := ranksOf(res(runner.VerdictSkip, 2, nil), nil, nil)

	if v, _ := Verdict(ranks); v != runner.VerdictError {
		t.Errorf("verdict = %v, want Error. Rank 0 declaring the test inapplicable before the other "+
			"ranks ever ran would certify every one of them on evidence from one — a runner may only "+
			"declare what it positively established, and rank 0 established something about rank 0", v)
	}
}

// Error and Fail are honoured however many ranks reported: a rank that
// positively erred or positively failed has established something, and a silent
// peer does not erase it.
func TestVerdict_APositiveFailureSurvivesAPartialTurnout(t *testing.T) {
	for _, c := range []struct {
		name string
		want runner.Verdict
		res  *runner.Result
	}{
		{"a fail is a fail", runner.VerdictFail, res(runner.VerdictFail, 1, nil)},
		{"an error is an error", runner.VerdictError, res(runner.VerdictError, 3, nil)},
	} {
		t.Run(c.name, func(t *testing.T) {
			if v, _ := Verdict(ranksOf(res(runner.VerdictPass, 0, nil), c.res, nil)); v != c.want {
				t.Errorf("verdict = %v, want %v", v, c.want)
			}
		})
	}
}

// Error outranks Fail, and both are taken from the LOWEST rank holding them so
// the exit code and the verdict describe the same execution.
func TestVerdict_PrecedenceAndExitCode(t *testing.T) {
	ranks := ranksOf(
		res(runner.VerdictFail, 1, nil),
		res(runner.VerdictError, 3, nil),
		res(runner.VerdictError, 9, nil),
	)
	v, exit := Verdict(ranks)
	if v != runner.VerdictError {
		t.Fatalf("verdict = %v, want Error — a machinery failure means the collective was never measured", v)
	}
	if exit != 3 {
		t.Errorf("exit = %d, want 3 (the LOWEST erring rank's), so the code and the verdict describe "+
			"the same execution", exit)
	}
}

func TestVerdict_AFullTurnoutOfPassesPasses(t *testing.T) {
	if v, _ := Verdict(ranksOf(res(runner.VerdictPass, 0, nil), res(runner.VerdictPass, 0, nil))); v != runner.VerdictPass {
		t.Errorf("verdict = %v, want Pass", v)
	}
}

func TestVerdict_NobodyReported(t *testing.T) {
	if v, _ := Verdict(ranksOf(nil, nil)); v != runner.VerdictError {
		t.Errorf("verdict = %v, want Error", v)
	}
	if v, _ := Verdict(nil); v != runner.VerdictError {
		t.Errorf("an empty group = %v, want Error", v)
	}
}

// The exit code is looked up by RANK NUMBER, never by slice position: a caller
// assembling ranks from files on N machines has no reason to hand them over
// sorted, and indexing would attribute rank 0's exit code to whoever came first.
func TestVerdict_ExitCodeFollowsTheRankNotThePosition(t *testing.T) {
	ranks := []Rank{
		{Rank: 2, Node: "c", Result: res(runner.VerdictPass, 22, nil)},
		{Rank: 0, Node: "a", Result: res(runner.VerdictPass, 0, nil)},
		{Rank: 1, Node: "b", Result: res(runner.VerdictPass, 11, nil)},
	}
	if _, exit := Verdict(ranks); exit != 0 {
		t.Errorf("exit = %d, want rank 0's (0), not the first slice element's", exit)
	}
}

// #121: elapsedS is Sum across windows and Max across ranks — eight ranks
// running 300s took 300s, not 2400s. Aggregation cannot stand in for
// Combination.
func TestElect_UsesTheMetricsOwnCombinationRule(t *testing.T) {
	// busBandwidthGBs is Collective: rank 0 is its reporting rank.
	got, ok := Elect("busBandwidthGBs", []Reading{{Rank: 1, Value: "40"}, {Rank: 0, Value: "97.2"}})
	if !ok || got != "97.2" {
		t.Errorf("busBandwidthGBs = %q (%v), want rank 0's 97.2", got, ok)
	}

	// elapsedS is Max across ranks.
	got, ok = Elect("elapsedS", []Reading{{Rank: 0, Value: "300"}, {Rank: 1, Value: "305"}})
	if !ok || got != "305" {
		t.Errorf("elapsedS = %q (%v), want 305 — eight ranks running 300s took 300s, not 2400s", got, ok)
	}
}

// An unclassified key keeps its value only while the ranks AGREE, which is the
// case where there is nothing to decide. The moment they disagree it is dropped
// and declared unmeasurable, so a threshold fails closed rather than certifying
// every node on one node's evidence.
func TestElect_AnUnclassifiedKeyNeedsUnanimity(t *testing.T) {
	got, ok := Elect("somethingNobodyRegistered", []Reading{{Rank: 0, Value: "7"}, {Rank: 1, Value: "7"}})
	if !ok || got != "7" {
		t.Errorf("unanimous = %q (%v), want 7", got, ok)
	}
	if _, ok := Elect("somethingNobodyRegistered", []Reading{{Rank: 0, Value: "7"}, {Rank: 1, Value: "9"}}); ok {
		t.Error("a disagreement on an unclassified key was elected; it must fail closed instead")
	}
}

// THE GB10 CASE from CLAUDE.md: ranks 0 and 1 emit eccErrors=0 and rank 2 is a
// part with no ECC to report. Electing 0 past that declaration certifies the
// node that said it could not tell.
func TestElect_ARankDeclaringUnmeasurableBlocksTheElection(t *testing.T) {
	vals := []Reading{{Rank: 0, Value: "0"}, {Rank: 1, Value: "0"}, {Rank: 2, Value: runner.Unmeasurable}}
	if _, ok := Elect("eccErrors", vals); ok {
		t.Error("a value was elected past a rank's declaration of unmeasurability — the gate would then " +
			"pass and certify all three nodes, including the one whose gate never ran")
	}
}

func TestCombine_AnUnmeasurableDeclarationSurvivesIntoTheResult(t *testing.T) {
	r0 := res(runner.VerdictPass, 0, map[string]string{"eccErrors": "0"})
	r1 := res(runner.VerdictPass, 0, map[string]string{"eccErrors": "0"})
	r2 := res(runner.VerdictPass, 0, nil)
	r2.Unmeasurable = map[string]bool{"eccErrors": true}

	c := Combine(ranksOf(r0, r1, r2))

	if _, present := c.Metrics["eccErrors"]; present {
		t.Errorf("eccErrors = %q is still a metric; a threshold on it would pass and certify the node "+
			"that declared it could not measure", c.Metrics["eccErrors"])
	}
	if !c.Unmeasurable["eccErrors"] {
		t.Error("eccErrors is neither a metric nor unmeasurable — it would simply vanish, and a gate on " +
			"it fails closed only because the metric is ABSENT rather than because anyone said so")
	}
}

// Readings pair every value with the rank that gave it. Attributing a discarded
// value to whichever rank happened to overwrite it names the wrong node, which
// on a per-node metric is the entire cost of the bug the note exists to report.
func TestCombine_ReadingsNameTheRankThatGaveEachValue(t *testing.T) {
	c := Combine(ranksOf(
		res(runner.VerdictPass, 0, map[string]string{"gpuTempC": "70"}),
		res(runner.VerdictPass, 0, map[string]string{"gpuTempC": "91"}),
	))

	got := map[int]string{}
	for _, rv := range c.Readings["gpuTempC"] {
		got[rv.Rank] = rv.Value
	}
	if got[0] != "70" || got[1] != "91" {
		t.Errorf("readings = %v, want rank 0=70 and rank 1=91", got)
	}
}

// pkg/runner adds an InvalidNames entry per offending LINE, so 64 ranks x 40
// samples is 2560 copies of one name. The set is the finding.
func TestCombine_InvalidNamesAreDeduped(t *testing.T) {
	a := res(runner.VerdictPass, 0, nil)
	a.InvalidNames = []string{"Bad_Name", "Bad_Name"}
	b := res(runner.VerdictPass, 0, nil)
	b.InvalidNames = []string{"Bad_Name", "other"}

	c := Combine(ranksOf(a, b))
	if len(c.InvalidNames) != 2 {
		t.Errorf("InvalidNames = %v, want the SET", c.InvalidNames)
	}
}
