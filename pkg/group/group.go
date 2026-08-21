// SPDX-License-Identifier: Apache-2.0
// Copyright the Glimmer authors.

// Package group folds N ranks' reports into the ONE result a collective is
// judged on.
//
// A GROUP VERDICT IS ABOUT THE COLLECTIVE, not about any one rank. Group scope
// runs one rank per target node, all rendezvous'd together, and produces
// exactly one TestResult naming every node. Splitting it per node would send an
// engineer to replace the wrong part: every healthy rank blocks on the faulty
// one and reports a timeout, so a per-rank reading indicts eleven innocent
// nodes and the twelfth equally.
//
// # Why this is a package
//
// The operator has always done this. The bare-metal dispatcher could not run
// Group scope at all, and the obvious way to add it is a second implementation
// of the fold — which would be the same category of mistake the two-dispatcher
// design spends its tests preventing, in the one place it would be hardest to
// notice. Two dispatchers electing DIFFERENT values from the same eight rank
// reports, under one metric name, is a fleet dashboard that cannot be read.
//
// So the DECISIONS live here: which verdict the group takes, and what each
// metric's value is once every rank has spoken. PRESENTATION does not — the
// message a reader sees is each dispatcher's own, because one has a pod name to
// quote and the other has a file path, and neither is a decision.
//
// # Kubernetes-free
//
// Deliberately, and it is measured (hack/invariants). The whole fold is
// expressed over pkg/runner's Result and pkg/contract's Combination, neither of
// which needs a cluster. That is what lets the bare-metal path use it at all.
package group

import (
	"sort"
	"strconv"

	"github.com/baldwinSPC/glimmer-burnin/pkg/contract"
	"github.com/baldwinSPC/glimmer-burnin/pkg/runner"
)

// RootRank is the rank that serves the rendezvous and, where a verdict needs a
// representative exit code, supplies it.
const RootRank = 0

// Rank is one rank's report.
//
// A nil Result means that rank NEVER PRODUCED ONE, which is a different
// statement from "it reported nothing useful" — and the difference decides the
// verdict. Keep them distinct.
type Rank struct {
	Rank   int
	Node   string
	Result *runner.Result
}

// Reading is one rank's answer for one metric key.
//
// Paired with the rank that gave it, always. Attributing a discarded value to
// whichever rank happened to overwrite it — the obvious shortcut — produces a
// note that names the wrong node, which on a per-node metric is the entire cost
// of the bug the note exists to report.
type Reading struct {
	Rank  int
	Value string
}

// Combined is what N ranks jointly established.
type Combined struct {
	Metrics      map[string]string
	Unmeasurable map[string]bool
	InvalidNames []string
	Verdict      runner.Verdict
	ExitCode     int
	// Readings is every rank's answer for every key, so a caller can report a
	// disagreement in its own words. It is DATA and not prose: which ranks said
	// what is a fact about the run, and how to phrase it is not.
	Readings map[string][]Reading
}

// Combine folds every rank's report into one.
func Combine(ranks []Rank) Combined {
	out := Combined{
		Metrics:      map[string]string{},
		Unmeasurable: map[string]bool{},
		Readings:     map[string][]Reading{},
	}

	// Highest rank first, so a LOWER rank overwrites it and rank 0 wins the
	// provisional election. The real election happens below, per metric; this
	// pass only collects.
	var invalid []string
	for i := len(ranks) - 1; i >= 0; i-- {
		m := ranks[i]
		if m.Result == nil {
			continue
		}
		for k, v := range m.Result.Metrics {
			out.Readings[k] = append(out.Readings[k], Reading{Rank: m.Rank, Value: v})
			out.Metrics[k] = v
		}
		invalid = append(invalid, m.Result.InvalidNames...)
	}

	// A declaration of unmeasurability never overwrites a real measurement,
	// which is the Pair rule and is right there: two ends of ONE link measuring
	// one quantity, so if either produced a number the quantity was measurable.
	//
	// AT GROUP SCOPE THAT REASONING DOES NOT CARRY, and swallowing the
	// declaration is a certification. Ranks 0 and 1 emit `eccErrors=0` and rank
	// 2 is a GB10 whose runner correctly emits `eccErrors=n/a` — the worked
	// example in CLAUDE.md — and the group stored eccErrors=0 with an empty
	// Unmeasurable set. An `eccErrors Equal 0` gate then passed and all three
	// nodes were certified, including the one whose gate never ran. The SAME
	// node under Node scope fails closed on that threshold.
	//
	// So the declaration is recorded as a reading, and the election below
	// refuses any key a rank declared unmeasurable.
	for _, m := range ranks {
		if m.Result == nil {
			continue
		}
		for k := range m.Result.Unmeasurable {
			if _, measured := out.Metrics[k]; !measured {
				out.Unmeasurable[k] = true
				continue
			}
			out.Readings[k] = append(out.Readings[k], Reading{Rank: m.Rank, Value: runner.Unmeasurable})
		}
	}

	// Now that every rank has been read, elect each key's value by what the
	// metric SAYS it is rather than by which rank happened to write last.
	for k, vals := range out.Readings {
		merged, ok := Elect(k, vals)
		if !ok {
			delete(out.Metrics, k)
			out.Unmeasurable[k] = true
			continue
		}
		out.Metrics[k] = merged
	}

	out.InvalidNames = dedupe(invalid)
	out.Verdict, out.ExitCode = Verdict(ranks)
	return out
}

// Verdict applies the precedence rule across every rank and returns the exit
// code of the rank that decided it.
func Verdict(ranks []Rank) (runner.Verdict, int) {
	reported := 0
	for _, m := range ranks {
		if m.Result != nil {
			reported++
		}
	}
	if reported == 0 || len(ranks) == 0 {
		return runner.VerdictError, -1
	}

	// Error and Fail are taken from the LOWEST rank holding them, so the exit
	// code and the verdict describe the same execution. Both are honoured
	// however many ranks reported: a rank that positively erred or positively
	// failed has established something, and a silent peer does not erase it.
	for _, want := range []runner.Verdict{runner.VerdictError, runner.VerdictFail} {
		for _, m := range ranks {
			if m.Result != nil && m.Result.Verdict == want {
				return want, m.Result.ExitCode
			}
		}
	}

	// EVERY REMAINING VERDICT REQUIRES A FULL TURNOUT, and Skip needs it for
	// exactly the same reason Pass does.
	//
	// A collective is only measured if every rank took part, so a rank that
	// never reported is a rank whose participation nobody can vouch for. That is
	// obvious for Pass. It is just as true for Skip, and Skip is the more
	// dangerous of the two to get wrong, because a Skip does not fail the RUN —
	// it records "acceptance does not apply to this hardware" and the run
	// settles Passed around it. Honouring rank 0's declaration for a group whose
	// other members never ran would certify every one of them on evidence from
	// one node. A runner may only declare what it positively established, and
	// rank 0 established something about rank 0.
	//
	// On the bare-metal path this is the guard that matters most, because there
	// is no controller creating the other ranks' pods: a person starts them by
	// hand, and the failure mode this refuses is a merge run before the last
	// machine finished — or before somebody noticed it never started.
	//
	// Pair scope does NOT have this guard and must not grow one by symmetry: at
	// n=2 the server terminating before the client is already an Error when it
	// exits 0 (no traffic crossed the link), and its Skip is a statement about
	// the only other participant in a link it is one half of. The asymmetry is
	// the point — a group has members whose hardware nobody consulted.
	if reported != len(ranks) {
		return runner.VerdictError, exitCodeOf(ranks, RootRank)
	}
	for _, m := range ranks {
		if m.Result.Verdict == runner.VerdictSkip {
			return runner.VerdictSkip, m.Result.ExitCode
		}
	}
	return runner.VerdictPass, exitCodeOf(ranks, RootRank)
}

// Missing names the ranks that never reported, in rank order.
//
// Exported because a caller REFUSING a partial collective owes the person a
// list: "rank 3 and rank 7 never reported" is something to act on, where "the
// collective was not measured" is not.
func Missing(ranks []Rank) []int {
	var out []int
	for _, m := range ranks {
		if m.Result == nil {
			out = append(out, m.Rank)
		}
	}
	sort.Ints(out)
	return out
}

// Elect decides one metric key's value across the ranks that answered.
//
// Returns false when the key cannot be elected at all, which the caller must
// treat as UNMEASURABLE rather than absent — a threshold on it then fails
// closed, instead of certifying every node on one node's evidence.
func Elect(key string, vals []Reading) (string, bool) {
	if len(vals) == 0 {
		return "", false
	}
	// A rank that declared the metric unmeasurable has not measured it, and a
	// number elected past that declaration would certify the node that said it
	// could not tell.
	for _, rv := range vals {
		if rv.Value == runner.Unmeasurable {
			return "", false
		}
	}

	lowest := vals[0]
	unanimous := true
	for _, rv := range vals[1:] {
		if rv.Rank < lowest.Rank {
			lowest = rv
		}
		if rv.Value != vals[0].Value {
			unanimous = false
		}
	}

	// pkg/contract.Combination is what makes this a decision about the METRIC
	// rather than about which rank happened to write last. Aggregation cannot
	// stand in for it: elapsedS is Sum across windows and Max across ranks —
	// eight ranks running 300s took 300s, not 2400s — and eccErrors is Last
	// across windows, which across ranks means "whatever rank 0 said", the
	// defect itself.
	switch c := contract.CombinationFor(key); c {
	case contract.CombineCollective:
		return lowest.Value, true
	case contract.CombineSum, contract.CombineMax, contract.CombineMin:
		return combineNumeric(c, vals)
	default:
		// Unclassified. Unanimity is not a decision about the metric, it is the
		// absence of anything to decide — so it is honoured, and the moment the
		// ranks disagree the key is dropped and declared unmeasurable.
		if unanimous {
			return lowest.Value, true
		}
		return "", false
	}
}

// combineNumeric applies a numeric Combination. A value that is not a number
// cannot be summed or compared, and inventing an ordering for it would be the
// same silent guess the classification exists to prevent.
func combineNumeric(c contract.Combination, vals []Reading) (string, bool) {
	acc, err := strconv.ParseFloat(vals[0].Value, 64)
	if err != nil {
		return "", false
	}
	for _, rv := range vals[1:] {
		f, err := strconv.ParseFloat(rv.Value, 64)
		if err != nil {
			return "", false
		}
		switch c {
		case contract.CombineSum:
			acc += f
		case contract.CombineMax:
			if f > acc {
				acc = f
			}
		case contract.CombineMin:
			if f < acc {
				acc = f
			}
		}
	}
	return strconv.FormatFloat(acc, 'f', -1, 64), true
}

// exitCodeOf is the named rank's exit code, or -1 where it did not report.
//
// Looked up by RANK NUMBER rather than by slice position: a caller assembling
// ranks from files on N machines has no reason to hand them over sorted, and
// indexing would silently attribute rank 0's exit code to whoever came first.
func exitCodeOf(ranks []Rank, want int) int {
	for _, m := range ranks {
		if m.Rank == want && m.Result != nil {
			return m.Result.ExitCode
		}
	}
	return -1
}

// dedupe keeps the set and drops the repetition.
//
// pkg/runner adds an InvalidNames entry per OFFENDING LINE and progressive
// reporting is expected there, so a runner emitting one bad key per sample
// gives ranks x samples entries — 64 ranks x 40 samples is 2560 copies of the
// same name in one message. The SET is the finding; the count of times it was
// printed is not.
func dedupe(in []string) []string {
	if len(in) == 0 {
		return nil
	}
	seen := map[string]bool{}
	out := make([]string, 0, len(in))
	for _, s := range in {
		if !seen[s] {
			seen[s] = true
			out = append(out, s)
		}
	}
	sort.Strings(out)
	return out
}
