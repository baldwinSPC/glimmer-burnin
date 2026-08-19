package main

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"os/signal"
	"path/filepath"
	"sort"
	"strings"
	"syscall"
	"time"

	api "github.com/baldwinSPC/glimmer-burnin/api/v1alpha1"
	"github.com/baldwinSPC/glimmer-burnin/pkg/group"
	"github.com/baldwinSPC/glimmer-burnin/pkg/localrun"
	"github.com/baldwinSPC/glimmer-burnin/pkg/runner"
)

// `burnin merge` is the half of Group scope that no rank can do for itself.
//
// A GROUP VERDICT IS ABOUT THE COLLECTIVE. Rank 0 holds the bandwidth figure,
// but whether every rank took part is a fact invisible from inside any one of
// them — and it is the fact that decides whether there is a verdict at all. So
// each rank writes a record and this reads all N.
//
// The fold is pkg/group's, the operator's own, so a bare-metal collective and
// an in-cluster one elect the same verdict and the same metric values from the
// same rank reports.

const mergeUsage = `burnin merge — fold every rank's record into the collective's verdict

USAGE
  burnin merge --results-dir DIR [flags]

  DIR holds every rank's record, as written by ` + "`burnin run --rank i`" + `. Copy
  each machine's <results-dir>/ranks/rank-NN.json into one directory, or point
  --results-dir at a shared filesystem every rank wrote to.

FLAGS
  --results-dir   directory holding ranks/rank-NN.json (repeatable)
  --nranks        how many ranks the collective had. Defaults to what the
                  records themselves claim; pass it to be certain
  --out           where to write the merged envelope (default: --results-dir)
  --sink-url      POST the merged envelope here
  --sink-token-file  file holding the bearer token

A PARTIAL COLLECTIVE HAS NO VERDICT. If fewer than --nranks records are found,
this refuses and names the missing ranks. A collective is only measured if every
rank took part, and rendering a verdict from the ranks that happened to report
would certify machines nobody looked at.

EXIT CODES
  0  the collective passed
  1  the collective FAILED — measured, and fell short
  2  nothing was judged
  3  machinery: a missing rank, an unreadable record, a partial collective
`

// rankRecord is what one rank writes and this reads.
//
// Its own struct, with explicit JSON tags, rather than a map: this file is the
// boundary between N machines and one verdict, and a typo in a key name would
// silently drop a rank — which the refusal below would then report as a machine
// that never ran.
type rankRecord struct {
	Rank       int                   `json:"rank"`
	NRanks     int                   `json:"nranks"`
	Node       string                `json:"node"`
	Phase      string                `json:"phase"`
	StartedAt  time.Time             `json:"startedAt"`
	FinishedAt time.Time             `json:"finishedAt"`
	Results    []localrun.TestResult `json:"results"`
	// Required names the tests that decide the run's exit code.
	//
	// Recorded rather than assumed, because a merge has no plan to read it
	// from — and assuming every merged test is required would make a failing
	// OPTIONAL collective exit 1, which is a hardware verdict the profile
	// deliberately did not ask for. Assuming none is required is the mistake
	// this field was added to fix: a passing collective exited 2, "nothing was
	// judged", which a CI job reads as hardware nobody measured.
	Required map[string]bool `json:"required,omitempty"`
	Note     string          `json:"note,omitempty"`
}

type mergeFlags struct {
	dirs   multiFlag
	nranks int
	out    string
	sink   sinkFlags
}

func runMerge(args []string) error {
	var f mergeFlags
	fs := flag.NewFlagSet("merge", flag.ContinueOnError)
	fs.SetOutput(os.Stderr)
	fs.Usage = func() { fmt.Fprint(os.Stderr, mergeUsage) }

	fs.Var(&f.dirs, "results-dir", "directory holding ranks/rank-NN.json (repeatable)")
	fs.IntVar(&f.nranks, "nranks", 0, "how many ranks the collective had")
	fs.StringVar(&f.out, "out", "", "where to write the merged envelope")
	fs.StringVar(&f.sink.url, "sink-url", "", "POST the merged envelope here")
	fs.StringVar(&f.sink.tokenFile, "sink-token-file", "", "file holding the bearer token")
	fs.BoolVar(&f.sink.insecure, "sink-insecure-skip-tls-verify", false, "development only")
	fs.DurationVar(&f.sink.timeout, "sink-timeout", 30*time.Second, "per-attempt timeout")

	if err := fs.Parse(args); err != nil {
		return err
	}
	if len(f.dirs) == 0 {
		fs.Usage()
		return exitWith(exitError, fmt.Errorf("no --results-dir: where are the rank records?"))
	}

	records, err := loadRankRecords(f.dirs)
	if err != nil {
		return exitWith(exitError, err)
	}
	if len(records) == 0 {
		return exitWith(exitError, fmt.Errorf(
			"no rank records under %s — a rank writes ranks/rank-NN.json only when `burnin run` was given --rank",
			strings.Join(f.dirs, ", ")))
	}

	nranks, err := expectedRanks(f.nranks, records)
	if err != nil {
		return exitWith(exitError, err)
	}

	report, err := mergeRanks(records, nranks)
	if err != nil {
		return exitWith(exitError, err)
	}

	id, err := NewRunIdentity(report.Node)
	if err != nil {
		return exitWith(exitError, err)
	}
	env, err := id.Final(report)
	if err != nil {
		return exitWith(exitError, err)
	}

	out := f.out
	if out == "" {
		out = f.dirs[0]
	}
	if err := writeJSON(filepath.Join(out, "envelopes", "001-RunPhaseChanged.json"), env); err != nil {
		warn("the merged envelope was not written: %v", err)
	}

	sender, err := f.sink.deliverer()
	if err != nil {
		return exitWith(exitError, err)
	}
	if sender != nil {
		ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
		defer stop()
		deliver(ctx, os.Stderr, sender, env)
	}

	printSummary(report)
	return exitWith(codeFor(report, requiredPlan(records)), nil)
}

// requiredPlan rebuilds just enough plan for codeFor to read which tests decide
// the exit code.
//
// A test is required if ANY rank recorded it as required. Ranks running one
// profile agree; a disagreement means they ran different profiles, and treating
// the test as required is the conservative half — it lets a failure reach the
// exit code rather than being silently discarded as optional.
func requiredPlan(records []rankRecord) localrun.Plan {
	required := map[string]bool{}
	for _, rec := range records {
		for name, req := range rec.Required {
			if req {
				required[name] = true
			}
		}
	}
	var p localrun.Plan
	for _, r := range records[0].Results {
		p.Tests = append(p.Tests, localrun.PlannedTest{Name: r.Name, Required: required[r.Name]})
	}
	return p
}

// expectedRanks decides how many ranks the collective HAD, which is the number
// the turnout is judged against.
//
// --nranks wins when given, because the records are exactly what a missing rank
// cannot contribute to: if rank 7's machine never ran, nothing in the directory
// knows rank 7 was expected. The records' own claim is the fallback and is
// checked for agreement — two ranks that disagree about the size of their own
// collective did not run the same collective, and merging them would produce a
// verdict about a group that never existed.
func expectedRanks(want int, records []rankRecord) (int, error) {
	if want > 0 {
		return want, nil
	}
	claimed := map[int][]int{}
	for _, r := range records {
		claimed[r.NRanks] = append(claimed[r.NRanks], r.Rank)
	}
	if len(claimed) != 1 {
		var parts []string
		for n, ranks := range claimed {
			sort.Ints(ranks)
			parts = append(parts, fmt.Sprintf("%v say %d", ranks, n))
		}
		sort.Strings(parts)
		return 0, fmt.Errorf(
			"the records disagree about how many ranks the collective had (%s) — they did not run the "+
				"same collective, and one verdict over them would describe a group that never existed",
			strings.Join(parts, "; "))
	}
	for n := range claimed {
		if n < 2 {
			return 0, fmt.Errorf("the records claim %d rank(s); a collective needs at least 2", n)
		}
		return n, nil
	}
	return 0, fmt.Errorf("no rank count could be established; pass --nranks")
}

// loadRankRecords reads every rank-NN.json under the given directories.
//
// A DUPLICATE RANK IS REFUSED rather than resolved. Two files claiming rank 3
// are two different machines that both believed they were rank 3, or one
// machine's record copied twice — and either way the collective did not run the
// way the merge is about to describe it.
func loadRankRecords(dirs []string) ([]rankRecord, error) {
	byRank := map[int]string{}
	var out []rankRecord

	for _, dir := range dirs {
		for _, pattern := range []string{
			filepath.Join(dir, "ranks", "rank-*.json"),
			filepath.Join(dir, "rank-*.json"),
		} {
			paths, err := filepath.Glob(pattern)
			if err != nil {
				return nil, err
			}
			for _, p := range paths {
				raw, err := os.ReadFile(p)
				if err != nil {
					return nil, fmt.Errorf("reading %s: %w", p, err)
				}
				var rec rankRecord
				if err := json.Unmarshal(raw, &rec); err != nil {
					return nil, fmt.Errorf("parsing %s: %w", p, err)
				}
				if prev, dup := byRank[rec.Rank]; dup {
					return nil, fmt.Errorf(
						"rank %d appears twice: %s and %s. Two machines that both believed they were "+
							"rank %d did not run the collective this merge would describe",
						rec.Rank, prev, p, rec.Rank)
				}
				byRank[rec.Rank] = p
				out = append(out, rec)
			}
		}
	}
	sort.Slice(out, func(i, j int) bool { return out[i].Rank < out[j].Rank })
	return out, nil
}

// mergeRanks folds N records into the one report a collective is judged on.
func mergeRanks(records []rankRecord, nranks int) (localrun.Report, error) {
	// THE REFUSAL. Checked before anything is folded, because a fold over a
	// partial turnout produces a plausible-looking verdict and the whole point
	// is that there is not one.
	present := map[int]bool{}
	for _, r := range records {
		if r.Rank < 0 || r.Rank >= nranks {
			return localrun.Report{}, fmt.Errorf(
				"a record claims rank %d, which is not a rank of %d", r.Rank, nranks)
		}
		present[r.Rank] = true
	}
	var missing []int
	for i := 0; i < nranks; i++ {
		if !present[i] {
			missing = append(missing, i)
		}
	}
	if len(missing) > 0 {
		return localrun.Report{}, fmt.Errorf(
			"only %d of %d ranks reported — %s never did. A collective is only MEASURED if every rank "+
				"took part, so there is no verdict here: a fold over the ranks that happened to report "+
				"would certify machines nobody looked at. Collect the missing record(s), or re-run",
			len(records), nranks, describeMissing(missing))
	}

	nodes := make([]string, 0, len(records))
	for _, r := range records {
		nodes = append(nodes, r.Node)
	}

	// Tests are folded BY NAME, in rank 0's order. A rank that ran a different
	// profile contributes nothing to a test the others ran, which the turnout
	// check below catches per test rather than assuming.
	rep := localrun.Report{
		Node:       strings.Join(nodes, ","),
		StartedAt:  records[0].StartedAt,
		FinishedAt: records[0].FinishedAt,
	}
	for _, seed := range records[0].Results {
		merged, err := mergeOneTest(seed, records, nodes)
		if err != nil {
			return localrun.Report{}, err
		}
		rep.Results = append(rep.Results, merged)
		switch merged.Phase {
		case api.RunPassed:
			rep.Summary.Passed++
		case api.RunFailed:
			rep.Summary.Failed++
		case api.RunSkipped:
			rep.Summary.Skipped++
		default:
			rep.Summary.Errored++
		}
	}
	rep.Phase = mergedPhase(rep.Results)
	return rep, nil
}

// mergeOneTest folds one test across every rank.
func mergeOneTest(seed localrun.TestResult, records []rankRecord, nodes []string) (localrun.TestResult, error) {
	ranks := make([]group.Rank, 0, len(records))
	for _, rec := range records {
		r := group.Rank{Rank: rec.Rank, Node: rec.Node}
		// A rank that has no result for this test contributes a NIL result, not
		// an absent entry — which is the difference pkg/group.Verdict turns into
		// a refusal rather than a pass.
		if res, ok := resultNamed(rec.Results, seed.Name); ok {
			r.Result = toRunnerResult(res)
		}
		ranks = append(ranks, r)
	}

	c := group.Combine(ranks)

	out := seed
	out.Nodes = append([]string(nil), nodes...)
	out.Metrics = c.Metrics
	out.Unmeasurable = sortedKeys(c.Unmeasurable)
	out.Phase = phaseOf(c.Verdict)
	out.Message = mergeMessage(ranks, c)
	// Violations came from ONE rank's own evaluation of its own metrics and do
	// not describe the collective. Cleared rather than merged: the gate belongs
	// to the merged metrics, and re-running it here would need the thresholds,
	// which a rank record does not carry.
	out.Violations = nil
	return out, nil
}

func resultNamed(in []localrun.TestResult, name string) (localrun.TestResult, bool) {
	for _, r := range in {
		if r.Name == name {
			return r, true
		}
	}
	return localrun.TestResult{}, false
}

// toRunnerResult reconstructs what the fold needs from a stored result.
func toRunnerResult(r localrun.TestResult) *runner.Result {
	out := &runner.Result{
		Metrics:      r.Metrics,
		Unmeasurable: map[string]bool{},
		Verdict:      verdictOf(r.Phase),
		Message:      r.Message,
		ExitCode:     -1,
	}
	for _, k := range r.Unmeasurable {
		out.Unmeasurable[k] = true
	}
	if n := len(r.Attempts); n > 0 {
		out.ExitCode = r.Attempts[n-1].ExitCode
	}
	return out
}

func verdictOf(p api.RunPhase) runner.Verdict {
	switch p {
	case api.RunPassed:
		return runner.VerdictPass
	case api.RunFailed:
		return runner.VerdictFail
	case api.RunSkipped:
		return runner.VerdictSkip
	default:
		return runner.VerdictError
	}
}

func phaseOf(v runner.Verdict) api.RunPhase {
	switch v {
	case runner.VerdictPass:
		return api.RunPassed
	case runner.VerdictFail:
		return api.RunFailed
	case runner.VerdictSkip:
		return api.RunSkipped
	default:
		return api.RunError
	}
}

// mergeMessage states the verdict's SUBJECT before its evidence.
//
// The lead clause is not decoration: a reader who sees a failure next to eleven
// node names will pick one of them, and on a collective that is nearly always
// the wrong one — every rank except the faulty one reports a timeout waiting
// for it.
func mergeMessage(ranks []group.Rank, c group.Combined) string {
	parts := []string{fmt.Sprintf(
		"group of %d (this verdict is about the COLLECTIVE, not about any one rank)", len(ranks))}

	// Ranks that DISSENT from the verdict are named in full; those that agree
	// are counted. Those are the ones worth walking to a rack for.
	var dissent []string
	agreed := 0
	for _, m := range ranks {
		if m.Result == nil {
			dissent = append(dissent, fmt.Sprintf("rank %d (%s) did not report", m.Rank, m.Node))
			continue
		}
		if m.Result.Verdict == c.Verdict {
			agreed++
			continue
		}
		dissent = append(dissent, fmt.Sprintf("rank %d (%s) %s (exit %d)",
			m.Rank, m.Node, strings.ToLower(string(m.Result.Verdict)), m.Result.ExitCode))
	}
	if agreed > 0 {
		parts = append(parts, fmt.Sprintf("%d rank(s) %s", agreed, strings.ToLower(string(c.Verdict))))
	}
	parts = append(parts, dissent...)

	if note := disagreementSummary(c); note != "" {
		parts = append(parts, note)
	}
	return strings.Join(parts, "; ")
}

// disagreementSummary names the keys the ranks answered differently.
//
// Recorded rather than resolved. A stored result saying "ranks disagreed about
// gpuTempC" is one an engineer can act on, where a silent 70 next to eight node
// names is one they cannot — and it changes no verdict, which is the point.
func disagreementSummary(c group.Combined) string {
	var keys []string
	for k, vals := range c.Readings {
		for _, rv := range vals {
			if rv.Value != c.Metrics[k] {
				keys = append(keys, k)
				break
			}
		}
	}
	if len(keys) == 0 {
		return ""
	}
	sort.Strings(keys)
	return "ranks disagreed about " + strings.Join(keys, ", ")
}

func sortedKeys(m map[string]bool) []string {
	if len(m) == 0 {
		return nil
	}
	out := make([]string, 0, len(m))
	for k := range m {
		out = append(out, k)
	}
	sort.Strings(out)
	return out
}

// mergedPhase is the run's own phase, by the same precedence the engine uses.
func mergedPhase(results []localrun.TestResult) api.RunPhase {
	phase := api.RunPassed
	judged := false
	for _, r := range results {
		switch r.Phase {
		case api.RunFailed:
			return api.RunFailed
		case api.RunError:
			phase = api.RunError
			judged = true
		case api.RunPassed:
			judged = true
		}
	}
	if !judged {
		return api.RunSkipped
	}
	return phase
}
