package main

import (
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io/fs"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"text/tabwriter"
	"time"

	"github.com/baldwinSPC/glimmer-burnin/pkg/compare"
	"github.com/baldwinSPC/glimmer-burnin/pkg/contract"
)

const baselineUsage = `burnin baseline — turn replicate runs into a baseline

USAGE
  burnin baseline promote --results-dir DIR [--results-dir DIR]... --as NAME --out FILE

  Reads the finished runs in each DIR (one node, one configuration) and writes
  each numeric metric's replicate distribution: the values, mean, sample SD
  and CV. A metric is ENFORCEABLE only from at least 5 clean runs, because
  fewer says nothing reliable about run-to-run variation. A run that errored
  on a test contributes nothing to that test's metrics.

  Runs whose fingerprints differ (another kernel, OS or accelerator) are
  refused: their spread would be partly configuration drift.
`

const compareUsage = `burnin compare — has anything changed?

USAGE
  burnin compare --baseline FILE --results-dir DIR [flags]
      one run against a promoted baseline
  burnin compare --results-dir DIR... --peer-results-dir DIR... [flags]
      two nodes' runs of one profile against each other

  A comparison is NOT a verdict. REGRESSION says a number moved against its
  reference by more than the tolerance and more than the reference's own
  spread allows; it does not say the hardware failed, because a baseline can
  be wrong. Thresholds decide Pass and Fail.

  Outcomes, first match wins: MISSING, NOT_COMPARABLE (with the cause, such as
  differing kernel or driver), UNUSABLE_INPUT (the run errored),
  INSUFFICIENT_PRECISION (too few replicates), then MATCH,
  NO_SIGNIFICANT_DIFFERENCE, REGRESSION or IMPROVEMENT.

FLAGS
  --tolerance       relative change that always counts as a MATCH (default 0.05)
  --tolerance-for   per metric, e.g. bandwidthGbps=0.02 (repeatable)
  --enforce         exit 1 on a REGRESSION in an enforceable baseline metric
  --json            print the results as JSON

EXIT
  0  compared (including regressions, unless --enforce)
  1  --enforce and an enforceable metric regressed
  3  could not compare
`

func runBaseline(args []string) error {
	if len(args) == 0 || args[0] != "promote" {
		fmt.Fprint(os.Stderr, baselineUsage)
		return exitWith(exitError, fmt.Errorf("baseline: the only subcommand is promote"))
	}
	fs := flag.NewFlagSet("baseline promote", flag.ContinueOnError)
	fs.SetOutput(os.Stderr)
	fs.Usage = func() { fmt.Fprint(os.Stderr, baselineUsage) }
	var dirs multiFlag
	var name, out string
	fs.Var(&dirs, "results-dir", "a run's results directory (repeatable)")
	fs.StringVar(&name, "as", "", "the baseline's name")
	fs.StringVar(&out, "out", "", "where to write the baseline file")
	if err := fs.Parse(args[1:]); err != nil {
		if errors.Is(err, flag.ErrHelp) {
			return nil
		}
		return exitWith(exitError, err)
	}
	if len(dirs) == 0 || name == "" || out == "" {
		fs.Usage()
		return exitWith(exitError, fmt.Errorf("--results-dir, --as and --out are required"))
	}
	side, err := loadSide(dirs)
	if err != nil {
		return exitWith(exitError, err)
	}
	b, err := compare.Promote(name, side, cliProducer(), time.Now())
	if err != nil {
		return exitWith(exitError, err)
	}
	if _, err := os.Stat(out); err == nil {
		return exitWith(exitError, fmt.Errorf("%s exists; a baseline is not overwritten in place", out))
	}
	if err := writeJSON(out, b); err != nil {
		return exitWith(exitError, err)
	}
	enforceable := 0
	for _, m := range b.Metrics {
		if m.Enforceable {
			enforceable++
		}
	}
	fmt.Printf("baseline %q: %d run(s) of %s, %d metric(s), %d enforceable (need %d clean runs each)\n",
		name, b.Replicates, b.Node, len(b.Metrics), enforceable, compare.MinReplicates)
	return nil
}

func runCompare(args []string) error {
	fs := flag.NewFlagSet("compare", flag.ContinueOnError)
	fs.SetOutput(os.Stderr)
	fs.Usage = func() { fmt.Fprint(os.Stderr, compareUsage) }
	var dirs, peers, tolFor multiFlag
	var baselineFile string
	var tol float64
	var enforce, asJSON bool
	fs.Var(&dirs, "results-dir", "the run(s) being compared (repeatable)")
	fs.Var(&peers, "peer-results-dir", "the other node's run(s) (repeatable)")
	fs.StringVar(&baselineFile, "baseline", "", "a promoted baseline file")
	fs.Float64Var(&tol, "tolerance", 0.05, "relative change that always counts as a MATCH")
	fs.Var(&tolFor, "tolerance-for", "metric=fraction (repeatable)")
	fs.BoolVar(&enforce, "enforce", false, "exit 1 on an enforceable regression")
	fs.BoolVar(&asJSON, "json", false, "print JSON")
	if err := fs.Parse(args); err != nil {
		if errors.Is(err, flag.ErrHelp) {
			return nil
		}
		return exitWith(exitError, err)
	}
	if len(dirs) == 0 || (baselineFile == "") == (len(peers) == 0) {
		fs.Usage()
		return exitWith(exitError, fmt.Errorf("give --results-dir and exactly one of --baseline or --peer-results-dir"))
	}
	o := compare.Options{DefaultTolerance: tol, Alpha: 0.05, Enforce: enforce, Tolerance: map[string]float64{}}
	for _, kv := range tolFor {
		k, v, ok := strings.Cut(kv, "=")
		f, err := strconv.ParseFloat(v, 64)
		if !ok || err != nil || f < 0 {
			return exitWith(exitError, fmt.Errorf("--tolerance-for %q: want metric=fraction", kv))
		}
		o.Tolerance[k] = f
	}

	cur, err := loadSide(dirs)
	if err != nil {
		return exitWith(exitError, err)
	}
	var results []compare.Result
	if baselineFile != "" {
		data, err := os.ReadFile(baselineFile)
		if err != nil {
			return exitWith(exitError, err)
		}
		var b compare.Baseline
		if err := json.Unmarshal(data, &b); err != nil || b.Schema != compare.BaselineSchema {
			return exitWith(exitError, fmt.Errorf("%s is not a %s file", baselineFile, compare.BaselineSchema))
		}
		if results, err = compare.AgainstBaseline(cur, &b, o); err != nil {
			return exitWith(exitError, err)
		}
	} else {
		peer, err := loadSide(peers)
		if err != nil {
			return exitWith(exitError, err)
		}
		results = compare.Symmetry(cur, peer, o)
	}

	if asJSON {
		b, _ := json.MarshalIndent(results, "", "  ")
		fmt.Println(string(b))
	} else {
		printResults(results)
	}
	for _, r := range results {
		if r.Enforced {
			return exitWith(exitFail, fmt.Errorf("an enforceable metric regressed: %s", r.Key))
		}
	}
	return nil
}

func printResults(rs []compare.Result) {
	w := tabwriter.NewWriter(os.Stdout, 0, 2, 2, ' ', 0)
	fmt.Fprintln(w, "METRIC\tOUTCOME\tCURRENT\tREFERENCE\tCHANGE\tDETAIL")
	num := func(p *float64) string {
		if p == nil {
			return "-"
		}
		return strconv.FormatFloat(*p, 'g', 6, 64)
	}
	for _, r := range rs {
		change := "-"
		if r.RelDelta != nil {
			change = fmt.Sprintf("%+.1f%%", 100**r.RelDelta)
		}
		detail := r.Cause
		if detail == "" && len(r.Interval) == 2 {
			detail = fmt.Sprintf("%s [%.4g, %.4g]", r.Method, r.Interval[0], r.Interval[1])
		}
		fmt.Fprintf(w, "%s\t%s\t%s\t%s\t%s\t%s\n", r.Key, r.Outcome, num(r.Current), num(r.Reference), change, detail)
	}
	w.Flush()
}

// loadSide reads every envelope in each results directory (or its envelopes/
// subdirectory). A sanitised directory is refused: its identity is a
// pseudonym, so "same hardware?" cannot be answered honestly (#548).
func loadSide(dirs []string) (*compare.Side, error) {
	var envs []*contract.Envelope
	for _, d := range dirs {
		if isSanitized(d) || isSanitized(filepath.Dir(d)) {
			return nil, fmt.Errorf("%s is a sanitised result, and a sanitised result is never compared", d)
		}
		root := d
		if fi, err := os.Stat(filepath.Join(d, "envelopes")); err == nil && fi.IsDir() {
			root = filepath.Join(d, "envelopes")
		}
		err := filepath.WalkDir(root, func(p string, e fs.DirEntry, err error) error {
			if err != nil || e.IsDir() || filepath.Ext(p) != ".json" {
				return err
			}
			data, err := os.ReadFile(p)
			if err != nil {
				return err
			}
			var env contract.Envelope
			dec := json.NewDecoder(strings.NewReader(string(data)))
			dec.DisallowUnknownFields()
			if err := dec.Decode(&env); err != nil {
				return fmt.Errorf("%s is not an envelope: %w", p, err)
			}
			envs = append(envs, &env)
			return nil
		})
		if err != nil {
			return nil, err
		}
	}
	return compare.Collect(envs)
}
