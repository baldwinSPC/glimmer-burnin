// Package compare turns repeated burn-in runs into a baseline and judges a
// later run, or another node, against it (#536).
//
// A comparison is NOT a verdict. REGRESSION says a number moved against a
// baseline beyond noise; it does not say the hardware failed, because the
// baseline can itself be wrong. Thresholds decide Pass and Fail; this package
// decides whether something changed. The outcome vocabulary and its
// precedence are ported from an independent GB10 acceptance toolkit:
//
//	MISSING                 one side has the metric and the other does not
//	NOT_COMPARABLE(cause)   the two numbers cannot be placed side by side
//	UNUSABLE_INPUT          one side's number is not a clean measurement
//	INSUFFICIENT_PRECISION  the baseline has too little spread information
//	MATCH / NO_SIGNIFICANT_DIFFERENCE / REGRESSION / IMPROVEMENT
//
// First match wins. NOT_COMPARABLE and INSUFFICIENT_PRECISION are statements
// about the setup, never about the part.
package compare

import (
	"fmt"
	"math"
	"regexp"
	"sort"
	"strconv"
	"strings"

	"github.com/baldwinSPC/glimmer-burnin/pkg/contract"
	"github.com/baldwinSPC/glimmer-burnin/pkg/stats"
)

type Outcome string

const (
	Match                   Outcome = "MATCH"
	Improvement             Outcome = "IMPROVEMENT"
	Regression              Outcome = "REGRESSION"
	NoSignificantDifference Outcome = "NO_SIGNIFICANT_DIFFERENCE"
	NotComparable           Outcome = "NOT_COMPARABLE"
	InsufficientPrecision   Outcome = "INSUFFICIENT_PRECISION"
	UnusableInput           Outcome = "UNUSABLE_INPUT"
	Missing                 Outcome = "MISSING"
)

// Direction says which way is worse, read off the registry's Aggregation:
// a metric whose windows combine by Min is a floor (higher is better), by Max
// or Sum a ceiling or a fault count (lower is better). Last and unregistered
// names are neutral: nothing is inferred about somebody else's measurement.
type Direction int

const (
	Neutral Direction = iota
	HigherIsBetter
	LowerIsBetter
)

func DirectionOf(metric string) Direction {
	m, ok := contract.Lookup(metric)
	if !ok {
		return Neutral
	}
	switch m.Aggregation {
	case contract.AggMin:
		return HigherIsBetter
	case contract.AggMax, contract.AggSum:
		return LowerIsBetter
	}
	return Neutral
}

// Key identifies one measurement across runs: the test, its variant cell, and
// the metric. Two runs of the same profile produce the same keys.
func Key(r contract.TestResult, metric string) string {
	k := r.Name + "/" + metric
	if len(r.VariantAxes) > 0 {
		axes := make([]string, 0, len(r.VariantAxes))
		for a, v := range r.VariantAxes {
			axes = append(axes, a+"="+v)
		}
		sort.Strings(axes)
		k += "[" + strings.Join(axes, ",") + "]"
	}
	return k
}

// Observation is one metric's values across the runs on one side.
type Observation struct {
	Kind     string
	Values   []float64
	Unusable string // why the latest run's value is not a clean measurement
}

// Side is what a set of runs of one node measured.
type Side struct {
	Node        string
	Fingerprint string
	Runs        []string // run UIDs, in the order given
	// RunFingerprints is each run's fingerprint of Node, in run order. They
	// must agree for the runs to be replicates of one configuration.
	RunFingerprints []string
	Metrics         map[string]*Observation
	// Identity holds label-valued identity evidence the latest run reported,
	// such as nvmeSerialDigests (#540). Never compared as a quantity.
	Identity map[string]string
}

// Collect reads the terminal verdict envelopes of runs on ONE node. Metric
// values that are not finite numbers are skipped: labels and identity strings
// are evidence, not quantities.
func Collect(envs []*contract.Envelope) (*Side, error) {
	s := &Side{Metrics: map[string]*Observation{}, Identity: map[string]string{}}
	for _, e := range envs {
		if e.Reason != contract.ReasonPhaseChanged || !terminal(e.Phase) {
			continue
		}
		for node, fp := range e.Fingerprint {
			if s.Node == "" {
				s.Node, s.Fingerprint = node, fp
			} else if node != s.Node {
				return nil, fmt.Errorf("runs from more than one node (%s and %s); a side is one node", s.Node, node)
			}
			s.RunFingerprints = append(s.RunFingerprints, fp)
		}
		s.Runs = append(s.Runs, e.Run.UID)
		for _, r := range e.Results {
			for metric, raw := range r.Metrics {
				if identityMetrics[metric] {
					if v := strings.TrimSpace(raw); v != "" {
						s.Identity[metric] = v
					}
					continue
				}
				v, err := strconv.ParseFloat(strings.TrimSpace(raw), 64)
				if err != nil || math.IsNaN(v) || math.IsInf(v, 0) {
					continue
				}
				k := Key(r, metric)
				o := s.Metrics[k]
				if o == nil {
					o = &Observation{Kind: r.Kind}
					s.Metrics[k] = o
				}
				if r.Phase == "Error" || r.Phase == "Skipped" {
					// An Error run measured nothing trustworthy, and a Skip
					// measured nothing at all. Neither may enter a baseline.
					o.Unusable = "the test ended " + r.Phase
					continue
				}
				o.Unusable = ""
				o.Values = append(o.Values, v)
			}
		}
	}
	if len(s.Runs) == 0 {
		return nil, fmt.Errorf("no finished runs: a comparison needs a terminal RunPhaseChanged envelope")
	}
	return s, nil
}

// identityMetrics are the label metrics that say WHICH hardware a result came
// from. Two nodes sharing a value are one piece of hardware seen twice.
var identityMetrics = map[string]bool{"nvmeSerialDigests": true}

// sharedIdentity names the identity values two sides have in common.
func sharedIdentity(a, b *Side) []string {
	var out []string
	for metric := range identityMetrics {
		av, bv := a.Identity[metric], b.Identity[metric]
		if av == "" || bv == "" {
			continue
		}
		seen := map[string]bool{}
		for _, x := range strings.Split(av, ",") {
			seen[strings.TrimSpace(x)] = true
		}
		for _, y := range strings.Split(bv, ",") {
			if y = strings.TrimSpace(y); y != "" && seen[y] {
				out = append(out, metric+" "+y)
			}
		}
	}
	sort.Strings(out)
	return out
}

func terminal(phase string) bool {
	switch phase {
	case "Passed", "Failed", "Error", "Cancelled", "Skipped":
		return true
	}
	return false
}

var fpKey = regexp.MustCompile(`(?:^|\s)([A-Za-z][\w./-]*)=`)

// FingerprintFields splits "kernel=6.11 os=Ubuntu 24.04 LTS arch=arm64" into
// its fields; a value runs to the next "key=".
func FingerprintFields(fp string) map[string]string {
	out := map[string]string{}
	idx := fpKey.FindAllStringSubmatchIndex(fp, -1)
	for i, m := range idx {
		end := len(fp)
		if i+1 < len(idx) {
			end = idx[i+1][0]
		}
		out[fp[m[2]:m[3]]] = strings.TrimSpace(fp[m[1]:end])
	}
	return out
}

// DifferingFields names the fingerprint fields two nodes disagree on. This is
// also the version-symmetry check (#540): two nodes of one fleet on different
// kernels or drivers are not comparable, and that is worth saying first.
func DifferingFields(a, b string) []string {
	fa, fb := FingerprintFields(a), FingerprintFields(b)
	var out []string
	for k, v := range fa {
		if fb[k] != v {
			out = append(out, k)
		}
	}
	for k := range fb {
		if _, ok := fa[k]; !ok {
			out = append(out, k)
		}
	}
	sort.Strings(out)
	return out
}

// Result is one metric's comparison.
type Result struct {
	Key       string    `json:"key"`
	Outcome   Outcome   `json:"outcome"`
	Cause     string    `json:"cause,omitempty"`
	Current   *float64  `json:"current,omitempty"`
	Reference *float64  `json:"reference,omitempty"`
	RelDelta  *float64  `json:"relDelta,omitempty"`
	Method    string    `json:"method,omitempty"`
	Interval  []float64 `json:"interval,omitempty"`
	Enforced  bool      `json:"enforced,omitempty"`
}

// Options tune a comparison.
type Options struct {
	// Tolerance is the relative change below which two numbers MATCH whatever
	// the statistics say, per metric name; Default applies otherwise.
	Tolerance        map[string]float64
	DefaultTolerance float64
	Alpha            float64 // prediction-interval / bootstrap significance, 0.05
	Enforce          bool
	Seed             uint64
}

func (o Options) tolFor(key string) float64 {
	metric := key[strings.Index(key, "/")+1:]
	if i := strings.Index(metric, "["); i >= 0 {
		metric = metric[:i]
	}
	if t, ok := o.Tolerance[metric]; ok {
		return t
	}
	return o.DefaultTolerance
}

func fp(v float64) *float64 { return &v }

// decide is the shared final step: a change within tolerance is a MATCH
// however significant; beyond it, a neutral metric is only different, and a
// directional one is a REGRESSION or IMPROVEMENT only if the interval test is
// conclusive.
func decide(r *Result, x, ref, tol float64, dir Direction, conclusive bool) {
	d := x - ref
	rel := d / math.Abs(ref)
	r.RelDelta = fp(rel)
	switch {
	case math.Abs(rel) <= tol:
		r.Outcome = Match
	case dir == Neutral || d == 0 || !conclusive:
		r.Outcome = NoSignificantDifference
	case (d < 0) == (dir == HigherIsBetter):
		r.Outcome = Regression
	default:
		r.Outcome = Improvement
	}
}

func sortResults(rs []Result) []Result {
	sort.Slice(rs, func(i, j int) bool { return rs[i].Key < rs[j].Key })
	return rs
}

// AgainstBaseline judges ONE run against a promoted baseline.
func AgainstBaseline(cur *Side, b *Baseline, o Options) ([]Result, error) {
	if len(cur.Runs) != 1 {
		return nil, fmt.Errorf("compare one run against a baseline; got %d", len(cur.Runs))
	}
	var fpDiff []string
	if cur.Fingerprint == "" || b.Fingerprint == "" {
		fpDiff = []string{"(fingerprint absent)"}
	} else {
		fpDiff = DifferingFields(cur.Fingerprint, b.Fingerprint)
	}
	var out []Result
	seen := map[string]bool{}
	for key, bm := range b.Metrics {
		seen[key] = true
		r := Result{Key: key, Reference: fp(bm.Mean), Method: "prediction_interval"}
		c := cur.Metrics[key]
		switch {
		case c == nil || (len(c.Values) == 0 && c.Unusable == ""):
			r.Outcome, r.Cause = Missing, "present only in the baseline"
		case len(fpDiff) > 0:
			r.Outcome, r.Cause = NotComparable, "fingerprint differs: "+strings.Join(fpDiff, ", ")
		case c.Unusable != "":
			r.Outcome, r.Cause = UnusableInput, c.Unusable
		case bm.Mean == 0:
			r.Current = fp(c.Values[len(c.Values)-1])
			r.Outcome, r.Cause = NotComparable, "zero reference: a relative change from 0 is undefined"
		case bm.N < 2 || bm.SD == nil:
			r.Current = fp(c.Values[len(c.Values)-1])
			r.Outcome, r.Cause = InsufficientPrecision, fmt.Sprintf("baseline has %d replicate(s)", bm.N)
		default:
			x := c.Values[len(c.Values)-1]
			r.Current = fp(x)
			lo, hi, err := stats.PredictionInterval(bm.Mean, *bm.SD, bm.N, o.Alpha)
			if err != nil {
				return nil, err
			}
			r.Interval = []float64{lo, hi}
			decide(&r, x, bm.Mean, o.tolFor(key), DirectionOf(bm.Metric), x < lo || x > hi)
			r.Enforced = r.Outcome == Regression && bm.Enforceable && o.Enforce
		}
		out = append(out, r)
	}
	for key, c := range cur.Metrics {
		if !seen[key] && len(c.Values) > 0 {
			out = append(out, Result{Key: key, Outcome: Missing, Cause: "present only in this run", Current: fp(c.Values[len(c.Values)-1])})
		}
	}
	return sortResults(out), nil
}

// Symmetry compares two nodes' runs of one profile. With two or more runs on
// each side the decision is a bootstrap of the difference; with fewer it is a
// point comparison against the tolerance alone. It is never enforced: two
// nodes disagreeing does not say which one is wrong.
func Symmetry(left, right *Side, o Options) []Result {
	var out []Result
	fpDiff := DifferingFields(left.Fingerprint, right.Fingerprint)
	keys := map[string]bool{}
	for k := range left.Metrics {
		keys[k] = true
	}
	for k := range right.Metrics {
		keys[k] = true
	}
	for key := range keys {
		l, r := left.Metrics[key], right.Metrics[key]
		res := Result{Key: key}
		switch {
		case l == nil || r == nil || (len(l.Values) == 0 && l.Unusable == "") || (len(r.Values) == 0 && r.Unusable == ""):
			res.Outcome, res.Cause = Missing, "present on one node only"
		case len(fpDiff) > 0:
			res.Outcome, res.Cause = NotComparable, "fingerprint differs: "+strings.Join(fpDiff, ", ")
		case l.Unusable != "" || r.Unusable != "":
			res.Outcome, res.Cause = UnusableInput, strings.TrimSpace(l.Unusable+" "+r.Unusable)
		default:
			x, ref := stats.Mean(l.Values), stats.Mean(r.Values)
			res.Current, res.Reference = fp(x), fp(ref)
			if ref == 0 {
				res.Outcome, res.Cause = NotComparable, "zero reference"
				break
			}
			conclusive := true
			res.Method = "point_only"
			if len(l.Values) >= 2 && len(r.Values) >= 2 {
				lo, hi, c, err := stats.BootstrapDiffCI(l.Values, r.Values, 10000, 1-o.Alpha, o.Seed)
				if err == nil {
					res.Method, res.Interval, conclusive = "bootstrap_diff", []float64{lo, hi}, c
				}
			}
			decide(&res, x, ref, o.tolFor(key), DirectionOf(metricOf(key)), conclusive)
		}
		out = append(out, res)
	}
	// #540: the same drive on two nodes is a cloned image or one machine
	// reached under two names, which makes every other row meaningless.
	if shared := sharedIdentity(left, right); len(shared) > 0 && left.Node != right.Node {
		out = append(out, Result{Key: "identity", Outcome: NotComparable,
			Cause: "the same hardware appears on both nodes (" + strings.Join(shared, "; ") +
				"): a cloned image, or one machine reached under two names"})
	}
	return sortResults(out)
}

func metricOf(key string) string {
	m := key[strings.Index(key, "/")+1:]
	if i := strings.Index(m, "["); i >= 0 {
		m = m[:i]
	}
	return m
}
