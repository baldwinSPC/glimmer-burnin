package compare

import (
	"fmt"
	"sort"
	"time"

	"github.com/baldwinSPC/glimmer-burnin/pkg/contract"
	"github.com/baldwinSPC/glimmer-burnin/pkg/stats"
)

// BaselineSchema versions the baseline file.
const BaselineSchema = "burnin.glimmer.ai/baseline/v1"

// MinReplicates is the least number of runs whose spread says anything about
// normal run-to-run variation. A metric promoted from fewer is recorded but not
// enforceable.
const MinReplicates = 5

// Baseline is a promoted set of replicate runs of ONE node.
type Baseline struct {
	Schema      string                    `json:"schema"`
	Name        string                    `json:"name"`
	PromotedAt  time.Time                 `json:"promotedAt"`
	Producer    *contract.Producer        `json:"producer,omitempty"`
	Node        string                    `json:"node"`
	Fingerprint string                    `json:"fingerprint"`
	SourceRuns  []string                  `json:"sourceRuns"`
	Replicates  int                       `json:"replicates"`
	Metrics     map[string]BaselineMetric `json:"metrics"`
}

// BaselineMetric is one metric's replicate distribution. Mean is the centre;
// SD and CV are the SAMPLE spread across runs, absent below two replicates.
type BaselineMetric struct {
	Test        string    `json:"test"`
	Kind        string    `json:"kind"`
	Metric      string    `json:"metric"`
	Values      []float64 `json:"values"`
	N           int       `json:"n"`
	Mean        float64   `json:"mean"`
	SD          *float64  `json:"sd,omitempty"`
	CV          *float64  `json:"cv,omitempty"`
	Enforceable bool      `json:"enforceable"`
	// Partial is true when fewer runs contributed than were promoted: a run
	// that errored on this test contributes nothing rather than a fake value.
	Partial bool `json:"partial,omitempty"`
}

// Promote builds a baseline from replicate runs of one node. Every run must
// carry the same fingerprint: a baseline mixing kernels or drivers would give
// a spread that is partly configuration drift.
func Promote(name string, s *Side, producer *contract.Producer, now time.Time) (*Baseline, error) {
	for _, f := range s.RunFingerprints {
		if d := DifferingFields(s.Fingerprint, f); len(d) > 0 {
			return nil, fmt.Errorf("the runs do not share one fingerprint (differ in %v); promote runs of one configuration", d)
		}
	}
	b := &Baseline{
		Schema: BaselineSchema, Name: name, PromotedAt: now.UTC(), Producer: producer,
		Node: s.Node, Fingerprint: s.Fingerprint, SourceRuns: s.Runs,
		Replicates: len(s.Runs), Metrics: map[string]BaselineMetric{},
	}
	keys := make([]string, 0, len(s.Metrics))
	for k := range s.Metrics {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	for _, k := range keys {
		o := s.Metrics[k]
		if len(o.Values) == 0 {
			continue
		}
		m := BaselineMetric{
			Test: k[:indexSlash(k)], Kind: o.Kind, Metric: metricOf(k),
			Values: o.Values, N: len(o.Values), Mean: stats.Mean(o.Values),
			Partial: len(o.Values) < len(s.Runs),
		}
		if sd, ok := stats.SampleSD(o.Values); ok {
			m.SD = &sd
		}
		if cv, ok := stats.CV(o.Values); ok {
			m.CV = &cv
		}
		m.Enforceable = m.N >= 2 && m.N >= MinReplicates && m.CV != nil
		b.Metrics[k] = m
	}
	if len(b.Metrics) == 0 {
		return nil, fmt.Errorf("no numeric metric came from a clean run; there is nothing to promote")
	}
	return b, nil
}

func indexSlash(k string) int {
	for i := 0; i < len(k); i++ {
		if k[i] == '/' {
			return i
		}
	}
	return len(k)
}
