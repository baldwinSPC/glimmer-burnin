package metrics

import (
	"sort"
	"strings"
	"testing"
	"time"

	"github.com/prometheus/client_golang/prometheus"

	"github.com/baldwinSPC/glimmer-burnin/pkg/contract"
)

// ─── Helpers ─────────────────────────────────────────────────────────────────

// newExporter returns an exporter registered on its own registry, so tests do
// not fight each other (or the package-level Default) over global state.
func newExporter(t *testing.T) (*Exporter, *prometheus.Registry) {
	t.Helper()
	e := New()
	reg := prometheus.NewRegistry()
	if err := e.Register(reg); err != nil {
		t.Fatalf("Register: %v", err)
	}
	return e, reg
}

// snapshot renders everything the registry would serve as a flat
// "name{k=v,...}" -> value map, which is what a scrape actually sees.
func snapshot(t *testing.T, g prometheus.Gatherer) map[string]float64 {
	t.Helper()
	families, err := g.Gather()
	if err != nil {
		t.Fatalf("Gather: %v", err)
	}
	out := map[string]float64{}
	for _, f := range families {
		for _, m := range f.GetMetric() {
			pairs := make([]string, 0, len(m.GetLabel()))
			for _, l := range m.GetLabel() {
				pairs = append(pairs, l.GetName()+"="+l.GetValue())
			}
			sort.Strings(pairs)
			key := f.GetName()
			if len(pairs) > 0 {
				key += "{" + strings.Join(pairs, ",") + "}"
			}
			switch {
			case m.Gauge != nil:
				out[key] = m.Gauge.GetValue()
			case m.Counter != nil:
				out[key] = m.Counter.GetValue()
			}
		}
	}
	return out
}

func seriesKey(name string, labels map[string]string) string {
	pairs := make([]string, 0, len(labels))
	for k, v := range labels {
		pairs = append(pairs, k+"="+v)
	}
	sort.Strings(pairs)
	if len(pairs) == 0 {
		return name
	}
	return name + "{" + strings.Join(pairs, ",") + "}"
}

func wantValue(t *testing.T, got map[string]float64, name string, labels map[string]string, want float64) {
	t.Helper()
	key := seriesKey(name, labels)
	v, ok := got[key]
	if !ok {
		t.Errorf("series %s is absent", key)
		return
	}
	if v != want {
		t.Errorf("series %s = %v, want %v", key, v, want)
	}
}

func wantAbsent(t *testing.T, got map[string]float64, name string, labels map[string]string) {
	t.Helper()
	key := seriesKey(name, labels)
	if v, ok := got[key]; ok {
		t.Errorf("series %s should not exist, got %v", key, v)
	}
}

func countSeries(got map[string]float64, prefix string) int {
	n := 0
	for k := range got {
		if strings.HasPrefix(k, prefix) {
			n++
		}
	}
	return n
}

var (
	testStart  = time.Unix(1750000000, 0).UTC()
	testFinish = time.Unix(1750003600, 0).UTC()
)

func envelopeFor(phase string, results ...contract.TestResult) *contract.Envelope {
	return &contract.Envelope{
		Version:    contract.Version,
		DeliveryID: contract.NewDeliveryID("uid-1", contract.ReasonPhaseChanged, phase),
		Reason:     contract.ReasonPhaseChanged,
		SentAt:     testFinish,
		Run:        contract.RunRef{Namespace: "burnin", Name: "run-1", UID: "uid-1", Profile: "acceptance"},
		Phase:      phase,
		Results:    results,
	}
}

// ─── Registration ────────────────────────────────────────────────────────────

// Registration must survive being done twice. The exporter is registered from
// package init AND is exposed as Register() for callers that want to assert it
// happened; if the second call panicked or errored, a manager doing both would
// crash on startup with metrics that were in fact fine.
func TestRegistrationIsIdempotent(t *testing.T) {
	e := New()
	reg := prometheus.NewRegistry()

	if err := e.Register(reg); err != nil {
		t.Fatalf("first Register: %v", err)
	}
	if err := e.Register(reg); err != nil {
		t.Fatalf("second Register: %v", err)
	}

	// The package-level exporter is already registered by init; calling it
	// again must still be a no-op success.
	if err := Register(); err != nil {
		t.Fatalf("package Register: %v", err)
	}
	if err := Register(); err != nil {
		t.Fatalf("package Register (repeat): %v", err)
	}

	// Gather fails outright on duplicate families, so reaching here with a
	// populated exporter proves nothing was registered twice.
	e.Observe(envelopeFor("Running"))
	if got := snapshot(t, reg); len(got) == 0 {
		t.Fatal("registry served no series after Observe")
	}
}

// ─── Exposition ──────────────────────────────────────────────────────────────

func TestGaugesReflectRunState(t *testing.T) {
	e, reg := newExporter(t)

	e.Observe(envelopeFor("Failed",
		contract.TestResult{
			Name: "nccl", Kind: "nccl", Scope: "Node", Phase: "Passed",
			Nodes:     []string{"node-a"},
			StartedAt: &testStart, FinishedAt: &testFinish,
			Metrics: map[string]string{
				"busBandwidthGBs": "23.4",
				// Evidence, not Acceptance: still charted, but labelled so
				// nobody alerts on it by accident.
				"throughputTflops": "12",
				"eccErrors":        "0",
				// Not in the contract registry: unbounded label input.
				"vendorSecretSauce": "7",
				// A number-shaped field that is not a number.
				"gpuTempC": "hot",
			},
		},
		contract.TestResult{
			Name: "thermal", Kind: "thermal-soak", Scope: "Node", Phase: "Failed",
			Nodes:     []string{"node-b"},
			StartedAt: &testFinish, FinishedAt: &testFinish,
			Metrics: map[string]string{"gpuTempC": "91"},
		},
	))
	got := snapshot(t, reg)

	run := map[string]string{"namespace": "burnin", "run": "run-1"}
	runProf := map[string]string{"namespace": "burnin", "run": "run-1", "profile": "acceptance"}

	// Phase: exactly one phase is 1, and the rest are explicitly 0 rather than
	// missing, so a dashboard can chart a transition instead of a gap.
	wantValue(t, got, "burnin_run_phase", merge(runProf, map[string]string{"phase": "Failed"}), 1)
	wantValue(t, got, "burnin_run_phase", merge(runProf, map[string]string{"phase": "Passed"}), 0)
	wantValue(t, got, "burnin_run_phase", merge(runProf, map[string]string{"phase": "Running"}), 0)

	// Tallies. Error and Skipped are published beside Passed and Failed
	// because "0 failed" next to eight unjudged tests is not a clean sweep.
	wantValue(t, got, "burnin_run_tests", merge(run, map[string]string{"phase": "Passed"}), 1)
	wantValue(t, got, "burnin_run_tests", merge(run, map[string]string{"phase": "Failed"}), 1)
	wantValue(t, got, "burnin_run_tests", merge(run, map[string]string{"phase": "Error"}), 0)
	wantValue(t, got, "burnin_run_tests", merge(run, map[string]string{"phase": "Skipped"}), 0)

	// Timestamps: earliest start, and — since Failed is terminal — a finish.
	wantValue(t, got, "burnin_run_start_time_seconds", run, float64(testStart.Unix()))
	wantValue(t, got, "burnin_run_finish_time_seconds", run, float64(testFinish.Unix()))

	// Per test/node outcome.
	wantValue(t, got, "burnin_test_phase", map[string]string{
		"namespace": "burnin", "run": "run-1", "test": "nccl", "kind": "nccl",
		"scope": "Node", "node": "node-a", "phase": "Passed",
	}, 1)
	wantValue(t, got, "burnin_test_phase", map[string]string{
		"namespace": "burnin", "run": "run-1", "test": "thermal", "kind": "thermal-soak",
		"scope": "Node", "node": "node-b", "phase": "Failed",
	}, 1)

	// Parsed metrics, labelled by name, with the unit and the registry's
	// threshold judgement carried alongside.
	wantValue(t, got, "burnin_test_metric", map[string]string{
		"namespace": "burnin", "run": "run-1", "test": "nccl", "kind": "nccl",
		"scope": "Node", "node": "node-a", "metric": "busBandwidthGBs",
		"unit": "GBs", "threshold_use": "Acceptance",
	}, 23.4)
	// Dimensionless counters carry no unit suffix and no unit label value.
	wantValue(t, got, "burnin_test_metric", map[string]string{
		"namespace": "burnin", "run": "run-1", "test": "nccl", "kind": "nccl",
		"scope": "Node", "node": "node-a", "metric": "eccErrors",
		"unit": "", "threshold_use": "Acceptance",
	}, 0)
	// An Evidence metric is exposed — it is worth charting, it is just not
	// worth gating on, and the label is what says so.
	wantValue(t, got, "burnin_test_metric", map[string]string{
		"namespace": "burnin", "run": "run-1", "test": "nccl", "kind": "nccl",
		"scope": "Node", "node": "node-a", "metric": "throughputTflops",
		"unit": "Tflops", "threshold_use": "Evidence",
	}, 12)

	// An unregistered name is unbounded label input and is counted, not
	// published; a value that is not a number is likewise refused.
	for k := range got {
		if strings.Contains(k, "vendorSecretSauce") {
			t.Errorf("unregistered metric was exposed: %s", k)
		}
	}
	wantAbsent(t, got, "burnin_test_metric", map[string]string{
		"namespace": "burnin", "run": "run-1", "test": "nccl", "kind": "nccl",
		"scope": "Node", "node": "node-a", "metric": "gpuTempC",
		"unit": "C", "threshold_use": "Acceptance",
	})
	wantValue(t, got, "burnin_exporter_metrics_dropped_total", map[string]string{"reason": "unregistered"}, 1)
	wantValue(t, got, "burnin_exporter_metrics_dropped_total", map[string]string{"reason": "non_numeric"}, 1)

	wantValue(t, got, "burnin_exporter_runs_tracked", nil, 1)
}

// A run that is still going has not finished, however many of its tests have.
// Publishing the last completed test's end time as the run's finish would show
// a soak that is hours from done as already over.
func TestFinishTimeIsAbsentUntilTerminal(t *testing.T) {
	e, reg := newExporter(t)
	result := contract.TestResult{
		Name: "smoke", Kind: "compute-smoke", Scope: "Node", Phase: "Passed",
		Nodes: []string{"node-a"}, StartedAt: &testStart, FinishedAt: &testFinish,
	}

	e.Observe(envelopeFor("Running", result))
	got := snapshot(t, reg)
	run := map[string]string{"namespace": "burnin", "run": "run-1"}
	wantValue(t, got, "burnin_run_start_time_seconds", run, float64(testStart.Unix()))
	wantAbsent(t, got, "burnin_run_finish_time_seconds", run)

	e.Observe(envelopeFor("Passed", result))
	got = snapshot(t, reg)
	wantValue(t, got, "burnin_run_finish_time_seconds", run, float64(testFinish.Unix()))
}

// A Group-scope measurement belongs to the set of nodes, not to each of them.
// The verdict does apply to each node — a failed collective leaves every rank
// unjudged — so the phase is emitted per node while the number is not.
func TestGroupResultFansOutPhaseButNotMeasurements(t *testing.T) {
	e, reg := newExporter(t)
	e.Observe(envelopeFor("Failed", contract.TestResult{
		Name: "allreduce", Kind: "nccl", Scope: "Group", Phase: "Failed",
		Nodes:   []string{"node-a", "node-b"},
		Metrics: map[string]string{"busBandwidthGBs": "180"},
	}))
	got := snapshot(t, reg)

	for _, node := range []string{"node-a", "node-b"} {
		wantValue(t, got, "burnin_test_phase", map[string]string{
			"namespace": "burnin", "run": "run-1", "test": "allreduce", "kind": "nccl",
			"scope": "Group", "node": node, "phase": "Failed",
		}, 1)
	}
	if n := countSeries(got, "burnin_test_metric{"); n != 1 {
		t.Errorf("group measurement produced %d series, want 1 (it must not be copied onto each node)", n)
	}
	wantValue(t, got, "burnin_test_metric", map[string]string{
		"namespace": "burnin", "run": "run-1", "test": "allreduce", "kind": "nccl",
		"scope": "Group", "node": "", "metric": "busBandwidthGBs",
		"unit": "GBs", "threshold_use": "Acceptance",
	}, 180)
}

// ─── Staleness ───────────────────────────────────────────────────────────────

// Each Observe replaces the run's whole series set. Without that, a test that
// moved from Running to Failed would sit at 1 in both phases forever, and a
// metric a runner stopped reporting would keep serving its last value as if it
// were current.
func TestObserveRemovesSeriesTheRunNoLongerHas(t *testing.T) {
	e, reg := newExporter(t)

	e.Observe(envelopeFor("Running", contract.TestResult{
		Name: "soak", Kind: "thermal-soak", Scope: "Node", Phase: "Running",
		Nodes:   []string{"node-a"},
		Metrics: map[string]string{"gpuTempC": "70", "powerDrawW": "300"},
	}))
	e.Observe(envelopeFor("Failed", contract.TestResult{
		Name: "soak", Kind: "thermal-soak", Scope: "Node", Phase: "Failed",
		Nodes:   []string{"node-a"},
		Metrics: map[string]string{"gpuTempC": "95"},
	}))
	got := snapshot(t, reg)

	base := map[string]string{
		"namespace": "burnin", "run": "run-1", "test": "soak",
		"kind": "thermal-soak", "scope": "Node", "node": "node-a",
	}
	wantValue(t, got, "burnin_test_phase", merge(base, map[string]string{"phase": "Failed"}), 1)
	wantAbsent(t, got, "burnin_test_phase", merge(base, map[string]string{"phase": "Running"}))
	if n := countSeries(got, "burnin_test_phase{"); n != 1 {
		t.Errorf("%d test_phase series after a transition, want 1", n)
	}

	wantValue(t, got, "burnin_test_metric",
		merge(base, map[string]string{"metric": "gpuTempC", "unit": "C", "threshold_use": "Acceptance"}), 95)
	wantAbsent(t, got, "burnin_test_metric",
		merge(base, map[string]string{"metric": "powerDrawW", "unit": "W", "threshold_use": "Acceptance"}))
}

// ─── Garbage collection ──────────────────────────────────────────────────────

func TestForgetDeletesEverySeriesForARun(t *testing.T) {
	e, reg := newExporter(t)
	e.Observe(envelopeFor("Passed", contract.TestResult{
		Name: "smoke", Kind: "compute-smoke", Scope: "Node", Phase: "Passed",
		Nodes: []string{"node-a"}, StartedAt: &testStart, FinishedAt: &testFinish,
		Metrics: map[string]string{"eccErrors": "0"},
	}))
	if n := countSeries(snapshot(t, reg), "burnin_run_"); n == 0 {
		t.Fatal("no run series to forget")
	}

	e.Forget("burnin", "run-1")
	got := snapshot(t, reg)

	for _, prefix := range []string{
		"burnin_run_phase{", "burnin_run_tests{", "burnin_run_start_time_seconds{",
		"burnin_run_finish_time_seconds{", "burnin_test_phase{", "burnin_test_metric{",
	} {
		if n := countSeries(got, prefix); n != 0 {
			t.Errorf("%d %s series survived Forget", n, prefix)
		}
	}
	wantValue(t, got, "burnin_exporter_runs_tracked", nil, 0)
	if tracked := e.Tracked(); len(tracked) != 0 {
		t.Errorf("Tracked() = %v after Forget, want empty", tracked)
	}
}

// Sweep is the mechanism that stops cardinality growing forever: hand it the
// runs that still exist and everything else — including runs whose delete
// event was missed while the operator was down — goes away.
func TestSweepDeletesGarbageCollectedRuns(t *testing.T) {
	e, reg := newExporter(t)
	for _, name := range []string{"run-1", "run-2", "run-3"} {
		env := envelopeFor("Passed", contract.TestResult{
			Name: "smoke", Kind: "compute-smoke", Scope: "Node", Phase: "Passed",
			Nodes: []string{"node-a"}, Metrics: map[string]string{"eccErrors": "0"},
		})
		env.Run.Name = name
		e.Observe(env)
	}
	if n := len(e.Tracked()); n != 3 {
		t.Fatalf("Tracked() has %d runs, want 3", n)
	}

	if forgotten := e.Sweep([]RunID{{Namespace: "burnin", Name: "run-2"}}); forgotten != 2 {
		t.Errorf("Sweep forgot %d runs, want 2", forgotten)
	}
	got := snapshot(t, reg)

	if n := countSeries(got, "burnin_test_metric{"); n != 1 {
		t.Errorf("%d test_metric series after Sweep, want 1 (only the live run)", n)
	}
	for _, gone := range []string{"run-1", "run-3"} {
		for k := range got {
			if strings.Contains(k, `run=`+gone) {
				t.Errorf("series for garbage-collected run %s survived Sweep: %s", gone, k)
			}
		}
	}
	wantValue(t, got, "burnin_exporter_runs_tracked", nil, 1)
	if tracked := e.Tracked(); len(tracked) != 1 || tracked[0].Name != "run-2" {
		t.Errorf("Tracked() = %v, want only run-2", tracked)
	}
}

// If nobody ever sweeps, the exporter must still not grow without bound —
// otherwise a cluster that creates runs forever eventually takes its own
// scrape target down. Eviction is lossy and is meant to be visible.
func TestEvictionBoundsCardinalityWhenNobodySweeps(t *testing.T) {
	e, reg := newExporter(t)
	e.SetMaxRuns(2)
	// A deterministic clock so "least recently observed" is unambiguous.
	tick := time.Unix(1750000000, 0)
	e.now = func() time.Time { tick = tick.Add(time.Second); return tick }

	for _, name := range []string{"run-1", "run-2", "run-3", "run-4", "run-5"} {
		env := envelopeFor("Passed", contract.TestResult{
			Name: "smoke", Kind: "compute-smoke", Scope: "Node", Phase: "Passed",
			Nodes: []string{"node-a"}, Metrics: map[string]string{"eccErrors": "0"},
		})
		env.Run.Name = name
		e.Observe(env)
	}

	got := snapshot(t, reg)
	if n := len(e.Tracked()); n != 2 {
		t.Errorf("Tracked() has %d runs, want the 2 the limit allows", n)
	}
	if n := countSeries(got, "burnin_test_metric{"); n != 2 {
		t.Errorf("%d test_metric series, want 2 — one per retained run", n)
	}
	if n := countSeries(got, "burnin_run_phase{"); n != 2*len(knownPhases) {
		t.Errorf("%d run_phase series, want %d", n, 2*len(knownPhases))
	}
	wantValue(t, got, "burnin_exporter_runs_evicted_total", nil, 3)
	wantValue(t, got, "burnin_exporter_runs_tracked", nil, 2)

	// The survivors are the most recently observed ones.
	names := []string{e.Tracked()[0].Name, e.Tracked()[1].Name}
	sort.Strings(names)
	if names[0] != "run-4" || names[1] != "run-5" {
		t.Errorf("retained %v, want the two most recently observed (run-4, run-5)", names)
	}
}

// A phase this build has never heard of must be visible as itself. Dropping it
// would make a run reported by a newer control plane look like it had no phase
// at all, which reads as "not running" rather than "not understood".
func TestUnknownPhaseIsStillExposed(t *testing.T) {
	e, reg := newExporter(t)
	e.Observe(envelopeFor("Quarantined"))
	got := snapshot(t, reg)
	wantValue(t, got, "burnin_run_phase", map[string]string{
		"namespace": "burnin", "run": "run-1", "profile": "acceptance", "phase": "Quarantined",
	}, 1)
	wantValue(t, got, "burnin_run_phase", map[string]string{
		"namespace": "burnin", "run": "run-1", "profile": "acceptance", "phase": "Running",
	}, 0)
}

func TestObserveIgnoresUnusableEnvelopes(t *testing.T) {
	e, reg := newExporter(t)
	e.Observe(nil)
	e.Observe(&contract.Envelope{Phase: "Running"}) // no run name: nothing to label
	if got := snapshot(t, reg); countSeries(got, "burnin_run_phase{") != 0 {
		t.Errorf("an unusable envelope produced series: %v", got)
	}
}

func merge(a, b map[string]string) map[string]string {
	out := make(map[string]string, len(a)+len(b))
	for k, v := range a {
		out[k] = v
	}
	for k, v := range b {
		out[k] = v
	}
	return out
}
