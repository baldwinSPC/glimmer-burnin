// Package metrics exposes the state of BurnInRuns on the operator's Prometheus
// endpoint.
//
// # Why this package exists
//
// BurnInSink accepts Type=Prometheus in its CRD enum, and until this package
// existed the reconciler then failed the delivery permanently with "Prometheus
// sinks are scraped, not delivered to". That is the worst kind of API: the
// object validates, the operator accepts it, and the only feedback is a
// permanent error on every export for the rest of the run. Either the enum
// value works or it should never have been accepted.
//
// # The design, stated honestly
//
// Prometheus is a pull system, so a "Prometheus sink" cannot be a destination
// in the way a webhook is. What it can be — and what it now is — is a
// SELECTOR. Naming a Prometheus BurnInSink in a profile's sinks list marks that
// profile's runs for exposition: every envelope the run would have delivered is
// instead reflected into this exporter's gauges, and Prometheus scrapes them
// from the manager's existing /metrics endpoint. Delivery therefore always
// succeeds, in-process and without I/O, and the sink's status records the
// exposition the same way it records a webhook POST. Nothing is queued and
// nothing is retried, because there is nothing that can fail.
//
// The consequence worth being explicit about: exposition is NOT durable
// delivery. A scrape that does not happen before the operator restarts loses
// that sample, and this exporter forgets a run when the run is swept or evicted
// (see below). The system of record for a verdict is the BurnInRun object and
// the envelope delivered to a Webhook or ConfigMap sink. A Prometheus sink is
// for dashboards and alerts, and a fleet that gates acceptance on a scrape has
// gated it on a cache. Configure a Prometheus sink ALONGSIDE a durable one, not
// instead of it.
//
// # The feed is the delivery envelope
//
// The exporter is fed contract.Envelope values — the same document every other
// sink receives — rather than the BurnInRun object. That is deliberate: the
// metrics view and the delivered verdict then cannot disagree, and a consumer
// comparing a dashboard against a webhook payload is looking at one source of
// truth. It also means the exporter inherits the envelope's limits, which are
// called out at each affected metric below.
//
// # Cardinality
//
// Every series here is per-run, and runs are created forever, so an exporter
// that only ever added series would grow without bound and eventually take the
// scrape target — and possibly the Prometheus server — down with it. Three
// things bound it:
//
//  1. Each Observe replaces a run's whole series set. A series that no longer
//     describes the run (a phase it has left, a test that lost a node, a metric
//     the runner stopped emitting) is deleted in the same call that writes the
//     replacement, so a long run does not accumulate history in the registry.
//  2. Forget and Sweep delete every series for a run. Sweep is level-based —
//     hand it the runs that currently exist and it removes the rest — so a
//     missed delete event cannot strand series the way an event-driven hook
//     would. This is the intended mechanism and it is the caller's to wire.
//  3. If nobody wires Sweep, MaxRuns evicts the least-recently-observed run.
//     This is a backstop, not a policy: evicting a run silently drops a
//     dashboard's data, which is only tolerable because the verdict itself
//     lives elsewhere. burnin_exporter_runs_evicted_total makes it visible
//     rather than mysterious.
//
// Only metrics REGISTERED in pkg/contract become series. See exposable for why
// that filter is registration rather than contract.SafeToThresholdOn.
package metrics

import (
	"errors"
	"math"
	"sort"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/prometheus/client_golang/prometheus"
	ctrlmetrics "sigs.k8s.io/controller-runtime/pkg/metrics"

	burninv1alpha1 "github.com/baldwinSPC/glimmer-burnin/api/v1alpha1"
	"github.com/baldwinSPC/glimmer-burnin/pkg/contract"
)

// DefaultMaxRuns bounds how many runs an exporter holds series for when no
// caller sweeps it.
//
// The number comes from arithmetic, not taste. A run costs roughly
// 16 + tests*nodes*(1 + metrics) series: a typical 4-test, 8-node run with six
// metrics apiece is about 250 series, so 128 runs is ~32k series — large but
// survivable for a scrape target. A pathological run (64 nodes, 8 tests, 10
// metrics) is ~5.6k series on its own, and 128 of those would not be
// survivable, which is exactly why this is called a backstop and Sweep is
// called the mechanism.
const DefaultMaxRuns = 128

// RunID identifies a run in label space.
//
// It is namespace/name and deliberately NOT the UID, because namespace/name is
// how an operator reading a dashboard identifies a run, and because a run
// recreated under the same name then takes over the existing series instead of
// leaving an orphaned set behind. The envelope still carries the UID, which
// remains the authoritative identity for anything that must not confuse two
// runs — this exporter is not that thing.
type RunID struct {
	Namespace string
	Name      string
}

// Vector identifiers, used as the first component of a series key.
const (
	vecRunPhase   = "run_phase"
	vecRunTests   = "run_tests"
	vecRunStart   = "run_start"
	vecRunFinish  = "run_finish"
	vecTestPhase  = "test_phase"
	vecTestMetric = "test_metric"
)

// knownPhases is the phase set enumerated by burnin_run_phase and
// burnin_run_tests. An observed phase that is not in this list is still
// exported (see Observe): a phase this build has never heard of is information,
// and silently omitting it would make a newer control plane's run look absent
// rather than unfamiliar.
var knownPhases = []string{
	string(burninv1alpha1.RunPending),
	string(burninv1alpha1.RunRunning),
	string(burninv1alpha1.RunPassed),
	string(burninv1alpha1.RunFailed),
	string(burninv1alpha1.RunError),
	string(burninv1alpha1.RunSkipped),
	string(burninv1alpha1.RunCancelled),
}

// Drop reasons for burnin_exporter_metrics_dropped_total.
const (
	dropUnregistered = "unregistered"
	dropNonNumeric   = "non_numeric"
	dropNonFinite    = "non_finite"
)

// sample is one gauge series and the value it should carry.
type sample struct {
	vec  string
	vals []string
	val  float64
}

// runState is everything the exporter remembers about one run.
type runState struct {
	lastSeen time.Time
	// series maps an identity key to the series currently published for it.
	// The identity key deliberately excludes the labels that CHANGE for a
	// given fact — a test's phase, for instance — so that a phase transition
	// is recognised as the same fact wearing new labels, and the old series is
	// deleted rather than left behind at 1 forever.
	series map[string]sample
}

// Exporter publishes run state as Prometheus gauges.
//
// It is safe for concurrent use: deliveries happen inside the reconcile loop,
// which controller-runtime may run with concurrency greater than one.
type Exporter struct {
	mu      sync.Mutex
	runs    map[RunID]*runState
	maxRuns int
	now     func() time.Time

	vecs map[string]*prometheus.GaugeVec

	runsTracked    prometheus.Gauge
	runsEvicted    prometheus.Counter
	metricsDropped *prometheus.CounterVec

	collectors []prometheus.Collector
}

// Default is the exporter the Prometheus sink writes to. It is registered on
// controller-runtime's registry by init, so the manager's existing /metrics
// endpoint serves it with no wiring in cmd/main.go.
var Default = New()

func init() {
	// A failure here is a duplicate descriptor, i.e. a programming error in
	// this package, and it must not be swallowed: the alternative is an
	// operator staring at a /metrics endpoint that silently omits burn-in.
	if err := Default.Register(ctrlmetrics.Registry); err != nil {
		panic("glimmer-burnin: registering burn-in metrics: " + err.Error())
	}
}

// Register registers Default on controller-runtime's registry. It is idempotent
// and exists for callers that want to assert registration happened rather than
// rely on package initialisation order.
func Register() error { return Default.Register(ctrlmetrics.Registry) }

// New builds an unregistered exporter.
func New() *Exporter {
	e := &Exporter{
		runs:    map[RunID]*runState{},
		maxRuns: DefaultMaxRuns,
		now:     time.Now,
	}

	runPhase := prometheus.NewGaugeVec(prometheus.GaugeOpts{
		Name: "burnin_run_phase",
		Help: "Current phase of a BurnInRun: 1 for the phase it is in, 0 for the others. Error means the hardware was not judged and is never folded into Failed; alert on the two separately.",
	}, []string{"namespace", "run", "profile", "phase"})

	runTests := prometheus.NewGaugeVec(prometheus.GaugeOpts{
		Name: "burnin_run_tests",
		Help: "Recorded test results in a run, by phase, counted in per-(test,node) executions. Passed equals the number of tests only for a single-node run.",
	}, []string{"namespace", "run", "phase"})

	runStart := prometheus.NewGaugeVec(prometheus.GaugeOpts{
		Name: "burnin_run_start_time_seconds",
		Help: "Unix time at which the run's earliest recorded test execution began. Absent until a test has started.",
	}, []string{"namespace", "run"})

	runFinish := prometheus.NewGaugeVec(prometheus.GaugeOpts{
		Name: "burnin_run_finish_time_seconds",
		Help: "Unix time at which the run's last recorded test execution ended. Published only once the run reaches a terminal phase; absent while it is still running.",
	}, []string{"namespace", "run"})

	testPhase := prometheus.NewGaugeVec(prometheus.GaugeOpts{
		Name: "burnin_test_phase",
		Help: "Outcome of one test on one node: always 1, with the outcome carried in the phase label. Emitted once per node the test covered, so a failed Group test marks every node in the group.",
	}, []string{"namespace", "run", "test", "kind", "scope", "node", "phase"})

	testMetric := prometheus.NewGaugeVec(prometheus.GaugeOpts{
		Name: "burnin_test_metric",
		Help: "A numeric metric parsed from a runner's stdout. The unit is in the name (see pkg/contract) and repeated in the unit label; threshold_use=Evidence marks a metric the registry says acceptance must NOT be decided on.",
	}, []string{"namespace", "run", "test", "kind", "scope", "node", "metric", "unit", "threshold_use"})

	e.vecs = map[string]*prometheus.GaugeVec{
		vecRunPhase:   runPhase,
		vecRunTests:   runTests,
		vecRunStart:   runStart,
		vecRunFinish:  runFinish,
		vecTestPhase:  testPhase,
		vecTestMetric: testMetric,
	}

	e.runsTracked = prometheus.NewGauge(prometheus.GaugeOpts{
		Name: "burnin_exporter_runs_tracked",
		Help: "Runs this exporter currently holds series for. Compare against burnin_exporter_runs_evicted_total to tell a healthy sweep from a full cache.",
	})
	e.runsEvicted = prometheus.NewCounter(prometheus.CounterOpts{
		Name: "burnin_exporter_runs_evicted_total",
		Help: "Runs whose series were dropped because the exporter hit its run limit. Non-zero means dashboards are losing runs and nothing is sweeping the exporter.",
	})
	e.metricsDropped = prometheus.NewCounterVec(prometheus.CounterOpts{
		Name: "burnin_exporter_metrics_dropped_total",
		Help: "Runner metrics that were not exposed, by reason. reason=unregistered means the name is not in pkg/contract's registry and is therefore unbounded label input; register the name to chart it.",
	}, []string{"reason"})

	e.collectors = []prometheus.Collector{
		runPhase, runTests, runStart, runFinish, testPhase, testMetric,
		e.runsTracked, e.runsEvicted, e.metricsDropped,
	}
	return e
}

// SetMaxRuns changes the eviction backstop. A value below 1 is ignored: an
// exporter that evicts everything it is told is worse than one that grows,
// because it reports nothing while looking healthy.
func (e *Exporter) SetMaxRuns(n int) {
	if n < 1 {
		return
	}
	e.mu.Lock()
	defer e.mu.Unlock()
	e.maxRuns = n
	e.evictLocked()
	e.runsTracked.Set(float64(len(e.runs)))
}

// Register adds every collector to r.
//
// It is idempotent: registering an exporter twice, or registering two exporters
// that own the same descriptors, is reported by client_golang as
// AlreadyRegisteredError and is treated as success. Anything else is a real
// failure and is returned, because a partially registered exporter serves a
// /metrics endpoint that is missing exactly the series someone is about to
// alert on.
func (e *Exporter) Register(r prometheus.Registerer) error {
	for _, c := range e.collectors {
		if err := r.Register(c); err != nil {
			var already prometheus.AlreadyRegisteredError
			if errors.As(err, &already) {
				continue
			}
			return err
		}
	}
	return nil
}

// Observe reflects one delivery envelope onto the exporter's gauges.
//
// It is a full replacement for the run: the envelope is cumulative (it carries
// every result recorded so far), so anything the previous envelope published
// and this one does not is stale by definition and is deleted. New series are
// written BEFORE stale ones are removed, so a scrape landing mid-update sees a
// superset rather than a hole.
func (e *Exporter) Observe(env *contract.Envelope) {
	if env == nil || env.Run.Name == "" {
		return
	}
	id := RunID{Namespace: env.Run.Namespace, Name: env.Run.Name}
	next := make(map[string]sample)
	add := func(vec, identity string, val float64, vals ...string) {
		next[vec+"\x00"+identity] = sample{vec: vec, vals: vals, val: val}
	}

	ns, name, profile := id.Namespace, id.Name, env.Run.Profile

	// --- Run phase ------------------------------------------------------
	known := false
	for _, p := range knownPhases {
		v := 0.0
		if p == env.Phase {
			v, known = 1, true
		}
		add(vecRunPhase, p, v, ns, name, profile, p)
	}
	if env.Phase != "" && !known {
		add(vecRunPhase, env.Phase, 1, ns, name, profile, env.Phase)
	}

	// --- Result tallies -------------------------------------------------
	//
	// Counted from Results rather than taken from Summary. Summary carries
	// Passed and Failed only — by contract, not by omission — and a dashboard
	// showing "0 failed" beside eight tests that errored is precisely the
	// collapse of Error into Fail that this project forbids everywhere else.
	// The two agree by construction: Summary is this same count over these
	// same results.
	counts := map[string]int{}
	for _, r := range env.Results {
		counts[r.Phase]++
	}
	for _, p := range knownPhases {
		add(vecRunTests, p, float64(counts[p]), ns, name, p)
		delete(counts, p)
	}
	for p, n := range counts {
		add(vecRunTests, p, float64(n), ns, name, p)
	}

	// --- Run timestamps -------------------------------------------------
	//
	// Derived from the results because the envelope — this exporter's only
	// input — carries per-test timestamps and not the run object's own. Start
	// is therefore when the first test EXECUTION began, which is slightly
	// later than status.startedAt (that includes scheduling and image pull);
	// the difference is minutes on a soak measured in hours, and keeping the
	// exporter fed by the delivered document matters more than those minutes.
	//
	// They are derived from results rather than from the envelope's SentAt for
	// a second reason: a terminal delivery is re-sent until a sink accepts it,
	// and SentAt moves on every attempt. A finish timestamp that walked
	// forward while the run sat finished would be a lie a dashboard could not
	// detect.
	if start, ok := earliestStart(env.Results); ok {
		add(vecRunStart, "", float64(start.Unix()), ns, name)
	}
	// Only once terminal: mid-run, the latest finished test is not the run's
	// finish, and publishing it would show a running soak as already over.
	if burninv1alpha1.RunPhase(env.Phase).IsTerminal() {
		if finish, ok := latestFinish(env.Results); ok {
			add(vecRunFinish, "", float64(finish.Unix()), ns, name)
		}
	}

	// --- Per test/node results and metrics -------------------------------
	for _, r := range env.Results {
		nodes := r.Nodes
		if len(nodes) == 0 {
			// A result recorded before a node was chosen — a resolution error,
			// or a scope this build cannot execute. It still has a phase, and
			// an unjudged test must be visible.
			nodes = []string{""}
		}
		for _, node := range nodes {
			add(vecTestPhase, r.Name+"\x00"+node, 1,
				ns, name, r.Name, r.Kind, r.Scope, node, r.Phase)
		}

		// A metric belongs to a node only when exactly one node produced it.
		// A Group-scope figure — NCCL bus bandwidth across eight ranks — is a
		// property of the set, and copying it onto each member would let a
		// per-node dashboard sum one measurement eight times.
		metricNode := ""
		if len(r.Nodes) == 1 {
			metricNode = r.Nodes[0]
		}
		for _, key := range sortedKeys(r.Metrics) {
			m, ok := e.exposable(key)
			if !ok {
				continue
			}
			v, ok := e.numeric(r.Metrics[key])
			if !ok {
				continue
			}
			add(vecTestMetric, r.Name+"\x00"+metricNode+"\x00"+key, v,
				ns, name, r.Name, r.Kind, r.Scope, metricNode,
				key, string(m.Unit), string(m.ThresholdUse))
		}
	}

	e.mu.Lock()
	defer e.mu.Unlock()
	e.applyLocked(id, next)
	e.evictLocked()
	e.runsTracked.Set(float64(len(e.runs)))
}

// exposable decides whether a parsed metric becomes a series.
//
// The filter is REGISTRATION in pkg/contract, and deliberately not
// contract.SafeToThresholdOn. That function answers true for an unregistered
// name — correctly, because the registry is an open world and a third-party
// runner's author knows what their metric means better than this package does.
// As an exposition filter it would therefore do the opposite of what is wanted
// in both directions: it would admit every unregistered name, which is
// unbounded label input and the one thing that can actually break a Prometheus
// server, and it would suppress ratedBoostClockMHz and throughputTflops, which
// are marked Evidence precisely because they are worth recording and charting
// but not worth gating on.
//
// So: registered names are exposed, whatever their ThresholdUse, and the
// registry's judgement travels with the series as the threshold_use label so
// nobody writes a page-at-3am alert on an evidence metric without seeing it.
// Unregistered names are counted, not published — the name is the contract, and
// a runner that wants its measurement on a dashboard should register it.
func (e *Exporter) exposable(name string) (contract.Metric, bool) {
	m, ok := contract.Lookup(name)
	if !ok {
		e.metricsDropped.WithLabelValues(dropUnregistered).Inc()
		return contract.Metric{}, false
	}
	return m, true
}

// numeric parses a metric value, refusing anything a gauge cannot honestly
// carry. NaN and Inf are legal float64 values and legal Prometheus samples, but
// as a measurement they mean the runner emitted garbage; publishing them makes
// every average and every threshold downstream silently non-finite too.
func (e *Exporter) numeric(raw string) (float64, bool) {
	v, err := strconv.ParseFloat(strings.TrimSpace(raw), 64)
	if err != nil {
		e.metricsDropped.WithLabelValues(dropNonNumeric).Inc()
		return 0, false
	}
	if math.IsNaN(v) || math.IsInf(v, 0) {
		e.metricsDropped.WithLabelValues(dropNonFinite).Inc()
		return 0, false
	}
	return v, true
}

// applyLocked writes next and deletes whatever the run published before that
// next does not.
func (e *Exporter) applyLocked(id RunID, next map[string]sample) {
	st := e.runs[id]
	if st == nil {
		st = &runState{}
		e.runs[id] = st
	}
	st.lastSeen = e.now()

	for _, s := range next {
		e.vecs[s.vec].WithLabelValues(s.vals...).Set(s.val)
	}
	for key, prev := range st.series {
		// Same identity with the same labels: already overwritten above.
		// Same identity with DIFFERENT labels (a phase transition) still needs
		// the old series removed, or the run would read as being in two phases
		// at once.
		if cur, ok := next[key]; ok && equalVals(prev.vals, cur.vals) {
			continue
		}
		e.vecs[prev.vec].DeleteLabelValues(prev.vals...)
	}
	st.series = next
}

// Forget deletes every series for one run. Call it when a run is deleted or
// garbage-collected; the series are a view of a live object and must not
// outlive it.
func (e *Exporter) Forget(namespace, name string) {
	e.mu.Lock()
	defer e.mu.Unlock()
	e.forgetLocked(RunID{Namespace: namespace, Name: name})
	e.runsTracked.Set(float64(len(e.runs)))
}

// Sweep keeps the runs in live and forgets every other run, returning how many
// it forgot.
//
// It is level-based on purpose. An exporter that only reacted to delete events
// would leak the series of any run deleted while the operator was down, and
// those series would then never be removed by anything. Handing Sweep the runs
// that currently exist makes a missed event self-correcting on the next pass.
func (e *Exporter) Sweep(live []RunID) int {
	keep := make(map[RunID]struct{}, len(live))
	for _, id := range live {
		keep[id] = struct{}{}
	}
	e.mu.Lock()
	defer e.mu.Unlock()
	forgotten := 0
	for id := range e.runs {
		if _, ok := keep[id]; ok {
			continue
		}
		e.forgetLocked(id)
		forgotten++
	}
	e.runsTracked.Set(float64(len(e.runs)))
	return forgotten
}

// Tracked lists the runs the exporter currently holds series for, sorted.
func (e *Exporter) Tracked() []RunID {
	e.mu.Lock()
	defer e.mu.Unlock()
	out := make([]RunID, 0, len(e.runs))
	for id := range e.runs {
		out = append(out, id)
	}
	sort.Slice(out, func(i, j int) bool {
		if out[i].Namespace != out[j].Namespace {
			return out[i].Namespace < out[j].Namespace
		}
		return out[i].Name < out[j].Name
	})
	return out
}

func (e *Exporter) forgetLocked(id RunID) {
	st, ok := e.runs[id]
	if !ok {
		return
	}
	for _, s := range st.series {
		e.vecs[s.vec].DeleteLabelValues(s.vals...)
	}
	delete(e.runs, id)
}

// evictLocked drops the least-recently-observed runs until the exporter is
// within its limit. See DefaultMaxRuns for why this exists and why it is not
// the mechanism anyone should be relying on.
func (e *Exporter) evictLocked() {
	for len(e.runs) > e.maxRuns {
		var oldest RunID
		var oldestAt time.Time
		first := true
		for id, st := range e.runs {
			if first || st.lastSeen.Before(oldestAt) {
				oldest, oldestAt, first = id, st.lastSeen, false
			}
		}
		if first {
			return
		}
		e.forgetLocked(oldest)
		e.runsEvicted.Inc()
	}
}

func earliestStart(results []contract.TestResult) (time.Time, bool) {
	var out time.Time
	found := false
	for _, r := range results {
		if r.StartedAt == nil || r.StartedAt.IsZero() {
			continue
		}
		if !found || r.StartedAt.Before(out) {
			out, found = *r.StartedAt, true
		}
	}
	return out, found
}

func latestFinish(results []contract.TestResult) (time.Time, bool) {
	var out time.Time
	found := false
	for _, r := range results {
		if r.FinishedAt == nil || r.FinishedAt.IsZero() {
			continue
		}
		if !found || r.FinishedAt.After(out) {
			out, found = *r.FinishedAt, true
		}
	}
	return out, found
}

func sortedKeys(m map[string]string) []string {
	out := make([]string, 0, len(m))
	for k := range m {
		out = append(out, k)
	}
	sort.Strings(out)
	return out
}

func equalVals(a, b []string) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}
