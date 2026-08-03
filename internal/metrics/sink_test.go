package metrics

import (
	"context"
	"strings"
	"testing"

	"github.com/baldwinSPC/glimmer-burnin/pkg/contract"
)

// The Prometheus sink's whole job is to reflect the envelope it was handed
// onto /metrics. Delivering it must therefore change what a scrape sees.
func TestSinkDeliverExposesTheRun(t *testing.T) {
	e, reg := newExporter(t)
	s := e.SinkFor("burnin", "scrape-me")

	err := s.Deliver(context.Background(), envelopeFor("Passed", contract.TestResult{
		Name: "smoke", Kind: "compute-smoke", Scope: "Node", Phase: "Passed",
		Nodes: []string{"node-a"}, StartedAt: &testStart, FinishedAt: &testFinish,
		Metrics: map[string]string{"eccErrors": "0"},
	}))
	if err != nil {
		t.Fatalf("Deliver = %v, want nil — exposition has no step that can fail", err)
	}

	got := snapshot(t, reg)
	wantValue(t, got, "burnin_run_phase", map[string]string{
		"namespace": "burnin", "run": "run-1", "profile": "acceptance", "phase": "Passed",
	}, 1)
	wantValue(t, got, "burnin_test_metric", map[string]string{
		"namespace": "burnin", "run": "run-1", "test": "smoke", "kind": "compute-smoke",
		"scope": "Node", "node": "node-a", "metric": "eccErrors",
		"unit": "", "threshold_use": "Acceptance",
	}, 0)
}

// Redelivery is routine — the reconciler resends a terminal envelope until a
// sink accepts it. Reflecting the same envelope twice must be indistinguishable
// from reflecting it once.
func TestSinkDeliverIsIdempotent(t *testing.T) {
	e, reg := newExporter(t)
	s := e.SinkFor("burnin", "scrape-me")
	env := envelopeFor("Passed", contract.TestResult{
		Name: "smoke", Kind: "compute-smoke", Scope: "Node", Phase: "Passed",
		Nodes: []string{"node-a"}, Metrics: map[string]string{"eccErrors": "0"},
	})

	if err := s.Deliver(context.Background(), env); err != nil {
		t.Fatalf("first Deliver: %v", err)
	}
	first := snapshot(t, reg)
	if err := s.Deliver(context.Background(), env); err != nil {
		t.Fatalf("second Deliver: %v", err)
	}
	second := snapshot(t, reg)

	if len(first) != len(second) {
		t.Errorf("redelivery changed the series count: %d then %d", len(first), len(second))
	}
	for k, v := range first {
		if second[k] != v {
			t.Errorf("redelivery changed %s: %v then %v", k, v, second[k])
		}
	}
}

func TestSinkDescribeNamesTheSinkAndTheEndpoint(t *testing.T) {
	e, _ := newExporter(t)
	got := e.SinkFor("burnin", "scrape-me").Describe()
	if !strings.Contains(got, "burnin/scrape-me") || !strings.Contains(got, "/metrics") {
		t.Errorf("Describe() = %q, want the sink's name and where the result can be read", got)
	}
}

// A nil envelope is a bug on this side, not a delivery problem, and it must not
// be quietly counted as a successful export.
func TestSinkRejectsNilEnvelope(t *testing.T) {
	e, _ := newExporter(t)
	if err := e.SinkFor("burnin", "scrape-me").Deliver(context.Background(), nil); err == nil {
		t.Error("Deliver(nil) = nil, want an error")
	}
}
