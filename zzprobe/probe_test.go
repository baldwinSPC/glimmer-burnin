package zzprobe

import (
	"encoding/json"
	"testing"
	"time"

	"github.com/baldwinSPC/glimmer-burnin/pkg/contract"
	"github.com/baldwinSPC/glimmer-burnin/pkg/report"
	"github.com/baldwinSPC/glimmer-burnin/pkg/report/nvvs"
)

func env(results []contract.TestResult, mutate func(*contract.Envelope)) *contract.Envelope {
	e := &contract.Envelope{
		Version: contract.Version, DeliveryID: "d-1", Reason: contract.ReasonPhaseChanged,
		Run:   contract.RunRef{Namespace: "burnin", Name: "run-42", UID: "uid-1", Profile: "acceptance"},
		Phase: "Failed", SentAt: time.Unix(1750000000, 0).UTC(), Results: results,
	}
	if mutate != nil {
		mutate(e)
	}
	return e
}

func dump(t *testing.T, in report.Input) {
	t.Helper()
	outs, err := nvvs.Renderer{}.Render(in)
	if err != nil {
		t.Fatalf("render: %v", err)
	}
	t.Logf("==== %d outputs", len(outs))
	for _, o := range outs {
		var m map[string]any
		if err := json.Unmarshal(o.Data, &m); err != nil {
			t.Fatal(err)
		}
		t.Logf("FILE %s", o.Filename)
		t.Logf("%s", string(o.Data))
		_ = m
	}
}

func TestProbe_FailHiddenByError(t *testing.T) {
	dump(t, report.Input{
		Envelopes: []*contract.Envelope{env([]contract.TestResult{
			{Name: "soak", Kind: "thermal-soak", Scope: "Node", Phase: "Failed", Nodes: []string{"spark-a"},
				Message:    "sustainedClockPct 61.2 < 80",
				Violations: []contract.Violation{{Metric: "sustainedClockPct", Cause: "Measurement", Reason: "61.2 < 80"}}},
			{Name: "fp4", Kind: "compute-smoke", Scope: "Node", Phase: "Error", Nodes: []string{"spark-a"},
				Message: "ImagePullBackOff"},
		}, nil)},
		Meta: report.Meta{Generator: "glimmer-burnin", Version: "v0.5.0"},
	})
}

func TestProbe_BaselinePass(t *testing.T) {
	dump(t, report.Input{
		Envelopes: []*contract.Envelope{env([]contract.TestResult{
			{Name: "soak", Kind: "thermal-soak", Scope: "Node", Phase: "Passed", Nodes: []string{"spark-a"}},
		}, func(e *contract.Envelope) { e.Baseline = true; e.Phase = "Passed" })},
		Meta: report.Meta{Generator: "glimmer-burnin", Version: "v0.5.0"},
	})
}

func TestProbe_BroadcastAndSerials(t *testing.T) {
	var gpus []report.GPUInfo
	for i := 0; i < 4; i++ {
		g := report.GPUInfo{Index: int32(i), Model: "NVIDIA GB10"}
		if i == 2 {
			g.Serial = "SERIAL-OF-GPU-2"
		}
		gpus = append(gpus, g)
	}
	dump(t, report.Input{
		Envelopes: []*contract.Envelope{env([]contract.TestResult{
			{Name: "soak", Kind: "thermal-soak", Scope: "Node", Phase: "Failed", Nodes: []string{"spark-a"},
				Violations: []contract.Violation{{Metric: "gpuTempC", Cause: "Measurement", Reason: "97 > 90"}}},
			{Name: "ram", Kind: "memory-stress", Scope: "Node", Phase: "Failed", Nodes: []string{"spark-a"}},
		}, nil)},
		Nodes: []report.NodeInfo{{Name: "spark-a", GPUs: gpus}},
		Meta:  report.Meta{Generator: "glimmer-burnin", Version: "v0.5.0"},
	})
}

func TestProbe_TargetedNodeWithNoResults(t *testing.T) {
	dump(t, report.Input{
		Envelopes: []*contract.Envelope{env([]contract.TestResult{
			{Name: "soak", Kind: "thermal-soak", Scope: "Node", Phase: "Passed", Nodes: []string{"spark-a"}},
		}, func(e *contract.Envelope) { e.Phase = "Passed" })},
		Nodes: []report.NodeInfo{{Name: "spark-a"}, {Name: "spark-b", GPUs: []report.GPUInfo{{Index: 0}}}},
		Meta:  report.Meta{Generator: "glimmer-burnin", Version: "v0.5.0"},
	})
}

func TestProbe_SkipReason(t *testing.T) {
	dump(t, report.Input{
		Envelopes: []*contract.Envelope{env([]contract.TestResult{
			{Name: "fp4", Kind: "compute-smoke", Scope: "Node", Phase: "Skipped", Nodes: []string{"spark-a"},
				Message: "FP4_GEMM_SKIP: device is CC 12.0, not 12.1"},
		}, func(e *contract.Envelope) { e.Phase = "Passed" })},
		Meta: report.Meta{Generator: "glimmer-burnin", Version: "v0.5.0"},
	})
}

func TestProbe_RunScopedErrorAcrossNodes(t *testing.T) {
	dump(t, report.Input{
		Envelopes: []*contract.Envelope{env([]contract.TestResult{
			{Name: "soak", Kind: "thermal-soak", Scope: "Node", Phase: "Passed", Nodes: []string{"spark-a"}},
			{Name: "soak", Kind: "thermal-soak", Scope: "Node", Phase: "Passed", Nodes: []string{"spark-b"}},
			{Name: "weird", Kind: "custom", Scope: "Cluster", Phase: "Error",
				Message: `scope "Cluster" is not a scope this operator version recognises`},
		}, nil)},
		Meta: report.Meta{Generator: "glimmer-burnin", Version: "v0.5.0"},
	})
}

func TestProbe_FilenameCollision(t *testing.T) {
	mk := func(run, uid string) report.Input {
		return report.Input{
			Envelopes: []*contract.Envelope{env([]contract.TestResult{
				{Name: "soak", Kind: "thermal-soak", Scope: "Node", Phase: "Failed", Nodes: []string{"spark-a"}},
			}, func(e *contract.Envelope) { e.Run.Name = run; e.Run.UID = uid })},
			Meta: report.Meta{Generator: "glimmer-burnin", Version: "v0.5.0"},
		}
	}
	a, _ := nvvs.Renderer{}.Render(mk("run-42", "uid-1"))
	b, _ := nvvs.Renderer{}.Render(mk("run-43", "uid-2"))
	t.Logf("run-42 -> %s ; run-43 -> %s", a[0].Filename, b[0].Filename)
}

func TestProbe_Link(t *testing.T) {
	dump(t, report.Input{
		Envelopes: []*contract.Envelope{env([]contract.TestResult{
			{Name: "ib", Kind: "ib-write-bw", Scope: "Pair", Phase: "Failed", Nodes: []string{"spark-a", "spark-b"},
				Metrics: map[string]string{"ibWriteBwGbps": "94.1"}},
		}, nil)},
		Nodes: []report.NodeInfo{{Name: "spark-a", GPUs: []report.GPUInfo{{Index: 0}}}},
		Meta:  report.Meta{Generator: "glimmer-burnin", Version: "v0.5.0"},
	})
}

func TestProbe_TwoTestsSameName(t *testing.T) {
	dump(t, report.Input{
		Envelopes: []*contract.Envelope{env([]contract.TestResult{
			{Name: "soak", Kind: "thermal-soak", Scope: "Node", Phase: "Passed", Nodes: []string{"spark-a"}},
			{Name: "soak", Kind: "thermal-soak", Scope: "Node", Phase: "Failed", Nodes: []string{"spark-a"},
				Message: "second execution failed"},
		}, nil)},
		Meta: report.Meta{Generator: "glimmer-burnin", Version: "v0.5.0"},
	})
}
