package verdict

import (
	"testing"

	burninv1alpha1 "github.com/baldwinSPC/glimmer-burnin/api/v1alpha1"
)

func th(metric string, op burninv1alpha1.Comparison, val string) burninv1alpha1.Threshold {
	return burninv1alpha1.Threshold{Metric: metric, Comparison: op, Value: val}
}

func TestEvaluate(t *testing.T) {
	cases := []struct {
		name       string
		metrics    map[string]string
		thresholds []burninv1alpha1.Threshold
		wantPass   bool
	}{
		{
			name:     "no thresholds passes",
			metrics:  map[string]string{"busBandwidthGBs": "23.4"},
			wantPass: true,
		},
		{
			name:       "nccl bus bandwidth meets floor",
			metrics:    map[string]string{"busBandwidthGBs": "23.4"},
			thresholds: []burninv1alpha1.Threshold{th("busBandwidthGBs", burninv1alpha1.GTE, "20")},
			wantPass:   true,
		},
		{
			name:       "nccl bus bandwidth below floor fails",
			metrics:    map[string]string{"busBandwidthGBs": "12.0"},
			thresholds: []burninv1alpha1.Threshold{th("busBandwidthGBs", burninv1alpha1.GTE, "20")},
			wantPass:   false,
		},
		{
			name:       "ecc errors must be zero",
			metrics:    map[string]string{"eccErrors": "0"},
			thresholds: []burninv1alpha1.Threshold{th("eccErrors", burninv1alpha1.EQ, "0")},
			wantPass:   true,
		},
		{
			name:       "ecc errors nonzero fails",
			metrics:    map[string]string{"eccErrors": "3"},
			thresholds: []burninv1alpha1.Threshold{th("eccErrors", burninv1alpha1.EQ, "0")},
			wantPass:   false,
		},
		{
			name:       "missing metric fails closed (must not silently pass acceptance)",
			metrics:    map[string]string{},
			thresholds: []burninv1alpha1.Threshold{th("busBandwidthGBs", burninv1alpha1.GTE, "20")},
			wantPass:   false,
		},
		{
			name:       "non-numeric metric fails",
			metrics:    map[string]string{"busBandwidthGBs": "fast"},
			thresholds: []burninv1alpha1.Threshold{th("busBandwidthGBs", burninv1alpha1.GTE, "20")},
			wantPass:   false,
		},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			pass, msg := Evaluate(c.metrics, c.thresholds)
			if pass != c.wantPass {
				t.Fatalf("Evaluate = %v (%q), want %v", pass, msg, c.wantPass)
			}
			if !pass && msg == "" {
				t.Error("a failing verdict must carry an explanatory message")
			}
		})
	}
}
