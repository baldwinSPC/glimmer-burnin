package stats

import (
	"math"
	"testing"
)

func near(a, b, tol float64) bool { return math.Abs(a-b) <= tol }

// Published values, the vectors the source toolkit pins.
func TestStudentTQuantileMatchesTables(t *testing.T) {
	for _, c := range []struct{ p, df, want float64 }{
		{0.975, 10, 2.2281}, {0.975, 2, 4.3027}, {0.975, 1, 12.7062}, {0.95, 30, 1.6973},
		{0.975, 1e6, 1.9600},
	} {
		got, err := StudentTQuantile(c.p, c.df)
		if err != nil || !near(got, c.want, 5e-4) {
			t.Errorf("t(%v, %v) = %v, %v; want %v", c.p, c.df, got, err, c.want)
		}
	}
}

func TestLocationAndSpread(t *testing.T) {
	x := []float64{10, 12, 11.5, 9.5, 13}
	if m := Mean(x); !near(m, 11.2, 1e-12) {
		t.Errorf("mean %v", m)
	}
	sd, ok := SampleSD(x)
	if !ok || !near(sd, 1.4405, 1e-4) {
		t.Errorf("sd %v %v", sd, ok)
	}
	if _, ok := SampleSD([]float64{5}); ok {
		t.Error("one sample produced a spread")
	}
	if _, ok := CV([]float64{-1, 1}); ok {
		t.Error("a zero mean produced a CV")
	}
	if q := Quantile7([]float64{1, 2, 3, 4}, 0.25); !near(q, 1.75, 1e-12) {
		t.Errorf("type-7 q25 = %v, want 1.75", q)
	}
	if m := TrimmedMean([]float64{1, 2, 3, 4, 100}, 0.2); !near(m, 3, 1e-12) {
		t.Errorf("trimmed mean %v", m)
	}
}

func TestPredictionInterval(t *testing.T) {
	lo, hi, err := PredictionInterval(100, 2, 11, 0.05)
	w := 2.2281 * 2 * math.Sqrt(1+1.0/11)
	if err != nil || !near(lo, 100-w, 1e-3) || !near(hi, 100+w, 1e-3) {
		t.Errorf("PI = [%v, %v] %v", lo, hi, err)
	}
	if _, _, err := PredictionInterval(100, 2, 1, 0.05); err == nil {
		t.Error("one replicate produced an interval")
	}
}

func TestBootstrapIsReproducibleAndDetectsAShift(t *testing.T) {
	a := []float64{97.5, 97.6, 97.4, 97.5, 97.55}
	b := []float64{90.1, 90.3, 90.0, 90.2, 90.15}
	lo1, hi1, c1, err := BootstrapDiffCI(a, b, 2000, 0.95, 7)
	lo2, hi2, _, _ := BootstrapDiffCI(a, b, 2000, 0.95, 7)
	if err != nil || lo1 != lo2 || hi1 != hi2 {
		t.Fatalf("not reproducible: [%v,%v] vs [%v,%v] %v", lo1, hi1, lo2, hi2, err)
	}
	if !c1 || lo1 < 7 || hi1 > 7.7 {
		t.Errorf("a 7.3-unit shift gave [%v, %v] conclusive=%v", lo1, hi1, c1)
	}
	_, _, c, _ := BootstrapDiffCI(a, a, 2000, 0.95, 7)
	if c {
		t.Error("identical samples were called different")
	}
	if _, _, _, err := BootstrapDiffCI([]float64{math.NaN()}, a, 10, 0.95, 1); err == nil {
		t.Error("a NaN sample was accepted")
	}
}

func TestSplitHalf(t *testing.T) {
	if d, ok := SplitHalfRelDelta([]float64{80, 80, 80, 80, 80}); !ok || d != 0 {
		t.Errorf("flat series: %v %v", d, ok)
	}
	if d, _ := SplitHalfRelDelta([]float64{100, 100, 90, 80}); !near(d, 0.15, 1e-12) {
		t.Errorf("declining series: %v, want 0.15", d)
	}
	if _, ok := SplitHalfRelDelta([]float64{1, 2, 3}); ok {
		t.Error("three samples judged")
	}
}
