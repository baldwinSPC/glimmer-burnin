// Package stats is the arithmetic behind baselines and comparison (#536):
// location and spread, the type-7 quantile, a percentile bootstrap, the
// normal-theory prediction interval with a stdlib-only Student-t quantile, and
// a split-half steady-state test.
//
// It has no dependency outside the standard library, so the same numbers come
// out of the operator, the bare-metal CLI and any consumer that imports it.
// The methods are ported from an independent GB10 acceptance toolkit, which
// cites its sources: Efron (the bootstrap), Hyndman & Fan (type-7 quantiles),
// Kalibera & Jones (replicates as the unit of run-to-run variation).
package stats

import (
	"errors"
	"math"
	"math/rand/v2"
	"sort"
)

// ErrNonFinite is returned for a NaN or infinite sample. A non-finite
// measurement is not data, and letting one through silently poisons every
// statistic downstream (NaN compares false against everything).
var ErrNonFinite = errors.New("non-finite sample")

func checkFinite(x []float64) error {
	for _, v := range x {
		if math.IsNaN(v) || math.IsInf(v, 0) {
			return ErrNonFinite
		}
	}
	return nil
}

// Mean is the arithmetic mean. It returns NaN for an empty slice.
func Mean(x []float64) float64 {
	if len(x) == 0 {
		return math.NaN()
	}
	var s float64
	for _, v := range x {
		s += v
	}
	return s / float64(len(x))
}

// SampleSD is the sample standard deviation (n-1). ok is false below two
// samples: one run says nothing about run-to-run variation.
func SampleSD(x []float64) (sd float64, ok bool) {
	if len(x) < 2 {
		return 0, false
	}
	m := Mean(x)
	var ss float64
	for _, v := range x {
		ss += (v - m) * (v - m)
	}
	return math.Sqrt(ss / float64(len(x)-1)), true
}

// CV is SD / |mean|. ok is false below two samples or at a zero mean, where a
// relative spread is undefined rather than infinite.
func CV(x []float64) (cv float64, ok bool) {
	sd, ok := SampleSD(x)
	m := Mean(x)
	if !ok || m == 0 {
		return 0, false
	}
	return sd / math.Abs(m), true
}

// Quantile7 is the Hyndman & Fan type-7 quantile of a SORTED slice:
// h = p·(n-1), linear interpolation between the neighbouring order statistics.
func Quantile7(sorted []float64, p float64) float64 {
	n := len(sorted)
	if n == 0 {
		return math.NaN()
	}
	h := p * float64(n-1)
	lo := int(math.Floor(h))
	if lo >= n-1 {
		return sorted[n-1]
	}
	return sorted[lo] + (h-float64(lo))*(sorted[lo+1]-sorted[lo])
}

// Median is Quantile7 at 0.5.
func Median(x []float64) float64 {
	s := append([]float64(nil), x...)
	sort.Float64s(s)
	return Quantile7(s, 0.5)
}

// TrimmedMean drops floor(trim·n) samples from each end, then averages.
func TrimmedMean(x []float64, trim float64) float64 {
	s := append([]float64(nil), x...)
	sort.Float64s(s)
	k := int(math.Floor(trim * float64(len(s))))
	if 2*k >= len(s) {
		return Median(s)
	}
	return Mean(s[k : len(s)-k])
}

// BootstrapDiffCI is a percentile bootstrap interval for mean(x) - mean(y):
// each resample draws x and then y independently, with replacement, from one
// seeded stream, so the interval is reproducible from the two vectors and the
// seed alone. conclusive is whether the interval excludes zero.
func BootstrapDiffCI(x, y []float64, resamples int, level float64, seed uint64) (lo, hi float64, conclusive bool, err error) {
	if len(x) == 0 || len(y) == 0 {
		return 0, 0, false, errors.New("bootstrap needs samples on both sides")
	}
	if err := checkFinite(x); err != nil {
		return 0, 0, false, err
	}
	if err := checkFinite(y); err != nil {
		return 0, 0, false, err
	}
	r := rand.New(rand.NewPCG(seed, 0x9e3779b97f4a7c15))
	stats := make([]float64, resamples)
	draw := func(v []float64) float64 {
		var s float64
		for range v {
			s += v[r.IntN(len(v))]
		}
		return s / float64(len(v))
	}
	for i := range stats {
		a := draw(x)
		b := draw(y)
		stats[i] = a - b
	}
	sort.Float64s(stats)
	lo = Quantile7(stats, (1-level)/2)
	hi = Quantile7(stats, (1+level)/2)
	return lo, hi, lo > 0 || hi < 0, nil
}

// PredictionInterval is the normal-theory interval for ONE new observation
// from the population a replicate baseline was drawn from:
// mean ± t(1-α/2, n-1) · sd · sqrt(1 + 1/n). It needs n ≥ 2.
func PredictionInterval(mean, sd float64, n int, alpha float64) (lo, hi float64, err error) {
	if n < 2 {
		return 0, 0, errors.New("a prediction interval needs at least two replicates")
	}
	t, err := StudentTQuantile(1-alpha/2, float64(n-1))
	if err != nil {
		return 0, 0, err
	}
	w := t * sd * math.Sqrt(1+1/float64(n))
	return mean - w, mean + w, nil
}

// StudentTQuantile inverts the Student-t CDF by bisection. The CDF comes from
// the regularised incomplete beta function, evaluated by Lentz's continued
// fraction — no dependency beyond the standard library.
func StudentTQuantile(p, df float64) (float64, error) {
	if !(p > 0 && p < 1) || !(df > 0) {
		return 0, errors.New("StudentTQuantile: p must be in (0,1) and df > 0")
	}
	lo, hi := -1e4, 1e4
	for i := 0; i < 200; i++ {
		mid := (lo + hi) / 2
		if studentTCDF(mid, df) < p {
			lo = mid
		} else {
			hi = mid
		}
		if hi-lo < 1e-13 {
			break
		}
	}
	return (lo + hi) / 2, nil
}

func studentTCDF(t, df float64) float64 {
	x := df / (df + t*t)
	tail := 0.5 * regIncBeta(df/2, 0.5, x)
	if t >= 0 {
		return 1 - tail
	}
	return tail
}

func regIncBeta(a, b, x float64) float64 {
	if x <= 0 {
		return 0
	}
	if x >= 1 {
		return 1
	}
	lga, _ := math.Lgamma(a)
	lgb, _ := math.Lgamma(b)
	lgab, _ := math.Lgamma(a + b)
	front := math.Exp(lgab - lga - lgb + a*math.Log(x) + b*math.Log(1-x))
	if x < (a+1)/(a+b+2) {
		return front * betaCF(a, b, x) / a
	}
	return 1 - front*betaCF(b, a, 1-x)/b
}

func betaCF(a, b, x float64) float64 {
	const eps, tiny = 1e-14, 1e-300
	c, d := 1.0, 1-(a+b)*x/(a+1)
	if math.Abs(d) < tiny {
		d = tiny
	}
	d = 1 / d
	h := d
	for m := 1; m <= 300; m++ {
		fm := float64(m)
		num := fm * (b - fm) * x / ((a + 2*fm - 1) * (a + 2*fm))
		d = 1 + num*d
		if math.Abs(d) < tiny {
			d = tiny
		}
		c = 1 + num/c
		if math.Abs(c) < tiny {
			c = tiny
		}
		d = 1 / d
		h *= d * c
		num = -(a + fm) * (a + b + fm) * x / ((a + 2*fm) * (a + 2*fm + 1))
		d = 1 + num*d
		if math.Abs(d) < tiny {
			d = tiny
		}
		c = 1 + num/c
		if math.Abs(c) < tiny {
			c = tiny
		}
		d = 1 / d
		del := d * c
		h *= del
		if math.Abs(del-1) < eps {
			break
		}
	}
	return h
}

// SplitHalfRelDelta is the steady-state test: the relative difference between
// the mean of the first half of a series and the mean of the second (the
// middle sample of an odd count in neither). ok is false below four samples.
// A value above a few percent says the measurement was still moving.
func SplitHalfRelDelta(x []float64) (relDelta float64, ok bool) {
	k := len(x)
	if k < 4 {
		return 0, false
	}
	m1 := Mean(x[:k/2])
	m2 := Mean(x[k-k/2:])
	den := math.Max(math.Max(math.Abs(m1), math.Abs(m2)), 1e-9)
	return math.Abs(m2-m1) / den, true
}
