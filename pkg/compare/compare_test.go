package compare

import (
	"strconv"
	"strings"
	"testing"
	"time"

	"github.com/baldwinSPC/glimmer-burnin/pkg/contract"
)

const gb10 = "kernel=6.11.0-1014-nvidia os=Ubuntu 24.04.3 LTS arch=arm64 accelerator=nvidia:0x2e12"

func run(node, fp, uid, phase string, metrics map[string]float64) *contract.Envelope {
	m := map[string]string{"gpuName": "NVIDIA GB10"}
	for k, v := range metrics {
		m[k] = strconv.FormatFloat(v, 'f', -1, 64)
	}
	return &contract.Envelope{
		Reason: contract.ReasonPhaseChanged, Phase: "Passed",
		Run:         contract.RunRef{Namespace: "local", Name: node, UID: uid},
		Fingerprint: map[string]string{node: fp},
		Results:     []contract.TestResult{{Name: "fabric", Kind: "ib-write-bw", Phase: phase, Metrics: m}},
	}
}

func side(t *testing.T, envs ...*contract.Envelope) *Side {
	t.Helper()
	s, err := Collect(envs)
	if err != nil {
		t.Fatal(err)
	}
	return s
}

// Five replicates at 97.50 ± 0.01 — the fabric-soak spread measured on the
// Sparks — promote to an enforceable baseline.
func baseline(t *testing.T) *Baseline {
	t.Helper()
	var envs []*contract.Envelope
	for i, v := range []float64{97.49, 97.50, 97.51, 97.50, 97.50} {
		envs = append(envs, run("spark-043a", gb10, "r"+strconv.Itoa(i), "Passed",
			map[string]float64{"bandwidthGbps": v, "xidEvents": 0}))
	}
	b, err := Promote("gb10-fabric", side(t, envs...), nil, time.Unix(0, 0))
	if err != nil {
		t.Fatal(err)
	}
	return b
}

var opts = Options{DefaultTolerance: 0.02, Alpha: 0.05, Enforce: true}

func outcomeOf(rs []Result, key string) Result {
	for _, r := range rs {
		if r.Key == key {
			return r
		}
	}
	return Result{}
}

func TestPromotionRecordsTheReplicateSpread(t *testing.T) {
	b := baseline(t)
	m := b.Metrics["fabric/bandwidthGbps"]
	if m.N != 5 || !m.Enforceable || m.SD == nil || m.Mean != 97.5 {
		t.Errorf("baseline metric = %+v", m)
	}
	if _, ok := b.Metrics["fabric/gpuName"]; ok {
		t.Error("a label was promoted as a quantity")
	}
}

func TestAHalvedLinkIsARegressionAndANormalRunMatches(t *testing.T) {
	b := baseline(t)
	bad := side(t, run("spark-043a", gb10, "now", "Passed", map[string]float64{"bandwidthGbps": 48.9, "xidEvents": 0}))
	rs, err := AgainstBaseline(bad, b, opts)
	if err != nil {
		t.Fatal(err)
	}
	if r := outcomeOf(rs, "fabric/bandwidthGbps"); r.Outcome != Regression || !r.Enforced {
		t.Errorf("48.9 against 97.5: %+v", r)
	}
	if r := outcomeOf(rs, "fabric/xidEvents"); r.Outcome != NotComparable {
		t.Errorf("a zero-mean counter should be not comparable, got %+v", r)
	}

	ok := side(t, run("spark-043a", gb10, "now", "Passed", map[string]float64{"bandwidthGbps": 97.49}))
	rs, _ = AgainstBaseline(ok, b, opts)
	if r := outcomeOf(rs, "fabric/bandwidthGbps"); r.Outcome != Match {
		t.Errorf("97.49: %+v", r)
	}
}

// Beyond the tolerance but inside the prediction interval is not a
// regression: the baseline's own spread says it could be noise.
func TestBeyondToleranceButInsideTheIntervalIsNotSignificant(t *testing.T) {
	var envs []*contract.Envelope
	for i, v := range []float64{90, 100, 110, 95, 105} {
		envs = append(envs, run("n", gb10, "r"+strconv.Itoa(i), "Passed", map[string]float64{"bandwidthGbps": v}))
	}
	b, _ := Promote("wide", side(t, envs...), nil, time.Unix(0, 0))
	rs, _ := AgainstBaseline(side(t, run("n", gb10, "now", "Passed", map[string]float64{"bandwidthGbps": 92})), b, opts)
	if r := outcomeOf(rs, "fabric/bandwidthGbps"); r.Outcome != NoSignificantDifference {
		t.Errorf("92 against 100±7.9: %+v", r)
	}
}

func TestSetupProblemsAreNeverVerdictsAboutThePart(t *testing.T) {
	b := baseline(t)
	newKernel := "kernel=6.14.0 os=Ubuntu 24.04.3 LTS arch=arm64 accelerator=nvidia:0x2e12"
	rs, _ := AgainstBaseline(side(t, run("spark-043a", newKernel, "now", "Passed", map[string]float64{"bandwidthGbps": 40})), b, opts)
	if r := outcomeOf(rs, "fabric/bandwidthGbps"); r.Outcome != NotComparable || r.Cause != "fingerprint differs: kernel" {
		t.Errorf("kernel change: %+v", r)
	}
	rs, _ = AgainstBaseline(side(t, run("spark-043a", gb10, "now", "Error", map[string]float64{"bandwidthGbps": 1})), b, opts)
	if r := outcomeOf(rs, "fabric/bandwidthGbps"); r.Outcome != UnusableInput {
		t.Errorf("errored run: %+v", r)
	}
	b1, _ := Promote("one", side(t, run("spark-043a", gb10, "a", "Passed", map[string]float64{"bandwidthGbps": 97.5})), nil, time.Unix(0, 0))
	rs, _ = AgainstBaseline(side(t, run("spark-043a", gb10, "now", "Passed", map[string]float64{"bandwidthGbps": 40})), b1, opts)
	if r := outcomeOf(rs, "fabric/bandwidthGbps"); r.Outcome != InsufficientPrecision {
		t.Errorf("one-replicate baseline: %+v", r)
	}
}

func TestPromotionRefusesMixedConfigurations(t *testing.T) {
	s := side(t,
		run("n", gb10, "a", "Passed", map[string]float64{"bandwidthGbps": 97.5}),
		run("n", "kernel=6.14.0 os=Ubuntu 24.04.3 LTS arch=arm64 accelerator=nvidia:0x2e12", "b", "Passed", map[string]float64{"bandwidthGbps": 97.4}))
	if _, err := Promote("mixed", s, nil, time.Unix(0, 0)); err == nil {
		t.Error("promoted runs on two kernels as one baseline")
	}
}

func TestSymmetryBetweenTwoNodes(t *testing.T) {
	var a, b []*contract.Envelope
	for i := 0; i < 5; i++ {
		v := 97.5 + float64(i%2)*0.01
		a = append(a, run("spark-043a", gb10, "a"+strconv.Itoa(i), "Passed", map[string]float64{"bandwidthGbps": v}))
		b = append(b, run("spark-85a9", gb10, "b"+strconv.Itoa(i), "Passed", map[string]float64{"bandwidthGbps": v - 12}))
	}
	rs := Symmetry(side(t, b...), side(t, a...), Options{DefaultTolerance: 0.02, Alpha: 0.05})
	if r := outcomeOf(rs, "fabric/bandwidthGbps"); r.Outcome != Regression || r.Method != "bootstrap_diff" {
		t.Errorf("a node 12 Gb/s slower: %+v", r)
	}
	rs = Symmetry(side(t, a...), side(t, a...), Options{DefaultTolerance: 0.02, Alpha: 0.05})
	if r := outcomeOf(rs, "fabric/bandwidthGbps"); r.Outcome != Match {
		t.Errorf("identical nodes: %+v", r)
	}
}

func TestFingerprintFieldsKeepSpacesInValues(t *testing.T) {
	f := FingerprintFields(gb10)
	if f["os"] != "Ubuntu 24.04.3 LTS" || f["kernel"] != "6.11.0-1014-nvidia" || f["accelerator"] != "nvidia:0x2e12" {
		t.Errorf("fields = %v", f)
	}
	op := FingerprintFields("kernel=6.11 nvidia.com/gpu.product=NVIDIA GB10")
	if op["nvidia.com/gpu.product"] != "NVIDIA GB10" {
		t.Errorf("operator-style key: %v", op)
	}
}

func withIdentity(e *contract.Envelope, digests string) *contract.Envelope {
	e.Results = append(e.Results, contract.TestResult{Name: "identity", Kind: "fingerprint-probe", Phase: "Passed",
		Metrics: map[string]string{"nvmeSerialDigests": digests, "nvmeCount": "1"}})
	return e
}

// #540: the same drive reported by two nodes is one machine seen twice, or a
// cloned image, and is said before anything else.
func TestTwoNodesSharingADriveAreFlagged(t *testing.T) {
	a := side(t, withIdentity(run("spark-043a", gb10, "a", "Passed", map[string]float64{"bandwidthGbps": 97.5}), "aaaa1111bbbb2222"))
	b := side(t, withIdentity(run("spark-85a9", gb10, "b", "Passed", map[string]float64{"bandwidthGbps": 97.5}), "cccc3333dddd4444,aaaa1111bbbb2222"))
	r := outcomeOf(Symmetry(a, b, opts), "identity")
	if r.Outcome != NotComparable || !strings.Contains(r.Cause, "aaaa1111bbbb2222") {
		t.Errorf("shared drive: %+v", r)
	}
	c := side(t, withIdentity(run("spark-85a9", gb10, "c", "Passed", map[string]float64{"bandwidthGbps": 97.5}), "cccc3333dddd4444"))
	if r := outcomeOf(Symmetry(a, c, opts), "identity"); r.Outcome != "" {
		t.Errorf("distinct drives flagged: %+v", r)
	}
	if _, ok := a.Metrics["identity/nvmeSerialDigests"]; ok {
		t.Error("an identity label was collected as a quantity")
	}
}
