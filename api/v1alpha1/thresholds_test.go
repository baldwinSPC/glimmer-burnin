package v1alpha1_test

import (
	"math"
	"os"
	"path/filepath"
	"regexp"
	"strconv"
	"strings"
	"testing"

	burninv1alpha1 "github.com/baldwinSPC/glimmer-burnin/api/v1alpha1"
	"github.com/baldwinSPC/glimmer-burnin/pkg/contract"
	"github.com/baldwinSPC/glimmer-burnin/pkg/verdict"
)

// typesFile is this package's own source, read back so the checks below are
// about what is DECLARED rather than about a second list someone maintained by
// hand. It is the same idiom runners/pins_test.go uses over the Dockerfiles: the
// declaration is the only thing that knows the whole set.
const typesFile = "burnintest_types.go"

func readTypesFile(t *testing.T) string {
	t.Helper()
	raw, err := os.ReadFile(typesFile)
	if err != nil {
		t.Fatalf("reading %s: %v", typesFile, err)
	}
	return string(raw)
}

var kindDecl = regexp.MustCompile(`(?m)^\s*(Kind[A-Za-z0-9]+)\s+TestKind\s*=\s*"([^"]+)"`)

// BuiltInKinds is hand-kept, and a hand-kept list beside an enum rots. The
// consequence of rot here is quiet: a new kind missing from the list is treated
// as somebody else's runner, so the one check that exists to catch a first-party
// gate on a metric no first-party runner emits simply stops applying to it, with
// nothing failing.
func TestBuiltInKindsCoversEveryDeclaredKind(t *testing.T) {
	declared := kindDecl.FindAllStringSubmatch(readTypesFile(t), -1)
	if len(declared) < 2 {
		t.Fatalf("found %d TestKind declarations in %s — the pattern no longer matches the source", len(declared), typesFile)
	}

	listed := map[burninv1alpha1.TestKind]bool{}
	for _, k := range burninv1alpha1.BuiltInKinds {
		if listed[k] {
			t.Errorf("BuiltInKinds lists %q twice", k)
		}
		listed[k] = true
	}

	for _, m := range declared {
		constName, kind := m[1], burninv1alpha1.TestKind(m[2])
		if kind == burninv1alpha1.KindCustom {
			// The escape hatch, and the one kind that must NOT be listed: it
			// exists to say "this runner is not ours".
			if kind.IsBuiltIn() {
				t.Errorf("%s (%q) is listed as built-in; it is the escape hatch for runners this project does not ship", constName, kind)
			}
			continue
		}
		if !kind.IsBuiltIn() {
			t.Errorf("%s (%q) is declared but missing from BuiltInKinds — add it there, "+
				"or the threshold linter will treat this project's own runner as a third party's", constName, kind)
		}
		delete(listed, kind)
	}
	for kind := range listed {
		t.Errorf("BuiltInKinds lists %q, which no TestKind constant declares", kind)
	}
}

// ─── The CRD's half of the threshold grammar ──────────────────────────────────
//
// Threshold.Metric and Threshold.Value carry Pattern markers, so the apiserver
// refuses the malformed forms at the moment of authoring — the earliest and
// cheapest place any of this can be said. The operator re-checks the same rules
// at plan time (verdict.ValidateThresholds, surfaced by buildPlan) because a CRD
// pattern cannot reach the registry and because objects predating the pattern
// still exist.
//
// Two rules enforced by one regex and one Go function is a disagreement waiting
// to happen, and the disagreement is not symmetric: a pattern STRICTER than the
// linter rejects thresholds the operator would have run, which is a rejection
// nothing explains. These tests hold the two together.

var patternMarker = regexp.MustCompile("(?m)^\\s*// \\+kubebuilder:validation:Pattern=`([^`]+)`")

// thresholdPatterns returns the Metric and Value patterns, in declaration order.
func thresholdPatterns(t *testing.T) []*regexp.Regexp {
	t.Helper()
	src := readTypesFile(t)
	// Only the Threshold struct's markers: scoping by the struct keeps this from
	// silently picking up a pattern added to some other field later.
	start := strings.Index(src, "type Threshold struct {")
	if start < 0 {
		t.Fatal("Threshold struct not found — this test no longer reads what it thinks it reads")
	}
	end := strings.Index(src[start:], "\n}\n")
	if end < 0 {
		t.Fatal("Threshold struct has no end")
	}

	found := patternMarker.FindAllStringSubmatch(src[start:start+end], -1)
	if len(found) != 2 {
		t.Fatalf("found %d Pattern markers on Threshold, want 2 (metric and value)", len(found))
	}
	out := make([]*regexp.Regexp, 0, 2)
	for _, m := range found {
		re, err := regexp.Compile(m[1])
		if err != nil {
			t.Fatalf("Pattern %q does not compile, so the apiserver would reject the CRD: %v", m[1], err)
		}
		out = append(out, re)
	}
	return out
}

// The generated manifest is what a cluster actually validates against, so the
// marker being present in the Go source proves nothing on its own.
//
// BOTH manifests, because a Threshold reaches a cluster by two routes: a
// BurnInTest, and a BurnInProfile's inline test. A rule enforced on one and not
// the other is a rule with a documented way around it.
func TestThresholdPatternsReachTheGeneratedCRDs(t *testing.T) {
	patterns := thresholdPatterns(t)
	for _, name := range []string{
		"burnin.glimmer.ai_burnintests.yaml",
		"burnin.glimmer.ai_burninprofiles.yaml",
	} {
		t.Run(name, func(t *testing.T) {
			raw, err := os.ReadFile(filepath.Join("..", "..", "config", "crd", name))
			if err != nil {
				t.Fatalf("reading the generated CRD: %v", err)
			}
			for _, re := range patterns {
				if !strings.Contains(string(raw), re.String()) {
					t.Errorf("pattern %q is declared on Threshold but absent from this CRD — run 'make manifests'", re)
				}
			}
		})
	}
}

// The metric pattern must accept exactly what the contract grammar accepts. It
// is deliberately the WEAKER of the two rules — it cannot know the registry, so
// it cannot enforce that a registered name declares its registered unit — but it
// must never reject a name the grammar allows.
func TestMetricPatternAgreesWithTheContractGrammar(t *testing.T) {
	metricPattern := thresholdPatterns(t)[0]

	legal := []string{
		"eccErrors", "busBandwidthGBs", "sustainedClockPct", "latencyUs",
		"pdWedgeSuspected", "vendorLinkUtilPct", "a", "x9", "nvLink4Errors",
	}
	for _, name := range legal {
		if err := contract.ValidateMetricName(name); err != nil {
			t.Fatalf("test corpus is wrong: %q is not actually legal: %v", name, err)
		}
		if !metricPattern.MatchString(name) {
			t.Errorf("the CRD pattern rejects %q, which the contract grammar allows — the apiserver would refuse a threshold the operator would have run", name)
		}
	}

	illegal := []string{
		"", "bus_bandwidth_gbs", "BusBandwidth", "9lives", "bus.bandwidth",
		"bus bandwidth", "bandwidth-gbps", "ecc\nErrors",
	}
	for _, name := range illegal {
		if contract.ValidateMetricName(name) == nil {
			t.Fatalf("test corpus is wrong: %q is actually legal", name)
		}
		if metricPattern.MatchString(name) {
			t.Errorf("the CRD pattern accepts %q, which the contract grammar rejects; such a name is dropped by the runner's parser and its gate can never be satisfied", name)
		}
	}
}

// The value pattern must accept exactly the FINITE numbers, which is the rule
// the linter enforces as Malformed and the field documentation promises. "NaN"
// and "Inf" parse as float64 and are the interesting rejections: NaN compares
// false against everything, which would quietly turn NotEqual into a gate that
// always passes.
func TestValuePatternAcceptsExactlyTheFiniteNumbers(t *testing.T) {
	valuePattern := thresholdPatterns(t)[1]

	corpus := []string{
		"0", "20", "10.8", "-3", "+3", "1e9", "1E9", "2.5e-3", ".5", "5.", "0090",
		"NaN", "nan", "Inf", "+Inf", "-Inf", "infinity",
		"", "twenty", "one hundred", "20%", "20 ", " 20", "0x1p3", "1_000",
	}
	for _, value := range corpus {
		t.Run(strconv.Quote(value), func(t *testing.T) {
			parsed, err := strconv.ParseFloat(value, 64)
			// The rule, stated once: a value is admissible exactly when it is a
			// finite float64. Everything else is a bound that expresses nothing.
			wantAdmissible := err == nil && !math.IsNaN(parsed) && !math.IsInf(parsed, 0)

			// Underscore and hex-float literals are the one deliberate
			// divergence: strconv takes them, nobody writes them in a threshold,
			// and admitting exotic spellings into a fleet-wide gate buys nothing.
			if strings.ContainsAny(value, "_xX") {
				wantAdmissible = false
			}

			if got := valuePattern.MatchString(value); got != wantAdmissible {
				t.Errorf("pattern accepts=%v, want %v", got, wantAdmissible)
			}

			// And the operator's own check must agree, so a value the apiserver
			// admits is never one the run then refuses, and vice versa.
			problems := verdict.ValidateThresholds([]burninv1alpha1.Threshold{{
				Metric: "eccErrors", Comparison: burninv1alpha1.GTE, Value: value,
			}})
			malformed := false
			for _, p := range problems {
				if p.Severity == verdict.SeverityMalformed {
					malformed = true
				}
			}
			if wantAdmissible && malformed {
				t.Errorf("the CRD admits %q but the linter calls it malformed: %v", value, problems)
			}
		})
	}
}
