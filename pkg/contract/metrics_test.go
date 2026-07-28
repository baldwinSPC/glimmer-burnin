package contract

import (
	"strings"
	"testing"
)

func TestUnitOf(t *testing.T) {
	cases := []struct {
		name string
		want Unit
	}{
		// The two that motivated the convention. They must stay distinguishable.
		{"bandwidthGbps", UnitGigabitsPerSecond},
		{"busBandwidthGBs", UnitGigabytesPerSecond},

		{"latencyUs", UnitMicroseconds},
		{"gpuTempC", UnitCelsius},
		{"powerDrawW", UnitWatts},
		{"throughputTflops", UnitTeraflops},
		{"utilisationPct", UnitPercent},
		{"durationS", UnitSeconds},

		// Dimensionless counters must NOT pick up a unit. These are the
		// dangerous ones: a lowercase plural "s" must never be read as the
		// Seconds suffix "S", or every counter silently becomes a duration.
		{"eccErrors", UnitNone},
		{"throttleEvents", UnitNone},
		{"nonfiniteCount", UnitNone},

		// A bare unit is not a name; it declares nothing about a measurand.
		{"tflops", UnitNone},
	}
	for _, c := range cases {
		if got := UnitOf(c.name); got != c.want {
			t.Errorf("UnitOf(%q) = %q, want %q", c.name, got, c.want)
		}
	}
}

// Every registered metric must agree with the unit its own name declares,
// otherwise the registry is lying about data consumers will store and chart.
func TestRegistryIsSelfConsistent(t *testing.T) {
	for _, name := range Registered() {
		m, ok := Lookup(name)
		if !ok {
			t.Fatalf("Registered() returned %q but Lookup could not find it", name)
		}
		if m.Name != name {
			t.Errorf("registry key %q holds metric named %q", name, m.Name)
		}
		if got := UnitOf(name); got != m.Unit {
			t.Errorf("registered %q declares unit %q by name but registry says %q", name, got, m.Unit)
		}
		if m.Description == "" {
			t.Errorf("registered metric %q has no description; two bandwidth figures in different units are only safe side by side if a reader can tell them apart", name)
		}
		if err := ValidateMetricName(name); err != nil {
			t.Errorf("registered metric %q fails its own validation: %v", name, err)
		}
	}
}

func TestValidateMetricName(t *testing.T) {
	valid := []string{
		"bandwidthGbps", "busBandwidthGBs", "eccErrors", "nonfiniteCount",
		// Unregistered but well-formed: a runner must be able to report a new
		// measurement without a release of this package.
		"someNewThingGBs", "widgets",
	}
	for _, n := range valid {
		if err := ValidateMetricName(n); err != nil {
			t.Errorf("ValidateMetricName(%q) = %v, want nil", n, err)
		}
	}

	invalid := []string{
		"",               // empty
		"BandwidthGbps",  // not lowerCamelCase
		"bandwidth_gbps", // snake_case is the runner's wire format, not canonical
		"bandwidth-gbps", // punctuation
		"bandwidth gbps", // space
	}
	for _, n := range invalid {
		if err := ValidateMetricName(n); err == nil {
			t.Errorf("ValidateMetricName(%q) = nil, want an error", n)
		}
	}
}

func TestValidateMetricsReportsEveryOffender(t *testing.T) {
	err := ValidateMetrics(map[string]string{
		"bandwidthGbps": "180",
		"bad_one":       "1",
		"AnotherBadOne": "2",
	})
	if err == nil {
		t.Fatal("ValidateMetrics returned nil for a map containing two malformed names")
	}
	// A runner author fixing names should see all of them at once.
	for _, want := range []string{"bad_one", "AnotherBadOne"} {
		if !strings.Contains(err.Error(), want) {
			t.Errorf("error %q does not mention offending name %q", err, want)
		}
	}
}
