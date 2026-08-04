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

		// The megahertz suffix must not be shadowed by the millisecond one:
		// both begin with "M", and a clock read as a duration is nonsense that
		// still charts.
		{"smClockMHz", UnitMegahertz},
		{"ratedBoostClockMHz", UnitMegahertz},
		{"elapsedMs", UnitMilliseconds},

		// "Us" must win over "S", or every latency becomes a duration in
		// seconds and reads six orders of magnitude too large.
		{"ioLatencyUs", UnitMicroseconds},
		{"elapsedS", UnitSeconds},

		// Megabytes and gigabytes per second are different suffixes on the
		// same shape of name; neither may fall through to Seconds.
		{"readBandwidthMBs", UnitMegabytesPerSecond},
		{"algBandwidthGBs", UnitGigabytesPerSecond},
		{"sustainedClockPct", UnitPercent},

		// Dimensionless counters must NOT pick up a unit. These are the
		// dangerous ones: a lowercase plural "s" must never be read as the
		// Seconds suffix "S", or every counter silently becomes a duration.
		{"eccErrors", UnitNone},
		{"throttleEvents", UnitNone},
		{"nonfiniteCount", UnitNone},
		{"miscompares", UnitNone},
		{"xidEvents", UnitNone},
		{"remappedRows", UnitNone},
		{"pcieReplayErrors", UnitNone},
		{"nicLinkDownEvents", UnitNone},
		{"memoryErrors", UnitNone},
		{"diagTestsFailed", UnitNone},
		{"iterationsCompleted", UnitNone},
		{"sdcDetections", UnitNone},

		// A bare unit is not a name; it declares nothing about a measurand.
		{"tflops", UnitNone},
		// Nor is a lowercase one: "mhz" is not the MHz suffix, which is the
		// whole reason a runner's "sm_clock_mhz" needs an alias rather than
		// generic normalisation.
		{"smClockMhz", UnitNone},
		{"memoryBandwidthGbs", UnitNone},
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
		// A registry entry that never states whether acceptance may be decided
		// on it is the failure this tri-state exists to catch: a bool would
		// have let the omission pass as a decision.
		switch m.ThresholdUse {
		case ThresholdUseAcceptance, ThresholdUseEvidence:
		case ThresholdUseUnspecified:
			t.Errorf("registered metric %q does not declare a ThresholdUse; say whether a profile may gate a fleet on it", name)
		default:
			t.Errorf("registered metric %q declares unknown ThresholdUse %q", name, m.ThresholdUse)
		}
	}
}

// The names below cross a repository boundary: a runner emits them, this
// operator thresholds on them, and an external consumer stores and charts them.
// A rename or a typo'd registry key is not a local refactor, so the expected
// set is written out here rather than derived from the registry it is checking.
func TestRegistryContainsTheNamesConsumersDependOn(t *testing.T) {
	want := map[string]struct {
		unit Unit
		use  ThresholdUse
	}{
		// Dimensional.
		"algBandwidthGBs":            {UnitGigabytesPerSecond, ThresholdUseAcceptance},
		"busBandwidthGBs":            {UnitGigabytesPerSecond, ThresholdUseAcceptance},
		"hostToDeviceBandwidthGBs":   {UnitGigabytesPerSecond, ThresholdUseAcceptance},
		"deviceToHostBandwidthGBs":   {UnitGigabytesPerSecond, ThresholdUseAcceptance},
		"deviceToDeviceBandwidthGBs": {UnitGigabytesPerSecond, ThresholdUseAcceptance},
		"memoryBandwidthGBs":         {UnitGigabytesPerSecond, ThresholdUseAcceptance},
		"bandwidthGbps":              {UnitGigabitsPerSecond, ThresholdUseAcceptance},
		"peakBandwidthGbps":          {UnitGigabitsPerSecond, ThresholdUseAcceptance},
		"readBandwidthMBs":           {UnitMegabytesPerSecond, ThresholdUseAcceptance},
		"writeBandwidthMBs":          {UnitMegabytesPerSecond, ThresholdUseAcceptance},
		"ioLatencyUs":                {UnitMicroseconds, ThresholdUseAcceptance},
		"latencyUs":                  {UnitMicroseconds, ThresholdUseAcceptance},
		"elapsedS":                   {UnitSeconds, ThresholdUseAcceptance},
		"durationRequestedS":         {UnitSeconds, ThresholdUseEvidence},
		"warmupS":                    {UnitSeconds, ThresholdUseEvidence},
		"sampleWindowS":              {UnitSeconds, ThresholdUseEvidence},
		"sustainedClockPct":          {UnitPercent, ThresholdUseAcceptance},
		"minSmClockPct":              {UnitPercent, ThresholdUseAcceptance},
		"maxSmClockPct":              {UnitPercent, ThresholdUseEvidence},
		"clockFloorPct":              {UnitPercent, ThresholdUseEvidence},
		"thermalClockFloorPct":       {UnitPercent, ThresholdUseEvidence},
		"clockFloorAppliedPct":       {UnitPercent, ThresholdUseEvidence},
		"gpuUtilizationPct":          {UnitPercent, ThresholdUseEvidence},
		"memUtilizationPct":          {UnitPercent, ThresholdUseEvidence},
		"powerLimitRatioPct":         {UnitPercent, ThresholdUseAcceptance},
		"throughputConsistencyPct":   {UnitPercent, ThresholdUseAcceptance},
		"smClockMHz":                 {UnitMegahertz, ThresholdUseAcceptance},
		"memClockMHz":                {UnitMegahertz, ThresholdUseAcceptance},
		"ratedBoostClockMHz":         {UnitMegahertz, ThresholdUseEvidence},
		"gpuTempC":                   {UnitCelsius, ThresholdUseAcceptance},
		"meanTempUnderLoadC":         {UnitCelsius, ThresholdUseAcceptance},
		"tempAtMinClockC":            {UnitCelsius, ThresholdUseEvidence},
		"thermalTempThresholdC":      {UnitCelsius, ThresholdUseEvidence},
		"powerDrawW":                 {UnitWatts, ThresholdUseAcceptance},
		"meanPowerW":                 {UnitWatts, ThresholdUseAcceptance},
		"enforcedPowerLimitW":        {UnitWatts, ThresholdUseAcceptance},
		"defaultPowerLimitW":         {UnitWatts, ThresholdUseEvidence},
		"throughputTflops":           {UnitTeraflops, ThresholdUseEvidence},
		"sustainedThroughputTflops":  {UnitTeraflops, ThresholdUseAcceptance},

		"sustainedFmaThroughputTflops": {UnitTeraflops, ThresholdUseAcceptance},
		"peakFmaThroughputTflops":      {UnitTeraflops, ThresholdUseEvidence},

		// Dimensionless counters.
		"miscompares":         {UnitNone, ThresholdUseAcceptance},
		"sdcDetections":       {UnitNone, ThresholdUseAcceptance},
		"nonfiniteCount":      {UnitNone, ThresholdUseAcceptance},
		"eccErrors":           {UnitNone, ThresholdUseAcceptance},
		"memoryErrors":        {UnitNone, ThresholdUseAcceptance},
		"xidEvents":           {UnitNone, ThresholdUseAcceptance},
		"remappedRows":        {UnitNone, ThresholdUseAcceptance},
		"pcieReplayErrors":    {UnitNone, ThresholdUseAcceptance},
		"nicLinkDownEvents":   {UnitNone, ThresholdUseAcceptance},
		"throttleEvents":      {UnitNone, ThresholdUseAcceptance},
		"throttledSamples":    {UnitNone, ThresholdUseAcceptance},
		"throttleReasonsMask": {UnitNone, ThresholdUseEvidence},
		"unsupportedReads":    {UnitNone, ThresholdUseEvidence},
		"diagTestsFailed":     {UnitNone, ThresholdUseAcceptance},
		"iterationsCompleted": {UnitNone, ThresholdUseAcceptance},
		"samplesTaken":        {UnitNone, ThresholdUseEvidence},
		"loadLaunches":        {UnitNone, ThresholdUseEvidence},
		"loadThreads":         {UnitNone, ThresholdUseEvidence},
		"loadItersPerLaunch":  {UnitNone, ThresholdUseEvidence},

		// Label-valued. Every one of these MUST be Evidence: their values are
		// not numbers, so a threshold on one fails closed on every node forever
		// and reads as a hardware verdict. Evidence is what makes an
		// authoring-time linter refuse the gate. Flipping any of them to
		// Acceptance would re-open exactly that hole, which is why they are
		// asserted here by name rather than left to review.
		"gpuName":                {UnitNone, ThresholdUseEvidence},
		"computeCap":             {UnitNone, ThresholdUseEvidence},
		"pciBusId":               {UnitNone, ThresholdUseEvidence},
		"driverVersion":          {UnitNone, ThresholdUseEvidence},
		"builtCudaArch":          {UnitNone, ThresholdUseEvidence},
		"migMode":                {UnitNone, ThresholdUseEvidence},
		"nvmlUnsupported":        {UnitNone, ThresholdUseEvidence},
		"configWarnings":         {UnitNone, ThresholdUseEvidence},
		"throttleReasons":        {UnitNone, ThresholdUseEvidence},
		"throttleClassification": {UnitNone, ThresholdUseEvidence},
		"pdWedgeSuspected":       {UnitNone, ThresholdUseEvidence},
		"clockFloorBasis":        {UnitNone, ThresholdUseEvidence},
	}

	for name, w := range want {
		m, ok := Lookup(name)
		if !ok {
			t.Errorf("metric %q is not registered; a consumer expecting it would store nothing", name)
			continue
		}
		if m.Unit != w.unit {
			t.Errorf("metric %q has unit %q, want %q", name, m.Unit, w.unit)
		}
		if m.ThresholdUse != w.use {
			t.Errorf("metric %q has ThresholdUse %q, want %q", name, m.ThresholdUse, w.use)
		}
	}

	// Catches the reverse drift: a name added to the registry but never
	// declared here has not been reviewed as a cross-boundary commitment.
	for _, name := range Registered() {
		if _, ok := want[name]; !ok {
			t.Errorf("registry contains %q but this test does not; every canonical name is a promise to a consumer, so add it here deliberately", name)
		}
	}
}

func TestSafeToThresholdOn(t *testing.T) {
	// A liveness signal. Gating on a single unwarmed launch produces a verdict
	// that looks like a hardware judgement and is not one.
	if SafeToThresholdOn("throughputTflops") {
		t.Error("throughputTflops is documented as unsafe to threshold on but SafeToThresholdOn said yes")
	}
	// The sustained figure is a real benchmark. If these two shared a name,
	// one of them would be wrong, which is why they do not.
	if !SafeToThresholdOn("sustainedThroughputTflops") {
		t.Error("sustainedThroughputTflops is a benchmark figure and must remain thresholdable")
	}
	// A nameplate constant identifies the SKU, not its health.
	if SafeToThresholdOn("ratedBoostClockMHz") {
		t.Error("ratedBoostClockMHz is a nameplate constant; gating on it fails a heterogeneous fleet for no hardware reason")
	}
	if !SafeToThresholdOn("busBandwidthGBs") {
		t.Error("busBandwidthGBs is the metric NCCL acceptance exists to gate on")
	}
	// The registry is an open world: a third-party runner's own measurement is
	// its author's to threshold on, not ours to forbid.
	if !SafeToThresholdOn("someVendorSpecificGBs") {
		t.Error("an unregistered metric was refused; that would forbid every threshold on every third-party runner")
	}
}

// A registered metric that was never classified must not be reported as safe:
// the registry having no opinion is not the same as the metric being sound to
// gate a fleet on. The registry test forbids this state, so it is constructed
// directly here.
func TestSafeToThresholdOn_UnclassifiedRegisteredMetricFailsClosed(t *testing.T) {
	m := Metric{Name: "someUnclassifiedGBs", Unit: UnitGigabytesPerSecond}
	if m.SafeToThresholdOn() {
		t.Error("an unclassified registry entry reported itself safe to threshold on")
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
