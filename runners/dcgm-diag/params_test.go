package main

import (
	"bytes"
	"io"
	"os"
	"strings"
	"testing"
	"time"

	"github.com/baldwinSPC/glimmer-burnin/pkg/runner"
)

// A level run is unchanged by the arrival of named tests: -r takes the number,
// diag_level is reported, and diag_tests is not.
func TestResolveRun_LevelIsUnchanged(t *testing.T) {
	spec, err := resolveRun("3", "", 0)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if spec.arg != "3" || spec.level != 3 {
		t.Errorf("spec = %+v, want -r 3 at level 3", spec)
	}
	if spec.tests != "" {
		t.Errorf("tests = %q, want empty for a level run", spec.tests)
	}
	if spec.source != "explicit" {
		t.Errorf("source = %q, want explicit", spec.source)
	}
}

func TestResolveRun_NamedTests(t *testing.T) {
	spec, err := resolveRun("", " memory , pcie ", 0)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if spec.arg != "memory,pcie" {
		t.Errorf("-r argument = %q, want memory,pcie", spec.arg)
	}
	if spec.tests != "memory,pcie" {
		t.Errorf("tests = %q, want memory,pcie", spec.tests)
	}
	// Zero, so run() omits diag_level entirely. A named-test run executed no
	// level, and reporting one a profile could threshold on would describe a
	// suite that was never asked for.
	if spec.level != 0 {
		t.Errorf("level = %d, want 0 for a named-test run", spec.level)
	}
	if spec.source != "named" {
		t.Errorf("source = %q, want named", spec.source)
	}
}

// The two variables ask for different scopes, and silently preferring one would
// certify a node against a suite it never ran — the same substitution
// resolveLevel refuses to make for a level that does not fit its duration.
func TestResolveRun_LevelAndTestsTogetherIsAnError(t *testing.T) {
	_, err := resolveRun("2", "memory", 0)
	if err == nil {
		t.Fatal("setting both BURNIN_DCGM_LEVEL and BURNIN_DCGM_TESTS was accepted")
	}
	for _, want := range []string{"BURNIN_DCGM_LEVEL", "BURNIN_DCGM_TESTS"} {
		if !strings.Contains(err.Error(), want) {
			t.Errorf("error %q does not name %s", err, want)
		}
	}
}

// "2" in BURNIN_DCGM_TESTS is a level, and dcgmi would honour it as one — but
// the runner would then report no diag_level and a diag_tests of "2", so the
// result would misdescribe a level-2 run.
//
// The identifier grammar rejects it either way, so what this pins is the
// MESSAGE: putting a level in the wrong variable is the likeliest mistake here,
// and "not a DCGM test name" does not tell an operator where to put it.
func TestResolveRun_NumericTestNameSaysWhereALevelGoes(t *testing.T) {
	for _, in := range []string{"2", "memory,3"} {
		_, err := resolveRun("", in, 0)
		if err == nil {
			t.Errorf("BURNIN_DCGM_TESTS=%q was accepted; it names a run level", in)
			continue
		}
		if !strings.Contains(err.Error(), "BURNIN_DCGM_LEVEL") {
			t.Errorf("BURNIN_DCGM_TESTS=%q: error %q does not say to use BURNIN_DCGM_LEVEL", in, err)
		}
	}
}

func TestResolveRun_MalformedTestLists(t *testing.T) {
	for _, in := range []string{"memory,", ",pcie", "memory,,pcie", "memory;pcie", "memory=1"} {
		if _, err := resolveRun("", in, 0); err == nil {
			t.Errorf("BURNIN_DCGM_TESTS=%q was accepted", in)
		}
	}
}

// The whole point of #307: a profile can enable a plugin DCGM's per-SKU
// allowlist gates off, without the runner holding any knowledge of which
// plugins exist or which parts are on the list.
func TestResolveParams_AllowExpandsUniformly(t *testing.T) {
	got, err := resolveParams("memory,pcie,targeted_stress", "")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	want := "memory.is_allowed=true;pcie.is_allowed=true;targeted_stress.is_allowed=true"
	if got != want {
		t.Errorf("params = %q, want %q", got, want)
	}
}

func TestResolveParams_RawPassesThrough(t *testing.T) {
	// The second clause is subtest-scoped, which is why the grammar accepts
	// more than one dot: requiring exactly one rejected DCGM's own vocabulary.
	raw := "memory.is_allowed=true;pcie.h2d_d2h_single_pinned.min_bandwidth=8000"
	got, err := resolveParams("", raw)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if got != raw {
		t.Errorf("params = %q, want it passed through as %q", got, raw)
	}
}

func TestResolveParams_AllowAndRawCombine(t *testing.T) {
	got, err := resolveParams("memory", "diagnostic.test_duration=60")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	want := "memory.is_allowed=true;diagnostic.test_duration=60"
	if got != want {
		t.Errorf("params = %q, want %q", got, want)
	}
}

// DCGM documents no precedence for a repeated key, so a runner that picked one
// would be guessing which of two contradictory instructions the operator meant.
func TestResolveParams_ContradictionIsRejected(t *testing.T) {
	_, err := resolveParams("memory", "memory.is_allowed=false")
	if err == nil {
		t.Fatal("memory.is_allowed set by both ALLOW and PARAMS was accepted")
	}
	if !strings.Contains(err.Error(), "memory.is_allowed") {
		t.Errorf("error %q does not name the contradicting parameter", err)
	}
}

func TestResolveParams_MalformedIsRejected(t *testing.T) {
	cases := []struct{ allow, raw string }{
		{"", "memory"},                  // no variable, no value
		{"", "memory.is_allowed"},       // no value
		{"", "is_allowed=true"},         // no test name
		{"", "memory.is_allowed="},      // empty value
		{"", ".is_allowed=true"},        // empty test name
		{"", "memory.=true"},            // empty variable
		{"", "9memory.is_allowed=true"}, // not an identifier
		{"memory pcie", ""},             // a comma was meant, not a space
		{"memory;pcie", ""},
		{"memory.is_allowed=true", ""}, // a raw clause in the curated variable
	}
	for _, tc := range cases {
		if _, err := resolveParams(tc.allow, tc.raw); err == nil {
			t.Errorf("resolveParams(%q, %q) was accepted", tc.allow, tc.raw)
		}
	}
}

func TestResolveParams_EmptyMeansNoFlag(t *testing.T) {
	got, err := resolveParams("", "")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if got != "" {
		t.Errorf("params = %q, want empty so -p is not passed at all", got)
	}
}

// Without -t, dcgmi's own timeout is UNLIMITED and the only bound is the
// runner's context — which expires as a SIGKILL that takes the JSON document
// with it. The derived value has to sit inside the budget for DCGM to end its
// own run and report what it got.
func TestResolveDiagTimeout_DerivedSitsInsideTheBudget(t *testing.T) {
	budget := 5 * time.Minute
	got := resolveDiagTimeout(0, budget)
	if got <= 0 {
		t.Fatalf("no timeout derived from a %s budget", budget)
	}
	if got >= budget {
		t.Errorf("derived timeout %s is not inside the %s budget, so it can never fire", got, budget)
	}
	if want := budget - diagTimeoutHeadroom; got != want {
		t.Errorf("derived timeout = %s, want %s", got, want)
	}
}

// A budget with no room for the headroom would derive a timeout that ends a
// diagnostic which had time to finish, which is worse than the SIGKILL.
func TestResolveDiagTimeout_TooSmallABudgetDerivesNothing(t *testing.T) {
	if got := resolveDiagTimeout(0, 40*time.Second); got != 0 {
		t.Errorf("derived %s from a 40s budget; want none", got)
	}
}

// An explicit value is honoured even when it sits outside the budget, on the
// same rule as an explicit level: the operator asked for it.
func TestResolveDiagTimeout_ExplicitWins(t *testing.T) {
	if got := resolveDiagTimeout(90*time.Second, 5*time.Minute); got != 90*time.Second {
		t.Errorf("timeout = %s, want the explicit 1m30s", got)
	}
	if got := resolveDiagTimeout(10*time.Minute, 5*time.Minute); got != 10*time.Minute {
		t.Errorf("timeout = %s, want the explicit 10m even though it exceeds the budget", got)
	}
}

// The metric is only worth reporting if it is what dcgmi was actually given.
func TestDiagArgsCarryTheParametersAndTimeout(t *testing.T) {
	cfg := config{address: "127.0.0.1:5555", params: "memory.is_allowed=true"}
	spec := runSpec{arg: "2", level: 2, source: "explicit"}

	args := diagArgs(cfg, spec, 105*time.Second)

	want := []string{
		"diag", "--host", "127.0.0.1:5555", "-r", "2", "-j",
		"-p", "memory.is_allowed=true",
		"-t", "105",
	}
	if strings.Join(args, "\x00") != strings.Join(want, "\x00") {
		t.Errorf("args = %q, want %q", args, want)
	}
}

// The parameter string is ONE argv element. Split on the ";" it contains, dcgmi
// would read the tail as a positional argument and the run would either fail or
// silently enable a different set of plugins than the metric claims.
func TestDiagArgsKeepTheParameterStringWhole(t *testing.T) {
	cfg := config{
		address: "127.0.0.1:5555",
		params:  "memory.is_allowed=true;pcie.is_allowed=true",
	}
	args := diagArgs(cfg, runSpec{arg: "2"}, 0)

	found := ""
	for i, a := range args {
		if a == "-p" && i+1 < len(args) {
			found = args[i+1]
		}
	}
	if found != cfg.params {
		t.Fatalf("-p carried %q, want the whole string %q", found, cfg.params)
	}
	for _, a := range args {
		if a == "pcie.is_allowed=true" {
			t.Error("the parameter string was split across argv elements")
		}
	}
}

// Neither flag is passed when nothing asked for it, so a profile that sets
// neither gets exactly the command line this runner has always issued.
func TestDiagArgsOmitBothFlagsWhenUnset(t *testing.T) {
	args := diagArgs(config{address: "h:1"}, runSpec{arg: "1"}, 0)
	want := []string{"diag", "--host", "h:1", "-r", "1", "-j"}
	if strings.Join(args, "\x00") != strings.Join(want, "\x00") {
		t.Errorf("args = %q, want %q", args, want)
	}
}

func TestDiagArgsCarryNamedTests(t *testing.T) {
	args := diagArgs(config{address: "h:1"}, runSpec{arg: "memory,pcie", tests: "memory,pcie"}, 0)
	for i, a := range args {
		if a == "-r" {
			if args[i+1] != "memory,pcie" {
				t.Fatalf("-r carried %q, want memory,pcie", args[i+1])
			}
			return
		}
	}
	t.Fatalf("no -r in %q", args)
}

// A mistyped parameter must cost nothing: the run ends before a host engine is
// started and before the GPU is touched.
func TestLoadConfigRejectsABadParameterBeforeAnythingRuns(t *testing.T) {
	t.Setenv("BURNIN_DCGM_PARAMS", "memory.is_allowed")
	if _, err := loadConfig(); err == nil {
		t.Fatal("loadConfig accepted a parameter with no value")
	}
}

func TestLoadConfigReadsTheParameterSurfaces(t *testing.T) {
	t.Setenv("BURNIN_DCGM_ALLOW", "memory")
	t.Setenv("BURNIN_DCGM_PARAMS", "diagnostic.test_duration=60")
	t.Setenv("BURNIN_DCGM_TESTS", "memory,pcie")
	t.Setenv("BURNIN_DCGM_TIMEOUT_SECONDS", "90")
	t.Setenv("BURNIN_DURATION_SECONDS", "600")

	cfg, err := loadConfig()
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if want := "memory.is_allowed=true;diagnostic.test_duration=60"; cfg.params != want {
		t.Errorf("params = %q, want %q", cfg.params, want)
	}
	if cfg.tests != "memory,pcie" {
		t.Errorf("tests = %q, want memory,pcie", cfg.tests)
	}
	if cfg.diagTimeout != 90*time.Second {
		t.Errorf("diagTimeout = %s, want 1m30s", cfg.diagTimeout)
	}
}

// #370: a profile that enables a gated plugin and leaves the level-derived
// budget in place gets cut off long before the plugin finishes — measured on
// GB10, the memory plugin alone ran 282-301s against the 105s a 2-minute
// duration derives. loadConfig refuses instead of silently deriving a budget
// already known to be wrong.
func TestLoadConfigRefusesAGatedPluginWithNoExplicitLongDuration(t *testing.T) {
	t.Setenv("BURNIN_DCGM_ALLOW", "memory")
	// Deliberately unset: this is the case the issue measured, where the level
	// (and so the derived -t) comes from BURNIN_DURATION_SECONDS alone.
	if _, err := loadConfig(); err == nil {
		t.Fatal("loadConfig accepted BURNIN_DCGM_ALLOW with no BURNIN_DURATION_SECONDS at all")
	}

	t.Setenv("BURNIN_DURATION_SECONDS", "120")
	if _, err := loadConfig(); err == nil {
		t.Fatal("loadConfig accepted BURNIN_DCGM_ALLOW with a 2-minute duration, " +
			"the exact case #370 measured cutting the memory plugin off")
	}
}

// The same guard fires for the raw BURNIN_DCGM_PARAMS surface, not just the
// ergonomic BURNIN_DCGM_ALLOW one — is_allowed=true has the same runtime
// impact however it was spelled.
func TestLoadConfigRefusesAGatedPluginNamedViaRawParams(t *testing.T) {
	t.Setenv("BURNIN_DCGM_PARAMS", "memory.is_allowed=true")
	t.Setenv("BURNIN_DURATION_SECONDS", "120")
	if _, err := loadConfig(); err == nil {
		t.Fatal("loadConfig accepted memory.is_allowed=true via BURNIN_DCGM_PARAMS with only a 2-minute duration")
	}
}

// A duration at or beyond the floor is accepted, and a run naming no gated
// plugin at all is never subject to the floor.
func TestLoadConfigAcceptsAGatedPluginWithALongEnoughDuration(t *testing.T) {
	t.Setenv("BURNIN_DCGM_ALLOW", "memory")
	t.Setenv("BURNIN_DURATION_SECONDS", "600")
	if _, err := loadConfig(); err != nil {
		t.Fatalf("loadConfig rejected a 600s duration, at the documented floor: %v", err)
	}

	t.Setenv("BURNIN_DCGM_ALLOW", "")
	t.Setenv("BURNIN_DURATION_SECONDS", "120")
	if _, err := loadConfig(); err != nil {
		t.Fatalf("loadConfig rejected a short duration with no gated plugin named: %v", err)
	}
}

// The acceptance criterion of #307: the runner REPORTS the exact parameter
// string it passed. Asserted end to end through run(), because the value being
// reported and the value reaching dcgmi are set in different places, and a
// diag_params that disagreed with the argv would be worse than none.
//
// dcgmi is deliberately absent, so run() reaches checkDCGMPresent and exits 3 —
// after the report is populated, which is the point. This needs no GPU, no host
// engine and no DCGM.
func TestRunReportsWhatItWasAskedToPass(t *testing.T) {
	t.Setenv("BURNIN_DCGM_ALLOW", "memory,pcie")
	t.Setenv("BURNIN_DCGM_TESTS", "memory,pcie")
	t.Setenv("BURNIN_DCGM_BIN", "/nonexistent/dcgmi-for-this-test")
	t.Setenv("BURNIN_DURATION_SECONDS", "600")

	stdout, code := captureRun(t)

	if code != exitError {
		t.Fatalf("exit = %d, want %d (dcgmi is absent, so the node is unjudged)", code, exitError)
	}
	res := runner.Parse("dcgm-diag", stdout, code)

	want := "memory.is_allowed=true;pcie.is_allowed=true"
	if got := res.Metrics["diagParams"]; got != want {
		t.Errorf("diagParams = %q, want %q", got, want)
	}
	if got := res.Metrics["diagTests"]; got != "memory,pcie" {
		t.Errorf("diagTests = %q, want memory,pcie", got)
	}
	if got := res.Metrics["diagLevelSource"]; got != "named" {
		t.Errorf("diagLevelSource = %q, want named", got)
	}
	// 600s budget less the 15s headroom. Reported so a run that ended at
	// DCGM's own timeout can be told from one that ended at the runner's.
	if got := res.Metrics["diagTimeoutS"]; got != "585" {
		t.Errorf("diagTimeoutS = %q, want 585", got)
	}
	// No level ran, so none is claimed.
	if got, present := res.Metrics["diagLevel"]; present {
		t.Errorf("diagLevel = %q on a named-test run; no level was executed", got)
	}
}

// captureRun runs the runner with os.Stdout replaced by a pipe. run() writes to
// os.Stdout directly, which is correct for a runner and is why this is needed.
func captureRun(t *testing.T) (string, int) {
	t.Helper()
	r, w, err := os.Pipe()
	if err != nil {
		t.Fatalf("pipe: %v", err)
	}
	saved := os.Stdout
	os.Stdout = w

	done := make(chan string, 1)
	go func() {
		var buf bytes.Buffer
		_, _ = io.Copy(&buf, r)
		done <- buf.String()
	}()

	code := run()

	os.Stdout = saved
	_ = w.Close()
	out := <-done
	_ = r.Close()
	return out, code
}

// diag_params holds a value containing both "=" and ";". The parser splits a
// metric line on the FIRST "=", so the whole parameter string must survive as
// one value — if it did not, the evidence for which plugins a run enabled would
// be truncated at the first clause.
func TestParameterStringSurvivesTheParserAsOneValue(t *testing.T) {
	params := "memory.is_allowed=true;pcie.h2d_d2h_single_pinned.min_bandwidth=8000"
	rep := newReport()
	rep.set(keyDiagParams, params)
	rep.set(keyDiagTests, "memory,pcie")
	rep.setInt(keyDiagTimeoutS, 105)

	var out bytes.Buffer
	rep.writeTo(&out)
	res := runner.Parse("dcgm-diag", out.String(), 0)

	if got := res.Metrics["diagParams"]; got != params {
		t.Errorf("diagParams = %q, want %q", got, params)
	}
	if got := res.Metrics["diagTests"]; got != "memory,pcie" {
		t.Errorf("diagTests = %q, want memory,pcie", got)
	}
	if got := res.Metrics["diagTimeoutS"]; got != "105" {
		t.Errorf("diagTimeoutS = %q, want 105", got)
	}
	if len(res.InvalidNames) > 0 {
		t.Errorf("the parser rejected %v", res.InvalidNames)
	}
}
