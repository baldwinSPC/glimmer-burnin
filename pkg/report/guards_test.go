package report_test

import (
	"go/build"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"strings"
	"testing"

	"github.com/baldwinSPC/glimmer-burnin/pkg/report"
)

// TestPackageImportsNothingUnexpected holds the package's first rule.
//
// pkg/report exists so that a consumer can render a verdict without carrying
// Kubernetes. pkg/verdict takes that dependency because it must speak Threshold;
// this one has no such excuse, and the audience most likely to import it — a CI
// job, the bare-metal CLI, an ingest service — is exactly the audience that
// should not be made to pull apimachinery to read a report.
//
// The failure this prevents is silent: an import added for one convenient type
// costs every downstream consumer megabytes of transitive dependency, and
// nobody notices until someone tries to vendor it somewhere small.
func TestPackageImportsNothingUnexpected(t *testing.T) {
	out, err := exec.Command("go", "list", "-deps", "github.com/baldwinSPC/glimmer-burnin/pkg/report").Output()
	if err != nil {
		t.Skipf("go list unavailable: %v", err)
	}
	for _, dep := range strings.Fields(string(out)) {
		switch {
		case strings.HasPrefix(dep, "k8s.io/"),
			strings.HasPrefix(dep, "sigs.k8s.io/"),
			strings.HasPrefix(dep, "github.com/baldwinSPC/glimmer-burnin/api/"),
			strings.HasPrefix(dep, "github.com/baldwinSPC/glimmer-burnin/internal/"):
			t.Errorf("pkg/report depends on %s — it must stay renderable without Kubernetes", dep)
		}
	}
}

// TestPhaseVocabularyMatchesTheAPI catches drift in the one thing this package
// duplicates on purpose.
//
// The phase strings are restated here rather than imported, because importing
// them would breach the rule above. Restating carries the usual cost: the API
// can grow a phase and this package would not know, and the symptom would be a
// report that quietly treats an unrecognised phase as non-terminal — an
// unfinished-looking document about a run that had in fact finished.
//
// So the guard reads the API's constants from source and requires every one to
// be known here.
func TestPhaseVocabularyMatchesTheAPI(t *testing.T) {
	src, err := os.ReadFile(filepath.Join("..", "..", "api", "v1alpha1", "burninrun_types.go"))
	if err != nil {
		t.Fatalf("reading the API types: %v", err)
	}

	// e.g.  RunPassed RunPhase = "Passed"
	re := regexp.MustCompile(`(?m)^\s*Run[A-Za-z]+\s+RunPhase\s*=\s*"([A-Za-z]+)"`)
	matches := re.FindAllStringSubmatch(string(src), -1)
	if len(matches) == 0 {
		t.Fatal("found no RunPhase constants — the guard's pattern has gone stale, which is itself the bug")
	}

	known := map[string]bool{
		report.PhasePending:   true,
		report.PhaseRunning:   true,
		report.PhasePassed:    true,
		report.PhaseFailed:    true,
		report.PhaseError:     true,
		report.PhaseSkipped:   true,
		report.PhaseCancelled: true,
	}
	for _, m := range matches {
		if !known[m[1]] {
			t.Errorf("the API defines phase %q and pkg/report does not know it — "+
				"an unknown phase would be treated as non-terminal, so a finished run would render as unfinished", m[1])
		}
	}
}

// TestTerminalityAgreesWithTheAPI pins the one behavioural claim the duplicated
// vocabulary makes.
func TestTerminalityAgreesWithTheAPI(t *testing.T) {
	terminal := map[string]bool{
		report.PhasePassed:    true,
		report.PhaseFailed:    true,
		report.PhaseError:     true,
		report.PhaseSkipped:   true,
		report.PhaseCancelled: true,
		report.PhasePending:   false,
		report.PhaseRunning:   false,
	}
	for phase, want := range terminal {
		if got := report.IsTerminal(phase); got != want {
			t.Errorf("IsTerminal(%q) = %v, want %v", phase, got, want)
		}
	}
	if report.IsTerminal("SomethingNobodyHasDefined") {
		t.Error("an unrecognised phase must not be treated as terminal")
	}
}

// keep build imported so the file documents its own toolchain assumption.
var _ = build.Default
