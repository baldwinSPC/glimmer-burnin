package playbook

import (
	"os"
	"path/filepath"
	"regexp"
	"strings"
	"testing"
)

// The playbook and the skill share ONE source — issue #180.
//
// The acceptance criterion is explicit: "the skill does not fork a second copy
// of the rules". That is not tidiness. Two copies of a rule become two different
// rules, and the second one to drift is the one somebody follows — while the
// guards in runners/pins_test.go continue to track only the first.
//
// So the skill must POINT at the document, and must not restate it.

func read(t *testing.T, rel string) string {
	t.Helper()
	data, err := os.ReadFile(filepath.Join("..", "..", rel))
	if err != nil {
		t.Fatalf("%s: %v", rel, err)
	}
	return string(data)
}

// The skill points at the playbook, by path, so an agent session loads the rules
// rather than rediscovering them.
func TestTheSkillPointsAtThePlaybook(t *testing.T) {
	skill := read(t, ".claude/skills/new-runner/SKILL.md")
	const path = "docs/dev/new-testkind-playbook.md"

	if !strings.Contains(skill, path) {
		t.Fatalf("the skill does not name %s, so a session using it never reads the "+
			"rules and rediscovers them — which is the expensive thing this exists to "+
			"stop", path)
	}
	if !strings.Contains(skill, "single source") && !strings.Contains(skill, "does not restate") {
		t.Error("the skill does not say it is a pointer rather than a copy, so the next " +
			"person to add a rule will reasonably add it in both places")
	}
}

// The skill does not fork the rules.
//
// Checked by looking for the SPECIFIC facts the playbook owns. A rule restated in
// the skill is a rule that will drift: the guards track the playbook, so the
// skill's copy would go stale silently and be trusted anyway.
func TestTheSkillDoesNotForkTheRules(t *testing.T) {
	skill := read(t, ".claude/skills/new-runner/SKILL.md")

	// Each of these is a concrete fact with exactly one correct home.
	forked := map[string]string{
		"BURNIN_ROOT_HOST":     "the Group environment contract",
		"BURNIN_PEER_HOST":     "the Pair environment contract",
		"Tflops":               "the unit-suffix list",
		"ldd":                  "the Dockerfile library guard",
		"^{}":                  "the pinned-SHA resolution recipe",
		"cluster.local":        "the DNS-qualification rule",
		"RequiredIfMeasurable": "the applicability rule",
	}
	for fact, owns := range forked {
		if strings.Contains(skill, fact) {
			t.Errorf("the skill restates %q, which is %s and belongs only in the "+
				"playbook. The guards track the playbook, so this copy drifts silently "+
				"and is trusted anyway.", fact, owns)
		}
	}
}

// The playbook names the guard for each rule, which is the second acceptance
// criterion: the reader has to learn which mistakes are caught mechanically and
// which are not.
func TestThePlaybookNamesItsGuards(t *testing.T) {
	doc := read(t, "docs/dev/new-testkind-playbook.md")

	// Every guard named in the playbook must actually exist, or the reader is
	// told a mistake is caught when it is not — which is worse than saying
	// nothing, because they will stop checking it themselves.
	//
	// Extracted from the guard callouts specifically, not from the whole
	// document: a bare `Test[A-Z]\w+` also matches the API's own type names —
	// TestKind, TestResult, TestScope — and a guard check that reports those as
	// missing tests is a check nobody will keep.
	var named []string
	for _, line := range strings.Split(doc, "\n") {
		if !strings.HasPrefix(strings.TrimSpace(line), ">") {
			continue
		}
		named = append(named, regexp.MustCompile(`Test[A-Z][A-Za-z]+`).FindAllString(line, -1)...)
	}
	if len(named) < 8 {
		t.Fatalf("the playbook names only %d guards; the point is that each rule says "+
			"what enforces it", len(named))
	}

	sources := strings.Join([]string{
		read(t, "runners/pins_test.go"),
		read(t, "runners/cxxtests_test.go"),
		read(t, "api/v1alpha1/samples_test.go"),
		read(t, "pkg/contract/metrics_test.go"),
		read(t, "pkg/runner/parse_test.go"),
	}, "\n")

	seen := map[string]bool{}
	for _, guard := range named {
		if seen[guard] {
			continue
		}
		seen[guard] = true
		if !strings.Contains(sources, "func "+guard+"(") {
			t.Errorf("the playbook names %s as a guard and no such test exists. A reader "+
				"told a mistake is caught mechanically will stop checking it themselves.",
				guard)
		}
	}
}

// The playbook covers every step the issue enumerates.
//
// Not a word count: each of these is a topic somebody adding a kind has to be
// told about, and the failure of omitting one is a runner that passes CI and is
// wrong on hardware.
func TestThePlaybookCoversEveryStep(t *testing.T) {
	doc := read(t, "docs/dev/new-testkind-playbook.md")

	for topic, must := range map[string]string{
		"the exit-code table":        "exit 3",
		"the declared-skip rule":     "_SKIP",
		"the reserved sentinel":      "n/a",
		"last-occurrence-wins":       "ccurrence-wins",
		"burst-only":                 "BurstOnly",
		"the unit suffixes":          "Tflops",
		"aggregation":                "Aggregation",
		"threshold-use for labels":   "Evidence",
		"the alias table":            "parse.go",
		"the Pair contract":          "BURNIN_PEER_HOST",
		"the Group contract":         "BURNIN_ROOT_HOST",
		"host access":                "hostPaths",
		"pinned upstreams":           "_SHA",
		"the redistributable guard":  "libcuda",
		"arch targets":               "PTX",
		"NOTICE":                     "NOTICE",
		"BuiltInKinds":               "BuiltInKinds",
		"the default image table":    "runnerimages",
		"immutable tags":             "immutable",
		"hardware before publishing": "verified on real hardware",
		"which runner to copy":       "host-health",
	} {
		if !strings.Contains(doc, must) {
			t.Errorf("the playbook does not cover %s (looked for %q). Omitting it means a "+
				"runner that passes CI and is wrong on hardware.", topic, must)
		}
	}
}
