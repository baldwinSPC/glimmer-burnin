package invariants

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// CLAUDE.md and docs/dev/invariants.md carry the same rules — issue #178.
//
// The design rationale used to live only in CLAUDE.md, a file addressed to AI
// assistants. A stranger evaluating whether to adopt or contribute had to read
// that to find out why anything is the way it is, and the design proposals live
// in a repository they cannot read at all.
//
// Publishing it created the risk that always comes with a second copy: two
// copies that drift are two different rules, and the one somebody follows is
// whichever they found first. This is what stops that — CLAUDE.md's own note
// promises this guard exists, and a promised guard that does not exist is worse
// than no promise.

func read(t *testing.T, rel string) string {
	t.Helper()
	data, err := os.ReadFile(filepath.Join("..", "..", rel))
	if err != nil {
		t.Fatalf("%s: %v", rel, err)
	}
	return string(data)
}

// Every invariant CLAUDE.md asserts appears in the public page.
//
// Matched on the SUBJECT of each rule rather than on prose, because the two are
// deliberately worded differently — one instructs a session, the other explains
// to a person. What must not differ is which rules exist.
func TestThePublicPageCarriesEveryInvariant(t *testing.T) {
	claude := read(t, "CLAUDE.md")
	public := read(t, "docs/dev/invariants.md")

	// Each entry is a rule CLAUDE.md states, keyed by a phrase that identifies
	// the SUBJECT — and a distinctive token the public page must also carry.
	subjects := map[string]string{
		"the standalone rule":                "BurnInSink",
		"permissive-only licensing":          "copyleft",
		"Error is not Fail":                  "retryOnErrorLimit",
		"verdicts fail closed":               "fail",
		"the n/a three-state rule":           "n/a",
		"declared skips":                     "_SKIP",
		"exact comparison, no epsilon":       "epsilon",
		"Pair is about the link":             "endpoint",
		"Group is about the collective":      "rank",
		"cordoning follows the wave":         "cordon",
		"destroy only after durable":         "resourceVersion",
		"host access is named and narrow":    "hostPaths",
		"immutable published tags":           "immutable",
		"deliberate arch targets":            "PTX",
		"a pod is never safely evictable":    "safe-to-evict",
		"a Node verdict covers every device": "deviceCount",
	}

	for subject, token := range subjects {
		if !strings.Contains(claude, token) {
			t.Errorf("CLAUDE.md no longer mentions %q (%s) — this guard is keyed on a "+
				"token that has moved, and is checking nothing for that rule",
				token, subject)
			continue
		}
		if !strings.Contains(public, token) {
			t.Errorf("CLAUDE.md asserts %s and docs/dev/invariants.md does not mention "+
				"%q. The public page is what a stranger reads; a rule missing from it is "+
				"a rule they will not know applies to their contribution.", subject, token)
		}
	}
}

// The public page is a real document, not a stub that satisfies the check above.
func TestThePublicPageIsSubstantial(t *testing.T) {
	public := read(t, "docs/dev/invariants.md")
	if len(public) < 8000 {
		t.Fatalf("docs/dev/invariants.md is %d bytes — too short to carry the rationale, "+
			"and a page that merely contains the right words is not documentation", len(public))
	}
	// NO PROSE-QUALITY CHECK HERE, deliberately.
	//
	// Two were tried and both removed. Counting "because", then counting a
	// broader set of explanatory connectives, each came down to picking a number
	// that the current text happens to clear — which measures nothing, and the
	// second attempt made that obvious by failing at 36 when the page is
	// perfectly well argued. A guard tuned until it passes is a guard that will
	// pass whatever comes next.
	//
	// Whether the page explains itself is a review question. What a test can
	// hold is that every invariant is PRESENT and that the page is not a stub,
	// and that is what it holds.
}

// CLAUDE.md points at the public page, so a session knows the canonical text
// exists and does not fork it.
func TestClaudeMdPointsAtThePublicPage(t *testing.T) {
	claude := read(t, "CLAUDE.md")
	if !strings.Contains(claude, "docs/dev/invariants.md") {
		t.Fatal("CLAUDE.md does not name the public invariants page, so a session " +
			"editing an invariant here will not know to change it there")
	}
}
