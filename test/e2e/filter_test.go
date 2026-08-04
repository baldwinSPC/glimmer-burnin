// SPDX-License-Identifier: Apache-2.0
// Copyright the Glimmer authors.

// This file deliberately carries NO `e2e` build tag, unlike every other file in
// this package. It needs no cluster — it reads the workflow and the test sources
// as text — and it has to run in the ordinary `go test ./...` that every PR
// executes, because what it guards is whether the e2e tests run AT ALL.
package e2e

import (
	"os"
	"regexp"
	"strings"
	"testing"
)

const ciWorkflow = "../../.github/workflows/ci.yml"

// runFilterPattern extracts the -run regex from each `go test -tags e2e`
// invocation in the workflow.
var runFilterPattern = regexp.MustCompile(`go test -tags e2e [^\n]*-run '([^']*)'`)

// testFuncPattern finds the test functions declared in this package's sources.
var testFuncPattern = regexp.MustCompile(`(?m)^func (Test[A-Za-z0-9_]*)\(`)

// TestEveryE2ETestIsSelectedByTheWorkflow closes a hole that cost a test its
// entire reason for existing.
//
// The e2e job does not run the package; it runs two hand-written `-run` regexes
// over it — one for the core suite on every PR, one for the chaos suite on main.
// A test whose name does not match either is compiled, is green, is reviewed,
// and NEVER EXECUTES. Nothing says so: the job passes, and the summary lists the
// tests that did run without hinting at the one that did not.
//
// That is exactly what happened to TestAnUndeclaredExitTwoIsAnErrorNotASkip. It
// was added to prove on a real cluster that a crashed runner is not recorded as
// "does not apply to this hardware" (#103), and the core filter's `TestA(...)`
// alternation did not admit a name beginning "TestAn". The PR went green having
// never run it.
//
// A silent omission is the worst failure mode a test suite can have, because the
// coverage is believed. So: every test in this package must be selected by
// exactly one of the filters — unselected means it never runs, and selected by
// both means the chaos job re-runs a core test and the reported timings stop
// meaning what they say.
func TestEveryE2ETestIsSelectedByTheWorkflow(t *testing.T) {
	wf, err := os.ReadFile(ciWorkflow)
	if err != nil {
		t.Fatalf("reading %s: %v", ciWorkflow, err)
	}
	rawFilters := runFilterPattern.FindAllStringSubmatch(string(wf), -1)
	if len(rawFilters) != 2 {
		t.Fatalf("found %d `go test -tags e2e ... -run` invocations in %s, want 2 (core and chaos). "+
			"If the job structure changed, this guard has to change with it.", len(rawFilters), ciWorkflow)
	}

	var filters []*regexp.Regexp
	for _, m := range rawFilters {
		// Go's -run anchors each slash-separated element, so a filter selects a
		// top-level test when the pattern matches anywhere in the name... except
		// that `go test` treats the pattern as unanchored per element. Compile it
		// as written, which is what `go test` does.
		re, err := regexp.Compile(m[1])
		if err != nil {
			t.Fatalf("workflow -run filter %q does not compile: %v", m[1], err)
		}
		filters = append(filters, re)
	}

	entries, err := os.ReadDir(".")
	if err != nil {
		t.Fatalf("reading test/e2e: %v", err)
	}
	var names []string
	for _, e := range entries {
		if e.IsDir() || !strings.HasSuffix(e.Name(), "_test.go") || e.Name() == "filter_test.go" {
			continue
		}
		src, err := os.ReadFile(e.Name())
		if err != nil {
			t.Fatalf("reading %s: %v", e.Name(), err)
		}
		for _, m := range testFuncPattern.FindAllStringSubmatch(string(src), -1) {
			// TestMain is the package entrypoint, not a test. `go test` calls it
			// regardless of -run, so requiring a filter to select it would be
			// asking for the one name that must NOT be selected.
			if m[1] == "TestMain" {
				continue
			}
			names = append(names, m[1])
		}
	}
	if len(names) == 0 {
		t.Fatal("no test functions found; this guard is not reading the e2e sources")
	}

	for _, name := range names {
		matched := 0
		for _, re := range filters {
			if re.MatchString(name) {
				matched++
			}
		}
		switch matched {
		case 0:
			t.Errorf("%s is selected by NEITHER workflow filter, so it never runs. It compiles, it is "+
				"green, and it proves nothing. Add it to the core or chaos -run pattern in %s.",
				name, ciWorkflow)
		case 1: // exactly right
		default:
			t.Errorf("%s is selected by BOTH workflow filters, so the chaos job re-runs a core test and "+
				"the reported timings stop meaning what they say.", name)
		}
	}
}
