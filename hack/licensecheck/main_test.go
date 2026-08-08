package main

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// The ways a licence check goes wrong are all SILENT PASSES — issue #177.
//
// That is what these tests are for. A guard that reports a graph nobody examined
// as clean is worse than no guard, because it converts "we have not checked" into
// "we have checked and it is fine", which is the statement the project's
// publishability rests on.

func allow(t *testing.T, licences ...string) map[string]bool {
	t.Helper()
	m := map[string]bool{}
	for _, l := range licences {
		m[l] = true
	}
	return m
}

// A copyleft dependency is named, with its licence, in the failure itself.
//
// "A licence check failed" sends the reader to run the tool by hand. The first
// thing they need is which dependency and what it is licensed under.
func TestACopyleftDependencyIsRefusedAndNamed(t *testing.T) {
	csv := []byte(`github.com/ok/one,https://example.com/one/LICENSE,Apache-2.0
github.com/juju/errors,https://github.com/juju/errors/blob/v1.0.0/LICENSE,LGPL-3.0
github.com/ok/two,https://example.com/two/LICENSE,MIT
`)
	f, err := classify(csv, allow(t, "Apache-2.0", "MIT"))
	if err != nil {
		t.Fatal(err)
	}
	if f.ok() {
		t.Fatal("an LGPL-3.0 dependency passed the licence check — this project is " +
			"distributed as Apache-2.0 source and as container images, and reciprocal " +
			"terms are terms it cannot meet")
	}
	if len(f.Disallowed) != 1 || f.Disallowed[0].Module != "github.com/juju/errors" {
		t.Fatalf("disallowed = %+v, want exactly the juju module", f.Disallowed)
	}
	text := f.text()
	for _, want := range []string{"github.com/juju/errors", "LGPL-3.0", "unpublishable"} {
		if !strings.Contains(text, want) {
			t.Errorf("the failure message omits %q:\n%s", want, text)
		}
	}
	// And the clean dependencies are not implicated.
	if strings.Contains(text, "github.com/ok/one") {
		t.Error("the failure names a dependency that is fine")
	}
}

// An UNCLASSIFIED licence is a failure, not a pass.
//
// It is the single most important line in this file. An unidentified licence is
// the one most likely to be a problem — a vendored blob, an unusual dual-licence
// header, a project with no LICENSE file — and defaulting it to acceptable
// defeats the check exactly where it matters.
func TestAnUnclassifiedLicenceIsAFailure(t *testing.T) {
	for _, licence := range []string{"Unknown", "unknown", ""} {
		csv := []byte("github.com/mystery/blob,https://example.com/LICENSE," + licence + "\n" +
			"github.com/ok/one,https://example.com/one/LICENSE,MIT\n")
		f, err := classify(csv, allow(t, "MIT"))
		if err != nil {
			t.Fatal(err)
		}
		if f.ok() {
			t.Fatalf("a dependency whose licence is %q passed — an unidentified licence "+
				"is the one most likely to be a problem", licence)
		}
		if len(f.Unclassified) != 1 {
			t.Fatalf("licence %q: unclassified = %+v, want one entry", licence, f.Unclassified)
		}
		if !strings.Contains(f.text(), "github.com/mystery/blob") {
			t.Errorf("licence %q: the failure does not name the module", licence)
		}
	}
}

// An EMPTY dependency list is a failure, not a clean graph.
//
// Zero violations out of zero dependencies is what this check looks like when it
// has stopped working — a changed flag, a build failure, a package pattern that
// matches nothing — and by exit code alone it is indistinguishable from success.
func TestAnEmptyGraphIsNotAPass(t *testing.T) {
	for name, csv := range map[string][]byte{
		"nothing":     nil,
		"empty":       []byte(""),
		"blank lines": []byte("\n\n   \n"),
	} {
		t.Run(name, func(t *testing.T) {
			if _, err := classify(csv, allow(t, "MIT")); err == nil {
				t.Fatal("an empty dependency list was reported as a clean graph — that is " +
					"what this check looks like when it has stopped working, and a copyleft " +
					"dependency would sail through it")
			}
		})
	}
}

// A licence URL containing a comma must not shift the licence field.
//
// go-licenses emits three comma-separated fields and the middle one is a URL,
// which may legitimately contain a comma. Splitting on the third field would
// then read part of the URL as the licence — and an unrecognised "licence" that
// happens to be a URL fragment would be reported as a violation on a clean
// dependency, or worse, a real licence name would be lost off the end.
func TestACommaInTheURLDoesNotShiftTheLicence(t *testing.T) {
	csv := []byte("github.com/odd/pkg,https://example.com/a,b/LICENSE,Apache-2.0\n")
	f, err := classify(csv, allow(t, "Apache-2.0"))
	if err != nil {
		t.Fatal(err)
	}
	if !f.ok() {
		t.Fatalf("a comma in the licence URL made a clean Apache-2.0 dependency fail: %+v", f)
	}
}

// The policy file is the policy, and it must parse to what it says.
func TestTheShippedAllowListIsPermissiveOnly(t *testing.T) {
	path := filepath.Join("..", "..", allowedFile)
	allowed, err := readAllowList(path)
	if err != nil {
		t.Fatal(err)
	}
	// Every licence the project was founded on is present.
	for _, want := range []string{"Apache-2.0", "MIT", "BSD-2-Clause", "BSD-3-Clause"} {
		if !allowed[want] {
			t.Errorf("%s is missing from the allow-list", want)
		}
	}
	// And nothing reciprocal has been added, by accident or in a hurry to make a
	// build pass. This is spelled out rather than derived, because the whole
	// point is that a human adding one of these has to delete this line first.
	for _, forbidden := range []string{
		"GPL-2.0", "GPL-3.0", "AGPL-3.0", "LGPL-2.1", "LGPL-3.0",
		"SSPL-1.0", "MPL-2.0", "Sleepycat", "CDDL-1.0", "EPL-2.0", "CPL-1.0",
	} {
		if allowed[forbidden] {
			t.Errorf("%s is on the allow-list. There is no exception process for copyleft: "+
				"it would make this project unpublishable, and no build failure is a "+
				"reason to add one", forbidden)
		}
	}
	// The file must carry its reasoning. An allow-list of bare identifiers is a
	// list somebody will append to without thinking.
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(string(data), "#") {
		t.Error("the allow-list carries no comments — every entry should say why it is safe")
	}
}

// An empty policy file is refused rather than treated as deny-all.
//
// Deny-all would fail every dependency at once, which reads like a catastrophic
// licensing problem rather than like a file that failed to load — and somebody
// debugging that at 2am will reach for the allow-list before they reach for the
// missing file.
func TestAnEmptyPolicyFileIsRefused(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "empty.txt")
	if err := os.WriteFile(path, []byte("# only comments\n\n"), 0o600); err != nil {
		t.Fatal(err)
	}
	if _, err := readAllowList(path); err == nil {
		t.Fatal("an allow-list with no licences in it was accepted")
	}
	if _, err := readAllowList(filepath.Join(dir, "does-not-exist.txt")); err == nil {
		t.Fatal("a missing policy file was accepted")
	}
}
