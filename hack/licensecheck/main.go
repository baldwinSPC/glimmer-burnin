// Command licensecheck enforces the licence policy over the whole Go module
// graph.
//
// # Why this exists
//
// CLAUDE.md states that a copyleft dependency would make this project
// unpublishable, and that rule was enforced by PROSE. Individual runner
// Dockerfiles assert their upstream C project's licence at build time — which is
// genuinely good and catches the case that has actually bitten — but nothing
// looked at the Go module graph. A transitive dependency arriving under GPL
// passed every check this repository had (#177).
//
// # Why it is a Go program and not three lines of shell
//
// Because the ways this check can go wrong are all SILENT PASSES, and a silent
// pass is worse than no check: it reports a graph nobody examined as clean.
// There are two, and both are tested in main_test.go:
//
//   - go-licenses returns nothing at all — a build failure, a changed flag, a
//     module path that matches no packages. Zero violations out of zero
//     dependencies is not a pass.
//   - go-licenses cannot classify a licence and reports "Unknown". An
//     unidentified licence is the one MOST likely to be a problem, so treating
//     it as acceptable defeats the check precisely where it matters.
//
// Neither is reachable from a shell one-liner's exit code, and a guard whose
// failure modes cannot be tested is a guard nobody should trust.
package main

import (
	"bufio"
	"bytes"
	"fmt"
	"os"
	"os/exec"
	"sort"
	"strings"
)

// goLicensesVersion is pinned. An unpinned tool is a tool whose verdict can
// change without anything in this repository changing — the same reason every
// runner upstream is pinned to a commit.
const goLicensesVersion = "v1.6.0"

const allowedFile = "hack/licenses-allowed.txt"

func main() {
	allowed, err := readAllowList(allowedFile)
	if err != nil {
		fatal("%v", err)
	}
	fmt.Printf("Allowed licences: %s\n\n", strings.Join(sortedSet(allowed), ", "))

	csv, err := runGoLicenses()
	if err != nil {
		fatal("go-licenses could not analyse the module graph: %v", err)
	}

	report, err := classify(csv, allowed)
	if err != nil {
		fatal("%v", err)
	}
	fmt.Print(report.text())
	if !report.ok() {
		os.Exit(1)
	}
}

func fatal(format string, args ...any) {
	fmt.Fprintf(os.Stderr, "::error::"+format+"\n", args...)
	os.Exit(1)
}

// dep is one row of go-licenses' CSV: module, licence URL, licence name.
type dep struct {
	Module  string
	URL     string
	Licence string
}

type findings struct {
	Total        int
	Disallowed   []dep
	Unclassified []dep
}

func (f findings) ok() bool { return len(f.Disallowed) == 0 && len(f.Unclassified) == 0 }

func (f findings) text() string {
	var b strings.Builder
	for _, d := range f.Disallowed {
		fmt.Fprintf(&b, "::error::DISALLOWED LICENCE %s in %s\n   %s\n", d.Licence, d.Module, d.URL)
	}
	for _, d := range f.Unclassified {
		fmt.Fprintf(&b, "::error::UNCLASSIFIED LICENCE for %s\n   %s\n", d.Module, d.URL)
		b.WriteString("   go-licenses could not identify this dependency's licence. That is a\n" +
			"   FAILURE and not a pass: an unidentified licence is the one most likely to\n" +
			"   be a problem, and defaulting it to acceptable would defeat the check where\n" +
			"   it matters most.\n")
	}
	if f.ok() {
		fmt.Fprintf(&b, "OK: %d dependencies, every one under an allowed licence.\n", f.Total)
		return b.String()
	}
	fmt.Fprintf(&b, "\n%d disallowed and %d unclassified, out of %d dependencies.\n\n",
		len(f.Disallowed), len(f.Unclassified), f.Total)
	b.WriteString(
		"A copyleft dependency would make this project unpublishable. It is distributed as\n" +
			"Apache-2.0 source and as container images, and reciprocal terms are terms this\n" +
			"project cannot meet. The fix is a different dependency, or an interface this\n" +
			"project defines and the consumer implements. Where an upstream is DUAL-licensed,\n" +
			"consume it under the permissive option and say so in NOTICE — linux-rdma/perftest\n" +
			"is the worked example.\n\n" +
			"See " + allowedFile + ". Adding a line there is a licensing decision, not a build fix.\n")
	return b.String()
}

// classify applies the allow-list to go-licenses' output.
//
// It refuses an EMPTY input rather than reporting it clean. Zero violations out
// of zero dependencies is what this check looks like when it has stopped
// working, and it is indistinguishable from success by exit code alone.
func classify(csv []byte, allowed map[string]bool) (findings, error) {
	var f findings
	scanner := bufio.NewScanner(bytes.NewReader(csv))
	scanner.Buffer(make([]byte, 0, 64*1024), 1024*1024)
	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if line == "" {
			continue
		}
		// module,url,licence — the URL may itself contain commas, so the licence
		// is taken from the LAST field rather than the third.
		fields := strings.Split(line, ",")
		if len(fields) < 2 {
			return f, fmt.Errorf("go-licenses produced a row this check cannot read: %q", line)
		}
		d := dep{
			Module:  fields[0],
			URL:     strings.Join(fields[1:len(fields)-1], ","),
			Licence: strings.TrimSpace(fields[len(fields)-1]),
		}
		f.Total++
		switch {
		case allowed[d.Licence]:
		case d.Licence == "" || strings.EqualFold(d.Licence, "Unknown"):
			f.Unclassified = append(f.Unclassified, d)
		default:
			f.Disallowed = append(f.Disallowed, d)
		}
	}
	if err := scanner.Err(); err != nil {
		return f, err
	}
	if f.Total == 0 {
		return f, fmt.Errorf(
			"go-licenses reported NO dependencies at all. That is not a pass — it means " +
				"this check analysed nothing, and a copyleft dependency would sail through it")
	}
	return f, nil
}

// readAllowList reads the policy. Comments and blank lines are ignored, which is
// what lets every entry carry the reason it is safe.
//
// An empty allow-list is an ERROR rather than a deny-all, because the failure it
// would produce ("every dependency is disallowed") reads like a catastrophic
// licensing problem rather than like a missing file.
func readAllowList(path string) (map[string]bool, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("cannot read the licence policy at %s: %w", path, err)
	}
	allowed := map[string]bool{}
	for _, line := range strings.Split(string(data), "\n") {
		line = strings.TrimSpace(line)
		if line == "" || strings.HasPrefix(line, "#") {
			continue
		}
		allowed[line] = true
	}
	if len(allowed) == 0 {
		return nil, fmt.Errorf("%s lists no licences at all — refusing to run, because an "+
			"empty allow-list fails every dependency and reads like a licensing catastrophe "+
			"rather than a missing file", path)
	}
	return allowed, nil
}

func runGoLicenses(pkgs ...string) ([]byte, error) {
	if len(pkgs) == 0 {
		pkgs = []string{"./..."}
	}
	// `report` rather than `check`: check exits non-zero and names the package,
	// but not the licence that made it fail. The value of this job is telling a
	// contributor WHICH dependency and WHAT it is licensed under, in the failure
	// itself — otherwise the first thing they do is run the tool again by hand.
	args := append([]string{"run", "github.com/google/go-licenses@" + goLicensesVersion, "report"}, pkgs...)
	cmd := exec.Command("go", args...)
	var out, errb bytes.Buffer
	cmd.Stdout, cmd.Stderr = &out, &errb
	if err := cmd.Run(); err != nil {
		return nil, fmt.Errorf("%w\n%s", err, errb.String())
	}
	return out.Bytes(), nil
}

func sortedSet(m map[string]bool) []string {
	out := make([]string, 0, len(m))
	for k := range m {
		out = append(out, k)
	}
	sort.Strings(out)
	return out
}
