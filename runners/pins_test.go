// SPDX-License-Identifier: Apache-2.0
// Copyright the Glimmer authors.

// Package runners holds no runner. It is the seam between the runner images and
// the repository, and it exists so that the rules which apply to ALL of them
// have somewhere to be enforced from — a directory listing is the only thing
// that knows how many runners there are.
//
// The rules below live here, and none of them is one a reviewer can enforce by
// reading a single diff:
//
//   - Every upstream a runner fetches is pinned to a COMMIT, and the build
//     asserts it. A published runner tag is immutable and a node's readiness
//     gate executes it, so a moved upstream tag would change what a fleet is
//     measured by with no audit trail and nothing to bisect.
//   - Every runner directory can actually be published. The publish workflow's
//     `runner` input is a hand-written choice list; a directory missing from it
//     cannot be built at all, and a defaultRunnerImages entry pointing at a tag
//     no workflow can produce is an ImagePullBackOff with no way out.
//   - Every C++ unit test a runner ships is actually compiled and run
//     (cxxtests_test.go). The C++ and CUDA runners cannot be tested from a Go
//     package of their own, so their tests are plain programs that nothing ran
//     until the sweep here existed.
//   - Every runner either READS the duration the operator injects, or its kind
//     says out loud that it does not. A runner that silently ignores the
//     duration it was given reports a burn-in that did not happen.
//   - No Dockerfile bakes in a fault-injection macro. Those switches exist so a
//     branch no available hardware can reach gets exercised anyway; an image
//     built with one would report a fabricated verdict for a whole fleet.
package runners

import (
	"fmt"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strings"
	"testing"

	burninv1alpha1 "github.com/baldwinSPC/glimmer-burnin/api/v1alpha1"
)

// publishWorkflow is the workflow whose choice list must name every runner.
const publishWorkflow = "../.github/workflows/publish-runner.yml"

// runnerDirs lists every directory under runners/, from the filesystem. It is
// deliberately not a hand-written list: the whole point of the checks below is
// that a twelfth runner cannot be added without them noticing it.
func runnerDirs(t *testing.T) []string {
	t.Helper()
	entries, err := os.ReadDir(".")
	if err != nil {
		t.Fatalf("reading runners/: %v", err)
	}
	var out []string
	for _, e := range entries {
		if e.IsDir() {
			out = append(out, e.Name())
		}
	}
	if len(out) == 0 {
		t.Fatal("no runner directories found; this test is not running from runners/")
	}
	sort.Strings(out)
	return out
}

// TestEveryRunnerDirectoryHasADockerfile is the floor. A runner directory with
// no Dockerfile is source that nothing can turn into the image a TestKind runs.
func TestEveryRunnerDirectoryHasADockerfile(t *testing.T) {
	for _, d := range runnerDirs(t) {
		if _, err := os.Stat(filepath.Join(d, "Dockerfile")); err != nil {
			t.Errorf("runners/%s has no Dockerfile: %v", d, err)
		}
	}
}

// TestDurationIsHonouredOrDeclaredBurstOnly keeps a TestKind's promise about
// BURNIN_DURATION_SECONDS tied to the code that does or does not read it.
//
// The operator injects that variable into every runner at every scope. For a
// long time compute-smoke was the one runner that never read it, while a shipped
// sample asked the kind for 120 seconds — so "node acceptance" burned nothing in
// and nothing anywhere said so (issue #25). The resolution was to declare the
// kind burst-only rather than to bolt a loop onto it, which means the honesty of
// the whole arrangement now rests on that declaration staying true.
//
// So both directions are checked, from the filesystem rather than from a list:
// a runner whose kind is NOT burst-only must actually read the variable, and a
// runner whose kind IS burst-only must not. The second half is what catches the
// happier failure — somebody adding a duration loop to compute-smoke and leaving
// the kind, the sample and the README describing a burst.
//
// Only SOURCE is scanned. A README that merely mentions the variable is exactly
// the kind of documentation this guard exists to stop trusting.
func TestDurationIsHonouredOrDeclaredBurstOnly(t *testing.T) {
	const durationEnv = "BURNIN_DURATION_SECONDS"

	for _, d := range runnerDirs(t) {
		// The directory name IS the TestKind value, which is what lets this run
		// off a directory listing; TestEveryRunnerDirectoryHasADockerfile and
		// the publish-workflow check rest on the same correspondence.
		kind := burninv1alpha1.TestKind(d)
		reads := false
		err := filepath.WalkDir(d, func(path string, e os.DirEntry, err error) error {
			if err != nil || e.IsDir() || reads {
				return err
			}
			switch filepath.Ext(path) {
			case ".go", ".cu", ".cuh", ".cc", ".h":
			default:
				return nil
			}
			// A test file naming the variable proves nothing about the runner.
			if strings.HasSuffix(path, "_test.go") || strings.HasSuffix(path, "_test.cc") {
				return nil
			}
			src, readErr := os.ReadFile(path)
			if readErr != nil {
				return readErr
			}
			if strings.Contains(string(src), durationEnv) {
				reads = true
			}
			return nil
		})
		if err != nil {
			t.Errorf("walking runners/%s: %v", d, err)
			continue
		}

		switch {
		case kind.BurstOnly() && reads:
			t.Errorf("runners/%s reads %s, but TestKind(%q).BurstOnly() says it does not use a "+
				"duration. If the runner now honours one, flip BurstOnly, restore durationSeconds "+
				"to the sample, and update the kind doc — the three must agree",
				d, durationEnv, d)
		case !kind.BurstOnly() && !reads:
			t.Errorf("no source file under runners/%s reads %s, yet TestKind(%q) is not declared "+
				"burst-only. A test that silently ignores the duration it was given reports a "+
				"burn-in that did not happen; either honour it, or declare the kind burst-only",
				d, durationEnv, d)
		}
	}
}

// faultInjectionMacros are compile-time switches a runner source defines for
// TEST builds only — the ones that make a branch reachable which the hardware in
// front of us cannot reach on its own.
//
// compute-smoke's pair is the case: exit 2 fires only on a part that is not
// compute capability 12.0/12.1, and no such accelerator is available to this
// project, so the Skip path had never once executed (issue #9). Defining these
// builds the same source with the same toolchain and tells it the device reports
// a capability it does not have, which is how that path gets exercised on a real
// GPU rather than only in a unit test.
//
// The guard below is the price of having them. A published runner image is what
// a node's readiness gate executes, and a Skip means "acceptance does not apply
// to this node" — so an image built with one of these defined would report a
// clean, fabricated not-applicable for an entire fleet, with the retry budget
// unspent and nothing in the result reading as wrong. The macros are
// deliberately compile-time so no environment variable can reach them; this
// makes sure no Dockerfile in the repository can bake one in either.
var faultInjectionMacros = []string{
	"BURNIN_FP4_FORCE_CC_MAJOR",
	"BURNIN_FP4_FORCE_CC_MINOR",
}

// TestNoDockerfileDefinesAFaultInjectionMacro keeps the test-only switches out
// of every image this repository can build.
//
// It sweeps every runner rather than only the one that has them today: the next
// runner to need a fault-injection switch will copy the pattern, and the guard
// should already be watching when it does.
func TestNoDockerfileDefinesAFaultInjectionMacro(t *testing.T) {
	for _, d := range runnerDirs(t) {
		path := filepath.Join(d, "Dockerfile")
		src, err := os.ReadFile(path)
		if err != nil {
			continue // TestEveryRunnerDirectoryHasADockerfile reports this.
		}
		for _, macro := range faultInjectionMacros {
			if strings.Contains(string(src), macro) {
				t.Errorf("%s defines the fault-injection macro %s; an image built with it would "+
					"report a fabricated verdict for every node it ran on. Build such a binary by "+
					"hand for a one-off experiment, never from a Dockerfile that can be published",
					path, macro)
			}
		}
	}
}

// TestFaultInjectionMacrosAnnounceThemselves asserts the other half of the same
// safeguard, in the source rather than the build: a binary that WAS built with
// one of these must say so on stdout.
//
// Belt and braces on purpose. The Dockerfile guard above stops the macro
// reaching a published image; this stops a hand-built binary's output from being
// mistaken for a real run — the marker is a key=value line, so it lands in the
// stored TestResult and in the delivered envelope, and any verdict such a build
// produced stays self-identifying for as long as the result exists.
func TestFaultInjectionMacrosAnnounceThemselves(t *testing.T) {
	src, err := os.ReadFile(filepath.Join("compute-smoke", "fp4_smoke.cu"))
	if err != nil {
		t.Fatalf("reading compute-smoke/fp4_smoke.cu: %v", err)
	}
	text := string(src)
	for _, macro := range faultInjectionMacros {
		if !strings.Contains(text, macro) {
			t.Errorf("fp4_smoke.cu no longer mentions %s; if the switch was removed, remove it from "+
				"faultInjectionMacros too rather than leaving a guard over nothing", macro)
		}
	}
	if !strings.Contains(text, `printf("forced_compute_cap=`) {
		t.Error("fp4_smoke.cu does not print a forced_compute_cap= marker; a fault-injected build " +
			"must announce itself in its own output, or its verdicts are indistinguishable from real ones")
	}
}

// refPattern finds the ARG declarations that name an upstream git ref, and
// shaPattern the commit each must be pinned to. The naming convention
// (<UPSTREAM>_REF / <UPSTREAM>_SHA) is what makes this mechanical; it is the
// convention every runner already follows.
var (
	refPattern   = regexp.MustCompile(`(?m)^ARG ([A-Z0-9_]+)_REF=(\S*)`)
	shaPattern   = regexp.MustCompile(`(?m)^ARG ([A-Z0-9_]+)_SHA=([0-9a-f]*)`)
	clonePattern = regexp.MustCompile(`git clone [^\n]*`)
)

// TestEveryUpstreamRefIsPinnedToACommit is the audit-trail rule with teeth.
//
// A tag is a mutable pointer. Upstreams do move them — to re-cut a release, to
// correct a packaging mistake, occasionally to hide one — and a runner image
// built from a moved tag contains different code under the same immutable image
// tag. Every node that gate then judges is judged by something no record in this
// repository describes.
//
// So each <UPSTREAM>_REF must be accompanied by an <UPSTREAM>_SHA that is a full
// 40-character commit id, and the Dockerfile must actually COMPARE the two: a
// declared-but-unused SHA is documentation, and documentation does not fail a
// build.
func TestEveryUpstreamRefIsPinnedToACommit(t *testing.T) {
	for _, d := range runnerDirs(t) {
		path := filepath.Join(d, "Dockerfile")
		src, err := os.ReadFile(path)
		if err != nil {
			continue // TestEveryRunnerDirectoryHasADockerfile reports this.
		}
		text := string(src)

		shas := map[string]string{}
		for _, m := range shaPattern.FindAllStringSubmatch(text, -1) {
			shas[m[1]] = m[2]
		}

		for _, upstream := range sortedUnique(refPattern.FindAllStringSubmatch(text, -1)) {
			sha, ok := shas[upstream]
			if !ok {
				t.Errorf("%s declares %s_REF with no %s_SHA. A tag can be moved, and a published "+
					"runner tag is immutable, so an image built from a moved tag silently changes what "+
					"a fleet is measured by. Resolve the commit with "+
					"`git ls-remote <repo> 'refs/tags/<ref>' 'refs/tags/<ref>^{}'` — for an ANNOTATED "+
					"tag take the peeled (^{}) value, which is what a shallow clone leaves at HEAD.",
					path, upstream, upstream)
				continue
			}
			if len(sha) != 40 {
				t.Errorf("%s: %s_SHA=%q is not a full 40-character commit id. An abbreviated id is "+
					"not a pin: it can become ambiguous as the upstream grows.", path, upstream, sha)
			}
			// The pin has to be load-bearing. Look for the SHA being used
			// somewhere other than its own declaration.
			uses := strings.Count(text, fmt.Sprintf("${%s_SHA}", upstream))
			if uses == 0 {
				t.Errorf("%s declares %s_SHA but never compares against it. Add the assertion that "+
					"refuses to build when the clone is not that commit; a pin nothing checks is a "+
					"comment.", path, upstream)
			}
		}
	}
}

// TestEveryCloneNamesAPinnedRef closes the other half of the same hole: an
// upstream fetched without -b takes the remote's default branch, which is
// "whatever main was that morning" and is not a reproducible acceptance verdict.
func TestEveryCloneNamesAPinnedRef(t *testing.T) {
	for _, d := range runnerDirs(t) {
		path := filepath.Join(d, "Dockerfile")
		src, err := os.ReadFile(path)
		if err != nil {
			continue
		}
		for _, clone := range clonePattern.FindAllString(string(src), -1) {
			if !strings.Contains(clone, "_REF}") {
				t.Errorf("%s clones without a pinned ref:\n    %s\nEvery upstream must be fetched at "+
					"an <UPSTREAM>_REF that a matching <UPSTREAM>_SHA asserts.", path, strings.TrimSpace(clone))
			}
		}
	}
}

// dockerPredefinedArgs are the build args BuildKit supplies without an ARG
// declaration. They may be used in a FROM at any point.
var dockerPredefinedArgs = map[string]bool{
	"BUILDPLATFORM": true, "BUILDOS": true, "BUILDARCH": true, "BUILDVARIANT": true,
	"TARGETPLATFORM": true, "TARGETOS": true, "TARGETARCH": true, "TARGETVARIANT": true,
}

var (
	argDeclPattern  = regexp.MustCompile(`(?i)^\s*ARG\s+([A-Za-z0-9_]+)`)
	fromLinePattern = regexp.MustCompile(`(?i)^\s*FROM\s`)
	varRefPattern   = regexp.MustCompile(`\$\{?([A-Za-z0-9_]+)`)
)

// TestFromArgsAreDeclaredBeforeTheFirstFrom is the structural check that would
// have stopped runners/memory-bw shipping a Dockerfile nobody had ever built.
//
// An ARG that a FROM expands must be declared in the GLOBAL scope — before the
// first FROM. An ARG declared after one belongs to that build stage and is
// invisible to every later FROM, so the later FROM expands to the empty string
// and the build dies with "base name should not be blank", a message that names
// neither the ARG nor the line. memory-bw shipped exactly that, and dcgm-diag's
// Dockerfile carries a comment about it, because both were written before
// anything in CI built them.
//
// This is deliberately a Go test rather than a Docker step: it needs no daemon,
// no QEMU and no minutes, so it runs on every PR and locally in `make test`,
// which is where the eleven full image builds cannot afford to be.
func TestFromArgsAreDeclaredBeforeTheFirstFrom(t *testing.T) {
	for _, d := range runnerDirs(t) {
		path := filepath.Join(d, "Dockerfile")
		src, err := os.ReadFile(path)
		if err != nil {
			continue
		}

		global := map[string]bool{}
		seenFrom := false
		for n, line := range strings.Split(string(src), "\n") {
			if m := argDeclPattern.FindStringSubmatch(line); m != nil && !seenFrom {
				global[m[1]] = true
				continue
			}
			if !fromLinePattern.MatchString(line) {
				continue
			}
			seenFrom = true
			for _, ref := range varRefPattern.FindAllStringSubmatch(line, -1) {
				name := ref[1]
				if dockerPredefinedArgs[name] || global[name] {
					continue
				}
				t.Errorf("%s:%d expands ${%s} in a FROM, but %s is not declared before the first "+
					"FROM.\n    %s\nAn ARG declared after a FROM belongs to that stage and is invisible "+
					"to any later FROM: the base name expands to the empty string and the build fails "+
					"with \"base name should not be blank\", which points at nothing. Move the ARG to "+
					"the global scope (and re-declare it inside any stage that also uses it).",
					path, n+1, name, name, strings.TrimSpace(line))
			}
		}
	}
}

// TestPublishWorkflowOffersEveryRunner keeps the publish workflow's hand-written
// choice list in step with the filesystem.
//
// workflow_dispatch inputs of type `choice` cannot be generated, so this list is
// the one place in the runner tooling that must be maintained by hand. A runner
// missing from it cannot be published at all — and the operator's
// defaultRunnerImages would point at a tag no workflow can produce, which
// reaches the fleet as an ImagePullBackOff on every node.
func TestPublishWorkflowOffersEveryRunner(t *testing.T) {
	src, err := os.ReadFile(publishWorkflow)
	if err != nil {
		t.Fatalf("reading %s: %v", publishWorkflow, err)
	}

	offered := map[string]bool{}
	inOptions := false
	for _, line := range strings.Split(string(src), "\n") {
		trimmed := strings.TrimSpace(line)
		switch {
		case trimmed == "options:":
			inOptions = true
		case inOptions && strings.HasPrefix(trimmed, "- "):
			offered[strings.TrimSpace(strings.TrimPrefix(trimmed, "- "))] = true
		case inOptions && trimmed != "" && !strings.HasPrefix(trimmed, "#"):
			inOptions = false
		}
	}
	if len(offered) == 0 {
		t.Fatalf("no runner choices found in %s; the parser no longer matches the workflow", publishWorkflow)
	}

	dirs := runnerDirs(t)
	for _, d := range dirs {
		if !offered[d] {
			t.Errorf("runners/%s is not offered by %s. It cannot be published, so no default runner "+
				"image can ever point at it.", d, publishWorkflow)
		}
		delete(offered, d)
	}
	for name := range offered {
		t.Errorf("%s offers %q, which is not a directory under runners/. Publishing it would fail "+
			"with an empty build context.", publishWorkflow, name)
	}
}

// sortedUnique returns the first submatch of each match, deduplicated. An ARG is
// re-declared once per build stage that consumes it, so the same upstream
// appears several times in one Dockerfile.
func sortedUnique(matches [][]string) []string {
	seen := map[string]bool{}
	var out []string
	for _, m := range matches {
		if seen[m[1]] {
			continue
		}
		seen[m[1]] = true
		out = append(out, m[1])
	}
	sort.Strings(out)
	return out
}
