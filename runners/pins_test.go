// SPDX-License-Identifier: Apache-2.0
// Copyright the Glimmer authors.

// Package runners holds no runner. It is the seam between the runner images and
// the repository, and it exists so that the rules which apply to ALL of them
// have somewhere to be enforced from — a directory listing is the only thing
// that knows how many runners there are.
//
// Three rules live here, and none of them is one a reviewer can enforce by
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
package runners

import (
	"fmt"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strings"
	"testing"
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
