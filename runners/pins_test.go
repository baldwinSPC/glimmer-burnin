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
	"github.com/baldwinSPC/glimmer-burnin/pkg/runner"
)

// publishWorkflow is the workflow whose choice list must name every runner.
const publishWorkflow = "../.github/workflows/publish-runner.yml"

// ciWorkflow is the other workflow that builds a runner image. It and
// publishWorkflow must agree about the disk they reclaim first.
const ciWorkflow = "../.github/workflows/ci.yml"

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

// reclaimPathsPattern finds every line in which a workflow declares the disk it
// frees before pulling a layer. EVERY, not the first: there are three copies
// across two files, and a pattern matched once per file is how the third one
// went unguarded for as long as it did.
var reclaimPathsPattern = regexp.MustCompile(`(?m)^\s*RECLAIM_PATHS='([^']*)'`)

// TestEveryWorkflowReclaimsTheSameDisk guards a rule a reviewer cannot enforce by
// reading one diff, because it spans copies that are never edited together.
//
// A hosted runner has ~14 GB free and ships 20-30 GB of preinstalled toolchains
// no build here uses. The two heaviest runners (nccl, gpudirect-rdma) pull a
// ~10 GB CUDA devel image plus a golang stage plus an ubuntu runtime stage, and
// since the images are built for linux/amd64 AND linux/arm64 in one job, that is
// two full sets of layers. Both runners failed with ENOSPC before the reclaim
// step existed — and the failure is close to undiagnosable, because the runner
// cannot write its own logs once the disk is full, so the build reports
// BlobNotFound rather than the real error.
//
// Every declaration must reclaim the SAME paths. A difference is not a style
// question: publish-runner.yml is the workflow whose failure costs a
// hand-published image and a burned version tag, and it is the one nobody
// exercises on a PR.
//
// It compares ALL declarations rather than one per file because it has already
// been fooled once. The e2e job added its own reclaim step with a hand-copied
// four-path list against the canonical six, and the earlier form of this test —
// FindStringSubmatch, first match per file — compared the runner-image copy to
// publish-runner.yml and reported agreement while the third copy sat two
// hundred lines above it, silently reclaiming two directories fewer.
func TestEveryWorkflowReclaimsTheSameDisk(t *testing.T) {
	type decl struct {
		where string
		paths string
	}
	var declared []decl

	for _, wf := range []string{ciWorkflow, publishWorkflow} {
		src, err := os.ReadFile(wf)
		if err != nil {
			t.Fatalf("reading %s: %v", wf, err)
		}
		ms := reclaimPathsPattern.FindAllStringSubmatch(string(src), -1)
		if len(ms) == 0 {
			t.Fatalf("%s builds a runner image but declares no RECLAIM_PATHS. A hosted runner "+
				"cannot hold two platforms of the CUDA devel image alongside its preinstalled "+
				"toolchains, and the resulting ENOSPC surfaces as an unretrievable log rather "+
				"than as an error.", wf)
		}
		for i, m := range ms {
			if strings.TrimSpace(m[1]) == "" {
				t.Errorf("%s declaration %d is an empty RECLAIM_PATHS, which reclaims nothing.", wf, i+1)
				continue
			}
			declared = append(declared, decl{
				where: fmt.Sprintf("%s (declaration %d)", wf, i+1),
				paths: strings.Join(strings.Fields(m[1]), " "),
			})
		}
	}

	if len(declared) == 0 {
		return // every declaration was empty; already reported above.
	}
	for _, d := range declared[1:] {
		if d.paths != declared[0].paths {
			t.Errorf("disk-reclaim declarations disagree:\n  %s: %s\n  %s: %s\n"+
				"All copies must match. publish-runner.yml is the one no PR exercises, so a "+
				"divergence there is discovered by a failed publish and a burned version tag.",
				declared[0].where, declared[0].paths, d.where, d.paths)
		}
	}
}

// emptyArchArgPattern matches an ARG that declares a gencode with NO default,
// i.e. one whose value is resolved later in the build.
var emptyArchArgPattern = regexp.MustCompile(`(?m)^ARG\s+CUDA_ARCH=\s*$`)

// rawArchUsePattern matches an nvcc invocation that takes its gencode straight
// from the ARG rather than from the resolved value.
var rawArchUsePattern = regexp.MustCompile(`-arch=["']?\$\{?CUDA_ARCH`)

// TestNoRunnerCompilesAgainstAnUnresolvedArch stops a build flag from silently
// becoming empty.
//
// A runner whose gencode depends on TARGETARCH cannot express that as an ARG
// default, because an ARG default cannot branch. compute-smoke therefore declares
// `ARG CUDA_ARCH=` (empty) and resolves the real value inside the build. Any step
// that then refers to ${CUDA_ARCH} gets the EMPTY string, and `nvcc -arch=` with
// no argument fails — or worse, a flag that merely reads oddly compiles for the
// wrong part.
//
// This is written from a break rather than from theory. The per-platform default
// (#86) and the #else-branch guard (#96) were developed on separate branches.
// Each was green: #86 resolved the arch into a shell local and used it, #96 added
// two steps referring to ${CUDA_ARCH} back when that ARG still carried a real
// default. Neither diff was wrong on its own, and a clean textual merge said
// nothing, so the failure appeared for the first time on main with both present.
// A guard that reads the whole file is the only thing that could have seen it.
func TestNoRunnerCompilesAgainstAnUnresolvedArch(t *testing.T) {
	for _, d := range runnerDirs(t) {
		path := filepath.Join(d, "Dockerfile")
		src, err := os.ReadFile(path)
		if err != nil {
			continue // TestEveryRunnerDirectoryHasADockerfile owns that failure.
		}
		if !emptyArchArgPattern.Match(src) {
			continue // A real ARG default means ${CUDA_ARCH} is never empty.
		}
		if loc := rawArchUsePattern.FindIndex(src); loc != nil {
			line := 1 + strings.Count(string(src[:loc[0]]), "\n")
			t.Errorf("%s:%d passes -arch=${CUDA_ARCH}, but this Dockerfile declares "+
				"`ARG CUDA_ARCH=` with no default, so that expands to the empty string on every "+
				"build that does not override it. Read the value the build resolved "+
				"(/out/cuda_arch) instead.", path, line)
		}
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

// quotedLiteral finds double-quoted string literals. Every runner in this repo
// defines its skip marker as one, in Go, C++ and CUDA alike.
var quotedLiteral = regexp.MustCompile(`"([^"\\\n]|\\.)*"`)

// skipMarkerCandidate decides whether a string literal is MEANT to be a skip
// marker. It is deliberately looser than runner.SkipMarkerPattern — case
// insensitive, and indifferent to the separator — because a guard whose
// selector shares a grammar with its validator can only ever agree with itself.
// "Host_Health_SKIP" and "HOST-HEALTH_SKIP" are both selected here and both
// rejected by the parser, which is the whole point.
//
// A marker is either the bare token or the head of a "MARKER: reason" line,
// possibly with a printf format tail, so only the part before the first colon
// is considered.
//
// A leading "_" means the literal is a SUFFIX, not a whole marker: the soak
// family (thermal-soak, gpu-burn) composes its markers as markerPrefix+"_SKIP"
// in the shared soak_core.cuh. Those are resolved against the prefixes declared
// in the same runner directory rather than waved through, because a composed
// marker is exactly as capable of being unparseable as a literal one.
func skipMarkerCandidate(literal string) (string, bool) {
	head, _, _ := strings.Cut(literal, ":")
	head = strings.TrimSpace(strings.TrimSuffix(head, `\n`))
	if head == "" || strings.ContainsAny(head, " \t%") {
		return "", false
	}
	if !strings.HasSuffix(strings.ToUpper(head), "_SKIP") {
		return "", false
	}
	return head, true
}

// TestEverySkipMarkerIsRecognisedByTheParser closes the loop the skip
// declaration rule opened (issue #103).
//
// Exit 2 is not a code a runner has sole control of — an unrecovered Go panic
// exits 2, and so does every Go runtime fatal error — so pkg/runner honours it
// as Skip only when the runner also DECLARES the skip with a marker line, and
// records an undeclared exit 2 as Error. That rule has a failure mode of its
// own, and it is the mirror image: a runner whose marker the parser does not
// recognise has every one of its LEGITIMATE skips recorded as an Error
// instead. Both dispatchers would do it, on every node, silently.
//
// The sweep is over the filesystem rather than a hand-written list, because the
// list that matters is "every marker any runner can print" and only a directory
// listing knows that. It reads sources, not just Dockerfiles: a marker is a
// string constant in Go, C++ or CUDA.
func TestEverySkipMarkerIsRecognisedByTheParser(t *testing.T) {
	found := 0
	for _, d := range runnerDirs(t) {
		sources := runnerSources(t, d)
		prefixes := markerPrefixes(sources)
		for path, body := range sources {
			for _, quoted := range quotedLiteral.FindAllString(body, -1) {
				candidate, ok := skipMarkerCandidate(strings.Trim(quoted, `"`))
				if !ok {
					continue
				}
				markers := []string{candidate}
				if strings.HasPrefix(candidate, "_") {
					if len(prefixes) == 0 {
						t.Errorf("%s: composes a skip marker as prefix+%q, but no markerPrefix is declared "+
							"anywhere in runners/%s, so nothing can say what it prints", path, candidate, d)
						continue
					}
					markers = nil
					for _, p := range prefixes {
						markers = append(markers, p+candidate)
					}
				}
				for _, marker := range markers {
					found++
					// Proven through Parse, which is what both dispatchers
					// actually call — not against the pattern alone, so the
					// anchoring and the line scan are covered too.
					for _, line := range []string{marker, marker + ": a reason"} {
						if got := runner.Parse("", line+"\n", 2); got.Verdict != runner.VerdictSkip {
							t.Errorf("%s: skip marker %q is not honoured by the parser (Parse(%q, exit 2) = %q) — "+
								"every skip this runner declares would be recorded as an Error instead",
								path, marker, line, got.Verdict)
						}
					}
				}
			}
		}
	}
	// A sweep that silently matched nothing is not a guard. Every runner that
	// can skip prints a marker, so zero means the scan itself broke.
	if found == 0 {
		t.Fatal("no skip markers found in any runner source; the sweep is not reading what it thinks it is")
	}
}

// runnerSources reads every source file in one runner directory, keyed by path.
// Markers are string constants, so this covers Go, C++ and CUDA alike.
func runnerSources(t *testing.T, dir string) map[string]string {
	t.Helper()
	sourceExt := map[string]bool{".go": true, ".cu": true, ".cc": true, ".cuh": true, ".h": true, ".sh": true}
	entries, err := os.ReadDir(dir)
	if err != nil {
		t.Fatalf("reading runners/%s: %v", dir, err)
	}
	out := map[string]string{}
	for _, e := range entries {
		if e.IsDir() || !sourceExt[strings.ToLower(filepath.Ext(e.Name()))] {
			continue
		}
		path := filepath.Join(dir, e.Name())
		body, err := os.ReadFile(path)
		if err != nil {
			t.Fatalf("reading %s: %v", path, err)
		}
		out[path] = string(body)
	}
	return out
}

// markerPrefixes finds the marker prefixes a runner declares, for the composed
// case. The line carries the field name and the value is the first quoted
// literal on it, which covers both the C++ designated-comment idiom
// (/*markerPrefix=*/"GPU_BURN") and a plain assignment.
func markerPrefixes(sources map[string]string) []string {
	seen := map[string]bool{}
	var out []string
	for _, body := range sources {
		for _, line := range strings.Split(body, "\n") {
			if !strings.Contains(line, "markerPrefix") {
				continue
			}
			for _, quoted := range quotedLiteral.FindAllString(line, -1) {
				p := strings.Trim(quoted, `"`)
				// The printf that CONSUMES the prefix sits on a line naming it
				// too, so its format string has to be excluded. A marker prefix
				// is a bare token: no verbs, no escapes, no separators. The
				// filter stays deliberately shy of requiring upper case, so a
				// mis-cased prefix is still selected and still fails below.
				if p == "" || seen[p] || strings.ContainsAny(p, "%\\, \t") {
					continue
				}
				seen[p] = true
				out = append(out, p)
			}
		}
	}
	sort.Strings(out)
	return out
}
