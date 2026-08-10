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
//     cannot be built at all, and a pkg/runnerimages entry pointing at a tag
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
// Keyed by the source file that implements each pair, because the two guards
// below read the structure differently: the Dockerfile guard flattens it (no
// image may define ANY of them), while the announce guard visits each file and
// demands ITS macros and ITS marker — a flat list would have demanded
// compute-smoke's file mention gemm-sweep's switches.
var faultInjectionMacros = map[string][]string{
	filepath.Join("compute-smoke", "fp4_smoke.cu"): {
		"BURNIN_FP4_FORCE_CC_MAJOR",
		"BURNIN_FP4_FORCE_CC_MINOR",
	},
	filepath.Join("gemm-sweep", "gemm_sweep.cu"): {
		"BURNIN_GEMM_FORCE_CC_MAJOR",
		"BURNIN_GEMM_FORCE_CC_MINOR",
	},
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
		for _, macros := range faultInjectionMacros {
			for _, macro := range macros {
				if strings.Contains(string(src), macro) {
					t.Errorf("%s defines the fault-injection macro %s; an image built with it would "+
						"report a fabricated verdict for every node it ran on. Build such a binary by "+
						"hand for a one-off experiment, never from a Dockerfile that can be published",
						path, macro)
				}
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
	for path, macros := range faultInjectionMacros {
		src, err := os.ReadFile(path)
		if err != nil {
			t.Errorf("reading %s: %v", path, err)
			continue
		}
		text := string(src)
		for _, macro := range macros {
			if !strings.Contains(text, macro) {
				t.Errorf("%s no longer mentions %s; if the switch was removed, remove it from "+
					"faultInjectionMacros too rather than leaving a guard over nothing", path, macro)
			}
		}
		if !strings.Contains(text, `printf("forced_compute_cap=`) {
			t.Errorf("%s does not print a forced_compute_cap= marker; a fault-injected build "+
				"must announce itself in its own output, or its verdicts are indistinguishable "+
				"from real ones", path)
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
// pkg/runnerimages would point at a tag no workflow can produce, which
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

// skipMarkerLiteral finds anything SHAPED like a skip marker in a runner's
// source, CASE-INSENSITIVELY.
//
// The case-insensitivity is the entire point and the first version of this test
// lacked it. Matching only the correct spelling meant that renaming
// FP4_GEMM_SKIP to Fp4_gemm_skip made the pattern find nothing at all, the loop
// below iterate over nothing, and the test pass — blind in precisely the case it
// exists to catch. A guard that only recognises the correct form cannot report
// the incorrect one.
//
// The trailing [a-z]* catches a rename to _SKIPPED, which DeclaresSkip's \b also
// rejects, and which would otherwise slip past a pattern anchored at _SKIP.
//
// Tokens with no upper-case letter at all are then discarded, because that is
// what separates a marker from a metric: dcgm-diag legitimately reports
// tests_skipped, which is a measurement and not a declaration. Every real marker
// is upper-case, and every mis-spelling worth catching (Fp4_gemm_skip,
// FP4_GEMM_SKIPPED) keeps at least one capital.
var skipMarkerLiteral = regexp.MustCompile(`(?i)[A-Za-z][A-Za-z0-9_]*_skip[a-z]*`)

// looksLikeAMarker discards the all-lower-case matches, which are metric names
// rather than skip declarations.
func looksLikeAMarker(tok string) bool { return tok != strings.ToLower(tok) }

// A marker is not always spelled out in one piece. The soak family builds
// its own at runtime — soak_core.cuh does
//
//	emitMarker(k, "_SKIP", reason, kExitSkip)   // prints markerPrefix + "_SKIP"
//
// so the only literal in thermal-soak's and gpu-burn's sources is the bare
// suffix, which skipMarkerLiteral cannot match: that pattern requires a leading
// letter. Both directories therefore contributed NOTHING to the sweep below,
// which is the same blindness the pattern's own comment describes, reached by a
// different route — and both runners really do skip (no accelerator visible,
// MIG enabled). Renaming markerPrefix to "Thermal-Soak" would have turned every
// one of those skips into an Error with no test saying so.
//
// So the two halves are found separately and composed. Only QUOTED literals
// count, or a comment mentioning _SKIP would invent a marker for a runner that
// has none.
var (
	quotedLiteral      = regexp.MustCompile(`"([^"\\\n]|\\.)*"`)
	composedSkipSuffix = regexp.MustCompile(`(?i)^_skip[a-z]*$`)
)

// markerPrefixesIn reads the prefix half out of one source file.
//
// The value is the first quoted literal on a line naming the field, which
// covers the C++ designated-comment idiom (/*markerPrefix=*/"GPU_BURN") and a
// plain assignment alike. The printf that CONSUMES the prefix sits on a line
// naming it too, so its format string has to be excluded — a prefix is a bare
// token with no verbs, escapes or separators. That filter stops deliberately
// short of requiring upper case, so a mis-cased prefix is still collected and
// still fails the check.
func markerPrefixesIn(src string) []string {
	var out []string
	for _, line := range strings.Split(src, "\n") {
		if !strings.Contains(line, "markerPrefix") {
			continue
		}
		for _, quoted := range quotedLiteral.FindAllString(line, -1) {
			if p := strings.Trim(quoted, `"`); p != "" && !strings.ContainsAny(p, "%\\, \t") {
				out = append(out, p)
			}
		}
	}
	return out
}

// TestEverySkipMarkerIsOneTheParserRecognises ties each runner's skip
// declaration to the parser that has to believe it.
//
// Exit 2 alone is no longer enough to be recorded as Skip, because a Go panic —
// and every Go runtime fatal error, which no runner can recover from — also
// exits 2, and Skip is the one phase that is never retried and never affects the
// run's verdict (#103). A skip must now be DECLARED, and pkg/runner.DeclaresSkip
// is what reads the declaration.
//
// That makes a marker's exact spelling load-bearing in a way it never was
// before. Renaming FP4_GEMM_SKIP to FP4_GEMM_SKIPPED, or lower-casing one,
// would not fail any test the runner has: it would quietly turn every legitimate
// skip that runner reports into an Error. This is the check that notices, and it
// runs from the filesystem so a twelfth runner is covered the day it is added.
func TestEverySkipMarkerIsOneTheParserRecognises(t *testing.T) {
	for _, d := range runnerDirs(t) {
		found := map[string]bool{}
		var suffixes, prefixes []string
		err := filepath.WalkDir(d, func(path string, e os.DirEntry, err error) error {
			if err != nil || e.IsDir() {
				return err
			}
			switch filepath.Ext(path) {
			case ".go", ".cu", ".cuh", ".cc", ".h":
			default:
				return nil
			}
			if strings.HasSuffix(path, "_test.go") || strings.HasSuffix(path, "_test.cc") {
				return nil
			}
			src, readErr := os.ReadFile(path)
			if readErr != nil {
				return readErr
			}
			for _, m := range skipMarkerLiteral.FindAllString(string(src), -1) {
				if looksLikeAMarker(m) {
					found[m] = true
				}
			}
			for _, quoted := range quotedLiteral.FindAllString(string(src), -1) {
				if lit := strings.Trim(quoted, `"`); composedSkipSuffix.MatchString(lit) {
					suffixes = append(suffixes, lit)
				}
			}
			prefixes = append(prefixes, markerPrefixesIn(string(src))...)
			return nil
		})
		if err != nil {
			t.Errorf("walking runners/%s: %v", d, err)
			continue
		}
		// The composed half. A suffix with no prefix anywhere in the directory
		// is a failure in itself: the runner prints a marker at runtime that
		// nothing in the repository can predict, so nothing can check it.
		if len(suffixes) > 0 && len(prefixes) == 0 {
			t.Errorf("runners/%s composes its skip marker from a %q suffix, but declares no markerPrefix "+
				"anywhere in the directory — nothing can say what this runner actually prints, so nothing "+
				"can check that the parser will believe it", d, suffixes[0])
		}
		for _, s := range suffixes {
			for _, p := range prefixes {
				found[p+s] = true
			}
		}
		for m := range found {
			if !runner.DeclaresSkip(m + ": reason") {
				t.Errorf("runners/%s prints %q, which pkg/runner.DeclaresSkip does not recognise as a skip "+
					"declaration. Every skip this runner reports would be recorded as an Error instead — "+
					"hardware unjudged rather than out of scope. Keep the marker upper-case, ending in _SKIP, "+
					"and at the start of its line", d, m)
			}
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

// TestNoRunnerBinaryCanBeCommittedAtTheRepositoryRoot closes the hole that put a
// 3.4 MB compiled binary into a public repository twice.
//
// Every runner directory is `package main`, so `go build ./runners/<name>/` —
// and `go build ./...` — drops an executable called <name> at the REPOSITORY
// ROOT, not in the runner's directory. It is then untracked, unignored, and
// swept up by the next `git add -A`. That is how `ib-write-bw` (4.2 MB) got
// committed, and then `host-health` (3.4 MB) on top of it while fixing an
// unrelated bug.
//
// .gitignore already carries the shape of the answer and the reason it has to be
// anchored:
//
//	# Anchored to the repo root ON PURPOSE. An unanchored `manager` matches any
//	# path component with that name, which silently swallowed config/manager/ ...
//	/manager
//
// The same reasoning applies to every runner and to nothing else — an unanchored
// `nccl` would swallow runners/nccl/ entirely and the images would stop being
// built. So the entries must exist, and they must be anchored, and the list has
// to come from the FILESYSTEM rather than from anyone remembering to extend it:
// a new runner directory is a new binary name on the very first `go build ./...`
// after it lands.
func TestNoRunnerBinaryCanBeCommittedAtTheRepositoryRoot(t *testing.T) {
	raw, err := os.ReadFile("../.gitignore")
	if err != nil {
		t.Fatalf("reading .gitignore: %v", err)
	}
	ignored := map[string]bool{}
	for _, line := range strings.Split(string(raw), "\n") {
		if line = strings.TrimSpace(line); strings.HasPrefix(line, "/") {
			ignored[strings.TrimPrefix(line, "/")] = true
		}
	}

	for _, d := range runnerDirs(t) {
		if !ignored[d] {
			t.Errorf("`go build ./runners/%s/` writes an executable named %q at the repository "+
				"root, and .gitignore has no anchored `/%s` entry — the next `git add -A` commits "+
				"a multi-megabyte binary into a public repo", d, d, d)
		}
		// On disk is fine and expected; TRACKED is the failure. The binary is
		// only ever produced by a local build, so its presence in the worktree
		// says nothing, but a checkout that materialises it has it committed.
		if info, err := os.Stat(filepath.Join("..", d)); err == nil && !info.IsDir() {
			t.Logf("note: %s exists at the repository root (a local build artefact); "+
				"the /%s ignore rule is what keeps it out of a commit", d, d)
		}
	}
}

// TestDriverInjectionIsDeclaredWhereItIsNeeded makes the fleet's most load-bearing
// accident visible in the repository.
//
// Issue #52: the GB10 cluster's CDI spec declares a hook its host toolkit is too
// old to implement, so every pod that gets its GPU through CDI dies at
// createContainer. The burn-in runners work anyway — and the reason is a property
// of THESE IMAGES, not of the host: containerd's default_runtime_name is the
// legacy nvidia-container-runtime, which injects from NVIDIA_VISIBLE_DEVICES and
// NVIDIA_DRIVER_CAPABILITIES in the image environment and never consults the CDI
// spec at all.
//
// So the fleet is one Dockerfile away from a cluster-wide outage, with nothing in
// the cluster recording why. A new GPU runner that omits those two variables
// would be caught by nothing here: it builds, it passes every test in this
// package, and it fails on a real node as a StartError naming a hook rather than
// the runner.
//
// The rule is a CORRESPONDENCE and not "every runner must set them". Two runners
// legitimately need no accelerator — ib-write-bw measures the wire through RDMA
// verbs and memory-stress exercises host RAM — and declaring GPU access they do
// not use would be a privilege grant nobody reviewed. What must not happen is a
// runner needing the driver and not saying so, so the list of exemptions is
// written down here and a new runner has to join it deliberately.
func TestDriverInjectionIsDeclaredWhereItIsNeeded(t *testing.T) {
	// Runners that touch no accelerator. Adding a name here is a claim that the
	// runner never opens a CUDA context and never executes nvidia-smi; check the
	// source before making it.
	noAccelerator := map[string]string{
		"fabric-soak":       "iterates ib_write_bw over the wire through RDMA verbs; no accelerator, same as ib-write-bw",
		"ib-write-bw":       "measures the wire through RDMA verbs; the CUDA variant is gpudirect-rdma",
		"memory-stress":     "exercises HOST memory through stressapptest",
		"fingerprint-probe": "READS ABOUT accelerators over a read-only sysfs mount and never opens one; it must work on a node whose device plugin never came up",
		"disk-io":           "measures storage through direct I/O against a declared path; no accelerator is involved",
		"tcp-baseline":      "measures the kernel TCP stack through iperf3; it is the accelerator-free half of the fabric story and runs on nodes with no GPU at all",
	}

	for _, d := range runnerDirs(t) {
		raw, err := os.ReadFile(filepath.Join(d, "Dockerfile"))
		if err != nil {
			t.Fatalf("reading %s/Dockerfile: %v", d, err)
		}
		body := string(raw)
		visible := strings.Contains(body, "NVIDIA_VISIBLE_DEVICES")
		caps := strings.Contains(body, "NVIDIA_DRIVER_CAPABILITIES")

		if why, exempt := noAccelerator[d]; exempt {
			if visible || caps {
				t.Errorf("%s is listed as needing no accelerator (%s) but its Dockerfile declares "+
					"NVIDIA driver injection — either the exemption is stale or the image is asking "+
					"for access it does not use", d, why)
			}
			continue
		}

		if !visible || !caps {
			t.Errorf("%s/Dockerfile does not declare BOTH NVIDIA_VISIBLE_DEVICES and "+
				"NVIDIA_DRIVER_CAPABILITIES (visible=%v capabilities=%v). On a host where the "+
				"legacy nvidia-container-runtime is the injection path — which is the GB10 "+
				"cluster today, see issue #52 — this runner gets no driver at all. If it really "+
				"needs no accelerator, add it to noAccelerator above with the reason.",
				d, visible, caps)
		}
	}
}

// TestGroupCapableRunnersReallyReadTheGroupContract keeps the operator's
// allow-list and the runner sources from drifting apart.
//
// internal/controller/plan.go carries groupCapableKinds: the kinds whose SHIPPED
// image speaks BURNIN_RANK/BURNIN_NRANKS/BURNIN_ROOT_HOST rather than branching
// on BURNIN_ROLE. It decides whether the operator will dispatch a Group test to
// a default image, and it is a claim about a runner's SOURCE that lives in a
// different package from that source.
//
// Both directions are failures with teeth. A kind on the list whose runner does
// not read the variables means the operator dispatches an image that will exit 2
// with a skip marker — a collective certified as inapplicable, the exact false
// negative of #118. A kind off the list whose runner does read them means the
// operator refuses a test it could have run, and an author is sent to pin an
// image by hand for no reason.
func TestGroupCapableRunnersReallyReadTheGroupContract(t *testing.T) {
	// Mirrors internal/controller/plan.go's groupCapableKinds. It is restated
	// rather than imported because runners/ is deliberately not coupled to the
	// controller — and a restatement that drifts is what this test catches.
	groupCapable := map[string]bool{"nccl": true}

	// BURNIN_ROOT_HOST is the discriminator, and picking the right one mattered:
	// a first attempt keyed on BURNIN_RANK/BURNIN_NRANKS and failed, because
	// ib-write-bw and gpudirect-rdma read exactly those in order to REFUSE Group
	// scope. Reading a variable to say no is not implementing the contract.
	//
	// The root host is the address a rank fetches the bootstrap handle FROM. A
	// runner that only refuses has no use for it and never will, so its presence
	// in a non-test source is the narrowest available evidence that this image
	// actually joins a collective.
	const rootVar = "BURNIN_ROOT_HOST"

	for _, d := range runnerDirs(t) {
		var reads bool
		err := filepath.WalkDir(d, func(path string, e os.DirEntry, err error) error {
			if err != nil || e.IsDir() || strings.HasSuffix(path, "_test.go") {
				return err
			}
			// SOURCE ONLY. Scanning every file meant runners/nccl/README.md
			// documenting the variable satisfied this on its own — so ripping the
			// Group path out of main.go while leaving the docs behind would keep
			// the test green and leave groupCapableKinds asserting something false
			// about the shipped image, which is the direction this test's own
			// error message calls a failure with teeth.
			switch filepath.Ext(path) {
			case ".go", ".c", ".cc", ".cu", ".h", ".cuh":
			default:
				return nil
			}
			body, rerr := os.ReadFile(path)
			if rerr != nil {
				return rerr
			}
			if strings.Contains(string(body), rootVar) {
				reads = true
			}
			return nil
		})
		if err != nil {
			t.Fatalf("walking %s: %v", d, err)
		}

		switch {
		case groupCapable[d] && !reads:
			t.Errorf("%s is on the operator's groupCapableKinds list, but no non-test source in "+
				"runners/%s reads %s. The operator will dispatch this image for a Group test and "+
				"it will read the absent BURNIN_ROLE as Node scope: exit 2 with a skip marker, "+
				"and a collective certified as inapplicable (#118).", d, d, rootVar)
		case !groupCapable[d] && reads:
			t.Errorf("runners/%s reads %s but is NOT on the operator's groupCapableKinds list, so "+
				"a Group test naming this kind is refused at plan time even though the image "+
				"could run it. Add it to groupCapableKinds in internal/controller/plan.go and to "+
				"the mirror in this test.", d, rootVar)
		}
	}
}

// TestAPinRefusalDiagnosesTheAnnotatedTagTrap holds every pin assertion to the
// same shape.
//
// The assertion itself has always worked — it caught a bad pin on tcp-baseline
// and, before that, on dcgm-diag. What it did not do is say WHY, and the reason
// has now been the same one twice: an annotated tag has two SHAs, `ls-remote`
// answers with the tag OBJECT, and `git rev-parse HEAD` after a shallow clone
// is the COMMIT. Record the unpeeled value and every build fails with a message
// that looks like the upstream moved.
//
// A shallow clone retains the tag object, so the build can tell the two apart
// and print the right one. This asserts every runner does.
//
// It is a guard rather than a shared snippet because each runner is its own
// Docker build context and COPY cannot reach outside it — the same reason
// soak_core.cuh and the fabric sources are physically duplicated. Duplication
// that cannot be removed is duplication that has to be checked.
//
// The diagnosis has a silent failure mode of its own, and the sweep checks for
// it: both of its git commands must run INSIDE the clone. The pin assertion
// addresses the repository explicitly (`git -C <dir>`, or a WORKDIR set to the
// clone), but a diagnosis pasted without `cd <clonedir> &&` runs wherever the
// RUN step happens to be — against no repository `git cat-file` errors,
// `2>/dev/null` eats the complaint, the `= "tag"` test is false, and the one
// message the block exists to print is the one that never appears: the exact
// silence #238 was filed about, reintroduced inside its own fix. Three of the
// eight copies shipped that way, distinguishable from the five working ones
// only by whether a WORKDIR happened to sit in the clone. So the shape is
// uniform and checked: every diagnosis command cd's into a directory this
// Dockerfile actually `git clone`s into, which keeps the snippet correct
// independent of whatever WORKDIR is in effect when a later edit moves it.
func TestAPinRefusalDiagnosesTheAnnotatedTagTrap(t *testing.T) {
	checked := 0
	for _, d := range runnerDirs(t) {
		raw, err := os.ReadFile(filepath.Join(d, "Dockerfile"))
		if err != nil {
			t.Fatalf("reading %s/Dockerfile: %v", d, err)
		}
		body := string(raw)
		if !strings.Contains(body, "refusing to build:") || !strings.Contains(body, "_SHA}") {
			// No upstream pinned here; nothing to diagnose.
			continue
		}
		checked++

		if !strings.Contains(body, `git cat-file -t`) {
			t.Errorf("%s asserts a pinned SHA but does not diagnose the annotated-tag trap. "+
				"A shallow clone retains the tag object, so the refusal can tell the tag object from "+
				"the commit and print the peeled value; without it the message reads as though the "+
				"upstream moved. See issue #238", d)
			continue
		}
		// The diagnosis is only useful if it hands over the value to use.
		if !strings.Contains(body, `^{commit}`) {
			t.Errorf("%s detects the tag object but never prints the peeled commit, which is the "+
				"whole point — the reader is left to work out the right value themselves", d)
		}

		dests := cloneDestinations(body)
		if len(dests) == 0 {
			t.Errorf("%s asserts a pinned SHA but no `git clone` destination could be parsed from "+
				"its Dockerfile, so this test cannot tell whether the diagnosis runs inside the "+
				"clone. If the clone moved into a form this parser does not recognise, teach "+
				"cloneDestinations the new shape rather than skipping the check.", d)
			continue
		}
		for i, line := range strings.Split(body, "\n") {
			var re *regexp.Regexp
			var what string
			switch {
			case strings.Contains(line, "git cat-file -t"):
				re, what = catFileInClone, "the tag-object probe (git cat-file -t)"
			case strings.Contains(line, "git rev-parse") && strings.Contains(line, "^{commit}"):
				re, what = peelInClone, "the peel (git rev-parse ...^{commit})"
			default:
				continue
			}
			m := re.FindStringSubmatch(line)
			if m == nil {
				t.Errorf("%s/Dockerfile:%d: %s has no `cd <clonedir> && ` in front of it, so it "+
					"runs wherever the RUN step happens to be — outside a repository git errors, "+
					"2>/dev/null eats the complaint, the tag test is false, and the diagnosis "+
					"silently does not print: the silence of #238, inside its own fix. Use the "+
					"$(cd <clonedir> && git ...) form.", d, i+1, what)
				continue
			}
			if !dests[m[1]] {
				t.Errorf("%s/Dockerfile:%d: %s cd's into %q, which is not a directory this "+
					"Dockerfile clones into %v. A cd into the wrong directory fails exactly like "+
					"no cd at all: silently.", d, i+1, what, m[1], sortedKeys(dests))
			}
		}
	}

	// Asserted rather than assumed: a refactor that renamed the refusal message
	// would empty this sweep and leave it passing while checking nothing.
	if checked == 0 {
		t.Fatal("found no runner asserting a pinned SHA, which cannot be right")
	}
}

// The two commands of the annotated-tag diagnosis, each required to enter the
// clone explicitly. The `cd` form is deliberate over `git -C`: it keeps the
// literal `git cat-file -t` intact for the presence check above, and it reads
// as what it is — a command that must be somewhere in particular to work.
var (
	catFileInClone = regexp.MustCompile(`cd\s+(\S+)\s+&&\s+git cat-file -t`)
	peelInClone    = regexp.MustCompile(`cd\s+(\S+)\s+&&\s+git rev-parse`)
)

// cloneDestinations parses the directory argument of every `git clone` in a
// Dockerfile body. It is how TestAPinRefusalDiagnosesTheAnnotatedTagTrap knows
// which directories actually contain the repository a diagnosis could run in —
// a `cd` into any other directory fails the same way as no `cd` at all.
func cloneDestinations(body string) map[string]bool {
	dests := map[string]bool{}
	for _, line := range strings.Split(body, "\n") {
		if !strings.Contains(line, "git clone") {
			continue
		}
		// A clone line may end in a shell continuation (`; \`); the destination
		// is the last real token.
		line = strings.TrimSpace(strings.TrimSuffix(strings.TrimSpace(line), `\`))
		line = strings.TrimSuffix(line, ";")
		fields := strings.Fields(line)
		if len(fields) < 2 {
			continue
		}
		dests[strings.Trim(fields[len(fields)-1], `"'`)] = true
	}
	return dests
}

func sortedKeys(m map[string]bool) []string {
	out := make([]string, 0, len(m))
	for k := range m {
		out = append(out, k)
	}
	sort.Strings(out)
	return out
}

// A PAIR SERVER MUST NOT REQUIRE BURNIN_PEER_HOST — issue #287.
//
// Every fabric runner learns its peer differently depending on which end it is.
// The CLIENT dials BURNIN_PEER_HOST. The SERVER cannot use it and never reads
// it: it starts first, and in-cluster its peer's DNS name does not resolve until
// the operator creates the client, which the operator will not do until the
// server is Ready. That is a deadlock, and the way out — documented at length in
// ib-write-bw's rendezvous.go — is that the server takes its peer's address from
// the accepted connection.
//
// Four runners nonetheless REQUIRED the variable for both roles, which turned it
// into a presence token. In-cluster that was invisible, because the operator
// sets it for both roles. On bare metal the CLI correctly declines to invent a
// peer the server cannot use, so `burnin run --role server` errored before
// binding and NO Pair test could run outside Kubernetes at all.
//
// This asserts the SHAPE rather than the message: a guard on an empty peer host
// must be qualified by the client role. tcp-baseline always had it right.
func TestPairServerDoesNotRequireAPeerHost(t *testing.T) {
	// Runners that speak the Pair rendezvous. A runner that refuses Pair
	// entirely has no peer host to guard and is not listed.
	pairRunners := []string{"ib-write-bw", "gpudirect-rdma", "nccl", "fabric-soak", "tcp-baseline"}

	checked := 0
	for _, r := range pairRunners {
		path := filepath.Join(r, "main.go")
		src, err := os.ReadFile(path)
		if err != nil {
			t.Errorf("%s: %v", path, err)
			continue
		}
		for i, line := range strings.Split(string(src), "\n") {
			trimmed := strings.TrimSpace(line)
			if !strings.HasPrefix(trimmed, "if ") || !strings.Contains(trimmed, `peerHost == ""`) {
				continue
			}
			checked++
			if !strings.Contains(trimmed, "RoleClient") && !strings.Contains(trimmed, "roleClient") {
				t.Errorf("%s:%d guards an empty BURNIN_PEER_HOST without qualifying it by the CLIENT "+
					"role:\n    %s\nThe server never reads that variable — it learns its peer from the "+
					"accepted connection — so requiring it makes the bare-metal dispatcher unable to "+
					"start a Pair server at all (#287).", path, i+1, trimmed)
			}
		}
	}
	// A rename or a shortened list would make this sweep pass while checking
	// nothing, which is the failure mode every guard in this file is written
	// against. The floor is a LITERAL and not len(pairRunners): comparing the
	// count against the list's own length is vacuous the moment the list is
	// emptied, which is exactly how this assertion first failed to bite.
	const minGuards = 5
	if checked < minGuards {
		t.Fatalf("found %d peer-host guards across %d Pair runners, want at least %d — the sweep is "+
			"not reading what it thinks it is", checked, len(pairRunners), minGuards)
	}
}
