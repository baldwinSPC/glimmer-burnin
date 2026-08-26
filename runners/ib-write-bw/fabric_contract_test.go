// SPDX-License-Identifier: Apache-2.0
// Copyright the Glimmer authors.

// Drift guard for the scaffolding the four fabric runners share.
//
// runners/ib-write-bw, runners/gpudirect-rdma, runners/nccl and
// runners/fabric-soak are four separate Docker build contexts. The publish
// workflow builds each with its OWN directory as the context and COPY cannot
// reach outside a build context, so every file they have in common is a COPY,
// not a symlink and not a shared package. That is a deliberate trade — but a
// copy nobody compares is a copy that drifts, and the drift is invisible in
// review because the two files are never in the same diff.
//
// fabric-soak joined this guard in #494, after shipping for a while with its
// own undeclared copies of rdma.go, memlock.go and (once #495 added it)
// rendezvous.go — exactly the kind of drift this file exists to catch, caught
// instead by a human reading #489's investigation closely enough to check by
// hand. Nothing was actually wrong at the time; nothing was watching either.
//
// The stakes are not tidiness. memlock.go carries the containerd RLIMIT_MEMLOCK
// finding: a container started by containerd inherits the runtime's
// LimitMEMLOCK (8 MiB on a systemd distribution) rather than the host's, which
// makes ibv_reg_mr/ibv_create_cq fail with a message that names neither the
// limit nor the resource and reads exactly like a dead HCA. A copy that lost
// that handling would not fail this repo's tests; it would fail on a customer's
// fleet, as a fabric verdict against healthy hardware.
//
// So this file states, for EVERY filename that appears in more than one fabric
// runner directory, whether the copies must be byte-identical. The table is
// total by construction: a new duplicated file fails the completeness test until
// somebody decides which it is.
//
// This is the fabric counterpart to TestSharedSoakSourcesAreIdentical in
// runners/thermal-soak/soak_contract_test.go, which does the same job for the
// two CUDA soak runners. It lives here, in the reference fabric runner, for the
// same reason that one lives in thermal-soak: ib-write-bw owns the canonical
// copy of the shared sources, and a guard belongs with the thing it protects.
package main

import (
	"os"
	"path/filepath"
	"sort"
	"strings"
	"testing"
)

// The four fabric runner directories, relative to this package.
const (
	ibWriteBWDir  = "."
	gpudirectDir  = "../gpudirect-rdma"
	ncclDir       = "../nccl"
	fabricSoakDir = "../fabric-soak"
)

var fabricRunnerDirs = []string{ibWriteBWDir, gpudirectDir, ncclDir, fabricSoakDir}

// fabricFile is one filename that exists in more than one fabric runner.
//
// sharedBy names the directories whose copies must be byte-identical. It may
// list fewer than all three: a file can be genuinely shared between two runners
// and deliberately forked in the third, which is exactly the situation nccl is
// in. reason records why the runners that are NOT in sharedBy diverge, so the
// next person to touch one of these files inherits the decision rather than
// re-deriving it from a diff.
type fabricFile struct {
	name     string
	sharedBy []string
	reason   string
}

// fabricFiles must name every file present in two or more of the directories in
// fabricRunnerDirs. TestFabricFileTableIsComplete enforces that.
var fabricFiles = []fabricFile{
	{
		name:     "groupscope_test.go",
		sharedBy: []string{ibWriteBWDir, gpudirectDir},
		reason: "the Group-scope guard, and nccl is a DELIBERATE FORK of it. All three runners once " +
			"read an absent BURNIN_ROLE as a Node-scope run and declared a SKIP for a collective " +
			"that never executed (#118); ib-write-bw and gpudirect-rdma still refuse Group scope " +
			"outright, because a point-to-point RDMA write and a GPUDirect peer-memory check are " +
			"measurements of a LINK and have no N-rank form — so their guard is one rule and stays " +
			"one file. nccl now speaks the Group rendezvous, so its copy asserts the opposite: that " +
			"a well-formed BURNIN_RANK/BURNIN_NRANKS/BURNIN_ROOT_HOST contract is ACCEPTED and a " +
			"malformed one is an Error. The two files must not be resynced — doing so would either " +
			"make nccl refuse a collective it can run, or make the link runners claim one they " +
			"cannot.",
	},
	{
		name:     "zz_runnerlib.go",
		sharedBy: []string{ibWriteBWDir, gpudirectDir, ncclDir, fabricSoakDir},
		reason: "the repo-wide runner helper stamp, generated from hack/runnerlib/runnerlib.go.src " +
			"by `go run ./hack/runnerlib`. runners/runnerlib_test.go already holds every copy " +
			"byte-identical to the source across ALL Go runners; this entry exists because this " +
			"table also enumerates shared files, and a file tracked by two guards must say so in " +
			"both or the next reader assumes one of them is stale.",
	},
	{
		name:     "rdma.go",
		sharedBy: []string{ibWriteBWDir, gpudirectDir, fabricSoakDir},
		reason: "device discovery, single-port selection and GID resolution are one implementation, " +
			"shared because ib-write-bw, gpudirect-rdma and fabric-soak all answer the same " +
			"single-link question — which ONE local RDMA port reaches this peer — and a fork would " +
			"let one runner SKIP a node the others test, which reads as a healthy node with no " +
			"fabric rather than as a runner that stopped looking. nccl's copy is a DELIBERATE fork " +
			"as of #489: NCCL is not point-to-point, its transport natively stripes a communicator " +
			"across every HCA named in NCCL_IB_HCA, so 'which ONE port' is the wrong question for it " +
			"on a node wired with more than one usable rail — measured on this fleet as a single " +
			"rail's ~12 GB/s reported in place of the ~22.6 GB/s both rails together sustain. nccl's " +
			"fork ADDS selectRailPorts/classifyRails/agreeingGIDIndex (local-topology rail " +
			"classification with no per-HCA GID-index syntax to lean on) alongside the same " +
			"discoverPorts/selectPort/resolveGID the other three runners still use unchanged — a " +
			"node with no qualifying multi-rail subnet falls straight through to the identical " +
			"single-device answer this file has always given.",
	},
	{
		name:     "perftest.go",
		sharedBy: []string{ibWriteBWDir, gpudirectDir, fabricSoakDir},
		reason: "the perftest output parser. nccl does not run perftest at all — it has results.go " +
			"for nccl_pair's own output — so there is no fourth copy to keep in step. fabric-soak " +
			"joined this table in #494; its copy was already byte-identical to ib-write-bw's, " +
			"undeclared until then.",
	},
	{
		name:     "assert-no-copyleft.sh",
		sharedBy: []string{ibWriteBWDir, gpudirectDir, fabricSoakDir},
		reason: "the build-time assertion that the perftest tree carries no GPL-only dependency. " +
			"The three perftest-based runners all fetch that tree; nccl builds NCCL, which is " +
			"BSD-3-Clause and asserted separately in its own Dockerfile.",
	},
	{
		name:     "rendezvous.go",
		sharedBy: []string{ibWriteBWDir, gpudirectDir},
		reason: "the server/client control channel. nccl's copy is a DELIBERATE fork: perftest runs " +
			"two phases over the link (bandwidth and latency) and nccl_pair runs exactly one " +
			"(a collective sweep inside a single communicator, because rebuilding the communicator " +
			"per message size would measure NCCL's initialisation as often as its bandwidth). " +
			"The phase vocabulary is part of the wire protocol, so it cannot be shared. fabric-soak's " +
			"copy (#495) is ALSO a deliberate fork, for a third reason: it needs none of the phase " +
			"negotiation either kind above carries, because every window is its own independent " +
			"ib_write_bw invocation the wrapper already orchestrates — the channel exists purely to " +
			"let the server learn its peer's address from the accepted connection, one hello, nothing " +
			"else. Three different reasons to diverge is not a coincidence to resolve; it is three " +
			"runners with three different relationships to the link they measure.",
	},
	{
		name:     "memlock.go",
		sharedBy: []string{ibWriteBWDir, gpudirectDir, fabricSoakDir},
		reason: "RLIMIT_MEMLOCK handling. nccl's copy is a DELIBERATE fork, and the divergence is a " +
			"measured fact rather than a style choice: the perftest runners own every byte they " +
			"register, so they SIZE THEMSELVES TO FIT the limit and still measure the link; NCCL's " +
			"registration is internal and no NCCL_BUFFSIZE or NCCL_MAX_NCHANNELS setting makes it " +
			"fit, so its copy REFUSES with an Error naming the two cluster-side remedies. fabric-soak " +
			"sizes itself to fit exactly like the perftest runners it shares this copy with. Every " +
			"copy must still carry the containerd lesson — see " +
			"TestEveryFabricMemlockKeepsTheContainerdLesson.",
	},
	{
		name:     "memlock_test.go",
		sharedBy: []string{ibWriteBWDir, gpudirectDir},
		reason:   "the tests for the above, and forked with it.",
	},
	{
		name: "rdma_test.go",
		reason: "ib-write-bw's copy tests the shared discoverPorts/selectPort/resolveGID " +
			"functions rdma.go's other entry still covers across three runners. nccl's copy " +
			"(#489) tests only what its own fork adds — classifyRails, agreeingGIDIndex — and " +
			"does not duplicate the shared-function coverage, so the two files test disjoint " +
			"code and were never going to be identical.",
	},
	{
		name: "main.go",
		reason: "each runner's own entrypoint: what it launches, which metrics it emits and how it " +
			"decides a verdict is the whole substance of a runner.",
	},
	{
		name: "metricnames_test.go",
		reason: "each runner's own metric vocabulary, checked against pkg/runner's alias table for " +
			"that TestKind.",
	},
	{
		name:   "Dockerfile",
		reason: "each runner's own image: different base, different upstream, different pins.",
	},
	{
		name:   "README.md",
		reason: "each runner's own operator documentation.",
	},
}

// TestFabricFileTableIsComplete is what makes the rest of this file a guarantee
// rather than a spot check.
//
// Without it, the guard covers whatever somebody remembered to list on the day
// they wrote it: a twelfth shared file could be copied into all three runners
// and drift forever, because nothing would be comparing it. So the table is
// checked against the filesystem in both directions — an undeclared duplicate
// fails, and so does a table entry that no longer describes two real files.
func TestFabricFileTableIsComplete(t *testing.T) {
	declared := map[string]bool{}
	for _, f := range fabricFiles {
		if declared[f.name] {
			t.Errorf("%q is listed twice in fabricFiles", f.name)
		}
		declared[f.name] = true
	}

	for name, dirs := range fabricDuplicates(t) {
		if !declared[name] {
			t.Errorf("%q exists in %v but fabricFiles says nothing about it. Add an entry: either "+
				"sharedBy (the copies must be byte-identical) or a reason the runners diverge. "+
				"An undeclared duplicate is a file that can drift with nobody comparing it.",
				name, dirs)
		}
	}

	dupes := fabricDuplicates(t)
	for _, f := range fabricFiles {
		if _, ok := dupes[f.name]; !ok {
			t.Errorf("fabricFiles declares %q, but it no longer exists in two or more of %v; "+
				"drop the entry", f.name, fabricRunnerDirs)
		}
	}
}

// TestSharedFabricSourcesAreIdentical is the guard itself.
//
// Both halves matter. A file declared shared must MATCH in every directory that
// shares it — that is the drift the containerd lesson could be lost to. And a
// file the table says is forked must actually DIFFER in the runners that fork
// it: a well-meaning "let's resync these" that copied ib-write-bw's memlock.go
// over nccl's would compile, pass every other test, and replace nccl's honest
// Error with a fit-to-budget path that cannot work for NCCL — a runner that
// shrinks buffers it does not own and then reports a number.
func TestSharedFabricSourcesAreIdentical(t *testing.T) {
	for _, f := range fabricFiles {
		if len(f.sharedBy) == 0 {
			assertNoTwoCopiesAgree(t, f)
			continue
		}

		ref := f.sharedBy[0]
		want, err := os.ReadFile(filepath.Join(ref, f.name))
		if err != nil {
			t.Errorf("fabricFiles says %s/%s is shared, but it cannot be read: %v", ref, f.name, err)
			continue
		}

		for _, dir := range f.sharedBy[1:] {
			got, err := os.ReadFile(filepath.Join(dir, f.name))
			if err != nil {
				t.Errorf("fabricFiles says %s/%s is shared, but it cannot be read: %v", dir, f.name, err)
				continue
			}
			if string(got) != string(want) {
				t.Errorf("%s has DRIFTED between %s and %s (%d vs %d bytes).\n"+
					"These are copies of one source; the publish workflow builds each runner with its own "+
					"directory as the Docker build context, so COPY cannot reach a shared path. "+
					"Copy one over the other, or — if the divergence is deliberate — move %s out of "+
					"sharedBy in fabricFiles and record why.\nWhy this file is shared: %s",
					f.name, ref, dir, len(want), len(got), dir, f.reason)
			}
		}

		// The forked runners must really be forked.
		for _, dir := range fabricRunnerDirs {
			if contains(f.sharedBy, dir) {
				continue
			}
			got, err := os.ReadFile(filepath.Join(dir, f.name))
			if err != nil {
				continue // it simply does not have the file, which is fine.
			}
			if string(got) == string(want) {
				t.Errorf("%s/%s is now byte-identical to the shared copy in %s, but fabricFiles has it "+
					"outside sharedBy. Either the fork was overwritten by a copy-paste — which is the "+
					"bug this check exists to catch — or the fork is genuinely over and %s should join "+
					"sharedBy.\nWhy it was forked: %s", dir, f.name, ref, dir, f.reason)
			}
		}
	}
}

// assertNoTwoCopiesAgree covers the entries with no sharedBy at all — the files
// that are per-runner by nature. Two of them becoming identical is not
// automatically wrong, but it does mean a duplicate appeared that nothing is
// keeping in step, which is the state this whole file exists to make visible.
func assertNoTwoCopiesAgree(t *testing.T, f fabricFile) {
	t.Helper()
	seen := map[string]string{} // content -> directory that had it first
	for _, dir := range fabricRunnerDirs {
		b, err := os.ReadFile(filepath.Join(dir, f.name))
		if err != nil {
			continue
		}
		if first, dup := seen[string(b)]; dup {
			t.Errorf("%s is byte-identical in %s and %s, but fabricFiles lists it as per-runner. "+
				"If these copies must now agree, give it a sharedBy so drift is caught; if the match "+
				"is a coincidence, they are still two copies nothing compares.\nRecorded reason: %s",
				f.name, first, dir, f.reason)
			continue
		}
		seen[string(b)] = dir
	}
}

// TestEveryFabricMemlockKeepsTheContainerdLesson guards the SUBSTANCE that
// byte-comparison cannot reach.
//
// nccl's memlock.go is a deliberate fork, so the identity check above says
// nothing about it — and it is the copy carrying the most expensive finding in
// this directory. These assertions are deliberately about what the file must
// still be able to TELL AN OPERATOR: the limit's name, and the containerd
// remedy. A rewrite that keeps the mechanism but drops the message leaves the
// next person with "Couldn't create CQ" and a bisect, which is precisely the
// cost this file was written to stop paying.
func TestEveryFabricMemlockKeepsTheContainerdLesson(t *testing.T) {
	required := []struct {
		substr string
		why    string
	}{
		{"RLIMIT_MEMLOCK", "the limit must be named; no message from the verbs stack ever names it"},
		{"LimitMEMLOCK=infinity", "the containerd drop-in is the actual remedy, and it is not guessable"},
		{"const rlimitMemlock = 8", "the RLIMIT_MEMLOCK constant is not exported by syscall and must not be re-derived"},
		{"const memlockUnlimited = ^uint64(0)", "RLIM_INFINITY as it appears in a Rlimit"},
	}

	for _, dir := range fabricRunnerDirs {
		src, err := os.ReadFile(filepath.Join(dir, "memlock.go"))
		if err != nil {
			t.Errorf("%s has no memlock.go: every fabric runner pins memory and so must handle the "+
				"limit that caps it (%v)", dir, err)
			continue
		}
		for _, r := range required {
			if !strings.Contains(string(src), r.substr) {
				t.Errorf("%s/memlock.go no longer contains %q — %s", dir, r.substr, r.why)
			}
		}
	}
}

// fabricDuplicates returns every filename present in two or more fabric runner
// directories, mapped to the directories that have it.
func fabricDuplicates(t *testing.T) map[string][]string {
	t.Helper()
	byName := map[string][]string{}
	for _, dir := range fabricRunnerDirs {
		entries, err := os.ReadDir(dir)
		if err != nil {
			t.Fatalf("reading %s: %v", dir, err)
		}
		for _, e := range entries {
			if e.IsDir() {
				continue
			}
			byName[e.Name()] = append(byName[e.Name()], dir)
		}
	}
	for name, dirs := range byName {
		if len(dirs) < 2 {
			delete(byName, name)
		} else {
			sort.Strings(dirs)
		}
	}
	return byName
}

func contains(haystack []string, needle string) bool {
	for _, s := range haystack {
		if s == needle {
			return true
		}
	}
	return false
}
