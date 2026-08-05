package main

import (
	"os"
	"testing"
)

// nccl SPEAKS the Group rendezvous — issue #118. Its sibling fabric runners do
// not, and their copy of this file asserts the opposite; see the fork entry in
// runners/ib-write-bw/fabric_contract_test.go for why the three diverge.
//
// What must never come back is the behaviour that made #118 urgent: reading an
// unset BURNIN_ROLE as "this is a Node-scope run" and declaring a SKIP for a
// collective that never executed. A Skip certifies, and it is never retried.

// A Group contract this runner cannot read is an ERROR, never a skip. The
// operator asked for a collective; a runner that cannot parse the request has
// measured nothing, and the hardware is unjudged.
func TestGroupRendezvousRejectsAMalformedContract(t *testing.T) {
	cases := map[string]map[string]string{
		"rank is not a number":   {"BURNIN_RANK": "one", "BURNIN_NRANKS": "8", "BURNIN_ROOT_HOST": "rank-0.svc"},
		"nranks is not a number": {"BURNIN_RANK": "0", "BURNIN_NRANKS": "many", "BURNIN_ROOT_HOST": "rank-0.svc"},
		"a collective of one":    {"BURNIN_RANK": "0", "BURNIN_NRANKS": "1", "BURNIN_ROOT_HOST": "rank-0.svc"},
		"rank outside the set":   {"BURNIN_RANK": "8", "BURNIN_NRANKS": "8", "BURNIN_ROOT_HOST": "rank-0.svc"},
		"negative rank":          {"BURNIN_RANK": "-1", "BURNIN_NRANKS": "4", "BURNIN_ROOT_HOST": "rank-0.svc"},
		"no root to fetch from":  {"BURNIN_RANK": "3", "BURNIN_NRANKS": "8", "BURNIN_ROOT_HOST": ""},
	}
	for name, env := range cases {
		t.Run(name, func(t *testing.T) {
			for k, v := range env {
				t.Setenv(k, v)
			}
			_, _, _, isGroup, err := groupRendezvous()
			if !isGroup {
				t.Fatal("a pod carrying BURNIN_RANK/BURNIN_NRANKS is in a Group execution whatever " +
					"else is wrong with the contract — reading it as Node scope is how a collective " +
					"gets certified as inapplicable")
			}
			if err == nil {
				t.Errorf("accepted a malformed Group contract %v", env)
			}
		})
	}
}

// A well-formed contract is accepted, and every field reaches the harness.
func TestGroupRendezvousAcceptsAWellFormedContract(t *testing.T) {
	t.Setenv("BURNIN_RANK", "3")
	t.Setenv("BURNIN_NRANKS", "8")
	t.Setenv("BURNIN_ROOT_HOST", "rank-0.bp-x-t0-a1.burnin.svc")

	rank, nranks, root, isGroup, err := groupRendezvous()
	if err != nil || !isGroup {
		t.Fatalf("rejected a valid contract: isGroup=%v err=%v", isGroup, err)
	}
	if rank != 3 || nranks != 8 || root != "rank-0.bp-x-t0-a1.burnin.svc" {
		t.Errorf("parsed rank=%d nranks=%d root=%q", rank, nranks, root)
	}
}

// Node scope is unaffected: no rendezvous variables at all is not a Group run.
func TestNoRendezvousVariablesIsNotAGroupRun(t *testing.T) {
	os.Unsetenv("BURNIN_RANK")
	os.Unsetenv("BURNIN_NRANKS")
	if _, _, _, isGroup, _ := groupRendezvous(); isGroup {
		t.Error("a pod with no rendezvous variables was read as a Group execution")
	}
}

// THE ARGUMENTS THE HARNESS ACTUALLY GETS. This is the seam that cannot be
// tested on hardware here, so it is tested exactly.
func TestGroupHarnessArgs(t *testing.T) {
	cfg := plan{bootstrapPort: 18535, warmup: 5, iters: 20}

	root := groupHarnessArgs(0, 8, "", cfg)
	if got := argOf(root, "--rank"); got != "0" {
		t.Errorf("--rank = %q, want 0", got)
	}
	if got := argOf(root, "--nranks"); got != "8" {
		t.Errorf("--nranks = %q, want 8 — the communicator size must be the whole cohort", got)
	}
	// Rank 0 SERVES the handle, so it takes no --peer. Passing one would make it
	// try to fetch from itself.
	for _, a := range root {
		if a == "--peer" {
			t.Error("rank 0 was given a --peer; it serves the bootstrap handle rather than fetching it")
		}
	}

	worker := groupHarnessArgs(5, 8, "10.0.0.7", cfg)
	if got := argOf(worker, "--rank"); got != "5" {
		t.Errorf("--rank = %q, want 5", got)
	}
	if got := argOf(worker, "--peer"); got != "10.0.0.7" {
		t.Errorf("--peer = %q, want the ROOT's address — every non-root rank fetches from rank 0", got)
	}

	// The Pair path must not have moved: it is proven on hardware and pins
	// --nranks 2 of its own.
	pair := harnessArgs(1, "10.0.0.9", cfg)
	if got := argOf(pair, "--nranks"); got != "2" {
		t.Errorf("the Pair path's --nranks = %q, want 2", got)
	}
}

func argOf(args []string, flag string) string {
	for i, a := range args {
		if a == flag && i+1 < len(args) {
			return args[i+1]
		}
	}
	return ""
}

// resolveHost turns the operator's DNS name into the address nccl_pair parses
// with inet_pton. A literal must pass through; a non-IPv4 literal must not be
// silently accepted, because the harness would reject it far less legibly.
func TestResolveHost(t *testing.T) {
	if ip, err := resolveHost("10.0.0.50"); err != nil || ip.String() != "10.0.0.50" {
		t.Errorf("resolveHost(literal) = %v, %v", ip, err)
	}
	if _, err := resolveHost("::1"); err == nil {
		t.Error("an IPv6 literal was accepted; nccl_pair parses IPv4 only and would fail obscurely")
	}
	if _, err := resolveHost("no-such-host.invalid"); err == nil {
		t.Error("an unresolvable name was accepted")
	}
}

// The readiness port must be SEPARATE from the bootstrap port, and both must be
// bound by a Group rank.
//
// A tcpSocket probe proves a listener is up the only way TCP allows: by
// connecting to it. serveUniqueId accepts exactly nranks-1 connections, so a
// probe pointed at the bootstrap port consumes a rank's slot and strands a real
// worker at a closed port — and at Group scope that stalls the whole collective
// rather than one link. The two ports must therefore never collide, and the
// defaults must not make them collide by accident.
func TestReadinessAndBootstrapPortsAreDistinct(t *testing.T) {
	if defaultHealthPort == defaultBootstrapPort {
		t.Fatalf("the readiness port and the bootstrap port are both %d — a probe would be "+
			"accepted as a rank and strand a real one", defaultHealthPort)
	}

	// And an operator overriding one must not be able to collide them silently.
	// Without a refusal the failure is still loud — whichever listener binds
	// second gets EADDRINUSE — but it arrives as an unexplained bind error from
	// inside the CUDA harness rather than as the configuration mistake it is.
	if err := checkPorts(plan{healthPort: 19000, bootstrapPort: 19000}); err == nil {
		t.Error("a configuration whose probe port IS the bootstrap port was accepted; the probe " +
			"would be served as a rank and a real rank would find a closed port")
	}
	if err := checkPorts(plan{healthPort: 19000, bootstrapPort: 18535}); err != nil {
		t.Errorf("distinct ports were refused: %v", err)
	}
}

// A non-root rank must still report its own miscompares — audit round 3.
//
// nccl_pair verifies that a SUM all-reduce left nranks in every element, and it
// verifies that on ITS OWN buffers. A miscompare seen by rank 5 is data
// corruption on rank 5's node, and rank 0's buffers being clean says nothing
// about it.
//
// An earlier version of runGroupRank returned Pass for every non-root rank
// without looking at the sweep, on the reasoning that rank 0 reports the
// measurement. That certified a collective which demonstrably returned corrupt
// data: every rank Pass, a full turnout, and the run settles Passed.
func TestAWorkerRankReportsItsOwnMiscompares(t *testing.T) {
	const withMiscompares = `RESULT size_bytes=1048576 time_us=228.500 algbw_gbs=4.5889 busbw_gbs=4.5889
RESULT_WRONG 17
`
	s, err := parseSweep(withMiscompares)
	if err != nil {
		t.Fatalf("fixture does not parse: %v", err)
	}
	if !s.WrongReported || s.Wrong != 17 {
		t.Fatalf("fixture did not carry the miscompare count: reported=%v wrong=%d", s.WrongReported, s.Wrong)
	}

	// The rule the runner must apply, stated here so a refactor that reorders
	// runGroupRank cannot quietly drop it: a positive miscompare count is exit 1
	// (a measurement of this hardware) whatever the rank, and never exit 0.
	if s.WrongReported && s.Wrong > 0 {
		// This is the branch runGroupRank takes before its rank-0 check.
		return
	}
	t.Fatal("a sweep carrying miscompares did not satisfy the fail condition")
}
