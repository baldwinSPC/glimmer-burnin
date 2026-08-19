// SPDX-License-Identifier: Apache-2.0
// Copyright the Glimmer authors.

package main

import (
	"math"
	"os"
	"strings"
	"testing"
)

// nodeScopeDecision is the entire Node-scope capability check, and it is
// pure so it is testable with no GPU — this project has no multi-GPU node,
// so this is the only place the >=2-device decision is exercised before real
// hardware (alongside collective.h's C++ arithmetic; see issue #406).
func TestNodeScopeDecision(t *testing.T) {
	if skip, reason := nodeScopeDecision(1); !skip || reason == "" {
		t.Errorf("one GPU: skip=%v reason=%q, want skip=true with a reason", skip, reason)
	}
	if skip, _ := nodeScopeDecision(1); !skip {
		t.Fatal("precondition failed")
	}
	if !strings.Contains(mustReason(t, 1), "1 accelerator") {
		t.Errorf("the reason must name the count found, got %q", mustReason(t, 1))
	}
	for _, gpus := range []int{2, 3, 8} {
		if skip, reason := nodeScopeDecision(gpus); skip {
			t.Errorf("gpus=%d: skip=%v reason=%q, want skip=false", gpus, skip, reason)
		}
	}
}

func mustReason(t *testing.T, gpus int) string {
	t.Helper()
	_, reason := nodeScopeDecision(gpus)
	return reason
}

// THE POSITIVE-CHECK INVARIANT. nodeScopeDecision must decide from the DEVICE
// COUNT alone — never from whether some other variable is absent — because
// reading an absence as permission is exactly the failure the
// groupCapableKinds guard on the Group side exists to catch (#118). This test
// cannot observe main.go's env-reading (that happens before
// nodeScopeDecision is ever called), so it asserts the property the function
// signature itself is supposed to guarantee: gpus is the ONLY input, so the
// same count always produces the same answer regardless of anything else in
// the process's environment.
func TestNodeScopeDecisionIsPositiveNotAbsence(t *testing.T) {
	for _, v := range []string{"", "server", "client", "garbage"} {
		t.Setenv("BURNIN_ROLE", v)
		if skip, _ := nodeScopeDecision(8); skip {
			t.Errorf("BURNIN_ROLE=%q changed the decision at 8 GPUs; nodeScopeDecision must depend only on the count", v)
		}
	}
}

func TestResolveCollective(t *testing.T) {
	t.Run("unset defaults to allreduce", func(t *testing.T) {
		os.Unsetenv("BURNIN_VARIANT_COLLECTIVE")
		got, err := resolveCollective()
		if err != nil || got != "allreduce" {
			t.Errorf("got %q, %v; want %q, nil", got, err, "allreduce")
		}
	})
	for _, name := range []string{"allreduce", "allgather", "reducescatter", "alltoall"} {
		t.Run(name, func(t *testing.T) {
			t.Setenv("BURNIN_VARIANT_COLLECTIVE", name)
			got, err := resolveCollective()
			if err != nil || got != name {
				t.Errorf("got %q, %v; want %q, nil", got, err, name)
			}
		})
	}
	t.Run("unrecognised is refused, not silently measured as allreduce", func(t *testing.T) {
		t.Setenv("BURNIN_VARIANT_COLLECTIVE", "broadcast")
		got, err := resolveCollective()
		if err == nil {
			t.Fatalf("got %q, nil error; want a refusal", got)
		}
		if !strings.Contains(err.Error(), "broadcast") {
			t.Errorf("error %q does not name the bad value", err)
		}
	})
	t.Run("case-sensitive: the operator's own axis values are lowercase", func(t *testing.T) {
		t.Setenv("BURNIN_VARIANT_COLLECTIVE", "AllReduce")
		if _, err := resolveCollective(); err == nil {
			t.Error("a capitalised spelling was silently accepted")
		}
	})
}

// nccl_pair.cu's collective.h ParseCollective must accept exactly the same
// vocabulary this map does, or a value that validates in one binary and is
// refused in the other silently changes which collective a profile actually
// measures. There is no cross-language import to enforce this, so it is
// pinned by name on both sides — this is the Go side's half.
func TestNcclCollectivesVocabulary(t *testing.T) {
	want := map[string]bool{"allreduce": true, "allgather": true, "reducescatter": true, "alltoall": true}
	if len(ncclCollectives) != len(want) {
		t.Fatalf("ncclCollectives = %v, want exactly %v", ncclCollectives, want)
	}
	for k := range want {
		if !ncclCollectives[k] {
			t.Errorf("ncclCollectives is missing %q", k)
		}
	}
}

func TestNodeScopeArgs(t *testing.T) {
	a := strings.Join(nodeScopeArgs("allgather", 3, 10, ""), " ")
	if !strings.Contains(a, "--local-multi-gpu") {
		t.Errorf("missing --local-multi-gpu: %q", a)
	}
	if !strings.Contains(a, "--collective allgather") {
		t.Errorf("missing --collective allgather: %q", a)
	}
	if !strings.Contains(a, "--warmup 3") || !strings.Contains(a, "--iters 10") {
		t.Errorf("warmup/iters not threaded through: %q", a)
	}
	if strings.Contains(a, "--rank") || strings.Contains(a, "--nranks") || strings.Contains(a, "--peer") {
		t.Errorf("the intra-node path must never pass rank/nranks/peer, which mean nothing to it: %q", a)
	}
	if strings.Contains(a, "--sizes") {
		t.Errorf("an empty sizes string must not become a bare --sizes flag: %q", a)
	}
	withSizes := strings.Join(nodeScopeArgs("allreduce", 5, 20, "8,1048576"), " ")
	if !strings.Contains(withSizes, "--sizes 8,1048576") {
		t.Errorf("sizes not passed through: %q", withSizes)
	}
}

func TestBusBandwidthFactorForCollective(t *testing.T) {
	// AllReduce must be BIT-FOR-BIT what busBandwidthFactor already computes —
	// this is the cross-scope invariant the doc comment promises: a Node-scope
	// all-reduce and a Pair-scope all-reduce must never disagree about the
	// formula for identical traffic.
	for _, n := range []int{2, 3, 4, 8} {
		if got, want := busBandwidthFactorForCollective("allreduce", n), busBandwidthFactor(n); got != want {
			t.Errorf("allreduce at n=%d: got %v, want %v (busBandwidthFactor's own answer)", n, got, want)
		}
	}
	// The default (empty string, used by every non-Node scope call site) must
	// also be the all-reduce factor.
	if got, want := busBandwidthFactorForCollective(defaultCollective, 4), busBandwidthFactor(4); got != want {
		t.Errorf("defaultCollective at n=4: got %v, want %v", got, want)
	}
	for _, c := range []string{"allgather", "reducescatter", "alltoall"} {
		got := busBandwidthFactorForCollective(c, 4)
		want := 3.0 / 4.0 // (n-1)/n
		if math.Abs(got-want) > 1e-12 {
			t.Errorf("%s at n=4: got %v, want %v ((n-1)/n)", c, got, want)
		}
		if got := busBandwidthFactorForCollective(c, 1); got != 0 {
			t.Errorf("%s at n=1: got %v, want 0 — one rank crosses no interconnect", c, got)
		}
	}
}

// classifyTransport is BEST-EFFORT and self-omitting — see results.go's doc
// comment for why this is the one place NCCL's own log text is read at all.
// These fixtures are CONSTRUCTED to match NCCL's documented internal
// transport-module names, not captured from real hardware: this project has
// no multi-GPU node, and issue #406 is where a real
// capture confirms or corrects the tokens matched here.
func TestClassifyTransport(t *testing.T) {
	cases := []struct {
		name string
		out  string
		want string
		ok   bool
	}{
		{"NVLS wins as the most specific claim", "NCCL INFO Channel 00/16 : 0 1 2 3 via NVLS", "nvlink", true},
		{"an anchored NVLink transport line counts", "NCCL INFO Channel 00 : 0 1 via NVLink/GDRDMA", "nvlink", true},
		// A bare topology mention must NOT count: NCCL_DEBUG_SUBSYS=INIT,GRAPH is
		// set for this run, and GRAPH's own topology dump names NVLink whenever
		// the hardware HAS it, whether or not the collective actually used it as
		// its transport. Matching this unanchored would misclassify every node
		// with NVLink present but not selected.
		{"a bare topology mention does NOT count", "NCCL INFO topology: GPU 0 -- NVLink -- GPU 1", "", false},
		{"P2P without NVLS", "NCCL INFO Channel 00 : 0 1 via P2P/CUMEM", "p2p", true},
		{"SHM fallback", "NCCL INFO Channel 00 : 0 1 via SHM/direct/direct", "shm", true},
		{"NET fallback", "NCCL INFO Channel 00 : 0 1 via NET/IB/0", "net", true},
		{"no recognisable line at all", "some ordinary log output with no NCCL info in it", "", false},
		{"empty output", "", "", false},
		{"NVLS beats a P2P line appearing earlier in the same log", "via P2P/direct pointer\nvia NVLS\n", "nvlink", true},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			got, ok := classifyTransport(c.out)
			if ok != c.ok || got != c.want {
				t.Errorf("classifyTransport(%q) = (%q, %v), want (%q, %v)", c.out, got, ok, c.want, c.ok)
			}
		})
	}
}

// A wrong or absent transport classification must never become a wrong
// VERDICT: it is Evidence, and its only consumer is a metric() call gated on
// `ok`. This test asserts that gate exists at the call site rather than
// trusting the doc comment — a future edit that always emits the metric,
// defaulting to "", would silently turn "could not tell" into a fabricated
// claim.
func TestClassifyTransportNeverFabricatesAValue(t *testing.T) {
	if _, ok := classifyTransport("nothing recognisable here"); ok {
		t.Fatal("precondition failed: this input must not classify")
	}
	// classifyTransport itself must return the zero value when ok is false, so
	// a caller that forgets to check ok cannot accidentally emit a plausible
	// empty string as if it were a real reading.
	if got, ok := classifyTransport("nothing recognisable here"); ok || got != "" {
		t.Errorf("got = %q, ok = %v; want \"\", false", got, ok)
	}
}
