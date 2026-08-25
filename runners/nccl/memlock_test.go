// SPDX-License-Identifier: Apache-2.0
// Copyright the Glimmer authors.

package main

import (
	"strings"
	"testing"

	"github.com/baldwinSPC/glimmer-burnin/pkg/runner"
)

// The container runtime's default is 8 MiB (containerd's systemd
// LimitMEMLOCK=8388608), and it is MEASURED to fail — including for an 8-byte
// all-reduce with NCCL_BUFFSIZE=256 KiB, which is what rules out any
// runner-side fix. It must be refused up front rather than discovered.
func TestContainerDefaultMemlockIsRefused(t *testing.T) {
	if memlockSufficient(8 << 20) {
		t.Fatal("8 MiB accepted; that is the container default, and it is measured to fail")
	}
}

func TestGenerousAndUnlimitedMemlockAreAccepted(t *testing.T) {
	for _, v := range []uint64{requiredMemlockBytes, 1 << 30, memlockUnlimited} {
		if !memlockSufficient(v) {
			t.Errorf("memlockSufficient(%s) = false", humanLimit(v))
		}
	}
}

// The message is the whole point: it must name the limit, the observed value,
// and BOTH remedies — and it must say plainly that this is not a hardware
// verdict, or a reader will go and inspect a perfectly good cable.
func TestMemlockAdviceIsActionable(t *testing.T) {
	got := memlockAdvice(8 << 20)
	for _, want := range []string{
		"RLIMIT_MEMLOCK", "8.0 MiB",
		"LimitMEMLOCK=infinity", "containerd", // remedy 1
		"uid 0", "CAP_IPC_LOCK", // remedy 2
		"IPC_LOCK] does NOT work", // the trap worth naming
		"ENVIRONMENT fault",       // not a fabric verdict
		"ib-write-bw",             // the test that still works here
	} {
		if !strings.Contains(got, want) {
			t.Errorf("advice should mention %q; got:\n%s", want, got)
		}
	}
}

// The whole point of #478: this message must NOT repeat memlockAdvice's
// "ENVIRONMENT fault" framing or its remedies, because at both call sites that
// use it (runClient, runGroupRank) the value it is handed was already
// confirmed sufficient before the harness ran — repeating memlockAdvice's
// claim here would have this runner contradict what it already established.
func TestMemlockRuledOutDoesNotRepeatMemlockAdvicesClaim(t *testing.T) {
	got := memlockRuledOut(1 << 30)
	for _, want := range []string{
		"already confirmed sufficient", // the actual fact
		"not the cause",                // the correction
		"genuine, unexplained failure", // what it is instead
	} {
		if !strings.Contains(got, want) {
			t.Errorf("memlockRuledOut should mention %q; got:\n%s", want, got)
		}
	}
	for _, mustNotContain := range []string{
		"ENVIRONMENT fault", // memlockAdvice's claim, which would be false here
		"LimitMEMLOCK",      // a remedy for a limit that was never the problem
	} {
		if strings.Contains(got, mustNotContain) {
			t.Errorf("memlockRuledOut must not repeat memlockAdvice's environment-fault framing; "+
				"found %q in:\n%s", mustNotContain, got)
		}
	}
}

// It must be an Error, never a Fail. The link was never measured, so there is no
// hardware verdict — and Error is the retryable phase, which is right for an
// environment somebody is about to fix. A Fail would permanently indict a link
// over a systemd default.
func TestInsufficientMemlockIsErrorNotFail(t *testing.T) {
	stdout := "NCCL_ERROR: " + memlockAdvice(8<<20) + "\n"
	res := runner.Parse(kind, stdout, exitError)
	if res.Verdict != runner.VerdictError {
		t.Fatalf("verdict = %v, want Error", res.Verdict)
	}
	if len(res.Metrics) != 0 {
		t.Errorf("a refused run must report no measurements, got %v", res.Metrics)
	}
}

// NCCL's and the verbs stack's own wording must be recognised, or the runner
// falls back to a message that reads like a dead fabric.
func TestLooksLikeMemlockExhaustion(t *testing.T) {
	for _, s := range []string{
		"NCCL WARN Call to ibv_reg_mr_iova2 failed with error Cannot allocate memory",
		"Couldn't create CQ",
	} {
		if !looksLikeMemlockExhaustion(s) {
			t.Errorf("not recognised: %q", s)
		}
	}
	for _, s := range []string{"unhandled system error", "connection refused", ""} {
		if looksLikeMemlockExhaustion(s) {
			t.Errorf("wrongly recognised: %q", s)
		}
	}
}

// humanLimit is deliberately distinct from results.go's humanBytes: one formats
// a limit, the other a message size, and a log that conflated them would restart
// the confusion this file exists to end.
func TestHumanLimit(t *testing.T) {
	for in, want := range map[uint64]string{
		8 << 20: "8.0 MiB", 64 << 20: "64.0 MiB", 1 << 30: "1.0 GiB", memlockUnlimited: "unlimited",
	} {
		if got := humanLimit(in); got != want {
			t.Errorf("humanLimit(%d) = %q, want %q", in, got, want)
		}
	}
}
