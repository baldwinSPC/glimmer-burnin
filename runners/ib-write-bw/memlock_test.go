// SPDX-License-Identifier: Apache-2.0
// Copyright the Glimmer authors.

package main

import (
	"strings"
	"testing"
)

// The numbers in this file are MEASURED, on a DGX Spark pod with the container
// runtime's default RLIMIT_MEMLOCK of 8 MiB:
//
//	1 MiB x 4 QPs = 8192 KiB pinned -> "Couldn't create CQ"
//	512 KiB x 4   = 4096 KiB pinned -> 99.60 Gb/s
//	1 MiB x 2     = 4096 KiB pinned -> 99.61 Gb/s
//	64 KiB x 4    =  512 KiB pinned -> 99.40 Gb/s
//	1 MiB x 1     = 2048 KiB pinned -> 97.47 Gb/s
//
// They are what justify both the headroom divisor and the order of reduction.

const eightMiB = 8 << 20

// The exact configuration that failed on the cluster must now be reduced to one
// that fits. This is the regression test for the whole file.
func TestTheConfigurationThatFailedInAPodNowFits(t *testing.T) {
	budget := budgetFor(eightMiB)
	gotBytes, gotQPs, notes, ok := fitToMemlock(1<<20, 4, budget)
	if !ok {
		t.Fatal("the default plan could not be fitted into an 8 MiB memlock at all")
	}
	if pinnedBytes(gotBytes, gotQPs) > budget {
		t.Fatalf("fitted plan pins %d bytes, over the %d budget", pinnedBytes(gotBytes, gotQPs), budget)
	}
	// The 8 MiB plan is exactly what ENOMEM'd; anything still that large is a
	// failure of this function.
	if pinnedBytes(gotBytes, gotQPs) >= eightMiB {
		t.Errorf("fitted plan still pins %d bytes, which is the amount that failed", pinnedBytes(gotBytes, gotQPs))
	}
	if len(notes) == 0 {
		t.Error("a reduction happened but produced no note; a silently changed measurement is the thing to avoid")
	}
	// Queue pairs are what saturate the link, so they must survive.
	if gotQPs != 4 {
		t.Errorf("queue pairs were reduced to %d before the message size hit its floor; "+
			"QPs cost ~2 Gb/s and message size ~0.2 Gb/s, so message size goes first", gotQPs)
	}
}

// A generous limit must leave the plan completely alone. A runner that "fitted"
// a plan nobody asked it to change would silently alter its own measurement.
func TestGenerousLimitChangesNothing(t *testing.T) {
	b, q, notes, ok := fitToMemlock(1<<20, 4, budgetFor(1<<30))
	if !ok || b != 1<<20 || q != 4 || notes != nil {
		t.Errorf("got %d bytes / %d qps / notes %v, want the plan untouched", b, q, notes)
	}
}

func TestUnlimitedMemlockChangesNothing(t *testing.T) {
	if got := budgetFor(memlockUnlimited); got != memlockUnlimited {
		t.Fatalf("budgetFor(unlimited) = %d", got)
	}
	b, q, notes, ok := fitToMemlock(1<<20, 4, memlockUnlimited)
	if !ok || b != 1<<20 || q != 4 || notes != nil {
		t.Errorf("got %d/%d notes %v, want untouched under an unlimited limit", b, q, notes)
	}
}

// Message size is surrendered before queue pairs, and only down to the floor.
func TestReductionOrderAndFloor(t *testing.T) {
	// Tight enough that the message size alone cannot get there.
	b, q, _, ok := fitToMemlock(1<<20, 4, 64<<10)
	if !ok {
		t.Fatalf("could not fit into 64 KiB")
	}
	if b < minMessageBytes {
		t.Errorf("message size %d went below the %d floor", b, minMessageBytes)
	}
	if q >= 4 {
		t.Errorf("queue pairs should have been reduced once the size floor was reached, got %d", q)
	}
	if pinnedBytes(b, q) > 64<<10 {
		t.Errorf("fitted plan pins %d, over budget", pinnedBytes(b, q))
	}
}

// When even the floor does not fit, the runner must REFUSE rather than measure
// something meaningless. A 4 KiB single-QP "bandwidth" says nothing about a link
// and would fail a correctly-set threshold as if the hardware were at fault.
func TestImpossibleBudgetIsRefused(t *testing.T) {
	if _, _, _, ok := fitToMemlock(1<<20, 4, 1024); ok {
		t.Fatal("fitToMemlock claimed success on a 1 KiB budget")
	}
}

// The headroom is real: a plan pinning the WHOLE limit is what failed on the
// cluster, so the budget must be strictly less than the limit.
func TestBudgetLeavesHeadroom(t *testing.T) {
	if budgetFor(eightMiB) >= eightMiB {
		t.Fatalf("budget %d leaves no headroom below %d", budgetFor(eightMiB), eightMiB)
	}
}

// Every unhelpful string the verbs stack produced on real hardware must be
// recognised, or the diagnosis reverts to "the fabric is broken".
func TestLooksLikeMemlockExhaustion(t *testing.T) {
	for _, s := range []string{
		"Couldn't create CQ\nFailed to create CQs\n Couldn't create IB resources",
		"Call to ibv_reg_mr_iova2 failed with error Cannot allocate memory",
		"Couldn't allocate MR",
	} {
		if !looksLikeMemlockExhaustion(s) {
			t.Errorf("not recognised as memlock exhaustion: %q", s)
		}
	}
	for _, s := range []string{
		"Unable to init the socket connection",
		"Couldn't connect to 10.0.0.1",
		"",
	} {
		if looksLikeMemlockExhaustion(s) {
			t.Errorf("wrongly recognised as memlock exhaustion: %q", s)
		}
	}
}

// The advice must name the limit and the remedy. Its whole purpose is that the
// next person does not repeat the bisect that found this.
func TestMemlockAdviceIsActionable(t *testing.T) {
	got := memlockAdvice(eightMiB, 8<<20)
	for _, want := range []string{"RLIMIT_MEMLOCK", "8.0 MiB", "LimitMEMLOCK", "containerd", "privileged"} {
		if !strings.Contains(got, want) {
			t.Errorf("advice should mention %q; got:\n%s", want, got)
		}
	}
}

func TestHumanBytes(t *testing.T) {
	for in, want := range map[uint64]string{
		8 << 20: "8.0 MiB", 512 << 10: "512.0 KiB", 1 << 30: "1.0 GiB", 42: "42 B",
		memlockUnlimited: "unlimited",
	} {
		if got := humanBytes(in); got != want {
			t.Errorf("humanBytes(%d) = %q, want %q", in, got, want)
		}
	}
}
