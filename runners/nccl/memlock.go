// SPDX-License-Identifier: Apache-2.0
// Copyright the Glimmer authors.

package main

import (
	"fmt"
	"strings"
	"syscall"
)

// RLIMIT_MEMLOCK, and why this runner refuses rather than adapts.
//
// # The failure
//
// NCCL's IB/RoCE transport registers pinned memory. Exceed RLIMIT_MEMLOCK and it
// fails during transport setup with:
//
//	NCCL WARN Call to ibv_reg_mr_iova2 failed with error Cannot allocate memory
//	unhandled system error
//
// which names neither the limit nor the remedy, and reads like a broken fabric.
//
// # Why this runner cannot size itself to fit, when ib-write-bw can
//
// runners/ib-write-bw owns every byte it registers, so it shrinks its plan to
// the budget and still measures the link: MEASURED, 4 MiB of registrations
// reaches 99.60 Gb/s where 8 MiB reaches 99.61. That runner therefore needs
// nothing from the cluster.
//
// NCCL does not expose the same lever. MEASURED on a DGX Spark pod with the
// container default of 8 MiB, ALL of these still fail in ibv_reg_mr_iova2:
//
//	default settings                       FAIL
//	NCCL_BUFFSIZE=2 MiB / 1 MiB / 512 KiB  FAIL
//	NCCL_BUFFSIZE + NCCL_MAX_NCHANNELS=2   FAIL
//	an 8-BYTE all-reduce, BUFFSIZE=256 KiB FAIL
//
// An eight-byte collective failing rules out the user buffers and the channel
// buffers alike: what does not fit is NCCL's own internal registration, and no
// environment variable makes it smaller. So there is no honest runner-side fix,
// and the only alternative to refusing would be NCCL_IB_DISABLE=1 — a silent
// fallback to a TCP socket transport, which would report a real, reproducible
// number about the wrong path. That is the exact failure this test exists to
// catch, so it must never be the way this test passes.
//
// # Why the environment cannot be fixed from the PodSpec, and what to do
//
// MEASURED on this cluster, all three of these are true at once:
//
//   - containerd's own LimitMEMLOCK is 8388608 (the systemd default), and every
//     container it starts inherits it;
//   - `securityContext.capabilities.add: [IPC_LOCK]` does NOT reach a container
//     running as a non-root UID — CapPrm and CapEff are both empty — because
//     Kubernetes has no ambient-capabilities field, so the permitted set is
//     cleared when runc drops to that UID;
//   - `privileged: true` has the same problem for the same reason: the SAME
//     privileged pod has CapEff 0 as uid 65532 and full capabilities as uid 0.
//
// CAP_IPC_LOCK is what actually matters, because it BYPASSES RLIMIT_MEMLOCK
// entirely — which is why the identical NCCL run passes as uid 0 and fails as
// uid 65532 with the very same 8 MiB limit. So there are exactly two remedies,
// and both are the cluster's rather than this image's:
//
//  1. raise the container runtime's limit (containerd under systemd: a drop-in
//     with [Service] LimitMEMLOCK=infinity, daemon-reload, restart) — this is
//     what every RDMA and NCCL deployment guide means by `--ulimit memlock=-1`;
//     or
//  2. run this runner as uid 0 in a privileged pod, so CAP_IPC_LOCK applies.
//
// This runner therefore checks the limit UP FRONT and reports an Error naming
// both. Error and not Fail: the link was never measured, so there is no hardware
// verdict to give, and Error is the retryable phase — which is the right
// response to an environment that someone is about to fix.

// memlockUnlimited is RLIM_INFINITY as it appears in a Rlimit.
const memlockUnlimited = ^uint64(0)

// rlimitMemlock is RLIMIT_MEMLOCK. Go's syscall package does not export it, and
// this runner's non-test sources are deliberately standard-library-only so the
// image build can run with GOPROXY=off. It is 8 on Linux for every architecture
// this project targets.
const rlimitMemlock = 8

// requiredMemlockBytes is the floor below which this runner refuses to start.
//
// It is a CONSERVATIVE floor, not a measured minimum, and the distinction is
// stated because the difference matters to whoever tunes it. The container
// default of 8 MiB is measured to fail; NCCL's actual requirement could not be
// measured, because the only way to give a container more is to grant
// CAP_IPC_LOCK or raise the runtime limit, and CAP_IPC_LOCK removes the ceiling
// altogether rather than raising it to a number. NCCL's own guidance is
// unlimited. 64 MiB is well above the default that fails and well below anything
// a deliberate operator would configure, so it separates "nobody thought about
// this" from "somebody chose a value".
//
// The Error message always reports the OBSERVED limit and both remedies, so it
// stays actionable even where this floor is imperfect.
const requiredMemlockBytes = 64 << 20

// memlock reports the process's RLIMIT_MEMLOCK soft limit in bytes, having first
// tried to raise the soft limit to the hard limit.
//
// Raising soft to hard needs no privilege and is always worth attempting: a
// runtime that sets a generous hard limit and a small soft one is a real
// configuration, and there this fixes the problem for free. Raising the HARD
// limit needs CAP_SYS_RESOURCE, which a non-root container does not have.
func memlock() (soft uint64, raised bool, err error) {
	var lim syscall.Rlimit
	if err := syscall.Getrlimit(rlimitMemlock, &lim); err != nil {
		return 0, false, fmt.Errorf("reading RLIMIT_MEMLOCK: %w", err)
	}
	if lim.Cur < lim.Max {
		want := syscall.Rlimit{Cur: lim.Max, Max: lim.Max}
		if err := syscall.Setrlimit(rlimitMemlock, &want); err == nil {
			return uint64(lim.Max), true, nil
		}
	}
	return uint64(lim.Cur), false, nil
}

// memlockSufficient reports whether NCCL can be expected to register its
// transport buffers.
func memlockSufficient(soft uint64) bool {
	return soft >= requiredMemlockBytes
}

// humanLimit formats a byte count for a human reading a pod log. It is named
// apart from results.go's humanBytes because that one formats MESSAGE SIZES in
// the units the sweep chose, and conflating a limit with a message size in a log
// is exactly the confusion this whole file is trying to end.
func humanLimit(n uint64) string {
	if n == memlockUnlimited {
		return "unlimited"
	}
	switch {
	case n >= 1<<30:
		return fmt.Sprintf("%.1f GiB", float64(n)/float64(1<<30))
	case n >= 1<<20:
		return fmt.Sprintf("%.1f MiB", float64(n)/float64(1<<20))
	case n >= 1<<10:
		return fmt.Sprintf("%.1f KiB", float64(n)/float64(1<<10))
	default:
		return fmt.Sprintf("%d B", n)
	}
}

// looksLikeMemlockExhaustion recognises how NCCL and the verbs stack report
// "out of pinned memory". None of these strings names the limit, which is why
// the runner translates them.
func looksLikeMemlockExhaustion(out string) bool {
	l := strings.ToLower(out)
	for _, s := range []string{
		"cannot allocate memory",
		"ibv_reg_mr",
		"couldn't create cq",
		"failed to create cqs",
		"couldn't allocate mr",
	} {
		if strings.Contains(l, s) {
			return true
		}
	}
	return false
}

// memlockAdvice is the message this whole file exists to be able to produce.
func memlockAdvice(soft uint64) string {
	return fmt.Sprintf(
		"RLIMIT_MEMLOCK is %s, and NCCL cannot register its transport buffers within it. "+
			"NCCL's registration is internal and no NCCL_BUFFSIZE or NCCL_MAX_NCHANNELS setting makes it fit — "+
			"measured, even an 8-byte all-reduce fails at this limit. This is an ENVIRONMENT fault, not a fabric "+
			"one: the link was never measured, so there is no hardware verdict here. "+
			"Two remedies, both the cluster's: (1) raise the container runtime's limit — for containerd under "+
			"systemd, a drop-in with [Service] LimitMEMLOCK=infinity, then daemon-reload and restart containerd, "+
			"which is what every RDMA/NCCL guide means by `--ulimit memlock=-1`; or (2) run this runner as uid 0 "+
			"in a privileged pod, since CAP_IPC_LOCK bypasses RLIMIT_MEMLOCK entirely. Note that "+
			"securityContext.capabilities.add: [IPC_LOCK] does NOT work here — Kubernetes has no "+
			"ambient-capabilities field, so a non-root container's capability sets are empty even when privileged. "+
			"runners/ib-write-bw needs none of this: it sizes its own registrations to fit and measures the same link",
		humanLimit(soft))
}

// memlockRuledOut is printed instead of memlockAdvice at a call site where
// looksLikeMemlockExhaustion matched the harness output but soft is a value
// THIS RUNNER ALREADY CONFIRMED SUFFICIENT before the harness ever started —
// main's pre-flight gate (memlockSufficient(soft)) refuses to reach either
// runClient or runGroupRank at all otherwise, so soft cannot be the cause of a
// failure surfacing here.
//
// Calling memlockAdvice in that situation would have this runner blame a
// value it already established was fine (#478) — the same "declare what you
// have not positively established" mistake this project refuses everywhere
// else, reached here by trusting a string-pattern match over what the runner
// itself already measured. This keeps the useful half of the match (the
// error SHAPE is a known one) without the false half (that THIS end's
// memlock explains it).
func memlockRuledOut(soft uint64) string {
	return fmt.Sprintf(
		"the error resembles RLIMIT_MEMLOCK exhaustion (a common verbs registration failure shape), but "+
			"this end's own limit (%s) was already confirmed sufficient before the harness started, so "+
			"that is not the cause here. Treating this as a genuine, unexplained failure rather than an "+
			"excused environment fault — the raw harness output is the only evidence available",
		humanLimit(soft))
}
