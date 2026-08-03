// SPDX-License-Identifier: Apache-2.0
// Copyright the Glimmer authors.

package main

import (
	"fmt"
	"strings"
	"syscall"
)

// RLIMIT_MEMLOCK: the limit that decides whether an RDMA test can run at all.
//
// # Why this file exists
//
// Every RDMA buffer is PINNED memory, so the total a verbs application may
// register is capped by RLIMIT_MEMLOCK. Exceed it and ibv_reg_mr / ibv_create_cq
// fail with ENOMEM, which perftest reports as:
//
//	Couldn't create CQ
//	Failed to create CQs
//	 Couldn't create IB resources
//
// That message names neither the limit nor the resource, and it is
// indistinguishable from a genuinely broken HCA. It cost a long bisect once; it
// should never cost one again, which is why this runner now measures the limit
// up front, sizes itself to fit, and — if it still fails — says the word
// "RLIMIT_MEMLOCK" out loud.
//
// # Why this bit the container and not the host
//
// The host allows ~15 GB. A container started by containerd inherits the
// runtime's own limit, which on a systemd distribution defaults to
// LimitMEMLOCK=8M — so the SAME binary, with the SAME arguments, gets 8 MiB in a
// pod and 15 GB on the node. `docker run --ulimit memlock=-1`, which is what
// this runner was first verified with and what every NVIDIA container guide
// tells you to pass, papers over it completely. That is the trap: verification
// under docker with an unlimited memlock cannot see this failure at all.
//
// Kubernetes has no PodSpec field for it. There is no securityContext knob, and
// privileged does not help: a container running as a non-root UID has an EMPTY
// effective capability set even when privileged (CapEff: 0000000000000000), so
// it cannot raise its own hard limit. Sizing the work to fit is therefore the
// only portable answer, and it is what this file does.

// memlockUnlimited is RLIM_INFINITY as it appears in a Rlimit.
const memlockUnlimited = ^uint64(0)

// rlimitMemlock is RLIMIT_MEMLOCK.
//
// Go's syscall package does not export it — golang.org/x/sys/unix does, but this
// runner's non-test sources are deliberately standard-library-only so the image
// build can run with GOPROXY=off and prove it fetched nothing. The constant is 8
// on Linux for every architecture this project targets, and this runner only
// ever executes on Linux.
const rlimitMemlock = 8

// memlockHeadroomDivisor reserves half the limit.
//
// The data buffers are not the only pinned pages: perftest also pins completion
// queues, queue pairs and doorbell pages, and the driver charges its own
// bookkeeping. MEASURED on a DGX Spark pod with an 8 MiB limit: a plan pinning
// exactly 8 MiB of data fails, 4 MiB succeeds and reaches 99.60 Gb/s. Half is
// therefore not a guess, it is the largest fraction observed to work with margin.
const memlockHeadroomDivisor = 2

// minMessageBytes is the floor the message size may be reduced to. Below this
// the test stops being a bandwidth measurement and becomes a message-rate one.
// MEASURED: 64 KiB across 4 queue pairs still reaches 99.40 Gb/s on this fabric,
// so there is a lot of room above this floor.
const minMessageBytes = 16 << 10

// memlock reports the process's RLIMIT_MEMLOCK soft limit in bytes, having first
// tried to raise the soft limit to the hard limit.
//
// Raising soft to hard needs no privilege and is always worth attempting: a
// runtime that sets a generous hard limit and a small soft one is a real and
// common configuration, and there the fix costs nothing. Raising the HARD limit
// needs CAP_SYS_RESOURCE, which a non-root container does not have, so it is not
// attempted — failing to do the impossible is not worth a log line.
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

// pinnedBytes is what perftest will register for a plan: one send and one
// receive region per queue pair, each of the message size.
func pinnedBytes(messageBytes, qps int) uint64 {
	return uint64(messageBytes) * uint64(qps) * 2
}

// budgetFor is how much of the limit this runner will spend on data buffers.
func budgetFor(soft uint64) uint64 {
	if soft == memlockUnlimited {
		return memlockUnlimited
	}
	return soft / memlockHeadroomDivisor
}

// fitToMemlock reduces a plan until its registration fits the budget, and
// explains every reduction it made.
//
// MESSAGE SIZE IS GIVEN UP BEFORE QUEUE PAIRS, and that order is a measurement,
// not a preference. On the reference fabric, dropping from 4 queue pairs to 1
// costs about 2 Gb/s (99.6 -> 97.5), while dropping the message size from 1 MiB
// to 64 KiB at 4 queue pairs costs about 0.2 (99.61 -> 99.40). Queue pairs are
// what saturate the link; message size mostly buys headroom above the point
// where it is already saturated.
//
// Returning ok=false means even the floor does not fit. The caller must then
// report an Error rather than measure something meaningless: a runner that
// silently shrank to a 4 KiB single-QP test would report a "bandwidth" that says
// nothing about the link and would fail a correctly-set threshold.
func fitToMemlock(messageBytes, qps int, budget uint64) (int, int, []string, bool) {
	var notes []string
	if budget == memlockUnlimited || pinnedBytes(messageBytes, qps) <= budget {
		return messageBytes, qps, nil, true
	}

	origBytes, origQPs := messageBytes, qps
	for pinnedBytes(messageBytes, qps) > budget && messageBytes > minMessageBytes {
		messageBytes /= 2
	}
	if messageBytes != origBytes {
		notes = append(notes, fmt.Sprintf("message size %d -> %d bytes", origBytes, messageBytes))
	}
	for pinnedBytes(messageBytes, qps) > budget && qps > 1 {
		qps /= 2
	}
	if qps != origQPs {
		notes = append(notes, fmt.Sprintf("queue pairs %d -> %d", origQPs, qps))
	}
	return messageBytes, qps, notes, pinnedBytes(messageBytes, qps) <= budget
}

// humanBytes formats a byte count for a human reading a pod log.
func humanBytes(n uint64) string {
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

// looksLikeMemlockExhaustion recognises the several unhelpful ways the verbs
// stack reports "you are out of pinned memory".
//
// None of these strings names the limit. Matching them is what lets this runner
// turn a message that reads like a dead HCA into one that names RLIMIT_MEMLOCK,
// so the next person does not repeat the bisect that found it.
func looksLikeMemlockExhaustion(out string) bool {
	l := strings.ToLower(out)
	for _, s := range []string{
		"couldn't create cq",
		"failed to create cqs",
		"couldn't create ib resources",
		"couldn't allocate mr",
		"failed to create mr",
		"cannot allocate memory",
		"couldn't create qp",
	} {
		if strings.Contains(l, s) {
			return true
		}
	}
	return false
}

// memlockAdvice is the message this whole file exists to be able to produce.
func memlockAdvice(soft uint64, needed uint64) string {
	return fmt.Sprintf(
		"this looks like RLIMIT_MEMLOCK exhaustion, not a fabric fault: RDMA buffers are PINNED memory, "+
			"this process's memlock limit is %s, and the test needed about %s of registrations. "+
			"Kubernetes has no PodSpec field for this and `privileged` does not help — a container running as a "+
			"non-root UID has an empty effective capability set and cannot raise its own limit. "+
			"Raise the CONTAINER RUNTIME's limit on the node (for containerd under systemd: a drop-in with "+
			"[Service] LimitMEMLOCK=infinity, then daemon-reload and restart containerd), which is what every "+
			"RDMA and NCCL deployment guide means by `--ulimit memlock=-1`",
		humanBytes(soft), humanBytes(needed))
}
