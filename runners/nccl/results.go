// SPDX-License-Identifier: Apache-2.0
// Copyright the Glimmer authors.

package main

import (
	"fmt"
	"strconv"
	"strings"
)

// Reduction of nccl_pair's per-size sweep to the handful of numbers the operator
// stores.
//
// ── THE CONSTRAINT THIS FILE EXISTS FOR ──────────────────────────────────────
//
// pkg/runner's parse is LAST-OCCURRENCE-WINS. A runner that swept message sizes
// and printed "busbw=" once per size would therefore report only the LAST size —
// silently, with no warning anywhere, and with a perfectly plausible-looking
// number. Whether that number is the best or the worst of the sweep would then
// depend on the order the sizes happened to be listed in.
//
// Two ways out were available:
//
//	(a) emit one metric per size under distinct names — busBandwidth1MiBGBs,
//	    busBandwidth256MiBGBs, ... — and
//	(b) sweep for evidence, but emit exactly ONE value per metric name.
//
// (b) is what this runner does, and the reason is that (a) makes the metric name
// carry a benchmark parameter. A profile author gating a fleet would have to
// know which sizes this image happens to sweep, every threshold would break the
// day the sweep changed, and two runner versions would report a link's
// bandwidth under two different names — which is precisely the drift
// pkg/contract's registry exists to prevent.
//
// So the sweep is printed in full to the log as evidence, and exactly one figure
// is promoted to each metric:
//
//	busBandwidthGBs / algBandwidthGBs   the PEAK across the sweep, and both from
//	                                    the SAME size, so the pair is coherent
//	latencyUs                           the SMALLEST size's per-collective time,
//	                                    where the figure is latency rather than
//	                                    bandwidth
//
// The peak is the right summary for a bandwidth gate because the small sizes in
// the sweep are latency-bound by construction: averaging them in would produce a
// number that is not the link's bandwidth and that moves when the sweep changes.

// busBandwidthFactor is nccl-tests' all-reduce scaling factor, 2*(n-1)/n.
//
// A ring all-reduce moves each byte around the ring twice — once reducing, once
// broadcasting — so each rank sees that many bytes of wire traffic per byte of
// buffer. It is duplicated here from nccl_pair.cu, which computes the reported
// figure, SO THAT THE ARITHMETIC CAN BE CHECKED FROM THE OTHER SIDE: the wrapper
// re-derives what the harness claimed and says so when they disagree, which is
// how a units or formula change in either file becomes visible instead of
// becoming a silently rescaled fleet metric.
//
// AT n=2 IT IS EXACTLY 1, so busBandwidthGBs equals algBandwidthGBs for a Pair.
// They are still two metrics, because they are two quantities that merely
// coincide at this rank count.
func busBandwidthFactor(nranks int) float64 {
	if nranks < 2 {
		return 0
	}
	return 2.0 * float64(nranks-1) / float64(nranks)
}

// busBandwidthFactorForCollective is busBandwidthFactor generalised to the
// Node-scope collectives — duplicated from collective.h's BusBandwidthScale
// for the SAME reason busBandwidthFactor is duplicated from nccl_pair.cu: so
// the arithmetic can be cross-checked from the Go side against whichever
// binary produced it.
//
// AllReduce keeps calling busBandwidthFactor unchanged — this function must
// never compute a different number for "allreduce" than that one does, or a
// Node-scope all-reduce and a Pair-scope all-reduce would cross-check against
// two different formulas for the same collective.
//
// AllGather, ReduceScatter and AllToAll each move every byte across the
// interconnect exactly once per rank's share — nccl-tests computes all three
// identically for that reason — so their factor is (n-1)/n rather than
// AllReduce's 2*(n-1)/n. An unrecognised collective name falls back to
// busBandwidthFactor: resolveCollective has already refused anything this
// runner does not know before a sweep can happen, so this branch exists only
// as the same defensive default the rest of this file uses rather than as a
// case this can actually reach.
func busBandwidthFactorForCollective(collective string, nranks int) float64 {
	if collective == "allgather" || collective == "reducescatter" || collective == "alltoall" {
		if nranks < 2 {
			return 0
		}
		return float64(nranks-1) / float64(nranks)
	}
	return busBandwidthFactor(nranks)
}

// sample is one message size's measurement.
type sample struct {
	Bytes  int64
	TimeUs float64
	AlgGBs float64
	BusGBs float64
}

// sweep is one nccl_pair run.
type sweep struct {
	Samples []sample
	// Wrong is the total number of elements that did not match the expected
	// all-reduce result across every size. It is a correctness counter, not a
	// performance one, and it is the reason this test can fail rather than
	// merely report.
	Wrong int64
	// WrongReported distinguishes "the program said zero" from "the program
	// never got that far". A miscompare count that was never produced must not
	// be reported as 0.
	WrongReported bool
}

// Peak returns the sample with the highest bus bandwidth.
func (s sweep) Peak() (sample, bool) {
	var best sample
	found := false
	for _, x := range s.Samples {
		if !found || x.BusGBs > best.BusGBs {
			best, found = x, true
		}
	}
	return best, found
}

// Smallest returns the sample with the smallest message, which is where the
// per-collective time is a latency figure rather than a bandwidth one.
func (s sweep) Smallest() (sample, bool) {
	var best sample
	found := false
	for _, x := range s.Samples {
		if !found || x.Bytes < best.Bytes {
			best, found = x, true
		}
	}
	return best, found
}

// parseSweep reads nccl_pair's RESULT lines.
//
// Only lines this program printed itself are parsed. NCCL's own diagnostics go
// to stderr and are echoed to the log, never scanned for numbers: NCCL's output
// format is not a contract and mining it would make this runner break on an
// NCCL upgrade.
func parseSweep(out string) (sweep, error) {
	var s sweep
	for _, line := range strings.Split(out, "\n") {
		line = strings.TrimSpace(line)
		switch {
		case strings.HasPrefix(line, "RESULT_WRONG "):
			n, err := strconv.ParseInt(strings.TrimSpace(strings.TrimPrefix(line, "RESULT_WRONG ")), 10, 64)
			if err != nil {
				return s, fmt.Errorf("unreadable miscompare count in %q: %w", line, err)
			}
			s.Wrong, s.WrongReported = n, true

		case strings.HasPrefix(line, "RESULT "):
			var x sample
			ok := true
			for _, field := range strings.Fields(strings.TrimPrefix(line, "RESULT ")) {
				k, v, found := strings.Cut(field, "=")
				if !found {
					continue
				}
				switch k {
				case "size_bytes":
					n, err := strconv.ParseInt(v, 10, 64)
					ok = ok && err == nil
					x.Bytes = n
				case "time_us":
					f, err := strconv.ParseFloat(v, 64)
					ok = ok && err == nil
					x.TimeUs = f
				case "algbw_gbs":
					f, err := strconv.ParseFloat(v, 64)
					ok = ok && err == nil
					x.AlgGBs = f
				case "busbw_gbs":
					f, err := strconv.ParseFloat(v, 64)
					ok = ok && err == nil
					x.BusGBs = f
				}
			}
			if !ok || x.Bytes == 0 {
				return s, fmt.Errorf("unreadable result line %q", line)
			}
			s.Samples = append(s.Samples, x)
		}
	}
	if len(s.Samples) == 0 {
		return s, fmt.Errorf("nccl_pair produced no RESULT lines")
	}
	return s, nil
}

// humanBytes formats a size for the log. The sweep is evidence a human reads, so
// it is printed in the units the sizes were chosen in.
func humanBytes(n int64) string {
	switch {
	case n >= 1<<20 && n%(1<<20) == 0:
		return fmt.Sprintf("%d MiB", n>>20)
	case n >= 1<<10 && n%(1<<10) == 0:
		return fmt.Sprintf("%d KiB", n>>10)
	default:
		return fmt.Sprintf("%d B", n)
	}
}

// classifyTransport is a BEST-EFFORT reading of which transport NCCL actually
// selected for an intra-node collective, from NCCL_DEBUG=INFO's own log text.
//
// THIS IS THE ONE DELIBERATE EXCEPTION to the rule at the top of this file:
// "NCCL's own diagnostics go to stderr and are echoed to the log, never
// scanned for numbers — NCCL's output format is not a contract and mining it
// would make this runner break on an NCCL upgrade." That rule protects the
// VERDICT — busbw, algbw, latency and miscompares are parsed only from lines
// this runner itself printed, and nothing here changes that. ncclTransport is
// Evidence, never Acceptance, and this function influences NOTHING about the
// exit code: if NCCL's wording moves under an upgrade, the worst outcome is
// that ok comes back false and the metric is simply absent from that run —
// never a wrong verdict, never a build break.
//
// It exists because an intra-node run raises a question a cross-node one
// never does: did the collective actually cross NVLink, or silently fall back
// to a slower path within the same box. The tokens matched
// ("NVLS", "P2P", "SHM", "NET", and any literal mention of NVLink) are NCCL's
// own internal transport names, but this has NOT been checked against a real
// multi-GPU NCCL_DEBUG=INFO capture — this project has no multi-GPU node. See
// issue #406, which names capturing this as one of the
// things a real run must confirm before this label is trusted.
//
// Precedence when a log shows more than one channel or more than one
// candidate word — NVLS is the most specific claim ("this used NVLink's own
// multicast tree"), so it wins over a bare P2P match, which in turn beats SHM,
// which beats NET. Absence of every token means "could not tell", which is
// omitted rather than guessed — a runner may only declare what it positively
// established.
func classifyTransport(out string) (transport string, ok bool) {
	upper := strings.ToUpper(out)
	switch {
	// Anchored the same way as the other three branches, and for the same
	// reason: NCCL_DEBUG_SUBSYS=INIT,GRAPH is set for this run so GRAPH's own
	// topology dump names NVLink whenever the hardware HAS it, whether or not
	// the collective actually used it as its transport. An unanchored match
	// would read that topology line as a transport decision.
	case strings.Contains(upper, "VIA NVLS") || strings.Contains(upper, "/NVLS/") ||
		strings.Contains(upper, "VIA NVLINK") || strings.Contains(upper, "/NVLINK/"):
		return "nvlink", true
	case strings.Contains(upper, "VIA P2P") || strings.Contains(upper, "/P2P/"):
		return "p2p", true
	case strings.Contains(upper, "VIA SHM") || strings.Contains(upper, "/SHM/"):
		return "shm", true
	case strings.Contains(upper, "VIA NET") || strings.Contains(upper, "/NET/"):
		return "net", true
	default:
		return "", false
	}
}
