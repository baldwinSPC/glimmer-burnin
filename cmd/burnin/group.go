package main

import (
	"fmt"
	"strings"

	"github.com/baldwinSPC/glimmer-burnin/pkg/group"
	"github.com/baldwinSPC/glimmer-burnin/pkg/localrun"
)

// A collective needs N machines, and at enrollment time none of them is a
// cluster member.
//
// # There is no --role, and that is not an omission
//
// Group scope has no roles. The env contract is BURNIN_RANK, BURNIN_NRANKS,
// BURNIN_ROOT_HOST and BURNIN_ROOT_NODE, and BURNIN_ROLE is DELIBERATELY
// ABSENT — a runner keying off server/client must fail loudly rather than
// treating rank 4 as a client. Offering --role here would let someone set it
// for a Group test and get exactly the silent misconfiguration the operator
// refuses to produce.
//
// # There is no gate on rank 0 being Ready, and that is not an omission either
//
// In a cluster the operator starts rank 0 and creates no other rank until it is
// Ready. Here there is no controller watching pods, so the ordering is the
// OPERATOR-IN-A-HUMAN's: start rank 0, wait for its listener, then start the
// rest. That is documented rather than faked, because a gate this command
// pretended to enforce would be worse than one it plainly does not have.
//
// The runners' connect-with-retry is the real gate, exactly as in a cluster: a
// rank that starts before the root must retry into success, never report a
// fabric fault.

// groupFlags are the N-host flags.
type groupFlags struct {
	rank   int
	nranks int
	root   string
	// rankSet records whether --rank was given at all, because rank 0 is a
	// legal value and its zero value is indistinguishable from absence. Getting
	// this wrong would make every machine that forgot the flag believe it was
	// the root.
	rankSet bool
}

// rendezvous validates the flags and builds what the plan needs.
//
// Strict, for the reason pairFlags is strict: every mistake caught here would
// otherwise surface as a rank that hangs waiting for a peer that never
// rendezvous'd, and a hang in a collective reads as a fabric fault.
func (f groupFlags) rendezvous(node string) (*localrun.Rendezvous, error) {
	if !f.rankSet && f.nranks == 0 && f.root == "" {
		return nil, nil
	}
	if !f.rankSet {
		return nil, fmt.Errorf("--nranks needs --rank: which rank is this machine?")
	}
	if f.nranks < 2 {
		return nil, fmt.Errorf(
			"--nranks is %d; a collective needs at least 2 ranks, and a group of one measures no interconnect at all",
			f.nranks)
	}
	if f.rank < 0 || f.rank >= f.nranks {
		return nil, fmt.Errorf("--rank %d is not a rank of %d: ranks are 0..%d", f.rank, f.nranks, f.nranks-1)
	}
	if f.rank != group.RootRank && f.root == "" {
		return nil, fmt.Errorf(
			"--rank %d needs --root <ip|host>: the address rank %d serves the rendezvous on",
			f.rank, group.RootRank)
	}

	rank := int32(f.rank)
	return &localrun.Rendezvous{
		Rank:   &rank,
		NRanks: int32(f.nranks),
		// The root's own address, which rank 0 does not need and every other
		// rank does. Rank 0 is told its own name for the messages a runner
		// prints, never for addressing.
		RootHost: f.root,
		RootNode: node,
	}, nil
}

// isGroup reports whether these flags describe a Group-scope run.
func (f groupFlags) isGroup() bool { return f.rankSet }

// rendezvous picks between the Pair and Group forms, refusing both at once.
//
// A machine is one end of a link or one rank of a collective, never both, and
// the two carry DIFFERENT env contracts — Pair sets BURNIN_ROLE and Group
// deliberately does not. Accepting both would produce a pod holding a rank AND
// a role, which is the one shape every fabric runner is written to refuse.
func (f runFlags) rendezvous(node string) (*localrun.Rendezvous, error) {
	if f.group.isGroup() && f.pair.role != "" {
		return nil, fmt.Errorf(
			"--rank and --role together: a machine is one end of a LINK or one rank of a COLLECTIVE, "+
				"never both. Pair scope sets BURNIN_ROLE and Group scope deliberately does not, so a runner "+
				"handed both would read rank %d as a client", f.group.rank)
	}
	if f.group.isGroup() {
		return f.group.rendezvous(node)
	}
	return f.pair.rendezvous()
}

// groupDeciding reports whether this rank writes the collective's envelope.
//
// NO RANK DOES, and that is the difference from Pair.
//
// At Pair scope the client holds both halves of the measurement — perftest and
// nccl-tests report there — so it can render the link's verdict alone. A
// collective's verdict is not held by any one rank: rank 0 has the bandwidth
// figure, but "did every rank take part" is a fact no rank can see. So every
// rank writes its OWN record and the verdict is rendered by `burnin merge`,
// which is the only thing that can count to N.
//
// Doing it any other way would mean rank 0 declaring a verdict for machines it
// never heard from, which is exactly what pkg/group.Verdict refuses.
func groupDeciding(*localrun.Rendezvous) bool { return false }

// rankRecordName is where one rank writes its record inside --results-dir.
//
// Rank-numbered rather than node-named, because the merge has to detect a
// MISSING rank and a filename derived from a node it never heard from is not
// something it can look for. Zero-padded so a shell glob lists them in rank
// order, which is the order a person reads them in.
func rankRecordName(rank int) string {
	return fmt.Sprintf("rank-%02d.json", rank)
}

// describeMissing renders the ranks a merge did not find.
//
// Named in full and never summarised: "the collective was not measured" is not
// something a person can act on, and "rank 3 and rank 7 never reported" is.
func describeMissing(missing []int) string {
	parts := make([]string, 0, len(missing))
	for _, r := range missing {
		parts = append(parts, fmt.Sprintf("rank %d", r))
	}
	return strings.Join(parts, ", ")
}
