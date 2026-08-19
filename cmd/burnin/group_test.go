package main

import (
	"strings"
	"testing"

	"github.com/baldwinSPC/glimmer-burnin/pkg/localrun"
)

// Rank 0 is a legal rank and its zero value is indistinguishable from absence.
// A machine that forgot the flag must not believe it is the root.
func TestGroupFlagsRankZeroIsDistinguishableFromAbsence(t *testing.T) {
	if rz, err := (groupFlags{}).rendezvous("n1"); err != nil || rz != nil {
		t.Errorf("no flags = %+v, %v; want no rendezvous at all", rz, err)
	}
	rz, err := groupFlags{rankSet: true, rank: 0, nranks: 2}.rendezvous("spark-a")
	if err != nil {
		t.Fatalf("--rank 0 --nranks 2: %v", err)
	}
	if rz == nil || rz.Rank == nil || *rz.Rank != 0 {
		t.Fatalf("rank 0 did not survive: %+v", rz)
	}
	if rz.NRanks != 2 {
		t.Errorf("NRanks = %d, want 2", rz.NRanks)
	}
}

func TestGroupFlagsAreValidatedBeforeAnythingRuns(t *testing.T) {
	for _, c := range []struct {
		name  string
		flags groupFlags
		want  string
	}{
		{"--nranks without --rank", groupFlags{nranks: 4}, "needs --rank"},
		{"a group of one measures nothing", groupFlags{rankSet: true, nranks: 1}, "at least 2"},
		{"a rank outside the group", groupFlags{rankSet: true, rank: 9, nranks: 4}, "not a rank of 4"},
		{"a non-root rank needs the root's address", groupFlags{rankSet: true, rank: 1, nranks: 2}, "needs --root"},
	} {
		t.Run(c.name, func(t *testing.T) {
			_, err := c.flags.rendezvous("n1")
			if err == nil {
				t.Fatalf("%+v was accepted", c.flags)
			}
			if !strings.Contains(err.Error(), c.want) {
				t.Errorf("error = %q, want it to mention %q", err, c.want)
			}
		})
	}
}

// A machine is one end of a LINK or one rank of a COLLECTIVE, never both. Pair
// sets BURNIN_ROLE and Group deliberately does not, so a runner handed both
// would read rank 4 as a client.
func TestRankAndRoleTogetherAreRefused(t *testing.T) {
	f := runFlags{
		pair:  pairFlags{role: localrun.RoleServer},
		group: groupFlags{rankSet: true, rank: 0, nranks: 2},
	}
	_, err := f.rendezvous("n1")
	if err == nil {
		t.Fatal("--rank and --role together were accepted")
	}
	if !strings.Contains(err.Error(), "BURNIN_ROLE") {
		t.Errorf("the refusal should say why the two cannot coexist: %v", err)
	}
}

// NO RANK DECIDES. rank 0 holds the bandwidth figure, but "did every rank take
// part" is a fact no rank can see, so the verdict is `burnin merge`'s.
func TestNoGroupRankWritesTheCollectivesEnvelope(t *testing.T) {
	rank := int32(0)
	rz := &localrun.Rendezvous{Rank: &rank, NRanks: 4}
	if deciding(rz) {
		t.Error("rank 0 was treated as the deciding end; it would render a verdict for machines it " +
			"never heard from, which pkg/group.Verdict exists to refuse")
	}
	// Pair is unchanged: the client still decides.
	if !deciding(&localrun.Rendezvous{Role: localrun.RoleClient}) {
		t.Error("the Pair client must still decide the link")
	}
	if deciding(&localrun.Rendezvous{Role: localrun.RoleServer}) {
		t.Error("the Pair server must still not decide")
	}
	if !deciding(nil) {
		t.Error("a single-machine run decides its own verdict")
	}
}
