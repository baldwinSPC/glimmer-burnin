package main

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/util/intstr"

	api "github.com/baldwinSPC/glimmer-burnin/api/v1alpha1"
	"github.com/baldwinSPC/glimmer-burnin/pkg/localrun"
)

func TestTheFlagsRefuseEveryShapeThatWouldLookLikeABadLink(t *testing.T) {
	// Each of these would otherwise surface as a connection error, and a
	// connection error in a fabric test reads as a fault in the fabric.
	// Refusing at the command line keeps a typo out of the hardware record.
	rows := []struct {
		name  string
		flags pairFlags
		want  string
	}{
		{"a client with nowhere to dial", pairFlags{role: "client"}, "--peer"},
		{"a server told where its peer is", pairFlags{role: "server", peer: "10.0.0.1"}, "takes no --peer"},
		{"a peer with no role", pairFlags{peer: "10.0.0.1"}, "needs --role"},
		{"a role that is neither end", pairFlags{role: "primary"}, "server or client"},
	}
	for _, r := range rows {
		_, err := r.flags.rendezvous()
		if err == nil {
			t.Errorf("%s: accepted", r.name)
			continue
		}
		if !strings.Contains(err.Error(), r.want) {
			t.Errorf("%s: error %q does not mention %q", r.name, err, r.want)
		}
	}

	// And the two good shapes.
	if rz, err := (pairFlags{role: "server"}).rendezvous(); err != nil || rz.Role != localrun.RoleServer {
		t.Errorf("a bare server was rejected: %v", err)
	}
	if rz, err := (pairFlags{role: "client", peer: "10.0.0.1"}).rendezvous(); err != nil || rz.PeerHost != "10.0.0.1" {
		t.Errorf("a client with a peer was rejected: %v", err)
	}
	if rz, err := (pairFlags{}).rendezvous(); err != nil || rz != nil {
		t.Errorf("no flags should mean no rendezvous: %v %v", rz, err)
	}
}

func TestOnlyALinkTestIsToldItIsHalfOfOne(t *testing.T) {
	// The rule with teeth: BURNIN_ROLE is gated on SCOPE, not on the flag being
	// present. A Node-scope test that receives it concludes it is one end of a
	// link and waits for a peer that does not exist — a hang, in a profile that
	// merely happened to be run with --role.
	plan := localrun.Plan{
		Node:       "spark-b",
		Rendezvous: &localrun.Rendezvous{Role: localrun.RoleClient, PeerHost: "10.0.0.11", PeerNode: "spark-a"},
		Tests: []localrun.PlannedTest{
			{Name: "smoke", Spec: api.BurnInTestSpec{Kind: "compute-smoke"}},
			{Name: "fabric", Spec: api.BurnInTestSpec{Kind: "ib-write-bw", Scope: api.ScopePair}},
		},
	}

	node, err := localrun.Translate(plan, plan.Tests[0])
	if err != nil {
		t.Fatal(err)
	}
	for _, k := range []string{"BURNIN_ROLE", "BURNIN_PEER_HOST", "BURNIN_PEER_NODE"} {
		if v, ok := node.Env[k]; ok {
			t.Errorf("a Node-scope test got %s=%q", k, v)
		}
	}
	if node.HostNetwork {
		t.Error("a Node-scope test was put on the host network because a link test was in the same profile")
	}

	pair, err := localrun.Translate(plan, plan.Tests[1])
	if err != nil {
		t.Fatal(err)
	}
	if pair.Env["BURNIN_ROLE"] != "client" || pair.Env["BURNIN_PEER_HOST"] != "10.0.0.11" {
		t.Errorf("the pair test did not get its rendezvous: %v", pair.Env)
	}
	if pair.Env["BURNIN_PEER_NODE"] != "spark-a" {
		t.Error("the peer's name should reach the runner for its messages")
	}
	// hostNetwork on bare metal takes port mapping out of the picture: a NAT'd
	// control channel is a connection error that reads as a bad link.
	if !pair.HostNetwork {
		t.Error("a pair test should run on the host network")
	}
}

func TestAPairScopeTestWithNoRoleSkipsRatherThanHalfRunning(t *testing.T) {
	// With no rendezvous at all, BURNIN_ROLE stays unset — which the runners
	// already treat as not-applicable and skip cleanly.
	plan := localrun.Plan{
		Node:  "spark-a",
		Tests: []localrun.PlannedTest{{Name: "fabric", Spec: api.BurnInTestSpec{Kind: "ib-write-bw", Scope: api.ScopePair}}},
	}
	spec, err := localrun.Translate(plan, plan.Tests[0])
	if err != nil {
		t.Fatal(err)
	}
	if _, ok := spec.Env["BURNIN_ROLE"]; ok {
		t.Error("BURNIN_ROLE was set with no rendezvous — the runner would wait for a peer nobody started")
	}
}

func TestGroupVariablesTravelTheSamePath(t *testing.T) {
	// Multi-host Group orchestration is not wired up, but the env contract is
	// one contract. Plumbing it here costs a few lines and is what keeps the
	// two dispatchers from growing two vocabularies.
	rank := int32(0)
	plan := localrun.Plan{
		Node: "spark-a",
		Rendezvous: &localrun.Rendezvous{
			Rank: &rank, NRanks: 3, RootHost: "10.0.0.11", RootNode: "spark-a",
		},
		Tests: []localrun.PlannedTest{{Name: "nccl", Spec: api.BurnInTestSpec{Kind: "nccl", Scope: api.ScopeGroup}}},
	}
	spec, err := localrun.Translate(plan, plan.Tests[0])
	if err != nil {
		t.Fatal(err)
	}
	for k, want := range map[string]string{
		"BURNIN_RANK": "0", "BURNIN_NRANKS": "3", "BURNIN_ROOT_HOST": "10.0.0.11", "BURNIN_ROOT_NODE": "spark-a",
	} {
		if spec.Env[k] != want {
			t.Errorf("%s = %q, want %q", k, spec.Env[k], want)
		}
	}
	if _, ok := spec.Env["BURNIN_ROLE"]; ok {
		// Deliberately absent at Group scope: rank 0 is the root, and a role
		// vocabulary alongside it would be a second way to say the same thing.
		t.Error("BURNIN_ROLE was set at Group scope")
	}
}

func TestALinkResultNamesBothEndsInTargetOrder(t *testing.T) {
	// A Pair verdict is about the LINK. Naming one node sends an engineer to
	// replace the wrong part; naming them in whichever order the local host
	// happens to be would not compare with the same link measured in-cluster,
	// where the operator emits [server, client].
	spec := api.BurnInTestSpec{Kind: "ib-write-bw", Scope: api.ScopePair}

	client := localrun.Plan{Node: "spark-b", Rendezvous: &localrun.Rendezvous{
		Role: localrun.RoleClient, PeerNode: "spark-a"}}
	server := localrun.Plan{Node: "spark-a", Rendezvous: &localrun.Rendezvous{
		Role: localrun.RoleServer, PeerNode: "spark-b"}}

	want := []string{"spark-a", "spark-b"}
	for _, p := range []localrun.Plan{client, server} {
		got := localrun.NodesFor(p, localrun.PlannedTest{Spec: spec})
		if len(got) != 2 || got[0] != want[0] || got[1] != want[1] {
			t.Errorf("from %s: nodes = %v, want %v", p.Node, got, want)
		}
	}
}

func TestTheServerWritesARecordAndNotASecondVerdict(t *testing.T) {
	// Two envelopes for one measurement would render as two results, and an
	// engineer comparing them would be comparing a measurement with its echo.
	dir := t.TempDir()
	rz := &localrun.Rendezvous{Role: localrun.RoleServer, PeerNode: "spark-b"}
	if err := writeResults(dir, sampleReport(), rz); err != nil {
		t.Fatalf("writeResults: %v", err)
	}

	if _, err := os.Stat(filepath.Join(dir, "envelopes")); err == nil {
		t.Error("the server end wrote an envelopes/ directory — report would render it as a verdict")
	}
	body, err := os.ReadFile(filepath.Join(dir, "sidecar", "server-record.json"))
	if err != nil {
		t.Fatalf("no server record: %v", err)
	}
	if !strings.Contains(string(body), "Not a verdict") {
		t.Error("the record does not say what it is not")
	}
	if !strings.Contains(string(body), "spark-b") {
		t.Error("the record should name the peer it was serving")
	}
}

func TestOnlyAPairTestsProbeIsWatched(t *testing.T) {
	probe := func(port int) *api.RunnerSpec {
		return &api.RunnerSpec{ReadinessProbe: tcpProbe(port)}
	}
	rows := []struct {
		name string
		plan localrun.Plan
		want int32
	}{
		{"a pair server's probe", localrun.Plan{Tests: []localrun.PlannedTest{
			{Spec: api.BurnInTestSpec{Scope: api.ScopePair, Runner: probe(18515)}}}}, 18515},
		{"a node test's probe is not a listener to wait for", localrun.Plan{Tests: []localrun.PlannedTest{
			{Spec: api.BurnInTestSpec{Runner: probe(9999)}}}}, 0},
		{"no probe at all", localrun.Plan{Tests: []localrun.PlannedTest{
			{Spec: api.BurnInTestSpec{Scope: api.ScopePair}}}}, 0},
	}
	for _, r := range rows {
		if got := probePort(r.plan); got != r.want {
			t.Errorf("%s: port = %d, want %d", r.name, got, r.want)
		}
	}
}

func TestDecidingSideIsTheClient(t *testing.T) {
	if !deciding(nil) {
		t.Error("an ordinary single-machine run decides its own verdict")
	}
	if !deciding(&localrun.Rendezvous{Role: localrun.RoleClient}) {
		t.Error("the client is the deciding side")
	}
	if deciding(&localrun.Rendezvous{Role: localrun.RoleServer}) {
		t.Error("the server must not produce the link's verdict")
	}
}

// tcpProbe is the shape a Pair server declares to say it is listening.
func tcpProbe(port int) *corev1.Probe {
	return &corev1.Probe{ProbeHandler: corev1.ProbeHandler{
		TCPSocket: &corev1.TCPSocketAction{Port: intstr.FromInt(port)},
	}}
}
