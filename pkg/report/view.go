package report

import (
	"sort"
	"strings"
	"time"

	"github.com/baldwinSPC/glimmer-burnin/pkg/contract"
)

// The view model every renderer shares.
//
// It exists so the formats cannot disagree. Two renderers deriving "which node
// failed" independently is exactly how a JUnit dashboard and an HTML report end
// up telling an engineer different things about the same run, and this package
// was created because consumers already did that to each other.
//
// It derives and never invents. Everything here is a rearrangement of what the
// envelopes said; where the record is silent the view is silent too.

// View is a run, organised the way a report reads it.
type View struct {
	Run RunView
	// Nodes are the per-node results, one entry per node the run touched, in
	// the order they first appear in the record.
	Nodes []NodeView
	// Links are the Pair- and Group-scope results, each appearing EXACTLY ONCE.
	//
	// This is the rule that matters and it is structural rather than a
	// convention a renderer must remember: a point-to-point measurement is a
	// property of the link, and attributing it to one endpoint sends an engineer
	// to replace the wrong part. There is no way to reach a link result through
	// Nodes, so no renderer can split one by accident.
	Links []LinkView
	Meta  Meta
}

// RunView is the banner: what ran, when, and how it ended.
type RunView struct {
	Namespace string
	Name      string
	UID       string
	Profile   string
	// Phase is the verdict of record, or the latest known phase when Final is
	// false. A renderer must present a non-final phase as progress, never as a
	// verdict.
	Phase string
	Final bool
	// Baseline says the run gated nothing and measured only.
	//
	// A renderer MUST show it. A thresholdless sweep and a certification both
	// end "Passed", and the flag is the only thing between them — a consumer
	// gating admission on "the last run passed" would otherwise certify hardware
	// against a run that gated nothing.
	Baseline bool
	// CancelReason is the operator's own words, when the run was cancelled.
	CancelReason string
	Cluster      *contract.ClusterRef
	StartedAt    string
	FinishedAt   string
	Summary      contract.Summary
	// Fingerprint is what the run captured about the hardware, as a flat map.
	// Read it with FingerprintValue; a missing key is absent, not empty.
	Fingerprint map[string]string
}

// NodeView is one node and everything the run concluded about it ALONE.
type NodeView struct {
	Name string
	// Results are the Node-scope results for this node. Link results are never
	// here; see View.Links.
	Results []contract.TestResult
	// Info is the structured inventory, when the caller supplied one. Known
	// distinguishes "no inventory collected" from "hardware reported nothing".
	Info  NodeInfo
	Known bool
}

// LinkView is one Pair or Group result: a single verdict naming every endpoint.
type LinkView struct {
	Result contract.TestResult
	// Endpoints are the nodes the collective or link ran across, in rank order
	// for a Group — rank i is Endpoints[i], which is how the plan pinned them.
	Endpoints []string
	// Scope is "Pair" or "Group", so a renderer can say which without guessing
	// from the endpoint count.
	Scope string
}

// Label is a link's endpoints as a reader sees them.
func (l LinkView) Label() string {
	switch len(l.Endpoints) {
	case 0:
		return l.Result.Name
	case 1:
		return l.Endpoints[0]
	case 2:
		return l.Endpoints[0] + " <-> " + l.Endpoints[1]
	default:
		return strings.Join(l.Endpoints, " + ")
	}
}

// BuildView derives the shared model from an Input.
func BuildView(in Input) View {
	v := View{Meta: in.Meta}
	verdict, final := in.Verdict()
	if verdict == nil {
		return v
	}

	v.Run = RunView{
		Namespace: verdict.Run.Namespace, Name: verdict.Run.Name, UID: verdict.Run.UID,
		Profile: verdict.Run.Profile, Phase: verdict.Phase, Final: final,
		Baseline: verdict.Baseline, CancelReason: verdict.CancelReason,
		Cluster: verdict.Cluster, Summary: verdict.Summary,
		Fingerprint: verdict.Fingerprint,
	}
	// The envelope carries no run-level start or finish, so the window is
	// DERIVED from the results: the earliest execution began and the latest one
	// ended. That is the run's real occupancy of the hardware, which is the
	// question a reader is asking. A run whose results carry no timestamps at
	// all leaves both empty rather than substituting SentAt, which is when the
	// delivery was sent and not when anything was measured.
	v.Run.StartedAt, v.Run.FinishedAt = executionWindow(verdict.Results)

	// Results are taken from the verdict of record, which is the run's own
	// account of itself. Earlier TestCompleted deliveries describe the same
	// tests and would duplicate every one of them.
	byNode := map[string]*NodeView{}
	var order []string
	for _, r := range verdict.Results {
		if isLinkScope(r) {
			v.Links = append(v.Links, LinkView{
				Result: r, Endpoints: append([]string(nil), r.Nodes...), Scope: linkScope(r),
			})
			continue
		}
		// A Node-scope result names one node. A result naming none — an
		// unresolvable test, a config error — still belongs to the report, under
		// the empty node name, because a required acceptance test that never ran
		// must never vanish from the document that certifies the fleet.
		name := ""
		if len(r.Nodes) > 0 {
			name = r.Nodes[0]
		}
		nv, ok := byNode[name]
		if !ok {
			nv = &NodeView{Name: name}
			byNode[name] = nv
			order = append(order, name)
		}
		nv.Results = append(nv.Results, r)
	}

	// Nodes the caller has inventory for but which produced no result still
	// appear: a node that was targeted and reported nothing is a fact, and a
	// report that omitted it would read as a fleet that was fully measured.
	for _, n := range in.Nodes {
		if _, ok := byNode[n.Name]; !ok {
			byNode[n.Name] = &NodeView{Name: n.Name}
			order = append(order, n.Name)
		}
	}

	for _, name := range order {
		nv := byNode[name]
		nv.Info, nv.Known = in.NodeByName(name)
		v.Nodes = append(v.Nodes, *nv)
	}
	sort.SliceStable(v.Nodes, func(i, j int) bool { return v.Nodes[i].Name < v.Nodes[j].Name })
	return v
}

// executionWindow is the span from the earliest execution to the latest.
//
// Formatted here rather than in each renderer so every format prints the same
// instant the same way, and in UTC because a report is read in a different
// timezone from the one it was produced in more often than not.
func executionWindow(results []contract.TestResult) (started, finished string) {
	var first, last *time.Time
	for i := range results {
		if s := results[i].StartedAt; s != nil && (first == nil || s.Before(*first)) {
			first = s
		}
		if f := results[i].FinishedAt; f != nil && (last == nil || f.After(*last)) {
			last = f
		}
	}
	if first != nil {
		started = first.UTC().Format(timeLayout)
	}
	if last != nil {
		finished = last.UTC().Format(timeLayout)
	}
	return started, finished
}

// timeLayout is how every renderer prints an instant.
const timeLayout = "2006-01-02 15:04:05 UTC"

// isLinkScope reports whether a result is a statement about a link or a
// collective rather than about one node.
//
// Scope is authoritative when present. The fallback on node count exists because
// Scope is omitempty and an envelope written before it was recorded still has to
// render — and getting this wrong in the lenient direction is the harmful one:
// a link result mistaken for a node result gets attributed to one endpoint,
// which sends an engineer to replace the wrong part.
func isLinkScope(r contract.TestResult) bool {
	switch r.Scope {
	case "Pair", "Group":
		return true
	case "Node":
		return false
	default:
		return len(r.Nodes) > 1
	}
}

func linkScope(r contract.TestResult) string {
	if r.Scope != "" {
		return r.Scope
	}
	if len(r.Nodes) == 2 {
		return "Pair"
	}
	return "Group"
}
