package envtest

import (
	"strings"
	"testing"
	"time"

	burninv1alpha1 "github.com/baldwinSPC/glimmer-burnin/api/v1alpha1"
)

// Issue #391: a profile whose testRef names a BurnInTest that does not exist
// produced a run that never reported anything — no phase, no condition, no
// result — while the manager's log filled with the same rejection forever:
//
//	status.results[0].scope: Unsupported value: "": supported values: "Node", "Pair", "Group"
//
// The operator was emitting a status its own CRD refuses. finalizeError records
// the cause as a SYNTHETIC TestResult — Name "resolve", empty Kind, which is how
// the controller marks a result that is about the run rather than about any
// hardware — and that result carried no Scope, because it has none: an
// unresolvable ref is not a Node, a Pair or a Group. TestResult.Scope was a
// required enum, so the write was refused whole, the reconcile returned an
// error, was requeued, built the same status and was refused again. Nothing
// ever landed, and the run looked queued rather than broken.
//
// The fake client stores whatever it is handed and reported the resolve path
// green for months. This is the tier that can see it: the run is driven through
// the real reconciler against the shipped CRD, and the assertion is on the
// EFFECT — a durable terminal phase whose message names the missing ref — not
// on any function being called.
func TestUnresolvableTestRefSettlesTheRunWithAReason(t *testing.T) {
	ns := newNamespace(t)
	node := nodeName(t, "unresolved")
	newNode(t, node)

	// The profile is real; the test it names is not. This is the shape a user
	// reaches by applying the samples into one namespace and the profile into
	// another, or by a plain typo.
	create(t,
		profile(ns, "acceptance", nil, burninv1alpha1.ProfileTest{TestRef: "does-not-exist"}),
		runFor(ns, "run", "acceptance", []string{node}, nil),
	)

	d := newDriver(t, ns)
	d.reconcilerOver(admin)

	// A young NotFound is given resolveGracePeriod to wait out the apply race.
	// That grace is not what is under test; move the clock past it.
	d.advanceClock(5 * time.Minute)

	run := d.run("run", 8)

	if run.Status.Phase != burninv1alpha1.RunError {
		t.Fatalf("phase = %q, want Error: an unresolvable testRef is a spec error the user can fix, "+
			"and a run that cannot say so has failed silently in the only place a user looks", run.Status.Phase)
	}
	if len(run.Status.Results) != 1 {
		t.Fatalf("results = %s, want exactly the synthetic resolve result", summarise(run))
	}
	res := run.Status.Results[0]
	if res.Name != "resolve" || res.Kind != "" || res.Phase != burninv1alpha1.RunError {
		t.Errorf("result = %+v, want the synthetic resolve result in Error", res)
	}
	if !strings.Contains(res.Message, "does-not-exist") {
		t.Errorf("message = %q, want it to name the missing testRef", res.Message)
	}
	// The scope is empty because the result has none, and the apiserver has to
	// accept that: this is the exact field whose enum wedged the run.
	if res.Scope != "" {
		t.Errorf("scope = %q on a synthetic result — a resolve failure is not a Node, a Pair or a Group, "+
			"and inventing one to satisfy the schema would be a lie a consumer might act on", res.Scope)
	}
	// Nothing was ever executed, so nothing may be held.
	if len(d.pods("run")) != 0 {
		t.Errorf("a run that could not resolve its profile created %d pod(s)", len(d.pods("run")))
	}
	if n := getNode(t, node); n.Spec.Unschedulable {
		t.Errorf("node %s was left cordoned by a run that never executed", node)
	}
}

// The admission refusal is the OTHER synthetic result — Name "admission",
// empty Kind, no Scope — built by refuse in admission.go, and it was wedged
// the same way for the same reason: a run refused because another run holds
// its targets sat with an empty status forever, and the sentence explaining
// which run was in the way lived only in the manager's log.
func TestAdmissionRefusalSettlesTheRunWithAReason(t *testing.T) {
	ns := newNamespace(t)
	node := nodeName(t, "busy")
	newNode(t, node)

	create(t,
		customTest(ns, "hold", func(bt *burninv1alpha1.BurnInTest) { bt.Spec.DurationSeconds = 3600 }),
		profile(ns, "acceptance", nil, burninv1alpha1.ProfileTest{TestRef: "hold"}),
		runFor(ns, "first", "acceptance", []string{node}, nil),
	)

	d := newDriver(t, ns)
	d.reconcilerOver(admin)
	d.script("", script{linger: true})

	// Get the first run holding the node: started, plan pinned, pod running.
	for i := 0; i < 6; i++ {
		if _, err := d.reconcile("first"); err != nil {
			t.Fatalf("reconcile first %d: %v", i, err)
		}
		d.playKubelet("first")
	}
	if got := getRun(t, ns, "first").Status.Phase; got != burninv1alpha1.RunRunning {
		t.Fatalf("first run phase = %q, want Running before the second arrives", got)
	}

	// The second run wants the same node and is not forced.
	create(t, runFor(ns, "second", "acceptance", []string{node}, nil))
	second := d.run("second", 8)

	if second.Status.Phase != burninv1alpha1.RunError {
		t.Fatalf("second run phase = %q, want Error: a refused run must SAY it was refused", second.Status.Phase)
	}
	if len(second.Status.Results) != 1 {
		t.Fatalf("results = %s, want exactly the synthetic admission result", summarise(second))
	}
	res := second.Status.Results[0]
	if res.Name != "admission" || res.Kind != "" || res.Phase != burninv1alpha1.RunError {
		t.Errorf("result = %+v, want the synthetic admission result in Error", res)
	}
	if !strings.Contains(res.Message, "first") || !strings.Contains(res.Message, node) {
		t.Errorf("message = %q, want it to name the run in the way and the contested node", res.Message)
	}
	if res.Scope != "" {
		t.Errorf("scope = %q on a synthetic result, want none", res.Scope)
	}
	// The first run must be untouched by the refusal.
	if got := getRun(t, ns, "first").Status.Phase; got != burninv1alpha1.RunRunning {
		t.Errorf("first run phase = %q after the second was refused, want still Running", got)
	}
}
