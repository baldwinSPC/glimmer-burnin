package controller

import (
	"testing"

	burninv1alpha1 "github.com/baldwinSPC/glimmer-burnin/api/v1alpha1"
)

// A burn-in pod is never safely evictable — issue #159.
//
// It is holding a node this operator CORDONED precisely so that nothing else
// lands there, and moving it discards a measurement in progress. The operator
// then records that as an Error and spends retry budget on a fault the cluster
// caused rather than the hardware.
//
// Over a multi-day soak the cluster's own housekeeping — an autoscaler
// consolidation pass, a drain, a preemption — is the most likely thing to end a
// run. This is the one protection Kubernetes offers that costs nothing.

// Every scope. A Pair's two pods and a Group's N ranks are as unevictable as a
// Node-scope one, and a rule that held for only some of them would be worse than
// none: the scope that lost it would fail in a way nobody was watching for.
func TestEveryRunnerPodAtEveryScopeRefusesEviction(t *testing.T) {
	spec := smokeTest("t").Spec
	run := newRun("r", "p", "spark-a", "spark-b", "spark-c")

	cases := map[string]*rendezvous{
		"node":         nil,
		"pair server":  {scope: burninv1alpha1.ScopePair, role: pairRoleServer, peerRole: pairRoleClient, peerNode: "spark-b", service: "svc"},
		"pair client":  {scope: burninv1alpha1.ScopePair, role: pairRoleClient, peerRole: pairRoleServer, peerNode: "spark-a", service: "svc"},
		"group rank 0": {scope: burninv1alpha1.ScopeGroup, role: groupRole(0), rank: 0, nranks: 3, rootNode: "spark-a", service: "svc"},
		"group rank 2": {scope: burninv1alpha1.ScopeGroup, role: groupRole(2), rank: 2, nranks: 3, rootNode: "spark-a", service: "svc"},
	}
	for name, rv := range cases {
		t.Run(name, func(t *testing.T) {
			pod, err := podForTest(run, 0, 1, "t", &spec, nil, "spark-a", "",
				burninv1alpha1.TargetSelector{}, rv)
			if err != nil {
				t.Fatal(err)
			}
			got := pod.Annotations[annotationSafeToEvict]
			if got != "false" {
				t.Fatalf("%s = %q, want \"false\". An autoscaler consolidation pass "+
					"discards this measurement, and the operator records the loss as an "+
					"Error against hardware that was fine.", annotationSafeToEvict, got)
			}
		})
	}
}

// The annotation is not a knob. There is no case where the honest answer is
// "yes, move this", so nothing in the spec can turn it off.
func TestNothingInTheSpecCanMakeAPodEvictable(t *testing.T) {
	spec := smokeTest("t").Spec
	spec.Runner = &burninv1alpha1.RunnerSpec{Privileged: true}
	spec.HostNetwork = true
	spec.DurationSeconds = 1

	pod, err := podForTest(newRun("r", "p", "spark-a"), 0, 1, "t", &spec, nil, "spark-a", "",
		burninv1alpha1.TargetSelector{}, nil)
	if err != nil {
		t.Fatal(err)
	}
	if pod.Annotations[annotationSafeToEvict] != "false" {
		t.Error("a test's own spec turned off the eviction guard")
	}
}

// PriorityClassName is plumbed, and defaults to unset.
func TestPriorityClassIsPassedThroughAndDefaultsToUnset(t *testing.T) {
	spec := smokeTest("t").Spec
	pod, err := podForTest(newRun("r", "p", "spark-a"), 0, 1, "t", &spec, nil, "spark-a", "",
		burninv1alpha1.TargetSelector{}, nil)
	if err != nil {
		t.Fatal(err)
	}
	if pod.Spec.PriorityClassName != "" {
		t.Errorf("PriorityClassName = %q with none requested; an operator that "+
			"invented a priority would change scheduling for every existing profile",
			pod.Spec.PriorityClassName)
	}

	spec.Runner = &burninv1alpha1.RunnerSpec{PriorityClassName: "burn-in-high"}
	pod, err = podForTest(newRun("r", "p", "spark-a"), 0, 1, "t", &spec, nil, "spark-a", "",
		burninv1alpha1.TargetSelector{}, nil)
	if err != nil {
		t.Fatal(err)
	}
	if pod.Spec.PriorityClassName != "burn-in-high" {
		t.Errorf("PriorityClassName = %q, want the requested class", pod.Spec.PriorityClassName)
	}
}

// A test with no runner block at all still gets a pod, and still refuses
// eviction. priorityClassOf must not panic on the nil it is handed.
func TestATestWithNoRunnerBlockStillRefusesEviction(t *testing.T) {
	spec := burninv1alpha1.BurnInTestSpec{
		Kind:            burninv1alpha1.KindComputeSmoke,
		Scope:           burninv1alpha1.ScopeNode,
		DurationSeconds: 60,
	}
	pod, err := podForTest(newRun("r", "p", "spark-a"), 0, 1, "t", &spec, nil, "spark-a", "",
		burninv1alpha1.TargetSelector{}, nil)
	if err != nil {
		t.Fatal(err)
	}
	if pod.Annotations[annotationSafeToEvict] != "false" {
		t.Error("a test with no runner block lost the eviction guard")
	}
	if pod.Spec.PriorityClassName != "" {
		t.Error("a test with no runner block acquired a priority class")
	}
}
