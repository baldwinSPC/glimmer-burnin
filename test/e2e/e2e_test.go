//go:build e2e

package e2e

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"
	"testing"
	"time"

	appsv1 "k8s.io/api/apps/v1"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/intstr"

	burninv1alpha1 "github.com/baldwinSPC/glimmer-burnin/api/v1alpha1"
)

// The runner one-liners. A runner's whole contract is an exit code plus
// key=value lines on stdout, so `sh -c` honours it exactly.
const (
	// passingRunner reports a clean dimensionless counter and exits 0.
	passingRunner = `echo starting acceptance; echo miscompares=0; echo eccErrors=0; exit 0`
	// failingRunner exits 0 but reports a number a threshold will refuse. Exit 0
	// matters: the FAIL has to come from the profile's acceptance bar, which is
	// a different route to Failed than a runner's own exit 1.
	failingRunner = `echo miscompares=7; exit 0`
	// skippingRunner is exit 2 — "this test does not apply to this hardware",
	// declared after looking. It must never read as a failure.
	skippingRunner = `echo no accelerator present; exit 2`
	// slowRunner keeps the node occupied long enough for the chaos tests to
	// interrupt the manager underneath it, and reports progressively so a
	// checkpoint has something to read.
	slowRunner = `i=0; while [ $i -lt 40 ]; do echo elapsedS=$i; i=$((i+1)); sleep 1; done; echo miscompares=0; exit 0`
	// lingeringRunner never finishes on its own, so the test decides when the
	// load stops. Used where the assertion is about something being taken away
	// from a run that is still in flight.
	lingeringRunner = `echo elapsedS=0; while true; do sleep 5; done`
)

// ─── the gate ─────────────────────────────────────────────────────────────────

// TestTheShippedManifestsAreInstalledAndTheOperatorIsRunning is deliberately
// the first test in the file, and it is the cheapest one here.
//
// It asserts the outcome of the CI step that applies config/crd, config/rbac
// and config/manager. THAT STEP IS WORTH MORE THAN EVERY OTHER TEST IN THIS
// PACKAGE COMBINED: an unanchored `manager` line in .gitignore once swallowed
// config/manager entirely and a release shipped with no way to deploy the
// operator, which no Go test could have caught because the Go code was fine.
// This test is the assertion form of it, so a failure says which piece is
// missing rather than leaving a pile of timeouts behind.
func TestTheShippedManifestsAreInstalledAndTheOperatorIsRunning(t *testing.T) {
	ctx := context.Background()
	for _, name := range []string{
		"burninruns.burnin.glimmer.ai",
		"burnintests.burnin.glimmer.ai",
		"burninprofiles.burnin.glimmer.ai",
		"burninsinks.burnin.glimmer.ai",
		"burninschedules.burnin.glimmer.ai",
		"nodefingerprints.burnin.glimmer.ai",
	} {
		// Read as unstructured: the CRD type lives in apiextensions-apiserver,
		// and this repository has no reason to take a direct dependency on it
		// just to look at one condition.
		crd := &unstructured.Unstructured{}
		crd.SetAPIVersion("apiextensions.k8s.io/v1")
		crd.SetKind("CustomResourceDefinition")
		if err := k8s.Get(ctx, types.NamespacedName{Name: name}, crd); err != nil {
			t.Fatalf("CRD %s is not installed: %v — config/crd did not apply", name, err)
		}
		if !crdEstablished(crd) {
			t.Errorf("CRD %s is present but not Established", name)
		}
	}

	awaitManagerAvailable(t, 3*time.Minute)

	// The shape the chaos tests depend on, asserted here so a change to it
	// shows up as one clear failure rather than as a flaky chaos run.
	dep := managerDeployment(t)
	if dep.Spec.Replicas == nil || *dep.Spec.Replicas != 1 {
		t.Errorf("manager replicas = %v, want 1 — leader election, and the overlap the rolling-update "+
			"chaos test relies on, are both written against a single replica", dep.Spec.Replicas)
	}
	// replicas:1 with the default RollingUpdate strategy has maxSurge 25%, which
	// rounds UP to 1: an apply therefore starts a second manager before the
	// first is gone. That overlap is not a misconfiguration — it is what keeps a
	// cluster reconciling across an upgrade — but it IS the condition under
	// which a lost status write turns into a false verdict, so it is stated
	// here rather than discovered.
	if s := dep.Spec.Strategy.Type; s != "" && s != "RollingUpdate" {
		t.Errorf("manager update strategy = %q; the chaos test assumes RollingUpdate", s)
	}
	if !hasLeaderElection(dep) {
		t.Error("the manager does not run with --leader-elect: two overlapping managers would both " +
			"reconcile every run, and the operator's writes are not built for that")
	}
}

func hasLeaderElection(dep *appsv1.Deployment) bool {
	for _, c := range dep.Spec.Template.Spec.Containers {
		for _, a := range c.Args {
			if strings.HasPrefix(a, "--leader-elect") {
				return true
			}
		}
	}
	return false
}

// ─── the core path ────────────────────────────────────────────────────────────

// TestANodeScopeRunSchedulesOntoTheNodeItCordoned is the regression test for
// the cordon deadlock.
//
// The operator cordons its target immediately before putting load on it, which
// the node controller expresses as node.kubernetes.io/unschedulable:NoSchedule.
// A runner pod that does not tolerate that taint can never be placed on the
// node the cordon was placed for — "0/3 nodes are available: 2 node(s) were
// unschedulable" — so the run holds a node out of service and then waits for a
// pod that will never land, until its deadline turns the whole thing into an
// Error. Every test, every node, always.
//
// Nothing without a scheduler can see it. The fake client assigns pods; envtest
// never places them at all. A run reaching Passed here is the proof, and it is
// the reason this package exists.
func TestANodeScopeRunSchedulesOntoTheNodeItCordoned(t *testing.T) {
	ns := newNamespace(t)
	node := workerNodes(t)[0]
	watcher := watchCordons(t, node)

	create(t,
		shellTest(ns, "smoke", passingRunner, func(bt *burninv1alpha1.BurnInTest) {
			bt.Spec.DurationSeconds = 10
			bt.Spec.Thresholds = []burninv1alpha1.Threshold{
				{Metric: "miscompares", Comparison: burninv1alpha1.EQ, Value: "0"},
			}
		}),
		profileFor(ns, "acceptance", nil, "smoke"),
		runFor(ns, "run", "acceptance", []string{node}, nil),
	)

	run := awaitTerminal(t, ns, "run", 5*time.Minute)
	if run.Status.Phase != burninv1alpha1.RunPassed {
		t.Fatalf("phase = %q, want Passed: %s", run.Status.Phase, summarise(run))
	}

	res := resultFor(run, "smoke", node)
	if res == nil {
		t.Fatalf("no result for smoke on %s: %s", node, summarise(run))
	}
	if res.Metrics["miscompares"] != "0" {
		t.Errorf("metrics = %v, want miscompares=0 — the runner's stdout did not reach the verdict",
			res.Metrics)
	}
	if res.StartedAt == nil {
		t.Error("result has no startedAt: the controller never observed the pod executing, " +
			"so this verdict is about a test that may not have run")
	}

	// The cordon has to have actually happened. A run that never cordoned would
	// pass this suite for the wrong reason — it would simply not be exercising
	// the interaction that deadlocked.
	if !watcher.sawCordon(node) {
		t.Errorf("node %s was never observed unschedulable during the run — the cordon/toleration "+
			"interaction this test exists for was not exercised", node)
	}
	assertNoStrandedCordon(t, ns, "run", node)
}

// TestAThresholdViolationIsAHardwareFailAndNotAnError keeps the two apart.
//
// Error is retryable and means the hardware was not judged; Failed is a
// measurement and is never retried. Collapsing them is how a real fault gets
// re-run until it comes out clean.
func TestAThresholdViolationIsAHardwareFailAndNotAnError(t *testing.T) {
	ns := newNamespace(t)
	node := workerNodes(t)[0]

	create(t,
		shellTest(ns, "gate", failingRunner, func(bt *burninv1alpha1.BurnInTest) {
			bt.Spec.DurationSeconds = 10
			bt.Spec.Thresholds = []burninv1alpha1.Threshold{
				{Metric: "miscompares", Comparison: burninv1alpha1.EQ, Value: "0"},
			}
		}),
		profileFor(ns, "acceptance", nil, "gate"),
		runFor(ns, "run", "acceptance", []string{node}, func(r *burninv1alpha1.BurnInRun) {
			// A retry budget that must go unspent: a Fail settles where it
			// happened, with the budget intact.
			r.Spec.RetryOnErrorLimit = int32p(2)
		}),
	)

	run := awaitTerminal(t, ns, "run", 5*time.Minute)
	if run.Status.Phase != burninv1alpha1.RunFailed {
		t.Fatalf("phase = %q, want Failed: %s", run.Status.Phase, summarise(run))
	}
	res := resultFor(run, "gate", node)
	if res == nil {
		t.Fatalf("no result: %s", summarise(run))
	}
	if res.ErrorRetries != 0 {
		t.Errorf("the run spent %d error retries on a FAILED test — re-running a measurement until "+
			"it comes out clean launders a hardware fault into an acceptance", res.ErrorRetries)
	}
	if len(res.Violations) == 0 {
		t.Error("a threshold failure recorded no violations, so nothing says WHICH gate was missed")
	}
	for _, v := range res.Violations {
		if v.Cause != "Measurement" {
			t.Errorf("violation on %q has cause %q, want Measurement — the hardware fell short and "+
				"the report has to say so", v.Metric, v.Cause)
		}
	}
	assertNoStrandedCordon(t, ns, "run", node)
}

// TestASkippingRunnerIsNotAFailure covers exit 2.
//
// A node that cannot take a test has not failed it. Collapsing skip into fail
// is the false negative that made healthy hardware look broken.
func TestASkippingRunnerIsNotAFailure(t *testing.T) {
	ns := newNamespace(t)
	node := workerNodes(t)[0]

	create(t,
		shellTest(ns, "inapplicable", skippingRunner, func(bt *burninv1alpha1.BurnInTest) {
			bt.Spec.DurationSeconds = 10
		}),
		profileFor(ns, "acceptance", nil, "inapplicable"),
		runFor(ns, "run", "acceptance", []string{node}, nil),
	)

	run := awaitTerminal(t, ns, "run", 5*time.Minute)
	if run.Status.Phase != burninv1alpha1.RunPassed {
		t.Fatalf("phase = %q, want Passed — a skip is not evidence either way: %s",
			run.Status.Phase, summarise(run))
	}
	if run.Status.Skipped != 1 || run.Status.Failed != 0 {
		t.Errorf("counters skipped=%d failed=%d, want 1 and 0", run.Status.Skipped, run.Status.Failed)
	}
	assertNoStrandedCordon(t, ns, "run", node)
}

// TestAPairScopeRunRendezvousThroughItsHeadlessService exercises the path that
// shipped against a ClusterRole with no `services` rule.
//
// The client runner does not just start — it RESOLVES its peer through the
// headless Service and opens a connection to it. That makes the assertion about
// the rendezvous rather than about two pods happening to run: per-pod DNS only
// works if the operator created the Service, gave each pod its hostname and
// subdomain, and held the client back until the server was Ready.
func TestAPairScopeRunRendezvousThroughItsHeadlessService(t *testing.T) {
	workers := workerNodes(t)
	if len(workers) < 2 {
		t.Skip("Pair scope needs two distinct worker nodes; this cluster has one")
	}
	ns := newNamespace(t)
	server, peer := workers[0], workers[1]
	watcher := watchCordons(t, server, peer)

	const port int32 = 5000
	// The server binds a socket and holds it. The readinessProbe below is what
	// turns "the container started" into "the listener is up", which is the
	// gate the client waits on.
	//
	// It accepts in a LOOP, and that is load-bearing rather than tidiness. A
	// tcpSocket probe proves a listener is up the only way TCP allows: by
	// connecting to it. That connection is a real one — busybox `nc -l` accepts
	// it, the probe closes it, and a single-shot listener then exits, having
	// served the probe and nobody else. The gate meant to guarantee the listener
	// is up is the thing that consumes it, and the client arrives to a closed
	// port a moment after the operator was told the server was ready.
	//
	// The operator read that correctly, which is why this was worth finding: the
	// run settled `Error` with "server pass (exit 0): peer disconnected; client
	// error (exit 3): could not reach peer". A server that exits 0 before the
	// client ever connected is an Error and never a pass, because no traffic
	// crossed the link — so the harness was wrong and the contract was right.
	//
	// Real fabric runners are exposed to the same trap. `ib_write_bw` and the
	// nccl-tests server accept a bounded number of connections, so a probe that
	// spends one can leave the client connecting to nothing. Prefer a probe on a
	// port the measurement does not use, or a server that re-accepts.
	serverSh := fmt.Sprintf(`echo listening on %d; while true; do nc -l -p %d >/dev/null 2>&1; done`, port, port)
	// The client is the deciding side, and it earns that here: it has to reach
	// the server by the name the operator handed it, which only resolves if the
	// headless Service exists and both pods carry a hostname and a subdomain.
	//
	// The connect is what tests the rendezvous, and it is deliberately the ONLY
	// check: it goes through the container's own resolver, which applies the
	// pod's DNS search path, whereas busybox's nslookup applet does not
	// reliably. A test that used nslookup would be testing busybox.
	//
	// The failure exit is 3 — the runner could not measure, hardware unjudged —
	// and never 1. A rendezvous that did not happen is not a statement about the
	// link, and exit 1 would permanently indict it.
	clientSh := fmt.Sprintf(`
i=0
while [ $i -lt 30 ]; do
  if nc -w 3 "$BURNIN_PEER_HOST" %d </dev/null >/dev/null 2>&1; then
    echo bandwidthGbps=99.61
    exit 0
  fi
  i=$((i+1)); sleep 1
done
echo could not reach peer "$BURNIN_PEER_HOST" - the rendezvous Service or its per-pod DNS is missing
exit 3`, port)

	// ONE image, ONE command, both ends. That is the operator's actual contract:
	// a fabric runner is a server on one node and a client on the other, and it
	// learns which it is from BURNIN_ROLE and nothing else. The dispatch is built
	// before the run exists because the plan is PINNED at start — editing the
	// BurnInTest afterwards cannot reach an in-flight attempt, which is the point
	// of pinning it.
	pairSh := fmt.Sprintf("if [ \"${BURNIN_ROLE:-}\" = client ]; then\n%s\nelse\n%s\nfi", clientSh, serverSh)

	create(t,
		shellTest(ns, "fabric", pairSh, func(bt *burninv1alpha1.BurnInTest) {
			bt.Spec.Scope = burninv1alpha1.ScopePair
			bt.Spec.DurationSeconds = 60
			bt.Spec.Runner.ReadinessProbe = &corev1.Probe{
				ProbeHandler: corev1.ProbeHandler{
					TCPSocket: &corev1.TCPSocketAction{Port: intstr.FromInt32(port)},
				},
				PeriodSeconds:    1,
				FailureThreshold: 60,
			}
			bt.Spec.Thresholds = []burninv1alpha1.Threshold{
				{Metric: "bandwidthGbps", Comparison: burninv1alpha1.GTE, Value: "1"},
			}
		}),
		profileFor(ns, "fabric-suite", nil, "fabric"),
		runFor(ns, "pair-run", "fabric-suite", []string{server, peer}, func(r *burninv1alpha1.BurnInRun) {
			// A pair is one indivisible unit of load holding BOTH nodes, so it
			// costs two of the cap's slots and can never start at the default 1.
			r.Spec.MaxConcurrentNodes = int32p(2)
		}),
	)

	run := awaitTerminal(t, ns, "pair-run", 8*time.Minute)
	if run.Status.Phase != burninv1alpha1.RunPassed {
		t.Fatalf("phase = %q, want Passed: %s", run.Status.Phase, summarise(run))
	}
	// ONE verdict, naming BOTH nodes. A point-to-point measurement is a property
	// of the link; attributing it to an endpoint sends an engineer to replace
	// the wrong part.
	if n := len(run.Status.Results); n != 1 {
		t.Fatalf("pair produced %d results, want exactly 1: %s", n, summarise(run))
	}
	res := run.Status.Results[0]
	if len(res.Nodes) != 2 || res.Nodes[0] != server || res.Nodes[1] != peer {
		t.Errorf("result nodes = %v, want [%s %s]", res.Nodes, server, peer)
	}
	if res.Metrics["bandwidthGbps"] == "" {
		t.Errorf("no bandwidthGbps on the result — the client's numbers did not reach the verdict: %v",
			res.Metrics)
	}
	// Both ends were held for the whole test, which is what "a pair costs two
	// slots" means in the room.
	for _, n := range []string{server, peer} {
		if !watcher.sawCordon(n) {
			t.Errorf("node %s was never cordoned during the pair test — a pair holds BOTH of its nodes", n)
		}
	}
	assertNoStrandedCordon(t, ns, "pair-run", server, peer)
}

// TestTheConfigMapSinkReceivesTheTerminalEnvelope is the export contract.
//
// BurnInSink is the ONLY integration seam with an external consumer, and the
// ConfigMap sink is its no-egress form. A verdict that never leaves the cluster
// is a verdict nobody acts on.
func TestTheConfigMapSinkReceivesTheTerminalEnvelope(t *testing.T) {
	ns := newNamespace(t)
	node := workerNodes(t)[0]

	create(t,
		&burninv1alpha1.BurnInSink{
			ObjectMeta: metav1.ObjectMeta{Name: "local", Namespace: ns},
			Spec: burninv1alpha1.BurnInSinkSpec{
				Type:      burninv1alpha1.SinkConfigMap,
				ConfigMap: &burninv1alpha1.ConfigMapSink{Name: "burnin-results"},
			},
		},
		shellTest(ns, "smoke", passingRunner, func(bt *burninv1alpha1.BurnInTest) {
			bt.Spec.DurationSeconds = 10
		}),
		profileFor(ns, "acceptance", []string{"local"}, "smoke"),
		runFor(ns, "run", "acceptance", []string{node}, nil),
	)

	run := awaitTerminal(t, ns, "run", 5*time.Minute)
	if run.Status.Phase != burninv1alpha1.RunPassed {
		t.Fatalf("phase = %q, want Passed: %s", run.Status.Phase, summarise(run))
	}

	var terminal map[string]any
	eventually(t, 2*time.Minute, "the terminal envelope to reach the sink", func() error {
		var cm corev1.ConfigMap
		if err := k8s.Get(context.Background(),
			types.NamespacedName{Namespace: ns, Name: "burnin-results"}, &cm); err != nil {
			return err
		}
		for _, body := range cm.Data {
			var env map[string]any
			if err := json.Unmarshal([]byte(body), &env); err != nil {
				return fmt.Errorf("sink wrote something that is not an envelope: %v", err)
			}
			if env["phase"] == string(burninv1alpha1.RunPassed) {
				terminal = env
				return nil
			}
		}
		return fmt.Errorf("%d envelope(s) delivered, none of them terminal", len(cm.Data))
	})

	// The envelope is a versioned contract with an external consumer on the
	// other end. These are the fields that consumer gates acceptance on.
	for _, key := range []string{"version", "deliveryId", "reason", "run", "phase", "results"} {
		if _, ok := terminal[key]; !ok {
			t.Errorf("terminal envelope has no %q field — the delivery contract is incomplete: %v",
				key, keysOf(terminal))
		}
	}
	if run.Status.Passed != 1 {
		t.Errorf("status.passed = %d, want 1", run.Status.Passed)
	}
	assertNoStrandedCordon(t, ns, "run", node)
}

// ─── helpers used only by this file ───────────────────────────────────────────

func keysOf(m map[string]any) []string {
	out := make([]string, 0, len(m))
	for k := range m {
		out = append(out, k)
	}
	return out
}

// crdEstablished reads the Established condition off an unstructured CRD.
func crdEstablished(crd *unstructured.Unstructured) bool {
	conditions, found, err := unstructured.NestedSlice(crd.Object, "status", "conditions")
	if err != nil || !found {
		return false
	}
	for _, raw := range conditions {
		c, ok := raw.(map[string]any)
		if !ok {
			continue
		}
		if c["type"] == "Established" && c["status"] == "True" {
			return true
		}
	}
	return false
}
