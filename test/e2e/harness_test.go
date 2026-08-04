//go:build e2e

package e2e

import (
	"context"
	"fmt"
	"os"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	appsv1 "k8s.io/api/apps/v1"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	clientgoscheme "k8s.io/client-go/kubernetes/scheme"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"

	burninv1alpha1 "github.com/baldwinSPC/glimmer-burnin/api/v1alpha1"
)

const (
	managerNamespace = "glimmer-burnin-system"
	managerName      = "burnin-controller-manager"
	managerSelector  = "burnin-controller"
)

// runnerImage is the CPU-only image every test in this package runs.
//
// It has to be present on the nodes: podForTest sets no imagePullPolicy, so a
// tagged (non-:latest) image defaults to IfNotPresent and a `kind load
// docker-image` is enough. That is deliberate — the e2e must not depend on a
// registry being reachable from inside the cluster.
var runnerImage = envOr("E2E_RUNNER_IMAGE", "busybox:1.37.0")

var (
	k8s    client.Client
	scheme *runtime.Scheme
	uniq   atomic.Int64
)

func TestMain(m *testing.M) {
	scheme = runtime.NewScheme()
	if err := clientgoscheme.AddToScheme(scheme); err != nil {
		fatal("scheme: %v", err)
	}
	if err := burninv1alpha1.AddToScheme(scheme); err != nil {
		fatal("scheme: %v", err)
	}

	cfg, err := ctrl.GetConfig()
	if err != nil {
		fatal("no cluster: %v\n\nThe e2e suite needs a cluster with the operator already installed; "+
			"see the package doc for the five commands that build one.", err)
	}
	k8s, err = client.New(cfg, client.Options{Scheme: scheme})
	if err != nil {
		fatal("build client: %v", err)
	}
	os.Exit(m.Run())
}

func fatal(format string, args ...any) {
	fmt.Fprintf(os.Stderr, "FATAL: "+format+"\n", args...)
	os.Exit(1)
}

func envOr(key, fallback string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return fallback
}

// ─── waiting ──────────────────────────────────────────────────────────────────

// eventually polls until fn returns nil or the deadline passes.
//
// The failure message is fn's LAST error, not a bare timeout, because "the run
// never reached a terminal phase" is useless and "the run is still Running with
// a pod stuck Pending: 0/3 nodes are available, 2 node(s) were unschedulable"
// is the whole diagnosis.
func eventually(t *testing.T, timeout time.Duration, what string, fn func() error) {
	t.Helper()
	deadline := time.Now().Add(timeout)
	var last error
	for {
		last = fn()
		if last == nil {
			return
		}
		if time.Now().After(deadline) {
			t.Fatalf("timed out after %s waiting for %s\nlast state: %v", timeout, what, last)
		}
		time.Sleep(500 * time.Millisecond)
	}
}

// ─── cluster fixtures ─────────────────────────────────────────────────────────

func newNamespace(t *testing.T) string {
	t.Helper()
	name := fmt.Sprintf("burnin-e2e-%d-%d", time.Now().Unix()%100000, uniq.Add(1))
	ns := &corev1.Namespace{ObjectMeta: metav1.ObjectMeta{Name: name}}
	if err := k8s.Create(context.Background(), ns); err != nil {
		t.Fatalf("create namespace: %v", err)
	}
	t.Cleanup(func() {
		// Deleting the namespace reaps the run, its pods and its ConfigMaps.
		// Not waited on: a wedged namespace must not mask the assertion that
		// already ran.
		_ = k8s.Delete(context.Background(), ns)
	})
	return name
}

// workerNodes are the nodes a burn-in may target: everything that is not a
// control-plane node, because the operator itself lives there and a run that
// cordoned its own host would be testing the wrong thing.
func workerNodes(t *testing.T) []string {
	t.Helper()
	var nodes corev1.NodeList
	if err := k8s.List(context.Background(), &nodes); err != nil {
		t.Fatalf("list nodes: %v", err)
	}
	var out []string
	for _, n := range nodes.Items {
		if _, isControlPlane := n.Labels["node-role.kubernetes.io/control-plane"]; isControlPlane {
			continue
		}
		out = append(out, n.Name)
	}
	if len(out) == 0 {
		t.Fatal("no worker nodes: the e2e cluster must have at least one node that is not the control plane")
	}
	return out
}

func getNode(t *testing.T, name string) *corev1.Node {
	t.Helper()
	var node corev1.Node
	if err := k8s.Get(context.Background(), types.NamespacedName{Name: name}, &node); err != nil {
		t.Fatalf("get node %s: %v", name, err)
	}
	return &node
}

func getRun(t *testing.T, ns, name string) *burninv1alpha1.BurnInRun {
	t.Helper()
	var run burninv1alpha1.BurnInRun
	if err := k8s.Get(context.Background(), types.NamespacedName{Namespace: ns, Name: name}, &run); err != nil {
		t.Fatalf("get run %s/%s: %v", ns, name, err)
	}
	return &run
}

func create(t *testing.T, objs ...client.Object) {
	t.Helper()
	for _, obj := range objs {
		if err := k8s.Create(context.Background(), obj); err != nil {
			t.Fatalf("create %T %s: %v", obj, obj.GetName(), err)
		}
	}
}

// ─── object builders ──────────────────────────────────────────────────────────

// shellTest builds a `custom` BurnInTest whose runner is a shell one-liner.
//
// The runner contract is an exit code plus key=value lines on stdout, and that
// is small enough to honour in one line of sh. Nothing here needs a purpose-built
// image, which is what keeps the suite runnable on a GPU-less hosted runner.
func shellTest(ns, name, sh string, mutate func(*burninv1alpha1.BurnInTest)) *burninv1alpha1.BurnInTest {
	bt := &burninv1alpha1.BurnInTest{
		ObjectMeta: metav1.ObjectMeta{Name: name, Namespace: ns},
		Spec: burninv1alpha1.BurnInTestSpec{
			Kind:            burninv1alpha1.KindCustom,
			Scope:           burninv1alpha1.ScopeNode,
			DurationSeconds: 60,
			Runner: &burninv1alpha1.RunnerSpec{
				Image:   runnerImage,
				Command: []string{"/bin/sh", "-c", sh},
			},
		},
	}
	if mutate != nil {
		mutate(bt)
	}
	return bt
}

func profileFor(ns, name string, sinks []string, testRefs ...string) *burninv1alpha1.BurnInProfile {
	p := &burninv1alpha1.BurnInProfile{
		ObjectMeta: metav1.ObjectMeta{Name: name, Namespace: ns},
		Spec:       burninv1alpha1.BurnInProfileSpec{Sinks: sinks},
	}
	for _, ref := range testRefs {
		p.Spec.Tests = append(p.Spec.Tests, burninv1alpha1.ProfileTest{TestRef: ref})
	}
	return p
}

func runFor(ns, name, profileRef string, nodes []string, mutate func(*burninv1alpha1.BurnInRun)) *burninv1alpha1.BurnInRun {
	r := &burninv1alpha1.BurnInRun{
		ObjectMeta: metav1.ObjectMeta{Name: name, Namespace: ns},
		Spec: burninv1alpha1.BurnInRunSpec{
			ProfileRef: profileRef,
			Target:     burninv1alpha1.TargetSelector{NodeNames: nodes},
			// Every run in this suite is bounded. A wedged run must fail the
			// test with a verdict to read, not hang until the job timeout.
			DeadlineSeconds: int32p(600),
		},
	}
	if mutate != nil {
		mutate(r)
	}
	return r
}

func int32p(v int32) *int32 { return &v }
func boolp(v bool) *bool    { return &v }

// ─── run assertions ───────────────────────────────────────────────────────────

func isTerminal(p burninv1alpha1.RunPhase) bool {
	switch p {
	case burninv1alpha1.RunPassed, burninv1alpha1.RunFailed, burninv1alpha1.RunError, burninv1alpha1.RunCancelled:
		return true
	}
	return false
}

// awaitTerminal waits for a verdict and reports what the cluster looked like if
// it never arrives.
func awaitTerminal(t *testing.T, ns, name string, timeout time.Duration) *burninv1alpha1.BurnInRun {
	t.Helper()
	eventually(t, timeout, "run "+ns+"/"+name+" to reach a terminal phase", func() error {
		run := getRun(t, ns, name)
		if isTerminal(run.Status.Phase) {
			return nil
		}
		return fmt.Errorf("phase=%q results=[%s]; %s", run.Status.Phase, summarise(run), podDiagnosis(t, ns, name))
	})
	return getRun(t, ns, name)
}

func summarise(run *burninv1alpha1.BurnInRun) string {
	parts := make([]string, 0, len(run.Status.Results))
	for _, res := range run.Status.Results {
		parts = append(parts, fmt.Sprintf("%s%v=%s(%s)", res.Name, res.Nodes, res.Phase, res.Message))
	}
	return strings.Join(parts, " | ")
}

// podDiagnosis is what makes a cordon deadlock legible instead of a timeout.
// The scheduler's own message ("N node(s) were unschedulable") is the answer,
// and it only exists on the pod.
func podDiagnosis(t *testing.T, ns, runName string) string {
	var pods corev1.PodList
	if err := k8s.List(context.Background(), &pods, client.InNamespace(ns),
		client.MatchingLabels{"burnin.glimmer.ai/run": runName}); err != nil {
		return "pods unreadable: " + err.Error()
	}
	if len(pods.Items) == 0 {
		return "no runner pods exist"
	}
	var parts []string
	for i := range pods.Items {
		pod := &pods.Items[i]
		desc := fmt.Sprintf("%s on %q: %s", pod.Name, pod.Spec.NodeName, pod.Status.Phase)
		for _, c := range pod.Status.Conditions {
			if c.Type == corev1.PodScheduled && c.Status != corev1.ConditionTrue {
				desc += fmt.Sprintf(" [NOT SCHEDULED: %s %s]", c.Reason, c.Message)
			}
		}
		for _, cs := range pod.Status.ContainerStatuses {
			if w := cs.State.Waiting; w != nil {
				desc += fmt.Sprintf(" [waiting: %s %s]", w.Reason, w.Message)
			}
		}
		parts = append(parts, desc)
	}
	return strings.Join(parts, "; ")
}

func resultFor(run *burninv1alpha1.BurnInRun, testName, node string) *burninv1alpha1.TestResult {
	for i := range run.Status.Results {
		res := &run.Status.Results[i]
		if res.Name != testName {
			continue
		}
		if node == "" {
			return res
		}
		for _, n := range res.Nodes {
			if n == node {
				return res
			}
		}
	}
	return nil
}

// assertNoStrandedCordon is the assertion the whole cordon design exists for: a
// node the fleet silently loses has nothing left in the cluster that knows it
// was taken.
//
// It POLLS rather than sampling once. terminate() now makes the verdict durable
// before it releases anything, so a run is observably terminal for a moment
// while it still holds its cordons — that window is the deliberate trade
// described in CLAUDE.md, and asserting into the middle of it would be a
// flake, not a finding. The bound is short: the release is the next thing that
// reconcile does.
func assertNoStrandedCordon(t *testing.T, ns, runName string, nodes ...string) {
	t.Helper()
	eventually(t, 2*time.Minute, "every cordon this run placed to be released", func() error {
		run := getRun(t, ns, runName)
		// The finalizer is the standing record that a run still owes the cluster
		// something. Its absence is the operator's own proof that there is
		// nothing left to give back — and waiting for it here is also what keeps
		// consecutive tests from colliding, since admission treats a run that
		// still holds its finalizer as still holding its nodes.
		for _, f := range run.Finalizers {
			if f == burninv1alpha1.FinalizerCordonCleanup {
				return fmt.Errorf("run (phase %s) still holds the cordon-cleanup finalizer, so it is "+
					"still accounted as testing %v", run.Status.Phase, nodes)
			}
		}
		if len(run.Status.CordonedNodes) != 0 {
			return fmt.Errorf("run (phase %s) still claims to hold %v", run.Status.Phase, run.Status.CordonedNodes)
		}
		for _, name := range nodes {
			node := getNode(t, name)
			if node.Spec.Unschedulable {
				return fmt.Errorf("node %s is STILL CORDONED after the run reached %s — "+
					"the cluster has silently lost it", name, run.Status.Phase)
			}
			for _, key := range []string{
				burninv1alpha1.AnnotationCordonOwner,
				burninv1alpha1.AnnotationPriorUnschedulable,
			} {
				if v, ok := node.Annotations[key]; ok {
					return fmt.Errorf("node %s still carries %s=%q — an ownership stamp with no owner "+
						"misleads the next run", name, key, v)
				}
			}
		}
		return nil
	})
}

// assertNoManufacturedVerdict checks that nothing blamed the hardware for what
// the machinery did to it. A signal death read back as evidence (128+n) is the
// exact fingerprint of a verdict derived from a pod the controller killed.
func assertNoManufacturedVerdict(t *testing.T, run *burninv1alpha1.BurnInRun) {
	t.Helper()
	for _, res := range run.Status.Results {
		if res.Phase == burninv1alpha1.RunFailed {
			t.Errorf("result %q on %v is Failed — a claim about the hardware, and nothing in this run "+
				"measured a fault: %s", res.Name, res.Nodes, res.Message)
		}
		for _, a := range res.Attempts {
			if a.ExitCode == nil {
				continue
			}
			if *a.ExitCode == 137 || *a.ExitCode == 143 {
				t.Errorf("result %q on %v recorded exit %d — the operator's own kill read back as "+
					"evidence about the node: %s", res.Name, res.Nodes, *a.ExitCode, res.Message)
			}
		}
	}
}

// ─── the manager ──────────────────────────────────────────────────────────────

func managerDeployment(t *testing.T) *appsv1.Deployment {
	t.Helper()
	var dep appsv1.Deployment
	if err := k8s.Get(context.Background(),
		types.NamespacedName{Namespace: managerNamespace, Name: managerName}, &dep); err != nil {
		t.Fatalf("get manager deployment: %v", err)
	}
	return &dep
}

func managerPods(t *testing.T) []corev1.Pod {
	t.Helper()
	var pods corev1.PodList
	if err := k8s.List(context.Background(), &pods, client.InNamespace(managerNamespace),
		client.MatchingLabels{"control-plane": managerSelector}); err != nil {
		t.Fatalf("list manager pods: %v", err)
	}
	return pods.Items
}

// awaitManagerAvailable waits for the Deployment to report a ready replica.
func awaitManagerAvailable(t *testing.T, timeout time.Duration) {
	t.Helper()
	eventually(t, timeout, "the operator Deployment to be Available", func() error {
		dep := managerDeployment(t)
		for _, c := range dep.Status.Conditions {
			if c.Type == appsv1.DeploymentAvailable && c.Status == corev1.ConditionTrue {
				return nil
			}
		}
		return fmt.Errorf("replicas ready=%d/%d updated=%d",
			dep.Status.ReadyReplicas, dep.Status.Replicas, dep.Status.UpdatedReplicas)
	})
}

// cordonWatcher records, in the background, whether a node was ever taken out
// of the scheduler while a run was in flight.
//
// Polled rather than watched on purpose: the assertion is about what the
// CLUSTER looked like, and a poll of the node's own spec is the same thing an
// operator would see from `kubectl get nodes`.
type cordonWatcher struct {
	saw  map[string]bool
	stop chan struct{}
	done chan struct{}
}

func watchCordons(t *testing.T, nodes ...string) *cordonWatcher {
	t.Helper()
	w := &cordonWatcher{saw: map[string]bool{}, stop: make(chan struct{}), done: make(chan struct{})}
	go func() {
		defer close(w.done)
		for {
			select {
			case <-w.stop:
				return
			case <-time.After(200 * time.Millisecond):
			}
			for _, name := range nodes {
				var node corev1.Node
				if err := k8s.Get(context.Background(), types.NamespacedName{Name: name}, &node); err != nil {
					continue
				}
				if node.Spec.Unschedulable {
					w.saw[name] = true
				}
			}
		}
	}()
	t.Cleanup(func() { w.finish() })
	return w
}

func (w *cordonWatcher) finish() {
	select {
	case <-w.stop:
	default:
		close(w.stop)
		<-w.done
	}
}

func (w *cordonWatcher) sawCordon(node string) bool {
	w.finish()
	return w.saw[node]
}
