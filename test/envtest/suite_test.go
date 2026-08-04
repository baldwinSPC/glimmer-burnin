package envtest

import (
	"bytes"
	"context"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	corev1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	utilyaml "k8s.io/apimachinery/pkg/util/yaml"
	clientgoscheme "k8s.io/client-go/kubernetes/scheme"
	"k8s.io/client-go/rest"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"
	crenvtest "sigs.k8s.io/controller-runtime/pkg/envtest"

	burninv1alpha1 "github.com/baldwinSPC/glimmer-burnin/api/v1alpha1"
	"github.com/baldwinSPC/glimmer-burnin/internal/controller"
)

var (
	testEnv *crenvtest.Environment
	// cfg is the admin (system:masters) connection to the test apiserver.
	cfg *rest.Config
	// admin is a full-privilege, uncached client. It stands in for the pieces
	// of a cluster envtest does not run: it writes pod status the way a kubelet
	// would, and it reads the apiserver's own copy of an object when a test
	// needs to know what is DURABLE rather than what a reconciler holds in
	// memory.
	admin  client.Client
	scheme *runtime.Scheme
)

// repoRoot is where config/ lives, relative to this package.
const repoRoot = "../.."

func TestMain(m *testing.M) {
	// The suite is skipped, loudly, when the control-plane binaries are not
	// installed — a laptop without them must still get a green `go test ./...`.
	// CI sets BURNIN_ENVTEST=required so the skip can never happen there: a
	// suite that quietly skips in CI has negative value, because it also
	// removes the pressure to notice.
	if os.Getenv("KUBEBUILDER_ASSETS") == "" {
		msg := "envtest control-plane binaries not found: set KUBEBUILDER_ASSETS (make envtest-assets)"
		if os.Getenv("BURNIN_ENVTEST") == "required" {
			fmt.Fprintln(os.Stderr, "FATAL: "+msg+" — BURNIN_ENVTEST=required forbids skipping")
			os.Exit(1)
		}
		fmt.Println("SKIP test/envtest: " + msg)
		os.Exit(0)
	}

	scheme = runtime.NewScheme()
	must(clientgoscheme.AddToScheme(scheme))
	must(burninv1alpha1.AddToScheme(scheme))

	testEnv = &crenvtest.Environment{
		// The SHIPPED manifests, not a copy. If config/crd drifts from the Go
		// types, every test in this package is running against the drift.
		CRDDirectoryPaths:     []string{filepath.Join(repoRoot, "config", "crd")},
		ErrorIfCRDPathMissing: true,
		Scheme:                scheme,
	}

	var err error
	cfg, err = testEnv.Start()
	if err != nil {
		fmt.Fprintf(os.Stderr, "FATAL: could not start envtest control plane: %v\n", err)
		os.Exit(1)
	}

	admin, err = client.New(cfg, client.Options{Scheme: scheme})
	if err != nil {
		fmt.Fprintf(os.Stderr, "FATAL: could not build admin client: %v\n", err)
		_ = testEnv.Stop()
		os.Exit(1)
	}

	code := m.Run()
	if err := testEnv.Stop(); err != nil {
		fmt.Fprintf(os.Stderr, "warning: stopping envtest control plane: %v\n", err)
	}
	os.Exit(code)
}

func must(err error) {
	if err != nil {
		panic(err)
	}
}

// ─── Fixtures ─────────────────────────────────────────────────────────────────

var uniq atomic.Int64

// newNamespace creates a namespace nothing else in the suite touches.
//
// It is never deleted. envtest runs no namespace controller, so a Terminating
// namespace never finishes terminating and would block every later test that
// happened to reuse the name.
func newNamespace(t *testing.T) string {
	t.Helper()
	name := fmt.Sprintf("burnin-%d-%d", time.Now().UnixNano()%1e6, uniq.Add(1))
	ns := &corev1.Namespace{ObjectMeta: metav1.ObjectMeta{Name: name}}
	if err := admin.Create(context.Background(), ns); err != nil {
		t.Fatalf("create namespace: %v", err)
	}
	return name
}

// newNode creates a schedulable Node carrying the hostname label runner pods
// select on. Nodes are cluster-scoped, so the name carries a unique suffix
// rather than a namespace.
func newNode(t *testing.T, name string) *corev1.Node {
	t.Helper()
	node := &corev1.Node{
		ObjectMeta: metav1.ObjectMeta{
			Name:   name,
			Labels: map[string]string{corev1.LabelHostname: name},
		},
	}
	if err := admin.Create(context.Background(), node); err != nil {
		t.Fatalf("create node %s: %v", name, err)
	}
	t.Cleanup(func() {
		_ = admin.Delete(context.Background(), &corev1.Node{ObjectMeta: metav1.ObjectMeta{Name: name}})
	})
	return node
}

func nodeName(t *testing.T, suffix string) string {
	t.Helper()
	return fmt.Sprintf("n-%d-%s", uniq.Add(1), suffix)
}

func getNode(t *testing.T, name string) *corev1.Node {
	t.Helper()
	var node corev1.Node
	if err := admin.Get(context.Background(), types.NamespacedName{Name: name}, &node); err != nil {
		t.Fatalf("get node %s: %v", name, err)
	}
	return &node
}

// getRun reads the run from the APISERVER — never from a reconciler's copy.
// Every assertion about what a run "knows" goes through here, because the bug
// class this package exists for is precisely a status the controller believed
// it had written and had not.
func getRun(t *testing.T, ns, name string) *burninv1alpha1.BurnInRun {
	t.Helper()
	var run burninv1alpha1.BurnInRun
	if err := admin.Get(context.Background(), types.NamespacedName{Namespace: ns, Name: name}, &run); err != nil {
		t.Fatalf("get run %s/%s: %v", ns, name, err)
	}
	return &run
}

// ─── Manifest application ─────────────────────────────────────────────────────

// readManifestObjects decodes every document of a manifest file into
// unstructured objects, exactly as `kubectl apply -f` would see them.
func readManifestObjects(path string) ([]*unstructured.Unstructured, error) {
	raw, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	var out []*unstructured.Unstructured
	dec := utilyaml.NewYAMLOrJSONDecoder(bytes.NewReader(raw), 4096)
	for i := 0; ; i++ {
		obj := &unstructured.Unstructured{}
		err := dec.Decode(obj)
		if err == io.EOF {
			break
		}
		if err != nil {
			return nil, fmt.Errorf("%s: document %d: %w", path, i, err)
		}
		if len(obj.Object) == 0 {
			continue // comment-only document
		}
		out = append(out, obj)
	}
	return out, nil
}

// applyManifest creates every document in a SHIPPED manifest file and removes
// them again when the test ends.
//
// Applying the real file rather than a hand-typed Go equivalent is the whole
// point: a re-typed copy tests the copy. mutate is called per document so a
// test can retarget a namespaced object into its own namespace; returning false
// skips the document.
func applyManifest(t *testing.T, path string, mutate func(*unstructured.Unstructured) bool) []*unstructured.Unstructured {
	t.Helper()
	applied := applyManifestPermanent(t, path, mutate)
	for _, obj := range applied {
		gone := obj.DeepCopy()
		t.Cleanup(func() { _ = admin.Delete(context.Background(), gone) })
	}
	return applied
}

// applyManifestPermanent is applyManifest without the cleanup, for objects that
// are shared inputs to the whole package (the ClusterRole, the operator's
// namespace and service account) and must outlive whichever test created them.
func applyManifestPermanent(t *testing.T, path string, mutate func(*unstructured.Unstructured) bool) []*unstructured.Unstructured {
	t.Helper()
	objs, err := readManifestObjects(path)
	if err != nil {
		t.Fatalf("read %s: %v", path, err)
	}
	var applied []*unstructured.Unstructured
	for _, obj := range objs {
		if mutate != nil && !mutate(obj) {
			continue
		}
		if err := admin.Create(context.Background(), obj); err != nil && !apierrors.IsAlreadyExists(err) {
			t.Fatalf("%s: create %s %s: %v", path, obj.GetKind(), obj.GetName(), err)
		}
		applied = append(applied, obj)
	}
	return applied
}

// ─── Driving a run ────────────────────────────────────────────────────────────

// script is what one test's runner pods "do": what they print and what they
// exit with. envtest has no kubelet, so the suite plays that part.
type script struct {
	stdout   string
	exitCode int32
	// linger keeps the pod Running forever instead of terminating. It is how a
	// test reaches the paths where the controller has to decide what to do
	// about a pod that is still executing — a deadline, a cancel, or a pair
	// server that outlives its client.
	linger bool
	// notReady leaves the pod Running but not Ready, which is what gates the
	// client half of a Pair rendezvous.
	notReady bool
}

// driver runs one BurnInRun to completion, playing kubelet in between
// reconciles.
type driver struct {
	t *testing.T
	// r is the reconciler under test. Its client may be impersonated, so it is
	// deliberately NOT the client the driver uses to move pods along.
	r  *controller.BurnInRunReconciler
	ns string
	// scripts is keyed by the pod's test label; "" is the fallback.
	scripts map[string]script
	// started records pods already moved to Running, so the controller observes
	// Running before it observes a terminal phase.
	started map[string]bool
	// onPodDelete, when set, is called with the last observed state of every pod
	// that disappeared, immediately after the reconcile that removed it.
	onPodDelete func(pod *corev1.Pod)
	seen        map[string]*corev1.Pod
	now         time.Time
}

func newDriver(t *testing.T, ns string) *driver {
	return &driver{
		t:       t,
		ns:      ns,
		scripts: map[string]script{},
		started: map[string]bool{},
		seen:    map[string]*corev1.Pod{},
		now:     time.Now(),
	}
}

func (d *driver) script(testName string, s script) *driver {
	d.scripts[testName] = s
	return d
}

func (d *driver) scriptFor(pod *corev1.Pod) script {
	if s, ok := d.scripts[pod.Labels["burnin.glimmer.ai/test"]]; ok {
		return s
	}
	return d.scripts[""]
}

// podLogs is the reconciler's log channel: whatever the pod's script says it
// printed. A pod with no script prints nothing, which is a state the controller
// has real opinions about.
func (d *driver) podLogs(_ context.Context, _, name string) (string, error) {
	if pod, ok := d.seen[name]; ok {
		return d.scriptFor(pod).stdout, nil
	}
	return "", nil
}

// reconcilerOver builds the reconciler under test over a given client, wired to
// this driver's log channel and clock.
func (d *driver) reconcilerOver(c client.Client) *controller.BurnInRunReconciler {
	d.r = &controller.BurnInRunReconciler{
		Client: c,
		Scheme: scheme,
		// The same client: everything in this package already talks straight to
		// the apiserver, so there is no cache to bypass. It is wired anyway so
		// the reads that MUST be uncached in production go through the field
		// they go through in production, rather than through the fallback.
		APIReader:     c,
		PodLogs:       d.podLogs,
		Now:           func() time.Time { return d.now },
		ForgetMetrics: func(string, string) {},
	}
	return d.r
}

// advanceClock moves the reconciler's clock forward. Deadlines and checkpoint
// cadences are the only things in the controller that depend on wall time, and
// both are read through Now.
func (d *driver) advanceClock(by time.Duration) { d.now = d.now.Add(by) }

func (d *driver) reconcile(name string) (ctrl.Result, error) {
	return d.r.Reconcile(context.Background(),
		ctrl.Request{NamespacedName: types.NamespacedName{Namespace: d.ns, Name: name}})
}

func (d *driver) pods(runName string) []corev1.Pod {
	d.t.Helper()
	var list corev1.PodList
	if err := admin.List(context.Background(), &list,
		client.InNamespace(d.ns), client.MatchingLabels{"burnin.glimmer.ai/run": runName}); err != nil {
		d.t.Fatalf("list pods: %v", err)
	}
	return list.Items
}

// playKubelet advances every pod of the run by one step: Pending becomes
// Running (and Ready), Running becomes terminated with the scripted exit code.
//
// The two steps are separate on purpose. A controller that only ever sees a pod
// after it finished takes a different path through beginAttempt than one that
// watched it run, and a Pair rendezvous cannot happen at all unless the server
// is observed Ready before the client exists.
func (d *driver) playKubelet(runName string) {
	d.t.Helper()
	ctx := context.Background()
	pods := d.pods(runName)
	for i := range pods {
		pod := pods[i]
		d.seen[pod.Name] = pod.DeepCopy()
		s := d.scriptFor(&pod)

		switch {
		case pod.Status.Phase == "" || pod.Status.Phase == corev1.PodPending:
			now := metav1.NewTime(d.now)
			pod.Status.Phase = corev1.PodRunning
			pod.Status.StartTime = &now
			pod.Status.ContainerStatuses = []corev1.ContainerStatus{{
				Name:  "runner",
				Ready: !s.notReady,
				State: corev1.ContainerState{Running: &corev1.ContainerStateRunning{StartedAt: now}},
			}}
			readiness := corev1.ConditionTrue
			if s.notReady {
				readiness = corev1.ConditionFalse
			}
			pod.Status.Conditions = []corev1.PodCondition{{Type: corev1.PodReady, Status: readiness}}
			if err := admin.Status().Update(ctx, &pod); err != nil && !apierrors.IsNotFound(err) {
				d.t.Fatalf("start pod %s: %v", pod.Name, err)
			}
			d.started[pod.Name] = true
			d.seen[pod.Name] = pod.DeepCopy()

		case pod.Status.Phase == corev1.PodRunning && d.started[pod.Name] && !s.linger:
			phase := corev1.PodSucceeded
			if s.exitCode != 0 {
				phase = corev1.PodFailed
			}
			now := metav1.NewTime(d.now)
			pod.Status.Phase = phase
			pod.Status.ContainerStatuses = []corev1.ContainerStatus{{
				Name: "runner",
				State: corev1.ContainerState{Terminated: &corev1.ContainerStateTerminated{
					ExitCode:   s.exitCode,
					Reason:     "Completed",
					FinishedAt: now,
				}},
			}}
			if err := admin.Status().Update(ctx, &pod); err != nil && !apierrors.IsNotFound(err) {
				d.t.Fatalf("finish pod %s: %v", pod.Name, err)
			}
			d.seen[pod.Name] = pod.DeepCopy()
		}
	}
}

// noteDeletions reports pods that vanished since the last sweep, so a test can
// assert on what the controller destroyed and on what the apiserver knew at
// that moment.
func (d *driver) noteDeletions(runName string) {
	if d.onPodDelete == nil {
		return
	}
	live := map[string]bool{}
	for _, pod := range d.pods(runName) {
		live[pod.Name] = true
	}
	for name, pod := range d.seen {
		if live[name] {
			continue
		}
		delete(d.seen, name)
		d.onPodDelete(pod)
	}
}

// run drives the reconcile loop until the run reaches a terminal phase or the
// step budget is spent.
func (d *driver) run(name string, maxSteps int) *burninv1alpha1.BurnInRun {
	d.t.Helper()
	for i := 0; i < maxSteps; i++ {
		if _, err := d.reconcile(name); err != nil {
			d.t.Fatalf("reconcile %d: %v%s", i, err, forbiddenHint(err))
		}
		d.noteDeletions(name)
		if isTerminalPhase(getRun(d.t, d.ns, name).Status.Phase) {
			// One more pass so the terminal path's own side effects — cordon
			// release, delivery retry, finalizer removal — converge.
			if _, err := d.reconcile(name); err != nil {
				d.t.Fatalf("terminal reconcile: %v%s", err, forbiddenHint(err))
			}
			d.noteDeletions(name)
			return getRun(d.t, d.ns, name)
		}
		d.playKubelet(name)
		d.noteDeletions(name)
	}
	run := getRun(d.t, d.ns, name)
	d.t.Fatalf("run %s did not reach a terminal phase in %d steps (phase=%q results: %s)",
		name, maxSteps, run.Status.Phase, summarise(run))
	return nil
}

// forbiddenHint turns an RBAC failure into the sentence that fixes it. A
// Forbidden reaching a reconcile is never a controller bug.
func forbiddenHint(err error) string {
	if err == nil || !apierrors.IsForbidden(err) {
		return ""
	}
	return "\n\nRBAC GAP: the operator's ClusterRole (config/rbac/role.yaml) does not grant a verb " +
		"the reconciler needs. Add the +kubebuilder:rbac marker beside the call and run `make manifests`."
}

func isTerminalPhase(p burninv1alpha1.RunPhase) bool {
	switch p {
	case burninv1alpha1.RunPassed, burninv1alpha1.RunFailed, burninv1alpha1.RunError, burninv1alpha1.RunCancelled:
		return true
	}
	return false
}

func summarise(run *burninv1alpha1.BurnInRun) string {
	if run == nil {
		return "<nil>"
	}
	parts := make([]string, 0, len(run.Status.Results))
	for _, res := range run.Status.Results {
		parts = append(parts, fmt.Sprintf("%s%v=%s(%s)", res.Name, res.Nodes, res.Phase, res.Message))
	}
	return strings.Join(parts, " | ")
}

// resultFor finds the result covering a node, the way the controller does.
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

// ─── Object builders ──────────────────────────────────────────────────────────

// customTest is a BurnInTest of kind "custom" — the one kind with no default
// runner image, which is what a test with no hardware wants: the image name is
// never resolved against a registry because envtest never runs the pod.
func customTest(ns, name string, mutate func(*burninv1alpha1.BurnInTest)) *burninv1alpha1.BurnInTest {
	t := &burninv1alpha1.BurnInTest{
		ObjectMeta: metav1.ObjectMeta{Name: name, Namespace: ns},
		Spec: burninv1alpha1.BurnInTestSpec{
			Kind:            burninv1alpha1.KindCustom,
			DurationSeconds: 5,
			Runner: &burninv1alpha1.RunnerSpec{
				Image:   "registry.invalid/never-pulled:v1",
				Command: []string{"/bin/true"},
			},
		},
	}
	if mutate != nil {
		mutate(t)
	}
	return t
}

func profile(ns, name string, sinks []string, tests ...burninv1alpha1.ProfileTest) *burninv1alpha1.BurnInProfile {
	return &burninv1alpha1.BurnInProfile{
		ObjectMeta: metav1.ObjectMeta{Name: name, Namespace: ns},
		Spec: burninv1alpha1.BurnInProfileSpec{
			Tests: tests,
			Sinks: sinks,
		},
	}
}

func runFor(ns, name, profileRef string, nodes []string, mutate func(*burninv1alpha1.BurnInRun)) *burninv1alpha1.BurnInRun {
	r := &burninv1alpha1.BurnInRun{
		ObjectMeta: metav1.ObjectMeta{Name: name, Namespace: ns},
		Spec: burninv1alpha1.BurnInRunSpec{
			ProfileRef: profileRef,
			Target:     burninv1alpha1.TargetSelector{NodeNames: nodes},
		},
	}
	if mutate != nil {
		mutate(r)
	}
	return r
}

func create(t *testing.T, objs ...client.Object) {
	t.Helper()
	for _, obj := range objs {
		if err := admin.Create(context.Background(), obj); err != nil {
			t.Fatalf("create %T %s: %v", obj, obj.GetName(), err)
		}
	}
}

func int32p(v int32) *int32 { return &v }
func boolp(v bool) *bool    { return &v }
