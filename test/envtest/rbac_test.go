package envtest

import (
	"context"
	"fmt"
	"path/filepath"
	"sync"
	"testing"

	authzv1 "k8s.io/api/authorization/v1"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/client-go/rest"
	"sigs.k8s.io/controller-runtime/pkg/client"

	burninv1alpha1 "github.com/baldwinSPC/glimmer-burnin/api/v1alpha1"
)

// RBAC, tested against an apiserver that actually enforces it.
//
// THIS IS THE FILE THAT WOULD HAVE CAUGHT THE MISSING `services` RULE. Pair
// scope creates a headless Service to rendezvous its two pods; the generated
// ClusterRole had no services rule, so every Pair test failed on a live cluster
// while the whole unit suite stayed green — controller-runtime's fake client
// has no authorizer and will create anything it is asked for.
//
// Two independent guards, because they fail in different ways:
//
//   - the RUN tests drive real reconciles through the SHIPPED service account,
//     so any verb the reconciler needs and does not have surfaces as a
//     Forbidden inside Reconcile. This half is self-maintaining: a new API call
//     in the controller is covered the day it is written.
//   - the SubjectAccessReview matrix covers what a reconcile in this
//     environment cannot reach — pod logs (injected into the reconciler),
//     leader-election leases, the other two controllers' resources — and it
//     asserts the NEGATIVES too, so nobody closes an RBAC gap by widening the
//     grant until it stops complaining.

const (
	shippedNamespace      = "glimmer-burnin-system"
	shippedServiceAccount = "burnin-controller-manager"
)

var shippedRBACOnce sync.Once

// installShippedRBAC applies config/rbac and config/manager exactly as a
// deployment would, and returns the username of the operator's own service
// account.
//
// It applies the SHIPPED files rather than a Go transcription: a transcription
// tests the transcription. The value here is that the ClusterRole under test is
// the one controller-gen wrote and CI checks for drift, and the leader-election
// Role is the one in the deployment manifest.
//
// Installed once for the package and never removed. These objects are
// cluster-scoped or live in a fixed namespace, so per-test creation would
// collide, and there is nothing to isolate: they are read-only inputs.
func installShippedRBAC(t *testing.T) string {
	t.Helper()
	shippedRBACOnce.Do(func() {
		applyManifestPermanent(t, filepath.Join(repoRoot, "config", "rbac", "role.yaml"), nil)
		applyManifestPermanent(t, filepath.Join(repoRoot, "config", "manager", "manager.yaml"), nil)
	})
	return fmt.Sprintf("system:serviceaccount:%s:%s", shippedNamespace, shippedServiceAccount)
}

// asShippedServiceAccount builds a client acting as the operator's own service
// account, holding exactly the permissions config/rbac and config/manager grant
// it and no others.
func asShippedServiceAccount(t *testing.T) client.Client {
	t.Helper()
	user := installShippedRBAC(t)
	impersonated := rest.CopyConfig(cfg)
	impersonated.Impersonate = rest.ImpersonationConfig{UserName: user}
	c, err := client.New(impersonated, client.Options{Scheme: scheme})
	if err != nil {
		t.Fatalf("build impersonated client: %v", err)
	}
	return c
}

// TestGeneratedRBACIsEnoughForANodeScopeRun drives a whole Node-scope run
// through the operator's own service account.
//
// Everything the reconciler touches on this path — the run and its status
// subresource, the profile, the test, the nodes it cordons and releases, the
// pods it creates and lists — goes through RBAC. A missing verb stops the run
// with a Forbidden instead of passing silently.
func TestGeneratedRBACIsEnoughForANodeScopeRun(t *testing.T) {
	ns := newNamespace(t)
	node := nodeName(t, "rbac-node")
	newNode(t, node)

	create(t,
		customTest(ns, "smoke", func(bt *burninv1alpha1.BurnInTest) {
			bt.Spec.Thresholds = []burninv1alpha1.Threshold{
				{Metric: "miscompares", Comparison: burninv1alpha1.EQ, Value: "0"},
			}
		}),
		profile(ns, "acceptance", nil, burninv1alpha1.ProfileTest{TestRef: "smoke"}),
		runFor(ns, "run", "acceptance", []string{node}, nil),
	)

	d := newDriver(t, ns)
	d.script("", script{stdout: "miscompares=0\n", exitCode: 0})
	d.reconcilerOver(asShippedServiceAccount(t))

	run := d.run("run", 40)
	if run.Status.Phase != burninv1alpha1.RunPassed {
		t.Fatalf("run phase = %q, want Passed: %s", run.Status.Phase, summarise(run))
	}
	if got := getNode(t, node); got.Spec.Unschedulable {
		t.Errorf("node %s is still cordoned after a completed run", node)
	}
}

// TestGeneratedRBACIsEnoughForAPairScopeRun is the regression test for the
// missing `services` rule.
//
// A Pair execution creates a headless Service before either pod exists. With no
// services rule the Create is Forbidden, startPair returns the error, and every
// Pair test on every cluster fails — which is what shipped.
func TestGeneratedRBACIsEnoughForAPairScopeRun(t *testing.T) {
	ns := newNamespace(t)
	server := nodeName(t, "pair-server")
	peer := nodeName(t, "pair-client")
	newNode(t, server)
	newNode(t, peer)

	create(t,
		customTest(ns, "fabric", func(bt *burninv1alpha1.BurnInTest) {
			bt.Spec.Scope = burninv1alpha1.ScopePair
		}),
		profile(ns, "fabric-suite", nil, burninv1alpha1.ProfileTest{TestRef: "fabric"}),
		runFor(ns, "pair-run", "fabric-suite", []string{server, peer}, func(r *burninv1alpha1.BurnInRun) {
			// A pair is one indivisible unit of load holding both of its nodes,
			// so it costs two slots and can never start at the default cap of 1.
			r.Spec.MaxConcurrentNodes = int32p(2)
		}),
	)

	sa := asShippedServiceAccount(t)
	d := newDriver(t, ns)
	d.script("", script{stdout: "bandwidthGbps=99.61\n", exitCode: 0})
	d.reconcilerOver(sa)

	run := d.run("pair-run", 60)
	if run.Status.Phase != burninv1alpha1.RunPassed {
		t.Fatalf("pair run phase = %q, want Passed: %s", run.Status.Phase, summarise(run))
	}

	// A pair is ONE result naming BOTH nodes — never one verdict per endpoint.
	if n := len(run.Status.Results); n != 1 {
		t.Fatalf("pair produced %d results, want exactly 1 (a link verdict is not per-node): %s", n, summarise(run))
	}
	if got := run.Status.Results[0].Nodes; len(got) != 2 || got[0] != server || got[1] != peer {
		t.Errorf("pair result nodes = %v, want [%s %s]", got, server, peer)
	}

	// The Service is the artefact the missing rule was about, and it is listed
	// through the SERVICE ACCOUNT's own client: read as admin, a run that could
	// not create one would still look fine here.
	var services corev1.ServiceList
	if err := sa.List(context.Background(), &services, client.InNamespace(ns)); err != nil {
		t.Fatalf("the operator's service account cannot list services: %v%s", err, forbiddenHint(err))
	}
	if len(services.Items) == 0 {
		t.Fatal("no headless Service was created for the pair rendezvous")
	}
	for _, svc := range services.Items {
		if svc.Spec.ClusterIP != corev1.ClusterIPNone {
			t.Errorf("rendezvous Service %s has clusterIP %q, want None — per-pod DNS is the whole point",
				svc.Name, svc.Spec.ClusterIP)
		}
	}
}

// TestShippedRBACCoversTheCallsAReconcileCannotExercise is the second guard: a
// SubjectAccessReview matrix over the verbs the runs above never reach.
//
// The positives are the calls that only happen on a real cluster. The negatives
// matter just as much: the cheapest way to make an RBAC failure go away is to
// grant more, and this is what keeps that from being invisible.
func TestShippedRBACCoversTheCallsAReconcileCannotExercise(t *testing.T) {
	user := installShippedRBAC(t)

	type check struct {
		group, resource, subresource, verb, namespace, why string
	}
	allowed := []check{
		// The runner's stdout IS the metrics channel; without this every verdict
		// would be exit-code-only. The reconciler takes it through an injected
		// func, so no reconcile in this package can prove the grant exists.
		{"", "pods", "log", "get", "default", "runner stdout is the metrics channel"},
		// Leader election. Namespaced, granted by config/manager's own Role, and
		// invisible to every test that does not start a manager.
		{"coordination.k8s.io", "leases", "", "get", shippedNamespace, "leader election"},
		{"coordination.k8s.io", "leases", "", "create", shippedNamespace, "leader election"},
		{"coordination.k8s.io", "leases", "", "update", shippedNamespace, "leader election"},
		{"", "events", "", "create", shippedNamespace, "NodeFingerprint records changes as events"},
		// The ConfigMap sink (airgapped export) and the webhook sink's bearer
		// token; neither is on the Node-scope path exercised above.
		{"", "configmaps", "", "create", "default", "ConfigMap sink"},
		{"", "configmaps", "", "update", "default", "ConfigMap sink"},
		{"", "secrets", "", "get", "default", "webhook bearer token"},
		// The other two reconcilers.
		{"burnin.glimmer.ai", "burninschedules", "", "watch", "default", "BurnInSchedule controller"},
		{"burnin.glimmer.ai", "burninschedules", "status", "update", "default", "BurnInSchedule controller"},
		{"burnin.glimmer.ai", "burninruns", "", "create", "default", "a schedule creates runs"},
		{"burnin.glimmer.ai", "nodefingerprints", "", "create", "default", "NodeFingerprint capture"},
		{"burnin.glimmer.ai", "nodefingerprints", "status", "update", "default", "NodeFingerprint capture"},
		{"burnin.glimmer.ai", "burninsinks", "status", "update", "default", "delivery health is recorded on the sink"},
		// Cordoning is a destructive edit to an object the operator does not own.
		{"", "nodes", "", "update", "", "cordon and release"},
		{"", "nodes", "", "watch", "", "target resolution by selector"},
		{"", "services", "", "create", "default", "Pair rendezvous"},
		{"", "services", "", "delete", "default", "Pair rendezvous cleanup"},
	}
	forbidden := []check{
		// One secret is read by name to resolve a sink token. Listing every
		// secret in the cluster is a different power, and cmd/main.go bypasses
		// the cache for Secrets precisely so the operator never needs it.
		{"", "secrets", "", "list", "default", "the manager bypasses the cache for Secrets to avoid needing this"},
		{"", "secrets", "", "watch", "default", "as above"},
		// The operator measures a fleet; it must not be able to remove nodes
		// from it, or to run commands inside a runner container.
		{"", "nodes", "", "delete", "", "an operator sent to measure a fleet must not delete its nodes"},
		{"", "pods", "exec", "create", "default", "the runner contract is an exit code and stdout, not a shell"},
		{"", "pods", "", "update", "default", "runner pods are created and deleted, never edited"},
		{"apps", "deployments", "", "get", "default", "the operator has no business with workloads"},
		{"rbac.authorization.k8s.io", "clusterroles", "", "create", "", "no privilege-escalation path"},
	}

	for _, c := range allowed {
		t.Run("allow/"+describePath(c.group, c.resource, c.subresource, c.verb), func(t *testing.T) {
			if !can(t, user, c.group, c.resource, c.subresource, c.verb, c.namespace) {
				t.Errorf("the shipped RBAC does NOT allow %s — needed for: %s",
					describePath(c.group, c.resource, c.subresource, c.verb), c.why)
			}
		})
	}
	for _, c := range forbidden {
		t.Run("deny/"+describePath(c.group, c.resource, c.subresource, c.verb), func(t *testing.T) {
			if can(t, user, c.group, c.resource, c.subresource, c.verb, c.namespace) {
				t.Errorf("the shipped RBAC ALLOWS %s and should not — %s",
					describePath(c.group, c.resource, c.subresource, c.verb), c.why)
			}
		})
	}
}

func describePath(group, resource, subresource, verb string) string {
	name := resource
	if group != "" {
		name = resource + "." + group
	}
	if subresource != "" {
		name += "_" + subresource
	}
	return verb + "_" + name
}

func can(t *testing.T, user, group, resource, subresource, verb, namespace string) bool {
	t.Helper()
	sar := &authzv1.SubjectAccessReview{
		Spec: authzv1.SubjectAccessReviewSpec{
			User: user,
			// The groups the apiserver attaches to a service-account request.
			// RBAC matches the ServiceAccount subject, but the review has to
			// describe the real request to be worth anything.
			Groups: []string{
				"system:authenticated",
				"system:serviceaccounts",
				"system:serviceaccounts:" + shippedNamespace,
			},
			ResourceAttributes: &authzv1.ResourceAttributes{
				Namespace:   namespace,
				Group:       group,
				Resource:    resource,
				Subresource: subresource,
				Verb:        verb,
			},
		},
	}
	if err := admin.Create(context.Background(), sar); err != nil {
		t.Fatalf("subjectaccessreview: %v", err)
	}
	return sar.Status.Allowed
}
