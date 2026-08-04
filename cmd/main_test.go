package main

import (
	"bytes"
	"io"
	"os"
	"path/filepath"
	"testing"

	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	utilyaml "k8s.io/apimachinery/pkg/util/yaml"
)

// The manager's own wiring, which nothing else in the suite can see.
//
// internal/controller tests the reconcilers, test/envtest tests them against a
// real apiserver, and test/e2e tests the deployed operator — but the options
// this binary hands to ctrl.NewManager sit between all three, and until now had
// no test at all. Each assertion below is about a decision another part of the
// system already depends on.

// TestSecretsAreKeptOutOfTheManagerCache guards the RBAC shape.
//
// A cached Secret read starts a cluster-wide Secret informer, which needs
// list+watch on EVERY secret in the cluster. The operator reads exactly one
// secret, by name, to resolve a sink's bearer token. test/envtest asserts the
// negative side of this — that the shipped ClusterRole does NOT grant secrets
// list or watch — and that assertion is only safe because of this option.
func TestSecretsAreKeptOutOfTheManagerCache(t *testing.T) {
	opts := managerOptions(":8080", ":8081", true, "glimmer-burnin-system")
	if opts.Client.Cache == nil {
		t.Fatal("no client cache options: Secrets would be read through the cache, which needs " +
			"list+watch RBAC on every secret in the cluster")
	}
	found := false
	for _, obj := range opts.Client.Cache.DisableFor {
		if _, ok := obj.(*corev1.Secret); ok {
			found = true
		}
	}
	if !found {
		t.Error("Secrets are not excluded from the manager cache — the operator would need " +
			"secrets list+watch cluster-wide to resolve one sink token by name")
	}
}

// TestLeaderElectionNamespaceIsExplicitRatherThanInferred.
//
// controller-runtime will otherwise resolve the namespace by reading the
// in-cluster service-account file. That works inside the shipped Deployment and
// fails everywhere else, and it is the one place this operator was still
// guessing at a namespace — the NodeFingerprint controller refuses to, and
// treats an unresolvable namespace as fatal rather than silently disabling
// itself.
//
// It matters beyond tidiness because the Lease is namespaced and
// config/manager grants leases in ONE namespace: electing in a different one
// fails as a permission error that reads like a bug in the operator.
func TestLeaderElectionNamespaceIsExplicitRatherThanInferred(t *testing.T) {
	opts := managerOptions(":8080", ":8081", true, "glimmer-burnin-system")
	if opts.LeaderElectionNamespace != "glimmer-burnin-system" {
		t.Errorf("LeaderElectionNamespace = %q, want the namespace POD_NAMESPACE supplied",
			opts.LeaderElectionNamespace)
	}
	if !opts.LeaderElection {
		t.Error("LeaderElection is off with --leader-elect set")
	}
	if opts.LeaderElectionID != leaderElectionID {
		t.Errorf("LeaderElectionID = %q, want %q — two managers that disagree about the Lease name "+
			"are two managers that both believe they are the leader",
			opts.LeaderElectionID, leaderElectionID)
	}

	// An empty POD_NAMESPACE keeps controller-runtime's own inference, so a pod
	// that does not project the downward API behaves exactly as before.
	if got := managerOptions(":8080", ":8081", true, "").LeaderElectionNamespace; got != "" {
		t.Errorf("LeaderElectionNamespace = %q with no POD_NAMESPACE, want the empty value that "+
			"leaves the existing inference in place", got)
	}
}

// TestTheShippedDeploymentProjectsPodNamespace closes the loop.
//
// The two settings above read POD_NAMESPACE, and so does the NodeFingerprint
// controller. A Go test that only checks the reading half would pass forever
// against a Deployment that never supplies it.
func TestTheShippedDeploymentProjectsPodNamespace(t *testing.T) {
	dep := shippedManagerContainer(t)

	env, found, err := unstructured.NestedSlice(dep, "env")
	if err != nil || !found {
		t.Fatal("the manager container declares no env at all, so POD_NAMESPACE cannot reach it")
	}
	for _, raw := range env {
		e, ok := raw.(map[string]any)
		if !ok || e["name"] != "POD_NAMESPACE" {
			continue
		}
		path, _, _ := unstructured.NestedString(e, "valueFrom", "fieldRef", "fieldPath")
		if path != "metadata.namespace" {
			t.Errorf("POD_NAMESPACE comes from %q, want the downward API's metadata.namespace — "+
				"a hard-coded namespace is wrong the moment the operator is installed elsewhere", path)
		}
		return
	}
	t.Error("config/manager/manager.yaml does not project POD_NAMESPACE. Leader election and the " +
		"NodeFingerprint capture both read it, and both fall back to guessing without it.")
}

// TestTheShippedDeploymentToleratesTheOperatorsOwnCordon.
//
// The manager has to keep reconciling while the nodes it is burning in are
// cordoned — including, on a small cluster, the node it is itself running on.
// Losing this toleration makes the operator unschedulable exactly when it is
// needed, and the symptom is a fleet of stranded cordons with nothing left
// running to release them.
func TestTheShippedDeploymentToleratesTheOperatorsOwnCordon(t *testing.T) {
	spec := shippedManagerPodSpec(t)
	tolerations, _, _ := unstructured.NestedSlice(spec, "tolerations")
	want := map[string]bool{
		"node-role.kubernetes.io/control-plane": false,
		"node.kubernetes.io/unschedulable":      false,
	}
	for _, raw := range tolerations {
		tol, ok := raw.(map[string]any)
		if !ok {
			continue
		}
		if key, _ := tol["key"].(string); key != "" {
			if _, expected := want[key]; expected {
				want[key] = true
			}
		}
	}
	for key, present := range want {
		if !present {
			t.Errorf("the manager does not tolerate %s — it becomes unschedulable exactly when a "+
				"burn-in is holding nodes out of the scheduler", key)
		}
	}
}

// ─── manifest reading ─────────────────────────────────────────────────────────

func shippedManagerPodSpec(t *testing.T) map[string]any {
	t.Helper()
	raw, err := os.ReadFile(filepath.Join("..", "config", "manager", "manager.yaml"))
	if err != nil {
		t.Fatalf("read config/manager/manager.yaml: %v — the operator's deployment manifest is "+
			"missing, which is how a release once shipped with no way to install it: %v", err, err)
	}
	dec := utilyaml.NewYAMLOrJSONDecoder(bytes.NewReader(raw), 4096)
	for {
		obj := &unstructured.Unstructured{}
		err := dec.Decode(obj)
		if err == io.EOF {
			break
		}
		if err != nil {
			t.Fatalf("decode manager manifest: %v", err)
		}
		if obj.GetKind() != "Deployment" {
			continue
		}
		spec, found, err := unstructured.NestedMap(obj.Object, "spec", "template", "spec")
		if err != nil || !found {
			t.Fatalf("the Deployment has no pod template spec: %v", err)
		}
		return spec
	}
	t.Fatal("config/manager/manager.yaml declares no Deployment")
	return nil
}

func shippedManagerContainer(t *testing.T) map[string]any {
	t.Helper()
	spec := shippedManagerPodSpec(t)
	containers, found, err := unstructured.NestedSlice(spec, "containers")
	if err != nil || !found {
		t.Fatal("the Deployment declares no containers")
	}
	for _, raw := range containers {
		c, ok := raw.(map[string]any)
		if !ok {
			continue
		}
		if name, _ := c["name"].(string); name == "manager" {
			return c
		}
	}
	t.Fatal("the Deployment has no container named \"manager\"")
	return nil
}
