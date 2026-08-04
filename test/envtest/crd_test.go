package envtest

import (
	"context"
	"path/filepath"
	"strings"
	"testing"

	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/types"

	burninv1alpha1 "github.com/baldwinSPC/glimmer-burnin/api/v1alpha1"
)

// TestApiserverAppliesTheCRDDefaults checks the defaults the RECONCILER is
// written against.
//
// Every one of these has a matching re-application in Go (plan.go's "resolved
// knobs", pods.go's hostPathReadOnly) because the controller must not assume
// apiserver defaulting has happened. That belt-and-braces arrangement only
// works if both halves agree: a marker deleted from the Go type but left in the
// manifest, or the reverse, produces an object whose meaning depends on which
// path created it. The fake client applies no defaults at all, so this is the
// first place the manifest's half is exercised.
func TestApiserverAppliesTheCRDDefaults(t *testing.T) {
	ns := newNamespace(t)
	ctx := context.Background()

	test := customTest(ns, "defaults", func(bt *burninv1alpha1.BurnInTest) {
		// Deliberately unset: Scope, RepeatCount, and the readOnly of a mount.
		bt.Spec.Runner.HostPaths = []burninv1alpha1.HostPathMount{
			{Path: "/dev/kmsg", MountPath: "/dev/kmsg"},
		}
		bt.Spec.Thresholds = []burninv1alpha1.Threshold{
			{Metric: "eccErrors", Comparison: burninv1alpha1.EQ, Value: "0"},
		}
	})
	create(t, test)

	var got burninv1alpha1.BurnInTest
	if err := admin.Get(ctx, types.NamespacedName{Namespace: ns, Name: "defaults"}, &got); err != nil {
		t.Fatalf("get: %v", err)
	}
	if got.Spec.Scope != burninv1alpha1.ScopeNode {
		t.Errorf("scope defaulted to %q, want Node", got.Spec.Scope)
	}
	if got.Spec.RepeatCount == nil || *got.Spec.RepeatCount != 1 {
		t.Errorf("repeatCount defaulted to %v, want 1", got.Spec.RepeatCount)
	}
	// The one default whose direction is a security property: an unset
	// privilege grant has to fall towards the harmless form.
	if ro := got.Spec.Runner.HostPaths[0].ReadOnly; ro == nil || !*ro {
		t.Errorf("hostPaths[0].readOnly defaulted to %v, want true — an unspecified host mount must not be writable", ro)
	}
	if got.Spec.Thresholds[0].Applicability != burninv1alpha1.Required {
		t.Errorf("threshold applicability defaulted to %q, want Required",
			got.Spec.Thresholds[0].Applicability)
	}

	prof := profile(ns, "defaults", nil, burninv1alpha1.ProfileTest{TestRef: "defaults"})
	create(t, prof)
	var gotProfile burninv1alpha1.BurnInProfile
	if err := admin.Get(ctx, types.NamespacedName{Namespace: ns, Name: "defaults"}, &gotProfile); err != nil {
		t.Fatalf("get profile: %v", err)
	}
	if req := gotProfile.Spec.Tests[0].Required; req == nil || !*req {
		t.Errorf("profile test required defaulted to %v, want true", req)
	}

	run := runFor(ns, "defaults", "defaults", []string{"node-a"}, nil)
	create(t, run)
	var gotRun burninv1alpha1.BurnInRun
	if err := admin.Get(ctx, types.NamespacedName{Namespace: ns, Name: "defaults"}, &gotRun); err != nil {
		t.Fatalf("get run: %v", err)
	}
	// The facility interlock. A nil that fell through as "no cap" would turn it
	// off exactly where nobody would look.
	if got := gotRun.Spec.MaxConcurrentNodes; got == nil || *got != 1 {
		t.Errorf("maxConcurrentNodes defaulted to %v, want 1", got)
	}
	if got := gotRun.Spec.RetryOnErrorLimit; got == nil || *got != 0 {
		t.Errorf("retryOnErrorLimit defaulted to %v, want 0", got)
	}
	if got := gotRun.Spec.Suspend; got == nil || *got {
		t.Errorf("suspend defaulted to %v, want false", got)
	}
	if got := gotRun.Spec.Cancel; got == nil || *got {
		t.Errorf("cancel defaulted to %v, want false", got)
	}
	if gotRun.Spec.CancelPolicy != burninv1alpha1.CancelGraceful {
		t.Errorf("cancelPolicy defaulted to %q, want Graceful", gotRun.Spec.CancelPolicy)
	}
}

// TestApiserverRejectsWhatTheSchemaForbids is the half of validation the fake
// client cannot have: it accepts anything that compiles.
//
// Each case below is a rule the operator relies on downstream. A schema that
// stopped enforcing one would not fail any unit test — the run would simply
// execute a spec nobody wrote.
func TestApiserverRejectsWhatTheSchemaForbids(t *testing.T) {
	ns := newNamespace(t)

	cases := []struct {
		name string
		obj  func() *unstructured.Unstructured
		want string
	}{{
		name: "unknown scope",
		// Scope selects a whole execution strategy. An unenumerated value would
		// fall through settleWithoutPod's default arm and be recorded as an
		// unexecutable scope — an Error against hardware nobody tested.
		obj:  func() *unstructured.Unstructured { return rawTest(ns, "bad-scope", map[string]any{"scope": "Cluster"}) },
		want: "scope",
	}, {
		name: "missing kind",
		obj: func() *unstructured.Unstructured {
			u := rawTest(ns, "no-kind", nil)
			unstructured.RemoveNestedField(u.Object, "spec", "kind")
			return u
		},
		want: "kind",
	}, {
		name: "relative host path",
		// A relative host path names nothing in particular on a node. plan.go
		// refuses it too, but only after the run has started.
		obj: func() *unstructured.Unstructured {
			return rawTest(ns, "rel-hostpath", map[string]any{
				"runner": map[string]any{
					"image":     "example.invalid/x:v1",
					"hostPaths": []any{map[string]any{"path": "dev/kmsg", "mountPath": "/dev/kmsg"}},
				},
			})
		},
		want: "path",
	}, {
		name: "hostPath type the kubelet would CREATE",
		// The enum admits only assert-only types. An operator sent to MEASURE a
		// fleet must never be able to mutate it.
		obj: func() *unstructured.Unstructured {
			return rawTest(ns, "creating-hostpath", map[string]any{
				"runner": map[string]any{
					"image": "example.invalid/x:v1",
					"hostPaths": []any{map[string]any{
						"path": "/var/log/burnin", "mountPath": "/var/log/burnin", "type": "DirectoryOrCreate",
					}},
				},
			})
		},
		want: "type",
	}, {
		name: "duplicate mountPath",
		// listMapKey=mountPath. Two mounts at one container path is an invalid
		// pod spec, and the apiserver would refuse every attempt while the run
		// held a cordon.
		obj: func() *unstructured.Unstructured {
			return rawTest(ns, "dup-mount", map[string]any{
				"runner": map[string]any{
					"image": "example.invalid/x:v1",
					"hostPaths": []any{
						map[string]any{"path": "/dev/kmsg", "mountPath": "/host"},
						map[string]any{"path": "/dev/infiniband", "mountPath": "/host"},
					},
				},
			})
		},
		want: "Duplicate value",
	}, {
		name: "unknown comparison",
		// verdict.Evaluate switches on this. An unknown comparison would reach
		// its default arm and decide a node's fate there.
		obj: func() *unstructured.Unstructured {
			return rawTest(ns, "bad-cmp", map[string]any{
				"thresholds": []any{map[string]any{
					"metric": "eccErrors", "comparison": "Approximately", "value": "0",
				}},
			})
		},
		want: "comparison",
	}, {
		name: "negative durationSeconds",
		obj: func() *unstructured.Unstructured {
			return rawTest(ns, "neg-duration", map[string]any{"durationSeconds": int64(-1)})
		},
		want: "durationSeconds",
	}, {
		name: "maxConcurrentNodes below the interlock floor",
		obj: func() *unstructured.Unstructured {
			return rawRun(ns, "zero-cap", map[string]any{"maxConcurrentNodes": int64(0)})
		},
		want: "maxConcurrentNodes",
	}, {
		name: "profile with no tests",
		obj: func() *unstructured.Unstructured {
			u := &unstructured.Unstructured{Object: map[string]any{
				"apiVersion": "burnin.glimmer.ai/v1alpha1",
				"kind":       "BurnInProfile",
				"metadata":   map[string]any{"name": "empty", "namespace": ns},
				"spec":       map[string]any{"tests": []any{}},
			}}
			return u
		},
		want: "tests",
	}}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			err := admin.Create(context.Background(), tc.obj())
			if err == nil {
				t.Fatalf("apiserver ACCEPTED an object the schema must refuse; "+
					"the CRD in config/crd no longer enforces %q", tc.want)
			}
			if !strings.Contains(err.Error(), tc.want) {
				t.Errorf("rejected, but not for the expected reason (%q not in %q)", tc.want, err.Error())
			}
		})
	}
}

// TestStatusIsARealSubresource asserts the separation every write in the
// controller depends on.
//
// terminate() writes metadata and status in two calls and restores its
// in-memory status around the metadata write, precisely because the apiserver
// treats them as different resources. With a fake client that dance is
// unnecessary and untested; here it is load-bearing.
func TestStatusIsARealSubresource(t *testing.T) {
	ns := newNamespace(t)
	ctx := context.Background()

	run := runFor(ns, "subresource", "nothing", []string{"node-a"}, nil)
	run.Status.Phase = burninv1alpha1.RunPassed
	run.Status.Passed = 99
	create(t, run)

	got := getRun(t, ns, "subresource")
	// The apiserver DISCARDS what the create body claimed about status. A
	// controller that trusted a status written through the main resource would
	// be reading its own fiction.
	if got.Status.Passed != 0 || got.Status.Phase != "" {
		t.Errorf("status survived a create on the main resource (phase=%q passed=%d) — status is not a subresource",
			got.Status.Phase, got.Status.Passed)
	}

	// Writing status through the subresource, with phase left unset, is where
	// the CRD default lands. markRunning relies on nothing here, but the run's
	// printcolumn does: a run with no phase at all reads as a run nobody started.
	if err := admin.Status().Update(ctx, got); err != nil {
		t.Fatalf("status update: %v", err)
	}
	if defaulted := getRun(t, ns, "subresource"); defaulted.Status.Phase != burninv1alpha1.RunPending {
		t.Errorf("status.phase = %q after the first status write, want the defaulted Pending",
			defaulted.Status.Phase)
	}

	got = getRun(t, ns, "subresource")
	got.Status.Phase = burninv1alpha1.RunRunning
	got.Spec.CancelReason = "written through the status subresource"
	if err := admin.Status().Update(ctx, got); err != nil {
		t.Fatalf("status update: %v", err)
	}
	after := getRun(t, ns, "subresource")
	if after.Status.Phase != burninv1alpha1.RunRunning {
		t.Errorf("status.phase = %q after a status update, want Running", after.Status.Phase)
	}
	if after.Spec.CancelReason != "" {
		t.Errorf("a spec field rode along on a status update (%q) — the subresource is not isolating spec",
			after.Spec.CancelReason)
	}
}

// TestShippedSamplesAreAcceptedByARealApiserver applies config/samples against
// the generated CRDs.
//
// api/v1alpha1/samples_test.go already decodes them strictly against the Go
// types, which catches a misspelled field. It cannot catch the other half: a
// sample that the GENERATED SCHEMA rejects, or one whose fields the apiserver
// silently prunes because the CRD does not know about them. Samples are the
// first thing a new user applies, so this is the closest thing the repo has to
// a smoke test of its own documentation.
func TestShippedSamplesAreAcceptedByARealApiserver(t *testing.T) {
	ns := newNamespace(t)
	files, err := filepath.Glob(filepath.Join(repoRoot, "config", "samples", "*.yaml"))
	if err != nil {
		t.Fatalf("glob: %v", err)
	}
	if len(files) == 0 {
		t.Fatal("no samples found — config/samples is the documented starting point and must not be empty")
	}
	for _, file := range files {
		t.Run(filepath.Base(file), func(t *testing.T) {
			applied := applyManifest(t, file, func(u *unstructured.Unstructured) bool {
				u.SetNamespace(ns)
				return true
			})
			if len(applied) == 0 {
				t.Fatal("sample file produced no objects")
			}
		})
	}
}

// TestShippedDeploymentManifestIsAcceptedByARealApiserver applies
// config/manager, which is the manifest a release ships for deploying the
// operator at all.
//
// It once shipped as nothing: an unanchored `manager` line in .gitignore
// swallowed config/manager/ for the whole of v0.2.0. A test that creates these
// objects would not have caught that by itself — but the kind e2e that APPLIES
// this directory would have, and this is its cheap in-process counterpart:
// after the file is restored, it stays syntactically deployable.
func TestShippedDeploymentManifestIsAcceptedByARealApiserver(t *testing.T) {
	// installShippedRBAC applies config/rbac and config/manager for the whole
	// package. Creating and then deleting these objects per test is not an
	// option: envtest runs no namespace controller, so a deleted namespace
	// stays Terminating forever and every later create into it is refused.
	installShippedRBAC(t)

	objs, err := readManifestObjects(filepath.Join(repoRoot, "config", "manager", "manager.yaml"))
	if err != nil {
		t.Fatalf("read manager manifest: %v", err)
	}
	var kinds []string
	for _, obj := range objs {
		kinds = append(kinds, obj.GetKind())
	}
	for _, want := range []string{"Namespace", "ServiceAccount", "ClusterRoleBinding", "Deployment"} {
		if !contains(kinds, want) {
			t.Errorf("config/manager/manager.yaml has no %s — the operator cannot be deployed from it (kinds: %v)",
				want, kinds)
		}
	}
}

func contains(hay []string, needle string) bool {
	for _, s := range hay {
		if s == needle {
			return true
		}
	}
	return false
}

// ─── raw object helpers ───────────────────────────────────────────────────────

// rawTest builds a minimally valid BurnInTest as unstructured data, with spec
// overrides merged in. Unstructured is required for the rejection cases: a
// typed object cannot express "scope: Cluster" at all, so a typed test would be
// asserting the Go compiler rather than the CRD.
func rawTest(ns, name string, specOverrides map[string]any) *unstructured.Unstructured {
	spec := map[string]any{"kind": "custom"}
	for k, v := range specOverrides {
		spec[k] = v
	}
	return &unstructured.Unstructured{Object: map[string]any{
		"apiVersion": "burnin.glimmer.ai/v1alpha1",
		"kind":       "BurnInTest",
		"metadata":   map[string]any{"name": name, "namespace": ns},
		"spec":       spec,
	}}
}

func rawRun(ns, name string, specOverrides map[string]any) *unstructured.Unstructured {
	spec := map[string]any{
		"profileRef": "whatever",
		"target":     map[string]any{"nodeNames": []any{"node-a"}},
	}
	for k, v := range specOverrides {
		spec[k] = v
	}
	return &unstructured.Unstructured{Object: map[string]any{
		"apiVersion": "burnin.glimmer.ai/v1alpha1",
		"kind":       "BurnInRun",
		"metadata":   map[string]any{"name": name, "namespace": ns},
		"spec":       spec,
	}}
}
