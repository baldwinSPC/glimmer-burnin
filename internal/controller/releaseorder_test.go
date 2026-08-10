package controller

import (
	"context"
	"fmt"
	"testing"
	"time"

	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/client/fake"
	"sigs.k8s.io/controller-runtime/pkg/client/interceptor"

	burninv1alpha1 "github.com/baldwinSPC/glimmer-burnin/api/v1alpha1"
)

// newHarnessWithInterceptor is newHarness with a client wrapper.
//
// The wrapper is built from the finished client rather than passed in, because
// an interceptor that needs to READ the store — as this one does, to ask what
// is running on a node at the moment it is released — cannot be constructed
// before the store exists.
func newHarnessWithInterceptor(t *testing.T, mk func(client.Client) interceptor.Funcs, objs ...client.Object) *harness {
	t.Helper()
	scheme := runtime.NewScheme()
	if err := corev1.AddToScheme(scheme); err != nil {
		t.Fatal(err)
	}
	if err := burninv1alpha1.AddToScheme(scheme); err != nil {
		t.Fatal(err)
	}
	build := func(funcs interceptor.Funcs) client.WithWatch {
		return fake.NewClientBuilder().
			WithScheme(scheme).
			WithObjects(objs...).
			WithStatusSubresource(&burninv1alpha1.BurnInRun{}, &burninv1alpha1.BurnInSink{}).
			WithInterceptorFuncs(funcs).
			Build()
	}
	// A plain client over the same fixtures, for the interceptor's own reads.
	// It shares nothing with the intercepted store, so a List through it cannot
	// re-enter the interceptor and recurse.
	h := &harness{
		t:      t,
		scheme: scheme,
		logs:   map[string]string{},
		nowVal: time.Unix(1750000000, 0).UTC(),
	}
	var inner client.Client
	c := build(interceptor.Funcs{
		Update: func(ctx context.Context, wc client.WithWatch, obj client.Object, opts ...client.UpdateOption) error {
			return mk(inner).Update(ctx, wc, obj, opts...)
		},
	})
	inner = c
	h.c = c
	h.r = h.newReconciler()
	return h
}

// The mirror of the #84 invariant, for issue #246.
//
// #84 says a pod may only be destroyed once the status justifying it is
// durable. This says the other half: A NODE MAY ONLY BE RETURNED TO THE
// SCHEDULER ONCE NOTHING OF THIS RUN IS STILL BURNING IT.
//
// Both are about `terminate`, and until #246 it satisfied only the first. It
// released the cordons, then delivered the terminal envelope — network I/O to
// somebody else's endpoint, bounded only by the 15s delivery timeout — and only
// then stopped the pods. For that whole window a node under a full-power
// burn-in was schedulable with this run's stamp already removed, so nothing in
// the cluster said it was loaded.
//
// The existing TestRun_FailFastCleansUpInFlightPods asserts the END STATE, and
// passes under either ordering; that is why it did not catch this. The
// assertion has to be made AT THE MOMENT OF RELEASE, which is what the
// interceptor below does.

// releaseWatcher fails the test if a node is made schedulable while a live pod
// of the run is still on it.
//
// It checks at the write, not afterwards, because afterwards is exactly the
// state that looks correct however wrong the ordering was.
type releaseWatcher struct {
	t          *testing.T
	violations []string
}

func (w *releaseWatcher) funcs(inner client.Client) interceptor.Funcs {
	return interceptor.Funcs{
		Update: func(ctx context.Context, c client.WithWatch, obj client.Object, opts ...client.UpdateOption) error {
			node, isNode := obj.(*corev1.Node)
			if !isNode || node.Spec.Unschedulable {
				// Not a release. A cordon going ON is the safe direction.
				return c.Update(ctx, obj, opts...)
			}

			var pods corev1.PodList
			if err := inner.List(ctx, &pods); err == nil {
				for i := range pods.Items {
					p := &pods.Items[i]
					if p.Labels[labelNode] != node.Name {
						continue
					}
					// NOT-TERMINATED is the test, rather than an allow-list
					// of live phases. A pod object that exists and has not
					// reached Succeeded or Failed is still holding the node —
					// including one with no phase set yet, which is what a
					// just-created pod looks like before its kubelet reports.
					// An allow-list missed exactly that case and made this
					// whole guard silently vacuous.
					switch p.Status.Phase {
					case corev1.PodSucceeded, corev1.PodFailed:
					default:
						phase := string(p.Status.Phase)
						if phase == "" {
							phase = "unstarted"
						}
						w.violations = append(w.violations, fmt.Sprintf(
							"%s was made schedulable while %s was still %s on it",
							node.Name, p.Name, phase))
					}
				}
			}
			return c.Update(ctx, obj, opts...)
		},
	}
}

func TestRun_ANodeIsNotHandedBackWhileItIsStillBeingBurned(t *testing.T) {
	w := &releaseWatcher{t: t}

	// Built in two steps: the watcher needs a client to List pods with, and the
	// interceptor needs the watcher. The inner client is the same fake store,
	// so the List sees exactly what the reconciler sees.
	h := newHarnessWithInterceptor(t, func(inner client.Client) interceptor.Funcs { return w.funcs(inner) },
		gb10Node("spark-a"), gb10Node("spark-b"),
		smokeTest("fp4"),
		profile("acceptance", nil, true, testRef("fp4")),
		withNodeCap(newRun("run1", "acceptance", "spark-a", "spark-b"), 2),
	)

	h.reconcile("run1")
	h.reconcile("run1")
	pods := h.pods("run1")
	if len(pods) != 2 {
		t.Fatalf("expected 2 pods, got %d", len(pods))
	}
	// Both pods actually RUNNING, which is the situation the invariant is
	// about. A pod the fake client never started has no phase at all, and an
	// earlier version of this test checked for Running and therefore passed
	// against the very ordering it was written to catch.
	h.startPod(pods["spark-a"])
	h.startPod(pods["spark-b"])

	// spark-a fails; spark-b is still running when failFast trips. This is the
	// ORDINARY fail-fast path — no conflict, no restart, no cache staleness.
	h.finishPod(pods["spark-a"], 1, "FP4_GEMM_FAIL\n", "Error")
	h.reconcileUntilSettled("run1")

	if run := h.run("run1"); run.Status.Phase != burninv1alpha1.RunFailed {
		t.Fatalf("phase = %q, want Failed", run.Status.Phase)
	}
	for _, v := range w.violations {
		t.Errorf("issue #246: %s", v)
	}
}

func TestRun_AnImmediateCancelStopsTheLoadBeforeGivingTheNodeBack(t *testing.T) {
	// The path this matters most on. CancelImmediate is documented as the tool
	// for a facility power or cooling event — where the load itself is the
	// reason for stopping — and it went straight through to terminate, so it
	// returned the nodes to service FIRST and stopped the load afterwards.
	w := &releaseWatcher{t: t}

	h := newHarnessWithInterceptor(t, func(inner client.Client) interceptor.Funcs { return w.funcs(inner) },
		gb10Node("spark-a"), gb10Node("spark-b"),
		smokeTest("fp4"),
		profile("acceptance", nil, false, testRef("fp4")),
		withNodeCap(newRun("run1", "acceptance", "spark-a", "spark-b"), 2),
	)

	h.reconcile("run1")
	h.reconcile("run1")
	pods := h.pods("run1")
	if len(pods) != 2 {
		t.Fatalf("expected 2 pods before cancelling")
	}
	h.startPod(pods["spark-a"])
	h.startPod(pods["spark-b"])

	run := h.run("run1")
	run.Spec.Cancel = boolp(true)
	run.Spec.CancelPolicy = burninv1alpha1.CancelImmediate
	if err := h.c.Update(context.Background(), run); err != nil {
		t.Fatal(err)
	}
	h.reconcileUntilSettled("run1")

	for _, v := range w.violations {
		t.Errorf("issue #246 on the CancelImmediate path: %s", v)
	}
}
