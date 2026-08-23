package controller

import (
	"context"
	"errors"
	"testing"

	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/types"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/client/interceptor"

	burninv1alpha1 "github.com/baldwinSPC/glimmer-burnin/api/v1alpha1"
	"github.com/baldwinSPC/glimmer-burnin/pkg/contract"
)

// Issue #461: terminate() marks the terminal status durable, then calls
// deleteLivePods and releaseCordons — either of which can fail and return
// early — and only AFTER both succeed did it used to set
// pendingDeliveryAnnotation. A failure in either call left the run durably
// terminal with the annotation never set, and reconcileTerminal's retry loop
// only fires when that annotation IS set — so the terminal envelope was
// silently and permanently undelivered, with the run otherwise reporting a
// correct verdict forever.
//
// This reproduces exactly that: a single failed node-release Update inside
// terminate()'s own releaseCordons call. CancelImmediate is the vehicle
// rather than an ordinary pass completion, because it reaches terminate()
// while the pod is still running — the node's cordon has not yet been
// released by the per-wave mechanism (cordon.go's own release, which fires
// as soon as a wave's pods finish and beats terminate() to it on an ordinary
// single-node run) — so terminate()'s own releaseCordons call is the one
// with something real left to release.
func TestRun_AFailedCordonReleaseDoesNotLoseTheTerminalDelivery(t *testing.T) {
	var releaseUpdates int
	failFirstRelease := func(inner client.Client) interceptor.Funcs {
		return interceptor.Funcs{
			Update: func(ctx context.Context, c client.WithWatch, obj client.Object, opts ...client.UpdateOption) error {
				// The precise signature of releaseCordons' own update, and
				// nothing else: it is the only write that REMOVES the cordon
				// stamp. cordonNode's "stamp first, cordon second" sequence
				// only ever ADDS it or flips Unschedulable with the stamp
				// already present.
				node, isNode := obj.(*corev1.Node)
				if isNode {
					var current corev1.Node
					if err := inner.Get(ctx, client.ObjectKeyFromObject(node), &current); err == nil {
						_, hadStamp := current.Annotations[burninv1alpha1.AnnotationCordonOwner]
						_, hasStamp := node.Annotations[burninv1alpha1.AnnotationCordonOwner]
						if hadStamp && !hasStamp {
							releaseUpdates++
							if releaseUpdates == 1 {
								return errors.New("injected: apiserver unavailable")
							}
						}
					}
				}
				return c.Update(ctx, obj, opts...)
			},
		}
	}

	h := newHarnessWithInterceptor(t, failFirstRelease,
		gb10Node("spark-a"),
		smokeTest("fp4"),
		profile("acceptance", []string{"results"}, false, testRef("fp4")),
		newRun("run1", "acceptance", "spark-a"),
	)
	if err := h.c.Create(context.Background(), cmSink("results", "results-cm")); err != nil {
		t.Fatal(err)
	}

	h.reconcile("run1")
	h.reconcile("run1")
	pods := h.pods("run1")
	if len(pods) != 1 {
		t.Fatalf("expected 1 pod before cancelling, got %d", len(pods))
	}
	h.startPod(pods["spark-a"])

	run := h.run("run1")
	run.Spec.Cancel = boolp(true)
	run.Spec.CancelPolicy = burninv1alpha1.CancelImmediate
	if err := h.c.Update(context.Background(), run); err != nil {
		t.Fatal(err)
	}

	// This pass reaches terminate() via cancel(): the status write durably
	// settles the run Cancelled, then releaseCordons' node Update fails (the
	// injected error) and terminate() returns early. Called directly,
	// bypassing h.reconcile, which treats any reconcile error as an
	// unconditional test failure — here the error is the point.
	req := ctrl.Request{NamespacedName: types.NamespacedName{Namespace: "burnin", Name: "run1"}}
	if _, err := h.r.Reconcile(context.Background(), req); err == nil {
		t.Fatal("setup: expected the injected node-update failure to surface as a reconcile error")
	}
	if got := h.run("run1").Status.Phase; !isTerminal(got) {
		t.Fatalf("phase = %q, want terminal — the verdict must be durable even though release failed", got)
	}
	if _, pending := h.run("run1").Annotations[pendingDeliveryAnnotation]; !pending {
		t.Fatal("issue #461: pendingDeliveryAnnotation was not set before the failing call, " +
			"so the terminal delivery this run owes would have been lost silently")
	}

	// The node update succeeds on every later call. Subsequent passes must
	// converge: release the cordon and deliver the terminal envelope the
	// annotation already promised. The RUNNING phase-change envelope (also
	// Reason=PhaseChanged) already delivered on the earlier passes, since
	// this test creates the sink up front — so the check below is specific
	// to the terminal phase, not just the reason.
	terminalDelivered := func() bool {
		for _, env := range h.envelopes("results-cm") {
			if env.Reason == contract.ReasonPhaseChanged && env.Phase == string(burninv1alpha1.RunCancelled) {
				return true
			}
		}
		return false
	}
	for i := 0; i < 5 && !terminalDelivered(); i++ {
		h.reconcile("run1")
	}
	if _, pending := h.run("run1").Annotations[pendingDeliveryAnnotation]; pending {
		t.Error("pending marker was not cleared once delivery converged")
	}
	if !terminalDelivered() {
		t.Error("the terminal Cancelled envelope was never delivered")
	}
}
