// Package controller holds the reconcilers for the Glimmer Burn-In Operator.
package controller

import (
	"context"

	"k8s.io/apimachinery/pkg/runtime"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/log"

	burninv1alpha1 "github.com/baldwinSPC/glimmer-burnin/api/v1alpha1"
)

// BurnInRunReconciler drives a BurnInRun from Pending to a terminal verdict:
// resolve the profile, capture the target NodeFingerprint, schedule each test's
// pod(s) (Node/Pair/Group scope), evaluate thresholds against parsed metrics,
// then export the verdict to the profile's sinks.
//
// This scaffold establishes the reconcile loop, RBAC, and owner-ref shape; the
// per-kind runner scheduling + result parsing land in follow-up PRs (they are
// the substance the CRDs were designed around).
type BurnInRunReconciler struct {
	client.Client
	Scheme *runtime.Scheme
}

// +kubebuilder:rbac:groups=burnin.glimmer.ai,resources=burninruns,verbs=get;list;watch;create;update;patch;delete
// +kubebuilder:rbac:groups=burnin.glimmer.ai,resources=burninruns/status,verbs=get;update;patch
// +kubebuilder:rbac:groups=burnin.glimmer.ai,resources=burninruns/finalizers,verbs=update
// +kubebuilder:rbac:groups=burnin.glimmer.ai,resources=burninprofiles;burnintests;burninsinks;nodefingerprints,verbs=get;list;watch
// +kubebuilder:rbac:groups=burnin.glimmer.ai,resources=nodefingerprints/status,verbs=get;update;patch
// +kubebuilder:rbac:groups="",resources=pods,verbs=get;list;watch;create;delete
// +kubebuilder:rbac:groups="",resources=nodes,verbs=get;list;watch
// +kubebuilder:rbac:groups="",resources=configmaps,verbs=get;list;watch;create;update;patch

// Reconcile is the entrypoint. Scaffold: it records that the run was observed
// and moves a Pending run to Running. Real scheduling is a follow-up.
func (r *BurnInRunReconciler) Reconcile(ctx context.Context, req ctrl.Request) (ctrl.Result, error) {
	logger := log.FromContext(ctx)

	var run burninv1alpha1.BurnInRun
	if err := r.Get(ctx, req.NamespacedName, &run); err != nil {
		return ctrl.Result{}, client.IgnoreNotFound(err)
	}

	if run.Status.Phase == "" {
		run.Status.Phase = burninv1alpha1.RunPending
	}
	if run.Status.ObservedGeneration != run.Generation {
		run.Status.ObservedGeneration = run.Generation
		logger.Info("observed BurnInRun", "profile", run.Spec.ProfileRef, "phase", run.Status.Phase)
		if err := r.Status().Update(ctx, &run); err != nil {
			return ctrl.Result{}, err
		}
	}

	return ctrl.Result{}, nil
}

// SetupWithManager wires the reconciler to BurnInRun events.
func (r *BurnInRunReconciler) SetupWithManager(mgr ctrl.Manager) error {
	return ctrl.NewControllerManagedBy(mgr).
		For(&burninv1alpha1.BurnInRun{}).
		Named("burninrun").
		Complete(r)
}
