package controller

import (
	"context"

	"k8s.io/apimachinery/pkg/types"
	"sigs.k8s.io/controller-runtime/pkg/log"

	burninv1alpha1 "github.com/baldwinSPC/glimmer-burnin/api/v1alpha1"
)

// vendorFor is the accelerator vendor of a node, for selecting a runner image.
//
// It reads the node's NodeFingerprint, which derives the vendor from a device
// label's DNS domain with no branch on the value. That is the whole reason this
// is a lookup rather than detection logic living here: the fingerprint
// controller already answers "what is this machine", and a second answer in the
// run controller is a second thing to keep true.
//
// IT NEVER FAILS THE RUN. An unreadable or absent fingerprint yields the empty
// vendor, and the empty vendor is handled where the consequence is:
// runnerImage, which refuses at plan time IF the test actually needs a vendor
// to resolve an image and answers the default otherwise. Returning an error
// here would make every single-vendor fleet — which is nearly all of them, and
// every fleet that worked before this field existed — depend on a fingerprint
// object that has nothing to contribute.
func (r *BurnInRunReconciler) vendorFor(ctx context.Context, node string) string {
	// No namespace means nobody told this operator where fingerprints live, so
	// there is nothing to read. Not an error: it is the pre-imagesByVendor
	// behaviour, and a test that actually needs a vendor still fails loudly in
	// runnerImage rather than silently getting the wrong image.
	if node == "" || r.FingerprintNamespace == "" {
		return ""
	}

	var fp burninv1alpha1.NodeFingerprint
	key := types.NamespacedName{Namespace: r.FingerprintNamespace, Name: fingerprintName(node)}
	if err := r.Get(ctx, key, &fp); err != nil {
		// Logged at V(1), not surfaced. On a fleet with no accelerators, or one
		// where the fingerprint controller has not caught up with a new node,
		// this is the ordinary case and not a problem — it becomes a problem
		// only if a test asked for imagesByVendor, and that is reported loudly
		// where it happens.
		log.FromContext(ctx).V(1).Info("no fingerprint for node; runner image will resolve without a vendor",
			"node", node, "error", err.Error())
		return ""
	}
	return vendorOf(&fp)
}

// vendorOf reads the vendor out of a fingerprint.
//
// GPUs[0], because a node's accelerators are the same part in every fleet this
// operator is built for, and a machine with two vendors of accelerator in it is
// not a case the image-per-vendor model can express anyway — one pod gets one
// image. Split out from vendorFor so the reading is testable without a client,
// and so the day a mixed-accelerator node turns up there is one place to change.
func vendorOf(fp *burninv1alpha1.NodeFingerprint) string {
	if fp == nil || len(fp.Status.GPUs) == 0 {
		return ""
	}
	// "unknown" is what the fingerprint records when it saw a device it could
	// not attribute. Normalised to empty here because a BurnInTest cannot
	// declare an image for it — the CRD's vendor enum rejects "unknown" — so
	// carrying it forward would produce a lookup that can never hit, and an
	// error message blaming a vendor nobody wrote.
	if v := fp.Status.GPUs[0].Vendor; v != "unknown" {
		return v
	}
	return ""
}
