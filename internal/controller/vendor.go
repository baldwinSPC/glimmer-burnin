package controller

import (
	"context"

	"k8s.io/apimachinery/pkg/types"
	"sigs.k8s.io/controller-runtime/pkg/log"

	burninv1alpha1 "github.com/baldwinSPC/glimmer-burnin/api/v1alpha1"
)

// device is what a node said about its accelerator: the two lookup keys image
// resolution uses, carried together.
//
// A struct rather than two adjacent string parameters because they are two
// adjacent strings — `podForTest(..., vendor, arch, ...)` transposed is a
// change no compiler catches, that resolves to no image on every node, and
// that reads as a missing entry in somebody's YAML. Neither field is ever
// branched on; they are compared and nothing else.
type device struct {
	vendor string
	arch   string
}

// deviceFor is what a node's accelerator is, for selecting a runner image.
//
// It reads the node's NodeFingerprint, which derives the vendor from a device
// label's DNS domain and records the architecture the node reported, with no
// branch on either value. That is the whole reason this is a lookup rather than
// detection logic living here: the fingerprint controller already answers "what
// is this machine", and a second answer in the run controller is a second thing
// to keep true.
//
// IT NEVER FAILS THE RUN. An unreadable or absent fingerprint yields the zero
// device, and that is handled where the consequence is: runnerImage, which
// refuses at plan time IF the test actually needs a vendor or an arch to
// resolve an image, and answers the default otherwise. Returning an error here
// would make every single-vendor fleet — which is nearly all of them, and every
// fleet that worked before these fields existed — depend on a fingerprint
// object that has nothing to contribute.
//
// Both keys come from ONE read.
//
// Answers a zero device when no fingerprint namespace is configured, there is
// no fingerprint object, or the fingerprint has nothing to say. That is not an error and never blocks a run — it
// resolves to the vendor's unqualified image entry, which is where every entry
// written before architectures were selectable lives, so a fleet that has never
// fingerprinted an arch behaves exactly as it did before this existed.
func (r *BurnInRunReconciler) deviceFor(ctx context.Context, node string) device {
	if node == "" || r.FingerprintNamespace == "" {
		return device{}
	}
	var fp burninv1alpha1.NodeFingerprint
	key := types.NamespacedName{Namespace: r.FingerprintNamespace, Name: fingerprintName(node)}
	if err := r.Get(ctx, key, &fp); err != nil {
		// Logged at V(1) and not surfaced: on a fleet
		// with no accelerators, or one where the fingerprint controller has not
		// caught up with a new node, this is ordinary. It becomes a problem only
		// if a test asked for an image by vendor or arch, and that is reported
		// loudly where it happens.
		log.FromContext(ctx).V(1).Info("no fingerprint for node; runner image will resolve without vendor or architecture",
			"node", node, "error", err.Error())
		return device{}
	}
	return device{vendor: vendorOf(&fp), arch: archOf(&fp)}
}

// archOf reads the architecture out of a fingerprint.
//
// GPUs[0], for the same reason vendorOf takes GPUs[0]: one pod gets one image,
// so a node whose accelerators differ from each other is not a case this model
// can express. The day that node turns up, both functions change together.
func archOf(fp *burninv1alpha1.NodeFingerprint) string {
	if fp == nil || len(fp.Status.GPUs) == 0 {
		return ""
	}
	return fp.Status.GPUs[0].Arch
}

// vendorOf reads the vendor out of a fingerprint.
//
// GPUs[0], because a node's accelerators are the same part in every fleet this
// operator is built for, and a machine with two vendors of accelerator in it is
// not a case the image-per-vendor model can express anyway — one pod gets one
// image. Split out from deviceFor so the reading is testable without a client,
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
