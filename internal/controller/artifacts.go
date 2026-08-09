package controller

import (
	"context"
	"fmt"
	"unicode/utf8"

	corev1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"

	burninv1alpha1 "github.com/baldwinSPC/glimmer-burnin/api/v1alpha1"
	"github.com/baldwinSPC/glimmer-burnin/pkg/runner"
)

// Storing the evidence a runner returned that is not a number — issue #143.
//
// # An artifact is evidence ABOUT a verdict, never part of one
//
// That single sentence decides every trade-off in this file. Storing an artifact
// is a second API write against an object the apiserver may reject for reasons
// that have nothing to do with the hardware — the ConfigMap is full, the run was
// deleted mid-flight, a quota is exhausted. If any of those could change a
// phase, then a node would be condemned or reprieved by the state of a storage
// object, which is exactly the confusion `Error` is kept distinct from `Fail` to
// prevent. So every failure here is recorded as a dropped ArtifactRef and
// returns no error to the caller.
//
// The converse is equally deliberate: a refusal is RECORDED and not merely
// swallowed. "The runner produced no artifact" and "the artifact was too large
// to keep" must not be the same observation, because only one of them is a
// reason to go and look at the node.
//
// # Why a ConfigMap
//
// It is the one durable, namespaced, RBAC-governed store this operator already
// has, it garbage-collects with the run through an owner reference, and it needs
// no egress credentials in a component whose job is running semi-trusted
// workloads on cordoned nodes. Its ceiling is what bounds the budget below.

const (
	// maxArtifactBytes bounds ONE artifact. It is pkg/runner's cap restated,
	// because the same number is enforced on both sides of the boundary: the
	// parser refuses to carry an oversized payload and the store refuses to
	// write one, and neither may rely on the other having looked.
	maxArtifactBytes = runner.MaxArtifactBytes

	// maxRunArtifactBytes bounds a RUN's whole artifact ConfigMap.
	//
	// etcd's value limit is 1 MiB and the apiserver enforces it on the encoded
	// object, so the budget leaves headroom for keys, metadata and the encoding
	// itself. It matches internal/sink's own ConfigMap ceiling for the same
	// reason and by the same arithmetic.
	maxRunArtifactBytes = 900 * 1024

	// artifactConfigMapSuffix names the run-owned ConfigMap.
	artifactConfigMapSuffix = "-artifacts"
)

// artifactConfigMapName is the run-owned ConfigMap holding every artifact of
// every test in the run.
//
// One per RUN rather than one per attempt: a run of forty tests would otherwise
// leave forty objects to garbage-collect and forty to look in, and the budget
// that actually matters — how much of this cluster's etcd one burn-in may
// consume — could then only be expressed forty times over.
func artifactConfigMapName(run *burninv1alpha1.BurnInRun) string {
	return run.Name + artifactConfigMapSuffix
}

// storeArtifacts persists what a runner fenced on stdout and returns references
// to it, in the order the runner emitted them.
//
// It is called from harvestPod, which is the single choke point every scope
// passes through — and, critically, BEFORE the pod is deleted. A container's log
// is the only copy of its stdout, and the kubelet reclaims it with the pod.
//
// It never returns an error. See the file comment: no failure to store evidence
// may reach a verdict.
func (r *BurnInRunReconciler) storeArtifacts(
	ctx context.Context,
	run *burninv1alpha1.BurnInRun,
	testName string,
	podName string,
	artifacts []runner.Artifact,
) []burninv1alpha1.ArtifactRef {
	if len(artifacts) == 0 {
		return nil
	}

	cmName := artifactConfigMapName(run)
	cm := &corev1.ConfigMap{}
	err := r.Get(ctx, types.NamespacedName{Namespace: run.Namespace, Name: cmName}, cm)
	switch {
	case apierrors.IsNotFound(err):
		cm = &corev1.ConfigMap{
			ObjectMeta: metav1.ObjectMeta{
				Namespace: run.Namespace,
				Name:      cmName,
				Labels:    map[string]string{labelRun: run.Name},
				// Owned by the run, so the evidence is collected exactly when the
				// verdict it explains is. An artifact outliving its run is
				// unattributable; one dying before it is a verdict nobody can check.
				OwnerReferences: []metav1.OwnerReference{
					*metav1.NewControllerRef(run,
						burninv1alpha1.GroupVersion.WithKind("BurnInRun")),
				},
			},
		}
	case err != nil:
		return dropAll(artifacts, fmt.Sprintf("could not read %s: %v", cmName, err))
	}

	used := configMapBytes(cm)
	refs := make([]burninv1alpha1.ArtifactRef, 0, len(artifacts))
	wrote := false

	for _, a := range artifacts {
		ref := burninv1alpha1.ArtifactRef{
			Name:      a.Name,
			MediaType: a.MediaType,
			SizeBytes: int64(a.SizeBytes),
		}
		// A drop decided by the parser stands: it already knows why, and
		// re-deciding it here could only disagree.
		if a.Dropped != "" {
			ref.Dropped = a.Dropped
			refs = append(refs, ref)
			continue
		}

		key, keyErr := artifactKey(testName, podName, a.Name)
		if keyErr != nil {
			ref.Dropped = keyErr.Error()
			refs = append(refs, ref)
			continue
		}
		switch {
		case len(a.Payload) > maxArtifactBytes:
			// Unreachable through pkg/runner, which refuses the same size. Kept
			// because this function's contract is with its own caller, and a
			// caller that assembled an Artifact by hand must not be able to
			// hand the apiserver an object it will reject.
			ref.Dropped = fmt.Sprintf("payload is %d bytes, over the %d-byte per-artifact cap",
				len(a.Payload), maxArtifactBytes)
			refs = append(refs, ref)
			continue
		case used+len(key)+len(a.Payload) > maxRunArtifactBytes:
			ref.Dropped = fmt.Sprintf(
				"the run's %d-byte artifact budget is exhausted (%d used); evidence not kept",
				maxRunArtifactBytes, used)
			refs = append(refs, ref)
			continue
		}

		setArtifactData(cm, key, a.Payload)
		used += len(key) + len(a.Payload)
		wrote = true

		ref.Digest = a.Digest
		ref.ConfigMap = cmName
		ref.Key = key
		refs = append(refs, ref)
	}

	if !wrote {
		return refs
	}
	if err := r.upsertConfigMap(ctx, cm); err != nil {
		// The payloads did not land, so every ref that claimed one is now a
		// promise the operator cannot keep. Downgrading them to drops is the
		// honest outcome: a dangling reference would send a reader to a key
		// that is not there, which reads as data loss rather than as a write
		// that failed.
		for i := range refs {
			if refs[i].Dropped == "" {
				refs[i].ConfigMap, refs[i].Key, refs[i].Digest = "", "", ""
				refs[i].Dropped = fmt.Sprintf("could not write %s: %v", cmName, err)
			}
		}
	}
	return refs
}

func (r *BurnInRunReconciler) upsertConfigMap(ctx context.Context, cm *corev1.ConfigMap) error {
	if cm.ResourceVersion == "" {
		err := r.Create(ctx, cm)
		if apierrors.IsAlreadyExists(err) {
			// Lost a race with another reconcile of this run. The next harvest
			// reads the winner and adds to it; nothing is corrupted, and this
			// attempt's refs are downgraded to drops by the caller.
			return err
		}
		return err
	}
	return r.Update(ctx, cm)
}

// setArtifactData writes the payload where it can survive a round trip.
//
// ConfigMap.Data is a map[string]string, and a runner's stdout carries no
// guarantee of valid UTF-8: a captured blob, or a log truncated mid-rune, is
// exactly the evidence somebody fenced.
//
// The failure mode is NOT a rejected write, which is what makes it dangerous.
// Measured against a real apiserver in test/envtest: the write is ACCEPTED, and
// the bytes round-trip intact to a protobuf client — but the same value read
// back over JSON, which is what kubectl and every REST consumer uses, comes back
// with each invalid byte replaced by U+FFFD. Silently. The evidence still looks
// present, one byte per corrupted byte, and the only thing that would reveal it
// is the digest in the ArtifactRef — which now fails to match, so a consumer
// doing the right thing concludes the evidence was tampered with.
//
// BinaryData is []byte and is base64-encoded on the wire, so it is exact for
// every consumer. Text still goes in Data, so `kubectl get cm -o yaml` shows a
// dcgmi document as a document rather than as base64.
func setArtifactData(cm *corev1.ConfigMap, key, payload string) {
	if utf8.ValidString(payload) {
		if cm.Data == nil {
			cm.Data = map[string]string{}
		}
		cm.Data[key] = payload
		delete(cm.BinaryData, key)
		return
	}
	if cm.BinaryData == nil {
		cm.BinaryData = map[string][]byte{}
	}
	cm.BinaryData[key] = []byte(payload)
	delete(cm.Data, key)
}

func configMapBytes(cm *corev1.ConfigMap) int {
	n := 0
	for k, v := range cm.Data {
		n += len(k) + len(v)
	}
	for k, v := range cm.BinaryData {
		n += len(k) + len(v)
	}
	return n
}

// artifactKey names the entry, and REFUSES a name it cannot make safe.
//
// The name comes from a runner's stdout, which is the least trusted input this
// operator has: a runner is a third-party image executing on a cordoned node.
// A ConfigMap key admits only alphanumerics, '-', '_' and '.', so a name with a
// '/' in it would make the apiserver reject the whole object and lose every
// artifact in the write — one runner's malformed name silently discarding
// another test's evidence.
//
// It is a REFUSAL and not a sanitisation. Rewriting "a/b" to "a_b" would make
// two different names collide into one key, so the second artifact would
// overwrite the first and the ref would point at the wrong payload while
// carrying the right digest. Better to keep neither and say so.
func artifactKey(testName, podName, artifactName string) (string, error) {
	if artifactName == "" {
		return "", fmt.Errorf("the artifact has no name")
	}
	if !validConfigMapKey(artifactName) {
		return "", fmt.Errorf(
			"the runner's artifact name %q is not usable as a ConfigMap key "+
				"(alphanumerics, '-', '_' and '.' only)", artifactName)
	}
	// The pod name carries the attempt: it is unique per attempt, which is what
	// keeps a repeat from overwriting the evidence of the attempt before it and
	// keeps a Pair's two endpoints apart.
	key := fmt.Sprintf("%s.%s.%s", testName, podName, artifactName)
	if len(key) > 253 {
		return "", fmt.Errorf("the artifact's key would be %d bytes, over the 253-byte limit", len(key))
	}
	return key, nil
}

func validConfigMapKey(s string) bool {
	if s == "." || s == ".." {
		return false
	}
	for _, c := range s {
		switch {
		case c >= 'a' && c <= 'z', c >= 'A' && c <= 'Z', c >= '0' && c <= '9':
		case c == '-', c == '_', c == '.':
		default:
			return false
		}
	}
	return true
}

// dropAll records every artifact as refused for one shared reason.
func dropAll(artifacts []runner.Artifact, reason string) []burninv1alpha1.ArtifactRef {
	refs := make([]burninv1alpha1.ArtifactRef, 0, len(artifacts))
	for _, a := range artifacts {
		d := a.Dropped
		if d == "" {
			d = reason
		}
		refs = append(refs, burninv1alpha1.ArtifactRef{
			Name: a.Name, MediaType: a.MediaType,
			SizeBytes: int64(a.SizeBytes), Dropped: d,
		})
	}
	return refs
}

// recordArtifacts attaches refs to the test's result.
//
// It is called AFTER completeAttempt, which is what guarantees the result
// exists: completeAttempt opens it through beginAttempt, and a ref recorded
// against a result that does not exist yet would be dropped silently.
//
// Refs ACCUMULATE across attempts, which is deliberately unlike Metrics,
// Violations and NotEvaluated — those are overwritten by the deciding attempt so
// the numbers always explain the phase beside them. A ref is not a measurement:
// it is a pointer to a payload that is already in the ConfigMap and stays there
// until the run is collected. Replacing the list on a retry would not reclaim
// one byte; it would only make the evidence of the earlier attempt unreachable,
// which is the worst of both — stored and unfindable. Each Key carries the pod
// name, so which attempt produced what is never in doubt.
func recordArtifacts(res *burninv1alpha1.TestResult, refs []burninv1alpha1.ArtifactRef) bool {
	if res == nil || len(refs) == 0 {
		return false
	}
	seen := make(map[string]bool, len(res.Artifacts))
	for _, a := range res.Artifacts {
		seen[artifactIdentity(a)] = true
	}
	dirty := false
	for _, a := range refs {
		id := artifactIdentity(a)
		if seen[id] {
			// A pod is harvested on every reconcile until it is deleted, so the
			// same artifact arrives repeatedly. Without this the list grows
			// without bound for as long as the pod lingers.
			continue
		}
		seen[id] = true
		res.Artifacts = append(res.Artifacts, a)
		dirty = true
	}
	return dirty
}

// artifactIdentity is what makes a re-harvest of the same pod idempotent. A
// dropped ref has no Key, so it falls back to the name it was refused under —
// otherwise every reconcile would append another copy of the same refusal.
func artifactIdentity(a burninv1alpha1.ArtifactRef) string {
	if a.Key != "" {
		return "k/" + a.Key
	}
	return "d/" + a.Name + "\x00" + a.Dropped
}

// harvestArtifacts stores a pod's artifacts and records the refs on the result,
// in one call, for the three harvest paths.
func (r *BurnInRunReconciler) harvestArtifacts(
	ctx context.Context,
	run *burninv1alpha1.BurnInRun,
	testName string,
	// nodes is the EXECUTION UNIT's node set — one node at Node scope, both ends
	// of a Pair, every rank of a Group.
	//
	// It has to be threaded through, and the reason is issue #231. This used
	// resultFor(run, testName, ""), and the empty-node form of resultFor returns
	// the FIRST result whose name matches — a form its own doc says exists for
	// node-LESS results (an unsupported scope, a resolution error). At Node scope
	// every target's result shares the test name, so all N resolved to
	// Results[0]: one node's verdict accumulated everybody's evidence and the
	// rest reported none.
	//
	// The payloads were stored correctly throughout, since artifactKey embeds the
	// pod name. Only the attribution was wrong, which is the worse failure —
	// evidence that LOOKS present against the wrong hardware sends an engineer to
	// the wrong rack, and is not recoverable from the delivered envelope.
	nodes []string,
	podName string,
	artifacts []runner.Artifact,
) {
	if len(artifacts) == 0 || podName == "" {
		return
	}
	// The empty-node lookup is kept for exactly the case it was written for: a
	// result that belongs to no node at all.
	node := ""
	if len(nodes) > 0 {
		node = nodes[0]
	}
	recordArtifacts(resultFor(run, testName, node), r.storeArtifacts(ctx, run, testName, podName, artifacts))
}
