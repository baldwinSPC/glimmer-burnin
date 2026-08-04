package controller

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"maps"
	"os"
	"reflect"
	"slices"
	"sort"
	"strconv"
	"strings"
	"time"

	corev1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/api/meta"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/validation"
	"k8s.io/client-go/tools/record"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/builder"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/event"
	"sigs.k8s.io/controller-runtime/pkg/handler"
	"sigs.k8s.io/controller-runtime/pkg/log"
	"sigs.k8s.io/controller-runtime/pkg/predicate"
	"sigs.k8s.io/controller-runtime/pkg/reconcile"

	burninv1alpha1 "github.com/baldwinSPC/glimmer-burnin/api/v1alpha1"
)

// NodeFingerprintReconciler captures each Node's hardware identity as a
// NodeFingerprint and flags drift.
//
// The point of the object is to make a verdict checkable after the fact. A
// BurnInRun says "this node passed"; the fingerprint says what "this node"
// was made of when it did. Without it, an acceptance record is a claim about a
// name, and names outlive hardware — a re-imaged, re-carded or re-flashed
// machine keeps its hostname and inherits an acceptance it never earned.
//
// Two rules shape the implementation:
//
//   - Everything is DERIVED FROM FACTS the node itself publishes. There is no
//     vendor branch here; the vendor is read off the DNS domain of the label
//     that published the fact, so supporting a new accelerator means that
//     vendor's device plugin publishing labels, not this file growing an `if`.
//
//   - Drift is REPORTED, never acted on. The reconciler writes a condition and
//     an Event and stops. Launching runs on drift is policy, and policy that
//     saturates accelerators across a fleet is not something a hash comparison
//     gets to trigger on its own.
//
// It also carries the orphaned-cordon reaper, which is not a fingerprinting
// concern and is here for one reason: this is the only reconciler that watches
// Nodes. A stranded cordon has to be found on a cluster where no BurnInRun is
// reconciling — that is the definition of the state — so the sweep cannot live
// in the run reconciler. See cordonreaper.go.
type NodeFingerprintReconciler struct {
	client.Client
	Scheme *runtime.Scheme

	// APIReader reads straight from the apiserver, bypassing the informer
	// cache. The manager wires it from mgr.GetAPIReader(); nil falls back to the
	// cached client.
	//
	// It exists for ONE read: the reaper's check that a cordon stamp's owning
	// run is gone. A cache miss is not an absence — an informer that has not yet
	// observed a freshly created run reports exactly that — and reaping on it
	// would release a node a live run is driving at full power. Nothing else
	// here needs it; fingerprinting a stale node costs a pass and is corrected
	// by the next one.
	APIReader client.Reader

	// Recorder emits the drift Event. Nil is tolerated (the condition is still
	// written) so the reconciler is usable without a manager.
	Recorder record.EventRecorder

	// Namespace is where NodeFingerprints are created. NodeFingerprint is
	// namespaced but Node is not, so the operator has to be told; it is
	// resolved from the downward API in SetupWithManager rather than guessed,
	// because writing cluster state into a namespace nobody chose is how
	// objects end up somewhere no one looks.
	Namespace string

	// Now is the clock, swappable in tests. Nil means time.Now.
	Now func() time.Time
}

func (r *NodeFingerprintReconciler) now() time.Time {
	if r.Now != nil {
		return r.Now()
	}
	return time.Now()
}

// uncached is the reader for the reaper's absence check. It degrades to the
// cached client when no APIReader is wired, so a reconciler built by hand still
// works — but the manager always wires one.
func (r *NodeFingerprintReconciler) uncached() client.Reader {
	if r.APIReader != nil {
		return r.APIReader
	}
	return r.Client
}

// digestSchemeVersion identifies the hashing scheme. It is part of both the
// hash input and the stored value, so a change to what counts as salient
// re-baselines every node instead of announcing a fleet-wide hardware change.
// Bump it whenever the set of digested facts changes.
const digestSchemeVersion = "v1"

// maxGPUsPerFingerprint bounds how many GPUInfo entries one label may
// materialise. A malformed or hostile `gpu.count` must not be able to inflate
// a status object without limit; the raw label is retained in Status.Labels
// either way, so nothing is lost from the record or from the digest.
const maxGPUsPerFingerprint = 64

// serviceAccountNamespaceFile is the in-cluster namespace, used when
// POD_NAMESPACE is not injected.
const serviceAccountNamespaceFile = "/var/run/secrets/kubernetes.io/serviceaccount/namespace"

// +kubebuilder:rbac:groups=burnin.glimmer.ai,resources=nodefingerprints,verbs=get;list;watch;create;update;patch
// +kubebuilder:rbac:groups=burnin.glimmer.ai,resources=nodefingerprints/status,verbs=get;update;patch
// +kubebuilder:rbac:groups="",resources=events,verbs=create;patch
// The reaper reads the run a cordon stamp names, and edits the node to give it
// back, so this controller needs more than the read access fingerprinting alone
// would call for. See cordonreaper.go.
// +kubebuilder:rbac:groups="",resources=nodes,verbs=get;list;watch;update;patch
// +kubebuilder:rbac:groups=burnin.glimmer.ai,resources=burninruns,verbs=get;list;watch

// Reconcile captures one node's identity. The request names a Node; the
// fingerprint is derived, compared with the stored one, and written only when
// something actually changed.
//
// It also reaps an orphaned burn-in cordon, and does that FIRST. The two jobs
// are independent, and the reap must not be reachable only through a successful
// fingerprint: a node whose fingerprint cannot be written — a name collision, a
// rejected status update — is still a node that may be stranded, and stranded
// capacity is the more urgent of the two.
func (r *NodeFingerprintReconciler) Reconcile(ctx context.Context, req ctrl.Request) (ctrl.Result, error) {
	logger := log.FromContext(ctx)

	var node corev1.Node
	if err := r.Get(ctx, types.NamespacedName{Name: req.Name}, &node); err != nil {
		if apierrors.IsNotFound(err) {
			// The node is gone. The fingerprint stays: it is the baseline a
			// past verdict was bound to, and deleting it would destroy the
			// record of what hardware that verdict describes — including in the
			// case that matters most, a machine pulled from the fleet because
			// it failed. Reaping stale fingerprints is a retention policy, and
			// this is not the code that should invent one.
			return ctrl.Result{}, nil
		}
		return ctrl.Result{}, err
	}

	// res carries the reaper's re-examination timer, which every return below
	// has to preserve: dropping it on the way out of a fingerprint no-op would
	// stop the only thing that ever comes back to look at a stamped node.
	res, err := r.reapCordon(ctx, &node)
	if err != nil {
		return ctrl.Result{}, err
	}

	fp, err := r.ensureFingerprint(ctx, &node)
	if err != nil || fp == nil {
		return res, err
	}

	desired := observeNode(&node)
	desired.Digest = fingerprintDigest(&desired)
	desired.ObservedGeneration = fp.Generation

	before := fp.Status.DeepCopy()
	desired.Conditions = slices.Clone(before.Conditions)

	// CapturedAt marks the last CHANGE, not the last look: nodes emit status
	// heartbeats constantly, and a timestamp that moved with them would rewrite
	// every fingerprint in the cluster every few seconds.
	if sameObservation(before, &desired) && before.CapturedAt != nil {
		desired.CapturedAt = before.CapturedAt
	} else {
		captured := metav1.NewTime(r.now())
		desired.CapturedAt = &captured
	}

	drift := r.classify(before.Digest, desired.Digest)
	change := ""
	if drift == driftChanged {
		change = describeDrift(before, &desired)
	}
	applyDriftCondition(&desired, drift, change, fp.Generation)

	if reflect.DeepEqual(*before, desired) {
		return res, nil
	}
	fp.Status = desired
	if err := r.Status().Update(ctx, fp); err != nil {
		return ctrl.Result{}, err
	}

	if drift == driftChanged {
		logger.Info("node hardware identity drifted",
			"node", node.Name, "fingerprint", fp.Name, "change", change)
		r.event(fp, corev1.EventTypeWarning, burninv1alpha1.EventReasonHardwareDrift,
			fmt.Sprintf("hardware identity of node %s changed since the baseline: %s "+
				"— verdicts recorded against the previous fingerprint no longer describe this node",
				node.Name, change))
	}
	return res, nil
}

// ensureFingerprint returns the NodeFingerprint for a node, creating it if it
// does not exist. A nil object with a nil error means the fingerprint that
// occupies the name belongs to a different node and must not be touched.
func (r *NodeFingerprintReconciler) ensureFingerprint(ctx context.Context, node *corev1.Node) (*burninv1alpha1.NodeFingerprint, error) {
	key := types.NamespacedName{Namespace: r.Namespace, Name: fingerprintName(node.Name)}

	var fp burninv1alpha1.NodeFingerprint
	err := r.Get(ctx, key, &fp)
	switch {
	case apierrors.IsNotFound(err):
		fp = burninv1alpha1.NodeFingerprint{
			ObjectMeta: metav1.ObjectMeta{
				Namespace: key.Namespace,
				Name:      key.Name,
				Labels:    map[string]string{labelNode: node.Name},
			},
			Spec: burninv1alpha1.NodeFingerprintSpec{NodeName: node.Name},
		}
		// No owner reference to the Node on purpose: garbage collection would
		// delete the fingerprint the moment the node object goes away, which is
		// exactly when the record of what the node WAS becomes most valuable.
		// AlreadyExists is returned rather than swallowed: it means another
		// writer got there first, and the retry re-reads what they wrote
		// instead of this pass proceeding against an object it never saw.
		if createErr := r.Create(ctx, &fp); createErr != nil {
			return nil, createErr
		}
	case err != nil:
		return nil, err
	case fp.Spec.NodeName != node.Name:
		// Someone else's object sits at this name. Overwriting it would
		// republish one machine's hardware under another machine's fingerprint,
		// which is the single worst thing this controller could do — every
		// verdict bound to it would then describe the wrong hardware. Refuse,
		// and say so; retrying cannot fix a name collision.
		log.FromContext(ctx).Error(
			fmt.Errorf("%s names node %q", key, fp.Spec.NodeName),
			"refusing to overwrite a NodeFingerprint that belongs to a different node",
			"node", node.Name)
		return nil, nil
	}
	return &fp, nil
}

func (r *NodeFingerprintReconciler) event(fp *burninv1alpha1.NodeFingerprint, eventType, reason, message string) {
	if r.Recorder == nil {
		return
	}
	r.Recorder.Event(fp, eventType, reason, message)
}

// ─── Drift classification ─────────────────────────────────────────────────────

type driftVerdict int

const (
	// driftBaseline: nothing to compare against yet.
	driftBaseline driftVerdict = iota
	// driftStable: same scheme, same digest.
	driftStable
	// driftChanged: same scheme, different digest — the hardware moved.
	driftChanged
	// driftIncomparable: different scheme — we cannot say, so we do not.
	driftIncomparable
)

func (r *NodeFingerprintReconciler) classify(stored, current string) driftVerdict {
	switch {
	case stored == "":
		return driftBaseline
	case stored == current:
		return driftStable
	case digestScheme(stored) != digestScheme(current):
		return driftIncomparable
	default:
		return driftChanged
	}
}

// applyDriftCondition writes the Drifted condition. The True state is sticky:
// once a node has drifted, a later stable pass does not clear it, because the
// fingerprint has by then been rewritten to describe the drifted hardware and
// "stable" would only mean "stable since we accepted the change".
func applyDriftCondition(status *burninv1alpha1.NodeFingerprintStatus, drift driftVerdict, change string, generation int64) {
	if drift != driftChanged && meta.IsStatusConditionTrue(status.Conditions, burninv1alpha1.ConditionNodeFingerprintDrifted) {
		return
	}

	cond := metav1.Condition{
		Type:               burninv1alpha1.ConditionNodeFingerprintDrifted,
		Status:             metav1.ConditionFalse,
		ObservedGeneration: generation,
	}
	switch drift {
	case driftBaseline:
		cond.Reason = burninv1alpha1.ReasonFingerprintBaselineCaptured
		cond.Message = "baseline captured; there is no earlier fingerprint to compare against"
	case driftStable:
		cond.Reason = burninv1alpha1.ReasonFingerprintStable
		cond.Message = "salient hardware identity is unchanged since the baseline"
	case driftIncomparable:
		cond.Reason = burninv1alpha1.ReasonFingerprintSchemeChanged
		cond.Message = "stored digest was produced by a different digest scheme; re-baselined without any claim about the hardware"
	case driftChanged:
		cond.Status = metav1.ConditionTrue
		cond.Reason = burninv1alpha1.ReasonFingerprintDigestChanged
		cond.Message = clampConditionMessage("salient hardware identity changed since the baseline: " + change)
	}
	meta.SetStatusCondition(&status.Conditions, cond)
}

// describeDrift names what moved, so the condition and the Event say more than
// "a hash changed" — a digest nobody can decode is a signal nobody can triage.
func describeDrift(before, after *burninv1alpha1.NodeFingerprintStatus) string {
	var parts []string
	if before.Kernel != after.Kernel {
		parts = append(parts, fmt.Sprintf("kernel %s → %s", orNone(before.Kernel), orNone(after.Kernel)))
	}
	if before.OSImage != after.OSImage {
		parts = append(parts, fmt.Sprintf("osImage %s → %s", orNone(before.OSImage), orNone(after.OSImage)))
	}
	if b, a := summarizeGPUs(before.GPUs), summarizeGPUs(after.GPUs); b != a {
		parts = append(parts, fmt.Sprintf("gpus %s → %s", b, a))
	}
	if b, a := summarizeNICs(before.NICs), summarizeNICs(after.NICs); b != a {
		parts = append(parts, fmt.Sprintf("nics %s → %s", b, a))
	}
	for _, k := range slices.Sorted(maps.Keys(unionKeys(before.Labels, after.Labels))) {
		b, hadBefore := before.Labels[k]
		a, hasNow := after.Labels[k]
		switch {
		case hadBefore && !hasNow:
			parts = append(parts, fmt.Sprintf("label %s=%s removed", k, b))
		case !hadBefore && hasNow:
			parts = append(parts, fmt.Sprintf("label %s=%s added", k, a))
		case b != a:
			parts = append(parts, fmt.Sprintf("label %s %s → %s", k, b, a))
		}
	}
	if len(parts) == 0 {
		return "no field-level difference is visible; the digest scheme covers facts not reproduced in status"
	}
	return strings.Join(parts, "; ")
}

func unionKeys(a, b map[string]string) map[string]struct{} {
	out := make(map[string]struct{}, len(a)+len(b))
	for k := range a {
		out[k] = struct{}{}
	}
	for k := range b {
		out[k] = struct{}{}
	}
	return out
}

func orNone(s string) string {
	if s == "" {
		return "(none)"
	}
	return s
}

// clampConditionMessage keeps a message inside the API server's 32KiB limit
// with room to spare; a status write rejected for length would lose the drift
// record entirely.
func clampConditionMessage(s string) string {
	const limit = 1024
	if len(s) <= limit {
		return s
	}
	return s[:limit-1] + "…"
}

func summarizeGPUs(gpus []burninv1alpha1.GPUInfo) string {
	if len(gpus) == 0 {
		return "(none)"
	}
	counts := map[string]int{}
	for _, g := range gpus {
		key := strings.TrimSpace(strings.Join([]string{g.Vendor, g.Model, g.Arch}, " "))
		if key == "" {
			key = "(unidentified)"
		}
		counts[key]++
	}
	out := make([]string, 0, len(counts))
	for _, k := range slices.Sorted(maps.Keys(counts)) {
		out = append(out, fmt.Sprintf("%d×%s", counts[k], k))
	}
	return strings.Join(out, ", ")
}

func summarizeNICs(nics []burninv1alpha1.NICInfo) string {
	if len(nics) == 0 {
		return "(none)"
	}
	out := make([]string, 0, len(nics))
	for _, n := range nics {
		if n.Role != "" {
			out = append(out, n.Name+"/"+n.Role)
			continue
		}
		out = append(out, n.Name)
	}
	return strings.Join(out, ", ")
}

// ─── Observation ──────────────────────────────────────────────────────────────

// observeNode reads a node's hardware identity. Everything it returns comes
// from the node itself: kubelet-reported system info, and labels published by
// whatever discovery runs on the cluster (NFD, a device plugin, a site probe).
func observeNode(node *corev1.Node) burninv1alpha1.NodeFingerprintStatus {
	return burninv1alpha1.NodeFingerprintStatus{
		OSImage: node.Status.NodeInfo.OSImage,
		Kernel:  node.Status.NodeInfo.KernelVersion,
		Labels:  salientLabels(node.Labels),
		GPUs:    gpusFromLabels(node.Labels),
		NICs:    nicsFromLabels(node.Labels),
	}
}

// sameObservation compares the observed identity, ignoring bookkeeping
// (CapturedAt, conditions, generation) so those cannot cause a write of their
// own.
func sameObservation(a, b *burninv1alpha1.NodeFingerprintStatus) bool {
	return a.OSImage == b.OSImage &&
		a.Kernel == b.Kernel &&
		a.Digest == b.Digest &&
		maps.Equal(a.Labels, b.Labels) &&
		slices.Equal(a.GPUs, b.GPUs) &&
		slices.Equal(a.NICs, b.NICs)
}

// salientLabelKeys are whole keys that identify the machine shape.
var salientLabelKeys = map[string]bool{
	corev1.LabelArchStable:             true, // kubernetes.io/arch
	"beta.kubernetes.io/arch":          true,
	corev1.LabelInstanceTypeStable:     true, // node.kubernetes.io/instance-type
	"beta.kubernetes.io/instance-type": true,
}

// salientLabelNamePrefixes match the NAME part of a label key — the part after
// the domain — so they hold for any publisher: `nvidia.com/gpu.count`,
// `amd.com/gpu.count` and a future vendor's `gpu.count` are all matched by one
// entry, with no vendor named anywhere.
//
// This is a vocabulary, not a policy: every match is handled identically. What
// is deliberately ABSENT matters as much as what is present — storage, network
// addressing, topology, scheduling state and taints are not here, because no
// TestKind this operator ships measures them, so changing one invalidates no
// verdict and must not raise a drift alarm.
var salientLabelNamePrefixes = []string{
	"accelerator",
	"board.", "board-",
	"cpu.", "cpu-",
	"cuda.", "cuda-",
	"gpu.", "gpu-", "gpus.",
	"hw.", "hw-",
	"infiniband",
	"kernel-version.",
	"memory.", "memory-",
	"mig.", "mig-",
	"nic.", "nic-",
	"nvlink",
	"pci-",
	"rdma.", "rdma-",
	"rocm.", "rocm-",
	"system-os",
}

// isSalientLabel reports whether a label states a hardware fact.
func isSalientLabel(key string) bool {
	domain, name := splitLabelKey(key)
	// This operator's own domain carries verdicts and run bookkeeping, never
	// hardware facts. Folding them in would make the fingerprint drift in
	// response to the burn-in that reads it.
	if domain == "burnin.glimmer.ai" || strings.HasSuffix(domain, ".burnin.glimmer.ai") {
		return false
	}
	name = strings.ToLower(name)
	if strings.HasPrefix(name, "burnin") || strings.Contains(name, "-burnin") || strings.Contains(name, ".burnin") {
		return false
	}
	if salientLabelKeys[key] {
		return true
	}
	for _, p := range salientLabelNamePrefixes {
		if strings.HasPrefix(name, p) {
			return true
		}
	}
	return false
}

func salientLabels(labels map[string]string) map[string]string {
	out := map[string]string{}
	for k, v := range labels {
		if isSalientLabel(k) {
			out[k] = v
		}
	}
	if len(out) == 0 {
		// nil, not an empty map: the field is omitempty, so an empty map does
		// not survive a round trip through the API server and would make every
		// comparison against the stored status report a spurious difference.
		return nil
	}
	return out
}

func splitLabelKey(key string) (domain, name string) {
	if i := strings.Index(key, "/"); i >= 0 {
		return key[:i], key[i+1:]
	}
	return "", key
}

// isNonVendorDomain reports whether a label domain belongs to a publisher that
// is not the silicon vendor — the Kubernetes ecosystem (NFD and friends) or
// this project. Facts from those domains are still used; they just do not name
// a vendor, so they are merged into whatever the vendor domains reported
// instead of inventing an accelerator made by "feature".
func isNonVendorDomain(domain string) bool {
	if domain == "" {
		return true
	}
	return domain == "kubernetes.io" || strings.HasSuffix(domain, ".kubernetes.io") ||
		domain == "glimmer.ai" || strings.HasSuffix(domain, ".glimmer.ai")
}

// vendorFromDomain reads the vendor off the domain that published the fact:
// `nvidia.com/gpu.product` was published by nvidia. That is where the vendor
// name comes from — not from a table in this file.
func vendorFromDomain(domain string) string {
	if i := strings.Index(domain, "."); i > 0 {
		return strings.ToLower(domain[:i])
	}
	return strings.ToLower(domain)
}

// gpuFacts is what one publishing domain said about the node's accelerators.
type gpuFacts struct {
	vendor      string
	model       string
	arch        string
	family      string
	driver      string
	driverMajor string
	driverMinor string
	driverRev   string
	memory      string
	count       string
}

// gpuField maps a label's name part onto a fact. The spellings are the ones
// device plugins and NFD actually publish; adding a spelling is adding data.
func gpuField(name string) string {
	switch strings.ToLower(name) {
	case "gpu.product", "gpu.model", "gpu.name", "gpu-product", "gpu-model":
		return "model"
	case "gpu.count", "gpu-count", "gpu.replicas":
		return "count"
	case "gpu.memory", "gpu-memory", "gpu.vram":
		return "memory"
	case "gpu.arch", "gpu-arch", "gpu.architecture", "gpu.compute-capability", "cuda.compute-capability":
		return "arch"
	case "gpu.family", "gpu-family":
		return "family"
	case "gpu.vendor", "gpu-vendor":
		return "vendor"
	case "gpu.driver-version", "gpu.driver", "gpu.driver.version", "cuda.driver-version":
		return "driver"
	case "cuda.driver.major", "gpu.driver.major":
		return "driver.major"
	case "cuda.driver.minor", "gpu.driver.minor":
		return "driver.minor"
	case "cuda.driver.rev", "gpu.driver.rev":
		return "driver.rev"
	}
	return ""
}

func (f *gpuFacts) set(field, value string) {
	switch field {
	case "model":
		f.model = value
	case "count":
		f.count = value
	case "memory":
		f.memory = value
	case "arch":
		f.arch = value
	case "family":
		f.family = value
	case "vendor":
		f.vendor = value
	case "driver":
		f.driver = value
	case "driver.major":
		f.driverMajor = value
	case "driver.minor":
		f.driverMinor = value
	case "driver.rev":
		f.driverRev = value
	}
}

// fillFrom completes a vendor's facts with unattributed ones (from NFD or from
// this project's own labels), never overwriting what the vendor itself said.
func (f *gpuFacts) fillFrom(other *gpuFacts) {
	if f.model == "" {
		f.model = other.model
	}
	if f.arch == "" {
		f.arch = other.arch
	}
	if f.family == "" {
		f.family = other.family
	}
	if f.memory == "" {
		f.memory = other.memory
	}
	if f.count == "" {
		f.count = other.count
	}
	if f.driver == "" {
		f.driver = other.driver
	}
	if f.driverMajor == "" {
		f.driverMajor, f.driverMinor, f.driverRev = other.driverMajor, other.driverMinor, other.driverRev
	}
}

// gpusFromLabels materialises the accelerator inventory the node advertises.
func gpusFromLabels(labels map[string]string) []burninv1alpha1.GPUInfo {
	byDomain := map[string]*gpuFacts{}
	unattributed := &gpuFacts{}

	for k, v := range labels {
		domain, name := splitLabelKey(k)
		field := gpuField(name)
		if field == "" {
			continue
		}
		target := unattributed
		if !isNonVendorDomain(domain) {
			if byDomain[domain] == nil {
				byDomain[domain] = &gpuFacts{vendor: vendorFromDomain(domain)}
			}
			target = byDomain[domain]
		}
		target.set(field, v)
	}

	groups := make([]*gpuFacts, 0, len(byDomain))
	for _, f := range byDomain {
		if f.model != "" || f.count != "" {
			groups = append(groups, f)
		}
	}
	if len(groups) == 0 && (unattributed.model != "" || unattributed.count != "") {
		// Only NFD or this project's own labels described the accelerators.
		// They are still facts about the node; they just do not name a vendor,
		// and "unknown" is the honest value for a field nothing published.
		groups = append(groups, unattributed)
	}
	// Filling from unattributed facts is only sound when there is one group:
	// with two vendors present, an unqualified `gpu-arch` names one of them and
	// copying it onto both would attribute an architecture to hardware that
	// never claimed it.
	if len(groups) == 1 {
		groups[0].fillFrom(unattributed)
	}
	sort.Slice(groups, func(i, j int) bool {
		if groups[i].vendor != groups[j].vendor {
			return groups[i].vendor < groups[j].vendor
		}
		return groups[i].model < groups[j].model
	})

	var out []burninv1alpha1.GPUInfo
	index := int32(0)
	for _, f := range groups {
		n, known := gpuCount(f.count)
		if !known {
			if f.model == "" && f.arch == "" && f.memory == "" {
				continue
			}
			// The node says an accelerator of this model is present but not how
			// many. One is what is proven; recording nothing would lose the
			// model entirely. The raw count label — credible or not — is still
			// in Status.Labels and therefore still in the digest, so a count
			// that later changes is detected regardless.
			n = 1
		}
		arch := f.arch
		if arch == "" {
			arch = f.family
		}
		for i := 0; i < n; i++ {
			out = append(out, burninv1alpha1.GPUInfo{
				Index:       index,
				Vendor:      f.vendor,
				Model:       f.model,
				Arch:        arch,
				MemoryBytes: parseGPUMemoryBytes(f.memory),
				DriverVer:   driverVersion(f),
			})
			index++
		}
	}
	return out
}

func gpuCount(raw string) (int, bool) {
	if raw == "" {
		return 0, false
	}
	n, err := strconv.Atoi(raw)
	if err != nil || n < 0 || n > maxGPUsPerFingerprint {
		return 0, false
	}
	return n, true
}

func driverVersion(f *gpuFacts) string {
	if f.driver != "" {
		return f.driver
	}
	parts := make([]string, 0, 3)
	for _, p := range []string{f.driverMajor, f.driverMinor, f.driverRev} {
		if p == "" {
			break
		}
		parts = append(parts, p)
	}
	return strings.Join(parts, ".")
}

// parseGPUMemoryBytes converts a published memory label to bytes.
//
// A bare integer is MEBIBYTES: that is what every publisher of a `gpu.memory`
// label emits (GFD included), and reading 81920 as bytes would record an 80KiB
// accelerator. A value carrying a unit ("16Gi", "16G") is parsed as a quantity
// and taken at face value.
func parseGPUMemoryBytes(v string) int64 {
	if v == "" {
		return 0
	}
	if n, err := strconv.ParseInt(v, 10, 64); err == nil {
		const mib = int64(1) << 20
		if n <= 0 || n > (1<<62)/mib {
			return 0
		}
		return n * mib
	}
	q, err := resource.ParseQuantity(v)
	if err != nil || q.Sign() <= 0 {
		return 0
	}
	return q.Value()
}

// nicsFromLabels reads per-interface facts from labels of the form
// `<domain>/nic.<device>.<field>` — the shape a node probe publishes when it
// classifies interfaces (management vs fabric, RDMA device, link layer). No
// such probe ships in this repo yet; where the labels are absent the list is
// simply empty, which is the honest answer rather than a guess.
func nicsFromLabels(labels map[string]string) []burninv1alpha1.NICInfo {
	byDevice := map[string]*burninv1alpha1.NICInfo{}
	for k, v := range labels {
		_, name := splitLabelKey(k)
		rest, ok := strings.CutPrefix(strings.ToLower(name), "nic.")
		if !ok {
			continue
		}
		// The field is the last dotted segment, so device names containing dots
		// survive intact.
		i := strings.LastIndex(rest, ".")
		if i <= 0 || i == len(rest)-1 {
			continue
		}
		device, field := rest[:i], rest[i+1:]
		nic := byDevice[device]
		if nic == nil {
			nic = &burninv1alpha1.NICInfo{Name: device}
			byDevice[device] = nic
		}
		switch field {
		case "role":
			nic.Role = v
		case "pci-vendor", "pcivendor":
			nic.PCIVendor = v
		case "model":
			nic.Model = v
		case "rdma-device", "rdmadevice":
			nic.RDMADevice = v
		case "link-layer", "linklayer":
			nic.LinkLayer = v
		case "speed-mbps", "speedmbps":
			nic.SpeedMbps = parseInt32(v)
		case "mtu":
			nic.MTU = parseInt32(v)
		}
	}
	if len(byDevice) == 0 {
		return nil
	}
	out := make([]burninv1alpha1.NICInfo, 0, len(byDevice))
	for _, name := range slices.Sorted(maps.Keys(byDevice)) {
		out = append(out, *byDevice[name])
	}
	return out
}

func parseInt32(v string) int32 {
	n, err := strconv.ParseInt(v, 10, 32)
	if err != nil || n < 0 {
		return 0
	}
	return int32(n)
}

// ─── Digest ───────────────────────────────────────────────────────────────────

// fingerprintDigest hashes the salient identity. Two properties are load
// bearing:
//
//   - DETERMINISM. Map iteration order must not reach the hash, or every
//     reconcile would look like drift. Labels are sorted; GPUs and NICs are
//     already emitted in a fixed order.
//
//   - COVERAGE BEYOND WHAT IS PARSED. Salient labels are hashed raw as well as
//     in derived form, so a fact this code does not yet understand — a count
//     spelling it does not recognise, a vendor label invented next year — still
//     moves the digest when it changes. Drift detection must not be limited to
//     the vocabulary that happened to be known when it was written.
func fingerprintDigest(s *burninv1alpha1.NodeFingerprintStatus) string {
	h := sha256.New()
	add := func(key, value string) {
		// NUL-delimited: label keys and values cannot contain NUL, so no pair
		// of distinct inputs can hash to the same byte stream.
		h.Write([]byte(key))
		h.Write([]byte{0})
		h.Write([]byte(value))
		h.Write([]byte{0})
	}
	add("scheme", digestSchemeVersion)
	add("osImage", s.OSImage)
	add("kernel", s.Kernel)
	for _, k := range slices.Sorted(maps.Keys(s.Labels)) {
		add("label:"+k, s.Labels[k])
	}
	for _, g := range s.GPUs {
		add("gpu", fmt.Sprintf("%d|%s|%s|%s|%d|%s", g.Index, g.Vendor, g.Model, g.Arch, g.MemoryBytes, g.DriverVer))
	}
	for _, n := range s.NICs {
		add("nic", fmt.Sprintf("%s|%s|%s|%s|%s|%s|%d|%d",
			n.Name, n.Role, n.PCIVendor, n.Model, n.RDMADevice, n.LinkLayer, n.SpeedMbps, n.MTU))
	}
	return digestSchemeVersion + ":" + hex.EncodeToString(h.Sum(nil))
}

func digestScheme(digest string) string {
	if i := strings.Index(digest, ":"); i >= 0 {
		return digest[:i]
	}
	return ""
}

// ─── Naming ───────────────────────────────────────────────────────────────────

// fingerprintName derives the object name from the node name. Node names are
// already DNS subdomains, so the common case is an identity mapping — which
// keeps `kubectl get nodefingerprint <node>` working. Anything that would not
// be a legal object name is sanitised and disambiguated with a hash of the
// original, so two odd names can never collide onto one fingerprint.
func fingerprintName(node string) string {
	const maxSanitised = 190
	if errs := validation.IsDNS1123Subdomain(node); len(errs) == 0 {
		return node
	}
	var b strings.Builder
	for _, ch := range strings.ToLower(node) {
		switch {
		case ch >= 'a' && ch <= 'z', ch >= '0' && ch <= '9', ch == '-', ch == '.':
			b.WriteRune(ch)
		default:
			b.WriteRune('-')
		}
	}
	sanitised := strings.Trim(b.String(), "-.")
	if len(sanitised) > maxSanitised {
		sanitised = strings.Trim(sanitised[:maxSanitised], "-.")
	}
	sum := sha256.Sum256([]byte(node))
	if sanitised == "" {
		return "node-" + hex.EncodeToString(sum[:8])
	}
	return sanitised + "-" + hex.EncodeToString(sum[:8])
}

// ─── Wiring ───────────────────────────────────────────────────────────────────

// SetupWithManager watches Nodes, filtered down to the events that can change a
// fingerprint, plus the NodeFingerprints themselves so a deleted or hand-edited
// one is restored.
func (r *NodeFingerprintReconciler) SetupWithManager(mgr ctrl.Manager) error {
	if r.Namespace == "" {
		raw, err := os.ReadFile(serviceAccountNamespaceFile)
		if err != nil {
			return fmt.Errorf("NodeFingerprint namespace is unknown: set POD_NAMESPACE from the downward API "+
				"(fieldRef metadata.namespace) or NodeFingerprintReconciler.Namespace; refusing to guess where to "+
				"write fingerprints: %w", err)
		}
		r.Namespace = strings.TrimSpace(string(raw))
	}
	if r.Namespace == "" {
		return fmt.Errorf("NodeFingerprint namespace resolved to the empty string")
	}

	return ctrl.NewControllerManagedBy(mgr).
		For(&corev1.Node{}, builder.WithPredicates(nodeIdentityChanged())).
		Watches(&burninv1alpha1.NodeFingerprint{}, handler.EnqueueRequestsFromMapFunc(r.nodeForFingerprint)).
		Named("nodefingerprint").
		Complete(r)
}

// nodeIdentityChanged drops the node updates this controller has no use for.
// Kubelets heartbeat every few seconds on every node in the cluster; without
// this the controller would wake once per heartbeat per node purely to hash
// facts that did not move.
//
// Create and Delete events are deliberately not filtered, so the initial list at
// manager startup reconciles every node once. That is when a stranded cordon
// left by a previous manager gets its first look, and it is the only pass a
// cluster with no BurnInRuns will ever get for free.
//
// The cordon stamp is NOT a fingerprint input — it is this operator's own
// bookkeeping, and salientLabels deliberately excludes everything in this
// project's domain so a burn-in cannot make the hardware look like it drifted.
// It is watched here anyway, because a node acquiring a stamp is what arms the
// reaper's re-examination timer. Without it, a node stamped after its last
// reconcile would not be looked at again until the manager's resync hours later.
func nodeIdentityChanged() predicate.Predicate {
	return predicate.Funcs{
		UpdateFunc: func(e event.UpdateEvent) bool {
			oldNode, okOld := e.ObjectOld.(*corev1.Node)
			newNode, okNew := e.ObjectNew.(*corev1.Node)
			if !okOld || !okNew {
				return true
			}
			return !maps.Equal(oldNode.Labels, newNode.Labels) ||
				oldNode.Status.NodeInfo != newNode.Status.NodeInfo ||
				oldNode.Annotations[burninv1alpha1.AnnotationCordonOwner] !=
					newNode.Annotations[burninv1alpha1.AnnotationCordonOwner]
		},
	}
}

// nodeForFingerprint maps a NodeFingerprint back to its node, so deleting a
// fingerprint re-captures it rather than leaving the node unrecorded.
func (r *NodeFingerprintReconciler) nodeForFingerprint(_ context.Context, obj client.Object) []reconcile.Request {
	fp, ok := obj.(*burninv1alpha1.NodeFingerprint)
	if !ok || fp.Namespace != r.Namespace || fp.Spec.NodeName == "" {
		return nil
	}
	return []reconcile.Request{{NamespacedName: types.NamespacedName{Name: fp.Spec.NodeName}}}
}
