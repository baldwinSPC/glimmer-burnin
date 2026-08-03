package controller

import (
	"context"
	"strings"
	"testing"
	"time"

	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/client-go/tools/record"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/client/fake"

	burninv1alpha1 "github.com/baldwinSPC/glimmer-burnin/api/v1alpha1"
)

const nfpNamespace = "burnin"

type nfpHarness struct {
	t        *testing.T
	r        *NodeFingerprintReconciler
	c        client.Client
	recorder *record.FakeRecorder
	nowVal   time.Time
}

func newNFPHarness(t *testing.T, objs ...client.Object) *nfpHarness {
	t.Helper()
	scheme := runtime.NewScheme()
	if err := corev1.AddToScheme(scheme); err != nil {
		t.Fatal(err)
	}
	if err := burninv1alpha1.AddToScheme(scheme); err != nil {
		t.Fatal(err)
	}
	c := fake.NewClientBuilder().
		WithScheme(scheme).
		WithObjects(objs...).
		WithStatusSubresource(&burninv1alpha1.NodeFingerprint{}).
		Build()

	h := &nfpHarness{
		t:        t,
		c:        c,
		recorder: record.NewFakeRecorder(32),
		nowVal:   time.Unix(1750000000, 0).UTC(),
	}
	h.r = &NodeFingerprintReconciler{
		Client:    c,
		Scheme:    scheme,
		Recorder:  h.recorder,
		Namespace: nfpNamespace,
		Now:       func() time.Time { return h.nowVal },
	}
	return h
}

func (h *nfpHarness) reconcile(node string) {
	h.t.Helper()
	if _, err := h.r.Reconcile(context.Background(),
		ctrl.Request{NamespacedName: types.NamespacedName{Name: node}}); err != nil {
		h.t.Fatalf("Reconcile(%s): %v", node, err)
	}
}

// fingerprint returns the captured fingerprint for a node, failing if absent.
func (h *nfpHarness) fingerprint(node string) *burninv1alpha1.NodeFingerprint {
	h.t.Helper()
	fp, err := h.lookup(node)
	if err != nil {
		h.t.Fatalf("get fingerprint for %s: %v", node, err)
	}
	return fp
}

func (h *nfpHarness) lookup(node string) (*burninv1alpha1.NodeFingerprint, error) {
	var fp burninv1alpha1.NodeFingerprint
	key := types.NamespacedName{Namespace: nfpNamespace, Name: fingerprintName(node)}
	if err := h.c.Get(context.Background(), key, &fp); err != nil {
		return nil, err
	}
	return &fp, nil
}

// relabel replaces a node's labels and reconciles.
func (h *nfpHarness) relabel(node string, labels map[string]string) {
	h.t.Helper()
	var n corev1.Node
	if err := h.c.Get(context.Background(), types.NamespacedName{Name: node}, &n); err != nil {
		h.t.Fatal(err)
	}
	n.Labels = labels
	if err := h.c.Update(context.Background(), &n); err != nil {
		h.t.Fatal(err)
	}
}

func (h *nfpHarness) events() []string {
	var out []string
	for {
		select {
		case e := <-h.recorder.Events:
			out = append(out, e)
		default:
			return out
		}
	}
}

func drifted(fp *burninv1alpha1.NodeFingerprint) *metav1.Condition {
	return meta.FindStatusCondition(fp.Status.Conditions, burninv1alpha1.ConditionNodeFingerprintDrifted)
}

// ─── Fixtures ─────────────────────────────────────────────────────────────────

// sparkLabels is a GB10 node as a GPU Feature Discovery / NFD deployment
// actually labels it, plus this project's own hardware annotations.
func sparkLabels() map[string]string {
	return map[string]string{
		corev1.LabelHostname:     "spark-a",
		corev1.LabelArchStable:   "arm64",
		"nvidia.com/gpu.product": "NVIDIA GB10",
		"nvidia.com/gpu.count":   "1",
		"nvidia.com/gpu.memory":  "131072",
		"glimmer.ai/gpu-arch":    "sm_121",
		"glimmer.ai/hw-class":    "dgx-spark",
	}
}

func nfpNode(name string, labels map[string]string) *corev1.Node {
	return &corev1.Node{
		ObjectMeta: metav1.ObjectMeta{Name: name, Labels: labels},
		Status: corev1.NodeStatus{NodeInfo: corev1.NodeSystemInfo{
			KernelVersion: "6.11.0-1014-nvidia",
			OSImage:       "Ubuntu 24.04 LTS",
			Architecture:  "arm64",
		}},
	}
}

// ─── Capture ──────────────────────────────────────────────────────────────────

func TestNodeFingerprint_CapturesTheNodesIdentity(t *testing.T) {
	h := newNFPHarness(t, nfpNode("spark-a", sparkLabels()))
	h.reconcile("spark-a")

	fp := h.fingerprint("spark-a")
	if fp.Spec.NodeName != "spark-a" {
		t.Errorf("spec.nodeName = %q, want spark-a", fp.Spec.NodeName)
	}
	if fp.Status.Kernel != "6.11.0-1014-nvidia" || fp.Status.OSImage != "Ubuntu 24.04 LTS" {
		t.Errorf("kernel/os not captured: %q / %q", fp.Status.Kernel, fp.Status.OSImage)
	}
	if fp.Status.Digest == "" {
		t.Fatal("no digest was computed")
	}
	if !strings.HasPrefix(fp.Status.Digest, digestSchemeVersion+":") {
		t.Errorf("digest %q does not carry its scheme — an operator upgrade could not be told from a hardware change", fp.Status.Digest)
	}
	if fp.Status.CapturedAt == nil || !fp.Status.CapturedAt.Time.Equal(h.nowVal) {
		t.Errorf("capturedAt = %v, want the harness clock", fp.Status.CapturedAt)
	}

	if len(fp.Status.GPUs) != 1 {
		t.Fatalf("gpus = %+v, want one entry", fp.Status.GPUs)
	}
	gpu := fp.Status.GPUs[0]
	if gpu.Vendor != "nvidia" || gpu.Model != "NVIDIA GB10" {
		t.Errorf("gpu identity = %+v, want vendor nvidia / model NVIDIA GB10", gpu)
	}
	if gpu.Arch != "sm_121" {
		t.Errorf("gpu arch = %q, want sm_121 filled in from the unattributed label", gpu.Arch)
	}
	if want := int64(131072) << 20; gpu.MemoryBytes != want {
		t.Errorf("gpu memory = %d, want %d — a bare gpu.memory label is MiB, not bytes", gpu.MemoryBytes, want)
	}

	// The salient labels are captured; the scheduling ones are not.
	if fp.Status.Labels["nvidia.com/gpu.count"] != "1" || fp.Status.Labels["glimmer.ai/hw-class"] != "dgx-spark" {
		t.Errorf("hardware labels not captured: %v", fp.Status.Labels)
	}
	if _, present := fp.Status.Labels[corev1.LabelHostname]; present {
		t.Errorf("hostname was captured as hardware identity: %v", fp.Status.Labels)
	}

	cond := drifted(fp)
	if cond == nil || cond.Status != metav1.ConditionFalse ||
		cond.Reason != burninv1alpha1.ReasonFingerprintBaselineCaptured {
		t.Fatalf("first capture condition = %+v, want Drifted=False/BaselineCaptured", cond)
	}
	if evts := h.events(); len(evts) != 0 {
		t.Errorf("first capture emitted drift events: %v", evts)
	}
}

// The whole point of a digest is that it does not move when the hardware does
// not. Node objects heartbeat constantly, so a fingerprint that rewrote itself
// on every pass would both churn etcd and drown the drift signal.
func TestNodeFingerprint_DigestIsStableAndDoesNotChurnTheObject(t *testing.T) {
	h := newNFPHarness(t, nfpNode("spark-a", sparkLabels()))
	h.reconcile("spark-a")
	// The second pass is the first that has something to compare against, so
	// it settles the condition from BaselineCaptured to DigestStable. From
	// there the object must not be touched again.
	h.reconcile("spark-a")

	first := h.fingerprint("spark-a")
	if cond := drifted(first); cond == nil || cond.Reason != burninv1alpha1.ReasonFingerprintStable {
		t.Fatalf("condition after a comparison = %+v, want DigestStable", cond)
	}
	for i := 0; i < 5; i++ {
		h.nowVal = h.nowVal.Add(time.Minute)
		h.reconcile("spark-a")
	}
	again := h.fingerprint("spark-a")

	if again.Status.Digest != first.Status.Digest {
		t.Errorf("digest changed across reconciles with unchanged hardware:\n%s\n%s", first.Status.Digest, again.Status.Digest)
	}
	if !again.Status.CapturedAt.Time.Equal(first.Status.CapturedAt.Time) {
		t.Errorf("capturedAt advanced without a change: %v → %v", first.Status.CapturedAt, again.Status.CapturedAt)
	}
	if again.ResourceVersion != first.ResourceVersion {
		t.Errorf("the fingerprint was rewritten by a no-op reconcile (rv %s → %s)", first.ResourceVersion, again.ResourceVersion)
	}
	if cond := drifted(again); cond == nil || cond.Status != metav1.ConditionFalse {
		t.Errorf("stable node reports Drifted = %+v", cond)
	}
}

// Nothing here may depend on Go's map iteration order.
func TestNodeFingerprint_DigestIsIndependentOfLabelOrderButNotOfContent(t *testing.T) {
	base := observeNode(nfpNode("spark-a", sparkLabels()))
	for i := 0; i < 50; i++ {
		if got := fingerprintDigest(&base); got != fingerprintDigest(&base) {
			t.Fatalf("digest is not deterministic: %s", got)
		}
	}

	// Two labels swapping values is a different machine, and must hash
	// differently — a digest that concatenated values without their keys would
	// miss it.
	swapped := sparkLabels()
	swapped["nvidia.com/gpu.product"], swapped["glimmer.ai/hw-class"] =
		swapped["glimmer.ai/hw-class"], swapped["nvidia.com/gpu.product"]
	other := observeNode(nfpNode("spark-a", swapped))
	if fingerprintDigest(&base) == fingerprintDigest(&other) {
		t.Error("swapping two label values left the digest unchanged")
	}
}

// ─── Derivation from facts ────────────────────────────────────────────────────

func TestNodeFingerprint_GPUCountMaterialisesOneEntryPerAccelerator(t *testing.T) {
	labels := map[string]string{
		"nvidia.com/gpu.product":        "NVIDIA H100 80GB HBM3",
		"nvidia.com/gpu.count":          "8",
		"nvidia.com/gpu.memory":         "81559",
		"nvidia.com/cuda.driver.major":  "570",
		"nvidia.com/cuda.driver.minor":  "86",
		"nvidia.com/cuda.driver.rev":    "15",
		"nvidia.com/gpu.compute-capab":  "ignored",
		"nvidia.com/cuda.compute-capab": "ignored",
	}
	gpus := gpusFromLabels(labels)
	if len(gpus) != 8 {
		t.Fatalf("gpus = %d, want 8", len(gpus))
	}
	for i, g := range gpus {
		if g.Index != int32(i) {
			t.Errorf("gpu[%d].Index = %d", i, g.Index)
		}
		if g.DriverVer != "570.86.15" {
			t.Errorf("gpu[%d].DriverVer = %q, want 570.86.15 assembled from the major/minor/rev labels", i, g.DriverVer)
		}
	}
}

// The vendor is read off the DNS domain that published the fact. A new
// accelerator vendor is supported by its device plugin publishing labels, not
// by a branch in this controller.
func TestNodeFingerprint_VendorComesFromTheLabelDomain(t *testing.T) {
	for domain, wantVendor := range map[string]string{
		"amd.com":          "amd",
		"tenstorrent.com":  "tenstorrent",
		"some-new-npu.dev": "some-new-npu",
	} {
		gpus := gpusFromLabels(map[string]string{
			domain + "/gpu.product": "Accelerator One",
			domain + "/gpu.count":   "2",
		})
		if len(gpus) != 2 {
			t.Fatalf("%s: gpus = %d, want 2", domain, len(gpus))
		}
		if gpus[0].Vendor != wantVendor {
			t.Errorf("%s: vendor = %q, want %q", domain, gpus[0].Vendor, wantVendor)
		}
	}

	// NFD is not a silicon vendor: its labels are facts, but they must not be
	// attributed to a vendor called "feature".
	gpus := gpusFromLabels(map[string]string{
		"feature.node.kubernetes.io/gpu.product": "Accelerator One",
		"feature.node.kubernetes.io/gpu.count":   "1",
	})
	if len(gpus) != 1 || gpus[0].Vendor != "" {
		t.Errorf("NFD-published GPU = %+v, want a single entry with no vendor claimed", gpus)
	}
}

// A count nobody can believe must not be allowed to inflate the status object;
// the raw label still reaches the digest, so nothing is lost.
func TestNodeFingerprint_ImplausibleGPUCountDoesNotInflateStatus(t *testing.T) {
	gpus := gpusFromLabels(map[string]string{
		"nvidia.com/gpu.product": "NVIDIA GB10",
		"nvidia.com/gpu.count":   "1000000",
	})
	if len(gpus) != 1 {
		t.Fatalf("gpus = %d, want 1 — an incredible count must fall back to what is proven", len(gpus))
	}

	if got := gpusFromLabels(map[string]string{"nvidia.com/gpu.count": "0"}); len(got) != 0 {
		t.Errorf("gpu.count=0 produced %+v, want no accelerators", got)
	}

	labels := sparkLabels()
	labels["nvidia.com/gpu.count"] = "1000000"
	before := observeNode(nfpNode("spark-a", sparkLabels()))
	after := observeNode(nfpNode("spark-a", labels))
	if fingerprintDigest(&before) == fingerprintDigest(&after) {
		t.Error("a count the parser rejected also vanished from the digest — the change would go undetected")
	}
}

func TestNodeFingerprint_NICLabelsBecomeNICInfo(t *testing.T) {
	nics := nicsFromLabels(map[string]string{
		"glimmer.ai/nic.enp1s0f0.role":        "fabric",
		"glimmer.ai/nic.enp1s0f0.rdma-device": "mlx5_0",
		"glimmer.ai/nic.enp1s0f0.link-layer":  "infiniband",
		"glimmer.ai/nic.enp1s0f0.speed-mbps":  "400000",
		"glimmer.ai/nic.enp1s0f0.mtu":         "9000",
		"glimmer.ai/nic.enp1s0f0.pci-vendor":  "0x15b3",
		"glimmer.ai/nic.eth0.role":            "management",
		"glimmer.ai/gpu-arch":                 "sm_121",
	})
	if len(nics) != 2 {
		t.Fatalf("nics = %+v, want 2", nics)
	}
	// Sorted by device name, so the record is stable across reconciles.
	if nics[0].Name != "enp1s0f0" || nics[1].Name != "eth0" {
		t.Fatalf("nics are not in a deterministic order: %+v", nics)
	}
	fabric := nics[0]
	if fabric.Role != "fabric" || fabric.RDMADevice != "mlx5_0" || fabric.LinkLayer != "infiniband" {
		t.Errorf("fabric NIC = %+v", fabric)
	}
	if fabric.SpeedMbps != 400000 || fabric.MTU != 9000 || fabric.PCIVendor != "0x15b3" {
		t.Errorf("fabric NIC numeric/pci fields = %+v", fabric)
	}
	if got := nicsFromLabels(map[string]string{"kubernetes.io/arch": "arm64"}); got != nil {
		t.Errorf("a node with no NIC labels reported %+v, want none rather than a guess", got)
	}
}

// ─── Drift ────────────────────────────────────────────────────────────────────

func TestNodeFingerprint_DriftSetsConditionAndEmitsEvent(t *testing.T) {
	h := newNFPHarness(t, nfpNode("spark-a", sparkLabels()))
	h.reconcile("spark-a")
	baseline := h.fingerprint("spark-a").Status.Digest
	h.events()

	// Half the accelerators are gone: an RMA, a reseat, or a lie. Whichever it
	// is, the verdict recorded against the old fingerprint no longer applies.
	drifting := sparkLabels()
	drifting["nvidia.com/gpu.count"] = "2"
	h.relabel("spark-a", drifting)
	h.nowVal = h.nowVal.Add(time.Hour)
	h.reconcile("spark-a")

	fp := h.fingerprint("spark-a")
	if fp.Status.Digest == baseline {
		t.Fatal("digest did not move when the accelerator count did")
	}
	if !fp.Status.CapturedAt.Time.Equal(h.nowVal) {
		t.Errorf("capturedAt = %v, want the time of the change", fp.Status.CapturedAt)
	}
	if len(fp.Status.GPUs) != 2 {
		t.Errorf("gpus = %d, want the new count of 2", len(fp.Status.GPUs))
	}

	cond := drifted(fp)
	if cond == nil || cond.Status != metav1.ConditionTrue ||
		cond.Reason != burninv1alpha1.ReasonFingerprintDigestChanged {
		t.Fatalf("condition = %+v, want Drifted=True/DigestChanged", cond)
	}
	if !strings.Contains(cond.Message, "gpu.count") {
		t.Errorf("condition message does not name what changed: %q", cond.Message)
	}

	evts := h.events()
	if len(evts) != 1 {
		t.Fatalf("events = %v, want exactly one drift event", evts)
	}
	if !strings.Contains(evts[0], burninv1alpha1.EventReasonHardwareDrift) ||
		!strings.Contains(evts[0], "Warning") {
		t.Errorf("event is not a HardwareDrift warning: %q", evts[0])
	}
	if !strings.Contains(evts[0], "spark-a") {
		t.Errorf("event does not name the node: %q", evts[0])
	}
}

// The reconciler reports drift; it does not act on it. Nothing may create a
// BurnInRun off the back of a hash comparison.
func TestNodeFingerprint_DriftDoesNotStartRuns(t *testing.T) {
	h := newNFPHarness(t, nfpNode("spark-a", sparkLabels()))
	h.reconcile("spark-a")
	drifting := sparkLabels()
	drifting["glimmer.ai/gpu-arch"] = "sm_120"
	h.relabel("spark-a", drifting)
	h.reconcile("spark-a")

	var runs burninv1alpha1.BurnInRunList
	if err := h.c.List(context.Background(), &runs); err != nil {
		t.Fatal(err)
	}
	if len(runs.Items) != 0 {
		t.Errorf("drift created %d BurnInRun(s); saturating a fleet's accelerators is policy, not a hash comparison's call", len(runs.Items))
	}
}

// Once raised, the flag stays raised. The fingerprint has by then been
// rewritten to describe the drifted hardware, so a self-clearing condition
// would erase the only record that anything happened.
func TestNodeFingerprint_DriftIsStickyOnceRaised(t *testing.T) {
	h := newNFPHarness(t, nfpNode("spark-a", sparkLabels()))
	h.reconcile("spark-a")

	drifting := sparkLabels()
	drifting["nvidia.com/gpu.count"] = "2"
	h.relabel("spark-a", drifting)
	h.reconcile("spark-a")
	firstSeen := drifted(h.fingerprint("spark-a")).LastTransitionTime

	for i := 0; i < 3; i++ {
		h.nowVal = h.nowVal.Add(time.Hour)
		h.reconcile("spark-a")
	}
	cond := drifted(h.fingerprint("spark-a"))
	if cond == nil || cond.Status != metav1.ConditionTrue {
		t.Fatalf("condition = %+v after later stable passes, want it still True", cond)
	}
	if !cond.LastTransitionTime.Equal(&firstSeen) {
		t.Errorf("LastTransitionTime moved to %v; it must stay at the first drift, %v", cond.LastTransitionTime, firstSeen)
	}
	if evts := h.events(); len(evts) != 1 {
		t.Errorf("events = %v, want one — a sticky condition must not re-emit on every pass", evts)
	}
}

// A kernel or OS change moves the whole driver stack under a verdict measured
// on the old one. That is drift, deliberately.
func TestNodeFingerprint_KernelUpgradeIsDrift(t *testing.T) {
	h := newNFPHarness(t, nfpNode("spark-a", sparkLabels()))
	h.reconcile("spark-a")

	var node corev1.Node
	if err := h.c.Get(context.Background(), types.NamespacedName{Name: "spark-a"}, &node); err != nil {
		t.Fatal(err)
	}
	node.Status.NodeInfo.KernelVersion = "6.11.0-1015-nvidia"
	if err := h.c.Status().Update(context.Background(), &node); err != nil {
		t.Fatal(err)
	}
	h.reconcile("spark-a")

	cond := drifted(h.fingerprint("spark-a"))
	if cond == nil || cond.Status != metav1.ConditionTrue {
		t.Fatalf("kernel change gave %+v, want Drifted=True", cond)
	}
	if !strings.Contains(cond.Message, "kernel") {
		t.Errorf("message does not name the kernel change: %q", cond.Message)
	}
}

// Verdict, scheduling and topology labels churn constantly — including the
// ones this operator's own consumers write. If they reached the digest, the
// operator would report hardware drift in response to its own output.
func TestNodeFingerprint_VerdictAndSchedulingLabelsDoNotDrift(t *testing.T) {
	h := newNFPHarness(t, nfpNode("spark-a", sparkLabels()))
	h.reconcile("spark-a")
	baseline := h.fingerprint("spark-a")

	noisy := sparkLabels()
	noisy["burnin.glimmer.ai/last-verdict"] = "Passed"
	noisy["glimmer.ai/burnin-gpu-state"] = "ready"
	noisy["node-role.kubernetes.io/worker"] = ""
	noisy["topology.kubernetes.io/zone"] = "rack-7"
	noisy["node.kubernetes.io/unschedulable"] = "true"
	h.relabel("spark-a", noisy)
	h.reconcile("spark-a")

	after := h.fingerprint("spark-a")
	if after.Status.Digest != baseline.Status.Digest {
		t.Fatalf("non-hardware labels moved the digest:\nbefore %v\nafter  %v", baseline.Status.Labels, after.Status.Labels)
	}
	if cond := drifted(after); cond == nil || cond.Status != metav1.ConditionFalse {
		t.Fatalf("condition = %+v, want Drifted=False — a verdict or scheduling label was read as hardware drift", cond)
	}
	if evts := h.events(); len(evts) != 0 {
		t.Errorf("non-hardware labels produced drift events: %v", evts)
	}
}

// A change to what the operator considers salient is an operator change. It
// re-baselines silently; announcing it as hardware drift would fire on every
// node at once, which is how a drift signal stops being read.
func TestNodeFingerprint_DigestSchemeChangeIsNotDrift(t *testing.T) {
	h := newNFPHarness(t, nfpNode("spark-a", sparkLabels()))
	h.reconcile("spark-a")

	fp := h.fingerprint("spark-a")
	fp.Status.Digest = "v0:0000000000000000000000000000000000000000000000000000000000000000"
	if err := h.c.Status().Update(context.Background(), fp); err != nil {
		t.Fatal(err)
	}
	h.events()
	h.reconcile("spark-a")

	after := h.fingerprint("spark-a")
	if !strings.HasPrefix(after.Status.Digest, digestSchemeVersion+":") {
		t.Errorf("digest was not re-baselined under the current scheme: %q", after.Status.Digest)
	}
	cond := drifted(after)
	if cond == nil || cond.Status != metav1.ConditionFalse ||
		cond.Reason != burninv1alpha1.ReasonFingerprintSchemeChanged {
		t.Fatalf("condition = %+v, want Drifted=False/DigestSchemeChanged", cond)
	}
	if evts := h.events(); len(evts) != 0 {
		t.Errorf("an operator-side scheme change emitted hardware events: %v", evts)
	}
}

// Drift detection must not be limited to the label spellings this code knows.
func TestNodeFingerprint_DigestCoversLabelsTheParserDoesNotUnderstand(t *testing.T) {
	before := sparkLabels()
	before["nvidia.com/gpu.clocks-throttle-reasons"] = "none"
	after := sparkLabels()
	after["nvidia.com/gpu.clocks-throttle-reasons"] = "hw-slowdown"

	b, a := observeNode(nfpNode("spark-a", before)), observeNode(nfpNode("spark-a", after))
	if fingerprintDigest(&b) == fingerprintDigest(&a) {
		t.Error("a hardware label the parser has no field for did not reach the digest")
	}
}

// ─── Object lifecycle ─────────────────────────────────────────────────────────

// The fingerprint is the record of what a verdict described. Losing the node
// object is precisely when that record matters most.
func TestNodeFingerprint_DeletedNodeKeepsTheFingerprint(t *testing.T) {
	node := nfpNode("spark-a", sparkLabels())
	h := newNFPHarness(t, node)
	h.reconcile("spark-a")
	captured := h.fingerprint("spark-a").Status.Digest

	if err := h.c.Delete(context.Background(), node); err != nil {
		t.Fatal(err)
	}
	h.reconcile("spark-a")

	fp, err := h.lookup("spark-a")
	if err != nil {
		t.Fatalf("the fingerprint was destroyed with its node: %v", err)
	}
	if fp.Status.Digest != captured {
		t.Errorf("digest = %q, want the captured %q", fp.Status.Digest, captured)
	}
}

func TestNodeFingerprint_DeletedFingerprintIsRecaptured(t *testing.T) {
	h := newNFPHarness(t, nfpNode("spark-a", sparkLabels()))
	h.reconcile("spark-a")
	original := h.fingerprint("spark-a")

	if err := h.c.Delete(context.Background(), original); err != nil {
		t.Fatal(err)
	}
	h.reconcile("spark-a")

	fp := h.fingerprint("spark-a")
	if fp.Status.Digest != original.Status.Digest {
		t.Errorf("recaptured digest %q differs from %q on unchanged hardware", fp.Status.Digest, original.Status.Digest)
	}
	// Deleting the fingerprint is how an operator re-establishes a baseline,
	// so the fresh capture must not inherit the old drift flag.
	if cond := drifted(fp); cond == nil || cond.Status != metav1.ConditionFalse ||
		cond.Reason != burninv1alpha1.ReasonFingerprintBaselineCaptured {
		t.Errorf("recapture condition = %+v, want a clean baseline", cond)
	}
}

// A NodeFingerprint watch has to reach the node that owns it, or a deleted
// fingerprint would never be recaptured.
func TestNodeFingerprint_WatchMapsFingerprintBackToItsNode(t *testing.T) {
	h := newNFPHarness(t)
	fp := &burninv1alpha1.NodeFingerprint{
		ObjectMeta: metav1.ObjectMeta{Namespace: nfpNamespace, Name: "spark-a"},
		Spec:       burninv1alpha1.NodeFingerprintSpec{NodeName: "spark-a"},
	}
	reqs := h.r.nodeForFingerprint(context.Background(), fp)
	if len(reqs) != 1 || reqs[0].Name != "spark-a" {
		t.Fatalf("requests = %+v, want one for node spark-a", reqs)
	}

	elsewhere := fp.DeepCopy()
	elsewhere.Namespace = "somewhere-else"
	if reqs := h.r.nodeForFingerprint(context.Background(), elsewhere); len(reqs) != 0 {
		t.Errorf("a fingerprint outside the operator's namespace enqueued %+v", reqs)
	}
}

// Republishing one machine's hardware under another machine's fingerprint
// would make every verdict bound to it describe the wrong node.
func TestNodeFingerprint_RefusesToOverwriteAnotherNodesFingerprint(t *testing.T) {
	squatter := &burninv1alpha1.NodeFingerprint{
		ObjectMeta: metav1.ObjectMeta{Namespace: nfpNamespace, Name: "spark-a"},
		Spec:       burninv1alpha1.NodeFingerprintSpec{NodeName: "spark-b"},
	}
	h := newNFPHarness(t, nfpNode("spark-a", sparkLabels()), squatter)
	h.reconcile("spark-a")

	fp := h.fingerprint("spark-a")
	if fp.Spec.NodeName != "spark-b" {
		t.Errorf("spec.nodeName = %q, want it left alone", fp.Spec.NodeName)
	}
	if fp.Status.Digest != "" {
		t.Errorf("spark-a's hardware was written into spark-b's fingerprint: %+v", fp.Status)
	}
}

// ─── Naming ───────────────────────────────────────────────────────────────────

func TestFingerprintName_IsIdentityForRealNodeNamesAndSafeOtherwise(t *testing.T) {
	for _, name := range []string{"spark-a", "gpu-node-01.rack7.dc2.example.com"} {
		if got := fingerprintName(name); got != name {
			t.Errorf("fingerprintName(%q) = %q, want the node name unchanged", name, got)
		}
	}

	// Names an API server would reject must still produce distinct, legal
	// object names — two odd nodes may never collide onto one fingerprint.
	weird := []string{"Node_A", "node a", "_", strings.Repeat("x", 400)}
	seen := map[string]string{}
	for _, name := range weird {
		got := fingerprintName(name)
		if got == "" || len(got) > 253 {
			t.Errorf("fingerprintName(%q) = %q, not a usable object name", name, got)
		}
		if strings.ContainsAny(got, "_ ABCDEFGHIJKLMNOPQRSTUVWXYZ") {
			t.Errorf("fingerprintName(%q) = %q, still illegal", name, got)
		}
		if prev, dup := seen[got]; dup {
			t.Errorf("fingerprintName collision: %q and %q both give %q", prev, name, got)
		}
		seen[got] = name
	}
	if fingerprintName("Node_A") == fingerprintName("node-a") {
		t.Error("sanitisation collapsed two distinct node names onto one fingerprint")
	}
}
