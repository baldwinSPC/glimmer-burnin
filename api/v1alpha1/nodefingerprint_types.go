package v1alpha1

import (
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

// NodeFingerprintSpec selects the node whose hardware identity is captured.
type NodeFingerprintSpec struct {
	// NodeName is the node this fingerprint describes.
	// +kubebuilder:validation:Required
	NodeName string `json:"nodeName"`
}

// GPUInfo is one accelerator as observed on the node.
type GPUInfo struct {
	Index       int32  `json:"index"`
	Vendor      string `json:"vendor,omitempty"` // nvidia|amd|intel|tenstorrent|unknown
	Model       string `json:"model,omitempty"`  // "NVIDIA GB10"
	Arch        string `json:"arch,omitempty"`   // "sm_121"
	MemoryBytes int64  `json:"memoryBytes,omitempty"`
	DriverVer   string `json:"driverVersion,omitempty"`
}

// NICInfo is one fabric-relevant network interface. It mirrors the vendor-
// neutral classification a node probe produces (management vs fabric, RDMA
// device, link layer) without importing any external inventory package.
type NICInfo struct {
	Name       string `json:"name"`
	Role       string `json:"role,omitempty"`       // management|fabric|other
	PCIVendor  string `json:"pciVendor,omitempty"`  // 0x15b3, ...
	Model      string `json:"model,omitempty"`      // "ConnectX-7"
	RDMADevice string `json:"rdmaDevice,omitempty"` // "mlx5_0"
	LinkLayer  string `json:"linkLayer,omitempty"`  // ethernet|infiniband
	SpeedMbps  int32  `json:"speedMbps,omitempty"`
	MTU        int32  `json:"mtu,omitempty"`
}

// NodeFingerprintStatus is the captured identity — the baseline a burn-in
// verdict is bound to, and the drift detector between runs.
type NodeFingerprintStatus struct {
	// CapturedAt is when this identity was last OBSERVED TO CHANGE, not when it
	// was last looked at. A field that moved on every reconcile would make
	// every node's fingerprint churn on every node heartbeat, and would say
	// nothing.
	CapturedAt *metav1.Time `json:"capturedAt,omitempty"`
	GPUs       []GPUInfo    `json:"gpus,omitempty"`
	NICs       []NICInfo    `json:"nics,omitempty"`
	OSImage    string       `json:"osImage,omitempty"`
	Kernel     string       `json:"kernel,omitempty"`

	// Labels is the HARDWARE-IDENTITY subset of the node's labels, not a copy
	// of all of them. Copying every label would fold scheduling state, taints,
	// topology and — worst — the burn-in verdict labels this operator's own
	// consumers write onto nodes into the digest, so the operator would report
	// hardware drift in response to its own writes.
	Labels map[string]string `json:"labels,omitempty"`
	// Digest is a stable hash of the salient hardware identity; a change between
	// runs means the hardware under a prior verdict is no longer the same.
	//
	// It is written as "<scheme>:<hex>". The scheme prefix exists so that a
	// change to WHAT the operator considers salient is distinguishable from a
	// change to the hardware itself: digests of different schemes are not
	// comparable, and an incomparable pair is never reported as drift.
	Digest             string `json:"digest,omitempty"`
	ObservedGeneration int64  `json:"observedGeneration,omitempty"`

	// Conditions carries the drift signal; see ConditionNodeFingerprintDrifted.
	Conditions []metav1.Condition `json:"conditions,omitempty"`
}

// Drift signalling.
//
// A burn-in verdict is a statement about a specific machine at a specific
// moment. The fingerprint is what binds the two: it records the hardware the
// verdict describes, so that "spark-a passed" can later be checked against
// "spark-a is still the machine that passed".
//
// The constants below are the vocabulary of that check. They live in the API
// package rather than in the controller because an external consumer — the
// thing that decides whether to trust an old verdict — has to match on them,
// and a constant only the controller knows is a constant no consumer can rely
// on and no test can prove the controller honours.
const (
	// ConditionNodeFingerprintDrifted is True when the node's salient hardware
	// identity has changed since the fingerprint was first captured.
	//
	// It is deliberately STICKY: it is not cleared when a later reconcile finds
	// the digest stable again. It has to be, because the fingerprint is updated
	// to describe the node as it is now — so one pass after the change, the new
	// state IS the baseline and a self-clearing condition would erase the only
	// record that anything happened, usually within seconds and always before a
	// human saw it. The condition's LastTransitionTime is when the node FIRST
	// drifted, its message is the most recent change, and the accompanying
	// Events carry the history.
	//
	// Clearing it is an act of judgement, not of bookkeeping: whoever
	// re-establishes the baseline (typically by re-running acceptance and
	// deleting the NodeFingerprint so it is recaptured) is asserting that the
	// node is trustworthy again. The operator does not make that assertion on
	// its own, and it does not launch runs on drift — that is future policy.
	ConditionNodeFingerprintDrifted = "Drifted"

	// ReasonFingerprintBaselineCaptured marks the first capture for a node.
	// There is nothing to compare against yet, so this is not evidence that the
	// hardware is unchanged — only that a baseline now exists.
	ReasonFingerprintBaselineCaptured = "BaselineCaptured"

	// ReasonFingerprintStable means the recomputed digest matches the stored
	// one: the salient identity is unchanged since the baseline.
	ReasonFingerprintStable = "DigestStable"

	// ReasonFingerprintDigestChanged means the recomputed digest differs from
	// the stored one under the SAME scheme — a real change in the hardware
	// identity, and the only reason that sets the condition True.
	ReasonFingerprintDigestChanged = "DigestChanged"

	// ReasonFingerprintSchemeChanged means the stored digest was produced by a
	// different digest scheme. The two hashes are not comparable, so the
	// operator re-baselines and says nothing about the hardware. Reporting an
	// operator upgrade as a fleet-wide hardware change would be a false alarm
	// on every node at once, which is how a drift signal stops being read.
	ReasonFingerprintSchemeChanged = "DigestSchemeChanged"

	// EventReasonHardwareDrift is the Event reason emitted alongside a
	// transition to Drifted=True.
	EventReasonHardwareDrift = "HardwareDrift"
)

// +kubebuilder:object:root=true
// +kubebuilder:subresource:status
// +kubebuilder:resource:shortName=nfp,categories=burnin
// +kubebuilder:printcolumn:name="Node",type=string,JSONPath=`.spec.nodeName`
// +kubebuilder:printcolumn:name="Digest",type=string,JSONPath=`.status.digest`

// NodeFingerprint is the captured hardware/network identity of a node.
type NodeFingerprint struct {
	metav1.TypeMeta   `json:",inline"`
	metav1.ObjectMeta `json:"metadata,omitempty"`
	Spec              NodeFingerprintSpec   `json:"spec,omitempty"`
	Status            NodeFingerprintStatus `json:"status,omitempty"`
}

// +kubebuilder:object:root=true

// NodeFingerprintList contains a list of NodeFingerprint.
type NodeFingerprintList struct {
	metav1.TypeMeta `json:",inline"`
	metav1.ListMeta `json:"metadata,omitempty"`
	Items           []NodeFingerprint `json:"items"`
}
