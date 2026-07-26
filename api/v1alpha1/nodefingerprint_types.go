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
	CapturedAt *metav1.Time      `json:"capturedAt,omitempty"`
	GPUs       []GPUInfo         `json:"gpus,omitempty"`
	NICs       []NICInfo         `json:"nics,omitempty"`
	OSImage    string            `json:"osImage,omitempty"`
	Kernel     string            `json:"kernel,omitempty"`
	Labels     map[string]string `json:"labels,omitempty"`
	// Digest is a stable hash of the salient hardware identity; a change between
	// runs means the hardware under a prior verdict is no longer the same.
	Digest             string `json:"digest,omitempty"`
	ObservedGeneration int64  `json:"observedGeneration,omitempty"`
}

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
