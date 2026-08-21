package v1alpha1

import (
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

// SinkType is where a BurnInRun's verdict is exported.
//
// This is the ONLY integration seam with an external control plane. The operator
// never imports a consumer; it POSTs a generic result document to
// a webhook, so a control plane — or Prometheus, or a customer's own system —
// consumes burn-in verdicts without a code dependency. That standalone contract
// is what makes this repo publishable on its own.
// +kubebuilder:validation:Enum=Webhook;Prometheus;ConfigMap
type SinkType string

const (
	// SinkWebhook POSTs the result JSON to a URL (a control plane registers one here).
	SinkWebhook SinkType = "Webhook"
	// SinkPrometheus exposes per-run metrics for scrape.
	SinkPrometheus SinkType = "Prometheus"
	// SinkConfigMap writes the result document into a ConfigMap (airgapped/no-egress).
	SinkConfigMap SinkType = "ConfigMap"
)

// BurnInSinkSpec configures a result export target.
type BurnInSinkSpec struct {
	// +kubebuilder:validation:Required
	Type SinkType `json:"type"`

	// Webhook config (Type=Webhook).
	Webhook *WebhookSink `json:"webhook,omitempty"`
	// ConfigMap config (Type=ConfigMap).
	ConfigMap *ConfigMapSink `json:"configMap,omitempty"`
}

// WebhookSink POSTs the result document to URL. Auth is a bearer token read from
// a Secret so no credential is stored on the CR.
type WebhookSink struct {
	// +kubebuilder:validation:Required
	URL string `json:"url"`
	// TokenSecretRef references a Secret key holding a bearer token (optional).
	TokenSecretRef *corev1.SecretKeySelector `json:"tokenSecretRef,omitempty"`
	// InsecureSkipVerify disables TLS verification (dev only).
	InsecureSkipVerify bool `json:"insecureSkipVerify,omitempty"`
}

// ConfigMapSink names the ConfigMap results are written to.
type ConfigMapSink struct {
	// +kubebuilder:validation:Required
	Name string `json:"name"`
}

type BurnInSinkStatus struct {
	LastDelivery *metav1.Time `json:"lastDelivery,omitempty"`
	LastError    string       `json:"lastError,omitempty"`
}

// +kubebuilder:object:root=true
// +kubebuilder:subresource:status
// +kubebuilder:resource:shortName=bis,categories=burnin
// +kubebuilder:printcolumn:name="Type",type=string,JSONPath=`.spec.type`

// BurnInSink is a result-export target — the generic contract that lets an
// external control plane consume burn-in verdicts with no code dependency.
type BurnInSink struct {
	metav1.TypeMeta   `json:",inline"`
	metav1.ObjectMeta `json:"metadata,omitempty"`
	Spec              BurnInSinkSpec   `json:"spec,omitempty"`
	Status            BurnInSinkStatus `json:"status,omitempty"`
}

// +kubebuilder:object:root=true

// BurnInSinkList contains a list of BurnInSink.
type BurnInSinkList struct {
	metav1.TypeMeta `json:",inline"`
	metav1.ListMeta `json:"metadata,omitempty"`
	Items           []BurnInSink `json:"items"`
}
