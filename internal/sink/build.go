package sink

import (
	"context"
	"fmt"
	"time"

	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/types"
	"sigs.k8s.io/controller-runtime/pkg/client"

	burninv1alpha1 "github.com/baldwinSPC/glimmer-burnin/api/v1alpha1"
	"github.com/baldwinSPC/glimmer-burnin/internal/metrics"
)

// Build turns a BurnInSink CR into a Deliverer.
//
// Configuration errors are Permanent: a sink pointing at a Secret that does not
// exist will not start working because we asked six times, and reporting it
// immediately is what lets an operator fix it before a run finishes.
func Build(ctx context.Context, c client.Client, s *burninv1alpha1.BurnInSink, timeout time.Duration) (Deliverer, error) {
	switch s.Spec.Type {
	case burninv1alpha1.SinkWebhook:
		if s.Spec.Webhook == nil {
			return nil, Permanent(fmt.Errorf("sink %s/%s has type Webhook but no webhook config", s.Namespace, s.Name))
		}
		token, err := resolveToken(ctx, c, s.Namespace, s.Spec.Webhook.TokenSecretRef)
		if err != nil {
			return nil, err
		}
		return NewWebhook(s.Spec.Webhook.URL, token, s.Spec.Webhook.InsecureSkipVerify, timeout), nil

	case burninv1alpha1.SinkConfigMap:
		if s.Spec.ConfigMap == nil {
			return nil, Permanent(fmt.Errorf("sink %s/%s has type ConfigMap but no configMap config", s.Namespace, s.Name))
		}
		return &ConfigMap{Client: c, Namespace: s.Namespace, Name: s.Spec.ConfigMap.Name}, nil

	case burninv1alpha1.SinkPrometheus:
		// Prometheus is a scrape target, not a push destination, so this sink
		// is a SELECTOR rather than an address: naming it in a profile marks
		// that profile's runs for exposition on the manager's /metrics
		// endpoint. "Delivery" reflects the envelope into the in-process
		// exporter and always succeeds — there is no network, so there is
		// nothing to retry — and the sink's status records it like any other.
		//
		// It carries no config, which is why there is no nil check here to
		// match the other two branches. Exposition is not durable delivery
		// (see internal/metrics), so pair a Prometheus sink with a Webhook or
		// ConfigMap sink rather than using it alone.
		return metrics.Default.SinkFor(s.Namespace, s.Name), nil

	default:
		return nil, Permanent(fmt.Errorf("sink %s/%s has unknown type %q", s.Namespace, s.Name, s.Spec.Type))
	}
}

// resolveToken reads the bearer token out of its Secret.
//
// The token is only ever held in memory and is never written to status or logs;
// Describe() on a Deliverer deliberately omits it.
func resolveToken(ctx context.Context, c client.Client, namespace string, ref *corev1.SecretKeySelector) (string, error) {
	if ref == nil {
		return "", nil
	}
	var secret corev1.Secret
	if err := c.Get(ctx, types.NamespacedName{Namespace: namespace, Name: ref.Name}, &secret); err != nil {
		return "", Permanent(fmt.Errorf("read token secret %s/%s: %w", namespace, ref.Name, err))
	}
	v, ok := secret.Data[ref.Key]
	if !ok {
		return "", Permanent(fmt.Errorf("secret %s/%s has no key %q", namespace, ref.Name, ref.Key))
	}
	if len(v) == 0 {
		return "", Permanent(fmt.Errorf("secret %s/%s key %q is empty", namespace, ref.Name, ref.Key))
	}
	return string(v), nil
}
