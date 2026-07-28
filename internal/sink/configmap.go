package sink

import (
	"context"
	"encoding/json"
	"fmt"

	corev1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"sigs.k8s.io/controller-runtime/pkg/client"

	"github.com/baldwinSPC/glimmer-burnin/pkg/contract"
)

// maxConfigMapBytes is the practical ceiling for a ConfigMap's data. etcd caps
// an object at ~1MiB; staying under it with headroom keeps a long run's
// accumulated deliveries from wedging the sink with an unwritable object.
const maxConfigMapBytes = 900 * 1024

// ConfigMap writes envelopes into a ConfigMap, for clusters with no egress.
//
// It is idempotent by construction: each envelope is stored under its
// DeliveryID, so redelivering overwrites the identical key with identical
// content rather than appending a duplicate.
type ConfigMap struct {
	Client    client.Client
	Namespace string
	Name      string
}

// Describe names the destination.
func (c *ConfigMap) Describe() string {
	return fmt.Sprintf("configmap %s/%s", c.Namespace, c.Name)
}

// Deliver stores the envelope under its DeliveryID.
func (c *ConfigMap) Deliver(ctx context.Context, env *contract.Envelope) error {
	body, err := json.Marshal(env)
	if err != nil {
		return Permanent(fmt.Errorf("encode envelope: %w", err))
	}

	var cm corev1.ConfigMap
	err = c.Client.Get(ctx, types.NamespacedName{Namespace: c.Namespace, Name: c.Name}, &cm)
	switch {
	case apierrors.IsNotFound(err):
		cm = corev1.ConfigMap{
			ObjectMeta: metav1.ObjectMeta{Namespace: c.Namespace, Name: c.Name},
			Data:       map[string]string{env.DeliveryID: string(body)},
		}
		if createErr := c.Client.Create(ctx, &cm); createErr != nil {
			// A racing create is not a failure: another delivery got there
			// first, and the next attempt will take the update path.
			if apierrors.IsAlreadyExists(createErr) {
				return fmt.Errorf("configmap %s/%s was created concurrently, retrying as update", c.Namespace, c.Name)
			}
			return fmt.Errorf("create configmap %s/%s: %w", c.Namespace, c.Name, createErr)
		}
		return nil

	case err != nil:
		return fmt.Errorf("get configmap %s/%s: %w", c.Namespace, c.Name, err)
	}

	if cm.Data == nil {
		cm.Data = map[string]string{}
	}
	if existing, ok := cm.Data[env.DeliveryID]; ok && existing == string(body) {
		// Already delivered, byte for byte. This is the redelivery case the
		// DeliveryID exists for.
		return nil
	}
	cm.Data[env.DeliveryID] = string(body)

	if total := dataSize(cm.Data); total > maxConfigMapBytes {
		// Failing loudly beats writing an object the API server will reject, or
		// silently dropping a verdict to make room.
		return Permanent(fmt.Errorf(
			"configmap %s/%s would reach %d bytes, over the %d limit; rotate or shard the sink",
			c.Namespace, c.Name, total, maxConfigMapBytes))
	}

	if err := c.Client.Update(ctx, &cm); err != nil {
		// Conflicts are expected under concurrent deliveries and are retryable.
		return fmt.Errorf("update configmap %s/%s: %w", c.Namespace, c.Name, err)
	}
	return nil
}

func dataSize(data map[string]string) int {
	n := 0
	for k, v := range data {
		n += len(k) + len(v)
	}
	return n
}
