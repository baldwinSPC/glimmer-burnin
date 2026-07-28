package sink

import (
	"context"
	"strings"
	"testing"

	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/client/fake"

	"github.com/baldwinSPC/glimmer-burnin/pkg/contract"
)

func newFakeClient(objs ...client.Object) client.Client {
	scheme := runtime.NewScheme()
	_ = corev1.AddToScheme(scheme)
	return fake.NewClientBuilder().WithScheme(scheme).WithObjects(objs...).Build()
}

func TestConfigMapCreatesWhenAbsent(t *testing.T) {
	c := newFakeClient()
	cm := &ConfigMap{Client: c, Namespace: "burnin", Name: "results"}
	env := testEnvelope()

	if err := cm.Deliver(context.Background(), env); err != nil {
		t.Fatalf("Deliver: %v", err)
	}

	var got corev1.ConfigMap
	if err := c.Get(context.Background(), types.NamespacedName{Namespace: "burnin", Name: "results"}, &got); err != nil {
		t.Fatalf("Get: %v", err)
	}
	if _, ok := got.Data[env.DeliveryID]; !ok {
		t.Errorf("envelope not stored under its DeliveryID; keys: %v", keysOf(got.Data))
	}
}

// Storing under DeliveryID is what makes this sink idempotent: a redelivery
// must overwrite the same key, not accumulate duplicates.
func TestConfigMapRedeliveryIsIdempotent(t *testing.T) {
	c := newFakeClient()
	cm := &ConfigMap{Client: c, Namespace: "burnin", Name: "results"}
	env := testEnvelope()

	for i := 0; i < 3; i++ {
		if err := cm.Deliver(context.Background(), env); err != nil {
			t.Fatalf("Deliver #%d: %v", i+1, err)
		}
	}

	var got corev1.ConfigMap
	if err := c.Get(context.Background(), types.NamespacedName{Namespace: "burnin", Name: "results"}, &got); err != nil {
		t.Fatalf("Get: %v", err)
	}
	if len(got.Data) != 1 {
		t.Errorf("three redeliveries produced %d entries, want 1: %v", len(got.Data), keysOf(got.Data))
	}
}

func TestConfigMapKeepsDistinctDeliveries(t *testing.T) {
	c := newFakeClient()
	cm := &ConfigMap{Client: c, Namespace: "burnin", Name: "results"}

	first := testEnvelope()
	second := testEnvelope()
	second.Reason = contract.ReasonTestCompleted
	second.DeliveryID = contract.NewDeliveryID("uid-1", contract.ReasonTestCompleted, "fp4")

	for _, e := range []*contract.Envelope{first, second} {
		if err := cm.Deliver(context.Background(), e); err != nil {
			t.Fatalf("Deliver: %v", err)
		}
	}

	var got corev1.ConfigMap
	_ = c.Get(context.Background(), types.NamespacedName{Namespace: "burnin", Name: "results"}, &got)
	if len(got.Data) != 2 {
		t.Errorf("two distinct deliveries produced %d entries, want 2: %v", len(got.Data), keysOf(got.Data))
	}
}

// Silently dropping a verdict to stay under the size limit would be worse than
// failing: the run would look delivered when it was not.
func TestConfigMapRefusesToExceedSizeLimit(t *testing.T) {
	big := strings.Repeat("x", maxConfigMapBytes)
	existing := &corev1.ConfigMap{
		ObjectMeta: metav1.ObjectMeta{Namespace: "burnin", Name: "results"},
		Data:       map[string]string{"old": big},
	}
	c := newFakeClient(existing)
	cm := &ConfigMap{Client: c, Namespace: "burnin", Name: "results"}

	err := cm.Deliver(context.Background(), testEnvelope())
	if err == nil {
		t.Fatal("Deliver = nil, want an error when the ConfigMap would exceed the limit")
	}
	if !IsPermanent(err) {
		t.Errorf("size overflow should be permanent, not retried forever: %v", err)
	}
}

func keysOf(m map[string]string) []string {
	out := make([]string, 0, len(m))
	for k := range m {
		out = append(out, k)
	}
	return out
}
