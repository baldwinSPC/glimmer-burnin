package sink

import (
	"context"
	"strings"
	"testing"
	"time"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"

	burninv1alpha1 "github.com/baldwinSPC/glimmer-burnin/api/v1alpha1"
)

// A Prometheus sink used to validate in the CRD and then fail permanently for
// the entire life of every run that named it. An enum value the API accepts
// must not be a trap.
func TestBuildPrometheusSinkIsUsable(t *testing.T) {
	s := &burninv1alpha1.BurnInSink{
		ObjectMeta: metav1.ObjectMeta{Namespace: "burnin", Name: "scrape-me"},
		Spec:       burninv1alpha1.BurnInSinkSpec{Type: burninv1alpha1.SinkPrometheus},
	}

	d, err := Build(context.Background(), newFakeClient(), s, time.Second)
	if err != nil {
		t.Fatalf("Build on a Prometheus sink = %v, want a Deliverer", err)
	}
	if d == nil {
		t.Fatal("Build returned a nil Deliverer with no error")
	}

	// Describe feeds sink status; it must say where the result actually went,
	// not imply a push that never happens.
	desc := d.Describe()
	if !strings.Contains(desc, "scrape-me") || !strings.Contains(desc, "/metrics") {
		t.Errorf("Describe() = %q, want it to name the sink and the scrape endpoint", desc)
	}

	// Exposition has no network step, so delivery has nothing that can fail
	// and nothing worth retrying.
	if err := NewSender(d, RetryPolicy{MaxAttempts: 1, Base: time.Millisecond}).Send(context.Background(), testEnvelope()); err != nil {
		t.Errorf("Send to a Prometheus sink = %v, want nil", err)
	}
}
