package controller

import (
	"context"
	"time"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"sigs.k8s.io/controller-runtime/pkg/log"

	burninv1alpha1 "github.com/baldwinSPC/glimmer-burnin/api/v1alpha1"
	"github.com/baldwinSPC/glimmer-burnin/internal/sink"
	"github.com/baldwinSPC/glimmer-burnin/pkg/contract"
)

// deliveryTimeout bounds one delivery fan-out. Deliveries run inside the
// reconcile loop, so an unreachable sink must cost seconds, not minutes —
// redelivery is safe (derived DeliveryID) and the next reconcile retries.
const deliveryTimeout = 15 * time.Second

// deliverRetry is deliberately shallow for the same reason: the durable retry
// mechanism is the reconcile loop itself, not a long in-process backoff chain.
var deliverRetry = sink.RetryPolicy{MaxAttempts: 2, Base: time.Second, Max: 2 * time.Second}

// deliver builds the envelope for (reason, eventKey) and sends it to every
// named sink. It reports whether EVERY sink accepted it. Failures are
// recorded on the sink's status and logged, never fatal to the run: a broken
// webhook must not stop hardware acceptance, and the derived DeliveryID makes
// the eventual redelivery safe.
func (r *BurnInRunReconciler) deliver(
	ctx context.Context,
	run *burninv1alpha1.BurnInRun,
	sinks []string,
	reason contract.Reason,
	eventKey string,
) bool {
	if len(sinks) == 0 {
		return true
	}
	logger := log.FromContext(ctx)
	env := sink.EnvelopeFor(run, run.Spec.ProfileRef, reason, eventKey, r.now())

	ctx, cancel := context.WithTimeout(ctx, deliveryTimeout)
	defer cancel()

	allOK := true
	for _, name := range sinks {
		var s burninv1alpha1.BurnInSink
		if err := r.Get(ctx, types.NamespacedName{Namespace: run.Namespace, Name: name}, &s); err != nil {
			logger.Error(err, "sink not found — delivery dropped for this transition", "sink", name)
			allOK = false
			continue
		}

		d, err := sink.Build(ctx, r.Client, &s, deliveryTimeout)
		if err == nil {
			err = sink.NewSender(d, deliverRetry).Send(ctx, env)
		}

		// Record the attempt on the sink so an operator can see delivery
		// health without reading controller logs. Best-effort: a status
		// conflict here must not fail the run's own reconcile.
		now := metav1.NewTime(r.now())
		if err != nil {
			logger.Error(err, "delivery failed", "sink", name, "reason", string(reason))
			s.Status.LastError = err.Error()
			allOK = false
		} else {
			s.Status.LastDelivery = &now
			s.Status.LastError = ""
		}
		if statusErr := r.Status().Update(ctx, &s); statusErr != nil {
			logger.V(1).Info("could not record delivery on sink status", "sink", name, "error", statusErr.Error())
		}
	}
	return allOK
}
