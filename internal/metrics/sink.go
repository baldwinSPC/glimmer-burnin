package metrics

import (
	"context"
	"errors"
	"fmt"

	"github.com/baldwinSPC/glimmer-burnin/pkg/contract"
)

// Sink is the Deliverer a BurnInSink of type Prometheus builds into.
//
// It satisfies internal/sink.Deliverer structurally, so this package does not
// import that one — the dependency runs sink -> metrics, which keeps the
// exporter free of retry policy, secrets and HTTP.
//
// What it does with an envelope is reflect it onto the exporter's gauges. That
// is the entire delivery: no network, no queue, no partial success. It is
// reported as a success so the reconciler's normal path records
// status.lastDelivery on the sink, which is the honest answer to "did this sink
// take the result?" — it did, and the result is on /metrics until the run is
// swept.
//
// The failure mode it deliberately does NOT have is the one it replaced. A
// Prometheus sink used to return a permanent error from Build, so every export
// for the whole run failed and the sink's status filled with the same message.
// A CRD that accepts a value must not then reject it on every use.
type Sink struct {
	exporter  *Exporter
	namespace string
	name      string
}

// SinkFor returns the Deliverer for a Prometheus BurnInSink.
//
// The BurnInSink object carries no configuration of its own — there is nothing
// to configure, because nothing is sent anywhere. Its role is to be NAMED by a
// profile: listing it marks that profile's runs for exposition, and not listing
// it means the run stays off /metrics. That is what makes exposition opt-in
// per profile rather than a global switch, so a cluster running a thousand
// throwaway smoke runs does not have to publish all of them.
func (e *Exporter) SinkFor(namespace, name string) *Sink {
	return &Sink{exporter: e, namespace: namespace, name: name}
}

// Describe names the destination for status and logs. It says where the result
// actually went, because "delivered to prometheus" would suggest a push that
// never happened.
func (s *Sink) Describe() string {
	return fmt.Sprintf("prometheus sink %s/%s (exposed on /metrics for scrape)", s.namespace, s.name)
}

// Deliver reflects the envelope onto the exporter.
//
// It cannot fail on anything external, so the only error it can return is a
// nil envelope, which would be a bug on this side rather than a delivery
// problem. Returning nil for everything else is the point: there is no retry
// worth scheduling and no backoff worth waiting.
func (s *Sink) Deliver(_ context.Context, env *contract.Envelope) error {
	if env == nil {
		return errors.New("prometheus sink: nil envelope")
	}
	s.exporter.Observe(env)
	return nil
}
