package main

import (
	"context"
	"fmt"
	"io"
	"os"
	"strings"
	"time"

	"github.com/baldwinSPC/glimmer-burnin/internal/sink"
	"github.com/baldwinSPC/glimmer-burnin/pkg/contract"
)

// Delivery from the CLI reuses the operator's own webhook deliverer and Sender.
//
// Not a parallel implementation, deliberately, and this is the whole point of
// the feature: a receiver already deduplicating the operator's traffic
// deduplicates the CLI's with no changes at all. DeliveryID is a pure function
// of run UID, reason and event key, so provisioning-time burn-ins and
// in-cluster burn-ins land in one coherent history per node.
//
// A second implementation would have to be kept identical by review, and the
// first divergence would show up as duplicate rows in someone's console rather
// than as a test failure.

// sinkFlags configure delivery.
type sinkFlags struct {
	url       string
	tokenFile string
	insecure  bool
	timeout   time.Duration
}

// deliverer builds the Sender, or nil when no sink was asked for.
func (f sinkFlags) deliverer() (*sink.Sender, error) {
	if f.url == "" {
		if f.tokenFile != "" {
			return nil, fmt.Errorf("--sink-token-file needs --sink-url")
		}
		return nil, nil
	}
	if !strings.HasPrefix(f.url, "http://") && !strings.HasPrefix(f.url, "https://") {
		return nil, fmt.Errorf("--sink-url must be http:// or https://, got %q", f.url)
	}

	token, err := readToken(f.tokenFile)
	if err != nil {
		return nil, err
	}
	timeout := f.timeout
	if timeout <= 0 {
		timeout = 30 * time.Second
	}
	// NewSender, not a struct literal: the constructor is what applies the
	// retry defaults and installs the clock.
	return sink.NewSender(sink.NewWebhook(f.url, token, f.insecure, timeout), sink.RetryPolicy{}), nil
}

// readToken reads a bearer token from a file.
//
// A FILE and not a flag. A credential in argv is a credential in every process
// listing on the box, readable by any user for as long as the run lasts — which
// on a burn-in is hours to weeks. The file's own permissions are the site's
// choice; putting it in the command line takes that choice away.
func readToken(path string) (string, error) {
	if path == "" {
		return "", nil
	}
	b, err := os.ReadFile(path)
	if err != nil {
		// The path is safe to name; the contents are not, and are never read
		// into an error message.
		return "", fmt.Errorf("reading the sink token from %s: %w", path, err)
	}
	token := strings.TrimSpace(string(b))
	if token == "" {
		// Failing here rather than sending an empty Authorization header, which
		// a receiver would reject as a 401 — a permanent error that reads like a
		// wrong credential rather than an absent one.
		return "", fmt.Errorf("the sink token file %s is empty", path)
	}
	return token, nil
}

// deliver posts the envelope, reporting what happened.
//
// A delivery failure never changes the run's exit code. The measurement
// happened and its verdict stands; failing to tell someone about it is an
// operational problem with the sink, and conflating the two would let a
// misconfigured endpoint read as failing hardware.
func deliver(ctx context.Context, w io.Writer, s *sink.Sender, env contract.Envelope) {
	if s == nil {
		return
	}
	fmt.Fprintf(w, "burnin: delivering to %s\n", s.Deliverer.Describe())

	if err := s.Send(ctx, &env); err != nil {
		if sink.IsPermanent(err) {
			// Named as permanent so nobody schedules a retry that cannot work.
			fmt.Fprintf(w, "burnin: delivery REJECTED (not retried): %v\n", err)
		} else {
			fmt.Fprintf(w, "burnin: delivery failed after retries: %v\n", err)
		}
		fmt.Fprintf(w, "burnin: the run's verdict stands; the envelope is in the results directory\n")
		return
	}
	fmt.Fprintf(w, "burnin: delivered %s\n", env.DeliveryID)
}
