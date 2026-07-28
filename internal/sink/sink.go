// Package sink delivers burn-in verdicts to a BurnInSink.
//
// Delivery is the integration seam: everything upstream of it is this
// operator's business, and everything downstream is a consumer's. The design
// constraint that shapes this package is that the network is unreliable and the
// verdict matters, so a delivery is retried until it succeeds or is provably
// pointless — which is only safe because every envelope carries a derived
// idempotency key the receiver can dedupe on.
package sink

import (
	"context"
	"errors"
	"fmt"
	"math/rand"
	"time"

	"github.com/baldwinSPC/glimmer-burnin/pkg/contract"
)

// Deliverer sends one envelope to one destination.
//
// Implementations must be safe to call again with the same envelope: a caller
// that cannot tell whether a delivery landed will retry it.
type Deliverer interface {
	// Deliver sends the envelope. Returning a PermanentError signals that
	// retrying is pointless; any other error is treated as retryable.
	Deliver(ctx context.Context, env *contract.Envelope) error
	// Describe names the destination for status and logs, without secrets.
	Describe() string
}

// PermanentError marks a failure that retrying cannot fix — a malformed
// envelope, or a receiver that rejected the request on its merits (4xx).
// Retrying these burns the run's remaining time and buries the real cause.
type PermanentError struct {
	Err error
}

func (e *PermanentError) Error() string { return e.Err.Error() }
func (e *PermanentError) Unwrap() error { return e.Err }

// Permanent wraps err as non-retryable.
func Permanent(err error) error { return &PermanentError{Err: err} }

// IsPermanent reports whether retrying err is pointless.
func IsPermanent(err error) bool {
	var p *PermanentError
	return errors.As(err, &p)
}

// RetryPolicy bounds how hard a delivery is retried.
//
// The defaults are deliberately patient rather than aggressive: a burn-in
// verdict is not latency-sensitive, and a consumer's ingest endpoint restarting
// is a routine event that should not lose a multi-day soak's result.
type RetryPolicy struct {
	// MaxAttempts includes the first try. Zero means DefaultRetryPolicy's value.
	MaxAttempts int
	// Base is the first backoff interval; it doubles each attempt.
	Base time.Duration
	// Max caps a single interval, so a long retry chain does not drift into
	// hour-long gaps.
	Max time.Duration
	// Jitter randomises each interval by up to +/-25%. Without it, every run in
	// a fleet that started together retries in lockstep and hammers a
	// recovering endpoint in synchronised waves.
	Jitter bool
}

// DefaultRetryPolicy is used when a zero RetryPolicy is supplied.
var DefaultRetryPolicy = RetryPolicy{
	MaxAttempts: 6,
	Base:        2 * time.Second,
	Max:         2 * time.Minute,
	Jitter:      true,
}

func (p RetryPolicy) withDefaults() RetryPolicy {
	if p.MaxAttempts <= 0 {
		p.MaxAttempts = DefaultRetryPolicy.MaxAttempts
	}
	if p.Base <= 0 {
		p.Base = DefaultRetryPolicy.Base
	}
	if p.Max <= 0 {
		p.Max = DefaultRetryPolicy.Max
	}
	return p
}

// backoff returns the wait before attempt n (1-based: attempt 1 needs no wait,
// and attempt 2 — the first retry — waits exactly Base).
func (p RetryPolicy) backoff(attempt int, rnd *rand.Rand) time.Duration {
	d := p.Base
	for i := 2; i < attempt; i++ {
		d *= 2
		if d >= p.Max {
			d = p.Max
			break
		}
	}
	if p.Jitter && rnd != nil {
		// +/-25%
		delta := float64(d) * 0.25
		d = time.Duration(float64(d) + (rnd.Float64()*2-1)*delta)
		if d < 0 {
			d = 0
		}
	}
	return d
}

// Sender retries a Deliverer according to a policy.
type Sender struct {
	Deliverer Deliverer
	Policy    RetryPolicy
	// sleep is swappable so tests do not actually wait.
	sleep func(context.Context, time.Duration) error
	rnd   *rand.Rand
}

// NewSender wraps d with retry behaviour.
func NewSender(d Deliverer, p RetryPolicy) *Sender {
	return &Sender{
		Deliverer: d,
		Policy:    p.withDefaults(),
		sleep:     sleepCtx,
		//nolint:gosec // jitter needs spread, not unpredictability
		rnd: rand.New(rand.NewSource(time.Now().UnixNano())),
	}
}

func sleepCtx(ctx context.Context, d time.Duration) error {
	if d <= 0 {
		return nil
	}
	t := time.NewTimer(d)
	defer t.Stop()
	select {
	case <-ctx.Done():
		return ctx.Err()
	case <-t.C:
		return nil
	}
}

// Send delivers env, retrying per the policy.
//
// It validates before the first attempt: an envelope that cannot be valid is a
// bug on this side, and hammering a consumer's endpoint with it would turn a
// local defect into someone else's incident.
func (s *Sender) Send(ctx context.Context, env *contract.Envelope) error {
	if err := env.Validate(); err != nil {
		return Permanent(err)
	}

	var last error
	for attempt := 1; attempt <= s.Policy.MaxAttempts; attempt++ {
		if attempt > 1 {
			if err := s.sleep(ctx, s.Policy.backoff(attempt, s.rnd)); err != nil {
				return fmt.Errorf("delivery to %s abandoned after %d attempt(s): %w (last error: %v)",
					s.Deliverer.Describe(), attempt-1, err, last)
			}
		}

		err := s.Deliverer.Deliver(ctx, env)
		if err == nil {
			return nil
		}
		last = err

		if IsPermanent(err) {
			return fmt.Errorf("delivery to %s failed permanently: %w", s.Deliverer.Describe(), err)
		}
		if ctx.Err() != nil {
			return fmt.Errorf("delivery to %s abandoned: %w (last error: %v)",
				s.Deliverer.Describe(), ctx.Err(), last)
		}
	}
	return fmt.Errorf("delivery to %s failed after %d attempts: %w",
		s.Deliverer.Describe(), s.Policy.MaxAttempts, last)
}
