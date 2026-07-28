package sink

import (
	"context"
	"errors"
	"testing"
	"time"

	"github.com/baldwinSPC/glimmer-burnin/pkg/contract"
)

type stubDeliverer struct {
	calls  int
	errs   []error // returned in order; nil past the end
	lastID string
}

func (s *stubDeliverer) Deliver(_ context.Context, env *contract.Envelope) error {
	s.lastID = env.DeliveryID
	s.calls++
	if s.calls-1 < len(s.errs) {
		return s.errs[s.calls-1]
	}
	return nil
}
func (s *stubDeliverer) Describe() string { return "stub" }

func newTestSender(d Deliverer, p RetryPolicy) *Sender {
	s := NewSender(d, p)
	s.sleep = func(context.Context, time.Duration) error { return nil }
	return s
}

func TestSendRetriesUntilSuccess(t *testing.T) {
	d := &stubDeliverer{errs: []error{errors.New("boom"), errors.New("boom")}}
	s := newTestSender(d, RetryPolicy{MaxAttempts: 5, Base: time.Millisecond})

	if err := s.Send(context.Background(), testEnvelope()); err != nil {
		t.Fatalf("Send = %v, want nil", err)
	}
	if d.calls != 3 {
		t.Errorf("made %d attempts, want 3", d.calls)
	}
}

func TestSendGivesUpAfterMaxAttempts(t *testing.T) {
	d := &stubDeliverer{errs: []error{
		errors.New("e1"), errors.New("e2"), errors.New("e3"), errors.New("e4"),
	}}
	s := newTestSender(d, RetryPolicy{MaxAttempts: 3, Base: time.Millisecond})

	err := s.Send(context.Background(), testEnvelope())
	if err == nil {
		t.Fatal("Send = nil, want an error")
	}
	if d.calls != 3 {
		t.Errorf("made %d attempts, want exactly MaxAttempts (3)", d.calls)
	}
}

// A permanent failure must stop immediately. Retrying it wastes the run's
// remaining time and buries the actual cause under attempt counts.
func TestSendStopsOnPermanentError(t *testing.T) {
	d := &stubDeliverer{errs: []error{Permanent(errors.New("bad credentials"))}}
	s := newTestSender(d, RetryPolicy{MaxAttempts: 5, Base: time.Millisecond})

	err := s.Send(context.Background(), testEnvelope())
	if err == nil {
		t.Fatal("Send = nil, want an error")
	}
	if d.calls != 1 {
		t.Errorf("made %d attempts on a permanent error, want 1", d.calls)
	}
	if !IsPermanent(err) {
		t.Errorf("permanence was lost through the retry wrapper: %v", err)
	}
}

// An invalid envelope is our bug. It must fail locally rather than being
// hammered at a consumer's ingest endpoint.
func TestSendRejectsInvalidEnvelopeWithoutDelivering(t *testing.T) {
	d := &stubDeliverer{}
	s := newTestSender(d, RetryPolicy{MaxAttempts: 5, Base: time.Millisecond})

	bad := testEnvelope()
	bad.Run.UID = ""

	err := s.Send(context.Background(), bad)
	if err == nil {
		t.Fatal("Send accepted an invalid envelope")
	}
	if d.calls != 0 {
		t.Errorf("attempted %d deliveries of an invalid envelope, want 0", d.calls)
	}
	if !IsPermanent(err) {
		t.Errorf("an invalid envelope should be permanent, got: %v", err)
	}
}

func TestSendAbandonsOnContextCancellation(t *testing.T) {
	d := &stubDeliverer{errs: []error{errors.New("e1"), errors.New("e2"), errors.New("e3")}}
	s := NewSender(d, RetryPolicy{MaxAttempts: 5, Base: time.Hour})
	ctx, cancel := context.WithCancel(context.Background())
	// Cancel while waiting to retry, as a manager shutdown would.
	s.sleep = func(context.Context, time.Duration) error {
		cancel()
		return context.Canceled
	}

	err := s.Send(ctx, testEnvelope())
	if err == nil {
		t.Fatal("Send = nil, want an error")
	}
	if d.calls != 1 {
		t.Errorf("made %d attempts, want 1 before cancellation", d.calls)
	}
	// The original failure must survive; "context canceled" alone would hide why
	// the delivery was being retried at all.
	if !errors.Is(err, context.Canceled) {
		t.Errorf("error does not wrap context.Canceled: %v", err)
	}
}

func TestBackoffGrowsAndIsCapped(t *testing.T) {
	p := RetryPolicy{MaxAttempts: 10, Base: time.Second, Max: 8 * time.Second}.withDefaults()
	want := []time.Duration{
		1 * time.Second, // attempt 2
		2 * time.Second,
		4 * time.Second,
		8 * time.Second,
		8 * time.Second, // capped
		8 * time.Second,
	}
	for i, w := range want {
		attempt := i + 2
		if got := p.backoff(attempt, nil); got != w {
			t.Errorf("backoff(attempt=%d) = %v, want %v", attempt, got, w)
		}
	}
}

func TestZeroPolicyGetsDefaults(t *testing.T) {
	p := RetryPolicy{}.withDefaults()
	if p.MaxAttempts != DefaultRetryPolicy.MaxAttempts || p.Base != DefaultRetryPolicy.Base || p.Max != DefaultRetryPolicy.Max {
		t.Errorf("zero policy did not pick up defaults: %+v", p)
	}
}
