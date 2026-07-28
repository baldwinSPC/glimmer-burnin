package sink

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/baldwinSPC/glimmer-burnin/pkg/contract"
)

func testEnvelope() *contract.Envelope {
	return &contract.Envelope{
		Version:    contract.Version,
		DeliveryID: contract.NewDeliveryID("uid-1", contract.ReasonPhaseChanged, "Passed"),
		Reason:     contract.ReasonPhaseChanged,
		SentAt:     time.Unix(1750000000, 0).UTC(),
		Run:        contract.RunRef{Namespace: "burnin", Name: "run-1", UID: "uid-1"},
		Phase:      "Passed",
		Summary:    contract.Summary{Passed: 1},
	}
}

func TestWebhookSendsEnvelopeWithAuthAndIdempotencyKey(t *testing.T) {
	env := testEnvelope()

	var gotAuth, gotKey, gotType string
	var gotBody contract.Envelope
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotAuth = r.Header.Get("Authorization")
		gotKey = r.Header.Get(IdempotencyKeyHeader)
		gotType = r.Header.Get("Content-Type")
		_ = json.NewDecoder(r.Body).Decode(&gotBody)
		w.WriteHeader(http.StatusAccepted)
	}))
	defer srv.Close()

	wh := NewWebhook(srv.URL, "s3cr3t", false, time.Second)
	if err := wh.Deliver(context.Background(), env); err != nil {
		t.Fatalf("Deliver: %v", err)
	}

	if gotAuth != "Bearer s3cr3t" {
		t.Errorf("Authorization = %q, want bearer token", gotAuth)
	}
	// The key must also be a header so a receiver can dedupe at its edge,
	// without parsing the body.
	if gotKey != env.DeliveryID {
		t.Errorf("%s = %q, want %q", IdempotencyKeyHeader, gotKey, env.DeliveryID)
	}
	if gotType != "application/json" {
		t.Errorf("Content-Type = %q", gotType)
	}
	if gotBody.DeliveryID != env.DeliveryID || gotBody.Phase != "Passed" {
		t.Errorf("decoded body mismatch: %+v", gotBody)
	}
}

func TestWebhookOmitsAuthorizationWhenNoToken(t *testing.T) {
	var hadAuth bool
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, hadAuth = r.Header["Authorization"]
		w.WriteHeader(http.StatusOK)
	}))
	defer srv.Close()

	if err := NewWebhook(srv.URL, "", false, time.Second).Deliver(context.Background(), testEnvelope()); err != nil {
		t.Fatalf("Deliver: %v", err)
	}
	if hadAuth {
		t.Error("sent an Authorization header despite having no token")
	}
}

// The retry/permanent split is the whole point: a 4xx must surface immediately
// rather than hide behind six attempts, while a 5xx or a 429 must be retried.
func TestWebhookErrorClassification(t *testing.T) {
	cases := []struct {
		status    int
		permanent bool
	}{
		{http.StatusUnauthorized, true},         // bad credentials: retrying never helps
		{http.StatusForbidden, true},            //
		{http.StatusBadRequest, true},           //
		{http.StatusNotFound, true},             //
		{http.StatusTooManyRequests, false},     // explicit "slow down"
		{http.StatusRequestTimeout, false},      // explicit "try again"
		{http.StatusInternalServerError, false}, // endpoint is unwell
		{http.StatusBadGateway, false},          //
		{http.StatusServiceUnavailable, false},  // typically a restart
	}
	for _, c := range cases {
		srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			w.WriteHeader(c.status)
			_, _ = w.Write([]byte("detail from receiver"))
		}))
		err := NewWebhook(srv.URL, "", false, time.Second).Deliver(context.Background(), testEnvelope())
		srv.Close()

		if err == nil {
			t.Errorf("status %d: Deliver = nil, want an error", c.status)
			continue
		}
		if got := IsPermanent(err); got != c.permanent {
			t.Errorf("status %d: IsPermanent = %v, want %v (err: %v)", c.status, got, c.permanent, err)
		}
		// The receiver's own explanation is the most useful part of the error.
		if !strings.Contains(err.Error(), "detail from receiver") {
			t.Errorf("status %d: error drops the receiver's response body: %v", c.status, err)
		}
	}
}

func TestWebhookTransportFailureIsRetryable(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {}))
	url := srv.URL
	srv.Close() // nothing is listening now

	err := NewWebhook(url, "", false, time.Second).Deliver(context.Background(), testEnvelope())
	if err == nil {
		t.Fatal("Deliver to a closed server = nil, want an error")
	}
	if IsPermanent(err) {
		t.Errorf("a connection failure was classified permanent; a restarting endpoint would lose the verdict: %v", err)
	}
}

// A retried delivery must be indistinguishable from the first, or the receiver
// cannot recognise it as a duplicate.
func TestRetriedDeliveryKeepsSameIdempotencyKey(t *testing.T) {
	var attempts int32
	var keys []string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		keys = append(keys, r.Header.Get(IdempotencyKeyHeader))
		if atomic.AddInt32(&attempts, 1) < 3 {
			w.WriteHeader(http.StatusServiceUnavailable)
			return
		}
		w.WriteHeader(http.StatusOK)
	}))
	defer srv.Close()

	s := NewSender(NewWebhook(srv.URL, "", false, time.Second), RetryPolicy{MaxAttempts: 5, Base: time.Millisecond})
	s.sleep = func(context.Context, time.Duration) error { return nil }

	if err := s.Send(context.Background(), testEnvelope()); err != nil {
		t.Fatalf("Send: %v", err)
	}
	if len(keys) != 3 {
		t.Fatalf("got %d attempts, want 3", len(keys))
	}
	for i, k := range keys {
		if k != keys[0] {
			t.Errorf("attempt %d used key %q, first used %q; the receiver could not dedupe", i+1, k, keys[0])
		}
	}
}
