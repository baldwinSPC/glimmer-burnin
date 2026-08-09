package main

import (
	"bytes"
	"context"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"

	api "github.com/baldwinSPC/glimmer-burnin/api/v1alpha1"
	"github.com/baldwinSPC/glimmer-burnin/internal/sink"
	"github.com/baldwinSPC/glimmer-burnin/pkg/contract"
)

func TestBothDispatchersDeriveTheSameDeliveryID(t *testing.T) {
	// The acceptance criterion for the whole feature: a receiver already
	// deduplicating the operator's traffic must deduplicate the CLI's with no
	// changes. If these ever diverge, a site pointing provisioning-time and
	// in-cluster burn-ins at one endpoint gets two histories per node and no
	// error anywhere.
	rep := sampleReport()
	fromCLI, err := EnvelopeFor(rep)
	if err != nil {
		t.Fatalf("EnvelopeFor: %v", err)
	}

	// The same run, as the operator would describe it: same UID, same phase.
	run := &api.BurnInRun{
		ObjectMeta: metav1.ObjectMeta{
			Namespace: "burnin", Name: "acceptance-spark-a",
			UID: types.UID(fromCLI.Run.UID),
		},
		Status: api.BurnInRunStatus{Phase: rep.Phase},
	}
	fromOperator := sink.EnvelopeFor(run, "acceptance", contract.ReasonPhaseChanged,
		sink.PhaseKey(rep.Phase), time.Now(), nil)

	if fromCLI.DeliveryID != fromOperator.DeliveryID {
		t.Errorf("delivery IDs diverged:\n  cli:      %s\n  operator: %s",
			fromCLI.DeliveryID, fromOperator.DeliveryID)
	}
	if fromCLI.Reason != fromOperator.Reason {
		t.Errorf("reason: cli %q, operator %q", fromCLI.Reason, fromOperator.Reason)
	}
}

func TestAFourHundredIsNotRetried(t *testing.T) {
	// A 4xx means the receiver will never accept this document. Retrying it
	// wastes a backoff chain and, on a long run, delays the operator noticing
	// the endpoint is misconfigured.
	var hits atomic.Int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		hits.Add(1)
		w.WriteHeader(http.StatusBadRequest)
	}))
	defer srv.Close()

	var log bytes.Buffer
	sender, err := sinkFlags{url: srv.URL}.deliverer()
	if err != nil {
		t.Fatal(err)
	}
	env, _ := EnvelopeFor(sampleReport())
	deliver(context.Background(), &log, sender, env)

	if n := hits.Load(); n != 1 {
		t.Errorf("a 400 was attempted %d times, want 1", n)
	}
	if !strings.Contains(log.String(), "REJECTED") {
		t.Errorf("a permanent rejection should be named as one:\n%s", log.String())
	}
}

func TestAFiveHundredIsRetriedAndTheVerdictSurvives(t *testing.T) {
	var hits atomic.Int32
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if hits.Add(1) < 3 {
			w.WriteHeader(http.StatusServiceUnavailable)
			return
		}
		w.WriteHeader(http.StatusOK)
	}))
	defer srv.Close()

	sender, err := sinkFlags{url: srv.URL}.deliverer()
	if err != nil {
		t.Fatal(err)
	}
	sender.Policy = sink.RetryPolicy{MaxAttempts: 5, Base: time.Millisecond, Max: time.Millisecond}

	var log bytes.Buffer
	env, _ := EnvelopeFor(sampleReport())
	deliver(context.Background(), &log, sender, env)

	if n := hits.Load(); n != 3 {
		t.Errorf("attempts = %d, want 3 (two failures then success)", n)
	}
	if !strings.Contains(log.String(), "delivered") {
		t.Errorf("a recovered delivery should say so:\n%s", log.String())
	}
}

func TestTheEnvelopeArrivesWithItsIdempotencyKeyAndToken(t *testing.T) {
	var gotKey, gotAuth, gotBody string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotKey = r.Header.Get(sink.IdempotencyKeyHeader)
		gotAuth = r.Header.Get("Authorization")
		b := make([]byte, 4096)
		n, _ := r.Body.Read(b)
		gotBody = string(b[:n])
		w.WriteHeader(http.StatusOK)
	}))
	defer srv.Close()

	tokenFile := filepath.Join(t.TempDir(), "token")
	if err := os.WriteFile(tokenFile, []byte("s3cret\n"), 0o600); err != nil {
		t.Fatal(err)
	}

	sender, err := sinkFlags{url: srv.URL, tokenFile: tokenFile}.deliverer()
	if err != nil {
		t.Fatal(err)
	}
	env, _ := EnvelopeFor(sampleReport())

	var log bytes.Buffer
	deliver(context.Background(), &log, sender, env)

	// The key rides in a header as well as the body so a gateway can dedupe at
	// its edge without parsing JSON.
	if gotKey != env.DeliveryID {
		t.Errorf("idempotency header = %q, want %q", gotKey, env.DeliveryID)
	}
	if gotAuth != "Bearer s3cret" {
		t.Errorf("authorization = %q — the trailing newline should be trimmed", gotAuth)
	}
	if !strings.Contains(gotBody, `"version"`) {
		t.Error("the receiver did not get an envelope")
	}
	// The token must not reach anything anyone reads.
	if strings.Contains(log.String(), "s3cret") {
		t.Errorf("the token leaked into the log:\n%s", log.String())
	}
	if strings.Contains(sender.Deliverer.Describe(), "s3cret") {
		t.Error("the token leaked into Describe()")
	}
}

func TestAnUnusableSinkIsRefusedBeforeAnythingRuns(t *testing.T) {
	// Caught in the first second rather than after a multi-hour soak has
	// produced the result it would have carried.
	rows := []struct {
		name  string
		flags sinkFlags
		want  string
	}{
		{"a token with no sink", sinkFlags{tokenFile: "/tmp/x"}, "needs --sink-url"},
		{"a URL that is not one", sinkFlags{url: "example.com/ingest"}, "http:// or https://"},
		{"a token file that is not there", sinkFlags{url: "https://x/y", tokenFile: "/nope/token"}, "reading the sink token"},
	}
	for _, r := range rows {
		_, err := r.flags.deliverer()
		if err == nil {
			t.Errorf("%s: accepted", r.name)
			continue
		}
		if !strings.Contains(err.Error(), r.want) {
			t.Errorf("%s: %v", r.name, err)
		}
	}

	// An empty token file fails rather than sending an empty Authorization
	// header, which a receiver rejects as a 401 — reading like a wrong
	// credential rather than an absent one.
	empty := filepath.Join(t.TempDir(), "empty")
	os.WriteFile(empty, []byte("  \n"), 0o600)
	if _, err := (sinkFlags{url: "https://x/y", tokenFile: empty}).deliverer(); err == nil ||
		!strings.Contains(err.Error(), "empty") {
		t.Errorf("an empty token file should be refused: %v", err)
	}

	// And no sink at all is not an error.
	if s, err := (sinkFlags{}).deliverer(); err != nil || s != nil {
		t.Errorf("no --sink-url should mean no sender: %v %v", s, err)
	}
}

func TestADeadSinkDoesNotChangeTheVerdict(t *testing.T) {
	// A misconfigured endpoint must never read as failing hardware. The
	// measurement happened; failing to tell anyone about it is an operational
	// problem with the sink.
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusInternalServerError)
	}))
	defer srv.Close()

	sender, _ := sinkFlags{url: srv.URL}.deliverer()
	sender.Policy = sink.RetryPolicy{MaxAttempts: 2, Base: time.Millisecond, Max: time.Millisecond}

	var log bytes.Buffer
	env, _ := EnvelopeFor(sampleReport())
	deliver(context.Background(), &log, sender, env) // must not panic, must not exit

	out := log.String()
	if !strings.Contains(out, "verdict stands") {
		t.Errorf("a failed delivery should say the verdict is unaffected:\n%s", out)
	}
}
