package sink

import (
	"bytes"
	"context"
	"crypto/tls"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"time"

	"github.com/baldwinSPC/glimmer-burnin/pkg/contract"
)

// IdempotencyKeyHeader carries the envelope's DeliveryID.
//
// The key is in a header as well as the body so a receiver can dedupe at its
// edge — a gateway or ingest handler can drop a repeat without parsing JSON.
const IdempotencyKeyHeader = "Idempotency-Key"

// Webhook POSTs an envelope to a URL.
type Webhook struct {
	URL string
	// Token is a bearer token, already resolved from its Secret. It is never
	// logged and never included in Describe.
	Token string
	// Client is the HTTP client to use. Required — the caller owns timeout and
	// TLS policy so they are configured in one place.
	Client *http.Client
}

// NewWebhook builds a Webhook with a sane client.
//
// timeout bounds a single attempt, not the whole retry chain: the Sender's
// context bounds that. insecureSkipVerify exists because the CRD exposes it for
// development, and is worth keeping visible at the call site.
func NewWebhook(url, token string, insecureSkipVerify bool, timeout time.Duration) *Webhook {
	if timeout <= 0 {
		timeout = 30 * time.Second
	}
	transport := http.DefaultTransport.(*http.Transport).Clone()
	if insecureSkipVerify {
		//nolint:gosec // surfaced as a documented dev-only field on the CRD
		transport.TLSClientConfig = &tls.Config{InsecureSkipVerify: true}
	}
	return &Webhook{
		URL:    url,
		Token:  token,
		Client: &http.Client{Timeout: timeout, Transport: transport},
	}
}

// Describe names the destination without leaking the token.
func (w *Webhook) Describe() string { return "webhook " + w.URL }

// Deliver POSTs the envelope.
func (w *Webhook) Deliver(ctx context.Context, env *contract.Envelope) error {
	body, err := json.Marshal(env)
	if err != nil {
		// A struct we built cannot be made marshalable by trying again.
		return Permanent(fmt.Errorf("encode envelope: %w", err))
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, w.URL, bytes.NewReader(body))
	if err != nil {
		return Permanent(fmt.Errorf("build request: %w", err))
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set(IdempotencyKeyHeader, env.DeliveryID)
	if w.Token != "" {
		req.Header.Set("Authorization", "Bearer "+w.Token)
	}

	resp, err := w.Client.Do(req)
	if err != nil {
		// Transport failures (DNS, refused, timeout) are exactly what retry is
		// for; a restarting ingest endpoint must not lose a soak's result.
		return fmt.Errorf("post to %s: %w", w.URL, err)
	}
	defer func() { _ = resp.Body.Close() }()

	// Drain a bounded amount so the connection can be reused, and so the error
	// message can quote what the receiver actually said.
	snippet, _ := io.ReadAll(io.LimitReader(resp.Body, 2048))

	switch {
	case resp.StatusCode >= 200 && resp.StatusCode < 300:
		return nil

	case resp.StatusCode == http.StatusTooManyRequests,
		resp.StatusCode == http.StatusRequestTimeout:
		// Explicit "slow down" / "try again" — retryable despite being 4xx.
		return fmt.Errorf("%s returned %s: %s", w.URL, resp.Status, snippet)

	case resp.StatusCode >= 400 && resp.StatusCode < 500:
		// The receiver rejected this request on its merits. Retrying sends the
		// identical bytes and gets the identical answer, while hiding the cause
		// behind attempt counts. Bad credentials land here, which is the case
		// most worth surfacing immediately rather than after six attempts.
		return Permanent(fmt.Errorf("%s rejected the delivery with %s: %s", w.URL, resp.Status, snippet))

	default:
		return fmt.Errorf("%s returned %s: %s", w.URL, resp.Status, snippet)
	}
}
