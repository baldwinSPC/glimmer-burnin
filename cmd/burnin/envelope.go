package main

import (
	"crypto/rand"
	"encoding/hex"
	"fmt"
	"time"

	"github.com/baldwinSPC/glimmer-burnin/pkg/contract"
	"github.com/baldwinSPC/glimmer-burnin/pkg/hostinfo"
	"github.com/baldwinSPC/glimmer-burnin/pkg/localrun"
)

// EnvelopeFor turns a local run into the SAME document the operator delivers.
//
// This is what makes the dual-path design pay off downstream: `burnin report`
// renders it, an ingest endpoint accepts it, and the delivery ID is derived the
// same way — so a receiver already deduplicating the operator's traffic
// deduplicates this with no changes at all.
//
// The run identity is synthetic. contract.Envelope.Validate requires a non-empty
// UID and nothing more, so a minted one is legal; namespace is "local", which
// says plainly that no apiserver ever knew about this run rather than inventing
// a namespace that sounds like it might have.
func EnvelopeFor(rep localrun.Report) (contract.Envelope, error) {
	uid, err := mintUID()
	if err != nil {
		return contract.Envelope{}, err
	}

	run := contract.RunRef{
		Namespace: "local",
		Name:      rep.Node,
		UID:       uid,
	}

	env := contract.Envelope{
		Version:    contract.Version,
		DeliveryID: contract.NewDeliveryID(uid, contract.ReasonPhaseChanged, string(rep.Phase)),
		Reason:     contract.ReasonPhaseChanged,
		SentAt:     time.Now().UTC(),
		Run:        run,
		Phase:      string(rep.Phase),
		Summary: contract.Summary{
			Passed:  rep.Summary.Passed,
			Failed:  rep.Summary.Failed,
			Errored: rep.Summary.Errored,
			Skipped: rep.Summary.Skipped,
		},
		Fingerprint: fingerprint(rep.Node),
	}

	for _, r := range rep.Results {
		env.Results = append(env.Results, toContractResult(r))
	}
	if err := env.Validate(); err != nil {
		return contract.Envelope{}, fmt.Errorf("built an envelope the contract rejects: %w", err)
	}
	return env, nil
}

func toContractResult(r localrun.TestResult) contract.TestResult {
	out := contract.TestResult{
		Name:         r.Name,
		Kind:         r.Kind,
		Scope:        string(r.Scope),
		Phase:        string(r.Phase),
		Nodes:        r.Nodes,
		Metrics:      r.Metrics,
		Message:      r.Message,
		Unmeasurable: r.Unmeasurable,
		VariantAxes:  r.VariantAxes,
	}
	if !r.StartedAt.IsZero() {
		t := r.StartedAt.UTC()
		out.StartedAt = &t
	}
	if !r.FinishedAt.IsZero() {
		t := r.FinishedAt.UTC()
		out.FinishedAt = &t
	}
	for _, v := range r.Violations {
		out.Violations = append(out.Violations, contract.Violation{
			Index:  v.Index,
			Metric: v.Metric,
			Cause:  v.Cause,
			Kind:   v.Kind,
			Reason: v.Reason,
		})
	}
	for _, n := range r.NotEvaluated {
		out.NotEvaluated = append(out.NotEvaluated, contract.NotEvaluated{Metric: n.Metric, Reason: n.Reason})
	}
	return out
}

// fingerprint records what the machine was, in the operator's own shape:
// keyed by node name, with a space-joined descriptor.
//
// Best effort. A field the probe could not establish is left out rather than
// filled with a placeholder, which is the same rule the probe itself follows.
func fingerprint(node string) map[string]string {
	h := hostinfo.Probe(hostinfo.Options{})

	var parts []string
	add := func(k, v string) {
		if v != "" {
			parts = append(parts, k+"="+v)
		}
	}
	add("kernel", h.Kernel)
	add("os", h.OSImage)
	add("arch", h.Arch)
	for _, a := range h.Accelerators {
		add("accelerator", a.Vendor+":"+a.DeviceID)
	}
	if len(parts) == 0 {
		return nil
	}
	return map[string]string{node: joinSpace(parts)}
}

func joinSpace(parts []string) string {
	out := ""
	for i, p := range parts {
		if i > 0 {
			out += " "
		}
		out += p
	}
	return out
}

// mintUID generates a stable identity for one local run.
//
// Random rather than derived from the node name and a timestamp: two runs on one
// machine are different runs, and a receiver that deduplicates on the delivery
// ID must not have the second silently discarded as a replay of the first.
func mintUID() (string, error) {
	b := make([]byte, 16)
	if _, err := rand.Read(b); err != nil {
		return "", fmt.Errorf("generating a run id: %w", err)
	}
	return hex.EncodeToString(b), nil
}
