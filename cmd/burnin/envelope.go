package main

import (
	"crypto/rand"
	"encoding/hex"
	"fmt"
	"strconv"
	"time"

	api "github.com/baldwinSPC/glimmer-burnin/api/v1alpha1"
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
	id, err := NewRunIdentity(rep.Node)
	if err != nil {
		return contract.Envelope{}, err
	}
	return id.Final(rep)
}

// RunIdentity is one local run's synthetic identity, minted ONCE.
//
// It exists because a run now delivers more than one envelope: every checkpoint
// is a delivery, and the final verdict is another. They must agree about WHICH
// RUN they describe, or a consumer receives a soak's progress under one UID and
// its verdict under a different one and has no way to join them — which is
// worse than no checkpoints at all, because it looks like two runs happened.
//
// The operator has this for free: a BurnInRun is an apiserver object with a UID.
// Here it is minted, and minting it per envelope was fine only while there was
// exactly one.
type RunIdentity struct {
	run contract.RunRef
	// fingerprint is captured ONCE, at the start, like the operator's. Hardware
	// does not change mid-run, and re-probing per delivery would make a
	// checkpoint's fingerprint disagree with the verdict's if a driver were
	// reloaded underneath.
	fingerprint map[string]string
}

// NewRunIdentity mints one, and probes THIS machine's hardware once.
func NewRunIdentity(node string) (*RunIdentity, error) {
	return NewRunIdentityFor(node, fingerprint(node))
}

// NewRunIdentityFor mints one with a fingerprint the caller already has.
//
// For `burnin merge`, which folds records written on OTHER machines. Probing
// the host that happens to be running the merge — usually a laptop — would put
// its kernel and PCI devices in the envelope as the hardware the collective's
// verdict applies to, which is a statement about machines that took no part in
// the test.
func NewRunIdentityFor(node string, fp map[string]string) (*RunIdentity, error) {
	uid, err := mintUID()
	if err != nil {
		return nil, err
	}
	if len(fp) == 0 {
		// Absent rather than empty: a fingerprint nobody established must not
		// look like one that was taken and found nothing.
		fp = nil
	}
	return &RunIdentity{
		run:         contract.RunRef{Namespace: "local", Name: node, UID: uid},
		fingerprint: fp,
	}, nil
}

// Checkpoint is the envelope for one in-progress sample.
//
// A CHECKPOINT IS EVIDENCE, NEVER A VERDICT, and the envelope says so in the
// only ways the contract has: Reason is Checkpoint, and Phase is Running. The
// TestResult it carries is also Running and has no Violations — nothing has
// been judged, because the execution is not over.
func (i *RunIdentity) Checkpoint(c localrun.Checkpoint, kind string, scope api.TestScope, nodes []string) (contract.Envelope, error) {
	at := c.At.UTC()
	env := contract.Envelope{
		Version: contract.Version,
		// The sequence is what makes this stable across a retry — see
		// localrun.checkpointSequence. A key that moved per attempt would mint
		// a new identity each time and defeat the receiver's dedupe.
		DeliveryID:         contract.NewDeliveryID(i.run.UID, contract.ReasonCheckpoint, strconv.Itoa(c.Sequence)),
		Reason:             contract.ReasonCheckpoint,
		SentAt:             time.Now().UTC(),
		Run:                i.run,
		Phase:              string(api.RunRunning),
		CheckpointSequence: c.Sequence,
		Fingerprint:        i.fingerprint,
		Results: []contract.TestResult{{
			Name:      c.Test,
			Kind:      kind,
			Scope:     string(scope),
			Phase:     string(api.RunRunning),
			Nodes:     nodes,
			Metrics:   c.Metrics,
			StartedAt: &at,
		}},
	}
	if err := env.Validate(); err != nil {
		return contract.Envelope{}, fmt.Errorf("built a checkpoint envelope the contract rejects: %w", err)
	}
	return env, nil
}

// Final is the envelope for the run's verdict.
func (i *RunIdentity) Final(rep localrun.Report) (contract.Envelope, error) {
	uid := i.run.UID
	env := contract.Envelope{
		Version:    contract.Version,
		DeliveryID: contract.NewDeliveryID(uid, contract.ReasonPhaseChanged, string(rep.Phase)),
		Reason:     contract.ReasonPhaseChanged,
		SentAt:     time.Now().UTC(),
		Run:        i.run,
		Phase:      string(rep.Phase),
		Summary: contract.Summary{
			Passed:  rep.Summary.Passed,
			Failed:  rep.Summary.Failed,
			Errored: rep.Summary.Errored,
			Skipped: rep.Summary.Skipped,
		},
		Fingerprint: i.fingerprint,
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
