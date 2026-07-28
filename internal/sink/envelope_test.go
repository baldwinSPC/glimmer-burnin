package sink

import (
	"context"
	"strings"
	"testing"
	"time"

	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"

	burninv1alpha1 "github.com/baldwinSPC/glimmer-burnin/api/v1alpha1"
	"github.com/baldwinSPC/glimmer-burnin/pkg/contract"
)

func sampleRun() *burninv1alpha1.BurnInRun {
	started := metav1.NewTime(time.Unix(1750000000, 0))
	finished := metav1.NewTime(time.Unix(1750000060, 0))
	return &burninv1alpha1.BurnInRun{
		ObjectMeta: metav1.ObjectMeta{Namespace: "burnin", Name: "run-1", UID: "uid-abc"},
		Spec:       burninv1alpha1.BurnInRunSpec{ProfileRef: "spark-acceptance"},
		Status: burninv1alpha1.BurnInRunStatus{
			Phase:       burninv1alpha1.RunPassed,
			Fingerprint: map[string]string{"gpuModel": "NVIDIA GB10"},
			Passed:      1,
			Results: []burninv1alpha1.TestResult{{
				Name:       "fp4",
				Kind:       burninv1alpha1.TestKind("compute-smoke"),
				Phase:      burninv1alpha1.RunPassed,
				StartedAt:  &started,
				FinishedAt: &finished,
				Metrics:    map[string]string{"throughputTflops": "108.77"},
			}},
		},
	}
}

func TestEnvelopeForProducesValidEnvelope(t *testing.T) {
	run := sampleRun()
	env := EnvelopeFor(run, run.Spec.ProfileRef, contract.ReasonPhaseChanged,
		PhaseKey(run.Status.Phase), time.Unix(1750000100, 0))

	if err := env.Validate(); err != nil {
		t.Fatalf("built envelope fails validation: %v", err)
	}
	if env.Run.UID != "uid-abc" || env.Run.Profile != "spark-acceptance" {
		t.Errorf("run reference wrong: %+v", env.Run)
	}
	if env.Phase != "Passed" || env.Summary.Passed != 1 {
		t.Errorf("phase/summary wrong: phase=%q summary=%+v", env.Phase, env.Summary)
	}
	if len(env.Results) != 1 || env.Results[0].Metrics["throughputTflops"] != "108.77" {
		t.Errorf("results not carried across: %+v", env.Results)
	}
	if env.Results[0].StartedAt == nil || env.Results[0].FinishedAt == nil {
		t.Error("timestamps dropped in conversion")
	}
}

// The DeliveryID must depend on what happened, not on when we built the
// envelope — otherwise every retry mints a new key and dedupe is impossible.
func TestEnvelopeDeliveryIDIsIndependentOfSendTime(t *testing.T) {
	run := sampleRun()
	a := EnvelopeFor(run, "p", contract.ReasonPhaseChanged, PhaseKey(run.Status.Phase), time.Unix(1750000100, 0))
	b := EnvelopeFor(run, "p", contract.ReasonPhaseChanged, PhaseKey(run.Status.Phase), time.Unix(1750009999, 0))

	if a.DeliveryID != b.DeliveryID {
		t.Errorf("DeliveryID changed with the clock (%q vs %q); retries would never be recognised as duplicates",
			a.DeliveryID, b.DeliveryID)
	}
	if a.SentAt.Equal(b.SentAt) {
		t.Error("SentAt should still reflect the actual send time")
	}
}

func TestDeliveryKeysDistinguishReasons(t *testing.T) {
	run := sampleRun()
	phase := EnvelopeFor(run, "p", contract.ReasonPhaseChanged, PhaseKey(run.Status.Phase), time.Now())
	test := EnvelopeFor(run, "p", contract.ReasonTestCompleted, TestKey("fp4"), time.Now())
	cp1 := EnvelopeFor(run, "p", contract.ReasonCheckpoint, CheckpointKey(1), time.Now())
	cp2 := EnvelopeFor(run, "p", contract.ReasonCheckpoint, CheckpointKey(2), time.Now())

	ids := map[string]string{
		"phase": phase.DeliveryID, "test": test.DeliveryID,
		"checkpoint1": cp1.DeliveryID, "checkpoint2": cp2.DeliveryID,
	}
	seen := map[string]string{}
	for what, id := range ids {
		if prev, dup := seen[id]; dup {
			t.Errorf("%s and %s share a DeliveryID; one would be discarded as a duplicate", what, prev)
		}
		seen[id] = what
	}
}

func TestBuildWebhookResolvesToken(t *testing.T) {
	secret := &corev1.Secret{
		ObjectMeta: metav1.ObjectMeta{Namespace: "burnin", Name: "sink-token"},
		Data:       map[string][]byte{"token": []byte("s3cr3t")},
	}
	c := newFakeClient(secret)
	s := &burninv1alpha1.BurnInSink{
		ObjectMeta: metav1.ObjectMeta{Namespace: "burnin", Name: "glimmer"},
		Spec: burninv1alpha1.BurnInSinkSpec{
			Type: burninv1alpha1.SinkWebhook,
			Webhook: &burninv1alpha1.WebhookSink{
				URL:            "https://example.invalid/api/v1/burnin/results",
				TokenSecretRef: &corev1.SecretKeySelector{LocalObjectReference: corev1.LocalObjectReference{Name: "sink-token"}, Key: "token"},
			},
		},
	}

	d, err := Build(context.Background(), c, s, time.Second)
	if err != nil {
		t.Fatalf("Build: %v", err)
	}
	wh, ok := d.(*Webhook)
	if !ok {
		t.Fatalf("Build returned %T, want *Webhook", d)
	}
	if wh.Token != "s3cr3t" {
		t.Errorf("token not resolved from secret, got %q", wh.Token)
	}
	// Describe feeds status and logs, so it must never carry the token.
	if got := d.Describe(); got == "" || strings.Contains(got, "s3cr3t") {
		t.Errorf("Describe() leaks the token: %q", got)
	}
}

func TestBuildRejectsMisconfiguredSinks(t *testing.T) {
	c := newFakeClient()
	cases := map[string]*burninv1alpha1.BurnInSink{
		"webhook without config": {
			ObjectMeta: metav1.ObjectMeta{Namespace: "burnin", Name: "a"},
			Spec:       burninv1alpha1.BurnInSinkSpec{Type: burninv1alpha1.SinkWebhook},
		},
		"configmap without config": {
			ObjectMeta: metav1.ObjectMeta{Namespace: "burnin", Name: "b"},
			Spec:       burninv1alpha1.BurnInSinkSpec{Type: burninv1alpha1.SinkConfigMap},
		},
		"missing secret": {
			ObjectMeta: metav1.ObjectMeta{Namespace: "burnin", Name: "c"},
			Spec: burninv1alpha1.BurnInSinkSpec{
				Type: burninv1alpha1.SinkWebhook,
				Webhook: &burninv1alpha1.WebhookSink{
					URL:            "https://example.invalid/",
					TokenSecretRef: &corev1.SecretKeySelector{LocalObjectReference: corev1.LocalObjectReference{Name: "nope"}, Key: "token"},
				},
			},
		},
		"prometheus is not a push target": {
			ObjectMeta: metav1.ObjectMeta{Namespace: "burnin", Name: "d"},
			Spec:       burninv1alpha1.BurnInSinkSpec{Type: burninv1alpha1.SinkPrometheus},
		},
	}
	for what, s := range cases {
		_, err := Build(context.Background(), c, s, time.Second)
		if err == nil {
			t.Errorf("%s: Build = nil, want an error", what)
			continue
		}
		// Configuration is not fixed by retrying it.
		if !IsPermanent(err) {
			t.Errorf("%s: error should be permanent, got %v", what, err)
		}
	}
}
