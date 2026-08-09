package sink

import (
	"context"
	"encoding/json"
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
		PhaseKey(run.Status.Phase), time.Unix(1750000100, 0), nil)

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
	a := EnvelopeFor(run, "p", contract.ReasonPhaseChanged, PhaseKey(run.Status.Phase), time.Unix(1750000100, 0), nil)
	b := EnvelopeFor(run, "p", contract.ReasonPhaseChanged, PhaseKey(run.Status.Phase), time.Unix(1750009999, 0), nil)

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
	phase := EnvelopeFor(run, "p", contract.ReasonPhaseChanged, PhaseKey(run.Status.Phase), time.Now(), nil)
	test := EnvelopeFor(run, "p", contract.ReasonTestCompleted, TestKey("fp4"), time.Now(), nil)
	cp1 := EnvelopeFor(run, "p", contract.ReasonCheckpoint, CheckpointKey(1), time.Now(), nil)
	cp2 := EnvelopeFor(run, "p", contract.ReasonCheckpoint, CheckpointKey(2), time.Now(), nil)

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
		"unknown type": {
			ObjectMeta: metav1.ObjectMeta{Namespace: "burnin", Name: "d"},
			Spec:       burninv1alpha1.BurnInSinkSpec{Type: burninv1alpha1.SinkType("Carrier Pigeon")},
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

// A run that failed three gates delivers all three — issue #139.
//
// Before this, Message carried one sentence about the FIRST violation (that
// field is frozen at those semantics), so a consumer had to parse English to
// learn whether a node was implicated. Cause is the field that decides whether
// anybody should walk to a rack, and it existed only inside the cluster.
func TestEnvelopeCarriesEveryViolation(t *testing.T) {
	run := &burninv1alpha1.BurnInRun{
		ObjectMeta: metav1.ObjectMeta{Namespace: "burnin", Name: "run1", UID: "uid-1"},
		Status: burninv1alpha1.BurnInRunStatus{
			Phase: burninv1alpha1.RunFailed,
			Results: []burninv1alpha1.TestResult{{
				Name: "soak", Kind: "thermal-soak", Phase: burninv1alpha1.RunFailed,
				Nodes:   []string{"spark-a"},
				Message: "sustainedClockPct 61.0 below the 70 floor",
				Violations: []burninv1alpha1.Violation{
					{Index: 0, Metric: "sustainedClockPct", Cause: "Measurement", Kind: "Unsatisfied", Reason: "61.0 < 70"},
					{Index: 1, Metric: "throttleEvents", Cause: "Measurement", Kind: "Unsatisfied", Reason: "3 != 0"},
					{Index: 2, Metric: "gpuTemperature", Cause: "Authoring", Kind: "MetricUnregistered", Reason: "no such metric"},
				},
			}},
		},
	}
	env := EnvelopeFor(run, "acceptance", contract.ReasonPhaseChanged, "", time.Unix(1750000000, 0), nil)

	got := env.Results[0].Violations
	if len(got) != 3 {
		t.Fatalf("envelope carried %d violation(s), want 3 — a consumer parsing Message sees "+
			"only the first", len(got))
	}
	// The field that decides whether anybody walks to a rack.
	if got[2].Cause != "Authoring" {
		t.Errorf("violation[2].Cause = %q, want Authoring — a broken threshold must not be "+
			"delivered as a hardware shortfall", got[2].Cause)
	}
	if got[0].Metric != "sustainedClockPct" || got[1].Metric != "throttleEvents" {
		t.Errorf("violations lost their metric names: %+v", got)
	}
}

// "This part has no ECC" and "this part reported zero ECC errors" are different
// statements about different hardware, and both used to deliver as prose.
func TestEnvelopeCarriesUnmeasurableAndNotEvaluated(t *testing.T) {
	run := &burninv1alpha1.BurnInRun{
		ObjectMeta: metav1.ObjectMeta{Namespace: "burnin", Name: "run1", UID: "uid-1"},
		Status: burninv1alpha1.BurnInRunStatus{
			Phase: burninv1alpha1.RunPassed,
			Results: []burninv1alpha1.TestResult{{
				Name: "health", Kind: "host-health", Phase: burninv1alpha1.RunPassed,
				Nodes:        []string{"spark-a"},
				Metrics:      map[string]string{"xidEvents": "0"},
				Unmeasurable: []string{"eccErrors", "remappedRows"},
				NotEvaluated: []burninv1alpha1.NotEvaluated{
					{Metric: "eccErrors", Reason: "unmeasurable on this hardware"},
				},
			}},
		},
	}
	env := EnvelopeFor(run, "acceptance", contract.ReasonPhaseChanged, "", time.Unix(1750000000, 0), nil)
	r := env.Results[0]

	if len(r.Unmeasurable) != 2 {
		t.Errorf("unmeasurable = %v, want both declarations — this is a claim about the PART "+
			"and must reach the consumer intact", r.Unmeasurable)
	}
	if len(r.NotEvaluated) != 1 || r.NotEvaluated[0].Metric != "eccErrors" {
		t.Errorf("notEvaluated = %+v — a Passed test whose ECC gate never ran must not be "+
			"indistinguishable from one whose ECC gate ran and was satisfied", r.NotEvaluated)
	}
	// And the run still passed: recording this changes no verdict.
	if env.Phase != string(burninv1alpha1.RunPassed) {
		t.Errorf("phase = %q, want Passed", env.Phase)
	}
}

// A cancelled run's envelope says WHY it was cancelled — issue #140.
//
// Without it a consumer cannot tell an operator-requested stop from a deadline
// expiry, and those mean opposite things about the hardware: one is a human
// deciding to stop, the other is a test that ran out of time and measured
// nothing.
func TestCancelledEnvelopeCarriesTheReason(t *testing.T) {
	run := &burninv1alpha1.BurnInRun{
		ObjectMeta: metav1.ObjectMeta{Namespace: "burnin", Name: "run1", UID: "uid-1"},
		Spec:       burninv1alpha1.BurnInRunSpec{CancelReason: "power event in row 4"},
		Status:     burninv1alpha1.BurnInRunStatus{Phase: burninv1alpha1.RunCancelled},
	}
	env := EnvelopeFor(run, "p", contract.ReasonPhaseChanged, "", time.Unix(1750000000, 0), nil)
	if env.CancelReason != "power event in row 4" {
		t.Errorf("cancelReason = %q, want the operator's own words", env.CancelReason)
	}

	// A run that was NOT cancelled carries none, so a consumer can read its
	// presence as a signal rather than checking a sentinel.
	run.Status.Phase = burninv1alpha1.RunPassed
	if got := EnvelopeFor(run, "p", contract.ReasonPhaseChanged, "", time.Unix(1750000000, 0), nil); got.CancelReason != "" {
		t.Errorf("a passed run carried cancelReason=%q", got.CancelReason)
	}
}

// Checkpoints are orderable. The sequence already exists — it is what makes
// DeliveryID stable across retries — and without it a consumer receiving them
// out of order cannot put a long soak's progress record back together.
func TestCheckpointEnvelopeCarriesItsSequence(t *testing.T) {
	run := &burninv1alpha1.BurnInRun{
		ObjectMeta: metav1.ObjectMeta{Namespace: "burnin", Name: "run1", UID: "uid-1"},
		Status:     burninv1alpha1.BurnInRunStatus{Phase: burninv1alpha1.RunRunning},
	}
	cp := EnvelopeFor(run, "p", contract.ReasonCheckpoint, CheckpointKey(7), time.Unix(1750000000, 0), nil)
	if cp.CheckpointSequence != 7 {
		t.Errorf("checkpointSequence = %d, want 7", cp.CheckpointSequence)
	}
	// Only on checkpoints: a phase change has no sequence to carry.
	phase := EnvelopeFor(run, "p", contract.ReasonPhaseChanged, PhaseKey(run.Status.Phase), time.Unix(1750000000, 0), nil)
	if phase.CheckpointSequence != 0 {
		t.Errorf("a phase-change envelope carried checkpointSequence=%d", phase.CheckpointSequence)
	}
}

// The cluster block is present only when the operator was told, and never
// guessed. A verdict attributed to the wrong fleet is worse than one attributed
// to none.
func TestClusterIsCarriedButNeverInvented(t *testing.T) {
	run := &burninv1alpha1.BurnInRun{
		ObjectMeta: metav1.ObjectMeta{Namespace: "burnin", Name: "run1", UID: "uid-1"},
		Status:     burninv1alpha1.BurnInRunStatus{Phase: burninv1alpha1.RunPassed},
	}
	with := EnvelopeFor(run, "p", contract.ReasonPhaseChanged, "", time.Unix(1750000000, 0),
		&contract.ClusterRef{Name: "spark-lab", UID: "ks-uid"})
	if with.Cluster == nil || with.Cluster.Name != "spark-lab" || with.Cluster.UID != "ks-uid" {
		t.Errorf("cluster = %+v, want both fields", with.Cluster)
	}

	without := EnvelopeFor(run, "p", contract.ReasonPhaseChanged, "", time.Unix(1750000000, 0), nil)
	if without.Cluster != nil {
		t.Errorf("an operator with no cluster configured emitted %+v", without.Cluster)
	}
	if err := without.Validate(); err != nil {
		t.Errorf("an envelope with no cluster must still validate: %v", err)
	}
}

// The summary accounts for every execution, which is the one thing a summary is
// for. It used to count two of the four phases.
func TestSummaryCountsErroredAndSkipped(t *testing.T) {
	run := &burninv1alpha1.BurnInRun{
		ObjectMeta: metav1.ObjectMeta{Namespace: "burnin", Name: "run1", UID: "uid-1"},
		Status: burninv1alpha1.BurnInRunStatus{
			Phase:   burninv1alpha1.RunError,
			Passed:  2,
			Failed:  1,
			Errored: 3,
			Skipped: 4,
			Results: make([]burninv1alpha1.TestResult, 10),
		},
	}
	s := EnvelopeFor(run, "p", contract.ReasonPhaseChanged, "", time.Unix(1750000000, 0), nil).Summary
	if s.Errored != 3 || s.Skipped != 4 {
		t.Errorf("summary = %+v, want errored=3 skipped=4", s)
	}
	if total := s.Passed + s.Failed + s.Errored + s.Skipped; total != 10 {
		t.Errorf("summary totals %d across 10 results — a consumer asking 'how many did not "+
			"pass' still cannot answer it from the summary", total)
	}
}

// The summary is EXHAUSTIVE, on every delivery reason — issue #190.
//
// A summary that counts some of the phases is worse than none: a consumer
// reading "8 passed, 0 failed" off a ten-execution run sees a clean sweep, and
// the two executions that errored — hardware nobody measured — are invisible.
// The one existing defence is a consumer that ignores Summary entirely and
// re-walks Results, which is the shape of a contract nobody can use.
//
// Asserted per REASON because the envelope is built by one function under four
// of them, and a checkpoint delivery mid-run is exactly where a partial tally
// would be easiest to justify and hardest to notice.
func TestSummaryIsExhaustiveOnEveryReason(t *testing.T) {
	run := &burninv1alpha1.BurnInRun{
		ObjectMeta: metav1.ObjectMeta{Namespace: "burnin", Name: "run1", UID: "uid-1"},
		Status: burninv1alpha1.BurnInRunStatus{
			Phase:   burninv1alpha1.RunError,
			Passed:  2,
			Failed:  3,
			Errored: 1,
			Skipped: 4,
			Results: make([]burninv1alpha1.TestResult, 10),
		},
	}
	for _, reason := range contract.Reasons {
		t.Run(string(reason), func(t *testing.T) {
			s := EnvelopeFor(run, "p", reason, "k", time.Unix(1750000000, 0), nil).Summary
			total := s.Passed + s.Failed + s.Errored + s.Skipped
			if total != int32(len(run.Status.Results)) {
				t.Errorf("summary totals %d across %d results (%+v) — a consumer asking "+
					"'how many did not pass' cannot answer it from this",
					total, len(run.Status.Results), s)
			}
			if s.Errored != 1 || s.Skipped != 4 {
				t.Errorf("summary = %+v, want errored=1 skipped=4 — an errored execution "+
					"measured nothing, and reporting it as anything else certifies "+
					"hardware nobody looked at", s)
			}
		})
	}
}

// The flag rides into every envelope, so a consumer can never mistake a
// thresholdless sweep for a certification — issue #142.
func TestEnvelopeCarriesBaseline(t *testing.T) {
	mk := func(baseline *bool) *contract.Envelope {
		run := &burninv1alpha1.BurnInRun{
			ObjectMeta: metav1.ObjectMeta{Namespace: "burnin", Name: "run1", UID: "uid-1"},
			Spec:       burninv1alpha1.BurnInRunSpec{Baseline: baseline},
			Status: burninv1alpha1.BurnInRunStatus{
				Phase:   burninv1alpha1.RunPassed,
				Results: []burninv1alpha1.TestResult{{Name: "sweep", Phase: burninv1alpha1.RunPassed}},
			},
		}
		return EnvelopeFor(run, "p", contract.ReasonPhaseChanged, "", time.Unix(1750000000, 0), nil)
	}

	sweep := mk(func() *bool { b := true; return &b }())
	if !sweep.Baseline {
		t.Error("a baseline run's envelope does not say so — a consumer gating admission on " +
			"'the last run passed' would certify hardware against a run that gated nothing")
	}
	// Both phases are Passed. The flag is the ONLY thing distinguishing them.
	if sweep.Phase != string(burninv1alpha1.RunPassed) {
		t.Errorf("phase = %q, want Passed — a baseline still passes on exit 0", sweep.Phase)
	}

	for _, unset := range []*bool{nil, func() *bool { b := false; return &b }()} {
		if mk(unset).Baseline {
			t.Error("an ordinary acceptance run was delivered as a baseline")
		}
	}
}

// A segmented soak must be legible from the envelope alone — issue #229.
//
// "40 of 288 segments" and "288 of 288" are different statements about a node,
// and before this they arrived looking identical: the envelope carried Metrics
// and nothing about how the test was executed. A consumer could not tell a fold
// from a single reading, nor a soak that completed from one that stopped a
// seventh of the way in — and the elapsedS >= 0.95 x duration gate in
// docs/soaks.md exists precisely because that difference matters.
//
// Attempts compound it: a segmented result's attempt history is capped, so a
// consumer counting attempts silently undercounts unless something says how many
// were dropped.
func TestEnvelopeDescribesASegmentedSoak(t *testing.T) {
	run := &burninv1alpha1.BurnInRun{
		ObjectMeta: metav1.ObjectMeta{Namespace: "burnin", Name: "run1", UID: "uid-1"},
		Status: burninv1alpha1.BurnInRunStatus{
			Phase: burninv1alpha1.RunError,
			Results: []burninv1alpha1.TestResult{{
				Name: "soak", Kind: "thermal-soak", Phase: burninv1alpha1.RunError,
				Nodes:             []string{"spark-a"},
				Metrics:           map[string]string{"elapsedS": "36000", "gpuTempC": "71.5"},
				SegmentsRequired:  288,
				SegmentsCompleted: 40,
				TruncatedAttempts: 23,
				AggregatedMetrics: map[string]string{"elapsedS": "36000", "gpuTempC": "71.5"},
			}},
		},
	}
	env := EnvelopeFor(run, "acceptance", contract.ReasonPhaseChanged, "", time.Unix(1750000000, 0), nil)
	r := env.Results[0]

	if r.Segments == nil {
		t.Fatalf("a 288-segment soak delivered with no segment description: a consumer cannot tell it "+
			"from a single execution, nor 40/288 from 288/288 (result: %+v)", r)
	}
	if r.Segments.Completed != 40 || r.Segments.Required != 288 {
		t.Errorf("segments = %d/%d, want 40/288", r.Segments.Completed, r.Segments.Required)
	}
	if r.Segments.TruncatedAttempts != 23 {
		t.Errorf("truncatedAttempts = %d, want 23 — without it a consumer counting the delivered "+
			"attempts undercounts and never learns that it did", r.Segments.TruncatedAttempts)
	}
}

// Absence is the signal: an ordinary execution carries no segment block at all,
// so a consumer reads `segments == null` as "this was one pod" rather than
// having to compare a zero against a sentinel.
func TestAnUnsegmentedResultCarriesNoSegmentBlock(t *testing.T) {
	run := &burninv1alpha1.BurnInRun{
		ObjectMeta: metav1.ObjectMeta{Namespace: "burnin", Name: "run1", UID: "uid-1"},
		Status: burninv1alpha1.BurnInRunStatus{
			Phase: burninv1alpha1.RunPassed,
			Results: []burninv1alpha1.TestResult{{
				Name: "smoke", Kind: "compute-smoke", Phase: burninv1alpha1.RunPassed,
				Nodes: []string{"spark-a"}, Metrics: map[string]string{"tflops": "31.2"},
			}},
		},
	}
	env := EnvelopeFor(run, "acceptance", contract.ReasonPhaseChanged, "", time.Unix(1750000000, 0), nil)
	if env.Results[0].Segments != nil {
		t.Errorf("an unsegmented result grew a segment block: %+v", env.Results[0].Segments)
	}
	// And it must not appear in the JSON either, or `omitempty` is not doing the
	// work the contract's "present only where it means something" posture needs.
	blob, err := json.Marshal(env.Results[0])
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	if strings.Contains(string(blob), "segments") {
		t.Errorf("an unsegmented result serialised a segments key: %s", blob)
	}
}
