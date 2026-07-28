package contract

import (
	"encoding/json"
	"strings"
	"testing"
	"time"
)

// The retry design rests entirely on this: a redelivery must carry the same key
// so the receiver can discard it, while a genuinely different delivery must
// not collide with it.
func TestDeliveryIDIsStableAndDistinct(t *testing.T) {
	const uid = "9f1c2f4e-0000-4000-8000-000000000001"

	a := NewDeliveryID(uid, ReasonPhaseChanged, "Passed")
	b := NewDeliveryID(uid, ReasonPhaseChanged, "Passed")
	if a != b {
		t.Fatalf("same delivery produced different IDs (%q vs %q); retries would be applied twice", a, b)
	}

	distinct := map[string]string{
		"other phase":  NewDeliveryID(uid, ReasonPhaseChanged, "Failed"),
		"other reason": NewDeliveryID(uid, ReasonTestCompleted, "Passed"),
		"other run":    NewDeliveryID("9f1c2f4e-0000-4000-8000-000000000002", ReasonPhaseChanged, "Passed"),
		"checkpoint":   NewDeliveryID(uid, ReasonCheckpoint, "1"),
	}
	for what, id := range distinct {
		if id == a {
			t.Errorf("%s collided with the original delivery ID; a distinct delivery would be silently dropped", what)
		}
	}
}

// Field separation matters: without it, (reason="a", key="bc") and
// (reason="ab", key="c") would hash identically.
func TestDeliveryIDFieldsCannotRunTogether(t *testing.T) {
	x := NewDeliveryID("uid", Reason("a"), "bc")
	y := NewDeliveryID("uid", Reason("ab"), "c")
	if x == y {
		t.Error("delivery ID fields are concatenated without separation; distinct deliveries collide")
	}
}

func validEnvelope() *Envelope {
	return &Envelope{
		Version:    Version,
		DeliveryID: NewDeliveryID("uid-1", ReasonPhaseChanged, "Passed"),
		Reason:     ReasonPhaseChanged,
		SentAt:     time.Unix(1750000000, 0).UTC(),
		Run:        RunRef{Namespace: "burnin", Name: "run-1", UID: "uid-1", Profile: "spark-acceptance"},
		Phase:      "Passed",
		Summary:    Summary{Passed: 2},
	}
}

func TestValidateAcceptsWellFormed(t *testing.T) {
	if err := validEnvelope().Validate(); err != nil {
		t.Fatalf("Validate() on a well-formed envelope = %v, want nil", err)
	}
}

func TestValidateRejects(t *testing.T) {
	cases := map[string]func(*Envelope){
		"missing version":    func(e *Envelope) { e.Version = "" },
		"unknown version":    func(e *Envelope) { e.Version = "burnin.glimmer.ai/v9" },
		"missing deliveryID": func(e *Envelope) { e.DeliveryID = "" },
		"missing reason":     func(e *Envelope) { e.Reason = "" },
		"missing run UID":    func(e *Envelope) { e.Run.UID = "" },
		"zero sentAt":        func(e *Envelope) { e.SentAt = time.Time{} },
	}
	for what, mutate := range cases {
		e := validEnvelope()
		mutate(e)
		if err := e.Validate(); err == nil {
			t.Errorf("Validate() with %s = nil, want an error", what)
		}
	}
}

// A malformed metric name must be caught before delivery, not stored by the
// consumer and discovered later when nothing can chart it.
func TestValidateRejectsMalformedMetricNames(t *testing.T) {
	e := validEnvelope()
	e.Results = []TestResult{{
		Name:    "nccl-allreduce-pair",
		Kind:    "nccl",
		Phase:   "Passed",
		Metrics: map[string]string{"bus_bandwidth": "23.4"},
	}}
	err := e.Validate()
	if err == nil {
		t.Fatal("Validate() accepted a snake_case metric name")
	}
	if !strings.Contains(err.Error(), "results[0].metrics") {
		t.Errorf("error %q does not locate the offending field", err)
	}
}

// The envelope crosses a repo boundary, so its wire form is the contract.
// This pins the field names a consumer will key off.
func TestJSONWireFormat(t *testing.T) {
	b, err := json.Marshal(validEnvelope())
	if err != nil {
		t.Fatalf("Marshal: %v", err)
	}
	for _, want := range []string{
		`"version":"burnin.glimmer.ai/v1alpha1"`,
		`"deliveryId":`,
		`"reason":"RunPhaseChanged"`,
		`"sentAt":`,
		`"run":{"namespace":"burnin","name":"run-1","uid":"uid-1","profile":"spark-acceptance"}`,
		`"phase":"Passed"`,
		`"summary":{"passed":2,"failed":0}`,
	} {
		if !strings.Contains(string(b), want) {
			t.Errorf("wire format missing %s\ngot: %s", want, b)
		}
	}
}

// A consumer must be able to decode what we encode.
func TestRoundTrip(t *testing.T) {
	in := validEnvelope()
	in.Fingerprint = map[string]string{"gpuModel": "NVIDIA GB10"}
	in.Results = []TestResult{{
		Name: "fp4", Kind: "compute-smoke", Phase: "Passed",
		Metrics: map[string]string{"throughputTflops": "108.77"},
	}}
	b, err := json.Marshal(in)
	if err != nil {
		t.Fatalf("Marshal: %v", err)
	}
	var out Envelope
	if err := json.Unmarshal(b, &out); err != nil {
		t.Fatalf("Unmarshal: %v", err)
	}
	if out.DeliveryID != in.DeliveryID || out.Phase != in.Phase {
		t.Errorf("round trip changed identity: %+v", out)
	}
	if out.Results[0].Metrics["throughputTflops"] != "108.77" {
		t.Errorf("round trip lost metrics: %+v", out.Results)
	}
	if err := out.Validate(); err != nil {
		t.Errorf("decoded envelope failed validation: %v", err)
	}
}
