package runner

import (
	"strings"
	"testing"
)

// A payload line must NEVER become a metric — issue #143.
//
// This is the rule the whole fence exists for. A dcgmi JSON document is full of
// `"key": value` lines, and pkg/runner splits on the first "=" and accepts
// anything whose left side has no whitespace. Without the fence, a runner
// returning its own diagnostic output would silently rewrite its own
// measurements — and the rewritten values would look exactly like real ones.
func TestArtifactPayloadNeverBecomesAMetric(t *testing.T) {
	const out = `busbw=15.97
-----BEGIN BURNIN ARTIFACT dcgmi.json application/json-----
{
  "busbw=99999": "a value that looks like a metric",
  "miscompares=41": 1
}
eccErrors=777
-----END BURNIN ARTIFACT-----
NCCL_PASS
`
	res := Parse("nccl", out, 0)

	// The real measurement survives, untouched by the payload.
	if res.Metrics["busBandwidthGBs"] != "15.97" {
		t.Errorf("busBandwidthGBs = %q, want 15.97 — the payload overwrote a real measurement",
			res.Metrics["busBandwidthGBs"])
	}
	for _, forbidden := range []string{"eccErrors", "miscompares"} {
		if v, ok := res.Metrics[forbidden]; ok {
			t.Errorf("%s = %q was parsed OUT OF AN ARTIFACT PAYLOAD — a runner returning "+
				"evidence just invented a measurement", forbidden, v)
		}
	}
	// And the verdict line still landed.
	if res.Message != "NCCL_PASS" {
		t.Errorf("Message = %q, want the marker after the fence", res.Message)
	}

	if len(res.Artifacts) != 1 {
		t.Fatalf("got %d artifacts, want 1", len(res.Artifacts))
	}
	a := res.Artifacts[0]
	if a.Name != "dcgmi.json" || a.MediaType != "application/json" {
		t.Errorf("artifact = %+v, want the name and media type from the fence", a)
	}
	if !strings.Contains(a.Payload, "a value that looks like a metric") {
		t.Errorf("payload was not captured: %q", a.Payload)
	}
	if !strings.HasPrefix(a.Digest, "sha256:") || a.SizeBytes != len(a.Payload) {
		t.Errorf("artifact is not verifiable: digest=%q size=%d len=%d", a.Digest, a.SizeBytes, len(a.Payload))
	}
}

// A malformed fence is REPORTED, never silently swallowed. Silence is
// indistinguishable from a runner that emitted no evidence at all.
func TestMalformedFencesAreReportedNotSwallowed(t *testing.T) {
	cases := map[string]struct {
		out  string
		want string // substring of the Dropped reason
	}{
		"never closed": {
			out:  "-----BEGIN BURNIN ARTIFACT a.json application/json-----\n{\"x\":1}\n",
			want: "never closed",
		},
		"nested begin": {
			out: "-----BEGIN BURNIN ARTIFACT a.json application/json-----\n" +
				"-----BEGIN BURNIN ARTIFACT b.json application/json-----\n" +
				"-----END BURNIN ARTIFACT-----\n",
			want: "inside the payload",
		},
		"oversized": {
			out: "-----BEGIN BURNIN ARTIFACT big.bin application/octet-stream-----\n" +
				strings.Repeat("x", MaxArtifactBytes+1) + "\n" +
				"-----END BURNIN ARTIFACT-----\n",
			want: "over the",
		},
	}
	for name, c := range cases {
		t.Run(name, func(t *testing.T) {
			res := Parse("custom", c.out, 0)
			if len(res.Artifacts) == 0 {
				t.Fatal("a malformed artifact vanished entirely — a report that quietly omits " +
					"evidence is indistinguishable from a runner that produced none")
			}
			a := res.Artifacts[0]
			if a.Dropped == "" {
				t.Fatalf("a malformed artifact was ACCEPTED: %+v", a)
			}
			if !strings.Contains(a.Dropped, c.want) {
				t.Errorf("Dropped = %q, want it to mention %q", a.Dropped, c.want)
			}
			if a.Payload != "" {
				t.Errorf("a dropped artifact carried a payload: %q", a.Payload)
			}
		})
	}
}

// An oversized artifact records its TRUE size, because "how much evidence did we
// lose" is the question a reader asks next.
func TestOversizedArtifactRecordsItsTrueSize(t *testing.T) {
	body := strings.Repeat("x", MaxArtifactBytes+100)
	res := Parse("custom", "-----BEGIN BURNIN ARTIFACT big.bin application/octet-stream-----\n"+
		body+"\n-----END BURNIN ARTIFACT-----\n", 0)

	a := res.Artifacts[0]
	if a.SizeBytes <= MaxArtifactBytes {
		t.Errorf("SizeBytes = %d, want the payload's true size, over the %d cap",
			a.SizeBytes, MaxArtifactBytes)
	}
}

// A line that merely looks like a marker is not one. Both fields are required:
// a nameless artifact cannot be referenced and an untyped one cannot be
// rendered, and inventing a default would put this package in the business of
// guessing what a runner meant.
func TestIncompleteMarkersAreNotFences(t *testing.T) {
	for _, line := range []string{
		"-----BEGIN BURNIN ARTIFACT-----",                     // no name, no type
		"-----BEGIN BURNIN ARTIFACT only-a-name-----",         // no media type
		"-----BEGIN BURNIN ARTIFACT a b c-----",               // too many fields
		"----BEGIN BURNIN ARTIFACT a.json application/json--", // wrong fence
	} {
		res := Parse("custom", line+"\nmiscompares=0\nDONE\n", 0)
		if len(res.Artifacts) != 0 {
			t.Errorf("%q was treated as an artifact fence: %+v", line, res.Artifacts)
		}
		// And crucially the metric AFTER it still parses — a half-recognised
		// marker must not swallow the rest of stdout.
		if res.Metrics["miscompares"] != "0" {
			t.Errorf("%q swallowed the metrics that followed it", line)
		}
	}
}

// Nothing about an artifact can change a verdict. It is evidence ABOUT one.
func TestArtifactsDoNotAffectTheVerdict(t *testing.T) {
	const withArtifact = `-----BEGIN BURNIN ARTIFACT x.json application/json-----
{"anything": true}
-----END BURNIN ARTIFACT-----
miscompares=0
DONE
`
	const without = "miscompares=0\nDONE\n"

	for _, exit := range []int{0, 1, 3} {
		a, b := Parse("custom", withArtifact, exit), Parse("custom", without, exit)
		if a.Verdict != b.Verdict || a.ExitCode != b.ExitCode {
			t.Errorf("exit %d: artifact changed the verdict (%q vs %q)", exit, a.Verdict, b.Verdict)
		}
		if a.Metrics["miscompares"] != b.Metrics["miscompares"] {
			t.Errorf("exit %d: artifact changed the metrics", exit)
		}
		if a.Message != b.Message {
			t.Errorf("exit %d: artifact changed the message (%q vs %q)", exit, a.Message, b.Message)
		}
	}
}

// FuzzArtifactFenceNeverCorruptsParsing is the acceptance criterion of #143:
// a malformed fence must never corrupt metric parsing.
//
// The property is stated with a PLANTED POISON METRIC rather than by hoping the
// fuzzer invents a colliding key. A first attempt asserted only that a metric
// written BEFORE the fence survived, and 358k executions could not distinguish
// that from an extractor which handed every payload line straight to the metric
// scanner — because nothing in the generated corpus happened to spell a metric
// name the assertion knew about. So the payload carries a key=value line this
// test chose, and the question becomes exact: did it leak?
//
// It may legitimately leak by exactly one route — the fuzzed bytes contain a
// line that IS the END marker, which closes the fence early and makes everything
// after it ordinary stdout. That route is computed here rather than excused, so
// the assertion stays total.
func FuzzArtifactFenceNeverCorruptsParsing(f *testing.F) {
	f.Add("{\"a\": 1}")
	f.Add("-----BEGIN BURNIN ARTIFACT n.json application/json-----")
	f.Add("-----END BURNIN ARTIFACT-----")
	f.Add("busBandwidthGBs=99999")
	f.Add("-----END BURNIN ARTIFACT-----\r")
	f.Add(strings.Repeat("-----END BURNIN ARTIFACT-----\n", 40))
	f.Add("\x00\xff\xfe not utf-8")
	f.Add(strings.Repeat("x", MaxArtifactBytes+1))
	f.Add("")
	f.Add("\n\n\n")
	f.Add("-----BEGIN BURNIN ARTIFACT a b-----\n-----BEGIN BURNIN ARTIFACT c d-----")

	f.Fuzz(func(t *testing.T, fuzzed string) {
		const poison = "poisonedByPayload=666"
		const before = "miscompares=0"
		const after = "THE FINAL WORD"

		out := before + "\n" +
			"-----BEGIN BURNIN ARTIFACT f.bin application/octet-stream-----\n" +
			fuzzed + "\n" + poison + "\n" +
			"-----END BURNIN ARTIFACT-----\n" +
			after + "\n"

		res := Parse("custom", out, 0)

		// The measurement reported BEFORE the fence is untouchable.
		if res.Metrics["miscompares"] != "0" {
			t.Fatalf("a payload rewrote a metric reported before it: miscompares=%q",
				res.Metrics["miscompares"])
		}
		if res.Verdict != VerdictPass || res.ExitCode != 0 {
			t.Fatalf("a payload changed the verdict: %q exit %d", res.Verdict, res.ExitCode)
		}

		// The poison sits inside the fence and may escape only if the fuzzed
		// bytes closed the fence ahead of it.
		closedEarly := false
		for _, line := range strings.Split(fuzzed, "\n") {
			if strings.TrimRight(line, "\r") == artifactEnd {
				closedEarly = true
				break
			}
		}
		if _, leaked := res.Metrics["poisonedByPayload"]; leaked && !closedEarly {
			t.Fatalf("a key=value line INSIDE a fence became a metric — a runner returning "+
				"evidence just invented a measurement (payload %q)", fuzzed)
		}

		// Every artifact is self-consistent: accepted ones are verifiable,
		// dropped ones carry no payload and say why.
		for _, a := range res.Artifacts {
			if a.Dropped != "" {
				if a.Payload != "" || a.Digest != "" {
					t.Fatalf("dropped artifact carries content: %+v", a)
				}
				continue
			}
			if a.SizeBytes != len(a.Payload) {
				t.Fatalf("SizeBytes %d != len(Payload) %d", a.SizeBytes, len(a.Payload))
			}
			if len(a.Payload) > MaxArtifactBytes {
				t.Fatalf("an oversized payload was accepted: %d bytes", len(a.Payload))
			}
			if !strings.HasPrefix(a.Digest, "sha256:") {
				t.Fatalf("accepted artifact is not verifiable: %+v", a)
			}
		}
	})
}
