package sanitize

import (
	"bytes"
	"encoding/json"
	"strings"
	"testing"
)

func mustNew(t *testing.T, o Options) *Sanitizer {
	t.Helper()
	if o.Salt == nil {
		o.Salt = bytes.Repeat([]byte{7}, 32)
	}
	s, err := New(o)
	if err != nil {
		t.Fatal(err)
	}
	return s
}

const envelope = `{
  "version": "burnin.glimmer.ai/v1alpha1",
  "deliveryId": "9f2c1ab0d5e34786b1c2a4f60e7d8931",
  "reason": "RunPhaseChanged",
  "sentAt": "2026-09-27T10:00:00Z",
  "run": {"namespace": "local", "name": "spark-043a", "uid": "6f0e6c62-0e6e-4d9a-9a5b-2c1f9a6d1a77"},
  "producer": {"name": "burnin", "version": "v0.9.1", "commit": "1e44a67665baeb3ed9e9b779915c1a797c788e7a"},
  "phase": "Passed",
  "fingerprint": {"spark-043a": "kernel=6.11.0-1013-nvidia os=Ubuntu 24.04.2 LTS arch=arm64"},
  "results": [{
    "name": "fabric", "kind": "ib-write-bw", "scope": "Pair", "phase": "Passed",
    "nodes": ["spark-043a", "spark-85a9"],
    "metrics": {"bandwidthGbps": "97.51", "gpuName": "NVIDIA GB10", "pciAddresses": "000f:01:00.0",
                "tcpTestInterface": "enP2p1s0f1np1"},
    "message": "peer 10.252.161.209 (spark-85a9) gid fe80:0000:0000:0000:5ebb:f6ff:fe12:3456 mac 5c:bb:f6:12:34:56",
    "unmeasurable": ["eccErrors"],
    "artifacts": [{"name": "per-device.json", "digest": "sha256:3f7a0c9d1e2b4a6c8e0f1a3b5c7d9e1f3a5b7c9d1e3f5a7b9c1d3e5f7a9b1c3d"}]
  }],
  "summary": {"passed": 1, "failed": 0, "errored": 0, "skipped": 0}
}`

func sanitizeEnvelope(t *testing.T, s *Sanitizer) (string, map[string]any) {
	t.Helper()
	out, err := s.JSON([]byte(envelope))
	if err != nil {
		t.Fatal(err)
	}
	var m map[string]any
	if err := json.Unmarshal(out, &m); err != nil {
		t.Fatal(err)
	}
	return string(out), m
}

func TestNodeNamesAddressesAndIDsAreGoneAndMeasurementsAreNot(t *testing.T) {
	s := mustNew(t, Options{Hosts: []string{"spark-043a", "spark-85a9"}})
	out, m := sanitizeEnvelope(t, s)

	for _, leak := range []string{
		"spark-043a", "spark-85a9", "10.252.161.209", "6f0e6c62-0e6e", "5c:bb:f6:12:34:56",
		"fe80:0000:0000:0000", "9f2c1ab0d5e34786b1c2a4f60e7d8931",
	} {
		if strings.Contains(out, leak) {
			t.Errorf("output still contains %q:\n%s", leak, out)
		}
	}
	res := m["results"].([]any)[0].(map[string]any)
	metrics := res["metrics"].(map[string]any)
	for k, want := range map[string]string{
		"bandwidthGbps": "97.51", "gpuName": "NVIDIA GB10", "pciAddresses": "000f:01:00.0",
	} {
		if metrics[k] != want {
			t.Errorf("metric %s = %v, want %q untouched", k, metrics[k], want)
		}
	}
	for _, keep := range []string{
		`"version": "burnin.glimmer.ai/v1alpha1"`, `"kind": "ib-write-bw"`,
		`"commit": "1e44a67665baeb3ed9e9b779915c1a797c788e7a"`,
		`"digest": "sha256:3f7a0c9d1e2b4a6c8e0f1a3b5c7d9e1f3a5b7c9d1e3f5a7b9c1d3e5f7a9b1c3d"`,
		`"eccErrors"`, `"passed": 1`,
	} {
		if !strings.Contains(out, keep) {
			t.Errorf("output lost %s:\n%s", keep, out)
		}
	}
	if got, _ := s.ResidualJSON([]byte(out)); len(got) != 0 {
		t.Errorf("residual scan found %v", got)
	}
}

// Within one output a name maps to one pseudonym everywhere — the run name,
// the fingerprint key, the node list and the free-text message — so the
// result still reads coherently.
func TestOneNameMapsToOnePseudonymEverywhere(t *testing.T) {
	s := mustNew(t, Options{Hosts: []string{"spark-043a", "spark-85a9"}})
	_, m := sanitizeEnvelope(t, s)
	name := m["run"].(map[string]any)["name"].(string)
	if !strings.HasPrefix(name, "host-") {
		t.Fatalf("run name = %q", name)
	}
	if _, ok := m["fingerprint"].(map[string]any)[name]; !ok {
		t.Errorf("fingerprint is not keyed by the run name's pseudonym %q: %v", name, m["fingerprint"])
	}
	nodes := m["results"].([]any)[0].(map[string]any)["nodes"].([]any)
	if nodes[0] != name {
		t.Errorf("nodes[0] = %v, want %q", nodes[0], name)
	}
}

// Two outputs of the same input share no pseudonym: a fresh salt per
// invocation is what stops two shared results being joined on node identity.
func TestTwoOutputsAreUnlinkable(t *testing.T) {
	a := mustNew(t, Options{Hosts: []string{"spark-043a"}, Salt: bytes.Repeat([]byte{1}, 32)})
	b := mustNew(t, Options{Hosts: []string{"spark-043a"}, Salt: bytes.Repeat([]byte{2}, 32)})
	x, _ := a.Text("spark-043a")
	y, _ := b.Text("spark-043a")
	if x == y {
		t.Errorf("both salts produced %q", x)
	}
	fresh, _ := New(Options{})
	if bytes.Equal(fresh.salt, make([]byte, 32)) {
		t.Error("New drew an all-zero salt")
	}
}

func TestInventoryMatchesWholeTokensOnly(t *testing.T) {
	s := mustNew(t, Options{Hosts: []string{"spark-1"}})
	for in, changed := range map[string]bool{
		"spark-1":          true,
		"on spark-1.":      true,
		"burnin-spark-1-7": true,
		"spark-10":         false,
		"spark-1x":         false,
		"spark-1_a":        false,
	} {
		out, _ := s.Text(in)
		if (out != in) != changed {
			t.Errorf("%q -> %q, changed=%v want %v", in, out, out != in, changed)
		}
	}
}

func TestAddressesMapIntoDocumentationRangesWithoutCollisions(t *testing.T) {
	s := mustNew(t, Options{})
	out, _ := s.Text("a 10.0.0.1 b 10.0.0.2 c 10.0.0.1 d 8.8.8.8 e 127.0.0.1 f 192.0.2.44 g ::1 h fd00::1")
	for _, want := range []string{"a 192.0.2.1 ", "b 192.0.2.2 ", "c 192.0.2.1 ", "d 192.0.2.3 ", "e 127.0.0.1 ", "f 192.0.2.44 ", "g ::1 ", "h 2001:db8::1"} {
		if !strings.Contains(out, want) {
			t.Errorf("missing %q in %q", want, out)
		}
	}
}

func TestSecretsAreMaskedAndKeyTriggeredValuesReplaced(t *testing.T) {
	s := mustNew(t, Options{})
	out, err := s.JSON([]byte(`{"token":"abc","serial":"S4XNNA0T123456","env":{"PATH":"/usr/bin","HOME":"/home/baldwin"},
		"note":"Authorization: Bearer sk-live-4f9a8b7c6d5e4f3a2b1c","log":"key=Zx9Qm2Lk8Pw3Rt7Yv1Bn6Hc4Jd0Fs5Ge"}`))
	if err != nil {
		t.Fatal(err)
	}
	o := string(out)
	for _, leak := range []string{"abc", "S4XNNA0T123456", "baldwin", "sk-live", "Zx9Qm2Lk8Pw3"} {
		if strings.Contains(o, leak) {
			t.Errorf("leaked %q:\n%s", leak, o)
		}
	}
	for _, keep := range []string{`"PATH": "/usr/bin"`, `"HOME": "<REDACTED-ENV>"`, "Authorization: <REDACTED-SECRET>", `"serial": "serial-`} {
		if !strings.Contains(o, keep) {
			t.Errorf("missing %s:\n%s", keep, o)
		}
	}
}

func TestHomePathUserIsReplacedAndAllowedDomainsKept(t *testing.T) {
	s := mustNew(t, Options{})
	out, _ := s.Text("/home/baldwin/results pulled ghcr.io/baldwinspc/glimmer-burnin-host-health:v0.7.2 from node7.lab.example")
	if strings.Contains(out, "/home/baldwin/") || strings.Contains(out, "node7.lab.example") {
		t.Errorf("identity survived: %q", out)
	}
	if !strings.Contains(out, "ghcr.io/baldwinspc/glimmer-burnin-host-health:v0.7.2") {
		t.Errorf("a public registry reference was rewritten: %q", out)
	}
}

func TestPathsRewriteOnlyInventoryNames(t *testing.T) {
	s := mustNew(t, Options{Hosts: []string{"spark-043a"}})
	p, _ := s.Path("raw/spark-043a/run.json")
	if strings.Contains(p, "spark-043a") || !strings.HasSuffix(p, "/run.json") || !strings.HasPrefix(p, "raw/host-") {
		t.Errorf("Path = %q", p)
	}
}

func TestResidualCatchesWhatTheReplacerMissed(t *testing.T) {
	s := mustNew(t, Options{})
	if got := s.Residual("peer 10.1.2.3"); len(got) == 0 {
		t.Error("residual missed a private address")
	}
	if got := s.Residual("peer host-0123456789ab at 192.0.2.1"); len(got) != 0 {
		t.Errorf("residual flagged a pseudonym or a documentation address: %v", got)
	}
}

func TestKeysThatCollideAfterSanitisingAreRefused(t *testing.T) {
	s := mustNew(t, Options{Hosts: []string{"spark-a"}})
	if _, err := s.JSON([]byte(`{"x":{"spark-a":"1","SPARK-A":"2"}}`)); err == nil {
		t.Error("two keys that sanitise to one were merged silently")
	}
}

// #540: a serial-derived identity metric is pseudonymised; no content rule
// would recognise a 16-hex digest, and left alone it links every shared result.
func TestASerialDigestMetricIsPseudonymised(t *testing.T) {
	s := mustNew(t, Options{})
	out, err := s.JSON([]byte(`{"results":[{"metrics":{"nvmeSerialDigests":"aaaa1111bbbb2222","nvmeCount":"1"}}]}`))
	if err != nil {
		t.Fatal(err)
	}
	if strings.Contains(string(out), "aaaa1111bbbb2222") || !strings.Contains(string(out), `"nvmeSerialDigests": "serial-`) {
		t.Errorf("digest not pseudonymised:\n%s", out)
	}
	if !strings.Contains(string(out), `"nvmeCount": "1"`) {
		t.Errorf("a number was touched:\n%s", out)
	}
}
