package main

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// sealedRun builds the shape a bare-metal run writes, with the identity a real
// one carries: the node name in the run record, the envelope's run name,
// fingerprint key and node list, and a peer address and GID in raw output.
func sealedRun(t *testing.T) string {
	t.Helper()
	dir := t.TempDir()
	files := map[string]string{
		"run.json": `{"node":"spark-043a","phase":"Passed"}` + "\n",
		"envelopes/001-RunPhaseChanged.json": `{"version":"burnin.glimmer.ai/v1alpha1",` +
			`"run":{"namespace":"local","name":"spark-043a","uid":"6f0e6c62-0e6e-4d9a-9a5b-2c1f9a6d1a77"},` +
			`"phase":"Passed","fingerprint":{"spark-043a":"kernel=6.11.0 arch=arm64"},` +
			`"results":[{"name":"fabric","kind":"ib-write-bw","phase":"Passed","nodes":["spark-043a","spark-85a9"],` +
			`"metrics":{"bandwidthGbps":"97.51"}}],"summary":{"passed":1}}` + "\n",
		"raw/fabric.log": "connecting to spark-85a9 at 10.252.161.209 gid fe80:0000:0000:0000:5ebb:f6ff:fe12:3456\nbandwidth_gbps=97.51\n",
	}
	for rel, body := range files {
		p := filepath.Join(dir, filepath.FromSlash(rel))
		if err := os.MkdirAll(filepath.Dir(p), 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(p, []byte(body), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	if err := sealDir(dir); err != nil {
		t.Fatal(err)
	}
	return dir
}

func readTree(t *testing.T, dir string) string {
	t.Helper()
	var b strings.Builder
	files, err := listRegular(dir)
	if err != nil {
		t.Fatal(err)
	}
	for _, rel := range files {
		data, _ := os.ReadFile(filepath.Join(dir, rel))
		b.WriteString(rel + "\n" + string(data))
	}
	return b.String()
}

func TestSanitizeWritesASealedTerminalCopyWithNoIdentity(t *testing.T) {
	in := sealedRun(t)
	out := filepath.Join(t.TempDir(), "shareable")
	if _, err := sanitizeDir(in, out, nil, nil); err != nil {
		t.Fatalf("sanitizeDir: %v", err)
	}
	tree := readTree(t, out)
	for _, leak := range []string{"spark-043a", "spark-85a9", "10.252.161.209", "6f0e6c62", "fe80:0000"} {
		if strings.Contains(tree, leak) {
			t.Errorf("sanitised output contains %q:\n%s", leak, tree)
		}
	}
	for _, keep := range []string{`"bandwidthGbps": "97.51"`, "bandwidth_gbps=97.51", `"kind": "ib-write-bw"`} {
		if !strings.Contains(tree, keep) {
			t.Errorf("sanitised output lost %s:\n%s", keep, tree)
		}
	}
	if p, _ := verifyDir(out); len(p) != 0 {
		t.Errorf("the sanitised copy is not sealed: %v", p)
	}
	if !isSanitized(out) {
		t.Error("the copy carries no sanitised marker")
	}
	if _, err := sanitizeDir(out, out+"2", nil, nil); err == nil {
		t.Error("a sanitised result was sanitised again; it is terminal")
	}
}

func TestSanitizeRefusesATamperedInput(t *testing.T) {
	in := sealedRun(t)
	if err := os.WriteFile(filepath.Join(in, "run.json"), []byte("{}"), 0o644); err != nil {
		t.Fatal(err)
	}
	out := filepath.Join(t.TempDir(), "x")
	if _, err := sanitizeDir(in, out, nil, nil); err == nil {
		t.Fatal("sanitised a directory that no longer matches its seal")
	}
	if _, err := os.Stat(out); err == nil {
		t.Error("output written despite the refusal")
	}
}

func TestSanitizeRefusesABinaryFileAndWritesNothing(t *testing.T) {
	in := sealedRun(t)
	if err := os.WriteFile(filepath.Join(in, "raw", "blob.bin"), []byte{0xff, 0xfe, 0x00, 0x81}, 0o644); err != nil {
		t.Fatal(err)
	}
	if err := sealDir(in); err != nil {
		t.Fatal(err)
	}
	out := filepath.Join(t.TempDir(), "x")
	if _, err := sanitizeDir(in, out, nil, nil); err == nil || !strings.Contains(err.Error(), "not text") {
		t.Fatalf("err = %v, want a not-text refusal", err)
	}
	if _, err := os.Stat(out); err == nil {
		t.Error("output written despite the refusal")
	}
	if _, err := os.Stat(out + ".sanitize-staging"); err == nil {
		t.Error("staging directory left behind")
	}
}

func TestSanitizeRefusesAnExistingOutput(t *testing.T) {
	in := sealedRun(t)
	if _, err := sanitizeDir(in, t.TempDir(), nil, nil); err == nil {
		t.Error("wrote into an existing directory")
	}
}
