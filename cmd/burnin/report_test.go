package main

import (
	"bytes"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/baldwinSPC/glimmer-burnin/pkg/report"
)

const envelopeJSON = `{
  "version": "burnin.glimmer.ai/v1alpha1",
  "deliveryId": "abc123",
  "reason": "RunPhaseChanged",
  "sentAt": "2026-08-06T12:00:00Z",
  "phase": "Failed",
  "run": {"namespace":"burnin","name":"acceptance","uid":"uid-1","profile":"node-acceptance"},
  "summary": {"passed":1,"failed":1,"errored":0,"skipped":0},
  "results": [
    {"name":"compute-smoke","kind":"compute-smoke","scope":"Node","phase":"Passed","nodes":["n1"],
     "metrics":{"throughputTflops":"101.99"}},
    {"name":"ib-write-bw","kind":"ib-write-bw","scope":"Pair","phase":"Failed","nodes":["n1","n2"],
     "violations":[{"index":0,"metric":"bandwidthGbps","cause":"Measurement","kind":"Unsatisfied","reason":"42.1 < 89"}]}
  ]
}`

func writeEnvelope(t *testing.T) string {
	t.Helper()
	dir := t.TempDir()
	path := filepath.Join(dir, "envelope.json")
	if err := os.WriteFile(path, []byte(envelopeJSON), 0o644); err != nil {
		t.Fatalf("writing fixture: %v", err)
	}
	return path
}

func TestEveryFormatRendersTheSameRun(t *testing.T) {
	// Four formats, one record. The point of the command is that a user can ask
	// for any of them and get a view of the same assembled input rather than
	// four independent reconstructions.
	src := writeEnvelope(t)

	for _, format := range formatNames() {
		t.Run(format, func(t *testing.T) {
			out := t.TempDir()
			if err := runReport([]string{"--from-file", src, "-o", format, "--out", out}); err != nil {
				t.Fatalf("report -o %s: %v", format, err)
			}
			entries, err := os.ReadDir(out)
			if err != nil || len(entries) == 0 {
				t.Fatalf("no documents written for %s (err %v)", format, err)
			}
			for _, e := range entries {
				b, err := os.ReadFile(filepath.Join(out, e.Name()))
				if err != nil {
					t.Fatalf("reading %s: %v", e.Name(), err)
				}
				if len(b) == 0 {
					t.Errorf("%s wrote an empty document", format)
				}
				// Every format must name the run it describes; a document that
				// cannot be traced back is not evidence.
				if !bytes.Contains(b, []byte("acceptance")) && !bytes.Contains(b, []byte("uid-1")) {
					t.Errorf("%s/%s does not identify the run", format, e.Name())
				}
			}
		})
	}
}

func TestAnUnknownFormatListsTheOnesThatExist(t *testing.T) {
	// A user who mistypes should be told the four available, not merely that
	// theirs is wrong.
	err := runReport([]string{"--from-file", writeEnvelope(t), "-o", "pdf"})
	if err == nil {
		t.Fatal("an unknown format was accepted")
	}
	for _, name := range formatNames() {
		if !strings.Contains(err.Error(), name) {
			t.Errorf("the error does not mention the available format %q: %v", name, err)
		}
	}
}

func TestExactlyOneInputIsRequired(t *testing.T) {
	src := writeEnvelope(t)

	if err := runReport([]string{"-o", "html"}); err == nil {
		t.Error("no input was accepted")
	} else if !strings.Contains(err.Error(), "--from-file") {
		t.Errorf("the error should say what to pass, got: %v", err)
	}

	// Two inputs is refused rather than silently prioritised: a user passing
	// both has a mistaken belief about which is being read, and picking a
	// winner leaves them holding a report about the wrong run.
	err := runReport([]string{"--from-file", src, "--run", "ns/name", "-o", "html"})
	if err == nil {
		t.Fatal("two inputs were accepted")
	}
	if !strings.Contains(err.Error(), "exactly one") {
		t.Errorf("the error should name the rule, got: %v", err)
	}
}

func TestARunReferenceMustBeNamespaced(t *testing.T) {
	err := runReport([]string{"--run", "just-a-name", "-o", "html"})
	if err == nil {
		t.Fatal("an unqualified run reference was accepted")
	}
	if !strings.Contains(err.Error(), "namespace") {
		t.Errorf("the error should say what the form is, got: %v", err)
	}
}

func TestAMultiDocumentFormatRefusesStdoutRatherThanConcatenating(t *testing.T) {
	// The NVVS renderer emits one document per node because that schema is
	// single-host. Concatenating them would produce a stream that is not valid
	// in its own format, which is worse than refusing.
	err := runReport([]string{"--from-file", writeEnvelope(t), "-o", "nvvs-json", "--out", "-"})
	if err == nil {
		t.Fatal("multiple documents were concatenated onto stdout")
	}
	if !strings.Contains(err.Error(), "--out") {
		t.Errorf("the error should say what to do instead, got: %v", err)
	}
}

func TestASingleDocumentFormatWritesToStdout(t *testing.T) {
	var buf bytes.Buffer
	in, err := loadInputForTest(writeEnvelope(t))
	if err != nil {
		t.Fatalf("loading: %v", err)
	}
	outs, err := renderers()["html"].Render(in)
	if err != nil {
		t.Fatalf("rendering: %v", err)
	}
	if err := write(outs, "-", &buf); err != nil {
		t.Fatalf("write to stdout: %v", err)
	}
	if !bytes.Contains(buf.Bytes(), []byte("<!DOCTYPE html>")) {
		t.Error("stdout did not receive the document")
	}
}

func loadInputForTest(path string) (report.Input, error) {
	return loadInput(reportFlags{fromFile: path, format: "html"})
}

func TestWriteRefusesAFilenameThatEscapesTheDirectory(t *testing.T) {
	// Renderers are documented to return a base name. This makes a renderer
	// that forgets unable to write outside the directory it was given.
	dir := t.TempDir()
	outs := []report.Output{{Filename: "../escaped.html", Data: []byte("x")}}
	if err := write(outs, dir, os.Stdout); err != nil {
		t.Fatalf("write: %v", err)
	}
	if _, err := os.Stat(filepath.Join(filepath.Dir(dir), "escaped.html")); err == nil {
		t.Fatal("a renderer's filename escaped the destination directory")
	}
	if _, err := os.Stat(filepath.Join(dir, "escaped.html")); err != nil {
		t.Errorf("the document was not written into the destination: %v", err)
	}
}

func TestAMalformedEnvelopeIsAnErrorNotAnEmptyReport(t *testing.T) {
	// A report assembled from "whatever parsed" has omissions nobody knows
	// about, and the file most likely to be malformed is the terminal delivery
	// carrying the verdict.
	dir := t.TempDir()
	path := filepath.Join(dir, "bad.json")
	if err := os.WriteFile(path, []byte(`{"version":"burnin.glimmer.ai/v1alpha1","deliveryId":`), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := runReport([]string{"--from-file", path, "-o", "html", "--out", t.TempDir()}); err == nil {
		t.Fatal("a malformed envelope produced a report instead of an error")
	}
}

func TestAResultsDirectoryIsRendered(t *testing.T) {
	dir := t.TempDir()
	if err := os.WriteFile(filepath.Join(dir, "001.json"), []byte(envelopeJSON), 0o644); err != nil {
		t.Fatal(err)
	}
	// A non-JSON sibling is ignored rather than treated as a delivery.
	if err := os.WriteFile(filepath.Join(dir, "notes.txt"), []byte("scratch"), 0o644); err != nil {
		t.Fatal(err)
	}

	out := t.TempDir()
	if err := runReport([]string{"--results-dir", dir, "-o", "markdown", "--out", out}); err != nil {
		t.Fatalf("report --results-dir: %v", err)
	}
	entries, _ := os.ReadDir(out)
	if len(entries) == 0 {
		t.Fatal("no document written from a results directory")
	}
}
