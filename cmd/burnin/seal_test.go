package main

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func sealedDir(t *testing.T) string {
	t.Helper()
	dir := t.TempDir()
	for rel, body := range map[string]string{
		"run.json":                           `{"node":"spark-a"}`,
		"envelopes/001-RunPhaseChanged.json": `{"phase":"Passed"}`,
		"raw/fp4-compute-smoke.log":          "throughput_tflops=12\n",
	} {
		p := filepath.Join(dir, filepath.FromSlash(rel))
		if err := os.MkdirAll(filepath.Dir(p), 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(p, []byte(body), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	if err := sealDir(dir); err != nil {
		t.Fatalf("sealDir: %v", err)
	}
	return dir
}

func mustVerify(t *testing.T, dir string) []string {
	t.Helper()
	p, err := verifyDir(dir)
	if err != nil {
		t.Fatalf("verifyDir: %v", err)
	}
	return p
}

func TestASealedDirectoryVerifies(t *testing.T) {
	dir := sealedDir(t)
	if p := mustVerify(t, dir); len(p) != 0 {
		t.Fatalf("a freshly sealed directory failed: %v", p)
	}
	b, _ := os.ReadFile(filepath.Join(dir, sumsFile))
	if n := strings.Count(string(b), "\n"); n != 3 {
		t.Errorf("SHA256SUMS lists %d files, want 3 (it must not list itself):\n%s", n, b)
	}
}

func TestOneChangedByteFails(t *testing.T) {
	dir := sealedDir(t)
	p := filepath.Join(dir, "raw", "fp4-compute-smoke.log")
	if err := os.WriteFile(p, []byte("throughput_tflops=13\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	if got := mustVerify(t, dir); len(got) != 1 || !strings.Contains(got[0], "has changed") {
		t.Errorf("got %v, want one 'has changed' problem", got)
	}
}

func TestAnAddedFileFails(t *testing.T) {
	dir := sealedDir(t)
	if err := os.WriteFile(filepath.Join(dir, "envelopes", "002-extra.json"), []byte("{}"), 0o644); err != nil {
		t.Fatal(err)
	}
	if got := mustVerify(t, dir); len(got) != 1 || !strings.Contains(got[0], "not written by the run") {
		t.Errorf("got %v, want one foreign-file problem", got)
	}
}

func TestARemovedFileFails(t *testing.T) {
	dir := sealedDir(t)
	if err := os.Remove(filepath.Join(dir, "run.json")); err != nil {
		t.Fatal(err)
	}
	if got := mustVerify(t, dir); len(got) != 1 || !strings.Contains(got[0], "missing") {
		t.Errorf("got %v, want one missing-file problem", got)
	}
}

// A symlink is refused at seal time and reported at verify time: copying a
// sealed directory with a link-following tool would otherwise carry whatever
// the link points at under a checksum that verified clean.
func TestASymlinkIsRefusedAndReported(t *testing.T) {
	dir := t.TempDir()
	if err := os.Symlink("/etc/hosts", filepath.Join(dir, "leak")); err != nil {
		t.Fatal(err)
	}
	if err := sealDir(dir); err == nil {
		t.Error("sealDir sealed a directory holding a symlink")
	}

	sealed := sealedDir(t)
	if err := os.Symlink("/etc/hosts", filepath.Join(sealed, "leak")); err != nil {
		t.Fatal(err)
	}
	if got := mustVerify(t, sealed); len(got) == 0 {
		t.Error("verify passed a directory holding a symlink")
	}
}

func TestAListedFileReachedThroughASymlinkedDirectoryIsRefused(t *testing.T) {
	dir := sealedDir(t)
	real := filepath.Join(t.TempDir(), "raw")
	if err := os.Rename(filepath.Join(dir, "raw"), real); err != nil {
		t.Fatal(err)
	}
	if err := os.Symlink(real, filepath.Join(dir, "raw")); err != nil {
		t.Fatal(err)
	}
	got := mustVerify(t, dir)
	if len(got) == 0 || !strings.Contains(strings.Join(got, "\n"), "not a directory") {
		t.Errorf("got %v, want the symlinked directory refused", got)
	}
}

func TestAnUnsealedDirectoryIsReportedNotVerified(t *testing.T) {
	got := mustVerify(t, t.TempDir())
	if len(got) != 1 || !strings.Contains(got[0], "not sealed") {
		t.Errorf("got %v", got)
	}
}

func TestSumsNamingAPathOutsideTheDirectoryIsRefused(t *testing.T) {
	dir := sealedDir(t)
	f, err := os.OpenFile(filepath.Join(dir, sumsFile), os.O_APPEND|os.O_WRONLY, 0)
	if err != nil {
		t.Fatal(err)
	}
	f.WriteString(strings.Repeat("0", 64) + "  ../outside\n")
	f.Close()
	got := strings.Join(mustVerify(t, dir), "\n")
	if !strings.Contains(got, "outside the directory") {
		t.Errorf("got %q", got)
	}
}

// Resealing after merge adds its envelope is what keeps merge's own output
// from reading as foreign.
func TestResealingCoversAFileAddedAfterTheFirstSeal(t *testing.T) {
	dir := sealedDir(t)
	if err := os.WriteFile(filepath.Join(dir, "envelopes", "merged.json"), []byte("{}"), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := sealDir(dir); err != nil {
		t.Fatal(err)
	}
	if got := mustVerify(t, dir); len(got) != 0 {
		t.Errorf("resealed directory failed: %v", got)
	}
}
