package main

import (
	"os"
	"path/filepath"
	"testing"
)

func TestParseKmsgRecord(t *testing.T) {
	cases := []struct {
		name string
		rec  string
		want string
	}{
		{
			name: "structured record",
			rec:  "3,1234,567890,-;NVRM: Xid (PCI:0000:01:00): 13, pid=1234, Graphics SM Warp Exception\n",
			want: "NVRM: Xid (PCI:0000:01:00): 13, pid=1234, Graphics SM Warp Exception",
		},
		{
			name: "continuation lines are dropped",
			rec:  "6,99,1,-;pcieport 0000:00:01.0: AER: Corrected error received\n SUBSYSTEM=pci\n DEVICE=+pci:0000:00:01.0\n",
			want: "pcieport 0000:00:01.0: AER: Corrected error received",
		},
		{
			name: "header carrying a key=value dictionary",
			rec:  "6,100,2,-,caller=T123;something happened\n",
			want: "something happened",
		},
		{
			// A line that is not kmsg-shaped must still be searchable: dropping
			// it could hide an Xid.
			name: "unstructured line passes through",
			rec:  "NVRM: Xid (PCI:0000:01:00): 79\n",
			want: "NVRM: Xid (PCI:0000:01:00): 79",
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if got := parseKmsgRecord(tc.rec); got != tc.want {
				t.Errorf("parseKmsgRecord() = %q, want %q", got, tc.want)
			}
		})
	}
}

func TestCountMessages(t *testing.T) {
	msgs := []string{
		"NVRM: Xid (PCI:0000:01:00): 13, pid=1234, Graphics SM Warp Exception",
		"nvidia-nvlink: Nvlink Error", // not an Xid
		"NVRM: Xid (PCI:0000:02:00): 48, DBE",
		"mce: [Hardware Error]: Machine check events logged",
		"EDAC MC0: 1 UE uncorrectable error",
		"amdgpu: GPU reset begin!",
		"pcieport 0000:00:01.0: AER: Multiple Corrected error received",
		"systemd[1]: Started Session 3 of user root.",
	}
	xid, hw := countMessages(msgs)
	if xid != 2 {
		t.Errorf("xid count = %d, want 2", xid)
	}
	// machine check (counted once even though it also matches "hardware
	// error"), the EDAC line, the amdgpu reset, and the AER line.
	if hw != 4 {
		t.Errorf("hardware-error count = %d, want 4", hw)
	}
}

// A kernel-log probe must count only what arrived DURING the window; anything
// already in the log is pre-existing and reported separately.
func TestFileSourceDifferencesTheWindow(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "kern.log")
	mustWrite(t, path, "boot: hello\nNVRM: Xid (PCI:0000:01:00): 31, old\n")

	src, err := openFileLog(path)
	if err != nil {
		t.Fatalf("openFileLog: %v", err)
	}
	first, err := src.scan()
	if err != nil {
		t.Fatalf("scan: %v", err)
	}
	if xid, _ := countMessages(first); xid != 1 {
		t.Fatalf("baseline xid = %d, want 1", xid)
	}

	appendTo(t, path, "NVRM: Xid (PCI:0000:01:00): 63, new\nunrelated\n")
	second, err := src.scan()
	if err != nil {
		t.Fatalf("scan: %v", err)
	}
	if len(second) != 2 {
		t.Fatalf("second scan returned %d lines, want 2: %q", len(second), second)
	}
	if xid, _ := countMessages(second); xid != 1 {
		t.Errorf("window xid = %d, want 1", xid)
	}
}

// A partially written trailing line must not be consumed: the rest of it
// belongs to the next scan, and consuming it would split an Xid line in two and
// lose it entirely.
func TestFileSourceHoldsBackPartialLine(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "kern.log")
	mustWrite(t, path, "complete line\nNVRM: Xid (PCI:0000:01:0")

	src, _ := openFileLog(path)
	first, _ := src.scan()
	if len(first) != 1 {
		t.Fatalf("first scan = %q, want only the complete line", first)
	}

	appendTo(t, path, "0): 48, split across writes\n")
	second, _ := src.scan()
	if len(second) != 1 {
		t.Fatalf("second scan = %q, want the completed line", second)
	}
	if xid, _ := countMessages(second); xid != 1 {
		t.Errorf("xid = %d, want the re-joined line to count once", xid)
	}
}

func TestFileSourceHandlesRotation(t *testing.T) {
	t.Run("renamed away by logrotate", func(t *testing.T) {
		dir := t.TempDir()
		path := filepath.Join(dir, "kern.log")
		mustWrite(t, path, "one\ntwo\nthree\n")

		src, _ := openFileLog(path)
		if first, _ := src.scan(); len(first) != 3 {
			t.Fatalf("first scan = %d lines, want 3", len(first))
		}
		if err := os.Rename(path, path+".1"); err != nil {
			t.Fatal(err)
		}
		// The replacement is LONGER than the old offset, so only the changed
		// identity of the file reveals the rotation.
		mustWrite(t, path, "NVRM: Xid (PCI:0000:01:00): 79, first line after rotation\n")
		second, _ := src.scan()
		if xid, _ := countMessages(second); xid != 1 {
			t.Errorf("post-rotation xid = %d, want 1", xid)
		}
	})

	t.Run("truncated in place", func(t *testing.T) {
		dir := t.TempDir()
		path := filepath.Join(dir, "kern.log")
		mustWrite(t, path, "one\ntwo\nthree\nfour\nfive\n")

		src, _ := openFileLog(path)
		if _, err := src.scan(); err != nil {
			t.Fatal(err)
		}
		mustWrite(t, path, "NVRM: Xid: 79\n")
		second, _ := src.scan()
		if xid, _ := countMessages(second); xid != 1 {
			t.Errorf("post-truncation xid = %d, want 1", xid)
		}
	})
}

// The kmsg reader is exercised against a regular file: read() reports EOF
// instead of EAGAIN, which is the other loop-termination path.
func TestKmsgSourceReadsRecords(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "kmsg")
	mustWrite(t, path, "3,1,1,-;NVRM: Xid (PCI:0000:01:00): 48, DBE\n")

	src, err := openKmsg(path)
	if err != nil {
		t.Fatalf("openKmsg: %v", err)
	}
	defer src.close()

	msgs, err := src.scan()
	if err != nil {
		t.Fatalf("scan: %v", err)
	}
	if len(msgs) != 1 || msgs[0] != "NVRM: Xid (PCI:0000:01:00): 48, DBE" {
		t.Fatalf("scan = %q", msgs)
	}
	if xid, _ := countMessages(msgs); xid != 1 {
		t.Errorf("xid = %d, want 1", xid)
	}
}

// One read of a captured log returns many records at once. Keeping only the
// first would silently lose every Xid but one.
func TestKmsgSourceReadsEveryRecordInOneRead(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "kmsg")
	mustWrite(t, path,
		"6,1,1,-;systemd[1]: Started Session 3 of user root.\n"+
			"3,2,2,-;NVRM: Xid (PCI:0000:01:00): 48, DBE\n"+
			" SUBSYSTEM=pci\n"+
			"3,3,3,-;NVRM: Xid (PCI:0000:02:00): 79, GPU has fallen off the bus\n")

	src, err := openKmsg(path)
	if err != nil {
		t.Fatalf("openKmsg: %v", err)
	}
	defer src.close()

	msgs, err := src.scan()
	if err != nil {
		t.Fatalf("scan: %v", err)
	}
	if len(msgs) != 3 {
		t.Fatalf("scan returned %d messages, want 3 (the continuation line is not one): %q", len(msgs), msgs)
	}
	if xid, _ := countMessages(msgs); xid != 2 {
		t.Errorf("xid = %d, want 2", xid)
	}
}

func TestSplitKmsgChunkHoldsBackPartialRecord(t *testing.T) {
	msgs, remainder := splitKmsgChunk("3,1,1,-;NVRM: Xid: 13\n3,2,2,-;NVRM: Xid")
	if len(msgs) != 1 {
		t.Fatalf("messages = %q, want only the complete record", msgs)
	}
	if remainder != "3,2,2,-;NVRM: Xid" {
		t.Errorf("remainder = %q, want the partial record carried over", remainder)
	}
	// The partial record completes on the next read and is counted once.
	msgs2, remainder2 := splitKmsgChunk(remainder + ": 48, DBE\n")
	if len(msgs2) != 1 || remainder2 != "" {
		t.Fatalf("second chunk = %q, remainder %q", msgs2, remainder2)
	}
	if xid, _ := countMessages(msgs2); xid != 1 {
		t.Errorf("xid = %d, want 1", xid)
	}
}

func TestKernelLogProbeFallsBackAndReportsSource(t *testing.T) {
	dir := t.TempDir()
	logPath := filepath.Join(dir, "kern.log")
	mustWrite(t, logPath, "NVRM: Xid (PCI:0000:01:00): 13, before\n")

	cfg := config{
		kmsgPath:     filepath.Join(dir, "does-not-exist"),
		kernLogPaths: []string{filepath.Join(dir, "also-missing"), logPath},
	}
	p := newKernelLogProbe(cfg)
	if p.source != "kernlog" {
		t.Fatalf("source = %q, want kernlog", p.source)
	}
	p.baseline()
	appendTo(t, logPath, "NVRM: Xid (PCI:0000:01:00): 63, during\n")
	p.collect()

	out := newEmitter()
	p.emit(out)
	if got, _ := out.get(keyXidCount); got != "1" {
		t.Errorf("%s = %q, want 1", keyXidCount, got)
	}
	if got, _ := out.get("xid_preexisting"); got != "1" {
		t.Errorf("xid_preexisting = %q, want 1", got)
	}
}

// With no readable source at all, the probe must emit NO xid_count. A zero here
// would be a measurement nobody took, and would pass a threshold the node never
// earned.
func TestKernelLogProbeEmitsNothingWithoutASource(t *testing.T) {
	dir := t.TempDir()
	cfg := config{kmsgPath: filepath.Join(dir, "nope"), kernLogPaths: []string{filepath.Join(dir, "nope2")}}
	p := newKernelLogProbe(cfg)
	p.baseline()
	p.collect()

	out := newEmitter()
	p.emit(out)
	if src, _ := out.get("xid_source"); src != "none" {
		t.Errorf("xid_source = %q, want none", src)
	}
	if _, ok := out.get(keyXidCount); ok {
		t.Error("xid_count must be omitted when no kernel log could be read")
	}
}

func mustWrite(t *testing.T, path, content string) {
	t.Helper()
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, []byte(content), 0o644); err != nil {
		t.Fatal(err)
	}
}

func appendTo(t *testing.T, path, content string) {
	t.Helper()
	f, err := os.OpenFile(path, os.O_APPEND|os.O_WRONLY, 0o644)
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = f.Close() }()
	if _, err := f.WriteString(content); err != nil {
		t.Fatal(err)
	}
}
