package main

import (
	"errors"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
)

// collect adapts the streaming scan back into a slice so these tests can keep
// asserting on the messages themselves.
//
// It is deliberately test-only. The production path (kernelLogProbe.count)
// never materialises the scan, because a slice of every record on the host is
// how this runner could run itself out of memory — see the kernelLog interface
// comment and issue #112.
func collect(src kernelLog) ([]string, error) {
	var out []string
	err := src.scan(func(m string) { out = append(out, m) })
	return out, err
}

// countMessages is the batch spelling of messageCounter, kept for the table
// test below. The production counter visits; this proves the same rule.
func countMessages(msgs []string) (xid, hw int64) {
	var c messageCounter
	for _, m := range msgs {
		c.visit(m)
	}
	return c.xid, c.hw
}

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
	first, err := collect(src)
	if err != nil {
		t.Fatalf("scan: %v", err)
	}
	if xid, _ := countMessages(first); xid != 1 {
		t.Fatalf("baseline xid = %d, want 1", xid)
	}

	appendTo(t, path, "NVRM: Xid (PCI:0000:01:00): 63, new\nunrelated\n")
	second, err := collect(src)
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
	first, _ := collect(src)
	if len(first) != 1 {
		t.Fatalf("first scan = %q, want only the complete line", first)
	}

	appendTo(t, path, "0): 48, split across writes\n")
	second, _ := collect(src)
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
		if first, _ := collect(src); len(first) != 3 {
			t.Fatalf("first scan = %d lines, want 3", len(first))
		}
		if err := os.Rename(path, path+".1"); err != nil {
			t.Fatal(err)
		}
		// The replacement is LONGER than the old offset, so only the changed
		// identity of the file reveals the rotation.
		mustWrite(t, path, "NVRM: Xid (PCI:0000:01:00): 79, first line after rotation\n")
		second, _ := collect(src)
		if xid, _ := countMessages(second); xid != 1 {
			t.Errorf("post-rotation xid = %d, want 1", xid)
		}
	})

	t.Run("truncated in place", func(t *testing.T) {
		dir := t.TempDir()
		path := filepath.Join(dir, "kern.log")
		mustWrite(t, path, "one\ntwo\nthree\nfour\nfive\n")

		src, _ := openFileLog(path)
		if _, err := collect(src); err != nil {
			t.Fatal(err)
		}
		mustWrite(t, path, "NVRM: Xid: 79\n")
		second, _ := collect(src)
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

	msgs, err := collect(src)
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

	msgs, err := collect(src)
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

func TestVisitKmsgChunkHoldsBackPartialRecord(t *testing.T) {
	visited := func(chunk string) ([]string, string) {
		var out []string
		rest := visitKmsgChunk(chunk, func(m string) { out = append(out, m) })
		return out, rest
	}

	msgs, remainder := visited("3,1,1,-;NVRM: Xid: 13\n3,2,2,-;NVRM: Xid")
	if len(msgs) != 1 {
		t.Fatalf("messages = %q, want only the complete record", msgs)
	}
	if remainder != "3,2,2,-;NVRM: Xid" {
		t.Errorf("remainder = %q, want the partial record carried over", remainder)
	}
	// The partial record completes on the next read and is counted once.
	msgs2, remainder2 := visited(remainder + ": 48, DBE\n")
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

// A scan must cost the same whatever the host's uptime — issue #112.
//
// This is the regression test for the mechanism, not for the symptom. Both
// sources used to accumulate every record into a []string for a caller that
// wanted two integers, so peak heap tracked the SIZE OF THE LOG: the kernel ring
// buffer on the kmsg path, and here the whole unread tail of /var/log/syslog,
// which nothing bounds. An allocation failure inside that is a Go runtime fatal
// error — unrecoverable, and it exits 2, the undeclared skip code, on the one
// runner in this repository that claims never to skip.
//
// The bound is deliberately loose (a fifth of the log). It is not a memory
// budget; it is the difference between "retains nothing" and "retains all of
// it", and only a change back to accumulating can cross it.
func TestFileSourceScanRetainsNothing(t *testing.T) {
	if testing.Short() {
		t.Skip("allocates a multi-megabyte log")
	}
	const (
		records   = 300000
		sampleGap = 16384
	)
	record := "Jun  1 00:00:00 node kernel: [12345.678901] systemd[1]: Started Session of user root, nothing to see here\n"

	dir := t.TempDir()
	path := filepath.Join(dir, "syslog")
	var log []byte
	for i := 0; i < records; i++ {
		log = append(log, record...)
	}
	logBytes := uint64(len(log))
	mustWrite(t, path, string(log))
	// `log` is deliberately not nilled here. It is never read again, so it is
	// already dead by liveness analysis and its backing array is collectable —
	// an explicit `log = nil` measured nothing and only looked like it did.

	src, err := openFileLog(path)
	if err != nil {
		t.Fatalf("openFileLog: %v", err)
	}
	defer src.close()

	runtime.GC()
	var base runtime.MemStats
	runtime.ReadMemStats(&base)

	var (
		seen  int
		peak  uint64
		stats runtime.MemStats
	)
	// Sampled INSIDE the scan: the retained set is what matters while the scan
	// is in flight, and by the time it returns the accumulating version would
	// have handed its slice back and be collectable again.
	err = src.scan(func(string) {
		seen++
		if seen%sampleGap != 0 {
			return
		}
		runtime.ReadMemStats(&stats)
		if grown := stats.HeapAlloc - min(stats.HeapAlloc, base.HeapAlloc); grown > peak {
			peak = grown
		}
	})
	if err != nil {
		t.Fatalf("scan: %v", err)
	}
	if seen != records {
		t.Fatalf("visited %d records, want %d — the scan is not reading the whole log", seen, records)
	}

	limit := logBytes / 5
	if peak > limit {
		t.Errorf("peak heap grew %d bytes while streaming a %d byte log (limit %d) — the scan is retaining records",
			peak, logBytes, limit)
	}
	t.Logf("streamed %d bytes with a peak heap growth of %d bytes", logBytes, peak)
}

// A scan that FAILED is not a sample — issue #112.
//
// Both of these were reachable on a shipped runner and each produced one of the
// two verdicts this project forbids most: a Fail against healthy hardware, and a
// Pass certified by a number nobody measured. They are the same root cause seen
// from two sides — count() returned (0, 0) for a failed scan, indistinguishably
// from a clean empty one, and emit() published that zero as a measurement.
//
// The fix is the rule already stated in this runner's package comment: "we could
// not look" is not a measurement, so the counter is OMITTED and fails closed
// upstream. It is deliberately NOT the `n/a` sentinel — that declares the
// hardware cannot produce the value, which is a claim about the part, and this
// is a claim about the probe.

func TestBaselineScanFailureDoesNotChargeHistoryToTheWindow(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "kern.log")
	mustWrite(t, path,
		"boot\nNVRM: Xid (PCI:0000:01:00): 48, months ago\nNVRM: Xid (PCI:0000:02:00): 63, also old\n")

	p := newKernelLogProbe(config{kernLogPaths: []string{path}})
	if p.source != "kernlog" {
		t.Fatalf("source = %q, want kernlog", p.source)
	}

	// The baseline scan's os.Open fails and then the file comes back. On a real
	// node that is a transient EMFILE under the fd pressure #112 was reported
	// under, or logrotate renaming the path in exactly that instant.
	away := path + ".1"
	if err := os.Rename(path, away); err != nil {
		t.Fatal(err)
	}
	p.baseline()
	if err := os.Rename(away, path); err != nil {
		t.Fatal(err)
	}

	// Nothing whatsoever happens during the observation window.
	p.collect()

	out := newEmitter()
	p.emit(out)

	// The window count is not a measurement: the baseline it would be measured
	// against was never taken, and the source's own read position never moved,
	// so the "window" would be the whole file.
	if v, ok := out.get(keyXidCount); ok {
		t.Errorf("%s=%q was emitted after the baseline scan failed — that is the node's whole "+
			"Xid history reported as events during the burn-in", keyXidCount, v)
	}
	if v, ok := out.get("xid_preexisting"); ok {
		t.Errorf("xid_preexisting=%q was emitted from a scan that failed", v)
	}
	if code, verdict := decide(out); code == exitFail {
		t.Errorf("a healthy node was FAILED for historical Xids: exit %d %q", code, verdict)
	}
}

func TestWindowScanFailureOmitsTheCounterRatherThanZeroingIt(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "kern.log")
	mustWrite(t, path, "boot\n")

	p := newKernelLogProbe(config{kernLogPaths: []string{path}})
	p.baseline()

	// A real Xid lands DURING the burn-in, and then the log rotates away before
	// the closing scan can read it.
	appendTo(t, path, "NVRM: Xid (PCI:0000:01:00): 48, double-bit ECC during the soak\n")
	if err := os.Remove(path); err != nil {
		t.Fatal(err)
	}
	p.collect()

	out := newEmitter()
	p.emit(out)

	if v, ok := out.get(keyXidCount); ok {
		t.Errorf("%s=%q was emitted for a window that was never read — a zero nobody measured "+
			"satisfies an `xidEvents Equal 0` gate and certifies a node that logged an Xid",
			keyXidCount, v)
	}
	// The evidence that something went wrong still has to be on the record.
	if _, ok := out.get("xid_log_dropped"); !ok {
		t.Error("xid_log_dropped was not emitted, so nothing records that the scan failed")
	}
}

// A clean run is untouched: both samples were taken, so both are published.
func TestSuccessfulScansStillPublishTheCounters(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "kern.log")
	mustWrite(t, path, "NVRM: Xid (PCI:0000:01:00): 31, before the window\n")

	p := newKernelLogProbe(config{kernLogPaths: []string{path}})
	p.baseline()
	appendTo(t, path, "NVRM: Xid (PCI:0000:02:00): 48, during the window\n")
	p.collect()

	out := newEmitter()
	p.emit(out)
	if v, _ := out.get(keyXidCount); v != "1" {
		t.Errorf("%s = %q, want 1", keyXidCount, v)
	}
	if v, _ := out.get("xid_preexisting"); v != "1" {
		t.Errorf("xid_preexisting = %q, want 1", v)
	}
	if _, dropped := out.get("xid_log_dropped"); dropped {
		t.Error("a clean pair of scans must not report a drop")
	}
}

// Exhausting the per-scan read bound is a TRUNCATED scan, not a complete one.
//
// The bound exists so a source that never returns EAGAIN cannot spin forever,
// and falling out of it used to return nil — success — with the drain cut off
// partway. The baseline is then short, the source is positioned somewhere
// arbitrary in the log, and everything past the cut is counted as a window event
// on the next scan. Same wrong verdict as a failed baseline, reached by a
// different route and without even xid_log_dropped to show for it.
//
// The bound is lowered here because it cannot otherwise be reached: with a
// 16 KiB buffer a regular file hits EOF in a handful of reads, and 200000 reads
// of real /dev/kmsg is not a state a test can construct. A guard nobody can
// execute is a guard nobody has checked.
func TestExhaustingTheReadBoundIsReportedAsAFailedScan(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "kmsg")

	var log []byte
	for i := 0; i < 64; i++ {
		log = append(log, "6,1,1,-;systemd[1]: a message\n"...)
	}
	mustWrite(t, path, string(log))

	src, err := openKmsg(path)
	if err != nil {
		t.Fatalf("openKmsg: %v", err)
	}
	defer src.close()
	src.maxReads = 1 // one read cannot drain the file

	seen := 0
	err = src.scan(func(string) { seen++ })
	if err == nil {
		t.Fatalf("a scan cut off at the read bound reported success after visiting %d records — "+
			"the caller cannot tell it from a complete drain", seen)
	}
	if seen == 0 {
		t.Error("the bounded scan visited nothing, so it is not exercising the truncation path")
	}

	// And the probe must treat it as an unmeasured sample rather than a zero.
	p := &kernelLogProbe{src: &boundedOnce{inner: src}, source: "kmsg", available: true}
	p.baseline()
	if p.baselineOK {
		t.Error("a truncated drain was recorded as a completed baseline sample")
	}
}

// boundedOnce hands the probe a source whose scan always reports truncation.
type boundedOnce struct{ inner *kmsgSource }

func (b *boundedOnce) scan(func(string)) error { return errTruncated }
func (b *boundedOnce) close()                  { b.inner.close() }

var errTruncated = errors.New("truncated")

// xid_source=none must say WHY — issue #134.
//
// On a real GB10 with /dev/kmsg mounted exactly as the CRD documents, the probe
// reported `xid_source=none` and nothing else. A profile gating xidEvents then
// fails closed — correctly, since nothing was measured — but with no way to tell
// a missing mount from a permission problem, and no hint that the cause is the
// image's own non-root uid meeting kernel.dmesg_restrict=1.
func TestKernelLogProbeSaysWhyItFoundNoSource(t *testing.T) {
	dir := t.TempDir()

	t.Run("nothing is mounted", func(t *testing.T) {
		p := newKernelLogProbe(config{
			kmsgPath:     filepath.Join(dir, "absent-kmsg"),
			kernLogPaths: []string{filepath.Join(dir, "absent-kern.log")},
		})
		if p.source != "none" {
			t.Fatalf("source = %q", p.source)
		}
		out := newEmitter()
		p.emit(out)

		detail, ok := out.get("xid_source_detail")
		if !ok {
			t.Fatal("xid_source=none was emitted with no explanation at all — the operator " +
				"cannot tell a missing mount from a permission problem")
		}
		if !strings.Contains(detail, "hostPaths") {
			t.Errorf("a missing path should point at the field that mounts it: %q", detail)
		}
	})

	t.Run("present but unreadable", func(t *testing.T) {
		if os.Geteuid() == 0 {
			t.Skip("running as root: mode bits do not deny this process")
		}
		locked := filepath.Join(dir, "locked-kern.log")
		mustWrite(t, locked, "boot\n")
		if err := os.Chmod(locked, 0o000); err != nil {
			t.Fatal(err)
		}

		p := newKernelLogProbe(config{kernLogPaths: []string{locked}})
		out := newEmitter()
		p.emit(out)
		detail, _ := out.get("xid_source_detail")

		if !strings.Contains(detail, "not readable") {
			t.Errorf("a permission failure must say so: %q", detail)
		}
		// The remedy nobody can guess from an errno: privileged is not enough,
		// because the effective capability set is dropped for a non-root uid.
		if !strings.Contains(detail, "runAsUser") {
			t.Errorf("the explanation does not name the field that fixes it: %q", detail)
		}
		if !strings.Contains(detail, "dmesg_restrict") {
			t.Errorf("the explanation does not name the host setting that causes it: %q", detail)
		}
	})

	t.Run("a working source says nothing extra", func(t *testing.T) {
		good := filepath.Join(dir, "kern.log")
		mustWrite(t, good, "boot\n")
		p := newKernelLogProbe(config{kernLogPaths: []string{good}})
		out := newEmitter()
		p.baseline()
		p.collect()
		p.emit(out)
		if _, ok := out.get("xid_source_detail"); ok {
			t.Error("a probe that found a source must not emit a failure explanation")
		}
	})
}
