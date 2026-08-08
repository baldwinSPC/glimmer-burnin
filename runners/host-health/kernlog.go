package main

import (
	"bufio"
	"errors"
	"fmt"
	"io"
	"os"
	"regexp"
	"strings"
	"syscall"
)

// Kernel-log probe: the Xid scan.
//
// An Xid is the NVIDIA driver's own fault report, and it is the single most
// informative host-side signal there is — it names double-bit ECC (Xid 48,
// 63, 64), a GPU that fell off the bus (79), MMU faults (13, 31) and thermal
// shutdowns (62). None of it reaches userspace through NVML; it only ever
// appears in the kernel log, which is why this probe exists at all.
//
// Two sources are tried, in order:
//
//   - /dev/kmsg, the kernel's structured log. Preferred: it is the primary,
//     it needs no logging daemon, and it can be differenced exactly, because
//     an open file descriptor keeps its position in the ring buffer.
//   - a plain text kernel log (/var/log/kern.log and friends), differenced by
//     byte offset, for hosts that mount one into the container.
//
// If neither is readable the probe reports xid_source=none and emits NO
// xid_count. That is the fails-closed outcome: a profile thresholding
// xidEvents fails on a node where the Xid scan could not run, rather than
// passing it on a zero nobody measured. See the README for what the pod needs
// in order for /dev/kmsg to be readable at all.

// xidPattern matches the driver's Xid line, e.g.
//
//	NVRM: Xid (PCI:0000:01:00): 13, pid=1234, Graphics SM Warp Exception
//
// Deliberately loose about what follows "Xid": the message body has changed
// shape across driver generations, and a scan that silently stops matching
// after a driver upgrade is worse than one that occasionally over-matches.
var xidPattern = regexp.MustCompile(`(?i)NVRM:\s*Xid`)

// hwErrorPatterns are vendor-neutral kernel hardware-fault markers. The Xid
// scan above is NVIDIA-specific by construction — an Xid is an NVIDIA concept —
// so this second, coarser scan is what gives the probe something to say on a
// node whose accelerator is not NVIDIA, and on host-side faults (MCE, EDAC)
// that no accelerator counter would ever show.
//
// It is heuristic, so it is EVIDENCE ONLY and never gated: a false positive
// here must not condemn a node. A non-zero kernelHwErrors is a "go read the
// kernel log" signal for a human.
var hwErrorPatterns = []*regexp.Regexp{
	regexp.MustCompile(`(?i)machine check`),           // x86 MCE
	regexp.MustCompile(`(?i)\[hardware error\]`),      // mce/EDAC printk prefix
	regexp.MustCompile(`(?i)uncorrectable error`),     // EDAC, NVMe, PCIe
	regexp.MustCompile(`(?i)amdgpu.*(gpu reset|ras)`), // AMD accelerator fault
	regexp.MustCompile(`(?i)pcieport.*aer`),           // AER reported through pcieport
}

// kernelLog is a differenceable source of kernel messages.
type kernelLog interface {
	// scan visits every message that appeared since the previous scan, in
	// order. The first call visits everything the source can already see.
	//
	// IT IS A VISITOR AND NOT A []string, AND THAT IS THE WHOLE POINT (#112).
	// Both implementations used to accumulate the entire scan into a slice for
	// a caller that immediately reduced it to two counters, so peak memory was
	// a function of how much log the host had — the kernel ring buffer on the
	// kmsg path, and on the file path the whole unread tail of
	// /var/log/syslog, which is bounded by nothing at all. A Go heap allocation
	// that fails is a runtime fatal error, no deferred recover can intercept
	// it, and the process exits 2: the undeclared skip code, on the one runner
	// in this repository that claims never to skip. Nothing is retained here
	// now, so the scan costs one buffer regardless of the host's uptime.
	scan(visit func(string)) error
	close()
}

type kernelLogProbe struct {
	src    kernelLog
	source string // kmsg | kernlog | none

	xidPre, hwPre int64
	xidNew, hwNew int64
	// baselineOK and windowOK say whether each SAMPLE was actually taken. They
	// are not the same statement as `dropped`, and conflating them is what
	// produced both halves of issue #112's second round: a scan that returned an
	// error yielded (0, 0), which is indistinguishable from a clean scan of a
	// quiet log, and that zero was then published as a measurement.
	baselineOK bool
	windowOK   bool
	dropped    bool
	available  bool
	// tried records every candidate source that did not open, so that
	// xid_source=none can say which and why. See why().
	tried []sourceAttempt
}

// newKernelLogProbe picks a source and, when it cannot, RECORDS WHY.
//
// "xid_source=none" on its own is the same dead end as "runner terminated
// abnormally" was before #52: it says a probe did not run and nothing about what
// to change. Issue #134 is that dead end met on real hardware — /dev/kmsg
// mounted exactly as the CRD documents, and still none, with no way to tell a
// missing mount from a permission problem from a path that was never tried.
//
// So every candidate's failure is kept and reported. The reasons are the ones a
// reader can act on:
//
//	ENOENT  the path is not in the container — the hostPaths mount is missing
//	        or names a different path
//	EACCES  it is there and this process may not read it. On /dev/kmsg that is
//	        almost always kernel.dmesg_restrict=1, which requires CAP_SYSLOG,
//	        which a container running as a NON-ROOT uid does not have in its
//	        effective set even when the pod is privileged. On a text kern.log it
//	        is the file's own mode — Debian ships -rw-r----- syslog:adm.
//
// Both are configuration, both are fixable, and neither is visible from
// "xid_source=none".
func newKernelLogProbe(cfg config) *kernelLogProbe {
	p := &kernelLogProbe{source: "none"}

	if cfg.kmsgPath != "" {
		src, err := openKmsg(cfg.kmsgPath)
		if err == nil {
			return &kernelLogProbe{src: src, source: "kmsg", available: true}
		}
		p.tried = append(p.tried, sourceAttempt{path: cfg.kmsgPath, err: err})
	}
	for _, path := range cfg.kernLogPaths {
		src, err := openFileLog(path)
		if err == nil {
			return &kernelLogProbe{src: src, source: "kernlog", available: true}
		}
		p.tried = append(p.tried, sourceAttempt{path: path, err: err})
	}
	return p
}

// sourceAttempt is one candidate that did not work out.
type sourceAttempt struct {
	path string
	err  error
}

// why renders the attempts as one line an operator can act on.
//
// It names the ERRNO rather than the Go error text, because the errno is what
// distinguishes the two configurations that produce this, and it appends the
// remedy for the one that is otherwise invisible.
func (k *kernelLogProbe) why() string {
	if len(k.tried) == 0 {
		return "no kernel-log source was configured (BURNIN_KMSG_PATH and BURNIN_KERN_LOG_PATHS are both empty)"
	}
	parts := make([]string, 0, len(k.tried))
	permission := false
	for _, a := range k.tried {
		reason := "cannot open"
		switch {
		case errors.Is(a.err, os.ErrNotExist), errors.Is(a.err, syscall.ENOENT):
			reason = "not present in this container (mount it with spec.runner.hostPaths)"
		case errors.Is(a.err, os.ErrPermission), errors.Is(a.err, syscall.EACCES), errors.Is(a.err, syscall.EPERM):
			reason = "present but not readable by this process"
			permission = true
		}
		parts = append(parts, a.path+": "+reason)
	}
	out := strings.Join(parts, "; ")
	if permission {
		// The remedy nobody can guess from an errno. A privileged pod does NOT
		// supply CAP_SYSLOG to a container running as a non-root uid, because
		// the effective set is dropped on the switch away from root — so
		// "privileged: true" looks like it should be enough and is not.
		out += ". This image runs as uid 65532 by design. /dev/kmsg needs CAP_SYSLOG " +
			"wherever kernel.dmesg_restrict=1, and a privileged pod does NOT grant it to a " +
			"non-root uid; a text kern.log is usually mode -rw-r----- syslog:adm and is " +
			"unreadable for the same reason. Set spec.runner.runAsUser: 0, or mount a " +
			"kernel log this uid can read and point BURNIN_KERN_LOG_PATHS at it"
	}
	return out
}

// baseline counts what the log already held before the window opened.
//
// Those Xids are reported separately as xid_preexisting rather than folded into
// xid_count. The distinction is the difference between "this node produced a
// fault while we watched" and "this node produced a fault at some point since
// boot", and only the first is what xidEvents is defined to mean. The second is
// at least as damning for acceptance, which is why it is emitted — a profile
// that wants to hold a node's whole boot against it thresholds xidPreexisting,
// and that choice stays visible in the BurnInTest instead of being hidden here.
func (k *kernelLogProbe) baseline() {
	if !k.available {
		return
	}
	k.xidPre, k.hwPre, k.baselineOK = k.count()
}

// collect closes the window and releases the source.
func (k *kernelLogProbe) collect() {
	if !k.available {
		return
	}
	k.xidNew, k.hwNew, k.windowOK = k.count()
	k.src.close()
}

// count drains the source and tallies as it goes, retaining nothing.
//
// The third return is whether the scan COMPLETED, and it is separate from the
// tallies on purpose. A failed scan yields zero messages, which is exactly what
// a clean scan of a quiet log yields, so a caller that reads only the counts
// cannot tell "nothing happened" from "we could not look" — and this runner's
// first rule is that those are different claims.
func (k *kernelLogProbe) count() (xid, hw int64, ok bool) {
	var c messageCounter
	if err := k.src.scan(c.visit); err != nil {
		k.dropped = true
		// The partial tally is discarded rather than returned. It is a floor on
		// an interval whose start we do not know, which is not a measurement of
		// anything a threshold could be written against.
		return 0, 0, false
	}
	return c.xid, c.hw, true
}

// emit publishes only the counters that were actually measured.
//
// THE WINDOW COUNT NEEDS BOTH SAMPLES, and that is the part that was wrong. It
// is a difference: the source is positioned by the baseline scan, so the closing
// scan returns what arrived since. If the BASELINE failed, the source was never
// positioned — a file source still sits at byte 0 — so the "window" is the whole
// log, and every Xid the node has ever logged is reported as having happened
// during the burn-in. That is exit 1, a hardware verdict, against healthy
// hardware, on the one phase this operator never retries. If the CLOSING scan
// failed, the window was never read at all, and publishing 0 hands a profile's
// `xidEvents Equal 0` gate a number nobody measured.
//
// So each counter is emitted only when the samples it is made of were taken, and
// omitted otherwise. Omission is the fails-closed outcome this runner already
// documents for an unreadable source, and it is deliberately NOT the `n/a`
// sentinel: `n/a` declares that the HARDWARE cannot produce the value, which is
// a claim about the part, and this is a claim about the probe.
func (k *kernelLogProbe) emit(out *emitter) {
	out.set("xid_source", k.source)
	if !k.available {
		// The whole point of #134: a probe that did not run says what to change.
		out.set("xid_source_detail", k.why())
		return
	}
	if k.baselineOK && k.windowOK {
		out.setInt(keyXidCount, k.xidNew)
		out.setInt("kernel_hw_errors", k.hwNew)
	}
	if k.baselineOK {
		out.setInt("xid_preexisting", k.xidPre)
		out.setInt("kernel_hw_errors_preexisting", k.hwPre)
	}
	if k.dropped {
		// The ring buffer wrapped, the reader fell behind, or a scan failed
		// outright. Worth knowing before trusting anything above it — and on the
		// record even when every counter was omitted, because otherwise the only
		// trace of a failed probe is an absence.
		out.setInt("xid_log_dropped", 1)
	}
}

// messageCounter is the entire reason the scan ever produced messages: two
// tallies. It is a struct with a visit method rather than a closure so the same
// counting rule is reachable from a test without re-stating it.
type messageCounter struct{ xid, hw int64 }

func (c *messageCounter) visit(m string) {
	if xidPattern.MatchString(m) {
		c.xid++
	}
	for _, p := range hwErrorPatterns {
		if p.MatchString(m) {
			c.hw++
			return
		}
	}
}

// ---------------------------------------------------------------------------
// /dev/kmsg
// ---------------------------------------------------------------------------

// kmsgSource reads /dev/kmsg through a raw file descriptor.
//
// The raw descriptor is not incidental. os.File registers a pollable character
// device with the Go runtime poller, and a Read on an empty /dev/kmsg would
// then BLOCK waiting for the next kernel message instead of returning EAGAIN —
// the runner would hang until the pod's deadline killed it. syscall.Read on a
// descriptor the runtime never adopted returns EAGAIN as the kernel intended.
type kmsgSource struct {
	fd int
	// maxReads bounds one drain. It is a field rather than a bare use of the
	// constant so a test can lower it: with a 16 KiB buffer a regular file
	// reaches EOF long before 200000 reads, so the exhaustion path is otherwise
	// unreachable from a test and would be a guard nobody had ever executed.
	maxReads int
	// pending holds a trailing partial line. A read of the real /dev/kmsg
	// always returns whole records, but BURNIN_KMSG_PATH may point at a
	// captured log, where one read returns many records and can stop
	// mid-record at the buffer boundary.
	pending string
}

// maxRecordsPerScan bounds one drain. The ring buffer is finite, but a source
// that never returns EAGAIN (a misconfigured BURNIN_KMSG_PATH, say) must not
// spin forever.
const maxRecordsPerScan = 200000

// kmsgBufSize exceeds the kernel's maximum record size; a short buffer makes
// read() fail with EINVAL rather than truncate.
const kmsgBufSize = 16384

func openKmsg(path string) (*kmsgSource, error) {
	fd, err := syscall.Open(path, syscall.O_RDONLY|syscall.O_NONBLOCK, 0)
	if err != nil {
		return nil, err
	}
	return &kmsgSource{fd: fd, maxReads: maxRecordsPerScan}, nil
}

func (k *kmsgSource) scan(visit func(string)) error {
	buf := make([]byte, kmsgBufSize)
	var dropped error
	for i := 0; i < k.maxReads; i++ {
		n, err := syscall.Read(k.fd, buf)
		switch {
		case errors.Is(err, syscall.EINTR):
			continue
		case errors.Is(err, syscall.EAGAIN), errors.Is(err, syscall.EWOULDBLOCK):
			// Nothing more to read right now: the window is drained.
			return dropped
		case errors.Is(err, syscall.EPIPE):
			// Records were overwritten before we read them. The descriptor has
			// already been repositioned at the next valid record, so keep
			// going — but remember that the count is now a floor.
			dropped = err
			continue
		case err != nil:
			return err
		case n == 0:
			// EOF. Only a regular file (a test fixture, or an operator pointing
			// BURNIN_KMSG_PATH at a captured log) can produce this.
			return dropped
		}
		k.pending = visitKmsgChunk(k.pending+string(buf[:n]), visit)
	}
	// The bound was exhausted, so the drain did NOT complete. Reporting success
	// here would hand the caller a truncated baseline that looks like a whole
	// one: the source is left positioned somewhere arbitrary, and everything past
	// the cut is counted as a window event by the next scan. Same wrong verdict
	// as a failed baseline, reached by a different route.
	return fmt.Errorf("kmsg drain stopped after %d reads without reaching the end of the log", k.maxReads)
}

// visitKmsgChunk hands one read's messages to visit and returns any trailing
// partial line to carry into the next read.
//
// A read of /dev/kmsg returns exactly one record, so this loop usually runs
// once — but it must not ASSUME that, because a single read of a regular file
// returns as many records as fit in the buffer and dropping all but the first
// would silently lose Xids.
func visitKmsgChunk(chunk string, visit func(string)) string {
	for {
		line, rest, ok := strings.Cut(chunk, "\n")
		if !ok {
			return chunk
		}
		chunk = rest
		// Continuation lines carry the record's key=value dictionary
		// (SUBSYSTEM=, DEVICE=), not text a fault would appear in.
		if line == "" || line[0] == ' ' || line[0] == '\t' {
			continue
		}
		visit(parseKmsgRecord(line))
	}
}

func (k *kmsgSource) close() {
	_ = syscall.Close(k.fd)
}

// parseKmsgRecord extracts the human-readable text of one /dev/kmsg record.
//
// A record is "<prio>,<seq>,<usec>,<flags>[,<k>=<v>...];<message>" optionally
// followed by continuation lines that start with a space. Anything that is not
// shaped like that is returned as-is rather than dropped: an unparsed line that
// contains "Xid" should still be counted.
func parseKmsgRecord(rec string) string {
	_, rest, ok := strings.Cut(rec, ";")
	if !ok {
		return strings.TrimRight(rec, "\n")
	}
	msg, _, _ := strings.Cut(rest, "\n")
	return msg
}

// ---------------------------------------------------------------------------
// text kernel log
// ---------------------------------------------------------------------------

// fileSource differences a plain text log by byte offset.
type fileSource struct {
	path string
	off  int64
	// info is the file we were reading last time, kept so that a rotation can
	// be told apart from an append.
	info os.FileInfo
}

func openFileLog(path string) (*fileSource, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	_ = f.Close()
	return &fileSource{path: path}, nil
}

// scan streams the unread tail of the file.
//
// The tail is bounded by nothing this runner controls — a node that has been up
// for months and never rotated /var/log/syslog presents the whole of it on the
// FIRST scan, which is the baseline taken before any output has been produced.
// Reading it into a slice was the largest unbounded allocation in the runner and
// the most plausible route to the exit 2 of #112.
func (f *fileSource) scan(visit func(string)) error {
	fh, err := os.Open(f.path)
	if err != nil {
		return err
	}
	defer func() { _ = fh.Close() }()

	if st, err := fh.Stat(); err == nil {
		// Two ways the log can move out from under a byte offset, and both must
		// reset it — otherwise the scan seeks past the end of a fresh file and
		// reports a clean window while Xids pile up in it.
		rotated := f.info != nil && !os.SameFile(f.info, st) // renamed away, new file created
		truncated := st.Size() < f.off                       // truncated in place
		if rotated || truncated {
			f.off = 0
		}
		f.info = st
	}
	if _, err := fh.Seek(f.off, io.SeekStart); err != nil {
		return err
	}

	var consumed int64
	r := bufio.NewReader(fh)
	for {
		line, err := r.ReadString('\n')
		if err != nil {
			// A trailing partial line is NOT consumed: the writer is mid-line,
			// and the rest of it belongs to the next scan.
			break
		}
		consumed += int64(len(line))
		visit(strings.TrimRight(line, "\n"))
	}
	f.off += consumed
	return nil
}

func (f *fileSource) close() {}
