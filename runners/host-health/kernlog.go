package main

import (
	"bufio"
	"errors"
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
	// scan returns the messages that appeared since the previous scan. The
	// first call returns everything the source can already see.
	scan() ([]string, error)
	close()
}

type kernelLogProbe struct {
	src    kernelLog
	source string // kmsg | kernlog | none

	xidPre, hwPre int64
	xidNew, hwNew int64
	dropped       bool
	available     bool
}

func newKernelLogProbe(cfg config) *kernelLogProbe {
	if cfg.kmsgPath != "" {
		if src, err := openKmsg(cfg.kmsgPath); err == nil {
			return &kernelLogProbe{src: src, source: "kmsg", available: true}
		}
	}
	for _, p := range cfg.kernLogPaths {
		if src, err := openFileLog(p); err == nil {
			return &kernelLogProbe{src: src, source: "kernlog", available: true}
		}
	}
	return &kernelLogProbe{source: "none"}
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
	msgs, err := k.src.scan()
	if err != nil {
		k.dropped = true
	}
	k.xidPre, k.hwPre = countMessages(msgs)
}

// collect closes the window and releases the source.
func (k *kernelLogProbe) collect() {
	if !k.available {
		return
	}
	msgs, err := k.src.scan()
	if err != nil {
		k.dropped = true
	}
	k.xidNew, k.hwNew = countMessages(msgs)
	k.src.close()
}

func (k *kernelLogProbe) emit(out *emitter) {
	out.set("xid_source", k.source)
	if !k.available {
		return
	}
	out.setInt(keyXidCount, k.xidNew)
	out.setInt("xid_preexisting", k.xidPre)
	out.setInt("kernel_hw_errors", k.hwNew)
	out.setInt("kernel_hw_errors_preexisting", k.hwPre)
	if k.dropped {
		// The ring buffer wrapped or the reader fell behind, so the counts
		// above are a floor. Worth knowing before trusting a clean scan.
		out.setInt("xid_log_dropped", 1)
	}
}

func countMessages(msgs []string) (xid, hw int64) {
	for _, m := range msgs {
		if xidPattern.MatchString(m) {
			xid++
		}
		for _, p := range hwErrorPatterns {
			if p.MatchString(m) {
				hw++
				break
			}
		}
	}
	return xid, hw
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
	return &kmsgSource{fd: fd}, nil
}

func (k *kmsgSource) scan() ([]string, error) {
	buf := make([]byte, kmsgBufSize)
	var (
		msgs    []string
		dropped error
	)
	for i := 0; i < maxRecordsPerScan; i++ {
		n, err := syscall.Read(k.fd, buf)
		switch {
		case errors.Is(err, syscall.EINTR):
			continue
		case errors.Is(err, syscall.EAGAIN), errors.Is(err, syscall.EWOULDBLOCK):
			// Nothing more to read right now: the window is drained.
			return msgs, dropped
		case errors.Is(err, syscall.EPIPE):
			// Records were overwritten before we read them. The descriptor has
			// already been repositioned at the next valid record, so keep
			// going — but remember that the count is now a floor.
			dropped = err
			continue
		case err != nil:
			return msgs, err
		case n == 0:
			// EOF. Only a regular file (a test fixture, or an operator pointing
			// BURNIN_KMSG_PATH at a captured log) can produce this.
			return msgs, dropped
		}
		records, remainder := splitKmsgChunk(k.pending + string(buf[:n]))
		k.pending = remainder
		msgs = append(msgs, records...)
	}
	return msgs, dropped
}

// splitKmsgChunk turns one read into the messages it contains, plus any
// trailing partial line to carry into the next read.
//
// A read of /dev/kmsg returns exactly one record, so this loop usually runs
// once — but it must not ASSUME that, because a single read of a regular file
// returns as many records as fit in the buffer and dropping all but the first
// would silently lose Xids.
func splitKmsgChunk(chunk string) ([]string, string) {
	var out []string
	for {
		line, rest, ok := strings.Cut(chunk, "\n")
		if !ok {
			return out, chunk
		}
		chunk = rest
		// Continuation lines carry the record's key=value dictionary
		// (SUBSYSTEM=, DEVICE=), not text a fault would appear in.
		if line == "" || line[0] == ' ' || line[0] == '\t' {
			continue
		}
		out = append(out, parseKmsgRecord(line))
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

func (f *fileSource) scan() ([]string, error) {
	fh, err := os.Open(f.path)
	if err != nil {
		return nil, err
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
		return nil, err
	}

	var (
		lines    []string
		consumed int64
	)
	r := bufio.NewReader(fh)
	for {
		line, err := r.ReadString('\n')
		if err != nil {
			// A trailing partial line is NOT consumed: the writer is mid-line,
			// and the rest of it belongs to the next scan.
			break
		}
		consumed += int64(len(line))
		lines = append(lines, strings.TrimRight(line, "\n"))
	}
	f.off += consumed
	return lines, nil
}

func (f *fileSource) close() {}
