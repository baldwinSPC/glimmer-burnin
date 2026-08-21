// kmsg_watch.h — attributes a driver-logged GPU fault to THIS test's own load
// window, for the whole soak family: thermal-soak, gpu-burn, and their -rocm
// counterparts.
//
// SPDX-License-Identifier: Apache-2.0
// Copyright the Glimmer authors.
//
// SHARED FILE, byte-identical in FOUR directories — runners/thermal-soak/kmsg/,
// runners/gpu-burn/kmsg/, runners/thermal-soak-rocm/kmsg/ and
// runners/gpu-burn-rocm/kmsg/ — guarded the same way soak_core.cuh already is.
// Edit one, copy it to the other three.
//
// One implementation rather than a leaner vendor-specific copy, on purpose: the
// highest-risk code here is the OS-facing fd/lseek/EPIPE mechanics (Watch,
// below), and a second, independently-maintained copy of it is exactly the kind
// of drift this project's "duplication that cannot be removed is duplication
// that has to be checked" rule exists to catch — a fix to the EPIPE handling
// landing in the NVIDIA copies and never reaching the ROCm ones would be
// invisible until somebody hit the same bug twice on AMD hardware. The
// NVIDIA-specific functions (LineHasXid, ExtractXidCode) go unused in a ROCm
// build and cost it nothing observable: they are never called from
// soak_core_rocm.h and are eliminated as dead code at -O3.
//
// It lives in this subdirectory, sibling to kmsg_watch_test.cc, rather than
// beside soak_core.cuh/soak_core_rocm.h directly, because runners/thermal-soak/
// already holds a Go package (soak_contract_test.go) and a directory holding Go
// sources cannot also hold a .cc file — the same reason runners/nccl/collective/
// exists. The other three directories have no such conflict but keep the
// identical kmsg/ layout anyway, so that a SINGLE relative #include path
// ("kmsg/kmsg_watch.h") is correct from every one of the four engine files.
//
// WHY THIS EXISTS
// ----------------
// "Zero Xid errors during the load" is the single most common pass criterion in
// every operator's own burn-in write-up, and before this file this operator
// could only half state it: runners/host-health already scans /dev/kmsg for
// NVRM Xid lines, but it runs as its OWN, separate, usually-short test — so an
// Xid it observes is attributed to "sometime before this Node's host-health
// ran", never to a SPECIFIC hour-long soak's own load window. A profile that
// wants "the part logged nothing while I was hammering it" has never been able
// to say that about the test doing the hammering.
//
// Two other designs were weighed and rejected before this one:
//
//   (i)  a second, companion pod running alongside the soak, watching /dev/kmsg
//        concurrently. Rejected: this operator runs tests STRICTLY IN PLAN
//        ORDER, one at a time — the next test does not start on a node until
//        the previous one owes no result there (see the comment in
//        internal/controller/burninrun_controller.go's advance loop) — and
//        podOutcome() reads the exit status of exactly ONE container, "runner".
//        A companion pod is a second thing this operator would have to create,
//        watch, and destroy IN LOCKSTEP with the first, which is the JobSet
//        argument CLAUDE.md already refuses for Group scope, reached again by a
//        different route. It would also double the node-count interlock's
//        accounting for one pod's worth of actual load.
//   (ii) run host-health AFTER the soak instead of changing the soak. Rejected:
//        it cannot attribute an Xid to the test that provoked it, only to "the
//        node, sometime before I ran" — and worse, a second soak segment's
//        clean host-health read could not un-implicate a fault that occurred
//        during the FIRST segment's window, because host-health has no notion
//        of "this soak's segment 3" at all.
//
// So this is (iii): the load-generating process reads its OWN /dev/kmsg,
// itself, inside the same window it is already measuring clock and temperature
// over. No second pod, no second schedule, no lockstep.
//
// THE MECHANISM MIRRORS runners/host-health/kernlog.go, on purpose
// ------------------------------------------------------------------
// An open file descriptor on /dev/kmsg keeps its own read position in the
// kernel's ring buffer, independent of any other open. host-health's Go probe
// exploits that by draining once at baseline (to position itself at "now") and
// draining again at the end (which then returns only what arrived since). This
// header does the SAME positional trick, cheaper: rather than draining and
// discarding whatever is already buffered, it seeks straight to the tail —
// lseek(fd, 0, SEEK_END) is a documented /dev/kmsg operation (the same one
// dmesg and journald use to "start from now"), and unlike a drain it costs one
// syscall regardless of how much history the ring buffer holds. That matters
// here specifically because this call happens before an hour-long GEMM even
// starts, and unlike host-health this runner has no use for a "preexisting"
// count to justify draining and counting the old content anyway — host-health
// already owns that number, at the node level.
//
// It is DELIBERATELY NOT the alternative design of recording a wall-clock
// timestamp and filtering each record's own embedded one at the end. That would
// need CLOCK_MONOTONIC to correlate with /dev/kmsg's own local_clock()-derived
// usec field, which is the same assumption `dmesg -T` and journald make and
// which this project has no real hardware to verify. A WRONG correlation would
// silently shift the window boundary — potentially COUNTING an Xid that
// happened moments before this test even started, which is the one direction
// this project's whole verdict philosophy refuses ("Fail is never retried").
// The lseek approach makes no clock assumption at all: the kernel itself
// decides what "the tail, right now" means, and everything this descriptor
// reads afterward genuinely arrived after that.
//
// WHAT IS OMITTED, AND WHY
// -------------------------
// If SEEK_END fails (older kernel, non-standard /dev/kmsg), the probe reports
// itself unavailable rather than falling back to reading from the ring buffer's
// start — a fallback there would silently include the node's whole prior
// history in "this test's window", which is the overcounting version of the
// exact bug the fails-closed rule exists to prevent. Absence beats a wrong
// number: xid_source=none, xidEvents omitted, exactly as host-health degrades.
//
// xidEvents is METRIC ONLY. Neither this header nor soak_core.cuh's run() lets
// an observed Xid change the runner's OWN pass/fail decision — that decision
// stays exactly what it was: miscompares, nonfiniteCount, throttleEvents, the
// clock floor, the optional temperature ceiling. xidEvents joins eccErrors as a
// metric a PROFILE gates externally via spec.thresholds, not one the runner
// bakes in — because, like eccErrors, it is only ever measurable when the
// operator has granted host access this runner does not have by default, and a
// runner's own hard verdict must never depend on whether that grant happened to
// be remembered. See runners/thermal-soak/README.md's "What the pod needs".
//
// PATTERNS ARE VENDOR-SPECIFIC; THE MECHANISM IS NOT. This file supplies the
// NVIDIA Xid pattern and code extraction, used identically by both NVIDIA soak
// kinds. runners/thermal-soak-rocm/soak_core_rocm.h supplies its OWN,
// deliberately narrower amdgpu reset/RAS pattern to the generic multi-substring
// matcher below — see that file for why it is narrower than
// runners/host-health's own amdgpu heuristic.

#ifndef GLIMMER_BURNIN_KMSG_WATCH_H
#define GLIMMER_BURNIN_KMSG_WATCH_H

#include <cctype>
#include <cerrno>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <functional>
#include <string>
#include <unistd.h>
#include <vector>

namespace burnin {
namespace kmsg {

// ── pure text handling: no file descriptor, no syscall, fully unit-testable ──
// See kmsg_watch_test.cc, compiled and run with plain c++, no CUDA — this
// project has no way to grant a test process /dev/kmsg access in CI, so this
// boundary is where the logic that decides a VERDICT-FEEDING count is checked.

// ToLowerAscii avoids the locale-sensitive std::tolower, which a kernel log
// line — English, ASCII, machine-generated — never needs and which a container
// running under an unexpected LC_ALL could answer differently on two nodes.
inline char ToLowerAscii(char c) {
  if (c >= 'A' && c <= 'Z') return static_cast<char>(c - 'A' + 'a');
  return c;
}

// ContainsCI is a case-insensitive substring search, hand-rolled rather than
// pulling in <regex>: every pattern this file matches is a fixed literal, never
// a shape that needs alternation or repetition, and <regex>'s compile cost is
// not free on a binary launched fresh for every burn-in pod.
inline bool ContainsCI(const std::string &haystack, const std::string &needle) {
  if (needle.empty()) return true;
  if (needle.size() > haystack.size()) return false;
  for (size_t i = 0; i + needle.size() <= haystack.size(); ++i) {
    bool match = true;
    for (size_t j = 0; j < needle.size(); ++j) {
      if (ToLowerAscii(haystack[i + j]) != ToLowerAscii(needle[j])) {
        match = false;
        break;
      }
    }
    if (match) return true;
  }
  return false;
}

// MatchesAll is the generic AND-matcher runners/thermal-soak-rocm/
// soak_core_rocm.h drives with its own amdgpu pattern list. Every substring in
// `all` must appear somewhere in the line, case-insensitively, in any order —
// mirroring the shape of host-health's own `(?i)amdgpu.*(gpu reset|ras)`
// pattern (an ordered "amdgpu, then later X" regex is, for a single line of log
// text, equivalent to "both amdgpu and X are present somewhere in it").
inline bool MatchesAll(const std::string &line, const std::vector<std::string> &all) {
  for (const auto &needle : all) {
    if (!ContainsCI(line, needle)) return false;
  }
  return true;
}

// LineHasXid mirrors host-health's xidPattern, `(?i)NVRM:\s*Xid`. Deliberately
// loose about what follows "Xid" — the message body has changed shape across
// driver generations, and a scan that silently stops matching after a driver
// upgrade is worse than one that occasionally over-matches on a line that was
// never going to be counted as anything else anyway.
inline bool LineHasXid(const std::string &line) {
  size_t pos = 0;
  for (;;) {
    size_t at = std::string::npos;
    for (size_t i = pos; i + 5 <= line.size(); ++i) {
      if (ToLowerAscii(line[i]) == 'n' && ToLowerAscii(line[i + 1]) == 'v' &&
          ToLowerAscii(line[i + 2]) == 'r' && ToLowerAscii(line[i + 3]) == 'm' && line[i + 4] == ':') {
        at = i;
        break;
      }
    }
    if (at == std::string::npos) return false;
    size_t j = at + 5;
    while (j < line.size() && std::isspace(static_cast<unsigned char>(line[j]))) ++j;
    if (j + 3 <= line.size() && ToLowerAscii(line[j]) == 'x' && ToLowerAscii(line[j + 1]) == 'i' &&
        ToLowerAscii(line[j + 2]) == 'd') {
      return true;
    }
    pos = at + 5; // keep scanning: a line can carry more than one "NVRM:" clause
  }
}

// ExtractXidCode reads the integer that follows the word "Xid" in a line
// LineHasXid already matched, e.g.
//
//   NVRM: Xid (PCI:0000:01:00): 13, pid=1234, Graphics SM Warp Exception
//   NVRM: Xid: 79
//
// Both shapes appear in host-health's own captured fixtures. The code is
// whatever contiguous run of digits appears after the NEXT ':' following "Xid"
// (skipping an optional parenthesised PCI clause first) — deliberately not
// "the first number anywhere on the line", because the PCI bus id embeds its
// own digits and would be read as the code otherwise.
inline bool ExtractXidCode(const std::string &line, long *code) {
  size_t xid = std::string::npos;
  for (size_t i = 0; i + 3 <= line.size(); ++i) {
    if (ToLowerAscii(line[i]) == 'x' && ToLowerAscii(line[i + 1]) == 'i' && ToLowerAscii(line[i + 2]) == 'd') {
      xid = i;
      break;
    }
  }
  if (xid == std::string::npos) return false;
  size_t i = xid + 3;
  // Skip one parenthesised clause, e.g. "(PCI:0000:01:00)", if present —
  // otherwise its own digits would be mistaken for the code.
  while (i < line.size() && std::isspace(static_cast<unsigned char>(line[i]))) ++i;
  if (i < line.size() && line[i] == '(') {
    size_t close = line.find(')', i);
    if (close == std::string::npos) return false;
    i = close + 1;
  }
  while (i < line.size() && std::isspace(static_cast<unsigned char>(line[i]))) ++i;
  if (i >= line.size() || line[i] != ':') return false;
  ++i;
  while (i < line.size() && std::isspace(static_cast<unsigned char>(line[i]))) ++i;
  size_t start = i;
  while (i < line.size() && std::isdigit(static_cast<unsigned char>(line[i]))) ++i;
  if (i == start) return false;
  *code = std::strtol(line.substr(start, i - start).c_str(), nullptr, 10);
  return true;
}

// ParseKmsgRecord extracts the human-readable text of one /dev/kmsg record.
// Byte-for-byte the same contract as host-health's parseKmsgRecord: a record is
// "<prio>,<seq>,<usec>,<flags>[,<k>=<v>...];<message>", optionally followed by
// continuation lines the caller has already stripped. Anything not shaped like
// that is returned as-is rather than dropped, so an unparsed line that still
// contains "Xid" is still counted.
inline std::string ParseKmsgRecord(const std::string &rec) {
  size_t semi = rec.find(';');
  if (semi == std::string::npos) return rec;
  std::string msg = rec.substr(semi + 1);
  size_t nl = msg.find('\n');
  if (nl != std::string::npos) msg = msg.substr(0, nl);
  return msg;
}

// VisitKmsgChunk hands one read's records to visit and returns any trailing
// partial line to prepend to the next chunk. A read of the real /dev/kmsg
// returns exactly one record, so this usually runs once — but it must not
// assume that: a single read of a regular file (BURNIN_KMSG_PATH pointed at a
// captured log, or a test fixture) returns as many records as fit in the
// buffer, and dropping all but the first would silently lose Xids.
inline std::string VisitKmsgChunk(const std::string &chunk, const std::function<void(const std::string &)> &visit) {
  size_t start = 0;
  for (;;) {
    size_t nl = chunk.find('\n', start);
    if (nl == std::string::npos) return chunk.substr(start);
    std::string line = chunk.substr(start, nl - start);
    start = nl + 1;
    // Continuation lines carry the record's key=value dictionary (SUBSYSTEM=,
    // DEVICE=), not text a fault would appear in.
    if (line.empty() || line[0] == ' ' || line[0] == '\t') continue;
    visit(ParseKmsgRecord(line));
  }
}

// Tally accumulates matches across however many drains the caller makes —
// CUMULATIVELY, across the whole soak, not reset between calls. That is what
// lets it be updated on the same periodic cadence soak_core.cuh already uses
// for every other metric: each periodic drain adds whatever is new since the
// last one, so a pod killed between two drains still has an up-to-date count
// from the last one it completed, exactly like throttleEvents already does.
struct Tally {
  long xidCount = 0;
  bool haveLastXidCode = false;
  long lastXidCode = 0;

  long faultCount = 0; // vendor-neutral: NVIDIA Xid OR (from soak_core_rocm.h) amdgpu

  // ObserveNvidia is what kmsg_watch.h's own OS-facing reader calls for the
  // NVIDIA soak kinds.
  void ObserveNvidia(const std::string &line) {
    if (!LineHasXid(line)) return;
    xidCount++;
    faultCount++;
    long code = 0;
    if (ExtractXidCode(line, &code)) {
      haveLastXidCode = true;
      lastXidCode = code;
    }
  }

  // ObserveGeneric is what runners/thermal-soak-rocm/soak_core_rocm.h drives
  // with its own amdgpu pattern list. No code is ever extracted here — AMD's
  // kernel messages carry no single canonical numeric field the way NVIDIA's
  // "Xid: NN" does, and inventing one would be declaring a measurement this
  // runner never took.
  void ObserveGeneric(const std::string &line, const std::vector<std::vector<std::string>> &patternSets) {
    for (const auto &set : patternSets) {
      if (MatchesAll(line, set)) {
        faultCount++;
        return; // one line counts once, even if it happens to match two sets
      }
    }
  }
};

// ── the OS-facing half: opens /dev/kmsg, positions at "now", drains on demand ─

// Watch owns one /dev/kmsg descriptor for the life of the process.
//
// Not copyable — it owns a raw fd — and there is never a reason to have two:
// one runner process, one accelerator, one window.
class Watch {
 public:
  Watch() {
    const char *path = std::getenv("BURNIN_KMSG_PATH");
    if (path == nullptr || *path == '\0') path = "/dev/kmsg";
    path_ = path;

    fd_ = open(path_.c_str(), O_RDONLY | O_NONBLOCK);
    if (fd_ < 0) {
      if (errno == ENOENT) {
        why_ = path_ + ": not present in this container (mount it with spec.runner.hostPaths)";
      } else {
        if (errno == EACCES || errno == EPERM) errnoWasPermission_ = true;
        why_ = path_ + ": cannot open (" + std::string(strerror(errno)) + ")";
      }
      return;
    }
    // Position at "now" — see this file's header comment for why this is a
    // seek rather than a drain-and-discard, and why a failure here must not
    // fall back to reading from the buffer's start.
    if (lseek(fd_, 0, SEEK_END) < 0) {
      why_ = path_ + ": opened, but this kernel refused to position /dev/kmsg at its tail (" +
             std::string(strerror(errno)) +
             "); reading from the start would count this node's prior history as part of this "
             "test's window, so this probe is refusing rather than risking that";
      close(fd_);
      fd_ = -1;
      return;
    }
    available_ = true;
  }

  ~Watch() {
    if (fd_ >= 0) close(fd_);
  }

  Watch(const Watch &) = delete;
  Watch &operator=(const Watch &) = delete;

  bool Available() const { return available_; }
  bool Dropped() const { return dropped_; }
  // Why explains an unavailable probe, mirroring host-health's kernelLogProbe.why():
  // the errno distinguishes "not mounted" from "mounted but unreadable", and only
  // the second needs the CAP_SYSLOG/uid-0 explanation appended.
  std::string Why() const {
    if (available_) return "";
    std::string out = why_;
    if (errnoWasPermission_) {
      out += ". Reading /dev/kmsg from a non-root container needs THREE things at once — a "
             "device-cgroup grant (privileged: true), uid 0 (a capability added to the bounding "
             "set does nothing for a non-root uid — this image runs as uid 65532 by default), and "
             "CAP_SYSLOG wherever kernel.dmesg_restrict=1, which is the default on Ubuntu and "
             "Debian. See runners/thermal-soak/README.md's \"What the pod needs\"";
    }
    return out;
  }

  // Collect drains whatever has arrived since the last call (or since Watch()
  // positioned the descriptor, on the first call), calling onMessage with each
  // record's text. A no-op, safely, when the probe is unavailable. The caller
  // supplies the vendor's own observation logic — Tally::ObserveNvidia for the
  // NVIDIA soak kinds, or ObserveGeneric with an amdgpu pattern list from
  // soak_core_rocm.h — so this class stays vendor-neutral.
  void Collect(const std::function<void(const std::string &)> &onMessage) {
    if (!available_) return;
    char buf[16384];
    std::string pending;
    for (int reads = 0; reads < kMaxReadsPerDrain; ++reads) {
      const ssize_t n = read(fd_, buf, sizeof(buf));
      if (n < 0) {
        if (errno == EINTR) continue;
        if (errno == EAGAIN || errno == EWOULDBLOCK) return; // drained; window stays open
        if (errno == EPIPE) {
          // Records were overwritten before we read them — the ring buffer
          // wrapped faster than this drain could keep up. The descriptor is
          // already repositioned at the next valid record; the count from
          // here on is a floor, and that is worth recording.
          dropped_ = true;
          continue;
        }
        dropped_ = true; // an unexpected error: the same "we do not fully trust this drain" signal
        return;
      }
      if (n == 0) return; // EOF: only a regular file (BURNIN_KMSG_PATH override, or a test) does this
      pending = VisitKmsgChunk(pending + std::string(buf, static_cast<size_t>(n)), onMessage);
    }
    // The bound was exhausted without reaching EAGAIN/EOF. Reporting success
    // here would leave the descriptor positioned somewhere arbitrary and let
    // everything past the cut be double-counted or lost on the next drain —
    // the same wrong-verdict shape as a truncated baseline, reached differently.
    dropped_ = true;
  }

 private:
  // Bounded for the same reason host-health's kmsgSource is: a misconfigured
  // BURNIN_KMSG_PATH pointing at something that never returns EAGAIN or EOF
  // must not spin this soak forever instead of running its GEMM.
  static constexpr int kMaxReadsPerDrain = 200000;

  std::string path_;
  int fd_ = -1;
  bool available_ = false;
  bool dropped_ = false;
  bool errnoWasPermission_ = false;
  std::string why_;
};

} // namespace kmsg
} // namespace burnin

#endif // GLIMMER_BURNIN_KMSG_WATCH_H
