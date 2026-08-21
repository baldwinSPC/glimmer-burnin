// kmsg_watch_test.cc — tests for kmsg_watch.h: the pure line-matching /
// code-extraction functions, and the real /dev/kmsg mechanics exercised
// against a temp regular file the same way runners/host-health's Go tests
// exercise theirs (mustWrite/appendTo, then open/collect).
//
// SPDX-License-Identifier: Apache-2.0
// Copyright the Glimmer authors.
//
// Build and run:
//
//   c++ -std=c++17 -Wall -Wextra -o /tmp/kmsg_watch_test kmsg_watch_test.cc && /tmp/kmsg_watch_test
//
// runners/cxxtests_test.go does exactly that under `make test`. It lives in
// this subdirectory, sibling to kmsg_watch.h — see that file's own header
// comment for why. This project has no way to grant a test process real
// /dev/kmsg access (in CI or otherwise), so this is the ONLY place the
// window-boundary logic is exercised before real hardware: the fixtures below
// stand in for a captured kmsg log the same way host-health's Go tests do.

#include "kmsg_watch.h"

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <unistd.h>

using namespace burnin::kmsg;

namespace {

int failures = 0;

void check(bool ok, const char *what) {
  if (!ok) {
    std::printf("FAIL: %s\n", what);
    ++failures;
  }
}

// ── pure functions: ContainsCI / MatchesAll ─────────────────────────────────

void testContainsCI() {
  check(ContainsCI("NVRM: Xid", "xid"), "case-insensitive match");
  check(ContainsCI("NVRM: Xid", "NVRM"), "exact-case match");
  check(!ContainsCI("NVRM: Xid", "amdgpu"), "no match when absent");
  check(ContainsCI("x", "x"), "single-char haystack equals needle");
  check(!ContainsCI("x", "xy"), "needle longer than haystack never matches");
  check(ContainsCI("abcXID", "xid"), "match at the very end");
  check(ContainsCI("anything", ""), "an empty needle matches everything, vacuously");
}

void testMatchesAllIsAndAcrossPatternsOrAcrossSets() {
  const std::vector<std::string> needsBoth = {"amdgpu", "reset"};
  check(MatchesAll("amdgpu 0000:01:00.0: [gpu_recover] GPU reset begin", needsBoth),
        "both substrings present, in order");
  check(MatchesAll("GPU reset (amdgpu driver) issued", needsBoth),
        "both present, out of the order they're listed in — AND, not a sequence");
  check(!MatchesAll("amdgpu 0000:01:00.0: firmware loaded", needsBoth),
        "amdgpu present alone does not match — an AND-matcher, not an OR");
  check(!MatchesAll("GPU reset requested by user", needsBoth),
        "reset present alone (no amdgpu) does not match");
}

// ── pure functions: LineHasXid / ExtractXidCode ─────────────────────────────
// Fixtures are the SAME message shapes runners/host-health/kernlog_test.go
// asserts against, so a reviewer who knows that file recognises these.

void testLineHasXid() {
  check(LineHasXid("NVRM: Xid (PCI:0000:01:00): 13, pid=1234, Graphics SM Warp Exception"),
        "the canonical shape");
  check(LineHasXid("NVRM: Xid: 79"), "the short shape, no PCI clause");
  check(LineHasXid("nvrm: xid (pci:0000:02:00): 48, DBE"), "case-insensitive");
  check(LineHasXid("prefix NVRM:Xid: 63"), "no whitespace between NVRM: and Xid");
  check(!LineHasXid("some ordinary kernel line about nothing in particular"), "no match on an unrelated line");
  check(!LineHasXid("NVRM: version 580.82.09"), "NVRM: alone, no Xid clause, does not match");
  check(!LineHasXid("a line that mentions Xid but never NVRM:"),
        "Xid alone, with no NVRM: prefix, does not match — a driver-attributed fault needs both");
}

void testExtractXidCode() {
  long code = -1;
  check(ExtractXidCode("NVRM: Xid (PCI:0000:01:00): 13, pid=1234, Graphics SM Warp Exception", &code) && code == 13,
        "code after a parenthesised PCI clause");
  check(ExtractXidCode("NVRM: Xid: 79", &code) && code == 79, "code with no PCI clause");
  check(ExtractXidCode("NVRM: Xid (PCI:0000:02:00): 48, DBE", &code) && code == 48,
        "the PCI clause's own digits (0000, 02, 00) are not mistaken for the code");
  check(!ExtractXidCode("NVRM: Xid (PCI:0000:01:00), pid=1234", &code),
        "no ':' follows the PCI clause, so there is no code to read — refused, not defaulted to 0");
  check(!ExtractXidCode("no Xid word here at all", &code), "refuses when the word Xid never appears");
}

// ── pure functions: ParseKmsgRecord / VisitKmsgChunk ────────────────────────
// Byte-for-byte the same contract as host-health's own, so these fixtures are
// deliberately the same shapes runners/host-health/kernlog_test.go uses.

void testParseKmsgRecord() {
  check(ParseKmsgRecord("3,1234,567890,-;NVRM: Xid (PCI:0000:01:00): 13, pid=1234, Graphics SM Warp Exception") ==
            "NVRM: Xid (PCI:0000:01:00): 13, pid=1234, Graphics SM Warp Exception",
        "strips the priority/sequence/timestamp/flags prefix");
  check(ParseKmsgRecord("NVRM: Xid (PCI:0000:01:00): 79") == "NVRM: Xid (PCI:0000:01:00): 79",
        "a record with no ';' is returned as-is rather than dropped");
}

void testVisitKmsgChunkSplitsAndSkipsContinuations() {
  std::vector<std::string> seen;
  auto visit = [&](const std::string &m) { seen.push_back(m); };

  std::string remainder = VisitKmsgChunk(
      "3,1,1,-;NVRM: Xid (PCI:0000:01:00): 48, DBE\n"
      " SUBSYSTEM=pci\n" // continuation line: must be skipped, not counted as a record
      "3,2,2,-;unrelated line\n"
      "3,3,3,-;NVRM: Xid (PCI:0000:02:00): 79, GPU has fallen off the bus\n",
      visit);

  check(remainder.empty(), "a chunk ending in a complete record leaves no remainder");
  check(seen.size() == 3, "three top-level records, the continuation line excluded");
  if (seen.size() == 3) {
    check(seen[0] == "NVRM: Xid (PCI:0000:01:00): 48, DBE", "first record's message");
    check(seen[1] == "unrelated line", "second record's message");
    check(seen[2] == "NVRM: Xid (PCI:0000:02:00): 79, GPU has fallen off the bus", "third record's message");
  }
}

void testVisitKmsgChunkHoldsBackPartialRecord() {
  std::vector<std::string> seen;
  auto visit = [&](const std::string &m) { seen.push_back(m); };
  std::string remainder = VisitKmsgChunk("3,1,1,-;NVRM: Xid: 13\n3,2,2,-;NVRM: Xid", visit);
  check(seen.size() == 1 && seen[0] == "NVRM: Xid: 13", "the complete record is visited");
  check(remainder == "3,2,2,-;NVRM: Xid", "the trailing partial record is held back, not visited early");
}

// ── Tally ────────────────────────────────────────────────────────────────

void testTallyAccumulatesCumulativelyAcrossCalls() {
  Tally t;
  t.ObserveNvidia("NVRM: Xid (PCI:0000:01:00): 13, first");
  t.ObserveNvidia("an unrelated kernel line");
  t.ObserveNvidia("NVRM: Xid: 79");
  check(t.xidCount == 2, "two Xid lines observed across three calls");
  check(t.faultCount == 2, "the vendor-neutral counter agrees with xidCount for the NVIDIA path");
  check(t.haveLastXidCode && t.lastXidCode == 79, "the LAST code wins, matching AggLast semantics");

  // A later call that observes nothing must not reset what was already tallied
  // — this is what lets the soak's periodic drain accumulate a running total
  // exactly like throttleEvents already does.
  t.ObserveNvidia("nothing interesting");
  check(t.xidCount == 2 && t.lastXidCode == 79, "an empty observation does not reset the running tally");
}

void testTallyObserveGenericIsAnOrAcrossPatternSetsAndCountsOnceOnOverlap() {
  Tally t;
  const std::vector<std::vector<std::string>> amdPatterns = {
      {"amdgpu", "reset"},
      {"amdgpu", "uncorrectable"},
  };
  t.ObserveGeneric("amdgpu 0000:01:00.0: GPU reset begin", amdPatterns);
  t.ObserveGeneric("amdgpu 0000:01:00.0: RAS: Uncorrectable Error detected", amdPatterns);
  t.ObserveGeneric("amdgpu 0000:01:00.0: firmware version 1.2.3", amdPatterns); // matches neither set
  check(t.faultCount == 2, "two amdgpu fault lines matched, the informational firmware line did not");
  check(!t.haveLastXidCode, "the generic (AMD) path never fabricates a code — there is none to extract");

  Tally overlap;
  overlap.ObserveGeneric("amdgpu: GPU reset due to an Uncorrectable RAS event", amdPatterns);
  check(overlap.faultCount == 1, "a line matching BOTH pattern sets counts once, not twice");
}

// ── Watch: real file mechanics, mirroring host-health's own test pattern ───
// mustWrite/appendTo/setenv a temp path — a real regular file, no privilege,
// no live /dev/kmsg needed — exactly what host-health's kernlog_test.go does
// for its own kmsgSource tests.

std::string tempPath() {
  char tmpl[] = "/tmp/burnin_kmsg_test_XXXXXX";
  int fd = mkstemp(tmpl);
  if (fd < 0) {
    std::perror("mkstemp");
    std::exit(1);
  }
  close(fd);
  return std::string(tmpl);
}

void mustWrite(const std::string &path, const std::string &content) {
  std::ofstream f(path, std::ios::trunc | std::ios::binary);
  f << content;
}

void appendTo(const std::string &path, const std::string &content) {
  std::ofstream f(path, std::ios::app | std::ios::binary);
  f << content;
}

// THE TEST THE FEATURE EXISTS FOR: a Xid logged BEFORE the window must not be
// counted, and one logged DURING it must be.
void testAXidBeforeTheWindowIsNotCountedAndOneDuringItIs() {
  const std::string path = tempPath();
  // Written BEFORE Watch() opens the file: this is "the node's prior history",
  // exactly like an Xid that happened last week and is still sitting in the
  // ring buffer when this soak starts.
  mustWrite(path, "NVRM: Xid (PCI:0000:01:00): 31, this happened before the window\n");

  setenv("BURNIN_KMSG_PATH", path.c_str(), 1);
  Watch w;
  check(w.Available(), "the probe opened the fixture file");
  if (!w.Available()) {
    std::printf("  (why: %s)\n", w.Why().c_str());
    unlink(path.c_str());
    return;
  }

  // Written AFTER Watch() positioned itself at the file's then-current end —
  // this is "during the window".
  appendTo(path, "NVRM: Xid (PCI:0000:02:00): 48, this happened during the window\n");

  Tally t;
  w.Collect([&](const std::string &line) { t.ObserveNvidia(line); });

  check(t.xidCount == 1, "exactly the during-window Xid was counted, not the before-window one");
  check(t.haveLastXidCode && t.lastXidCode == 48,
        "the code belongs to the during-window Xid (48), never the before-window one (31)");
  check(!w.Dropped(), "a clean single drain reports no drop");

  unlink(path.c_str());
  unsetenv("BURNIN_KMSG_PATH");
}

// A window with nothing in it at all: available, zero count, no code.
void testAQuietWindowReportsZeroNeverAbsence() {
  const std::string path = tempPath();
  mustWrite(path, "NVRM: Xid: 13, all of this is before the window\n");
  setenv("BURNIN_KMSG_PATH", path.c_str(), 1);
  Watch w;
  check(w.Available(), "the probe opened the fixture file");

  appendTo(path, "an ordinary, uninteresting kernel line\n");
  Tally t;
  w.Collect([&](const std::string &line) { t.ObserveNvidia(line); });
  check(t.xidCount == 0, "nothing Xid-shaped arrived during the window");
  check(!t.haveLastXidCode, "no code to report when nothing was observed");

  unlink(path.c_str());
  unsetenv("BURNIN_KMSG_PATH");
}

// Two drains, matching how soak_core.cuh calls Collect() periodically: the
// SECOND drain must see only what arrived since the FIRST, not re-see anything.
void testTwoDrainsDoNotDoubleCount() {
  const std::string path = tempPath();
  mustWrite(path, "");
  setenv("BURNIN_KMSG_PATH", path.c_str(), 1);
  Watch w;
  check(w.Available(), "the probe opened the fixture file");

  appendTo(path, "NVRM: Xid: 13, first period\n");
  Tally t;
  w.Collect([&](const std::string &line) { t.ObserveNvidia(line); });
  check(t.xidCount == 1, "first drain sees the first period's Xid");

  appendTo(path, "NVRM: Xid: 79, second period\n");
  w.Collect([&](const std::string &line) { t.ObserveNvidia(line); });
  check(t.xidCount == 2, "second drain ADDS the second period's Xid rather than re-seeing the first");
  check(t.lastXidCode == 79, "the running tally's last code reflects the most recent drain");

  unlink(path.c_str());
  unsetenv("BURNIN_KMSG_PATH");
}

// A path that does not exist: the probe must say so, and never fabricate a
// zero count from "nothing to open".
void testAMissingPathIsUnavailableWithAReason() {
  setenv("BURNIN_KMSG_PATH", "/nonexistent/path/burnin-test-does-not-exist", 1);
  Watch w;
  check(!w.Available(), "a path that was never mounted is not an available source");
  check(w.Why().find("not present") != std::string::npos, "the reason names the actual failure: not present");
  unsetenv("BURNIN_KMSG_PATH");
}

void testCollectOnAnUnavailableWatchIsANoOp() {
  setenv("BURNIN_KMSG_PATH", "/nonexistent/path/burnin-test-does-not-exist", 1);
  Watch w;
  Tally t;
  bool called = false;
  w.Collect([&](const std::string &) { called = true; });
  check(!called, "Collect on an unavailable probe never invokes the callback");
  check(t.xidCount == 0 && !t.haveLastXidCode, "and the tally stays exactly as it started");
  unsetenv("BURNIN_KMSG_PATH");
}

} // namespace

int main() {
  testContainsCI();
  testMatchesAllIsAndAcrossPatternsOrAcrossSets();
  testLineHasXid();
  testExtractXidCode();
  testParseKmsgRecord();
  testVisitKmsgChunkSplitsAndSkipsContinuations();
  testVisitKmsgChunkHoldsBackPartialRecord();
  testTallyAccumulatesCumulativelyAcrossCalls();
  testTallyObserveGenericIsAnOrAcrossPatternSetsAndCountsOnceOnOverlap();
  testAXidBeforeTheWindowIsNotCountedAndOneDuringItIs();
  testAQuietWindowReportsZeroNeverAbsence();
  testTwoDrainsDoNotDoubleCount();
  testAMissingPathIsUnavailableWithAReason();
  testCollectOnAnUnavailableWatchIsANoOp();

  if (failures != 0) {
    std::printf("%d failure(s)\n", failures);
    return 1;
  }
  std::printf("kmsg_watch_test: all checks passed\n");
  return 0;
}
