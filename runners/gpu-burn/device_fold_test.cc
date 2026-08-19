// device_fold_test.cc — tests for every decision in device_fold.h: the
// allocation budget, the iteration plan, the concurrency mode and window, the
// fold, the spreads, the verdict precedence across devices, and the artifact.
//
// SPDX-License-Identifier: Apache-2.0
// Copyright the Glimmer authors.
//
// Build and run:
//
//   c++ -std=c++17 -Wall -Wextra -o /tmp/device_fold_test device_fold_test.cc && /tmp/device_fold_test
//
// runners/cxxtests_test.go does exactly that under `make test`, with no Docker,
// no CUDA toolchain and no GPU — which is the whole reason the logic is in a
// CUDA-free header: this project has no multi-GPU node, so the fold over eight
// devices, the heterogeneous board, the MIG case and the allocation mismatch are
// reachable ONLY here. A plain program with no framework, after arch_match_test.cc.

#include "device_fold.h"

#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

using namespace burnin::devices;

namespace {

int failures = 0;

void check(bool ok, const char *what) {
  if (!ok) {
    std::printf("FAIL: %s\n", what);
    ++failures;
  }
}

bool near(double a, double b) { return std::fabs(a - b) < 1e-6; }

VendorResources nvidia() { return nvidiaResources(); }

DeviceReport dev(int index, const char *bus, const char *name, const char *cc,
                 std::map<std::string, double> values, bool identityRead = true) {
  DeviceReport d;
  d.index = index;
  d.busId = bus;
  d.name = name;
  d.computeCap = cc;
  d.identityRead = identityRead;
  d.values = std::move(values);
  return d;
}

// ── budget ─────────────────────────────────────────────────────────────────

void testBudget() {
  // Absent: not set at all. Bare metal, or a test with no limits.
  check(parseBudget(nullptr, nvidia()).state == Budget::Absent, "unset variable is Absent");
  check(parseBudget("", nvidia()).state == Budget::Absent, "empty variable is Absent (the operator never emits it)");

  // The intended configuration.
  Budget b = parseBudget("memory=2Gi,nvidia.com/gpu=8,rdma/hca=1", nvidia());
  check(b.state == Budget::Established && b.count == 8, "nvidia.com/gpu=8 among other limits is 8");

  // MIG instances sum, and a MIG fleet gates the instance count.
  b = parseBudget("nvidia.com/mig-1g.5gb=4,nvidia.com/mig-2g.10gb=2", nvidia());
  check(b.state == Budget::Established && b.count == 6, "MIG profiles sum to the instance count");

  // gpu + mig together also sum: the pod was allocated both.
  b = parseBudget("nvidia.com/gpu=1,nvidia.com/mig-1g.5gb=2", nvidia());
  check(b.state == Budget::Established && b.count == 3, "gpu and mig sum");

  // A vendor-domain resource the allow-set does not name: HAMi/vGPU's
  // gpumem/gpucores, a renamed resource. Not ignored, not summed —
  // Unrecognised, naming it, so the runner exits 3 rather than iterating on a
  // guess. Summing everything under the prefix would have made this 8192.
  b = parseBudget("nvidia.com/gpu=8,nvidia.com/gpumem=8192", nvidia());
  check(b.state == Budget::Unrecognised && b.detail == "nvidia.com/gpumem",
        "a vendor-domain resource outside the allow-set is Unrecognised and named");

  // Another vendor's resources are not ours and are ignored.
  b = parseBudget("amd.com/gpu=8", nvidia());
  check(b.state == Budget::Established && b.count == 0, "another vendor's limit is not ours: allocated zero");

  // Limits set, none of them accelerators.
  b = parseBudget("cpu=4,memory=8Gi", nvidia());
  check(b.state == Budget::Established && b.count == 0, "no accelerator limit at all is an allocation of zero");

  // Malformed quantities and entries.
  check(parseBudget("nvidia.com/gpu=eight", nvidia()).state == Budget::Malformed, "non-numeric quantity is Malformed");
  check(parseBudget("nvidia.com/gpu=-1", nvidia()).state == Budget::Malformed, "negative quantity is Malformed");
  check(parseBudget("nvidia.com/gpu", nvidia()).state == Budget::Malformed, "entry without = is Malformed");
  // A trailing comma or doubled comma is tolerated: nothing is between them.
  b = parseBudget("nvidia.com/gpu=2,,", nvidia());
  check(b.state == Budget::Established && b.count == 2, "empty entries are skipped");
}

// ── plan ───────────────────────────────────────────────────────────────────

void testPlan() {
  const Budget absent{Budget::Absent, 0, ""};
  const Budget eight{Budget::Established, 8, ""};
  const Budget one{Budget::Established, 1, ""};
  const Budget zero{Budget::Established, 0, ""};
  const Budget unrec{Budget::Unrecognised, 0, "nvidia.com/gpumem"};
  const Budget malformed{Budget::Malformed, 0, "nvidia.com/gpu=x"};

  // No device: skip, whatever the budget said. The runner prints its marker.
  check(planIteration(0, absent).outcome == Plan::Skip, "no visible device is Skip (absent budget)");
  check(planIteration(0, eight).outcome == Plan::Skip, "no visible device is Skip (even with a budget)");

  // Absent budget: iterate what is visible. Bare metal.
  Plan p = planIteration(8, absent);
  check(p.outcome == Plan::Iterate && p.count == 8, "absent budget iterates every visible device");
  p = planIteration(1, absent);
  check(p.outcome == Plan::Iterate && p.count == 1, "absent budget on a Spark iterates its one device");

  // The intended configuration: budget == visible.
  p = planIteration(8, eight);
  check(p.outcome == Plan::Iterate && p.count == 8, "budget == visible iterates them all");
  p = planIteration(1, one);
  check(p.outcome == Plan::Iterate && p.count == 1, "one of one iterates");

  // THE LEAK: allocated one, shown eight. Never a silent cap at one — device 0
  // may be another pod's — and never eight. Error, naming both figures.
  p = planIteration(8, one);
  check(p.outcome == Plan::Error && p.count == 0, "budget < visible is an Error, not a cap");
  check(p.message.find("allocated 1") != std::string::npos && p.message.find("shows 8") != std::string::npos,
        "the leak message names both figures");
  check(p.message.find("NVIDIA_VISIBLE_DEVICES=all") != std::string::npos,
        "the leak message names the likely cause");

  // Allocated eight, shown four: a partial board. Error, not a fold over four.
  p = planIteration(4, eight);
  check(p.outcome == Plan::Error && p.message.find("partial board") != std::string::npos,
        "budget > visible is an Error about a partial board");

  // Allocated nothing, can see something: the leak with no request at all.
  p = planIteration(8, zero);
  check(p.outcome == Plan::Error, "budget 0 with devices visible is an Error");

  // Unrecognised and malformed budgets never iterate.
  p = planIteration(8, unrec);
  check(p.outcome == Plan::Error && p.message.find("nvidia.com/gpumem") != std::string::npos,
        "unrecognised resource is an Error naming it");
  check(planIteration(8, malformed).outcome == Plan::Error, "malformed budget is an Error");
}

// ── concurrency and window ─────────────────────────────────────────────────

void testConcurrency() {
  ConcurrencyChoice c = resolveConcurrency(nullptr, Concurrency::All);
  check(c.mode == Concurrency::All && c.recognised, "unset takes the kind's default (soak: all)");
  c = resolveConcurrency(nullptr, Concurrency::Sequential);
  check(c.mode == Concurrency::Sequential && c.recognised, "unset takes the kind's default (measurement: sequential)");
  c = resolveConcurrency("sequential", Concurrency::All);
  check(c.mode == Concurrency::Sequential && c.recognised, "sequential overrides a concurrent kind");
  c = resolveConcurrency("all", Concurrency::Sequential);
  check(c.mode == Concurrency::All && c.recognised, "all overrides a sequential kind");
  c = resolveConcurrency("ALL", Concurrency::Sequential);
  check(c.mode == Concurrency::Sequential && !c.recognised,
        "an unrecognised value falls back to the default AND says so — a silent mode change would change what the numbers mean");

  // Sequential divides the test's budget; concurrent does not. Never zero.
  check(deviceWindowSeconds(120, 8, Concurrency::Sequential) == 15, "sequential: 120 s over 8 devices is 15 s each");
  check(deviceWindowSeconds(120, 8, Concurrency::All) == 120, "concurrent: every device gets the full 120 s");
  check(deviceWindowSeconds(120, 1, Concurrency::Sequential) == 120, "one device gets the whole window either way");
  check(deviceWindowSeconds(5, 8, Concurrency::Sequential) == 1, "a squeezed window floors at 1 s, not 0");
  check(deviceWindowSeconds(120, 0, Concurrency::Sequential) == 120, "a zero count is treated as one, not divided by");
}

// ── fold ───────────────────────────────────────────────────────────────────

void testFold() {
  const std::vector<FoldRule> rules = {
      {"sustained_clock_pct", Fold::Min},  // a floor: the worst device is the smallest
      {"gpu_temp_c", Fold::Max},           // a ceiling: the worst device is the hottest
      {"miscompares", Fold::Sum},          // a counter: the board's total
      {"elapsed_s", Fold::Once},           // wall-clock: the pod's, emitted once
  };
  std::vector<DeviceReport> board = {
      dev(0, "0000:01:00.0", "NVIDIA H100", "9.0", {{"sustained_clock_pct", 95}, {"gpu_temp_c", 70}, {"miscompares", 0}, {"elapsed_s", 300}}),
      dev(1, "0000:02:00.0", "NVIDIA H100", "9.0", {{"sustained_clock_pct", 60}, {"gpu_temp_c", 84}, {"miscompares", 2}, {"elapsed_s", 300}}),
      dev(2, "0000:03:00.0", "NVIDIA H100", "9.0", {{"sustained_clock_pct", 94}, {"gpu_temp_c", 71}, {"miscompares", 0}, {"elapsed_s", 300}}),
  };

  Folded f = fold(board, rules, "sustained_clock_pct");
  check(near(f.values["sustained_clock_pct"], 60), "Min keeps the worst device's clock (60), not the mean");
  check(near(f.values["gpu_temp_c"], 84), "Max keeps the hottest device");
  check(near(f.values["miscompares"], 2), "Sum totals the board's miscompares");
  check(near(f.values["elapsed_s"], 300), "Once takes the first device's value and does not sum (900 would be wrong)");
  check(f.worstIndex == 1 && f.worstBusId == "0000:02:00.0",
        "the primary key's worst device (index 1, the 60% part) is named, with its bus id");
  check(f.homogeneityKnown && f.homogeneous, "three identical H100s are homogeneous");

  // The primary key under a Max rule names the LARGEST reading.
  Folded g = fold(board, rules, "gpu_temp_c");
  check(g.worstIndex == 1, "under a Max primary the hottest device is the worst");

  // A key one device did not report contributes nothing from that device —
  // absence is not a zero — and a key no device reported stays absent.
  std::vector<DeviceReport> partial = {
      dev(0, "a", "X", "1", {{"sustained_clock_pct", 90}}),
      dev(1, "b", "X", "1", {}),
  };
  Folded h = fold(partial, rules, "sustained_clock_pct");
  check(near(h.values["sustained_clock_pct"], 90), "a device that reported nothing does not drag the fold to 0");
  check(h.values.count("gpu_temp_c") == 0, "a key no device reported is absent from the fold (fails closed downstream)");
  check(h.worstIndex == 0, "worst device is the only one that reported");

  // Heterogeneous board: identities read and different.
  std::vector<DeviceReport> mixed = {
      dev(0, "a", "NVIDIA A100", "8.0", {{"sustained_clock_pct", 90}}),
      dev(1, "b", "NVIDIA L40S", "8.9", {{"sustained_clock_pct", 88}}),
  };
  Folded m = fold(mixed, rules, "sustained_clock_pct");
  check(m.homogeneityKnown && !m.homogeneous, "A100 beside L40S is heterogeneous, positively");

  // An unread identity: homogeneity is UNKNOWN, not false. A runner may only
  // declare what it positively established.
  std::vector<DeviceReport> unread = {
      dev(0, "a", "NVIDIA A100", "8.0", {{"sustained_clock_pct", 90}}),
      dev(1, "b", "", "", {{"sustained_clock_pct", 88}}, /*identityRead=*/false),
  };
  Folded u = fold(unread, rules, "sustained_clock_pct");
  check(!u.homogeneityKnown, "an unread identity leaves homogeneity unknown; nothing is declared");

  // Single device: worst is itself.
  std::vector<DeviceReport> spark = {dev(0, "0000:01:00.0", "NVIDIA GB10", "12.1", {{"sustained_clock_pct", 83}})};
  Folded s = fold(spark, rules, "sustained_clock_pct");
  check(s.worstIndex == 0 && s.worstBusId == "0000:01:00.0", "a single device is its own worst");
  check(near(s.values["sustained_clock_pct"], 83), "a single device's fold is its reading, unchanged");
}

// ── spreads ────────────────────────────────────────────────────────────────

void testSpreads() {
  const std::vector<FoldRule> rules = {{"sustained_clock_pct", Fold::Min}, {"achieved_tflops", Fold::Min}};
  std::vector<DeviceReport> board = {
      dev(0, "a", "H100", "9.0", {{"sustained_clock_pct", 95}, {"achieved_tflops", 800}}),
      dev(1, "b", "H100", "9.0", {{"sustained_clock_pct", 60}, {"achieved_tflops", 500}}),
      dev(2, "c", "H100", "9.0", {{"sustained_clock_pct", 94}, {"achieved_tflops", 790}}),
  };
  Folded f = fold(board, rules, "sustained_clock_pct");
  Spread sp = spreadOf(board, "sustained_clock_pct", /*absolute=*/false, f, /*mig=*/false);
  check(sp.measurable && near(sp.pct, (95.0 - 60.0) / 95.0 * 100.0), "spread is (max − min) / max × 100 across the board");
  sp = spreadOf(board, "achieved_tflops", true, f, false);
  check(sp.measurable && near(sp.pct, (800.0 - 500.0) / 800.0 * 100.0), "an absolute spread on a homogeneous board is measurable");

  // n/a is a POSITIVE claim, in each of its three cases, and the reason is kept.
  std::vector<DeviceReport> spark = {dev(0, "a", "GB10", "12.1", {{"sustained_clock_pct", 83}})};
  Folded s = fold(spark, rules, "sustained_clock_pct");
  sp = spreadOf(spark, "sustained_clock_pct", false, s, false);
  check(!sp.measurable && std::string(sp.whyNot).find("one device") != std::string::npos, "one device: n/a, nothing to spread across");

  sp = spreadOf(board, "achieved_tflops", true, f, /*mig=*/true);
  check(!sp.measurable && std::string(sp.whyNot).find("MIG") != std::string::npos, "under MIG every spread is n/a");

  std::vector<DeviceReport> mixed = {
      dev(0, "a", "A100", "8.0", {{"sustained_clock_pct", 90}, {"achieved_tflops", 300}}),
      dev(1, "b", "L40S", "8.9", {{"sustained_clock_pct", 88}, {"achieved_tflops", 180}}),
  };
  Folded m = fold(mixed, rules, "sustained_clock_pct");
  sp = spreadOf(mixed, "achieved_tflops", true, m, false);
  check(!sp.measurable && std::string(sp.whyNot).find("heterogeneous") != std::string::npos,
        "an absolute spread on a heterogeneous board is n/a — it would read ~40% on healthy hardware");
  sp = spreadOf(mixed, "sustained_clock_pct", false, m, false);
  check(sp.measurable, "a percentage-of-own-rating spread IS measurable on a heterogeneous board");

  // Unknown homogeneity blocks an absolute spread but not a relative one.
  std::vector<DeviceReport> unread = {
      dev(0, "a", "A100", "8.0", {{"achieved_tflops", 300}, {"sustained_clock_pct", 90}}),
      dev(1, "b", "", "", {{"achieved_tflops", 290}, {"sustained_clock_pct", 89}}, false),
  };
  Folded u = fold(unread, rules, "sustained_clock_pct");
  check(!spreadOf(unread, "achieved_tflops", true, u, false).measurable, "unknown homogeneity: absolute spread n/a");
  check(spreadOf(unread, "sustained_clock_pct", false, u, false).measurable, "unknown homogeneity: relative spread still measurable");

  // A key one device did not report: the spread is over the devices that did.
  std::vector<DeviceReport> partial = {
      dev(0, "a", "H100", "9.0", {{"sustained_clock_pct", 90}}),
      dev(1, "b", "H100", "9.0", {}),
      dev(2, "c", "H100", "9.0", {{"sustained_clock_pct", 45}}),
  };
  Folded p = fold(partial, rules, "sustained_clock_pct");
  sp = spreadOf(partial, "sustained_clock_pct", false, p, false);
  check(sp.measurable && near(sp.pct, 50), "spread over the two devices that reported");

  // A zero maximum admits no ratio.
  std::vector<DeviceReport> zeros = {dev(0, "a", "X", "1", {{"k", 0}}), dev(1, "b", "X", "1", {{"k", 0}})};
  Folded z = fold(zeros, {{"k", Fold::Min}}, "k");
  check(!spreadOf(zeros, "k", false, z, false).measurable, "all-zero readings admit no ratio: n/a");
}

// ── precedence ─────────────────────────────────────────────────────────────

void testPrecedence() {
  check(combineExitCodes({0, 0, 0}) == 0, "all pass is pass");
  check(combineExitCodes({0, 2, 0}) == 2, "a skip among passes is skip");
  check(combineExitCodes({0, 3, 0}) == 3, "an error among passes is error");
  check(combineExitCodes({0, 1, 0}) == 1, "a fail among passes is fail");
  // The inversion of Pair, deliberately: a device is a PART.
  check(combineExitCodes({1, 3}) == 1, "Fail outranks Error across devices — device 3's miscompare is a fact device 6's enumeration failure does not erase");
  check(combineExitCodes({3, 2}) == 3, "Error outranks Skip");
  check(combineExitCodes({2, 0}) == 2, "Skip outranks Pass");
  check(combineExitCodes({137}) == 3, "any other code is error");
  check(combineExitCodes({}) == 0, "no devices combined is pass (the caller has already refused that case)");
}

// ── artifact ───────────────────────────────────────────────────────────────

void testArtifact() {
  std::vector<DeviceReport> board = {
      dev(0, "0000:01:00.0", "NVIDIA H100", "9.0", {{"sustained_clock_pct", 95.5}}),
      dev(1, "0000:02:00.0", "Odd \"name\"\n", "9.0", {{"sustained_clock_pct", 60}}),
  };
  const std::string a = renderPerDeviceArtifact(board);
  check(a.rfind("-----BEGIN BURNIN ARTIFACT per-device.json application/json-----\n", 0) == 0,
        "the artifact opens with the fence pkg/runner recognises, name and media type");
  check(a.find("-----END BURNIN ARTIFACT-----\n") != std::string::npos, "and closes it");
  check(a.find("\"index\": 0") != std::string::npos && a.find("\"index\": 1") != std::string::npos, "one object per device");
  check(a.find("\"pciBusId\": \"0000:02:00.0\"") != std::string::npos, "bus id per device");
  check(a.find("\"sustained_clock_pct\": 95.5") != std::string::npos, "values per device");
  check(a.find("Odd \\\"name\\\"\\n") != std::string::npos, "strings are JSON-escaped");
  // The payload lines look like metrics ("key": value) — which is exactly why
  // they must be fenced. Nothing outside the fence.
  check(a.find("\"sustained_clock_pct\"") > a.find("BEGIN") && a.find("\"sustained_clock_pct\"") < a.find("END"),
        "every value line is inside the fence");

  // A non-finite value renders as null, never as nan (JSON has no NaN).
  std::vector<DeviceReport> nan = {dev(0, "a", "X", "1", {{"k", std::nan("")}})};
  check(renderPerDeviceArtifact(nan).find("\"k\": null") != std::string::npos, "NaN renders as null");
}

// ── printFold ──────────────────────────────────────────────────────────────

void testPrintFold() {
  const std::vector<FoldRule> rules = {{"sustained_clock_pct", Fold::Min}};
  std::vector<DeviceReport> board = {
      dev(0, "0000:01:00.0", "H100", "9.0", {{"sustained_clock_pct", 95}}),
      dev(1, "0000:02:00.0", "H100", "9.0", {{"sustained_clock_pct", 60}}),
  };
  Folded f = fold(board, rules, "sustained_clock_pct");
  char *buf = nullptr;
  size_t len = 0;
  std::FILE *mem = open_memstream(&buf, &len);
  printFold(mem, board, 2, 15, Concurrency::Sequential, f,
            {{"sustained_clock_spread_pct", "sustained_clock_pct", false},
             {"gemm_throughput_spread_pct", "achieved_tflops", true}},
            false);
  std::fclose(mem);
  std::string out(buf, len);
  std::free(buf);
  check(out.find("device_count=2\n") != std::string::npos, "device_count printed");
  check(out.find("devices_visible=2\n") != std::string::npos, "devices_visible printed");
  check(out.find("device_window_s=15\n") != std::string::npos, "device_window_s printed");
  check(out.find("device_concurrency=sequential\n") != std::string::npos, "device_concurrency printed");
  check(out.find("worst_device_index=1\n") != std::string::npos, "worst_device_index printed");
  check(out.find("worst_device_pci_bus_id=0000:02:00.0\n") != std::string::npos, "worst_device_pci_bus_id printed");
  check(out.find("device_homogeneous=true\n") != std::string::npos, "device_homogeneous printed when known");
  check(out.find("sustained_clock_spread_pct=36.84\n") != std::string::npos, "a measurable spread prints its number");
  check(out.find("gemm_throughput_spread_pct=n/a\n") != std::string::npos,
        "a spread whose base key no device reported prints n/a — a positive claim, never a fabricated 0");
}

}  // namespace

int main() {
  testBudget();
  testPlan();
  testConcurrency();
  testFold();
  testSpreads();
  testPrecedence();
  testArtifact();
  testPrintFold();
  if (failures != 0) {
    std::printf("%d failure(s)\n", failures);
    return 1;
  }
  std::printf("device_fold_test: all checks passed\n");
  return 0;
}
