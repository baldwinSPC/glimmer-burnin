// SPDX-License-Identifier: Apache-2.0
// Copyright the Glimmer authors.
//
// Unit tests for cpu_topology.h, compiled and run by runners/cxxtests_test.go.
#include "cpu_topology.h"

#include <cstdio>

static int failures = 0;
#define CHECK(cond, ...)                                          \
  do {                                                            \
    if (!(cond)) {                                                \
      std::fprintf(stderr, "FAIL %s:%d: ", __FILE__, __LINE__);   \
      std::fprintf(stderr, __VA_ARGS__);                          \
      std::fprintf(stderr, "\n");                                 \
      failures++;                                                 \
    }                                                             \
  } while (0)

int main() {
  using namespace cpubench;
  std::vector<int> v;
  CHECK(parseCpuList("0-19\n", &v) && v.size() == 20 && v.back() == 19, "0-19");
  CHECK(parseCpuList("0-3,8,10-11", &v) && v.size() == 7 && v[4] == 8, "mixed list");
  CHECK(!parseCpuList("", &v), "empty accepted");
  CHECK(!parseCpuList("3-1", &v), "descending accepted");
  CHECK(!parseCpuList("0-", &v), "open range accepted");
  CHECK(!parseCpuList("a", &v), "letters accepted");

  // GB10: A725 at 718-731 on cpus 0-9, X925 at 997-1024 on 10-19, max on 19.
  std::vector<CpuCap> gb10;
  for (int c = 0; c < 10; ++c) gb10.push_back({c, 718 + c});
  for (int c = 10; c < 20; ++c) gb10.push_back({c, 997 + (c - 10) * 3});
  CHECK(pickPerformanceCore(gb10) == 19, "GB10 perf core %d, want 19", pickPerformanceCore(gb10));
  // Only the efficiency cluster allowed by the cpuset: the best of those.
  std::vector<CpuCap> small(gb10.begin(), gb10.begin() + 10);
  CHECK(pickPerformanceCore(small) == 9, "restricted cpuset core %d", pickPerformanceCore(small));
  // Ties go to the lowest id; no capacities means the lowest allowed id.
  CHECK(pickPerformanceCore({{5, 1024}, {3, 1024}}) == 3, "tie");
  CHECK(pickPerformanceCore({{7, -1}, {4, -1}, {6, -1}}) == 4, "no capacities");

  CHECK(parseCacheSize("16384K\n") == 16LL * 1024 * 1024, "16384K");
  CHECK(parseCacheSize("8M") == 8LL * 1024 * 1024, "8M");
  CHECK(parseCacheSize("") == -1 && parseCacheSize("12Q") == -1 && parseCacheSize("K") == -1, "garbage");

  CHECK(streamArrayBytes(16LL * 1024 * 1024) == 64LL * 1024 * 1024, "GB10 array 64 MiB");
  CHECK(streamArrayBytes(-1) == kFallbackArrayBytes, "fallback");
  CHECK(streamArrayBytes(1000) % 64 == 0, "alignment");

  const long long gib = 1024LL * 1024 * 1024;
  CHECK(headroomOK(128 * gib, 64LL * 1024 * 1024), "GB10 has room");
  CHECK(!headroomOK(1 * gib, 300LL * 1024 * 1024), "900 MiB of 1 GiB allowed");
  CHECK(headroomOK(4 * gib, 1 * gib), "exactly 3/4 must pass");
  CHECK(!headroomOK(0, 1), "unknown memory must refuse");

  CHECK(median({3, 1, 2}) == 2 && median({4, 1, 3, 2}) == 2.5, "median");
  Windows w = splitWindows(100);
  CHECK(w.fmaSingle == 20 && w.fmaAll == 30 && w.stream == 50, "split");
  CHECK(splitWindows(1).stream == 5, "a tiny duration is clamped to 10 s");

  if (failures) {
    std::fprintf(stderr, "%d failure(s)\n", failures);
    return 1;
  }
  std::printf("cpu_topology ok\n");
  return 0;
}
