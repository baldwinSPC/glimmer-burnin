// SPDX-License-Identifier: Apache-2.0
// Copyright the Glimmer authors.
//
// cpu_topology.h — every decision cpu-bench makes that could produce a wrong
// verdict, kept free of any syscall so runners/cpu-bench/cpu_topology_test.cc
// can test it exhaustively under `make test`.
//
// The sizing and core-selection rules are ported from an independent GB10
// acceptance toolkit, which measured why each one matters on real hardware:
// on GB10 cpu0 is an EFFICIENCY core (the ten Cortex-X925 read cpu_capacity
// 997-1024, the ten Cortex-A725 718-731), and L3 is two asymmetric instances
// (8 MiB and 16 MiB), so "cpu0" and "the L3 size" are both wrong answers.
#ifndef GLIMMER_BURNIN_CPU_TOPOLOGY_H
#define GLIMMER_BURNIN_CPU_TOPOLOGY_H

#include <algorithm>
#include <cctype>
#include <string>
#include <utility>
#include <vector>

namespace cpubench {

// parseCpuList parses the kernel's cpulist format, "0-9,12,14-15". Returns
// false on anything malformed rather than guessing.
inline bool parseCpuList(const std::string &s, std::vector<int> *out) {
  out->clear();
  size_t i = 0;
  const size_t n = s.size();
  auto num = [&](long *v) {
    if (i >= n || !std::isdigit(static_cast<unsigned char>(s[i]))) return false;
    long x = 0;
    while (i < n && std::isdigit(static_cast<unsigned char>(s[i]))) {
      x = x * 10 + (s[i] - '0');
      if (x > 1000000) return false;
      ++i;
    }
    *v = x;
    return true;
  };
  while (i < n) {
    if (s[i] == '\n' || s[i] == ' ') {
      ++i;
      continue;
    }
    long a = 0, b = 0;
    if (!num(&a)) return false;
    b = a;
    if (i < n && s[i] == '-') {
      ++i;
      if (!num(&b) || b < a) return false;
    }
    for (long c = a; c <= b; ++c) out->push_back(static_cast<int>(c));
    if (i < n && s[i] == ',') ++i;
  }
  return !out->empty();
}

// A CPU and its cpu_capacity, or -1 where the kernel publishes none.
struct CpuCap {
  int cpu;
  long capacity;
};

// pickPerformanceCore is the single-thread measurement's core: the highest
// cpu_capacity among the CPUs this process may run on, ties to the lowest id.
// With no capacity published anywhere, every core is the same and the lowest
// allowed id is as good as any.
inline int pickPerformanceCore(const std::vector<CpuCap> &cpus) {
  int best = -1;
  long bestCap = -2;
  for (const auto &c : cpus) {
    if (c.capacity > bestCap || (c.capacity == bestCap && c.cpu < best)) {
      best = c.cpu;
      bestCap = c.capacity;
    }
  }
  return best;
}

// parseCacheSize reads sysfs's cache "size" ("16384K", "8M", "1G"). Returns -1
// for anything it cannot read.
inline long long parseCacheSize(const std::string &raw) {
  std::string s;
  for (char ch : raw) {
    if (!std::isspace(static_cast<unsigned char>(ch))) s += ch;
  }
  if (s.empty()) return -1;
  long long mult = 1;
  const char last = static_cast<char>(std::toupper(static_cast<unsigned char>(s.back())));
  if (last == 'K') mult = 1024;
  if (last == 'M') mult = 1024LL * 1024;
  if (last == 'G') mult = 1024LL * 1024 * 1024;
  if (mult != 1) s.pop_back();
  if (s.empty()) return -1;
  long long v = 0;
  for (char ch : s) {
    if (!std::isdigit(static_cast<unsigned char>(ch))) return -1;
    v = v * 10 + (ch - '0');
  }
  return v * mult;
}

// kStreamMultiple: each STREAM array is this many times the LARGEST SINGLE L3
// instance, so no array fits in any one cluster's cache. On GB10 that is
// 4 x 16 MiB = 64 MiB per array. Not the SUM of instances: a cluster reads
// through its own L3, and the larger one is the one to defeat.
constexpr long long kStreamMultiple = 4;
// kFallbackArrayBytes is used only where no L3 is published at all, and the
// runner says so in its output.
constexpr long long kFallbackArrayBytes = 256LL * 1024 * 1024;

inline long long streamArrayBytes(long long largestL3) {
  long long b = largestL3 > 0 ? kStreamMultiple * largestL3 : kFallbackArrayBytes;
  return b - b % 64;  // whole cache lines of whole doubles
}

// headroomOK is the guard that keeps a memory benchmark from becoming a memory
// incident: three arrays may use at most three quarters of the memory this
// process can actually get (the smaller of MemTotal and a cgroup limit).
inline bool headroomOK(long long availableBytes, long long arrayBytes) {
  if (availableBytes <= 0 || arrayBytes <= 0) return false;
  return 3.0L * arrayBytes <= 0.75L * availableBytes;
}

// median of a copy; NaN-free input is the caller's duty.
inline double median(std::vector<double> v) {
  if (v.empty()) return 0.0;
  std::sort(v.begin(), v.end());
  const size_t m = v.size() / 2;
  return v.size() % 2 ? v[m] : 0.5 * (v[m - 1] + v[m]);
}

// Phase windows: the test's duration is split 20/30/50 between the
// single-core FMA, the all-core FMA and STREAM, which needs the most because
// each iteration moves hundreds of megabytes. A window below one second per
// phase measures nothing worth reporting.
struct Windows {
  double fmaSingle, fmaAll, stream;
};
inline Windows splitWindows(long durationSeconds) {
  const double d = static_cast<double>(std::max(10L, durationSeconds));
  return {0.2 * d, 0.3 * d, 0.5 * d};
}

}  // namespace cpubench

#endif  // GLIMMER_BURNIN_CPU_TOPOLOGY_H
