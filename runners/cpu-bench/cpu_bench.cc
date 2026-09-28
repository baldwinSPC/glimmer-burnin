// SPDX-License-Identifier: Apache-2.0
// Copyright the Glimmer authors.
//
// cpu-bench — the host CPU's compute and memory bandwidth (#542).
//
// Three measurements, one after another, never overlapping each other or
// anything else on the node (on an APU, CPU load moves the GPU clock and the
// shared memory system moves both — see #534):
//
//   fmaSingleCoreGflops  8 independent double FMA chains on ONE core, the
//                        highest cpu_capacity core this process may use
//   fmaAllCoreGflops     the same kernel on every allowed core at once
//   hostStream*GBs       McCalpin's STREAM copy/scale/add/triad, one thread
//                        per allowed core, arrays 4x the largest single L3
//
// Every figure is the MEDIAN of its samples. The kernels and sizing rules are
// ported from an independent GB10 acceptance toolkit; this runner uses plain
// threads rather than OpenMP so the image carries no GPL runtime.
//
// Contract (a runner may only declare what it positively established):
//   exit 0  CPU_BENCH_PASS    everything measured
//   exit 1  CPU_BENCH_FAIL    STREAM produced wrong values: host memory or
//                             the core computing through it is corrupting data
//   exit 3  CPU_BENCH_ERROR   something stopped the measurement; unjudged
// There is no skip: every node has a CPU. A STREAM phase refused by the RAM
// headroom guard omits its metrics and says why — a gate on them then fails
// closed rather than certifying a measurement nobody took.
#include <pthread.h>
#include <sched.h>
#include <unistd.h>

#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include "cpu_topology.h"

namespace {

using Clock = std::chrono::steady_clock;
double now() { return std::chrono::duration<double>(Clock::now().time_since_epoch()).count(); }

std::string readFile(const std::string &p) {
  std::ifstream f(p);
  if (!f) return "";
  std::stringstream ss;
  ss << f.rdbuf();
  return ss.str();
}

int finish(int code, const char *marker, const std::string &msg) {
  std::fflush(stdout);
  std::printf("%s: %s\n", marker, msg.c_str());
  return code;
}

bool pinTo(int cpu) {
  cpu_set_t set;
  CPU_ZERO(&set);
  CPU_SET(cpu, &set);
  return pthread_setaffinity_np(pthread_self(), sizeof(set), &set) == 0;
}

// ── FMA ──────────────────────────────────────────────────────────────────────
constexpr int kChains = 8;
constexpr long kInner = 300000;
constexpr double kFlopsPerCall = 2.0 * kChains * kInner;

// fmaCall runs the chains and returns their sum so the compiler cannot discard
// them. Built with -ffp-contract=fast so each acc*a+b is one fused instruction.
__attribute__((noinline)) double fmaCall(double seed) {
  double acc[kChains];
  for (int c = 0; c < kChains; ++c) acc[c] = seed + c;
  const double a = 1.0000001, b = 0.9999999;
  for (long i = 0; i < kInner; ++i) {
    for (int c = 0; c < kChains; ++c) acc[c] = acc[c] * a + b;
  }
  double s = 0;
  for (int c = 0; c < kChains; ++c) s += acc[c];
  return s;
}

std::atomic<double> gSink{0};

// A reusable barrier: std::barrier is C++20, and this runner builds as C++17.
class Barrier {
 public:
  explicit Barrier(int n) : n_(n) {}
  void wait() {
    const int gen = gen_.load();
    if (count_.fetch_add(1) + 1 == n_) {
      count_.store(0);
      gen_.fetch_add(1);
    } else {
      while (gen_.load() == gen) std::this_thread::yield();
    }
  }

 private:
  const int n_;
  std::atomic<int> count_{0};
  std::atomic<int> gen_{0};
};

std::vector<double> fmaSingle(int core, double seconds) {
  std::vector<double> out;
  std::thread t([&] {
    pinTo(core);
    const double warmEnd = now() + std::min(1.0, seconds / 5);
    while (now() < warmEnd) gSink = gSink + fmaCall(1.0);
    const double end = now() + seconds;
    while (now() < end) {
      const double t0 = now();
      const double r = fmaCall(t0);
      const double t1 = now();
      gSink = gSink + r;
      if (t1 > t0) out.push_back(kFlopsPerCall / (t1 - t0) / 1e9);
    }
  });
  t.join();
  return out;
}

std::vector<double> fmaAll(const std::vector<int> &cpus, double seconds) {
  const int n = static_cast<int>(cpus.size());
  const int callsPerSample = 20;
  Barrier bar(n);
  std::atomic<bool> stop{false};
  std::vector<double> out;
  double t0 = 0;
  const double end = now() + seconds;
  std::vector<std::thread> ts;
  for (int i = 0; i < n; ++i) {
    ts.emplace_back([&, i] {
      pinTo(cpus[i]);
      double local = 0;
      for (;;) {
        bar.wait();
        if (stop.load()) break;
        if (i == 0) t0 = now();
        bar.wait();
        for (int k = 0; k < callsPerSample; ++k) local += fmaCall(1.0 + k * 1e-9);
        bar.wait();
        if (i == 0) {
          const double dt = now() - t0;
          if (dt > 0) out.push_back(n * callsPerSample * kFlopsPerCall / dt / 1e9);
          if (now() >= end) stop.store(true);
        }
      }
      gSink = gSink + local;
    });
  }
  for (auto &t : ts) t.join();
  // The first sample includes thread start-up and frequency ramp.
  if (out.size() > 3) out.erase(out.begin());
  return out;
}

// ── STREAM ───────────────────────────────────────────────────────────────────
struct StreamResult {
  std::vector<double> copy, scale, add, triad;
  bool verified = false;
  std::string why;
};

StreamResult stream(const std::vector<int> &cpus, long long arrayBytes, double seconds) {
  StreamResult r;
  const long long N = arrayBytes / 8;
  const int n = static_cast<int>(cpus.size());
  double *a = static_cast<double *>(std::aligned_alloc(64, arrayBytes));
  double *b = static_cast<double *>(std::aligned_alloc(64, arrayBytes));
  double *c = static_cast<double *>(std::aligned_alloc(64, arrayBytes));
  if (!a || !b || !c) {
    r.why = "could not allocate the STREAM arrays";
    std::free(a), std::free(b), std::free(c);
    return r;
  }
  // STREAM's classic scalar is 3, which makes every value grow each
  // iteration: a run of more than a few hundred iterations overflows to
  // infinity and fails its own check. With s = sqrt(2) - 1 one full
  // copy/scale/add/triad pass maps a -> a(2s + s^2) = a exactly, so the
  // values stay put however long the phase runs and the check stays exact.
  const double s = std::sqrt(2.0) - 1.0;
  Barrier bar(n);
  std::atomic<bool> stop{false};
  std::atomic<long> iterations{0};
  double t0 = 0;
  const double end = now() + seconds;
  std::vector<std::thread> ts;
  for (int i = 0; i < n; ++i) {
    ts.emplace_back([&, i] {
      pinTo(cpus[i]);
      const long long lo = N * i / n, hi = N * (i + 1) / n;
      // Local restrict pointers to this thread's slice. Indexing the captured
      // arrays through the closure instead lets the compiler assume any store
      // might change the pointers themselves, so it reloads them per element
      // and does not vectorise: measured on GB10, that read ~20 GB/s where the
      // memory system delivers ~100.
      const long long len = hi - lo;
      double *__restrict A = a + lo;
      double *__restrict B = b + lo;
      double *__restrict C = c + lo;
      // First touch by the owning thread, so each page lands where it is used.
      for (long long j = 0; j < len; ++j) A[j] = 1.0, B[j] = 2.0, C[j] = 0.0;
      auto timed = [&](std::vector<double> &v, double bytes, auto &&op) {
        bar.wait();
        if (i == 0) t0 = now();
        bar.wait();
        op();
        bar.wait();
        if (i == 0) {
          const double dt = now() - t0;
          if (dt > 0) v.push_back(bytes / dt / 1e9);
        }
      };
      for (;;) {
        bar.wait();
        if (stop.load()) break;
        timed(r.copy, 2.0 * 8 * N, [&] { for (long long j = 0; j < len; ++j) C[j] = A[j]; });
        timed(r.scale, 2.0 * 8 * N, [&] { for (long long j = 0; j < len; ++j) B[j] = s * C[j]; });
        timed(r.add, 3.0 * 8 * N, [&] { for (long long j = 0; j < len; ++j) C[j] = A[j] + B[j]; });
        timed(r.triad, 3.0 * 8 * N, [&] { for (long long j = 0; j < len; ++j) A[j] = B[j] + s * C[j]; });
        bar.wait();
        if (i == 0) {
          iterations.fetch_add(1);
          if (now() >= end) stop.store(true);
        }
      }
    });
  }
  for (auto &t : ts) t.join();

  // STREAM's own check: replay the recurrence on scalars and compare every
  // element. A mismatch is data corruption, not noise.
  double ea = 1.0, eb = 2.0, ec = 0.0;
  for (long k = 0; k < iterations.load(); ++k) {
    ec = ea;
    eb = s * ec;
    ec = ea + eb;
    ea = eb + s * ec;
  }
  auto close = [](double x, double e) { return std::isfinite(x) && std::fabs(x - e) <= 1e-12 * std::fabs(e); };
  r.verified = true;
  for (long long j = 0; j < N; ++j) {
    if (!close(a[j], ea) || !close(b[j], eb) || !close(c[j], ec)) {
      r.verified = false;
      char buf[160];
      std::snprintf(buf, sizeof buf, "element %lld holds a=%.17g b=%.17g c=%.17g, expected %.17g %.17g %.17g",
                    j, a[j], b[j], c[j], ea, eb, ec);
      r.why = buf;
      break;
    }
  }
  // The first iteration pays page faults and frequency ramp.
  for (auto *v : {&r.copy, &r.scale, &r.add, &r.triad}) {
    if (v->size() > 3) v->erase(v->begin());
  }
  std::free(a), std::free(b), std::free(c);
  return r;
}

// availableMemory is the smaller of MemTotal and this cgroup's limit: a pod
// limited to 4 GiB on a 128 GiB host has 4 GiB.
long long availableMemory() {
  long long total = -1;
  std::istringstream mi(readFile("/proc/meminfo"));
  std::string key;
  long long kb;
  std::string unit;
  while (mi >> key >> kb >> unit) {
    if (key == "MemTotal:") total = kb * 1024;
  }
  const std::string lim = readFile("/sys/fs/cgroup/memory.max");
  if (!lim.empty() && lim.rfind("max", 0) != 0) {
    const long long l = std::atoll(lim.c_str());
    if (l > 0 && (total < 0 || l < total)) total = l;
  }
  return total;
}

}  // namespace

int main() {
  std::setvbuf(stdout, nullptr, _IOLBF, 0);
#if defined(__x86_64__)
  // The amd64 build targets x86-64-v3. A part without FMA would die on an
  // illegal instruction, which reads as a crash rather than a reason.
  __builtin_cpu_init();
  if (!__builtin_cpu_supports("fma") || !__builtin_cpu_supports("avx2")) {
    return finish(3, "CPU_BENCH_ERROR", "this image is built for x86-64-v3 (AVX2 + FMA) and the CPU lacks it");
  }
#endif

  long duration = 60;
  if (const char *d = std::getenv("BURNIN_DURATION_SECONDS")) {
    const long v = std::atol(d);
    if (v > 0) duration = v;
  }
  const cpubench::Windows w = cpubench::splitWindows(duration);

  cpu_set_t allowed;
  CPU_ZERO(&allowed);
  if (sched_getaffinity(0, sizeof(allowed), &allowed) != 0) {
    return finish(3, "CPU_BENCH_ERROR", std::string("sched_getaffinity: ") + std::strerror(errno));
  }
  std::vector<int> cpus;
  std::vector<cpubench::CpuCap> caps;
  long long largestL3 = -1;
  for (int c = 0; c < CPU_SETSIZE; ++c) {
    if (!CPU_ISSET(c, &allowed)) continue;
    cpus.push_back(c);
    const std::string base = "/sys/devices/system/cpu/cpu" + std::to_string(c);
    const std::string cap = readFile(base + "/cpu_capacity");
    caps.push_back({c, cap.empty() ? -1 : std::atol(cap.c_str())});
    for (int idx = 0; idx < 8; ++idx) {
      const std::string cb = base + "/cache/index" + std::to_string(idx);
      if (readFile(cb + "/level").rfind("3", 0) == 0) {
        largestL3 = std::max(largestL3, cpubench::parseCacheSize(readFile(cb + "/size")));
      }
    }
  }
  if (cpus.empty()) return finish(3, "CPU_BENCH_ERROR", "no CPU in this process's affinity mask");
  const int perfCore = cpubench::pickPerformanceCore(caps);

  std::printf("cpuBenchThreads=%zu\n", cpus.size());
  std::printf("cpuBenchPerfCore=%d\n", perfCore);
  std::printf("durationRequestedS=%ld\n", duration);

  const std::vector<double> st = fmaSingle(perfCore, w.fmaSingle);
  if (!st.empty()) std::printf("fmaSingleCoreGflops=%.2f\n", cpubench::median(st));
  const std::vector<double> mt = fmaAll(cpus, w.fmaAll);
  if (!mt.empty()) std::printf("fmaAllCoreGflops=%.2f\n", cpubench::median(mt));

  const long long arrayBytes = cpubench::streamArrayBytes(largestL3);
  std::printf("streamArrayBytes=%lld\n", arrayBytes);
  if (largestL3 <= 0) std::printf("streamArrayBasis=fallback\n");
  const long long avail = availableMemory();
  if (!cpubench::headroomOK(avail, arrayBytes)) {
    std::fprintf(stderr, "cpu-bench: STREAM skipped: three %lld-byte arrays exceed three quarters of the %lld "
                         "bytes available; its metrics are omitted\n", arrayBytes, avail);
    std::printf("streamStatus=insufficient_memory\n");
  } else {
    StreamResult r = stream(cpus, arrayBytes, w.stream);
    if (!r.why.empty() && !r.verified && r.copy.empty()) {
      return finish(3, "CPU_BENCH_ERROR", r.why);
    }
    // Metrics first, verdict after: a failed check still leaves its numbers.
    if (!r.copy.empty()) std::printf("hostStreamCopyGBs=%.2f\n", cpubench::median(r.copy));
    if (!r.scale.empty()) std::printf("hostStreamScaleGBs=%.2f\n", cpubench::median(r.scale));
    if (!r.add.empty()) std::printf("hostStreamAddGBs=%.2f\n", cpubench::median(r.add));
    if (!r.triad.empty()) std::printf("hostStreamTriadGBs=%.2f\n", cpubench::median(r.triad));
    if (!r.verified) return finish(1, "CPU_BENCH_FAIL", "STREAM produced wrong values: " + r.why);
  }
  if (st.empty() || mt.empty()) return finish(3, "CPU_BENCH_ERROR", "an FMA phase completed no sample");
  std::fprintf(stderr, "cpu-bench: sink %g\n", gSink.load());  // keeps the FMA work observable
  return finish(0, "CPU_BENCH_PASS", "measured");
}
