// SPDX-License-Identifier: Apache-2.0
// Copyright the Glimmer authors.
//
// soak_series.h — the per-sample series a soak used to throw away (#546), and
// the steady-state test run over it (#547). CUDA-free and NVML-free, so it is
// unit-tested by runners/gpu-burn/soak_series_test.cc under `make test`.
//
// The soak family samples clocks, temperature, power and throttle reasons every
// few hundred milliseconds and reported only running statistics: a slow slide
// into throttling, or a clock that never settled, was visible only as its
// endpoint. This keeps the series, bounded, and emits it as a fenced artifact —
// evidence ABOUT the verdict, never part of it (pkg/runner lifts the fence out
// before the metric scanner sees it).
//
// BYTE-IDENTICAL in runners/thermal-soak, runners/gpu-burn and
// runners/power-swing; runners/thermal-soak/soak_contract_test.go enforces it.
#ifndef GLIMMER_BURNIN_SOAK_SERIES_H
#define GLIMMER_BURNIN_SOAK_SERIES_H

#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

namespace soakseries {

// Point is one sample. A field the driver did not return this sample is
// flagged absent and omitted from the artifact — never written as a zero it
// never read.
struct Point {
  float t = 0;           // seconds since this device's load started
  bool warmup = false;   // inside the warm-up window
  unsigned int smMHz = 0;
  bool haveTemp = false;
  float tempC = 0;
  bool havePower = false;
  float powerW = 0;
  bool haveMask = false;
  unsigned long long throttleMask = 0;
};

// Series keeps at most kMax points. When full it drops every other point and
// doubles its stride, so a week-long soak keeps an evenly spaced record of the
// whole run in bounded memory rather than only its last few minutes.
class Series {
 public:
  static constexpr size_t kMax = 2048;

  void add(const Point &p) {
    if (seen_++ % stride_ != 0) return;
    pts_.push_back(p);
    if (pts_.size() > kMax) {
      std::vector<Point> kept;
      kept.reserve(kMax / 2 + 1);
      for (size_t i = 0; i < pts_.size(); i += 2) kept.push_back(pts_[i]);
      pts_.swap(kept);
      stride_ *= 2;
    }
  }
  const std::vector<Point> &points() const { return pts_; }
  long stride() const { return stride_; }

 private:
  std::vector<Point> pts_;
  long stride_ = 1;
  long seen_ = 0;
};

// smClockDriftPct is the split-half steady-state test on the SM clock over the
// post-warm-up points: 100·(mean of the second half − mean of the first half)
// / max(|first|, |second|), the middle point of an odd count in neither half.
// SIGNED, unlike the source toolkit's absolute form, because a soak's
// direction matters: negative is a clock still sliding down when the test
// ended. ok is false below four post-warm-up points.
inline bool smClockDriftPct(const Series &s, double *out) {
  std::vector<double> v;
  for (const auto &p : s.points()) {
    if (!p.warmup) v.push_back(p.smMHz);
  }
  const size_t k = v.size();
  if (k < 4) return false;
  double a = 0, b = 0;
  const size_t h = k / 2;
  for (size_t i = 0; i < h; ++i) a += v[i];
  for (size_t i = k - h; i < k; ++i) b += v[i];
  a /= h;
  b /= h;
  const double den = std::fmax(std::fmax(std::fabs(a), std::fabs(b)), 1e-9);
  *out = 100.0 * (b - a) / den;
  return true;
}

// renderArtifact is the telemetry.jsonl artifact: one JSON object per point,
// every device, within budgetBytes (pkg/runner refuses an artifact over 256 KiB,
// and a refusal would lose the whole series). Each device gets an equal share
// of lines, thinned evenly, and a header line says how much was thinned, so a
// reader never mistakes a sparse record for a sparse sampler.
struct DeviceSeries {
  int index;
  const Series *series;
};

inline std::string renderArtifact(const std::vector<DeviceSeries> &devs, size_t budgetBytes,
                                  double sampleIntervalS) {
  const size_t kLineBytes = 128;  // a generous upper bound on one rendered line
  size_t maxLines = budgetBytes / kLineBytes;
  if (maxLines < devs.size() + 1) maxLines = devs.size() + 1;
  const size_t perDev = devs.empty() ? 0 : (maxLines - 1) / devs.size();

  std::string s = "-----BEGIN BURNIN ARTIFACT telemetry.jsonl application/x-ndjson-----\n";
  char buf[256];
  std::snprintf(buf, sizeof buf, "{\"sampleIntervalS\": %.3f, \"devices\": %zu, \"linesPerDeviceMax\": %zu}\n",
                sampleIntervalS, devs.size(), perDev);
  s += buf;
  for (const auto &d : devs) {
    const auto &pts = d.series->points();
    const size_t step = pts.size() > perDev && perDev > 0 ? (pts.size() + perDev - 1) / perDev : 1;
    for (size_t i = 0; i < pts.size(); i += step) {
      const Point &p = pts[i];
      int n = std::snprintf(buf, sizeof buf, "{\"device\": %d, \"t\": %.2f, \"warmup\": %s, \"smMHz\": %u",
                            d.index, p.t, p.warmup ? "true" : "false", p.smMHz);
      std::string line(buf, n > 0 ? static_cast<size_t>(n) : 0);
      if (p.haveTemp) {
        std::snprintf(buf, sizeof buf, ", \"tempC\": %.1f", p.tempC);
        line += buf;
      }
      if (p.havePower) {
        std::snprintf(buf, sizeof buf, ", \"powerW\": %.2f", p.powerW);
        line += buf;
      }
      if (p.haveMask) {
        std::snprintf(buf, sizeof buf, ", \"throttleMask\": %llu", p.throttleMask);
        line += buf;
      }
      s += line + "}\n";
    }
  }
  s += "-----END BURNIN ARTIFACT-----\n";
  return s;
}

}  // namespace soakseries

#endif  // GLIMMER_BURNIN_SOAK_SERIES_H
