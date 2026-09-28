// SPDX-License-Identifier: Apache-2.0
// Copyright the Glimmer authors.
//
// Unit tests for soak_series.h, compiled and run by runners/cxxtests_test.go.
// It lives in gpu-burn, not thermal-soak, because a runner directory holding a
// .cc file cannot also hold Go files, and thermal-soak holds its contract test.
#include "soak_series.h"

#include <cstdio>
#include <cstdlib>
#include <string>

static int failures = 0;
#define CHECK(cond, ...)                              \
  do {                                                \
    if (!(cond)) {                                    \
      std::fprintf(stderr, "FAIL %s:%d: ", __FILE__, __LINE__); \
      std::fprintf(stderr, __VA_ARGS__);              \
      std::fprintf(stderr, "\n");                     \
      failures++;                                     \
    }                                                 \
  } while (0)

static soakseries::Point pt(float t, unsigned sm, bool warm = false) {
  soakseries::Point p;
  p.t = t;
  p.smMHz = sm;
  p.warmup = warm;
  return p;
}

int main() {
  // A series longer than kMax stays bounded, evenly thinned, and still spans
  // the whole run.
  soakseries::Series s;
  const int total = 10000;
  for (int i = 0; i < total; ++i) s.add(pt(static_cast<float>(i), 2400));
  CHECK(s.points().size() <= soakseries::Series::kMax, "size %zu", s.points().size());
  CHECK(s.points().size() > soakseries::Series::kMax / 4, "thinned too far: %zu", s.points().size());
  CHECK(s.points().front().t == 0.0f, "first point t=%f", s.points().front().t);
  CHECK(s.points().back().t > total * 0.9f, "last point t=%f does not reach the end", s.points().back().t);

  // Drift: a clock sliding from 2400 to 2000 is negative; a flat one is 0;
  // warm-up points are excluded; under four points is not judged.
  soakseries::Series slide;
  for (int i = 0; i < 3; ++i) slide.add(pt(i, 900, /*warm=*/true));
  for (int i = 0; i < 8; ++i) slide.add(pt(3 + i, i < 4 ? 2400 : 2000));
  double d = 0;
  CHECK(soakseries::smClockDriftPct(slide, &d), "drift not computed");
  CHECK(std::abs(d - (-100.0 * 400 / 2400)) < 1e-9, "drift %f", d);

  soakseries::Series flat;
  for (int i = 0; i < 6; ++i) flat.add(pt(i, 2450));
  CHECK(soakseries::smClockDriftPct(flat, &d) && d == 0.0, "flat drift %f", d);

  soakseries::Series tiny;
  for (int i = 0; i < 3; ++i) tiny.add(pt(i, 2450));
  CHECK(!soakseries::smClockDriftPct(tiny, &d), "three points were judged");

  // Artifact: fenced, within budget, absent fields omitted rather than zeroed.
  soakseries::Point withTemp = pt(1, 2400);
  withTemp.haveTemp = true;
  withTemp.tempC = 71.5f;
  soakseries::Series one;
  one.add(withTemp);
  one.add(pt(2, 2401));
  std::vector<soakseries::DeviceSeries> devs{{0, &one}, {1, &s}};
  const size_t budget = 200 * 1024;
  std::string a = soakseries::renderArtifact(devs, budget, 0.25);
  CHECK(a.rfind("-----BEGIN BURNIN ARTIFACT telemetry.jsonl application/x-ndjson-----\n", 0) == 0, "no fence");
  CHECK(a.size() <= budget + 256, "artifact %zu bytes over budget", a.size());
  CHECK(a.find("\"tempC\": 71.5") != std::string::npos, "temperature missing");
  CHECK(a.find("{\"device\": 0, \"t\": 2.00, \"warmup\": false, \"smMHz\": 2401}") != std::string::npos,
        "a point with no temperature must omit the field:\n%s", a.substr(0, 400).c_str());
  CHECK(a.find("\"device\": 1") != std::string::npos, "second device missing");

  if (failures) {
    std::fprintf(stderr, "%d failure(s)\n", failures);
    return 1;
  }
  std::printf("soak_series ok\n");
  return 0;
}
