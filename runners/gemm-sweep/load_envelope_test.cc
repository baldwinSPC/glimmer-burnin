// SPDX-License-Identifier: Apache-2.0
// Copyright the Glimmer authors.
//
// Unit tests for load_envelope.h, compiled and run by runners/cxxtests_test.go.
#include "load_envelope.h"

#include <cstdio>

static int failures = 0;
#define CHECK(cond, ...)                                        \
  do {                                                          \
    if (!(cond)) {                                              \
      std::fprintf(stderr, "FAIL %s:%d: ", __FILE__, __LINE__); \
      std::fprintf(stderr, __VA_ARGS__);                        \
      std::fprintf(stderr, "\n");                               \
      failures++;                                               \
    }                                                           \
  } while (0)

int main() {
  loadenv::Envelope e;
  std::map<std::string, double> v;
  e.store(&v);
  CHECK(v.empty(), "an envelope that read nothing stored %zu fields", v.size());

  e.addClock(2400);
  e.addClock(2000);
  e.addTemp(70);
  e.addTemp(78);
  e.addTemp(74);
  e.addPower(80.5);
  e.addPower(62.0);
  e.addLimit(100);
  e.addLimit(90);
  e.store(&v);
  CHECK(v["sm_clock_mhz"] == 2200, "mean clock %g", v["sm_clock_mhz"]);
  CHECK(v["gpu_temp_c"] == 78, "peak temp %g", v["gpu_temp_c"]);
  CHECK(v["power_draw_w"] == 80.5, "peak power %g", v["power_draw_w"]);
  CHECK(v["enforced_power_limit_w"] == 90, "lowest limit %g", v["enforced_power_limit_w"]);

  // Only the clock read: temperature and power must stay absent.
  loadenv::Envelope c;
  c.addClock(1000);
  std::map<std::string, double> cv;
  c.store(&cv);
  CHECK(cv.size() == 1 && cv.count("gpu_temp_c") == 0, "absent fields were stored");

  CHECK(loadenv::reasonLabels(0) == "none", "no reasons");
  CHECK(loadenv::reasonLabels(0x4 | 0x40) == "swPowerCap,hwThermalSlowdown", "labels %s",
        loadenv::reasonLabels(0x44).c_str());
  loadenv::Envelope m;
  m.addMask(0x4);
  m.addMask(0x40);
  CHECK(m.mask == 0x44 && m.haveMask, "mask union");

  if (failures) {
    std::fprintf(stderr, "%d failure(s)\n", failures);
    return 1;
  }
  std::printf("load_envelope ok\n");
  return 0;
}
