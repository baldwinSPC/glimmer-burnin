// sysfs_clocks_test.cc — unit tests for sysfs_clocks.h.
//
// SPDX-License-Identifier: Apache-2.0
// Copyright the Glimmer authors.
//
// Plain C++17, no framework, no HIP, no hardware — compiled and run by
// runners/cxxtests_test.go with:
//
//   c++ -std=c++17 -O1 -Wall -Wextra -o sysfs_clocks_test sysfs_clocks_test.cc
//
// It builds a fake sysfs tree in a temp directory for the discovery and reader
// tests, and drives Judge through the truth table the runner's verdict rests
// on. Every row that matters is one the GPU-free header can be wrong about:
// the pp_dpm_sclk grammar, the vendor filter, the millidegree/microwatt
// scaling, and the fail-closed rules around unknown temperature and unknown
// utilization.

#include "sysfs_clocks.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>

namespace fs = std::filesystem;
using namespace sysfsclocks;

namespace {

int failures = 0;

void check(bool ok, const char *what) {
	if (!ok) {
		std::fprintf(stderr, "FAIL: %s\n", what);
		failures++;
	}
}

void write(const fs::path &p, const std::string &content) {
	fs::create_directories(p.parent_path());
	std::ofstream f(p);
	f << content;
}

void testParseDpmSclk() {
	DpmTable t;
	std::string err;

	// The ordinary Strix Halo shape: several levels, star on the active one.
	check(ParseDpmSclk("0: 400Mhz\n1: 1100Mhz\n2: 2900Mhz *\n", &t, &err),
	      "parse of a normal ladder");
	check(t.levelsMHz.size() == 3, "three levels parsed");
	check(t.currentIndex == 2, "star marks level 2");
	check(t.ratedMHz() == 2900.0, "rated is the top level");
	check(t.currentMHz() == 2900.0, "current follows the star");

	// Star mid-ladder, trailing CRLF, and the unit's case shifted — all seen
	// in the wild across kernel versions.
	check(ParseDpmSclk("0: 400MHz *\r\n1: 2900MHz\r\n", &t, &err),
	      "parse with CRLF and MHz casing");
	check(t.currentIndex == 0, "star on the lowest level");
	check(t.ratedMHz() == 2900.0, "rated unaffected by star position");

	// No star at all: levels are still a ladder, current is unknown.
	check(ParseDpmSclk("0: 400Mhz\n1: 2900Mhz\n", &t, &err), "parse without a star");
	check(t.currentIndex == -1, "no star means unknown current");
	check(t.currentMHz() == 0.0, "unknown current reads as zero");

	// Garbage must fail the parse, not read as a ladder.
	check(!ParseDpmSclk("", &t, &err), "empty input refused");
	check(!ParseDpmSclk("not a ladder\n", &t, &err), "no-colon line refused");
	check(!ParseDpmSclk("0: fastMhz\n", &t, &err), "no-number line refused");
	// A future driver reporting a different unit must fail rather than be
	// silently misread as MHz.
	check(!ParseDpmSclk("0: 400Ghz\n", &t, &err), "wrong unit refused");
}

void testDiscoveryAndReaders() {
	const fs::path root = fs::temp_directory_path() / "clockprobe-rocm-test-sysfs";
	fs::remove_all(root);

	// card0: an NVIDIA device — must be passed over, not merely tolerated.
	write(root / "class/drm/card0/device/vendor", "0x10de\n");
	// card0-DP-1: a connector entry whose name would match a naive "card*".
	write(root / "class/drm/card0-DP-1/status", "connected\n");
	// card1: the AMD APU, with a ladder, utilization and sensors.
	write(root / "class/drm/card1/device/vendor", "0x1002\n");
	write(root / "class/drm/card1/device/pp_dpm_sclk", "0: 400Mhz\n1: 2900Mhz *\n");
	write(root / "class/drm/card1/device/gpu_busy_percent", "97\n");
	write(root / "class/drm/card1/device/hwmon/hwmon3/temp1_input", "91000\n");
	write(root / "class/drm/card1/device/hwmon/hwmon3/power1_average", "119000000\n");

	auto card = FindAmdgpuCard(root);
	check(card.has_value(), "discovery finds the AMD card");
	if (card) {
		check(card->device.filename() == "device", "device dir returned");
		check(card->device.parent_path().filename() == "card1", "card1 selected, card0 skipped");

		const auto temp = ReadTempC(*card);
		check(temp.has_value() && *temp == 91.0, "millidegree temperature scaled to C");
		const auto power = ReadPowerW(*card);
		check(power.has_value() && *power == 119.0, "microwatt power scaled to W");
		const auto busy = ReadBusyPct(*card);
		check(busy.has_value() && *busy == 97.0, "gpu_busy_percent read");
	}

	// A tree with no AMD accelerator answers nothing — the runner's skip path.
	const fs::path empty = root / "no-amd";
	write(empty / "class/drm/card0/device/vendor", "0x10de\n");
	check(!FindAmdgpuCard(empty).has_value(), "no AMD card means no discovery");

	// An AMD display device without pp_dpm_sclk is not a judgeable part.
	const fs::path noladder = root / "no-ladder";
	write(noladder / "class/drm/card0/device/vendor", "0x1002\n");
	check(!FindAmdgpuCard(noladder).has_value(), "AMD device without a ladder is passed over");

	fs::remove_all(root);
}

void testJudge() {
	const double floor = 60.0, thermalFloor = 40.0, thermalTemp = 90.0;

	// Fast and cool: pass on the general floor, nothing suspected.
	auto j = Judge(95.0, true, 70.0, true, 99.0, floor, thermalFloor, thermalTemp);
	check(j.pass, "fast+cool passes");
	check(std::strcmp(j.floorBasis, "general") == 0, "fast+cool judged on the general floor");
	check(std::strcmp(j.throttleClassification, "none") == 0, "fast+cool classifies none");
	check(std::strcmp(j.idleClockLock, "false") == 0, "fast+cool suspects nothing");

	// Slow and hot, above the lenient floor: a thermal throttle doing its job.
	j = Judge(50.0, true, 95.0, true, 99.0, floor, thermalFloor, thermalTemp);
	check(j.pass, "slow+hot above the thermal floor passes");
	check(std::strcmp(j.floorBasis, "thermal") == 0, "thermal evidence buys the thermal floor");
	check(std::strcmp(j.throttleClassification, "thermal") == 0, "slow+hot classifies thermal");
	check(std::strcmp(j.idleClockLock, "false") == 0, "hot and slow is a throttle, not a lock");

	// Slow and hot, below even the lenient floor: fail, still thermal.
	j = Judge(30.0, true, 95.0, true, 99.0, floor, thermalFloor, thermalTemp);
	check(!j.pass, "slow+hot below the thermal floor fails");
	check(std::strcmp(j.floorBasis, "thermal") == 0, "failure judged on the thermal floor");

	// Slow, cool, busy: the #5750 idle-clock-lock signature.
	j = Judge(15.0, true, 55.0, true, 97.0, floor, thermalFloor, thermalTemp);
	check(!j.pass, "slow+cool fails");
	check(std::strcmp(j.floorBasis, "general") == 0, "no thermal evidence, strict floor");
	check(std::strcmp(j.throttleClassification, "unknown") == 0,
	      "slow without thermal evidence classifies unknown, never guesses");
	check(std::strcmp(j.idleClockLock, "true") == 0, "slow+cool+busy is the lock signature");

	// Slow and cool but utilization unreadable: the lock cannot be established.
	j = Judge(15.0, true, 55.0, false, 0.0, floor, thermalFloor, thermalTemp);
	check(!j.pass, "slow+cool without busy still fails");
	check(std::strcmp(j.idleClockLock, "unknown") == 0,
	      "no utilization reading means unknown, not true and not false");

	// Slow with no temperature at all: strict floor, and unknown everywhere —
	// an unknown measurement never buys leniency and never reads as all-clear.
	j = Judge(50.0, false, 0.0, true, 99.0, floor, thermalFloor, thermalTemp);
	check(!j.pass, "unknown temperature is judged on the strict floor");
	check(std::strcmp(j.floorBasis, "general") == 0, "no thermal evidence, no thermal floor");
	check(std::strcmp(j.idleClockLock, "unknown") == 0,
	      "without a temperature the lock cannot be told from a throttle");

	// Slow, cool, and only moderately busy: suspicious but not the signature.
	j = Judge(15.0, true, 55.0, true, 40.0, floor, thermalFloor, thermalTemp);
	check(std::strcmp(j.idleClockLock, "unknown") == 0,
	      "cool+slow but not busy stays unknown");
}

} // namespace

int main() {
	testParseDpmSclk();
	testDiscoveryAndReaders();
	testJudge();
	if (failures > 0) {
		std::fprintf(stderr, "%d check(s) failed\n", failures);
		return 1;
	}
	std::printf("sysfs_clocks_test: all checks passed\n");
	return 0;
}
