// sysfs_clocks.h — amdgpu sysfs reading and clock judgement for clockprobe-rocm.
//
// SPDX-License-Identifier: Apache-2.0
// Copyright the Glimmer authors.
//
// This header is deliberately free of HIP and of every ROCm header, for the
// same reason clockprobe keeps NVML behind nvml_dynamic.h and memory-bw keeps
// its scanner in a plain header: the part of a runner worth unit-testing is
// never the part that needs a GPU. Everything here reads ordinary files under
// a caller-supplied root and decides things, so sysfs_clocks_test.cc compiles
// with any C++17 compiler and runs on any machine — including CI, via
// runners/cxxtests_test.go.
//
// WHY SYSFS AND NOT A LIBRARY. On AMD APUs (Strix Halo / gfx1151) the vendor
// management library is the gap, not the kernel: amdsmi reports essentially
// every monitoring field as N/A on gfx1151 while the amdgpu driver exposes
// clocks, temperature, power and utilization through sysfs the whole time
// (ROCm issue #6035). rocm-smi itself reads these same files. So this runner
// goes to the source that actually answers, which also keeps the shipped image
// free of a library dependency that would misreport on exactly the hardware
// this variant exists for.
//
// The files read, all documented sysfs ABI of the amdgpu driver:
//
//   class/drm/card*/device/vendor           PCI vendor id; 0x1002 is AMD
//   class/drm/card*/device/pp_dpm_sclk      DPM levels, "N: <mhz>Mhz [*]"
//   class/drm/card*/device/gpu_busy_percent utilization, integer percent
//   .../device/hwmon/hwmon*/temp1_input     edge temperature, millidegree C
//   .../device/hwmon/hwmon*/power1_average  board/package power, microwatt
//   .../device/hwmon/hwmon*/power1_input    fallback for power1_average

#pragma once

#include <algorithm>
#include <cctype>
#include <filesystem>
#include <fstream>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

namespace sysfsclocks {

// DpmTable is one parse of pp_dpm_sclk: the driver's own clock ladder.
//
// The TOP level is this part's rated clock for judgement purposes. It is the
// driver's declared ceiling for the part in this machine as configured, which
// is the honest denominator on an APU: there is no NVML-style nameplate boost
// to read, and a BIOS or power-mode change that lowers the ladder lowers the
// denominator with it rather than failing the part against a ceiling it was
// never allowed to reach.
struct DpmTable {
	std::vector<double> levelsMHz;
	// currentIndex is the level the driver marked active with '*', or -1 when
	// no line carried the marker (seen on some kernels between DPM updates).
	int currentIndex = -1;

	double ratedMHz() const {
		return levelsMHz.empty() ? 0.0 : *std::max_element(levelsMHz.begin(), levelsMHz.end());
	}
	double currentMHz() const {
		if (currentIndex < 0 || currentIndex >= static_cast<int>(levelsMHz.size())) return 0.0;
		return levelsMHz[static_cast<size_t>(currentIndex)];
	}
};

// ParseDpmSclk parses the pp_dpm_sclk format: one level per line,
// "0: 400Mhz" with " *" appended to the active level. The unit spelling is
// the driver's own ("Mhz"), matched case-insensitively because it has changed
// case across kernels before and a unit-spelling change must not read as a
// zero-level ladder.
inline bool ParseDpmSclk(const std::string &text, DpmTable *out, std::string *err) {
	DpmTable t;
	std::istringstream in(text);
	std::string line;
	while (std::getline(in, line)) {
		// Trim trailing CR/space so the '*' check is positional, not fragile.
		while (!line.empty() && (line.back() == '\r' || line.back() == ' ' || line.back() == '\t')) {
			line.pop_back();
		}
		if (line.empty()) continue;
		bool active = false;
		if (line.size() >= 1 && line.back() == '*') {
			active = true;
			line.pop_back();
			while (!line.empty() && (line.back() == ' ' || line.back() == '\t')) line.pop_back();
		}
		const auto colon = line.find(':');
		if (colon == std::string::npos) {
			if (err) *err = "pp_dpm_sclk line has no level index: '" + line + "'";
			return false;
		}
		const std::string value = line.substr(colon + 1);
		size_t i = 0;
		while (i < value.size() && std::isspace(static_cast<unsigned char>(value[i]))) i++;
		size_t start = i;
		while (i < value.size() &&
		       (std::isdigit(static_cast<unsigned char>(value[i])) || value[i] == '.')) {
			i++;
		}
		if (start == i) {
			if (err) *err = "pp_dpm_sclk line has no clock value: '" + line + "'";
			return false;
		}
		// The unit must be megahertz, whatever its case. A future driver that
		// reports a different unit here must fail the parse rather than be
		// silently read as MHz — a ladder in the wrong unit would misjudge
		// every node it touches.
		std::string unit = value.substr(i);
		std::transform(unit.begin(), unit.end(), unit.begin(),
		               [](unsigned char c) { return std::tolower(c); });
		while (!unit.empty() && std::isspace(static_cast<unsigned char>(unit.front()))) {
			unit.erase(unit.begin());
		}
		if (unit.rfind("mhz", 0) != 0) {
			if (err) *err = "pp_dpm_sclk level is not in Mhz: '" + line + "'";
			return false;
		}
		t.levelsMHz.push_back(std::stod(value.substr(start, i - start)));
		if (active) t.currentIndex = static_cast<int>(t.levelsMHz.size()) - 1;
	}
	if (t.levelsMHz.empty()) {
		if (err) *err = "pp_dpm_sclk is empty";
		return false;
	}
	*out = t;
	return true;
}

// Card is one discovered amdgpu device: its sysfs device directory, its hwmon
// directory (empty when the driver exposed no sensors), and its PCI address —
// the target of the `device` symlink, e.g. "0000:c1:00.0" — which is what lets
// a specific card be matched to a specific HIP device ordinal. See
// FindAmdgpuCardForDevice.
struct Card {
	std::filesystem::path device; // <root>/class/drm/cardN/device
	std::filesystem::path hwmon;  // <device>/hwmon/hwmonM, or empty
	std::string pciAddress;
};

inline std::optional<std::string> ReadFileTrim(const std::filesystem::path &p) {
	std::ifstream f(p);
	if (!f) return std::nullopt;
	std::stringstream ss;
	ss << f.rdbuf();
	std::string s = ss.str();
	while (!s.empty() && std::isspace(static_cast<unsigned char>(s.back()))) s.pop_back();
	return s;
}

// AllAmdgpuCards scans <root>/class/drm for every card whose PCI vendor is
// AMD (0x1002) and which exposes pp_dpm_sclk, in name order for determinism.
//
// The pp_dpm_sclk requirement is part of the selection, not a later failure:
// a display-only device or a partially initialised driver directory is not a
// part this probe can judge.
//
// FindAmdgpuCard (single-device callers) takes the first; multi-device
// callers use FindAmdgpuCardForDevice to match a SPECIFIC card to a specific
// HIP device ordinal by PCI address, never by this list's order — HIP's
// device enumeration and sysfs's directory order have no documented
// relationship.
inline std::vector<Card> AllAmdgpuCards(const std::filesystem::path &root) {
	const auto drm = root / "class" / "drm";
	std::error_code ec;
	std::vector<std::filesystem::path> entries;
	for (const auto &e : std::filesystem::directory_iterator(drm, ec)) {
		entries.push_back(e.path());
	}
	std::vector<Card> out;
	if (ec) return out;
	std::sort(entries.begin(), entries.end());
	for (const auto &p : entries) {
		const std::string name = p.filename().string();
		// cardN only; card0-DP-1 style connector entries also live here.
		if (name.rfind("card", 0) != 0 || name.find('-') != std::string::npos) continue;
		const auto device = p / "device";
		const auto vendor = ReadFileTrim(device / "vendor");
		if (!vendor || *vendor != "0x1002") continue;
		if (!std::filesystem::exists(device / "pp_dpm_sclk", ec)) continue;
		Card c;
		c.device = device;
		std::error_code linkEc;
		const auto resolved = std::filesystem::canonical(device, linkEc);
		if (!linkEc) c.pciAddress = resolved.filename().string();
		std::vector<std::filesystem::path> hwmons;
		for (const auto &h : std::filesystem::directory_iterator(device / "hwmon", ec)) {
			hwmons.push_back(h.path());
		}
		if (!hwmons.empty()) {
			std::sort(hwmons.begin(), hwmons.end());
			c.hwmon = hwmons.front();
		}
		out.push_back(c);
	}
	return out;
}

// FindAmdgpuCard is the single-device convenience: the first card
// AllAmdgpuCards finds, deterministic on a machine with more than one.
inline std::optional<Card> FindAmdgpuCard(const std::filesystem::path &root) {
	const auto all = AllAmdgpuCards(root);
	if (all.empty()) return std::nullopt;
	return all.front();
}

// FindAmdgpuCardForDevice matches a HIP device's own reported PCI
// domain/bus/device against every card sysfs shows — the same cross-check
// CUDA gets for free via cudaDeviceGetPCIBusId + nvmlDeviceGetHandleByPciBusId.
// Returns nullopt if nothing matches; callers treat that as "no sysfs
// telemetry for this device" rather than silently sampling the wrong one.
inline std::optional<Card> FindAmdgpuCardForDevice(const std::filesystem::path &root, int domain,
                                                   int bus, int device) {
	char want[16];
	std::snprintf(want, sizeof(want), "%04x:%02x:%02x.0", domain, bus, device);
	for (auto &c : AllAmdgpuCards(root)) {
		if (c.pciAddress == want) return c;
	}
	return std::nullopt;
}

inline std::optional<double> ReadNumber(const std::filesystem::path &p) {
	const auto s = ReadFileTrim(p);
	if (!s || s->empty()) return std::nullopt;
	try {
		size_t pos = 0;
		const double v = std::stod(*s, &pos);
		if (pos == 0) return std::nullopt;
		return v;
	} catch (...) {
		return std::nullopt;
	}
}

// ReadTempC reads hwmon temp1_input, which is millidegrees Celsius.
inline std::optional<double> ReadTempC(const Card &c) {
	if (c.hwmon.empty()) return std::nullopt;
	const auto v = ReadNumber(c.hwmon / "temp1_input");
	if (!v) return std::nullopt;
	return *v / 1000.0;
}

// ReadPowerW reads hwmon power1_average (microwatt), falling back to
// power1_input — the same order Glimmer's own sysfs probe uses.
inline std::optional<double> ReadPowerW(const Card &c) {
	if (c.hwmon.empty()) return std::nullopt;
	if (const auto v = ReadNumber(c.hwmon / "power1_average")) return *v / 1e6;
	if (const auto v = ReadNumber(c.hwmon / "power1_input")) return *v / 1e6;
	return std::nullopt;
}

inline std::optional<double> ReadBusyPct(const Card &c) {
	return ReadNumber(c.device / "gpu_busy_percent");
}

// Judgement is the runner's decision, separated from I/O so the truth table in
// sysfs_clocks_test.cc can exercise every row without hardware.
struct Judgement {
	bool pass = false;
	// floorBasis is "general" or "thermal" — which floor the run was judged
	// against, mirroring clockprobe's clock_floor_basis.
	const char *floorBasis = "general";
	double floorAppliedPct = 0.0;
	// throttleClassification: "none" at speed; "thermal" with thermal evidence;
	// "unknown" when slow without it. sysfs has no NVML-style reason mask, so
	// this vocabulary is deliberately smaller than clockprobe's and never
	// guesses a cause it cannot observe.
	const char *throttleClassification = "none";
	// idleClockLock is TRI-STATE ("true"|"false"|"unknown") for the same
	// reason clockprobe's pd_wedge_suspected is: without a temperature the
	// lock cannot be told from a thermal throttle, and without a utilization
	// reading "busy while slow" cannot be established. An unknown must never
	// collapse into an all-clear.
	const char *idleClockLock = "false";
};

// Judge applies the same fail-closed rules clockprobe applies, with the amdgpu
// idle-clock lock (ROCm issue #5750) in the role the PD wedge plays on GB10:
// the part reports healthy — busy, cool — and simply runs at a fraction of its
// ladder's top. The signature is slow + cool + busy.
//
// The lenient thermal floor is bought ONLY by thermal evidence: a part whose
// temperature could not be read is judged on the strict general floor, because
// an unknown measurement must never buy leniency.
inline Judgement Judge(double sustainedPct,
                       bool tempKnown, double maxTempC,
                       bool busyKnown, double meanBusyPct,
                       double floorPct, double thermalFloorPct, double thermalTempC) {
	Judgement j;
	const bool thermal = tempKnown && maxTempC >= thermalTempC;

	j.floorAppliedPct = floorPct;
	if (thermal) {
		j.floorAppliedPct = std::min(floorPct, thermalFloorPct);
		j.floorBasis = "thermal";
	}
	j.pass = sustainedPct >= j.floorAppliedPct;

	if (thermal) {
		j.throttleClassification = "thermal";
	} else if (!j.pass) {
		j.throttleClassification = "unknown";
	}

	if (j.pass) {
		j.idleClockLock = "false";
	} else if (!tempKnown || thermal) {
		// Hot and slow is a throttle, not a lock; unknown temperature cannot
		// distinguish the two.
		j.idleClockLock = thermal ? "false" : "unknown";
	} else if (!busyKnown) {
		j.idleClockLock = "unknown";
	} else {
		// Slow, cool, and the driver says the part is working: the #5750
		// signature. 80% is a deliberately high bar — the load this runner
		// applies saturates a healthy part, so anything busier than that
		// while cool and slow is the lock, not an idle machine.
		j.idleClockLock = meanBusyPct >= 80.0 ? "true" : "unknown";
	}
	return j;
}

} // namespace sysfsclocks
