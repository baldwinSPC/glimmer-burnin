// device_fold.h — how a runner turns N per-device readings into ONE Node
// verdict, and how it decides which devices it may iterate at all.
//
// SPDX-License-Identifier: Apache-2.0
// Copyright the Glimmer authors.
//
// Host-only, and deliberately free of CUDA and HIP, so a plain C++ compiler
// exercises every decision here with no GPU anywhere in sight
// (device_fold_test.cc; runners/cxxtests_test.go compiles and runs it under
// `make test`). BYTE-IDENTICAL across every runner that carries it — each
// runner is its own Docker build context and COPY cannot leave one, so the file
// is duplicated and runners/sharedsource_test.go refuses drift. Edit it in one
// place and copy.
//
// The design it implements is docs/dev/multi-device.md. The one sentence:
//
//   A Node verdict describes EVERY accelerator on the node, gated on the WORST;
//   a single-device measurement on a multi-device node is a bug.
//
// Everything below is that sentence applied to a specific decision. Nothing
// here touches a device: the runner enumerates, sets, launches and reads; this
// header decides how many it may touch, how long each gets, how the readings
// combine, which device the verdict names, and what the stdout says.
//
// The vocabulary is the registry's (pkg/contract/metrics.go). Keys are the
// runner's raw snake_case, which pkg/runner canonicalises: device_count →
// deviceCount, worst_device_pci_bus_id → worstDevicePciBusId, and so on.

#ifndef BURNIN_DEVICE_FOLD_H
#define BURNIN_DEVICE_FOLD_H

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <vector>

namespace burnin {
namespace devices {

// ── Direction ──────────────────────────────────────────────────────────────

// Fold is how one raw key combines across the devices of ONE window. The
// direction is the metric's, not the runner's: it must agree with the
// registry's Aggregation for a Min/Max/Sum metric, and runners/devicefold_test.go
// reads each runner's table to check that it does. Once marks a key that is
// emitted a single time and never folded — a wall-clock figure describes the
// pod, an identity label keeps the first device's meaning.
enum class Fold { Min, Max, Sum, Once };

struct FoldRule {
  const char *key;
  Fold fold;
};

// ── Iteration mode ─────────────────────────────────────────────────────────

// Concurrency is the iteration mode. Sequential gives each device
// duration/deviceCount alone; All runs every device at once for the full
// duration. The DEFAULT is the kind's — a soak heats the board and is
// concurrent; a measurement isolates a device and is sequential — and
// BURNIN_DEVICE_CONCURRENCY overrides it.
enum class Concurrency { Sequential, All };

struct ConcurrencyChoice {
  Concurrency mode;
  // recognised is false when the variable was set to something that is not
  // "all" or "sequential"; the kind's default is used and the runner should
  // say so, because a typo that silently changed the mode would change what
  // the numbers mean with nothing in the result recording it.
  bool recognised;
};

inline ConcurrencyChoice resolveConcurrency(const char *env, Concurrency kindDefault) {
  if (env == nullptr || *env == '\0') return {kindDefault, true};
  if (std::strcmp(env, "all") == 0) return {Concurrency::All, true};
  if (std::strcmp(env, "sequential") == 0) return {Concurrency::Sequential, true};
  return {kindDefault, false};
}

inline const char *concurrencyName(Concurrency c) {
  return c == Concurrency::All ? "all" : "sequential";
}

// deviceWindowSeconds is the load window ONE device gets. The whole iteration
// must fit inside BURNIN_DURATION_SECONDS — the operator's pod deadline is
// derived from it and does not grow with the board — so sequential mode
// divides, concurrent mode does not. Never below one second: a zero window is
// a measurement that never happened, and saying "1" is at least honest about
// having been squeezed.
inline long deviceWindowSeconds(long durationSeconds, int deviceCount, Concurrency c) {
  if (deviceCount < 1) deviceCount = 1;
  long w = c == Concurrency::All ? durationSeconds : durationSeconds / deviceCount;
  return w < 1 ? 1 : w;
}

// ── Allocation budget ──────────────────────────────────────────────────────

// Budget is what BURNIN_RESOURCE_LIMITS says this pod was ALLOCATED, read
// against the runner's own vendor. The operator injects the pod's limits
// verbatim ("nvidia.com/gpu=8,rdma/hca=1"); it interprets nothing, because
// which name is the accelerator is vendor knowledge and this image is that
// vendor's.
struct Budget {
  enum State {
    // The variable was not set at all: bare metal (where "all" IS the
    // allocation) or a test with no limits. The runner iterates every visible
    // device and says so. Never returned for a variable that is set.
    Absent,
    // Every resource under the vendor's domain matched the allow-set; count
    // is their sum.
    Established,
    // A resource under the vendor's domain was NOT in the allow-set — a MIG
    // profile the set does not name, a vGPU stack's gpumem/gpucores, a
    // renamed resource. The allocation cannot be established, and a runner
    // may only declare what it positively established: exit 3.
    Unrecognised,
    // The variable was set but a matched quantity did not parse as a whole
    // number. Also exit 3.
    Malformed,
  };
  State state;
  long count;          // Established only
  std::string detail;  // the offending name or value, for the message
};

struct VendorResources {
  // The vendor's domain prefix, "nvidia.com/". Any resource under it that is
  // in neither list makes the budget Unrecognised.
  const char *domain;
  // Exact count-shaped names: "nvidia.com/gpu".
  std::vector<std::string> exact;
  // Count-shaped prefixes: "nvidia.com/mig-". A MIG fleet gates the instance
  // count; deviceCount counts instances, not parts.
  std::vector<std::string> prefixes;
};

// The allow-sets, one per vendor image, in ONE place. A runner passes the one
// for the vendor it is built for. Every exact name here must also be a key of
// pkg/localrun's vendorResources table (the bare-metal CLI reads the same
// profile), and runners/devicefold_test.go refuses drift between the two.
// The prefixes are count-shaped resource families the CLI never sees.
inline VendorResources nvidiaResources() { return {"nvidia.com/", {"nvidia.com/gpu"}, {"nvidia.com/mig-"}}; }
inline VendorResources amdResources() { return {"amd.com/", {"amd.com/gpu"}, {}}; }

namespace detail {
inline bool startsWith(const std::string &s, const std::string &p) {
  return s.size() >= p.size() && s.compare(0, p.size(), p) == 0;
}
inline bool parseWhole(const std::string &s, long *out) {
  if (s.empty()) return false;
  char *end = nullptr;
  const long v = std::strtol(s.c_str(), &end, 10);
  if (end == nullptr || *end != '\0' || v < 0) return false;
  *out = v;
  return true;
}
}  // namespace detail

inline Budget parseBudget(const char *csv, const VendorResources &vendor) {
  if (csv == nullptr) return {Budget::Absent, 0, ""};
  // Set but empty is not a state the operator produces (it omits the variable
  // when there are no limits), so treat it as absent rather than invent one.
  if (*csv == '\0') return {Budget::Absent, 0, ""};

  long total = 0;
  bool matched = false;
  std::string s(csv);
  size_t pos = 0;
  while (pos <= s.size()) {
    size_t comma = s.find(',', pos);
    if (comma == std::string::npos) comma = s.size();
    const std::string item = s.substr(pos, comma - pos);
    pos = comma + 1;
    if (item.empty()) continue;
    const size_t eq = item.find('=');
    if (eq == std::string::npos) return {Budget::Malformed, 0, item};
    const std::string name = item.substr(0, eq);
    const std::string qty = item.substr(eq + 1);
    if (!detail::startsWith(name, vendor.domain)) continue;  // not ours; ignored

    bool ours = std::find(vendor.exact.begin(), vendor.exact.end(), name) != vendor.exact.end();
    for (const auto &p : vendor.prefixes) {
      if (detail::startsWith(name, p)) ours = true;
    }
    if (!ours) return {Budget::Unrecognised, 0, name};
    long n = 0;
    if (!detail::parseWhole(qty, &n)) return {Budget::Malformed, 0, item};
    total += n;
    matched = true;
  }
  if (!matched) {
    // Limits were set, none of them ours. The pod asked for no accelerator
    // at all — a CPU-only test image would never call this — so the honest
    // count is zero, and the runner reports "the pod was allocated no device"
    // rather than iterating whatever the leak shows it.
    return {Budget::Established, 0, ""};
  }
  return {Budget::Established, total, ""};
}

// ── Iteration plan ─────────────────────────────────────────────────────────

// Plan is whether the runner may iterate, and over how many devices.
struct Plan {
  enum Outcome {
    Iterate,  // over `count` devices, indices 0..count-1
    Skip,     // nothing to measure: no device visible (exit 2, with the marker)
    Error,    // the allocation cannot be established (exit 3)
  };
  Outcome outcome;
  int count;
  std::string message;
};

// planIteration decides from what the runtime SHOWED and what the pod was
// ALLOCATED. The rules, each of which is a way a multi-device runner could
// otherwise certify hardware it did not measure or measure hardware it did not
// own:
//
//   - visible == 0: Skip. A node with no accelerator has nothing to test; the
//     runner prints its own marker.
//   - Absent budget: iterate every visible device. Safe where the variable is
//     absent — bare metal, where `--gpus all` is the allocation.
//   - budget == visible: iterate them all. The intended configuration.
//   - budget < visible: ERROR. A budget is a count, not an identity; under the
//     legacy runtime nothing marks WHICH N of the visible devices are this
//     pod's, so capping at N would still iterate device 0, which may be
//     another pod's. The mismatch IS the misconfiguration.
//   - budget > visible: ERROR. The plugin handed fewer than it promised, or
//     the runtime hid some; either way the allocation is not what the pod
//     asked for and a fold over what showed up would certify a partial board.
//   - budget == 0 with devices visible: ERROR — the pod asked for no
//     accelerator and can see some. That is the leak, named.
//   - Unrecognised / Malformed: ERROR, naming the resource.
inline Plan planIteration(int visible, const Budget &b) {
  if (visible <= 0) return {Plan::Skip, 0, "no accelerator visible to this container"};
  switch (b.state) {
    case Budget::Absent:
      return {Plan::Iterate, visible, ""};
    case Budget::Unrecognised:
      return {Plan::Error, 0,
              "BURNIN_RESOURCE_LIMITS names " + b.detail +
                  ", a resource under this runner's vendor domain that it does not know how to count; the allocation cannot be established, so no device is measured"};
    case Budget::Malformed:
      return {Plan::Error, 0, "BURNIN_RESOURCE_LIMITS entry " + b.detail + " is not name=whole-number; the allocation cannot be established"};
    case Budget::Established:
      break;
  }
  if (b.count == static_cast<long>(visible)) return {Plan::Iterate, visible, ""};
  char buf[256];
  if (b.count < visible) {
    std::snprintf(buf, sizeof(buf),
                  "the runtime shows %d device(s) but this pod was allocated %ld; a budget is a count, not an identity, so which devices are this pod's cannot be established (legacy-runtime host injecting NVIDIA_VISIBLE_DEVICES=all?) — no device is measured",
                  visible, b.count);
  } else {
    std::snprintf(buf, sizeof(buf),
                  "this pod was allocated %ld device(s) but the runtime shows %d; the allocation is not what the pod asked for, and a fold over a partial board would certify devices nobody measured",
                  b.count, visible);
  }
  return {Plan::Error, 0, buf};
}

// ── Per-device readings and the fold ───────────────────────────────────────

// DeviceReport is what the runner measured on ONE device. values are the raw
// numeric keys the fold table names; identity is what makes the board's
// homogeneity decidable.
struct DeviceReport {
  int index = 0;
  std::string busId;
  std::string name;
  std::string computeCap;  // "12.1", or a gfx target on AMD
  bool identityRead = false;  // name AND computeCap were positively read
  std::map<std::string, double> values;
};

// Verdict precedence ACROSS DEVICES: Fail > Error > Skip > Pass. This inverts
// Pair's Error > Fail deliberately. A device is a PART: a measured miscompare
// on device 3 is a fact about device 3, and device 6 failing to enumerate does
// not erase it. A Pair endpoint is HALF A LINK: a machinery failure at either
// end means the link was never measured, and a client's "connection refused"
// is then an artifact of its peer. memory-bw already orders devices this way.
// Codes are the runner contract's: 0 pass, 1 fail, 2 skip, 3 error.
inline int combineExitCodes(const std::vector<int> &codes) {
  bool fail = false, error = false, skip = false;
  for (int c : codes) {
    switch (c) {
      case 0: break;
      case 1: fail = true; break;
      case 2: skip = true; break;
      default: error = true; break;
    }
  }
  if (fail) return 1;
  if (error) return 3;
  if (skip) return 2;
  return 0;
}

// Spread is one homogeneity figure: (max − min) / max × 100 across devices,
// or n/a — a POSITIVE claim: one device, a heterogeneous board, MIG, or a
// key no device reported — with the reason kept so the runner can say which.
struct Spread {
  bool measurable;
  double pct;
  const char *whyNot;  // when !measurable
};

struct Folded {
  std::map<std::string, double> values;  // folded keys, Once keys from device 0
  int worstIndex = 0;
  std::string worstBusId;
  bool homogeneous = true;      // every identity read and equal
  bool homogeneityKnown = true;  // false if any identity was not read
};

// fold combines per-device readings by the rules. primaryKey is the gated
// metric whose worst device the verdict names; its own rule decides which end
// is worst (Min → the smallest reading, Max → the largest, Sum/Once → the
// first device). A key a device did not report is simply absent from that
// device's contribution: absence is not a zero, and a key NO device reported
// stays absent from the fold, where a threshold on it fails closed.
inline Folded fold(const std::vector<DeviceReport> &devs, const std::vector<FoldRule> &rules,
                   const char *primaryKey) {
  Folded out;
  if (devs.empty()) return out;

  for (const auto &r : rules) {
    bool seen = false;
    double acc = 0;
    int worstIdx = devs.front().index;
    for (const auto &d : devs) {
      auto it = d.values.find(r.key);
      if (it == d.values.end()) continue;
      const double v = it->second;
      if (!seen) {
        acc = v;
        worstIdx = d.index;
        seen = true;
        if (r.fold == Fold::Once) break;
        continue;
      }
      switch (r.fold) {
        case Fold::Min:
          if (v < acc) { acc = v; worstIdx = d.index; }
          break;
        case Fold::Max:
          if (v > acc) { acc = v; worstIdx = d.index; }
          break;
        case Fold::Sum:
          acc += v;
          break;
        case Fold::Once:
          break;
      }
    }
    if (!seen) continue;
    out.values[r.key] = acc;
    if (primaryKey != nullptr && std::strcmp(r.key, primaryKey) == 0) {
      out.worstIndex = worstIdx;
      for (const auto &d : devs) {
        if (d.index == worstIdx) out.worstBusId = d.busId;
      }
    }
  }
  if (out.worstBusId.empty()) out.worstBusId = devs.front().busId;

  // Homogeneity: a POSITIVE establishment. If any identity was not read, the
  // answer is unknown and nothing is declared either way.
  for (const auto &d : devs) {
    if (!d.identityRead) { out.homogeneityKnown = false; break; }
  }
  if (out.homogeneityKnown) {
    for (const auto &d : devs) {
      if (d.name != devs.front().name || d.computeCap != devs.front().computeCap) {
        out.homogeneous = false;
        break;
      }
    }
  }
  return out;
}

// spreadOf computes one spread. absoluteFigure says the base metric is an
// absolute reading (TFLOP/s, GB/s, W, C) rather than a percentage of the part's
// own rating; on a heterogeneous board an absolute spread reads ~100% on
// healthy hardware and is declared n/a. underMig declares every spread n/a:
// MIG instances share one power, thermal and clock domain, so a spread across
// them measures one physical part several times and reads 0 on a degraded one.
inline Spread spreadOf(const std::vector<DeviceReport> &devs, const char *key, bool absoluteFigure,
                       const Folded &f, bool underMig) {
  if (underMig) return {false, 0, "MIG instances share one clock, power and thermal domain"};
  if (absoluteFigure && !f.homogeneityKnown) return {false, 0, "device identities were not all read, so homogeneity is unknown"};
  if (absoluteFigure && !f.homogeneous) return {false, 0, "heterogeneous board; an absolute spread across unlike parts is not a measurement"};
  double lo = 0, hi = 0;
  int n = 0;
  for (const auto &d : devs) {
    auto it = d.values.find(key);
    if (it == d.values.end()) continue;
    if (n == 0) { lo = hi = it->second; } else { lo = std::min(lo, it->second); hi = std::max(hi, it->second); }
    ++n;
  }
  if (n == 0) return {false, 0, "no device reported the base metric"};
  if (n == 1) return {false, 0, "one device; nothing to spread across"};
  if (!(hi > 0) || !std::isfinite(hi) || !std::isfinite(lo)) return {false, 0, "readings admit no ratio"};
  return {true, (hi - lo) / hi * 100.0, nullptr};
}

// ── Output ─────────────────────────────────────────────────────────────────

// SpreadSpec names one spread key and the raw key it is a spread of.
struct SpreadSpec {
  const char *spreadKey;  // "sustained_clock_spread_pct"
  const char *ofKey;      // "sustained_clock_pct"
  bool absoluteFigure;
};

// printFold writes the multi-device keys. The folded metric keys themselves
// are the runner's to print (it owns their formatting); this prints the
// device bookkeeping the registry knows: device_count, devices_visible,
// device_window_s, device_concurrency, worst_device_index,
// worst_device_pci_bus_id, device_homogeneous (only when known), and each
// spread as a number or as n/a.
inline void printFold(std::FILE *out, const std::vector<DeviceReport> &devs, int visible, long windowS,
                      Concurrency c, const Folded &f, const std::vector<SpreadSpec> &spreads,
                      bool underMig) {
  std::fprintf(out, "device_count=%zu\n", devs.size());
  std::fprintf(out, "devices_visible=%d\n", visible);
  std::fprintf(out, "device_window_s=%ld\n", windowS);
  std::fprintf(out, "device_concurrency=%s\n", concurrencyName(c));
  std::fprintf(out, "worst_device_index=%d\n", f.worstIndex);
  if (!f.worstBusId.empty()) std::fprintf(out, "worst_device_pci_bus_id=%s\n", f.worstBusId.c_str());
  if (f.homogeneityKnown) std::fprintf(out, "device_homogeneous=%s\n", f.homogeneous ? "true" : "false");
  for (const auto &s : spreads) {
    const Spread sp = spreadOf(devs, s.ofKey, s.absoluteFigure, f, underMig);
    if (sp.measurable) {
      std::fprintf(out, "%s=%.2f\n", s.spreadKey, sp.pct);
    } else {
      // n/a is a positive claim and the reason rides beside it, on stderr,
      // where it cannot be mistaken for a metric.
      std::fprintf(out, "%s=n/a\n", s.spreadKey);
      std::fprintf(stderr, "%s=n/a: %s\n", s.spreadKey, sp.whyNot);
    }
  }
}

namespace detail {
inline std::string jsonEscape(const std::string &s) {
  std::string o;
  o.reserve(s.size() + 2);
  for (char ch : s) {
    switch (ch) {
      case '"': o += "\\\""; break;
      case '\\': o += "\\\\"; break;
      case '\n': o += "\\n"; break;
      case '\r': o += "\\r"; break;
      case '\t': o += "\\t"; break;
      default:
        if (static_cast<unsigned char>(ch) < 0x20) {
          char b[8];
          std::snprintf(b, sizeof(b), "\\u%04x", ch);
          o += b;
        } else {
          o += ch;
        }
    }
  }
  return o;
}
}  // namespace detail

// renderPerDeviceArtifact is the per-device table, as the fenced artifact
// pkg/runner lifts out BEFORE the metric scanner sees it — so the "key": value
// lines inside can never rewrite a measurement. One object per device: index,
// bus id, name, compute capability, and every value the fold consumed. This is
// the ONLY sound attribution across the windows of a segmented soak;
// worst_device_index names one window's device.
inline std::string renderPerDeviceArtifact(const std::vector<DeviceReport> &devs) {
  std::string s = "-----BEGIN BURNIN ARTIFACT per-device.json application/json-----\n[";
  bool firstDev = true;
  for (const auto &d : devs) {
    if (!firstDev) s += ",";
    firstDev = false;
    s += "\n  {\"index\": " + std::to_string(d.index);
    s += ", \"pciBusId\": \"" + detail::jsonEscape(d.busId) + "\"";
    s += ", \"name\": \"" + detail::jsonEscape(d.name) + "\"";
    s += ", \"computeCap\": \"" + detail::jsonEscape(d.computeCap) + "\"";
    s += ", \"values\": {";
    bool firstVal = true;
    for (const auto &kv : d.values) {
      if (!firstVal) s += ", ";
      firstVal = false;
      char num[64];
      if (std::isfinite(kv.second)) {
        std::snprintf(num, sizeof(num), "%.6g", kv.second);
      } else {
        std::snprintf(num, sizeof(num), "null");  // JSON has no NaN; null says "no number"
      }
      s += "\"" + detail::jsonEscape(kv.first) + "\": " + num;
    }
    s += "}}";
  }
  s += "\n]\n-----END BURNIN ARTIFACT-----\n";
  return s;
}

}  // namespace devices
}  // namespace burnin

#endif  // BURNIN_DEVICE_FOLD_H
