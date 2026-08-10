// memory_bw.cc — runner for the "memory-bw" TestKind.
//
// SPDX-License-Identifier: Apache-2.0
// Copyright the Glimmer authors.
//
// This file is original work licensed under Apache-2.0. It drives NVIDIA
// nvbandwidth (Apache-2.0) as a child process and translates its output into
// the burn-in runner contract: key=value metrics on stdout, and an exit code
// that means pass (0) / fail (1) / not-applicable (2) / error (anything else).
//
// Why a wrapper rather than shipping nvbandwidth as the entrypoint: nvbandwidth
// prints a human-readable bandwidth matrix and exits 1 for every failure it can
// have, hardware or otherwise. Both halves of the runner contract — the metric
// names and, more importantly, the Fail/Error distinction — have to be produced
// by something that knows which is which.
//
// The device probe is done through dlopen("libcuda.so.1") rather than by
// linking the CUDA driver stub. That keeps this binary's DT_NEEDED list free of
// every NVIDIA name, so the image's "no NVIDIA redistributable" build assertion
// has only nvbandwidth's own dependencies to reason about, and it means a node
// with no driver injected reports a clean Skip instead of failing to start.

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include <dlfcn.h>
#include <errno.h>
#include <poll.h>
#include <signal.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>

// Both are set by the Dockerfile so the image records exactly which nvbandwidth
// it shipped. The defaults exist only so this file compiles on its own.
#ifndef NVBANDWIDTH_BIN
#define NVBANDWIDTH_BIN "/usr/local/bin/nvbandwidth"
#endif
#ifndef NVBANDWIDTH_REF
#define NVBANDWIDTH_REF "unknown"
#endif

namespace {

using Clock = std::chrono::steady_clock;

// The runner contract. Anything other than 0/1/2 is an Error to the operator;
// 3 is used so an Error is distinguishable from a crash in the logs.
constexpr int kExitPass = 0;
constexpr int kExitFail = 1;
constexpr int kExitSkip = 2;
constexpr int kExitError = 3;

// CUDA driver API constants we need without linking against it.
constexpr int kCudaSuccess = 0;
constexpr int kCudaErrorNoDevice = 100;

// nvbandwidth's own default buffer size (MiB) and sample count. They are the
// defaults here too: a burn-in verdict should be reproducible against the
// upstream tool's published numbers unless somebody deliberately changed the
// shape of the measurement.
constexpr long kDefaultBufferMiB = 512;
constexpr long kDefaultSamples = 3;

// Used when BURNIN_DURATION_SECONDS is unset — a runner launched by hand must
// still be bounded. Matches the operator's own defaultDurationSeconds.
constexpr long kDefaultDurationSeconds = 600;

int pass() {
  std::printf("MEMORY_BW_PASS\n");
  return kExitPass;
}

int fail(const std::string &why) {
  std::printf("MEMORY_BW_FAIL: %s\n", why.c_str());
  return kExitFail;
}

int skip(const std::string &why) {
  std::printf("MEMORY_BW_SKIP: %s\n", why.c_str());
  return kExitSkip;
}

int error(const std::string &why) {
  std::printf("MEMORY_BW_ERROR: %s\n", why.c_str());
  return kExitError;
}

long envLong(const char *name, long fallback) {
  const char *raw = std::getenv(name);
  if (raw == nullptr || *raw == '\0') return fallback;
  char *end = nullptr;
  errno = 0;
  const long value = std::strtol(raw, &end, 10);
  // A malformed value falls back rather than aborting: a typo in a profile's
  // env must not turn into an Error that reads like a hardware problem.
  if (errno != 0 || end == raw || *end != '\0') return fallback;
  return value;
}

long clampLong(long v, long lo, long hi) { return v < lo ? lo : (v > hi ? hi : v); }

std::vector<std::string> splitLines(const std::string &text) {
  std::vector<std::string> lines;
  size_t begin = 0;
  while (begin <= text.size()) {
    size_t end = text.find('\n', begin);
    if (end == std::string::npos) {
      lines.push_back(text.substr(begin));
      break;
    }
    lines.push_back(text.substr(begin, end - begin));
    begin = end + 1;
  }
  return lines;
}

std::vector<std::string> splitTokens(const std::string &line) {
  std::vector<std::string> tokens;
  size_t i = 0;
  while (i < line.size()) {
    while (i < line.size() && std::isspace(static_cast<unsigned char>(line[i]))) ++i;
    size_t start = i;
    while (i < line.size() && !std::isspace(static_cast<unsigned char>(line[i]))) ++i;
    if (i > start) tokens.push_back(line.substr(start, i - start));
  }
  return tokens;
}

bool containsCI(const std::string &haystack, const char *needle) {
  const size_t n = std::strlen(needle);
  if (n == 0 || haystack.size() < n) return false;
  for (size_t i = 0; i + n <= haystack.size(); ++i) {
    size_t j = 0;
    while (j < n && std::tolower(static_cast<unsigned char>(haystack[i + j])) ==
                        std::tolower(static_cast<unsigned char>(needle[j])))
      ++j;
    if (j == n) return true;
  }
  return false;
}

// decimalToken accepts exactly the shape nvbandwidth prints a bandwidth in:
// digits, one '.', digits (it formats the matrix with std::fixed and
// setprecision(2)). Requiring the '.' is what separates a measurement from the
// bare integers in the matrix's own row and column index headers, and refusing
// a sign keeps a stray "-" out of the sample set.
bool decimalToken(const std::string &token, double &out) {
  if (token.empty()) return false;
  size_t dot = std::string::npos;
  for (size_t i = 0; i < token.size(); ++i) {
    const char c = token[i];
    if (c == '.') {
      if (dot != std::string::npos) return false;
      dot = i;
      continue;
    }
    if (std::isdigit(static_cast<unsigned char>(c)) == 0) return false;
  }
  if (dot == std::string::npos || dot == 0 || dot + 1 == token.size()) return false;
  out = std::strtod(token.c_str(), nullptr);
  return true;
}

// Sample is the set of bandwidth figures one nvbandwidth testcase produced.
struct Sample {
  double min = 0.0;
  double max = 0.0;
  int count = 0;
};

// scanBandwidth reads every bandwidth cell out of one nvbandwidth run.
//
// The tool prints a description line ending in "(GB/s)", then a header row of
// column indices, then one line per row: a row label followed by one cell per
// column, each either a fixed-point number or "N/A". It closes with
// "SUM <key> <value>" and "COEFFICIENT_OF_VARIATION <key> <value>".
//
// This is deliberately a tolerant scan and not a positional parse. Column
// widths, row labels (device index on some testcases, NUMA node on others) and
// the exact matrix shape all differ between testcases and between releases; the
// two things that do not are the "(GB/s)" anchor and the fixed-point cell
// format. Anything the scan cannot read comes back as count == 0, which the
// caller turns into an Error — never into a Fail. A parser that has lost track
// of the tool's output has not learned anything about the hardware.
//
// Each nvbandwidth invocation runs exactly one testcase, so there is no need to
// attribute rows to a testcase name: the whole capture belongs to one.
Sample scanBandwidth(const std::string &text) {
  Sample sample;
  const std::vector<std::string> lines = splitLines(text);

  size_t anchor = lines.size();
  for (size_t i = 0; i < lines.size(); ++i) {
    if (lines[i].find("(GB/s)") != std::string::npos) {
      anchor = i;
      break;
    }
  }
  if (anchor == lines.size()) return sample;

  bool seenRow = false;
  for (size_t i = anchor + 1; i < lines.size(); ++i) {
    const std::vector<std::string> tokens = splitTokens(lines[i]);
    if (tokens.empty()) {
      if (seenRow) break;  // the blank line after the matrix
      continue;
    }
    const std::string &head = tokens[0];
    if (head == "SUM" || head == "COEFFICIENT_OF_VARIATION" || head == "&&&&" ||
        head == "Running" || lines[i].find("(GB/s)") != std::string::npos) {
      break;
    }
    seenRow = true;
    for (const std::string &token : tokens) {
      double value = 0.0;
      if (!decimalToken(token, value)) continue;
      // A 0.00 cell is a pair the testcase did not measure (the diagonal of a
      // peer matrix), not a link that moved no data. Counting it would drag
      // every minimum to zero.
      if (value <= 0.0) continue;
      if (sample.count == 0 || value < sample.min) sample.min = value;
      if (sample.count == 0 || value > sample.max) sample.max = value;
      ++sample.count;
    }
  }
  return sample;
}

// ── running nvbandwidth ───────────────────────────────────────────────────────

struct Child {
  int exitCode = -1;
  bool spawnFailed = false;
  bool timedOut = false;
  std::string output;
};

// readUntilEof drains fd, returning false if the deadline passed first.
bool readUntilEof(int fd, bool hasDeadline, Clock::time_point deadline, std::string &out) {
  for (;;) {
    int waitMs = -1;
    if (hasDeadline) {
      const long long remaining =
          std::chrono::duration_cast<std::chrono::milliseconds>(deadline - Clock::now()).count();
      if (remaining <= 0) return false;
      waitMs = static_cast<int>(std::min<long long>(remaining, 500));
    }
    struct pollfd pfd;
    pfd.fd = fd;
    pfd.events = POLLIN;
    pfd.revents = 0;
    const int rc = ::poll(&pfd, 1, waitMs);
    if (rc < 0) {
      if (errno == EINTR) continue;
      return true;  // treat as end of output; the exit code still decides
    }
    if (rc == 0) continue;  // timeout slice elapsed, re-check the deadline
    char buf[8192];
    const ssize_t n = ::read(fd, buf, sizeof(buf));
    if (n < 0) {
      if (errno == EINTR) continue;
      return true;
    }
    if (n == 0) return true;
    out.append(buf, static_cast<size_t>(n));
  }
}

Child runNvbandwidth(const std::vector<std::string> &args, bool hasDeadline,
                     Clock::time_point deadline) {
  Child child;
  int fds[2];
  if (::pipe(fds) != 0) {
    child.spawnFailed = true;
    return child;
  }
  const pid_t pid = ::fork();
  if (pid < 0) {
    ::close(fds[0]);
    ::close(fds[1]);
    child.spawnFailed = true;
    return child;
  }
  if (pid == 0) {
    ::close(fds[0]);
    ::dup2(fds[1], STDOUT_FILENO);
    ::dup2(fds[1], STDERR_FILENO);
    ::close(fds[1]);
    std::vector<char *> argv;
    argv.reserve(args.size() + 1);
    for (const std::string &a : args) argv.push_back(const_cast<char *>(a.c_str()));
    argv.push_back(nullptr);
    ::execv(argv[0], argv.data());
    ::_exit(127);
  }

  ::close(fds[1]);
  const bool drained = readUntilEof(fds[0], hasDeadline, deadline, child.output);
  ::close(fds[0]);
  if (!drained) {
    child.timedOut = true;
    ::kill(pid, SIGTERM);
    struct timespec ts;
    ts.tv_sec = 0;
    ts.tv_nsec = 500L * 1000L * 1000L;
    ::nanosleep(&ts, nullptr);
    // A copy wedged in the driver will not act on SIGTERM. SIGKILL is what
    // guarantees this process can still report, which matters more than a
    // clean child shutdown for a test we have already given up on.
    ::kill(pid, SIGKILL);
  }
  int status = 0;
  while (::waitpid(pid, &status, 0) < 0 && errno == EINTR) {
  }
  child.exitCode = WIFEXITED(status) ? WEXITSTATUS(status) : -1;
  return child;
}

// echoChild copies nvbandwidth's output to stderr so the pod log keeps the raw
// evidence behind every number reported below.
//
// The "nvbandwidth| " prefix is load-bearing, not decoration. The controller
// parses the pod log, and a line is only read as a metric when the text before
// its first '=' contains no whitespace. The prefix puts a space there on every
// echoed line, so nothing nvbandwidth prints can ever be mistaken for one of
// this runner's metrics.
void echoChild(const char *testcase, const std::string &out) {
  std::fflush(stdout);
  std::fprintf(stderr, "nvbandwidth| ---- %s ----\n", testcase);
  size_t begin = 0;
  while (begin < out.size()) {
    size_t end = out.find('\n', begin);
    if (end == std::string::npos) end = out.size();
    std::fprintf(stderr, "nvbandwidth| %.*s\n", static_cast<int>(end - begin), out.data() + begin);
    begin = end + 1;
  }
  std::fflush(stderr);
}

// ── device probe ──────────────────────────────────────────────────────────────

struct Device {
  int count = 0;
  int driverVersion = 0;
  std::string name;
};

enum class Probe { Ok, NoDevice, Error };

Probe probeCuda(Device &device, std::string &detail) {
  void *lib = ::dlopen("libcuda.so.1", RTLD_NOW | RTLD_LOCAL);
  if (lib == nullptr) {
    // No driver library was injected: either the node has no NVIDIA
    // accelerator, or the container toolkit is not wired up. Either way this
    // test does not apply to what is in front of us.
    detail = "libcuda.so.1 is not present, so this node exposes no CUDA device";
    return Probe::NoDevice;
  }

  auto cuInit = reinterpret_cast<int (*)(unsigned int)>(::dlsym(lib, "cuInit"));
  auto cuDeviceGetCount = reinterpret_cast<int (*)(int *)>(::dlsym(lib, "cuDeviceGetCount"));
  auto cuDeviceGet = reinterpret_cast<int (*)(int *, int)>(::dlsym(lib, "cuDeviceGet"));
  auto cuDeviceGetName = reinterpret_cast<int (*)(char *, int, int)>(::dlsym(lib, "cuDeviceGetName"));
  auto cuDriverGetVersion = reinterpret_cast<int (*)(int *)>(::dlsym(lib, "cuDriverGetVersion"));
  if (cuInit == nullptr || cuDeviceGetCount == nullptr || cuDeviceGet == nullptr ||
      cuDeviceGetName == nullptr || cuDriverGetVersion == nullptr) {
    detail = "libcuda.so.1 does not export the expected driver API symbols";
    return Probe::Error;
  }

  const int rc = cuInit(0);
  if (rc == kCudaErrorNoDevice) {
    detail = "the NVIDIA driver reports no CUDA device on this node";
    return Probe::NoDevice;
  }
  if (rc != kCudaSuccess) {
    // A driver that is present but unusable (version mismatch, wedged) is an
    // infrastructure fault. Reporting Skip here would quietly excuse a node
    // that was never tested.
    detail = "cuInit failed with CUDA driver error " + std::to_string(rc);
    return Probe::Error;
  }
  if (cuDeviceGetCount(&device.count) != kCudaSuccess) {
    detail = "cuDeviceGetCount failed";
    return Probe::Error;
  }
  if (device.count <= 0) {
    detail = "the NVIDIA driver reports 0 CUDA devices on this node";
    return Probe::NoDevice;
  }

  cuDriverGetVersion(&device.driverVersion);
  int handle = 0;
  char name[256];
  name[0] = '\0';
  if (cuDeviceGet(&handle, 0) == kCudaSuccess &&
      cuDeviceGetName(name, static_cast<int>(sizeof(name)), handle) == kCudaSuccess) {
    name[sizeof(name) - 1] = '\0';
    device.name = name;
  }
  return Probe::Ok;
}

// ── the measurement plan ──────────────────────────────────────────────────────

// One nvbandwidth testcase and the metrics it feeds.
//
// All three are copy-engine testcases, chosen so the reported figures do not
// depend on which SM architecture the image was compiled for.
//
// device_local_copy — not device_to_device_memcpy_read_ce — is what backs
// deviceToDeviceBandwidthGBs. The two are different measurands and each has its
// own column: device_local_copy is the on-device figure ("bounded by the memory
// subsystem rather than by the host link"), while the device_to_device cases are
// GPU-to-GPU PEER copies over NVLink, xGMI or a PCIe switch. Putting a peer
// figure in the on-device column would put two quantities in one place, which is
// why the peer cases below carry names of their own rather than sharing.
//
// The peer cases are where a multi-GPU node's interesting failures live — issue
// #161. A degraded NVLink lane, an xGMI link that trained narrow, a switch port
// that fell back to a lower width: none of them move a device-local copy, and
// all of them halve a training job that assumed the fabric.
struct Case {
  const char *testcase;
  const char *minKey;  // the acceptance metric — see README on why it is the min
  const char *maxKey;  // evidence, so a one-bad-link node is visible as a spread
  // peer marks a case that measures a link BETWEEN accelerators and therefore
  // needs at least two. On a single-GPU node it is not merely absent: it is
  // declared unmeasurable — see the n/a emission below.
  bool peer;
};

const Case kCases[] = {
    {"host_to_device_memcpy_ce", "h2d_bandwidth_gbs", "hostToDeviceBandwidthMaxGBs", false},
    {"device_to_host_memcpy_ce", "d2h_bandwidth_gbs", "deviceToHostBandwidthMaxGBs", false},
    {"device_local_copy", "d2d_bandwidth_gbs", "deviceToDeviceBandwidthMaxGBs", false},
    // The all-pairs peer matrix, read and write. Copy-engine variants for the
    // same reason as the three above: the figure must not depend on which SM
    // architecture this image was compiled for.
    //
    // The MINIMUM cell is the acceptance figure, matching this runner's existing
    // convention and for a sharper reason here — a fabric is as good as its
    // worst link, and a mean over an all-pairs matrix hides exactly the single
    // degraded lane the case exists to find.
    {"device_to_device_memcpy_read_ce", "peer_read_bandwidth_gbs", "peerReadBandwidthMaxGBs", true},
    {"device_to_device_memcpy_write_ce", "peer_write_bandwidth_gbs", "peerWriteBandwidthMaxGBs", true},
};

// reportsCorruption identifies the one condition under which nvbandwidth's
// non-zero exit is a hardware verdict rather than an infrastructure fault: its
// post-copy data verification found bytes that did not survive the transfer.
//
// Every other failure inside nvbandwidth — a CUDA API error, an unknown
// testcase, an allocation it could not make — exits 1 through the same ASSERT
// macro, so the exit code alone cannot tell them apart. Matching narrowly and
// falling through to Error is the safe direction: an Error leaves the node
// unaccepted and retried, while a wrong Fail condemns working hardware.
bool reportsCorruption(const std::string &out) {
  return containsCI(out, "mismatch") || containsCI(out, "h_errorFlag");
}

}  // namespace

int main() {
  // stdout is a pipe under the kubelet, so it is fully buffered by default.
  // Line buffering is what guarantees the metrics printed before a decision
  // survive even if this process is killed at the pod's activeDeadlineSeconds.
  std::setvbuf(stdout, nullptr, _IOLBF, 0);

  const Clock::time_point start = Clock::now();

  // BURNIN_DURATION_SECONDS is honoured as a budget, not as a workload length:
  // nvbandwidth's work is set by its sample count, not by a clock, so a bigger
  // number here does not buy a longer soak (use thermal-soak or gpu-burn for
  // that). What it does buy is a deadline this runner enforces itself, so a
  // wedged copy is reported as an Error with its evidence rather than being
  // SIGKILLed by the kubelet with an empty log.
  const long durationSeconds = envLong("BURNIN_DURATION_SECONDS", kDefaultDurationSeconds);
  const bool hasDeadline = durationSeconds > 0;
  const Clock::time_point deadline = start + std::chrono::seconds(hasDeadline ? durationSeconds : 0);

  std::printf("nvbandwidth_ref=%s\n", NVBANDWIDTH_REF);

  Device device;
  std::string detail;
  switch (probeCuda(device, detail)) {
    case Probe::NoDevice:
      return skip(detail);
    case Probe::Error:
      return error(detail);
    case Probe::Ok:
      break;
  }

  std::printf("gpu_name=%s\ngpu_count=%d\ncuda_driver_version=%d\n", device.name.c_str(),
              device.count, device.driverVersion);

  const long bufferMiB = clampLong(envLong("BURNIN_MEMORY_BW_BUFFER_MIB", kDefaultBufferMiB), 1, 65536);
  const long samples = clampLong(envLong("BURNIN_MEMORY_BW_SAMPLES", kDefaultSamples), 1, 1000);
  const bool disableAffinity = envLong("BURNIN_MEMORY_BW_DISABLE_AFFINITY", 0) != 0;

  std::printf("transfer_size_bytes=%lld\ntest_samples=%ld\n",
              static_cast<long long>(bufferMiB) * 1024LL * 1024LL, samples);

  // Only one of these is ever reported, in this order of precedence. A
  // verification mismatch is a fact about the hardware and outranks everything;
  // an Error ("we do not know") outranks a Skip, because a run that partly
  // failed has not established that the test was inapplicable.
  std::string failReason;
  std::string errorReason;
  std::string skipReason;

  for (const Case &c : kCases) {
    // A PEER CASE ON A SINGLE-GPU NODE IS UNMEASURABLE, AND SAYS SO.
    //
    // Not a silent omission and not a skip of the whole test. Omitting the key
    // would make a threshold on it fail closed, condemning every single-GPU
    // node in the fleet for hardware it does not have; exiting 2 would throw
    // away the three device-local figures this run DID measure, which on a DGX
    // Spark is the entire value of the kind.
    //
    // `n/a` is the reserved value for exactly this: the runner looked, and the
    // part has nothing to report. pkg/runner puts it in Result.Unmeasurable
    // rather than Metrics, so a threshold with applicability
    // RequiredIfMeasurable is reported NOT EVALUATED instead of failed. It is a
    // declaration the runner is entitled to make because it positively
    // established the device count — the same rule host-health follows for ECC.
    if (c.peer && device.count < 2) {
      std::printf("%s=n/a\n%s=n/a\n", c.minKey, c.maxKey);
      continue;
    }

    std::vector<std::string> args{NVBANDWIDTH_BIN,        "-t", c.testcase,
                                  "-b", std::to_string(bufferMiB),
                                  "-i", std::to_string(samples)};
    if (disableAffinity) args.push_back("-d");

    // One invocation per testcase. It costs a CUDA init each time and buys the
    // guarantee that every number in the capture belongs to the testcase we
    // asked for, with no cross-testcase attribution to get wrong.
    const Child child = runNvbandwidth(args, hasDeadline, deadline);
    echoChild(c.testcase, child.output);

    if (child.spawnFailed) {
      if (errorReason.empty())
        errorReason = std::string("could not start ") + NVBANDWIDTH_BIN;
      continue;
    }
    if (child.exitCode == 127) {
      if (errorReason.empty())
        errorReason = std::string(NVBANDWIDTH_BIN) + " is missing from the runner image";
      break;
    }

    const Sample sample = scanBandwidth(child.output);
    // Metrics first, decision after: a run that ends in Fail or Error still has
    // to leave behind whatever it did manage to measure.
    if (sample.count > 0) {
      std::printf("%s=%.2f\n%s=%.2f\n", c.minKey, sample.min, c.maxKey, sample.max);
    }

    if (child.timedOut) {
      if (errorReason.empty())
        errorReason = std::string("nvbandwidth ") + c.testcase +
                      " exceeded the BURNIN_DURATION_SECONDS budget";
      break;  // the budget is spent; the remaining testcases cannot run
    }
    if (child.exitCode != 0) {
      if (reportsCorruption(child.output)) {
        if (failReason.empty())
          failReason = std::string(c.testcase) +
                       " failed nvbandwidth's post-copy data verification";
      } else if (errorReason.empty()) {
        errorReason = std::string("nvbandwidth ") + c.testcase + " exited " +
                      std::to_string(child.exitCode) + " without a data-verification failure";
      }
      continue;
    }
    if (containsCI(child.output, "waived")) {
      if (c.peer) {
        // Two or more accelerators and nvbandwidth still declined: no peer path
        // between them (no NVLink, no P2P over the switch). That is a fact about
        // the topology, so it is declared unmeasurable rather than waiving the
        // whole test — the device-local figures are still good.
        std::printf("%s=n/a\n%s=n/a\n", c.minKey, c.maxKey);
        continue;
      }
      if (skipReason.empty())
        skipReason = std::string(c.testcase) + " is not supported on this hardware";
      continue;
    }
    if (sample.count == 0) {
      // The tool reported success and we could not read a figure out of it.
      // That is a statement about this parser or about a changed output
      // format, not about the memory subsystem, so it must not become a Fail.
      if (errorReason.empty())
        errorReason = std::string("no bandwidth figure could be read from nvbandwidth's ") +
                      c.testcase + " output";
      continue;
    }
  }

  const double elapsed =
      std::chrono::duration_cast<std::chrono::milliseconds>(Clock::now() - start).count() / 1000.0;
  std::printf("elapsed_s=%.1f\n", elapsed);

  if (!failReason.empty()) return fail(failReason);
  if (!errorReason.empty()) return error(errorReason);
  if (!skipReason.empty()) return skip(skipReason);
  return pass();
}
