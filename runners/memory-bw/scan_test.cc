// scan_test.cc — tests for the nvbandwidth output scanner in memory_bw.cc.
//
// SPDX-License-Identifier: Apache-2.0
// Copyright the Glimmer authors.
//
// Build and run:
//
//   c++ -std=c++17 -Wall -Wextra -o /tmp/scan_test scan_test.cc && /tmp/scan_test
//
// It is a plain program with no test framework so it needs nothing beyond a C++
// compiler, and it is NOT compiled into the runner image — the Dockerfile copies
// only memory_bw.cc.
//
// It includes the runner source with main() renamed rather than asking that file
// to carry a #ifdef. The shipped entrypoint has no test hook in it, and cannot
// grow one by accident.
//
// What is worth testing here is the scanner, not the plumbing: fork/exec and
// dlopen are exercised the moment the image runs at all, whereas the scanner is
// the part that quietly returns a wrong number if nvbandwidth's output shifts.

#define main memory_bw_main
#include "memory_bw.cc"
#undef main

#include <cmath>
#include <cstdio>

namespace {

int failures = 0;

void expect(const char *name, bool ok, const std::string &detail) {
  if (ok) {
    std::printf("ok   %s\n", name);
    return;
  }
  ++failures;
  std::printf("FAIL %s: %s\n", name, detail.c_str());
}

void expectScan(const char *name, const std::string &text, int wantCount, double wantMin,
                double wantMax) {
  const Sample s = scanBandwidth(text);
  bool ok = s.count == wantCount;
  if (ok && wantCount > 0) {
    ok = std::fabs(s.min - wantMin) < 1e-6 && std::fabs(s.max - wantMax) < 1e-6;
  }
  char detail[256];
  std::snprintf(detail, sizeof(detail), "count=%d min=%.2f max=%.2f", s.count, s.min, s.max);
  expect(name, ok, detail);
}

}  // namespace

int main() {
  // A healthy single-GPU run, with the preamble nvbandwidth prints first. The
  // preamble carries numbers of its own ("13.0", "580.82.09") and none of them
  // may be harvested as a bandwidth.
  expectScan("single GPU, full preamble",
             "nvbandwidth Version: v0.10.0\n"
             "Built from Git version: v0.10.0\n"
             "\n"
             "CUDA Runtime Version: 13.0\n"
             "Driver Version: 580.82.09\n"
             "\n"
             "Device 0: NVIDIA GB10\n"
             "\n"
             "Running host_to_device_memcpy_ce.\n"
             "memcpy CE CPU(row) -> GPU(column) bandwidth (GB/s)\n"
             "           0\n"
             " 0     24.83\n"
             "\n"
             "SUM host_to_device_memcpy_ce 24.83\n"
             "COEFFICIENT_OF_VARIATION host_to_device_memcpy_ce 0.01\n",
             1, 24.83, 24.83);

  // A peer matrix: the 0.00 diagonal is a pair the testcase did not measure, and
  // counting it would drag the minimum — the figure acceptance is decided on —
  // to zero on every multi-GPU node.
  expectScan("peer matrix, diagonal excluded",
             "Running device_to_device_memcpy_read_ce.\n"
             "memcpy CE GPU(row) <- GPU(column) bandwidth (GB/s)\n"
             "          0         1         2         3\n"
             "0      0.00    276.07    276.36    276.14\n"
             "1    276.19      0.00    276.29    276.29\n"
             "2    276.30    150.05      0.00    276.11\n"
             "3    276.21    276.19    276.09      0.00\n"
             "\n"
             "SUM device_to_device_memcpy_read_ce 3312.29\n",
             12, 150.05, 276.36);

  expectScan("N/A cells skipped",
             "memcpy local GPU(row) bandwidth (GB/s)\n"
             "          0         1\n"
             "0    612.40       N/A\n"
             "1       N/A    598.72\n"
             "\n"
             "SUM device_local_copy 1211.12\n",
             2, 598.72, 612.40);

  // A leading row with nothing readable in it must not end the scan, or every
  // device after the first unmeasured one would vanish from the sample.
  expectScan("all-N/A row does not truncate the matrix",
             "memcpy CE bandwidth (GB/s)\n"
             "          0         1\n"
             "0       N/A       N/A\n"
             "1    100.00    200.00\n"
             "\n",
             2, 100.00, 200.00);

  // Row labels are device indices on some testcases and NUMA nodes on others,
  // and a label may be more than one token. The scan keys on the cell format,
  // not on a column position, so this must still work.
  expectScan("multi-token row labels",
             "memcpy CE CPU(row) -> GPU(column) bandwidth (GB/s)\n"
             "          0         1\n"
             "numa 0     24.83     25.10\n"
             "numa 1     24.11     25.02\n"
             "\n",
             4, 24.11, 25.10);

  expectScan("prose after the matrix is not harvested",
             "memcpy CE CPU(row) -> GPU(column) bandwidth (GB/s)\n"
             "           0\n"
             " 0     24.83\n"
             "\n"
             "NOTE: a later note that happens to contain 99.99\n",
             1, 24.83, 24.83);

  // The two ways nothing is measurable. Both must come back empty so the caller
  // reports Error — "we could not read this" is never a hardware verdict.
  expectScan("no matrix at all",
             "Running host_to_device_memcpy_ce.\n"
             "ASSERT in expression cuMemAlloc(&ptr, size) at memcpy.cpp:12\n",
             0, 0, 0);
  expectScan("preamble only", "CUDA Runtime Version: 13.0\nDriver Version: 580.82.09\n", 0, 0, 0);

  // Fail is reserved for corruption. Everything else nvbandwidth can exit 1 for
  // must fall through to Error, so these classifications are the difference
  // between condemning a node and retrying it.
  expect("corruption: read-only pattern mismatch",
         reportsCorruption("ERROR: Read-only pattern mismatch at element 42"), "");
  expect("corruption: write-only mismatch count",
         reportsCorruption("ERROR: Write-only test found 17 mismatched elements!"), "");
  expect("corruption: generic memcmp assert",
         reportsCorruption("ASSERT in expression h_errorFlag == 0 in memcmpPatternHelper()"), "");
  expect("not corruption: CUDA API failure",
         !reportsCorruption("[CUDA_ERROR_OUT_OF_MEMORY] out of memory in expression "
                            "cuMemAlloc(&ptr, size) in doMemcpy() : memcpy.cpp:88"),
         "");
  expect("not corruption: usage text mentioning verification",
         !reportsCorruption("  -s, --skipVerification    Skips data verification after copy"), "");

  expect("waived is detected case-insensitively",
         containsCI("Testcase device_local_copy_tma WAIVED: TMA unsupported", "waived"), "");

  // decimalToken is what separates a measurement from the matrix's own index
  // headers and from version strings in the preamble.
  double v = 0;
  expect("token: rejects a version triple", !decimalToken("580.82.09", v), "");
  expect("token: rejects a bare integer", !decimalToken("13000", v), "");
  expect("token: rejects N/A", !decimalToken("N/A", v), "");
  expect("token: rejects a signed value", !decimalToken("-1.00", v), "");
  expect("token: accepts a cell", decimalToken("276.07", v) && std::fabs(v - 276.07) < 1e-9, "");

  std::printf("%s\n", failures == 0 ? "ALL OK" : "FAILURES");
  return failures == 0 ? 0 : 1;
}
