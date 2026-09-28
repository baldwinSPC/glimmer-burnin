// SPDX-License-Identifier: Apache-2.0
// Copyright the Glimmer authors.
//
// managed_triad — STREAM triad over cudaMallocManaged memory (#543).
//
// nvbandwidth measures copy engines and device-local copies; it has no case
// for managed (unified) memory. On a part with one physical pool shared by CPU
// and GPU — GB10's LPDDR5X over NVLink-C2C — managed memory is what
// unified-memory workloads actually use, so its bandwidth is a first-class
// measurement rather than a portability detail.
//
// Method (McCalpin's triad, y = a·x + z, on float arrays): each of three
// managed arrays holds a fraction of the device's free memory between them,
// is first touched ON THE DEVICE by an init kernel, then the triad runs
// `iters` times, each timed with CUDA events. Bandwidth per iteration is
// 3·E·4 bytes / seconds; the reported figure is the median. Every visible
// device is measured and the WORST is reported, matching the rest of this
// runner's fold.
//
// Contract with memory_bw.cc (stdout, key=value):
//   exit 0  managed_triad_gbs=<worst device, GB/s>
//   exit 1  managed_triad_miscompare=<device> — the triad produced wrong data
//   exit 2  managed_triad_unsupported=<device> — the device reports no
//           managed-memory support (cudaDevAttrManagedMemory == 0)
//   exit 3  anything else: the measurement could not be made
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

__global__ void initKernel(float *x, float *y, float *z, size_t n) {
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) {
    x[i] = 1.0f;
    y[i] = 0.0f;
    z[i] = 2.0f;
  }
}

__global__ void triadKernel(float *y, const float *x, const float *z, float a, size_t n) {
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) {
    y[i] = a * x[i] + z[i];
  }
}

static int fail3(const char *what, cudaError_t e) {
  std::printf("managed_triad_error=%s: %s\n", what, cudaGetErrorString(e));
  return 3;
}

int main(int argc, char **argv) {
  const double fraction = argc > 1 ? std::atof(argv[1]) : 0.1;
  const int iters = argc > 2 ? std::max(3, std::atoi(argv[2])) : 20;
  if (!(fraction > 0 && fraction < 0.9)) {
    std::printf("managed_triad_error=fraction %.3f out of range\n", fraction);
    return 3;
  }

  int count = 0;
  cudaError_t e = cudaGetDeviceCount(&count);
  if (e != cudaSuccess) return fail3("cudaGetDeviceCount", e);
  if (count == 0) {
    std::printf("managed_triad_error=no device visible\n");
    return 3;
  }

  double worst = INFINITY;
  for (int dev = 0; dev < count; ++dev) {
    if ((e = cudaSetDevice(dev)) != cudaSuccess) return fail3("cudaSetDevice", e);
    int managed = 0;
    if ((e = cudaDeviceGetAttribute(&managed, cudaDevAttrManagedMemory, dev)) != cudaSuccess)
      return fail3("cudaDeviceGetAttribute", e);
    if (!managed) {
      std::printf("managed_triad_unsupported=%d\n", dev);
      return 2;
    }
    size_t freeB = 0, totalB = 0;
    if ((e = cudaMemGetInfo(&freeB, &totalB)) != cudaSuccess) return fail3("cudaMemGetInfo", e);
    const size_t n = static_cast<size_t>(fraction * freeB / (3 * sizeof(float)));
    if (n < 1024) {
      std::printf("managed_triad_error=only %zu bytes free on device %d\n", freeB, dev);
      return 3;
    }

    float *x = nullptr, *y = nullptr, *z = nullptr;
    if ((e = cudaMallocManaged(&x, n * sizeof(float))) != cudaSuccess) return fail3("cudaMallocManaged", e);
    if ((e = cudaMallocManaged(&y, n * sizeof(float))) != cudaSuccess) return fail3("cudaMallocManaged", e);
    if ((e = cudaMallocManaged(&z, n * sizeof(float))) != cudaSuccess) return fail3("cudaMallocManaged", e);

    const int threads = 256, blocks = 1024;
    initKernel<<<blocks, threads>>>(x, y, z, n);
    triadKernel<<<blocks, threads>>>(y, x, z, 3.0f, n);  // warm-up
    if ((e = cudaDeviceSynchronize()) != cudaSuccess) return fail3("warm-up", e);

    cudaEvent_t a, b;
    cudaEventCreate(&a);
    cudaEventCreate(&b);
    std::vector<double> gbs;
    for (int i = 0; i < iters; ++i) {
      cudaEventRecord(a);
      triadKernel<<<blocks, threads>>>(y, x, z, 3.0f, n);
      cudaEventRecord(b);
      if ((e = cudaEventSynchronize(b)) != cudaSuccess) return fail3("triad", e);
      float ms = 0;
      cudaEventElapsedTime(&ms, a, b);
      if (ms > 0) gbs.push_back(3.0 * n * sizeof(float) / 1e9 / (ms / 1000.0));
    }
    cudaEventDestroy(a);
    cudaEventDestroy(b);

    // 3·1 + 2 = 5 exactly in float. Checked on the host at evenly spaced
    // points plus both ends: a wrong value is corruption, not noise.
    const size_t step = std::max<size_t>(1, n / 4096);
    for (size_t i = 0; i < n; i += step) {
      if (y[i] != 5.0f) {
        std::printf("managed_triad_miscompare=%d\n", dev);
        return 1;
      }
    }
    if (y[n - 1] != 5.0f) {
      std::printf("managed_triad_miscompare=%d\n", dev);
      return 1;
    }
    cudaFree(x);
    cudaFree(y);
    cudaFree(z);

    if (gbs.empty()) {
      std::printf("managed_triad_error=no timed iteration on device %d\n", dev);
      return 3;
    }
    std::sort(gbs.begin(), gbs.end());
    const double median = gbs[gbs.size() / 2];
    std::printf("managed_triad_device=%d gbs=%.2f elements=%zu\n", dev, median, n);
    worst = std::min(worst, median);
  }
  std::printf("managed_triad_gbs=%.2f\n", worst);
  return 0;
}
