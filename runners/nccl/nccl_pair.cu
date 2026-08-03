// nccl_pair: a two-rank NCCL all-reduce benchmark, one process per node.
//
// SPDX-License-Identifier: Apache-2.0
// Copyright the Glimmer authors.
//
// ── WHY THIS EXISTS RATHER THAN nccl-tests ───────────────────────────────────
//
// NVIDIA's nccl-tests is the obvious tool and it is BSD-3-Clause, so licensing
// is not the reason. The reason is the launcher. nccl-tests only spans more than
// one process when it is BUILT WITH MPI, and then it must be started by mpirun,
// which needs a launcher (ssh, or a resource manager) able to start a process on
// the OTHER node. A Kubernetes Pair is two independently scheduled pods with no
// such launcher between them, and adding one would mean shipping an sshd in a
// burn-in image and giving it a key — a standing remote-execution service on
// every accelerator node in the fleet, to run a bandwidth test.
//
// The operator already provides exactly the rendezvous that is missing: it
// starts a server on one node and a client on the other and tells each where the
// other is. So this program takes its rank from the wrapper, exchanges NCCL's
// bootstrap handle over one TCP connection, and runs the same collective
// nccl-tests runs, computing algorithm and bus bandwidth WITH THE SAME FORMULAS
// (see busBandwidth below). What is lost relative to nccl-tests is its breadth
// of collectives and its multi-node topology reporting; what is gained is that
// two pods are enough.
//
// ── WHY THE UNIQUE ID IS EXCHANGED HERE AND NOT BY THE WRAPPER ───────────────
//
// ncclGetUniqueId() does not merely mint a token: it opens a listening socket IN
// THE CALLING PROCESS and records its address inside the ID. The process that
// generates the ID must therefore be the same process that later joins the
// communicator as rank 0. A wrapper that generated the ID in a separate
// invocation and passed it in would hand over the address of a socket that had
// already closed, and the symptom would be a hang inside ncclCommInitRank that
// looks exactly like a dead fabric.
//
// ── OUTPUT ───────────────────────────────────────────────────────────────────
//
// One RESULT line per message size, and nothing else machine-readable. The
// wrapper — not this program — decides which single figure becomes
// busBandwidthGBs, because the operator's parser is last-occurrence-wins and a
// program that printed "busbw=" once per size would silently report only the
// last size measured. See README.md.

#include <arpa/inet.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <sys/socket.h>
#include <unistd.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <string>
#include <vector>

#include <cuda_runtime.h>
#include <nccl.h>

namespace {

// Exit codes are the runner contract's, so a crash of this program is still an
// Error and never a hardware verdict.
constexpr int kOK = 0;
constexpr int kError = 3;

#define CUDA_CHECK(expr)                                                         \
	do {                                                                         \
		cudaError_t _e = (expr);                                                 \
		if (_e != cudaSuccess) {                                                 \
			std::fprintf(stderr, "cuda error at %s:%d: %s (%s)\n", __FILE__,     \
			             __LINE__, cudaGetErrorString(_e), #expr);               \
			return kError;                                                       \
		}                                                                        \
	} while (0)

#define NCCL_CHECK(expr)                                                         \
	do {                                                                         \
		ncclResult_t _e = (expr);                                                \
		if (_e != ncclSuccess) {                                                 \
			std::fprintf(stderr, "nccl error at %s:%d: %s (%s)\n", __FILE__,     \
			             __LINE__, ncclGetErrorString(_e), #expr);               \
			return kError;                                                       \
		}                                                                        \
	} while (0)

// sendAll / recvAll: the ncclUniqueId is 128 bytes and a short read on a
// loopback-fast socket is still possible. Looping is not paranoia, it is the
// difference between a deterministic handshake and an intermittent one.
bool sendAll(int fd, const void *buf, size_t n) {
	const char *p = static_cast<const char *>(buf);
	while (n > 0) {
		ssize_t k = ::send(fd, p, n, 0);
		if (k <= 0) return false;
		p += k;
		n -= static_cast<size_t>(k);
	}
	return true;
}

bool recvAll(int fd, void *buf, size_t n) {
	char *p = static_cast<char *>(buf);
	while (n > 0) {
		ssize_t k = ::recv(fd, p, n, 0);
		if (k <= 0) return false;
		p += k;
		n -= static_cast<size_t>(k);
	}
	return true;
}

// serveUniqueId: rank 0 publishes the bootstrap handle.
//
// The listen backlog is 1 and the socket is closed as soon as the ID is
// delivered, because this port exists for exactly one connection. It is NOT the
// pod's readiness port — the wrapper owns that one, and it must stay separate so
// that a kubelet probe cannot be mistaken for rank 1.
int serveUniqueId(int port, const ncclUniqueId &id) {
	int ln = ::socket(AF_INET, SOCK_STREAM, 0);
	if (ln < 0) return -1;
	int one = 1;
	::setsockopt(ln, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));
	sockaddr_in addr{};
	addr.sin_family = AF_INET;
	addr.sin_addr.s_addr = htonl(INADDR_ANY);
	addr.sin_port = htons(static_cast<uint16_t>(port));
	if (::bind(ln, reinterpret_cast<sockaddr *>(&addr), sizeof(addr)) != 0) {
		::close(ln);
		return -1;
	}
	if (::listen(ln, 1) != 0) {
		::close(ln);
		return -1;
	}
	std::fprintf(stderr, "nccl_pair: rank 0 publishing the NCCL bootstrap handle on port %d\n", port);
	int fd = ::accept(ln, nullptr, nullptr);
	::close(ln);
	if (fd < 0) return -1;
	bool ok = sendAll(fd, &id, sizeof(id));
	::close(fd);
	return ok ? 0 : -1;
}

int fetchUniqueId(const std::string &peer, int port, ncclUniqueId *id) {
	// Retry: rank 1 is released by the wrapper only after rank 0 has been
	// started, but "started" and "bound" are a few milliseconds apart and a
	// single refused connect here would be reported as a fabric fault.
	for (int attempt = 0; attempt < 120; ++attempt) {
		int fd = ::socket(AF_INET, SOCK_STREAM, 0);
		if (fd < 0) return -1;
		sockaddr_in addr{};
		addr.sin_family = AF_INET;
		addr.sin_port = htons(static_cast<uint16_t>(port));
		if (::inet_pton(AF_INET, peer.c_str(), &addr.sin_addr) != 1) {
			::close(fd);
			std::fprintf(stderr, "nccl_pair: %s is not an IPv4 address\n", peer.c_str());
			return -1;
		}
		if (::connect(fd, reinterpret_cast<sockaddr *>(&addr), sizeof(addr)) == 0) {
			bool ok = recvAll(fd, id, sizeof(*id));
			::close(fd);
			if (ok) return 0;
		} else {
			::close(fd);
		}
		::usleep(500 * 1000);
	}
	std::fprintf(stderr, "nccl_pair: could not fetch the bootstrap handle from %s:%d\n", peer.c_str(), port);
	return -1;
}

// busBandwidth converts algorithm bandwidth to bus bandwidth for an all-reduce,
// using nccl-tests' own factor.
//
// The factor is 2*(n-1)/n because a ring all-reduce moves each byte around the
// ring twice — once reducing, once broadcasting — and each rank therefore sees
// 2*(n-1)/n bytes of wire traffic per byte of buffer. AT n=2 THIS FACTOR IS
// EXACTLY 1, so busBandwidthGBs and algBandwidthGBs are equal for a Pair and
// they will not be equal for a larger Group. They are reported as two metrics
// anyway because they are two different quantities that happen to coincide here,
// and collapsing them would make a Pair's history incomparable with a Group's.
double busBandwidth(double algbw, int nranks) {
	return algbw * 2.0 * static_cast<double>(nranks - 1) / static_cast<double>(nranks);
}

struct Options {
	int rank = -1;
	int nranks = 2;
	std::string peer;
	int port = 18535;
	int warmup = 5;
	int iters = 20;
	std::vector<size_t> sizes;
};

bool parseArgs(int argc, char **argv, Options *o) {
	for (int i = 1; i < argc; ++i) {
		std::string a = argv[i];
		auto next = [&](long long *out) {
			if (i + 1 >= argc) return false;
			*out = std::atoll(argv[++i]);
			return true;
		};
		long long v = 0;
		if (a == "--rank" && next(&v)) o->rank = static_cast<int>(v);
		else if (a == "--nranks" && next(&v)) o->nranks = static_cast<int>(v);
		else if (a == "--port" && next(&v)) o->port = static_cast<int>(v);
		else if (a == "--warmup" && next(&v)) o->warmup = static_cast<int>(v);
		else if (a == "--iters" && next(&v)) o->iters = static_cast<int>(v);
		else if (a == "--peer" && i + 1 < argc) o->peer = argv[++i];
		else if (a == "--sizes" && i + 1 < argc) {
			std::string s = argv[++i];
			size_t start = 0;
			while (start <= s.size()) {
				size_t comma = s.find(',', start);
				std::string tok = s.substr(start, comma == std::string::npos ? std::string::npos : comma - start);
				if (!tok.empty()) o->sizes.push_back(static_cast<size_t>(std::atoll(tok.c_str())));
				if (comma == std::string::npos) break;
				start = comma + 1;
			}
		} else {
			std::fprintf(stderr, "nccl_pair: unrecognised argument %s\n", a.c_str());
			return false;
		}
	}
	if (o->sizes.empty()) o->sizes = {8, 1u << 20, 1u << 23, 1u << 26, 1u << 28};
	return o->rank == 0 || o->rank == 1;
}

int runBenchmark(const Options &o) {
	int devices = 0;
	CUDA_CHECK(cudaGetDeviceCount(&devices));
	if (devices < 1) {
		std::fprintf(stderr, "nccl_pair: no CUDA device visible\n");
		return kError;
	}
	CUDA_CHECK(cudaSetDevice(0));

	cudaDeviceProp prop{};
	CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
	std::fprintf(stderr, "nccl_pair: rank %d of %d on %s (sm_%d%d)\n", o.rank, o.nranks, prop.name, prop.major, prop.minor);

	int major = 0, minor = 0, patch = 0;
	ncclGetVersion(&major);
	minor = (major / 100) % 100;
	patch = major % 100;
	std::fprintf(stderr, "nccl_pair: NCCL runtime version %d.%d.%d\n", major / 10000, minor, patch);

	// ── bootstrap ────────────────────────────────────────────────────────────
	ncclUniqueId id;
	if (o.rank == 0) {
		NCCL_CHECK(ncclGetUniqueId(&id));
		if (serveUniqueId(o.port, id) != 0) {
			std::fprintf(stderr, "nccl_pair: could not publish the bootstrap handle on port %d\n", o.port);
			return kError;
		}
	} else {
		if (fetchUniqueId(o.peer, o.port, &id) != 0) return kError;
	}

	// The largest buffer is allocated ONCE and reused for every size, so the
	// measurement is of the collective and not of an allocator.
	size_t maxBytes = 0;
	for (size_t s : o.sizes) maxBytes = s > maxBytes ? s : maxBytes;
	size_t maxCount = maxBytes / sizeof(float);
	if (maxCount == 0) maxCount = 1;

	float *sendBuf = nullptr, *recvBuf = nullptr;
	CUDA_CHECK(cudaMalloc(&sendBuf, maxCount * sizeof(float)));
	CUDA_CHECK(cudaMalloc(&recvBuf, maxCount * sizeof(float)));

	// Every rank contributes 1.0, so a correct SUM all-reduce leaves nranks in
	// every element. That makes the correctness check a real check rather than a
	// tautology: a collective that silently dropped a rank's contribution
	// produces 1.0 and is caught, and one that corrupted data produces neither.
	std::vector<float> host(maxCount, 1.0f);
	CUDA_CHECK(cudaMemcpy(sendBuf, host.data(), maxCount * sizeof(float), cudaMemcpyHostToDevice));

	cudaStream_t stream;
	CUDA_CHECK(cudaStreamCreate(&stream));

	ncclComm_t comm;
	NCCL_CHECK(ncclCommInitRank(&comm, o.nranks, id, o.rank));

	long long wrong = 0;

	for (size_t bytes : o.sizes) {
		size_t count = bytes / sizeof(float);
		if (count == 0) count = 1;

		for (int i = 0; i < o.warmup; ++i) {
			NCCL_CHECK(ncclAllReduce(sendBuf, recvBuf, count, ncclFloat, ncclSum, comm, stream));
		}
		CUDA_CHECK(cudaStreamSynchronize(stream));

		auto t0 = std::chrono::steady_clock::now();
		for (int i = 0; i < o.iters; ++i) {
			NCCL_CHECK(ncclAllReduce(sendBuf, recvBuf, count, ncclFloat, ncclSum, comm, stream));
		}
		CUDA_CHECK(cudaStreamSynchronize(stream));
		auto t1 = std::chrono::steady_clock::now();

		double seconds = std::chrono::duration<double>(t1 - t0).count() / static_cast<double>(o.iters);
		double actualBytes = static_cast<double>(count * sizeof(float));
		double algbw = actualBytes / seconds / 1e9; // GB/s, decimal, as nccl-tests reports
		double busbw = busBandwidth(algbw, o.nranks);

		// Correctness, checked at every size rather than once: a fault that only
		// appears above a protocol switchover threshold (NCCL changes algorithm
		// with message size) would be invisible to a single-size check.
		std::vector<float> back(count);
		CUDA_CHECK(cudaMemcpy(back.data(), recvBuf, count * sizeof(float), cudaMemcpyDeviceToHost));
		const float expect = static_cast<float>(o.nranks);
		for (size_t i = 0; i < count; ++i) {
			if (back[i] != expect) ++wrong;
		}

		std::printf("RESULT size_bytes=%zu time_us=%.3f algbw_gbs=%.4f busbw_gbs=%.4f\n",
		            count * sizeof(float), seconds * 1e6, algbw, busbw);
		std::fflush(stdout);
	}

	std::printf("RESULT_WRONG %lld\n", wrong);
	std::fflush(stdout);

	ncclCommDestroy(comm);
	CUDA_CHECK(cudaStreamDestroy(stream));
	CUDA_CHECK(cudaFree(sendBuf));
	CUDA_CHECK(cudaFree(recvBuf));
	return kOK;
}

} // namespace

int main(int argc, char **argv) {
	Options o;
	if (!parseArgs(argc, argv, &o)) {
		std::fprintf(stderr,
		             "usage: nccl_pair --rank <0|1> [--nranks 2] [--peer <ipv4>] [--port N]\n"
		             "                 [--warmup N] [--iters N] [--sizes b1,b2,...]\n");
		return kError;
	}
	if (o.rank == 1 && o.peer.empty()) {
		std::fprintf(stderr, "nccl_pair: rank 1 needs --peer\n");
		return kError;
	}
	return runBenchmark(o);
}
