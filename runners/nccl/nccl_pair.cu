// nccl_pair: an N-rank NCCL all-reduce benchmark, one process per node.
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
// starts one pod per node and tells each of them which rank it is and where rank
// 0 answers. So this program takes its rank from the wrapper, exchanges NCCL's
// bootstrap handle over TCP — rank 0 serves it to the other nranks-1 — and runs
// the same collective nccl-tests runs, computing algorithm and bus bandwidth
// WITH THE SAME FORMULAS (see busBandwidth below). What is lost relative to
// nccl-tests is its breadth of collectives and its multi-node topology
// reporting; what is gained is that no launcher is needed at any rank count.
//
// It serves BOTH scopes the operator executes. At Pair scope the wrapper passes
// --nranks 2 and the roles come from BURNIN_ROLE; at Group scope they come from
// BURNIN_RANK/BURNIN_NRANKS and there may be any number of them. The only thing
// that changes inside this file is how many peers rank 0 serves the handle to.
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
//
// ── THE SECOND PATH: --local-multi-gpu ────────────────────────────────────────
//
// Everything above is the ORIGINAL path: one process is one rank, and N of
// them (N=2 at Pair scope, N>=2 at Group scope) find each other over TCP and
// join ONE communicator across NODES. --local-multi-gpu is a second, disjoint
// path added for Node scope: one process, EVERY device the pod was allocated,
// joined into N communicators via ncclCommInitAll — NCCL's own API for a
// single process driving multiple devices, no bootstrap handle exchange
// needed because there is no second process to exchange it with. See
// collective.h for the collective-specific arithmetic and buffer sizing this
// path needs that the single-collective, single-buffer-pair path above never
// did.
//
// It exists because DGX BasePOD validation runs the 8-GPU intra-node
// all-reduce BEFORE any multi-node test, and AMD's CVF gates intra-node
// all-reduce bandwidth the same way — neither is expressible as two pods
// exchanging a handle over a fabric that does not exist between two GPUs in
// one box.

#include <arpa/inet.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <unistd.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <string>
#include <vector>

#include <cuda_runtime.h>
#include <nccl.h>

#include "collective/collective.h"

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

// kRankHello is the four bytes a fetching rank sends before it is counted, and
// kHelloTimeoutSec is how long rank 0 waits for them. The value is arbitrary and
// only has to be something a probe will not send, which is anything at all.
constexpr uint32_t kRankHello = 0x4E43434CU;  // "NCCL"
constexpr int kHelloTimeoutSec = 5;

// serveUniqueId: rank 0 publishes the bootstrap handle to every other rank.
//
// It serves EXACTLY nranks-1 connections and then closes, which is the whole
// difference between two ranks and N. The count is load-bearing in both
// directions: serving fewer leaves a rank that can never join, and the collective
// then blocks until something kills it; serving more would keep a socket open for
// a peer that does not exist.
//
// It is NOT the pod's readiness port — the wrapper owns that one, and it must
// stay separate so that a kubelet probe cannot be mistaken for a rank. That
// separation matters more at Group scope than it did at Pair: a probe accepted
// here would consume one of the N-1 slots and strand a real rank.
int serveUniqueId(int port, int nranks, const ncclUniqueId &id) {
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
	// The backlog holds every peer that connects before we accept it. Sized to
	// the whole cohort because at Group scope the wrapper releases all of them at
	// once, so they arrive together rather than in turn.
	const int waiting = nranks > 1 ? nranks - 1 : 1;
	if (::listen(ln, waiting) != 0) {
		::close(ln);
		return -1;
	}
	std::fprintf(stderr, "nccl_pair: rank 0 publishing the NCCL bootstrap handle on port %d to %d peer(s)\n",
	             port, waiting);

	// A CONNECTION IS NOT A RANK, and counting accepts instead of ranks strands
	// one.
	//
	// A kubelet tcpSocket probe is exactly connect-then-close. sendAll writes 128
	// bytes into the send buffer, which SUCCEEDS even when the peer has already
	// sent FIN — so a probe was indistinguishable from a rank that received the
	// handle. After nranks-1 such accepts this function closed the listener and
	// returned 0, logging that it had served every peer, while a real rank got
	// ECONNREFUSED for its whole fetch budget and the collective blocked until
	// something killed it. Reproduced by compiling this function verbatim and
	// driving it with one probe plus two real fetchers.
	//
	// So a peer must IDENTIFY ITSELF before it is counted. This is the same rule
	// rendezvous.go's acceptPeer already applies to the wrapper's own readiness
	// listener, for the same reason and in the same words: connections that are
	// not the peer are expected and are not errors, they are read with a short
	// deadline and dropped, and the loop keeps waiting for a real one.
	int served = 0;
	int strangers = 0;
	while (served < waiting) {
		int fd = ::accept(ln, nullptr, nullptr);
		if (fd < 0) {
			::close(ln);
			return -1;
		}
		// A short receive deadline: a real rank sends its hello immediately, and a
		// probe sends nothing at all and must not hold the listener.
		timeval tv{};
		tv.tv_sec = kHelloTimeoutSec;
		::setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));

		uint32_t hello = 0;
		if (!recvAll(fd, &hello, sizeof(hello)) || ntohl(hello) != kRankHello) {
			::close(fd);
			if (++strangers % 10 == 1) {
				std::fprintf(stderr, "nccl_pair: ignored %d connection(s) that did not identify as a "
				                     "rank (a readiness probe looks like this); still waiting for %d\n",
				             strangers, waiting - served);
			}
			continue;
		}
		bool ok = sendAll(fd, &id, sizeof(id));
		::close(fd);
		if (!ok) {
			::close(ln);
			return -1;
		}
		++served;
		std::fprintf(stderr, "nccl_pair: served the bootstrap handle to %d of %d peer(s)\n",
		             served, waiting);
	}
	::close(ln);
	return 0;
}

int fetchUniqueId(const std::string &peer, int port, ncclUniqueId *id) {
	// Retry: a non-root rank is released by the wrapper only after rank 0 has
	// been started, but "started" and "bound" are a few milliseconds apart and a
	// single refused connect here would be reported as a fabric fault. At Group
	// scope the whole cohort is released at once, so several ranks may retry
	// against the same listener simultaneously — the backlog in serveUniqueId is
	// what absorbs that.
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
			// Identify as a rank before reading. Rank 0 counts only peers that do
			// this, so that a readiness probe cannot consume a rank's slot.
			const uint32_t hello = htonl(kRankHello);
			bool ok = sendAll(fd, &hello, sizeof(hello)) && recvAll(fd, id, sizeof(*id));
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
	// localMultiGpu selects the SECOND path — see the file header. Every field
	// above is meaningless under it: there is no rank, no peer, no bootstrap
	// port, because there is only one process and it never talks to another.
	bool localMultiGpu = false;
	burnin::Collective collective = burnin::Collective::AllReduce;
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
		else if (a == "--local-multi-gpu") o->localMultiGpu = true;
		else if (a == "--collective" && i + 1 < argc) {
			const std::string raw = argv[++i];
			if (!burnin::ParseCollective(raw, &o->collective)) {
				std::fprintf(stderr,
				             "nccl_pair: --collective %s is not one of allreduce, allgather, "
				             "reducescatter, alltoall\n",
				             raw.c_str());
				return false;
			}
		} else if (a == "--sizes" && i + 1 < argc) {
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
	if (o->localMultiGpu) {
		// rank/nranks/peer are unused under this path; the device count that
		// takes their place is read from the runtime, not the command line,
		// because it is a hardware fact rather than something a wrapper decides.
		return true;
	}
	// Any rank in [0, nranks), and at least two of them: an all-reduce over one
	// rank measures nothing. The upper bound is what stops a wrapper bug from
	// launching a rank the communicator has no room for, which NCCL would report
	// as a hang rather than as the argument error it is.
	return o->nranks >= 2 && o->rank >= 0 && o->rank < o->nranks;
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
		if (serveUniqueId(o.port, o.nranks, id) != 0) {
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

// launchGroup issues one collective across every device, batched as a SINGLE
// NCCL group so the N per-device calls execute as one synchronized operation
// rather than N independent ones. AllToAll has no native NCCL primitive — it
// is composed from nranks*nranks point-to-point ncclSend/ncclRecv pairs inside
// the same group, the pattern NCCL's own user guide gives for implementing an
// all-to-all; every other collective is one native call per device.
//
// A device must be cudaSetDevice'd as current immediately before the NCCL call
// that targets it, group or not — NCCL routes each call by the CURRENT device,
// not by which comms[i] was passed.
int launchGroup(burnin::Collective collective, int devices, const std::vector<int> &devList,
                 std::vector<ncclComm_t> &comms, std::vector<cudaStream_t> &streams,
                 std::vector<float *> &sendBufs, std::vector<float *> &recvBufs,
                 const burnin::BufferPlan &plan) {
	NCCL_CHECK(ncclGroupStart());
	if (collective == burnin::Collective::AllToAll) {
		for (int i = 0; i < devices; ++i) {
			CUDA_CHECK(cudaSetDevice(devList[i]));
			for (int j = 0; j < devices; ++j) {
				NCCL_CHECK(ncclSend(sendBufs[i] + static_cast<size_t>(j) * plan.ChunkCount, plan.ChunkCount,
				                     ncclFloat, j, comms[i], streams[i]));
				NCCL_CHECK(ncclRecv(recvBufs[i] + static_cast<size_t>(j) * plan.ChunkCount, plan.ChunkCount,
				                     ncclFloat, j, comms[i], streams[i]));
			}
		}
	} else {
		for (int i = 0; i < devices; ++i) {
			CUDA_CHECK(cudaSetDevice(devList[i]));
			switch (collective) {
			case burnin::Collective::AllReduce:
				NCCL_CHECK(
				    ncclAllReduce(sendBufs[i], recvBufs[i], plan.SendCount, ncclFloat, ncclSum, comms[i], streams[i]));
				break;
			case burnin::Collective::AllGather:
				NCCL_CHECK(ncclAllGather(sendBufs[i], recvBufs[i], plan.SendCount, ncclFloat, comms[i], streams[i]));
				break;
			case burnin::Collective::ReduceScatter:
				NCCL_CHECK(ncclReduceScatter(sendBufs[i], recvBufs[i], plan.RecvCount, ncclFloat, ncclSum, comms[i],
				                              streams[i]));
				break;
			case burnin::Collective::AllToAll:
				break;  // handled above; unreachable here
			}
		}
	}
	NCCL_CHECK(ncclGroupEnd());
	return kOK;
}

// runLocalMultiGpu is the SECOND path — see the file header. One process,
// every visible device, one collective chosen by --collective, joined via
// ncclCommInitAll rather than the bootstrap handshake runBenchmark uses.
//
// It ignores o.rank, o.nranks, o.peer and o.port entirely: there is one
// process and it never talks to another, so none of those has a meaning here.
int runLocalMultiGpu(const Options &o) {
	int devices = 0;
	CUDA_CHECK(cudaGetDeviceCount(&devices));
	// The wrapper (main.go) already refuses fewer than two devices before ever
	// invoking this binary — a POSITIVE check on the device count, never on
	// BURNIN_ROLE/BURNIN_RANK being absent (see main.go's runNodeScope). Reaching
	// here with devices<2 means the two disagree about the hardware, so refuse
	// loudly rather than silently measure a "collective" with no second rank.
	if (devices < 2) {
		std::fprintf(stderr, "nccl_pair: --local-multi-gpu needs at least 2 visible devices, found %d\n", devices);
		return kError;
	}

	std::vector<int> devList(devices);
	for (int i = 0; i < devices; ++i) devList[i] = i;

	for (int i = 0; i < devices; ++i) {
		cudaDeviceProp prop{};
		CUDA_CHECK(cudaGetDeviceProperties(&prop, i));
		std::fprintf(stderr, "nccl_pair: device %d of %d: %s (sm_%d%d)\n", i, devices, prop.name, prop.major,
		             prop.minor);
	}
	int nvVer = 0;
	ncclGetVersion(&nvVer);
	std::fprintf(stderr, "nccl_pair: NCCL runtime version %d.%d.%d; collective=%s\n", nvVer / 10000,
	             (nvVer / 100) % 100, nvVer % 100, burnin::CollectiveName(o.collective));

	std::vector<ncclComm_t> comms(devices);
	NCCL_CHECK(ncclCommInitAll(comms.data(), devices, devList.data()));

	// The largest buffer is allocated ONCE per device and reused for every
	// size, exactly as runBenchmark does — the measurement is of the
	// collective, not of an allocator.
	size_t maxTotal = 0;
	for (size_t s : o.sizes) maxTotal = s > maxTotal ? s : maxTotal;
	size_t maxCount = maxTotal / sizeof(float);
	if (maxCount == 0) maxCount = 1;
	const burnin::BufferPlan maxPlan = burnin::PlanBuffers(o.collective, maxCount, devices);

	std::vector<cudaStream_t> streams(devices);
	std::vector<float *> sendBufs(devices), recvBufs(devices);

	for (int i = 0; i < devices; ++i) {
		CUDA_CHECK(cudaSetDevice(devList[i]));
		CUDA_CHECK(cudaStreamCreate(&streams[i]));
		CUDA_CHECK(cudaMalloc(&sendBufs[i], maxPlan.BufElements * sizeof(float)));
		CUDA_CHECK(cudaMalloc(&recvBufs[i], maxPlan.BufElements * sizeof(float)));

		// Seeded ONCE, at the largest allocation, and never again — for
		// AllReduce, ReduceScatter and AllGather ONLY. AllReduce and
		// ReduceScatter sum a constant (1.0), and AllGather fills the WHOLE
		// buffer with one value keyed on this device's index; neither pattern
		// depends on where a chunk boundary falls, so the same seed is correct
		// at every swept size. AllToAll is NOT seeded here — see the per-size
		// loop below for why.
		std::vector<float> host(maxPlan.BufElements);
		switch (o.collective) {
		case burnin::Collective::AllReduce:
		case burnin::Collective::ReduceScatter:
			std::fill(host.begin(), host.end(), 1.0f);
			CUDA_CHECK(cudaMemcpy(sendBufs[i], host.data(), maxPlan.BufElements * sizeof(float), cudaMemcpyHostToDevice));
			break;
		case burnin::Collective::AllGather:
			std::fill(host.begin(), host.end(), burnin::SeedForAllGather(i));
			CUDA_CHECK(cudaMemcpy(sendBufs[i], host.data(), maxPlan.BufElements * sizeof(float), cudaMemcpyHostToDevice));
			break;
		case burnin::Collective::AllToAll:
			break;  // seeded per size, below — see why there
		}
	}

	long long wrong = 0;

	for (size_t bytes : o.sizes) {
		size_t totalCount = bytes / sizeof(float);
		if (totalCount == 0) totalCount = 1;
		const burnin::BufferPlan plan = burnin::PlanBuffers(o.collective, totalCount, devices);

		// AllToAll's seed is keyed on CHUNK OFFSET (which peer a range of
		// elements is destined for), and the chunk size varies with the swept
		// message size — plan.ChunkCount here is smaller than maxPlan.ChunkCount
		// at every size but the largest. Seeding once at the max layout and
		// launching at a smaller layout would read each peer's data from the
		// WRONG offset (every peer but the last would receive peer 0's chunk),
		// which the correctness check would then report as a hardware
		// miscompare that never happened — a fabricated Fail, permanent and
		// never retried, on hardware that measured nothing wrong. So this is
		// reseeded at THIS size's own chunk boundaries before every size.
		if (o.collective == burnin::Collective::AllToAll) {
			for (int i = 0; i < devices; ++i) {
				// AllToAll's BufElements is SendCount == RecvCount == the full
				// reconstructed total (ChunkCount * devices) — see PlanBuffers.
				std::vector<float> host(plan.BufElements, 0.0f);
				for (int j = 0; j < devices; ++j) {
					const float v = burnin::SeedForAllToAll(i, j);
					const size_t base = static_cast<size_t>(j) * plan.ChunkCount;
					for (size_t k = 0; k < plan.ChunkCount && base + k < host.size(); ++k) host[base + k] = v;
				}
				CUDA_CHECK(cudaSetDevice(devList[i]));
				CUDA_CHECK(cudaMemcpy(sendBufs[i], host.data(), host.size() * sizeof(float), cudaMemcpyHostToDevice));
			}
		}

		for (int w = 0; w < o.warmup; ++w) {
			int rc = launchGroup(o.collective, devices, devList, comms, streams, sendBufs, recvBufs, plan);
			if (rc != kOK) return rc;
		}
		for (int i = 0; i < devices; ++i) {
			CUDA_CHECK(cudaSetDevice(devList[i]));
			CUDA_CHECK(cudaStreamSynchronize(streams[i]));
		}

		auto t0 = std::chrono::steady_clock::now();
		for (int it = 0; it < o.iters; ++it) {
			int rc = launchGroup(o.collective, devices, devList, comms, streams, sendBufs, recvBufs, plan);
			if (rc != kOK) return rc;
		}
		for (int i = 0; i < devices; ++i) {
			CUDA_CHECK(cudaSetDevice(devList[i]));
			CUDA_CHECK(cudaStreamSynchronize(streams[i]));
		}
		auto t1 = std::chrono::steady_clock::now();

		// The swept size IS the total collective payload — the same "count" the
		// original path has always used, so a fleet reading busBandwidthGBs
		// against a swept size reads it against the same numbers whichever path
		// produced it. algbw is bytes moved per device per second, matching
		// nccl-tests' own convention of reporting per-rank throughput.
		double seconds = std::chrono::duration<double>(t1 - t0).count() / static_cast<double>(o.iters);
		double actualBytes = static_cast<double>(totalCount * sizeof(float));
		double algbw = actualBytes / seconds / 1e9;
		double busbw = burnin::BusBandwidth(o.collective, algbw, devices);

		// Correctness, checked at every size — the same reasoning runBenchmark
		// already gives: a fault that only appears above a protocol switchover
		// threshold would be invisible to a single-size check.
		for (int i = 0; i < devices; ++i) {
			// Every device is cudaSetDevice'd as current before the call that
			// targets it, group or not — see launchGroup's own comment. The
			// same rule applies here: cudaMemcpy is UVA-transparent across
			// devices on this project's own toolchain, but this loop otherwise
			// enters with whichever device the prior synchronize loop left
			// current, and depending on that rather than stating it is the kind
			// of implicit assumption this file elsewhere refuses to make.
			CUDA_CHECK(cudaSetDevice(devList[i]));
			std::vector<float> back(plan.RecvCount);
			CUDA_CHECK(cudaMemcpy(back.data(), recvBufs[i], plan.RecvCount * sizeof(float), cudaMemcpyDeviceToHost));
			switch (o.collective) {
			case burnin::Collective::AllReduce:
			case burnin::Collective::ReduceScatter: {
				const float expect = static_cast<float>(devices);
				for (float v : back) {
					if (v != expect) ++wrong;
				}
				break;
			}
			case burnin::Collective::AllGather:
				for (int seg = 0; seg < devices; ++seg) {
					const float expect = burnin::ExpectedAllGatherSegment(seg);
					const size_t base = static_cast<size_t>(seg) * plan.SendCount;
					for (size_t k = 0; k < plan.SendCount && base + k < back.size(); ++k) {
						if (back[base + k] != expect) ++wrong;
					}
				}
				break;
			case burnin::Collective::AllToAll:
				for (int sender = 0; sender < devices; ++sender) {
					const float expect = burnin::ExpectedAllToAllChunk(sender, i);
					const size_t base = static_cast<size_t>(sender) * plan.ChunkCount;
					for (size_t k = 0; k < plan.ChunkCount && base + k < back.size(); ++k) {
						if (back[base + k] != expect) ++wrong;
					}
				}
				break;
			}
		}

		std::printf("RESULT size_bytes=%zu time_us=%.3f algbw_gbs=%.4f busbw_gbs=%.4f\n", totalCount * sizeof(float),
		            seconds * 1e6, algbw, busbw);
		std::fflush(stdout);
	}

	std::printf("RESULT_WRONG %lld\n", wrong);
	std::fflush(stdout);

	for (int i = 0; i < devices; ++i) {
		CUDA_CHECK(cudaSetDevice(devList[i]));
		ncclCommDestroy(comms[i]);
		CUDA_CHECK(cudaStreamDestroy(streams[i]));
		CUDA_CHECK(cudaFree(sendBufs[i]));
		CUDA_CHECK(cudaFree(recvBufs[i]));
	}
	return kOK;
}

} // namespace

int main(int argc, char **argv) {
	Options o;
	if (!parseArgs(argc, argv, &o)) {
		std::fprintf(stderr,
		             "usage: nccl_pair --rank <0..nranks-1> --nranks <N, at least 2> [--peer <ipv4>] [--port N]\n"
		             "                 [--warmup N] [--iters N] [--sizes b1,b2,...]\n"
		             "   or: nccl_pair --local-multi-gpu [--collective allreduce|allgather|reducescatter|alltoall]\n"
		             "                 [--warmup N] [--iters N] [--sizes b1,b2,...]\n");
		return kError;
	}
	if (o.localMultiGpu) return runLocalMultiGpu(o);
	if (o.rank != 0 && o.peer.empty()) {
		std::fprintf(stderr, "nccl_pair: rank %d needs --peer (the address rank 0 answers on)\n", o.rank);
		return kError;
	}
	return runBenchmark(o);
}
