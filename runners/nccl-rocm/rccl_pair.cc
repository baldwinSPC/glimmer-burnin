// rccl_pair: an N-rank RCCL all-reduce benchmark, one process per node — the
// AMD runner image for the "nccl" TestKind, selected per node via
// spec.runner.imagesByVendor ({vendor: amd}).
//
// SPDX-License-Identifier: Apache-2.0
// Copyright the Glimmer authors.
//
// ── THE KIND IS CALLED "nccl" AND THIS RUNS RCCL. THAT IS DELIBERATE ─────────
//
// The TestKind names a MEASURAND — "all-reduce bus-bandwidth over the fabric" —
// not a library. RCCL is AMD's port of NCCL and declares the same C API
// (ncclGetUniqueId, ncclCommInitRank, ncclAllReduce…), versions itself against
// NCCL releases, and reads the same NCCL_* environment variables. rccl-tests
// computes busbw with the identical 2*(n-1)/n factor. So both vendors' images
// measure the same quantity in the same units and emit the same metric name,
// and one profile with imagesByVendor serves a mixed fleet.
//
// A separate "rccl" kind would fragment that: every profile targeting both
// vendors would need two entries, and a fleet dashboard would need to know
// which axis to read. The vocabulary is inherited and slightly awkward on AMD;
// the alternative is worse.
//
// ── WHY THIS EXISTS RATHER THAN rccl-tests ───────────────────────────────────
//
// The same reason nccl_pair.cu exists rather than nccl-tests, and it survives
// the port intact: rccl-tests spans more than one process only when built with
// MPI, and then it must be started by mpirun, which needs a launcher able to
// start a process on the OTHER node. A Kubernetes Pair is two independently
// scheduled pods with no launcher between them, and adding one would mean
// shipping an sshd in a burn-in image — a standing remote-execution service on
// every accelerator node in the fleet, to run a bandwidth test.
//
// The operator already provides the rendezvous that is missing: it starts one
// pod per node and tells each which rank it is and where rank 0 answers.
//
// ── WHAT THIS PORT DOES NOT CARRY ────────────────────────────────────────────
//
// The NVIDIA image is a Go wrapper around its harness that also raises memlock
// limits and configures NCCL_IB_* for an RDMA fabric. This runner is TCP/socket
// only, which is what AMD themselves support on this hardware: the ROCm 7.13
// notes describe multi-node collectives for Ryzen AI Max 300 "connected over
// Ethernet", with no RDMA requirement. A RoCE path (the community E810 setup)
// is a follow-up, not a silent omission — see the README.
//
// ── RCCL NEEDS ROCm 7.12 OR NEWER, AND 7.2.x WILL NOT DO ─────────────────────
//
// gfx1151 was added to RCCL's DEFAULT_GPUS by rocm-systems PR #3415 (merged
// 2026-02-26) and first shipped in 7.12. The rocm-7.2.x line is a maintenance
// branch that never received it, so a "7.x" floor is not sufficient — see the
// Dockerfile, which pins accordingly and asserts at build time.
//
// OUTPUT CONTRACT
//   metrics as key=value lines, ALWAYS printed before the decision, then one of
//     NCCL_PAIR_PASS               exit 0
//     NCCL_PAIR_FAIL:  <why>       exit 1   the collective ran and was WRONG
//     NCCL_PAIR_SKIP:  <why>       exit 2   not applicable to this hardware
//     NCCL_PAIR_ERROR: <why>       exit 3   unjudged; NOT a hardware verdict

#include <arpa/inet.h>
#include <netdb.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include <hip/hip_runtime.h>
#include <rccl/rccl.h>

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "collective_bw.h"

namespace {

constexpr int kExitPass = 0;
constexpr int kExitFail = 1;
constexpr int kExitSkip = 2;
constexpr int kExitError = 3;

int emitMarker(const char *marker, const std::string &why, int code) {
	if (why.empty()) {
		std::printf("%s\n", marker);
	} else {
		std::printf("%s: %s\n", marker, why.c_str());
	}
	return code;
}

int pass() { return emitMarker("NCCL_PAIR_PASS", "", kExitPass); }
int fail(const std::string &w) { return emitMarker("NCCL_PAIR_FAIL", w, kExitFail); }
int skip(const std::string &w) { return emitMarker("NCCL_PAIR_SKIP", w, kExitSkip); }
int errored(const std::string &w) { return emitMarker("NCCL_PAIR_ERROR", w, kExitError); }

std::string env(const char *k) {
	const char *v = std::getenv(k);
	return v == nullptr ? std::string() : std::string(v);
}

bool envInt(const char *k, int dflt, int *out) {
	const std::string v = env(k);
	if (v.empty()) {
		*out = dflt;
		return true;
	}
	try {
		std::size_t pos = 0;
		const int parsed = std::stoi(v, &pos);
		if (pos != v.size()) return false;
		*out = parsed;
		return true;
	} catch (...) {
		return false;
	}
}

// ── socket rendezvous ────────────────────────────────────────────────────────
// Ported from nccl_pair.cu unchanged in behaviour, including the hello
// handshake below, which is hard-won and must not be simplified away.

bool sendAll(int fd, const void *buf, std::size_t n) {
	const char *p = static_cast<const char *>(buf);
	while (n > 0) {
		const ssize_t k = ::send(fd, p, n, 0);
		if (k <= 0) return false;
		p += k;
		n -= static_cast<std::size_t>(k);
	}
	return true;
}

bool recvAll(int fd, void *buf, std::size_t n) {
	char *p = static_cast<char *>(buf);
	while (n > 0) {
		const ssize_t k = ::recv(fd, p, n, 0);
		if (k <= 0) return false;
		p += k;
		n -= static_cast<std::size_t>(k);
	}
	return true;
}

// kRankHello is the four bytes a fetching rank sends before it is counted.
//
// A CONNECTION IS NOT A RANK, and counting accepts instead of ranks strands
// one. A kubelet tcpSocket probe is exactly connect-then-close, and sendAll
// succeeds writing into the send buffer even when the peer has already sent
// FIN — so without this handshake a probe is indistinguishable from a rank that
// received the handle. Rank 0 would then close its listener believing every
// peer was served while a real rank got ECONNREFUSED, and the collective would
// block until something killed it. That was reproduced on the NVIDIA harness;
// the port keeps the fix rather than rediscovering the bug.
constexpr std::uint32_t kRankHello = 0x4E43434CU;  // "NCCL"
constexpr int kHelloTimeoutSec = 5;

int listenOn(int port, int backlog) {
	const int ln = ::socket(AF_INET, SOCK_STREAM, 0);
	if (ln < 0) return -1;
	int one = 1;
	::setsockopt(ln, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));
	sockaddr_in addr{};
	addr.sin_family = AF_INET;
	addr.sin_addr.s_addr = htonl(INADDR_ANY);
	addr.sin_port = htons(static_cast<std::uint16_t>(port));
	if (::bind(ln, reinterpret_cast<sockaddr *>(&addr), sizeof(addr)) != 0 ||
	    ::listen(ln, backlog) != 0) {
		::close(ln);
		return -1;
	}
	return ln;
}

// serveUniqueId: rank 0 publishes the bootstrap handle to exactly nranks-1
// peers. The count is load-bearing in both directions — serving fewer leaves a
// rank that can never join, serving more holds a socket for a peer that does
// not exist.
int serveUniqueId(int port, int nranks, const ncclUniqueId &id) {
	const int waiting = nranks - 1;
	const int ln = listenOn(port, waiting + 8);
	if (ln < 0) return -1;
	std::fprintf(stderr, "rccl_pair: rank 0 publishing the RCCL bootstrap handle on port %d to %d peer(s)\n",
	             port, waiting);

	int served = 0, strangers = 0;
	while (served < waiting) {
		const int fd = ::accept(ln, nullptr, nullptr);
		if (fd < 0) {
			::close(ln);
			return -1;
		}
		timeval tv{};
		tv.tv_sec = kHelloTimeoutSec;
		::setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));

		std::uint32_t hello = 0;
		if (!recvAll(fd, &hello, sizeof(hello)) || ntohl(hello) != kRankHello) {
			::close(fd);
			if (++strangers % 10 == 1) {
				std::fprintf(stderr,
				             "rccl_pair: ignored %d connection(s) that did not identify as a rank "
				             "(a readiness probe looks like this); still waiting for %d\n",
				             strangers, waiting - served);
			}
			continue;
		}
		const bool ok = sendAll(fd, &id, sizeof(id));
		::close(fd);
		if (!ok) {
			::close(ln);
			return -1;
		}
		++served;
		std::fprintf(stderr, "rccl_pair: served the bootstrap handle to %d of %d peer(s)\n", served,
		             waiting);
	}
	::close(ln);
	return 0;
}

// resolveHost turns either an IPv4 literal or a Service DNS name into an
// address. The operator hands ranks a headless-Service name, so the DNS path is
// the normal one and the literal path is the fallback, not the reverse.
bool resolveHost(const std::string &host, in_addr *out) {
	if (::inet_pton(AF_INET, host.c_str(), out) == 1) return true;
	addrinfo hints{};
	hints.ai_family = AF_INET;
	hints.ai_socktype = SOCK_STREAM;
	addrinfo *res = nullptr;
	if (::getaddrinfo(host.c_str(), nullptr, &hints, &res) != 0 || res == nullptr) return false;
	*out = reinterpret_cast<sockaddr_in *>(res->ai_addr)->sin_addr;
	::freeaddrinfo(res);
	return true;
}

// fetchUniqueId: every rank but 0 pulls the bootstrap handle from the root.
//
// It RETRIES for a budget rather than failing on the first refusal, because the
// ranks are independently scheduled pods: a client that starts before rank 0
// has bound its listener gets ECONNREFUSED, and that is ordinary startup
// ordering rather than a fabric fault. Resolution is retried too — a headless
// Service's DNS record does not exist until its first endpoint is ready.
int fetchUniqueId(const std::string &peer, int port, int budgetSec, ncclUniqueId *id) {
	const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(budgetSec);
	while (std::chrono::steady_clock::now() < deadline) {
		sockaddr_in addr{};
		addr.sin_family = AF_INET;
		addr.sin_port = htons(static_cast<std::uint16_t>(port));
		if (!resolveHost(peer, &addr.sin_addr)) {
			::sleep(1);
			continue;
		}
		const int fd = ::socket(AF_INET, SOCK_STREAM, 0);
		if (fd < 0) return -1;
		if (::connect(fd, reinterpret_cast<sockaddr *>(&addr), sizeof(addr)) == 0) {
			// Identify as a rank before the root will count us; see kRankHello.
			const std::uint32_t hello = htonl(kRankHello);
			if (sendAll(fd, &hello, sizeof(hello)) && recvAll(fd, id, sizeof(*id))) {
				::close(fd);
				return 0;
			}
		}
		::close(fd);
		::sleep(1);
	}
	std::fprintf(stderr, "rccl_pair: could not fetch the bootstrap handle from %s:%d within the budget\n",
	             peer.c_str(), port);
	return -1;
}

}  // namespace

#define HIP_CHECK(x)                                                                     \
	do {                                                                                 \
		const hipError_t e_ = (x);                                                       \
		if (e_ != hipSuccess) {                                                          \
			return errored(std::string("hip error: ") + hipGetErrorString(e_) +           \
			               "; hardware unjudged");                                        \
		}                                                                                \
	} while (0)

#define RCCL_CHECK(x)                                                                    \
	do {                                                                                 \
		const ncclResult_t r_ = (x);                                                     \
		if (r_ != ncclSuccess) {                                                          \
			return errored(std::string("rccl error: ") + ncclGetErrorString(r_) +         \
			               "; hardware unjudged");                                        \
		}                                                                                \
	} while (0)

int main() {
	// ── who am I ─────────────────────────────────────────────────────────────
	// Group scope sets BURNIN_RANK/BURNIN_NRANKS/BURNIN_ROOT_HOST; Pair scope
	// sets BURNIN_ROLE plus BURNIN_PEER_HOST. Reading the Group variables FIRST
	// matters: at Group scope BURNIN_ROLE is deliberately unset, and a runner
	// that branched on its absence would read a Group execution as a Node one
	// and skip — certifying a collective that never ran (issue #118).
	int rank = 0, nranks = 0, durationSeconds = 0, port = 0, bootstrapBudget = 0;
	if (!envInt("BURNIN_RANK", 0, &rank) || !envInt("BURNIN_NRANKS", 0, &nranks) ||
	    !envInt("BURNIN_DURATION_SECONDS", 60, &durationSeconds) ||
	    !envInt("NCCL_PAIR_PORT", 29500, &port) ||
	    !envInt("NCCL_PAIR_BOOTSTRAP_TIMEOUT_S", 300, &bootstrapBudget)) {
		return errored("a BURNIN_* or NCCL_PAIR_* variable is not an integer");
	}
	const std::string role = env("BURNIN_ROLE");
	std::string rootHost = env("BURNIN_ROOT_HOST");

	if (nranks == 0) {
		// Pair scope: two ranks, server is rank 0 and the client fetches from
		// BURNIN_PEER_HOST.
		if (role.empty()) {
			return skip("no BURNIN_ROLE and no BURNIN_NRANKS: this image runs a collective and "
			            "needs at least two ranks, so it is not applicable to a Node-scope run");
		}
		nranks = 2;
		rank = (role == "server") ? 0 : 1;
		if (rootHost.empty()) rootHost = env("BURNIN_PEER_HOST");
	}

	// A COLLECTIVE OF ONE MEASURES NOTHING. The bus-bandwidth factor 2*(n-1)/n
	// is exactly zero at one rank, so such a run would print busbw=0.00 on
	// perfectly healthy hardware and fail every floor gate forever. Refused as
	// a config error rather than reported as a number.
	if (nranks < 2) {
		return errored("a collective needs at least two ranks; this run was given " +
		               std::to_string(nranks) +
		               ", whose bus-bandwidth factor is zero by construction");
	}
	if (rank < 0 || rank >= nranks) {
		return errored("rank " + std::to_string(rank) + " is outside a set of " +
		               std::to_string(nranks));
	}
	if (rank != 0 && rootHost.empty()) {
		return errored("rank " + std::to_string(rank) +
		               " has no root host to fetch the bootstrap handle from");
	}

	std::printf("rank=%d\nnranks=%d\n", rank, nranks);

	// ── the device ───────────────────────────────────────────────────────────
	int devCount = 0;
	const hipError_t de = hipGetDeviceCount(&devCount);
	if (de != hipSuccess || devCount == 0) {
		return errored(std::string("no usable HIP device (") +
		               (de == hipSuccess ? "zero devices" : hipGetErrorString(de)) +
		               "); a collective needs one — hardware unjudged");
	}
	HIP_CHECK(hipSetDevice(0));
	hipDeviceProp_t props;
	std::memset(&props, 0, sizeof(props));
	HIP_CHECK(hipGetDeviceProperties(&props, 0));
	std::printf("gpu_name=%s\ngfx_target=%s\n", props.name, props.gcnArchName);

	int major = 0, minor = 0, patch = 0;
	{
		int version = 0;
		if (ncclGetVersion(&version) == ncclSuccess) {
			major = version / 10000;
			minor = (version / 100) % 100;
			patch = version % 100;
			std::printf("rccl_version=%d.%d.%d\n", major, minor, patch);
		}
	}

	// ── readiness ────────────────────────────────────────────────────────────
	// A listening socket the kernel completes handshakes on without this
	// process ever calling accept(), so a kubelet tcpSocket probe succeeds
	// while the benchmark runs — and, critically, on a DIFFERENT port from the
	// bootstrap listener, so a probe can never be mistaken for a rank even
	// before the hello handshake gets a chance to reject it.
	int readinessPort = 0;
	if (!envInt("NCCL_PAIR_READY_PORT", 29499, &readinessPort)) {
		return errored("NCCL_PAIR_READY_PORT is not an integer");
	}
	const int readyFd = listenOn(readinessPort, 64);
	if (readyFd < 0) {
		return errored("could not open the readiness listener on port " +
		               std::to_string(readinessPort));
	}
	std::printf("readiness_port=%d\nbootstrap_port=%d\n", readinessPort, port);

	// ── bootstrap ────────────────────────────────────────────────────────────
	// ncclGetUniqueId does not merely mint a token: it opens a listening socket
	// IN THE CALLING PROCESS and records its address inside the ID. The process
	// that generates it must be the same one that joins as rank 0, which is why
	// this happens here rather than in a wrapper.
	ncclUniqueId id;
	if (rank == 0) {
		RCCL_CHECK(ncclGetUniqueId(&id));
		if (serveUniqueId(port, nranks, id) != 0) {
			::close(readyFd);
			return errored("rank 0 could not publish the bootstrap handle on port " +
			               std::to_string(port) + "; hardware unjudged");
		}
	} else {
		if (fetchUniqueId(rootHost, port, bootstrapBudget, &id) != 0) {
			::close(readyFd);
			return errored("rank " + std::to_string(rank) + " could not fetch the bootstrap handle from " +
			               rootHost + "; hardware unjudged");
		}
	}

	// ── the collective ───────────────────────────────────────────────────────
	const std::size_t maxCount = 32 * 1024 * 1024;  // 128 MiB of float
	float *sendBuf = nullptr, *recvBuf = nullptr;
	HIP_CHECK(hipMalloc(&sendBuf, maxCount * sizeof(float)));
	HIP_CHECK(hipMalloc(&recvBuf, maxCount * sizeof(float)));

	std::vector<float> host(maxCount, 1.0f);
	HIP_CHECK(hipMemcpy(sendBuf, host.data(), maxCount * sizeof(float), hipMemcpyHostToDevice));

	hipStream_t stream;
	HIP_CHECK(hipStreamCreate(&stream));

	ncclComm_t comm;
	RCCL_CHECK(ncclCommInitRank(&comm, nranks, id, rank));

	burnin::Sweep sweep;
	long long wrong = 0;
	const auto started = std::chrono::steady_clock::now();

	for (std::size_t count = 256 * 1024; count <= maxCount; count *= 4) {
		const std::size_t bytes = count * sizeof(float);

		// Warm up so the timed pass excludes connection setup and first-touch.
		RCCL_CHECK(ncclAllReduce(sendBuf, recvBuf, count, ncclFloat, ncclSum, comm, stream));
		HIP_CHECK(hipStreamSynchronize(stream));

		const auto t0 = std::chrono::steady_clock::now();
		RCCL_CHECK(ncclAllReduce(sendBuf, recvBuf, count, ncclFloat, ncclSum, comm, stream));
		HIP_CHECK(hipStreamSynchronize(stream));
		const double seconds =
		    std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();

		double algbw = 0;
		if (burnin::algBandwidthGBs(bytes, seconds, &algbw)) {
			const double busbw = burnin::busBandwidth(algbw, nranks);
			sweep.observe(bytes, algbw, busbw);
			std::fprintf(stderr, "rccl_pair: size=%zu time=%.3fus algbw=%.4f busbw=%.4f\n", bytes,
			             seconds * 1e6, algbw, busbw);
		}

		// Every rank summed 1.0, so every element must be exactly nranks. A
		// collective that ran fast and produced the wrong answer is the one
		// hardware verdict this runner can return.
		HIP_CHECK(hipMemcpy(host.data(), recvBuf, bytes, hipMemcpyDeviceToHost));
		for (std::size_t i = 0; i < count; i++) {
			if (host[i] != static_cast<float>(nranks)) {
				wrong++;
				break;
			}
		}
		HIP_CHECK(hipMemcpy(sendBuf, std::vector<float>(count, 1.0f).data(), bytes,
		                    hipMemcpyHostToDevice));

		if (std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count() >
		    static_cast<double>(durationSeconds)) {
			break;
		}
	}

	ncclCommDestroy(comm);
	(void)hipStreamDestroy(stream);
	(void)hipFree(sendBuf);
	(void)hipFree(recvBuf);
	::close(readyFd);

	const double elapsed =
	    std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
	std::printf("elapsed_s=%.2f\n", elapsed);

	if (!sweep.any) {
		return errored("no message size produced a timed result; hardware unjudged");
	}
	// ONE line per metric, the peak of the sweep — see collective_bw.h on why
	// the selection happens there rather than by emitting one line per size.
	std::printf("peak_size_bytes=%zu\nalgbw_gbs=%.4f\nbusbw_gbs=%.4f\n", sweep.peakSizeBytes,
	            sweep.peakAlgBw, sweep.peakBusBw);

	if (wrong > 0) {
		return fail("the all-reduce returned incorrect data at " + std::to_string(wrong) +
		            " message size(s): every element should equal the rank count");
	}
	return pass();
}
