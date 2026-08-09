// Package runnerimages is the one place that says which image implements which
// TestKind.
//
// It lives here rather than in internal/controller because the operator is not
// the only thing that runs these images. GEP-0178's design is one brain and two
// dispatchers: a bare-metal path runs the same runners on a host that is not a
// cluster member, and if it kept its own copy of this table the two would drift.
// The drift would be invisible until a node passed under one dispatcher and
// failed under the other — and this solution already has one instance of that
// exact failure on record, a second implementation running a DIFFERENT test
// under the same TestKind name.
//
// So the table is public, both dispatchers read it, and a release that repins
// every image edits one file.
package runnerimages

import (
	api "github.com/baldwinSPC/glimmer-burnin/api/v1alpha1"
)

// defaults maps a TestKind to its default runner image. A kind
// without an entry requires spec.runner.image — scheduling a pod with no image
// would produce an ImagePullBackOff blamed on the hardware.
//
// Images are pinned by version, never :latest: a readiness/acceptance verdict
// must be reproducible, and a floating tag changes the test under every
// existing profile with no audit trail.
//
// PUBLICATION STATUS — every entry below is published, public, and immutable.
//
// All eleven images are v0.5.0, published to GHCR, public, and anonymously
// pullable.
//
// THESE TAGS WERE NOT VERIFIED ON REAL HARDWARE BEFORE PUBLICATION, which this
// file's own rule below ("publish only after the kernel has been verified on
// real hardware") requires. The GPU fleet was unavailable and the release was
// cut anyway, deliberately and with the trade understood. What DOES stand behind
// them: the unit, envtest and kind-based e2e tiers, including a real three-node
// rendezvous for Group scope, and four rounds of adversarial review that found
// twenty defects in this release's own code. What does not: no GB10 has run
// these exact tags, NO 3-RANK COLLECTIVE HAS EVER EXECUTED through the N-rank
// path on any hardware, and CI builds runner images by CHANGED DIRECTORY — so
// the runners whose sources did not move are republished from sources CI did not
// rebuild for this release.
//
// The v0.3.0 tags remain published and immutable, and the measurements behind
// THEM were taken on a two-node DGX Spark cluster with the v0.2.0 build of the
// same sources: the Node-scope suite passed 10/10 across both nodes, and the
// Pair-scope fabric suite passed with ib-write-bw at 99.61 Gb/s, nccl at
// 12.02 GB/s bus bandwidth, and gpudirect-rdma correctly reporting exit 2 on
// hardware whose unified memory has no peer-memory provider. Pin v0.3.0 if
// hardware-verified images matter more to you than the v0.4.0 fixes — but read
// the release notes first: two of those fixes are false negatives that were
// certifying hardware nobody measured.
//
// ARCHITECTURE: linux/amd64 AND linux/arm64. A node pulls only its own
// platform's layers, so one tag serves an x86 rack and a Grace rack alike, and
// x86 users no longer need to set spec.runner.image to run at all.
//
// That is the HOST axis and it is not the GPU axis. A tag resolving on a host
// says nothing about whether the image contains code for the accelerator in it —
// compute-smoke carries only the CC 12.x kernel and reports a B200 as an ERROR,
// hardware unjudged — not a Skip, because a B200 does do NVFP4 and the test
// applies to it — and nccl ships one gencode that an unlisted fleet must
// rebuild. Each runner's README states its
// GPU coverage; conflating the two is how an operator ends up reporting Error on
// hardware it simply was not built for.
//
// A missing entry fails fast and legibly at plan time ("set spec.runner.image");
// a present-but-unpullable entry fails slowly, per node, and looks like an
// infrastructure fault rather than a configuration one. That asymmetry is why
// nothing is listed here before it is actually published.
//
// The names follow the publish workflow's own construction —
// ghcr.io/<owner lowercased>/glimmer-burnin-<kind>:<version> — so an entry here
// and a workflow_dispatch of publish-runner with the same version agree by
// construction rather than by anyone remembering to keep them in step.
var defaults = map[api.TestKind]string{
	// compute-smoke moves off v0.1.0 deliberately. That tag is public and
	// immutable and stays exactly as it is, but it reports "no usable CUDA
	// device", a wrong-arch image, and every CUDA runtime error as exit 1 —
	// a hardware FAILURE verdict against the node. v0.2.0 was the first build
	// where those are exit 3, Error, hardware unjudged. Anyone still pinning
	// v0.1.0 keeps the old behaviour, which is why those are new tags and not
	// repushed ones.
	api.KindComputeSmoke: "ghcr.io/baldwinspc/glimmer-burnin-compute-smoke:v0.5.0",

	api.KindClockProbe:   "ghcr.io/baldwinspc/glimmer-burnin-clockprobe:v0.5.0",
	api.KindDCGMDiag:     "ghcr.io/baldwinspc/glimmer-burnin-dcgm-diag:v0.5.0",
	api.KindHostHealth:   "ghcr.io/baldwinspc/glimmer-burnin-host-health:v0.5.0",
	api.KindMemoryBW:     "ghcr.io/baldwinspc/glimmer-burnin-memory-bw:v0.5.0",
	api.KindMemoryStress: "ghcr.io/baldwinspc/glimmer-burnin-memory-stress:v0.5.0",
	api.KindThermalSoak:  "ghcr.io/baldwinspc/glimmer-burnin-thermal-soak:v0.5.0",
	api.KindGPUBurn:      "ghcr.io/baldwinspc/glimmer-burnin-gpu-burn:v0.5.0",
	api.KindIBWriteBW:    "ghcr.io/baldwinspc/glimmer-burnin-ib-write-bw:v0.5.0",
	api.KindNCCL:         "ghcr.io/baldwinspc/glimmer-burnin-nccl:v0.5.0",
	api.KindGPUDirect:    "ghcr.io/baldwinspc/glimmer-burnin-gpudirect-rdma:v0.5.0",

	// KindCustom has no default by definition: it exists so a user can point
	// any image at the contract, and inventing a default would defeat it.
}

// Default returns the image that implements a kind, and whether one exists.
//
// The bool is the whole interface. A kind with no default is not an oversight:
// it fails fast and legibly at plan time asking for spec.runner.image, which is
// far better than scheduling a pod with no image and getting an
// ImagePullBackOff that reads as a hardware fault.
func Default(kind api.TestKind) (string, bool) {
	img, ok := defaults[kind]
	return img, ok
}

// All returns every default, copied so a caller cannot edit the table.
//
// A mutable package-level map shared by two dispatchers is a way for one of them
// to change what the other runs.
func All() map[api.TestKind]string {
	out := make(map[api.TestKind]string, len(defaults))
	for k, v := range defaults {
		out[k] = v
	}
	return out
}

// WithoutDefault lists the kinds that deliberately have no image here, so a test
// can tell "decided against" from "forgotten".
//
// KindCustom exists so a user can point any image at the runner contract;
// inventing a default for it would defeat the point.
//
// KindTCPBaseline and KindDiskIO have runner source in this repo but NO
// PUBLISHED IMAGE yet.
// Publishing is manual and hardware-gated by policy, and a default pointing at
// a tag that does not exist is worse than no default at all: it turns a
// plan-time error that names the problem into an ImagePullBackOff on every
// targeted node. Add it here the moment the tag is published, and delete this
// paragraph with it.
func WithoutDefault() []api.TestKind {
	return []api.TestKind{api.KindCustom, api.KindTCPBaseline, api.KindDiskIO}
}
