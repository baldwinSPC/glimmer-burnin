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
	"fmt"
	"sort"
	"strings"

	"github.com/baldwinSPC/glimmer-burnin/pkg/contract"

	api "github.com/baldwinSPC/glimmer-burnin/api/v1alpha1"
)

// image is a default runner image and the vendor's hardware it can measure.
//
// THE VENDOR IS NOT DECORATION. Every image this project ships is an NVIDIA
// image except the two that touch no accelerator at all, so "fall through to the
// kind's default" is a claim about hardware — and on a mixed fleet, usually a
// false one. Recording it here makes the claim checkable instead of assumed,
// which is the same discipline pkg/contract applies by refusing a metric with an
// Unspecified Aggregation: the answer lives beside the name.
type image struct {
	Ref string
	// Vendor is VendorAny where the image measures no accelerator and therefore
	// runs on any node. Never empty; TestEveryDefaultDeclaresItsVendor refuses
	// that, because an undeclared vendor would silently read as "serves
	// everything" — the fail-open this column exists to close.
	Vendor string
}

// Vendor names, matching NodeFingerprint's own vocabulary, which is derived from
// the DNS domain of the label or resource that published the fact. They are
// LOOKUP KEYS: nothing in this package or in internal/controller branches on
// their values.
const (
	VendorNVIDIA = "nvidia"
	// VendorAny marks an image that measures no accelerator and so runs
	// anywhere. Deliberately a character no DNS domain can produce, so it can
	// never collide with a real vendor name.
	VendorAny = "*"
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
// All eleven images are v0.6.0, published to GHCR, public, and anonymously
// pullable.
//
// v0.6.0 IS THE FIRST RELEASE WHERE SOME OF THESE RAN ON REAL SILICON BEFORE
// PUBLICATION — the rule below ("publish only after the kernel has been verified
// on real hardware") is now partly satisfied rather than knowingly broken, and
// the split is recorded here rather than left for a reader to assume.
//
// Verified on a two-node DGX Spark (GB10, CC 12.1, driver 580.82.09) against
// THESE sources before the tag:
//
//	compute-smoke   FP4_GEMM_PASS — the sm_121a path executed and got the right
//	                answer, which is the one claim this runner exists to make
//	host-health     passed on both nodes, 28 metrics, nvmlStatus=ok
//	thermal-soak    a full 300 s window, peaking at 80 C
//
// NOT verified on hardware, and shipping on the unit, envtest and kind e2e tiers
// alone: clockprobe, dcgm-diag, memory-bw, memory-stress, gpu-burn, ib-write-bw,
// nccl, gpudirect-rdma. Two further gaps that no tier covers: NO 3-RANK
// COLLECTIVE HAS EVER EXECUTED through the N-rank path on any hardware, and CI
// builds runner images by CHANGED DIRECTORY — so a runner whose source did not
// move is republished from sources CI did not rebuild for this release.
//
// One measured caveat that belongs with these tags rather than in a release
// note, because it decides how a soak is scheduled: on this fleet a
// thermal-soak pod is SIGKILLed after 60-106 s IN KUBERNETES while the identical
// image runs its full window under the bare-metal CLI on the same GPU. That is
// not thermal — it was reproduced from a 53 C start, dying at 68 C — and it is
// unresolved (issue #280). A long soak scheduled through the operator on a
// GB10 will not complete today.
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
var defaults = map[contract.TestKind]image{
	// compute-smoke moves off v0.1.0 deliberately. That tag is public and
	// immutable and stays exactly as it is, but it reports "no usable CUDA
	// device", a wrong-arch image, and every CUDA runtime error as exit 1 —
	// a hardware FAILURE verdict against the node. v0.2.0 was the first build
	// where those are exit 3, Error, hardware unjudged. Anyone still pinning
	// v0.1.0 keeps the old behaviour, which is why those are new tags and not
	// repushed ones.
	contract.KindComputeSmoke: {Ref: "ghcr.io/baldwinspc/glimmer-burnin-compute-smoke:v0.6.0", Vendor: VendorNVIDIA},

	contract.KindClockProbe:   {Ref: "ghcr.io/baldwinspc/glimmer-burnin-clockprobe:v0.6.0", Vendor: VendorNVIDIA},
	contract.KindDCGMDiag:     {Ref: "ghcr.io/baldwinspc/glimmer-burnin-dcgm-diag:v0.6.0", Vendor: VendorNVIDIA},
	contract.KindHostHealth:   {Ref: "ghcr.io/baldwinspc/glimmer-burnin-host-health:v0.6.0", Vendor: VendorNVIDIA},
	contract.KindMemoryBW:     {Ref: "ghcr.io/baldwinspc/glimmer-burnin-memory-bw:v0.6.0", Vendor: VendorNVIDIA},
	contract.KindMemoryStress: {Ref: "ghcr.io/baldwinspc/glimmer-burnin-memory-stress:v0.6.0", Vendor: VendorAny},
	contract.KindThermalSoak:  {Ref: "ghcr.io/baldwinspc/glimmer-burnin-thermal-soak:v0.6.0", Vendor: VendorNVIDIA},
	contract.KindGPUBurn:      {Ref: "ghcr.io/baldwinspc/glimmer-burnin-gpu-burn:v0.6.0", Vendor: VendorNVIDIA},
	contract.KindIBWriteBW:    {Ref: "ghcr.io/baldwinspc/glimmer-burnin-ib-write-bw:v0.6.0", Vendor: VendorAny},
	contract.KindNCCL:         {Ref: "ghcr.io/baldwinspc/glimmer-burnin-nccl:v0.6.0", Vendor: VendorNVIDIA},
	contract.KindGPUDirect:    {Ref: "ghcr.io/baldwinspc/glimmer-burnin-gpudirect-rdma:v0.6.0", Vendor: VendorNVIDIA},

	// KindCustom has no default by definition: it exists so a user can point
	// any image at the contract, and inventing a default would defeat it.
}

// Default returns the image that implements a kind, and whether one exists.
//
// The bool is the whole interface. A kind with no default is not an oversight:
// it fails fast and legibly at plan time asking for spec.runner.image, which is
// far better than scheduling a pod with no image and getting an
// ImagePullBackOff that reads as a hardware fault.
func Default(kind contract.TestKind) (string, bool) {
	e, ok := defaults[kind]
	return e.Ref, ok
}

// VendorOf reports which vendor's hardware a kind's default image can measure.
//
// VendorAny means it measures no accelerator at all and therefore runs anywhere.
func VendorOf(kind contract.TestKind) (string, bool) {
	e, ok := defaults[kind]
	return e.Vendor, ok
}

// All returns every default, copied so a caller cannot edit the table.
//
// A mutable package-level map shared by two dispatchers is a way for one of them
// to change what the other runs.
func All() map[contract.TestKind]string {
	out := make(map[contract.TestKind]string, len(defaults))
	for k, v := range defaults {
		out[k] = v.Ref
	}
	return out
}

// WithoutDefault lists the kinds that deliberately have no image here, so a test
// can tell "decided against" from "forgotten".
//
// KindCustom exists so a user can point any image at the runner contract;
// inventing a default for it would defeat the point.
//
// KindTCPBaseline, KindDiskIO, KindFingerprintProbe and KindGemmSweep have
// runner source in this repo but NO PUBLISHED IMAGE yet.
// Publishing is manual and hardware-gated by policy, and a default pointing at
// a tag that does not exist is worse than no default at all: it turns a
// plan-time error that names the problem into an ImagePullBackOff on every
// targeted node. Add it here the moment the tag is published, and delete this
// paragraph with it.
func WithoutDefault() []contract.TestKind {
	return []contract.TestKind{
		contract.KindCustom, contract.KindTCPBaseline, contract.KindDiskIO, contract.KindFingerprintProbe,
		contract.KindFabricSoak,
		contract.KindGemmSweep,
	}
}

// Resolve picks the image that runs a test on a node of a given vendor.
//
// THIS IS THE ONE LADDER, and it is here rather than in either dispatcher for
// the reason this whole package exists: the operator is not the only thing that
// runs these images. A bare-metal path runs the same runners on a host that is
// not a cluster member, and when it kept its own copy of the resolution it
// resolved a DIFFERENT image for the same test on the same hardware — the exact
// failure the package comment above already records once, reintroduced through
// imagesByVendor. Two dispatchers, one ladder.
//
// vendor is what the node said about itself: NodeFingerprint's derived vendor in
// the cluster, hostinfo's PCI-derived vendor on bare metal. EMPTY MEANS UNKNOWN
// and is not a mismatch — a node nothing has fingerprinted declared nothing, and
// absence is not a declaration. That case is every cluster before this field
// existed and it resolves exactly as it did then.
//
// The order is: an explicit image, then the vendor's own entry, then the kind's
// default — but only if the default can actually measure this node.
func Resolve(kind contract.TestKind, runner *api.RunnerSpec, vendor string) (string, error) {
	if runner != nil && runner.Image != "" {
		// Pinned by the author for every node. Their call, including on a mixed
		// fleet, and no vendor check applies: naming an image IS the declaration.
		return runner.Image, nil
	}
	if runner != nil {
		for _, vi := range runner.ImagesByVendor {
			if vi.Vendor == vendor && vi.Image != "" {
				return vi.Image, nil
			}
		}
	}

	// THE FALL-THROUGH IS A CLAIM ABOUT HARDWARE, so it is checked.
	//
	// Before the vendor column this branch was unconditional, and since every
	// built-in default is an NVIDIA image, `imagesByVendor: [{nvidia: ...}]` on
	// an AMD node quietly pulled the NVIDIA one. What happens next belongs to the
	// runner and ranges from wasteful to catastrophic: a well-behaved runner
	// exits 3 (Error, unjudged, retryable — fine); compute-smoke v0.1.0 is on
	// record exiting 1 for exactly "no usable CUDA device", which is a permanent
	// hardware indictment with the retry budget unspent; and a runner exiting 2
	// with a declared _SKIP marker reports acceptance as not-applicable to
	// hardware nobody measured.
	//
	// UNKNOWN VENDOR IS NOT A MISMATCH. A node nothing has fingerprinted declared
	// nothing, and absence is not a declaration — that case is every cluster
	// before this field existed and resolves exactly as it did then. The refusal
	// fires only where the node positively declared a vendor and the image
	// positively declared it cannot measure it. Two declarations, not an
	// inference.
	e, hasDefault := defaults[kind]
	if hasDefault && (e.Vendor == VendorAny || vendor == "" || e.Vendor == vendor) {
		return e.Ref, nil
	}

	// Unresolvable. An ERROR — machinery, hardware unjudged, retryable — and
	// never a Fail, and never a skip: a node silently not being tested is how a
	// fleet gets certified without being measured.
	if runner != nil && len(runner.ImagesByVendor) > 0 {
		// The author has already shown they know about the field: they listed
		// some vendors and not this one, which on a mixed fleet is far more
		// likely to be an oversight than a decision. Reported separately because
		// the fix is different.
		return "", fmt.Errorf(
			"no image for vendor %s on kind %q: spec.runner.imagesByVendor lists %s, and %s. "+
				"Add a %s entry, or set spec.runner.image to pin one image for every node",
			vendorOrUnknown(vendor), kind, listVendors(runner.ImagesByVendor),
			defaultClause(e, hasDefault), vendorOrUnknown(vendor))
	}
	if !hasDefault {
		return "", fmt.Errorf("no default runner image for kind %q — set spec.runner.image", kind)
	}
	return "", fmt.Errorf(
		"no image for vendor %s on kind %q: %s, and spec.runner.imagesByVendor is not set. "+
			"Add a %s entry to it, or set spec.runner.image to pin one image for every node",
		vendorOrUnknown(vendor), kind, defaultClause(e, hasDefault), vendorOrUnknown(vendor))
}

// defaultClause says what the built-in default can do for this node, which is
// the half of the message that tells the reader whether adding an entry is
// even necessary.
func defaultClause(e image, hasDefault bool) string {
	if !hasDefault {
		return "this kind has no built-in default to fall back to"
	}
	return fmt.Sprintf("the built-in default measures %q hardware", e.Vendor)
}

// vendorOrUnknown keeps an empty vendor from rendering as `vendor ""`, which
// sends the reader looking for a typo in their YAML. The real problem is that
// nothing established what this node is.
//
// Worded for BOTH dispatchers, because both call this: in a cluster the vendor
// comes from a NodeFingerprint, on bare metal from the PCI bus.
func vendorOrUnknown(vendor string) string {
	if vendor == "" {
		return `"unknown" (nothing established an accelerator vendor for this node — ` +
			`no fingerprint in-cluster, no PCI vendor match on bare metal)`
	}
	return `"` + vendor + `"`
}

// listVendors renders the declared vendors in a stable order for a message.
func listVendors(list []api.VendorImage) string {
	names := make([]string, 0, len(list))
	for _, vi := range list {
		names = append(names, vi.Vendor)
	}
	sort.Strings(names)
	return strings.Join(names, ", ")
}
