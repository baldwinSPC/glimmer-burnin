// Package report turns a stored burn-in verdict into a document.
//
// # Why this is not in the sinks
//
// The three sinks deliver JSON to a webhook, a ConfigMap or Prometheus. That is
// TRANSPORT. Nothing in this repository turned a verdict into something a person
// reads or a pipeline consumes, so every consumer built its own renderer and
// they disagreed — and the only report anyone could produce was a closed one,
// while an operator running this project standalone got nothing.
//
// # Kubernetes stops at the door
//
// This package depends on pkg/contract and the standard library. Nothing else.
// Not api/v1alpha1 — pkg/verdict takes that dependency, and apimachinery with
// it, because it must speak Threshold; a consumer that only wants to render a
// result should not pay that cost. The audience here is CI jobs, a CLI and a
// third-party ingest, which is exactly the audience that should not be pulling
// in client-go to format a table. TestNoKubernetesDependency asserts it against
// the real build graph rather than against a promise in this comment.
//
// # A report never fabricates
//
// This is the rule the whole package is built around, and every renderer
// inherits it. A serial number that was not captured is ABSENT — not the empty
// string, not "unknown", not "N/A". The distinction is the same one the verdict
// engine spends its whole existence preserving: a measurement nobody took and a
// measurement that came back clean are different facts, and a report that
// renders them identically converts the first into the second. An RMA
// conversation opened on a fabricated serial is worse than one opened with a
// gap in it.
//
// The typed shapes below express that in the type system where they can: a
// field a caller could not populate is left at its zero value and renderers must
// omit it, and the few fields where "zero" is a legitimate MEASUREMENT rather
// than an absence are pointers.
package report

import (
	"time"

	"github.com/baldwinSPC/glimmer-burnin/pkg/contract"
)

// Input is everything a renderer is given. It is assembled by a caller — the
// CLI from a results directory, a kubectl plugin from the cluster, a consumer
// from its own inventory — and rendered without further I/O.
//
// A renderer is a pure function of this struct. That is deliberate: a renderer
// that could read a file or call an API would produce documents that depend on
// when they were rendered, and a burn-in report's whole value is that it
// describes one moment that has already passed.
type Input struct {
	// Envelopes describes one run, in any order, with at least one entry.
	//
	// The terminal RunPhaseChanged delivery is the VERDICT OF RECORD. Earlier
	// TestCompleted and Checkpoint deliveries are enrichment: they carry
	// mid-flight metrics and per-test detail that the terminal envelope's own
	// Results may have overwritten, and a long soak's progress is only visible
	// through them.
	Envelopes []*contract.Envelope

	// Nodes is the structured hardware inventory, and is OPTIONAL.
	//
	// When it is empty a renderer falls back to Envelope.Fingerprint, which is
	// a flat map of whatever the run captured, and omits everything it cannot
	// know. It never fills a gap with a placeholder.
	Nodes []NodeInfo

	// Artifacts is raw evidence a runner returned — a dcgmi JSON document, a
	// captured topology. Renderers that understand a media type may merge it;
	// the rest ignore it. A renderer must never RE-SYNTHESISE something it was
	// handed for real: a genuine vendor document and our reconstruction of one
	// are different evidence, and only the first is worth attaching to an RMA.
	Artifacts []Artifact

	// Meta is provenance. It is mandatory in every rendered document.
	Meta Meta
}

// Meta identifies what produced a document, so a reader can tell a report this
// project generated from one a vendor's own tool did.
//
// It is not decoration. A renderer that emits a vendor-compatible schema is
// producing something that LOOKS like the vendor's output, and the difference
// has to survive being forwarded, attached and quoted.
type Meta struct {
	// Generator names the producing tool, e.g. "glimmer-burnin".
	Generator string
	// Version is the generator's version.
	Version string
	// GeneratedAt is when the document was rendered — not when the run
	// happened, which is in the envelope.
	GeneratedAt time.Time
}

// NodeInfo is a plain-Go mirror of a NodeFingerprint's status.
//
// It is a MIRROR and not an import for the reason in the package comment. It is
// populated by the caller: the kubectl plugin from the CRs, a CLI from a host
// probe, a consumer from its own inventory.
type NodeInfo struct {
	// Name is the node this describes. It is the only required field: an
	// inventory entry that cannot say which node it is about is not evidence.
	Name string

	GPUs []GPUInfo
	NICs []NICInfo

	// OSImage, Kernel and Digest are omitted when empty. A report that printed
	// "kernel: unknown" would read as a captured observation.
	OSImage string
	Kernel  string
	Digest  string

	// Labels is the hardware-identity subset of the node's labels, never a copy
	// of all of them — scheduling state and the burn-in verdict labels this
	// operator's own consumers write are not hardware identity.
	Labels map[string]string

	// CapturedAt is when this identity was last observed to CHANGE, not when it
	// was last looked at. Nil when the caller does not know.
	CapturedAt *time.Time
}

// GPUInfo mirrors v1alpha1.GPUInfo.
type GPUInfo struct {
	// Index is the device ordinal, and is meaningful at zero, so it is not
	// omitted the way an empty string is.
	Index int32
	// Vendor is nvidia|amd|intel|tenstorrent, or empty when it was not
	// established. Never "unknown": a report does not guess.
	Vendor string
	Model  string // "NVIDIA GB10"
	Arch   string // "sm_121"
	// MemoryBytes is zero when uncaptured. Zero is not a plausible measurement
	// for an accelerator's memory, so the ambiguity that motivates a pointer
	// elsewhere does not arise here.
	MemoryBytes int64
	DriverVer   string
	// Serial is the field an RMA turns on, and the one most often absent. It is
	// omitted rather than blanked; see the package comment.
	Serial string
	// PCIBusID identifies the physical slot, which is what an engineer walking
	// to the rack needs.
	PCIBusID string
}

// NICInfo mirrors v1alpha1.NICInfo.
type NICInfo struct {
	Name       string
	Role       string // management|fabric|other
	PCIVendor  string // "0x15b3"
	Model      string // "ConnectX-7"
	RDMADevice string // "mlx5_0"
	LinkLayer  string // ethernet|infiniband
	SpeedMbps  int32
	MTU        int32
}

// Artifact is one piece of raw evidence, already fetched by the caller.
//
// It mirrors what internal/controller stored and what the envelope referenced,
// with the payload resolved: pkg/report does no I/O, so whoever assembles the
// Input is the one that reads the ConfigMap or the file.
type Artifact struct {
	// Node is which node produced it; empty for a run-scoped artifact.
	Node string
	// Test is which test produced it.
	Test string
	// Name is the runner's own name for it, e.g. "dcgmi.json".
	Name string
	// MediaType is what the payload IS.
	MediaType string
	// Payload is the evidence.
	Payload []byte
	// Digest is "sha256:<hex>" as recorded when it was stored, so a renderer
	// that passes it through can state it was not modified.
	Digest string
}

// Output is one rendered file. A renderer may return several — the NVVS
// renderer emits one document per node, because that schema is single-host.
type Output struct {
	// Filename is a base name with no directory component. The caller decides
	// where documents land; a renderer that chose a path could escape the
	// directory it was given.
	Filename string
	Data     []byte
}

// Renderer turns an Input into documents.
//
// Render must not mutate its Input. Callers render the same run through several
// renderers, and a renderer that normalised something in place would change what
// the next one saw — so two formats of the same run would disagree, which is
// the exact failure this package exists to end.
type Renderer interface {
	// Name is the format's identifier as a user types it: "junit", "html",
	// "nvvs-json", "markdown".
	Name() string
	// ContentType is the MIME type of the rendered documents.
	ContentType() string
	Render(Input) ([]Output, error)
}
