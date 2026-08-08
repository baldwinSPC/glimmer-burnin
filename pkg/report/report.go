package report

import (
	"time"

	"github.com/baldwinSPC/glimmer-burnin/pkg/contract"
)

// Input is everything a renderer is given.
//
// Only Envelopes is required. Nodes and Artifacts are enrichment a caller
// supplies if it has them — the kubectl plugin reads NodeFingerprint objects,
// the bare-metal CLI probes the host it just tested, an ingest service uses its
// own inventory. A caller with none of that still gets a valid report; it is
// simply a report that says less, which is the correct failure mode.
type Input struct {
	// Envelopes are the deliveries describing ONE run. Resolve refuses a mix
	// from different runs rather than rendering them as one document.
	Envelopes []*contract.Envelope

	// Nodes describes the hardware, richer than Envelope.Fingerprint can.
	// Optional; when absent, Resolve falls back to the fingerprint map and says
	// so in Warnings.
	Nodes []NodeInfo

	// Artifacts are the payloads behind the ArtifactRefs on a result. The
	// envelope carries only references — a consumer that wants the bytes fetches
	// them and passes them here. Optional: a reference with no payload still
	// renders as "this evidence exists, here is where".
	Artifacts []Artifact

	// Meta stamps provenance into every document.
	Meta Meta
}

// Meta identifies what produced a document and when.
//
// Every renderer is required to emit this. A report that cannot say what wrote
// it is not evidence, and the NVVS-schema renderer in particular MUST carry it —
// emitting another vendor's schema is only defensible if the document says whose
// it actually is.
type Meta struct {
	// Generator names the producing software, e.g. "glimmer-burnin".
	Generator string
	// Version is the generator's version. Empty is allowed and renders as
	// absent; a guessed version would be worse than none.
	Version string
	// GeneratedAt is stamped by the caller rather than read from the clock here,
	// so a golden test can produce byte-identical output.
	GeneratedAt time.Time
}

// NodeInfo describes one machine.
//
// A plain-Go mirror of the NodeFingerprint status shape, so that a caller can
// populate it from the CRD, from a local host probe, or from its own inventory
// without this package learning about any of them.
type NodeInfo struct {
	Name    string
	OSImage string
	Kernel  string
	Arch    string
	GPUs    []GPUInfo
	NICs    []NICInfo
}

// GPUInfo is one accelerator.
//
// Serial is frequently unknown — a GB10 fingerprint carries none — and an
// unknown serial stays empty so a renderer omits the field rather than emitting
// a blank one. Downstream, NVVS has a "GPU Device Serials" key that a consumer
// may reasonably trust; filling it with "" would be worse than not writing it.
type GPUInfo struct {
	Index         int32
	Vendor        string
	Model         string
	Arch          string
	MemoryBytes   int64
	DriverVersion string
	Serial        string
}

// NICInfo is one network interface.
type NICInfo struct {
	Name       string
	Role       string // management | fabric | other
	Model      string
	PCIVendor  string
	RDMADevice string
	LinkLayer  string // ethernet | infiniband
	SpeedMbps  int32
	MTU        int32
}

// Artifact is the payload behind an ArtifactRef.
//
// Name and TestName together identify which reference this satisfies. Node
// disambiguates when the same test ran on several machines.
type Artifact struct {
	TestName  string
	Node      string
	Name      string
	MediaType string
	Data      []byte
}

// Output is one rendered file.
//
// Renderers return a slice because some formats are inherently multi-file: the
// NVVS schema is single-host by construction, so a run across eight nodes
// renders eight documents rather than one that misrepresents its own shape.
type Output struct {
	Filename string
	Data     []byte
}

// Renderer turns a resolved run into documents.
type Renderer interface {
	// Name is the format's stable identifier, used as the CLI's -o value.
	Name() string
	// ContentType is the MIME type of the Outputs.
	ContentType() string
	// Render must be deterministic: the same Resolved renders byte-identically
	// every time, which is what makes golden-fixture tests meaningful.
	Render(Resolved) ([]Output, error)
}

// Phase strings a result or a run can hold.
//
// Duplicated from api/v1alpha1 rather than imported, for the reason stated in
// the package doc: this package stays free of Kubernetes types. The duplication
// is narrow and guarded by TestPhaseVocabularyMatchesTheAPI, which fails if the
// API grows a phase this package has not been taught.
const (
	PhasePending   = "Pending"
	PhaseRunning   = "Running"
	PhasePassed    = "Passed"
	PhaseFailed    = "Failed"
	PhaseError     = "Error"
	PhaseSkipped   = "Skipped"
	PhaseCancelled = "Cancelled"
)

// IsTerminal reports whether a phase is final.
//
// Skipped counts: a run never takes that phase but a test does, and a skipped
// test is finished.
func IsTerminal(phase string) bool {
	switch phase {
	case PhasePassed, PhaseFailed, PhaseError, PhaseSkipped, PhaseCancelled:
		return true
	default:
		return false
	}
}

// Cause values carried on a Violation, restated for renderers that group by
// them. The vocabulary is pkg/verdict's; these are the strings that cross the
// envelope boundary.
const (
	// CauseMeasurement — the hardware fell short. This one is about the part.
	CauseMeasurement = "Measurement"
	// CauseEvidence — the runner's report could not support a judgement. The
	// node is unjudged, NOT condemned, and a renderer must not present it as a
	// hardware verdict.
	CauseEvidence = "Evidence"
	// CauseAuthoring — the threshold itself is broken. No hardware is implicated
	// and no node should be touched.
	CauseAuthoring = "Authoring"
)
