// Package nvvs renders a run in the schema NVIDIA's own diagnostic emits.
//
// # Why borrow someone else's schema
//
// An engineer qualifying accelerators already knows what `dcgmi diag -j` output
// looks like. Their tooling parses it, their runbooks reference its fields, and
// an RMA conversation goes faster when the attachment has the shape the vendor's
// own tool produces. This operator measures strictly more than DCGM does for the
// tests it runs — per-threshold violations classified by cause, link-scoped
// verdicts, gates that were measured rather than assumed — and delivering all of
// that in a shape nobody reads is a self-inflicted wound.
//
// # What is and is not being claimed
//
// These documents are STRUCTURALLY COMPATIBLE with the NVVS schema. They are not
// NVIDIA output, not endorsed by NVIDIA, and not a claim to have run NVIDIA's
// tests. Three rules keep that honest and none of them is negotiable:
//
//   - Every document carries provenance in aux_data naming this generator and
//     stating plainly that NVIDIA did not produce it.
//   - Our test names stay ours. A synthesised entry is never labelled with an
//     NVIDIA plugin name like sm_stress, because a consumer must not be able to
//     mistake our reconstruction for that plugin's own result. NVVS consumers
//     iterate test_categories generically, so an unfamiliar name renders fine.
//   - A field we did not measure is ABSENT. Not empty, not "unknown", not
//     zero. An empty serial number in an RMA is a claim that the part has none.
//
// # Key names are pinned, the nesting is not yet
//
// The field names below are taken from NVIDIA's own nvvs/include/NvvsJsonStrings.h
// (Apache-2.0), which is what makes structural compatibility achievable rather
// than approximate — they are constants in the vendor's source, not guesses.
//
// The NESTING of metadata, entity_groups and "Overall Result" relative to the
// "DCGM Diagnostic" object is now CHECKED against a real capture, and the check
// found it wrong.
//
// The capture was already here: runners/dcgm-diag/diagjson_test.go holds
// `dcgm4GB10`, a genuine `dcgmi diag -j` document from this project's own GB10
// fleet under DCGM 4.2.3. Its root keys are ["DCGM Diagnostic", "entity_groups",
// "metadata"], with only "test_categories" inside the first — so metadata,
// entity_groups and "Overall Result" are ROOT SIBLINGS, not children.
// TestTheDocumentTreeMatchesARealCapture compares the two documents directly, so
// the tree cannot be re-inferred.
//
// What remains unverified is narrower and is stated where it applies: the
// entity_group_id enum values, and the key spellings this package deliberately
// does not emit.
package nvvs

// Field names, verbatim from NvvsJsonStrings.h. Kept as constants so a typo is
// a compile error and a rename upstream is a single edit here.
const (
	keyRoot         = "DCGM Diagnostic"
	keyVersion      = "version"
	keyRuntimeError = "runtime_error"
	keyGlobalWarn   = "Warning"
	keyMetadata     = "metadata"
	keyEntityGroups = "entity_groups"
	keyCategories   = "test_categories"
	keyOverall      = "Overall Result"
	keyAuxData      = "aux_data"

	keyDriverVersion = "Driver Version Detected"
	keyGPUDeviceIDs  = "GPU Device IDs"
	keyGPUSerials    = "GPU Device Serials"
)

// Document is the top-level object.
//
// ONLY test_categories nests under "DCGM Diagnostic". metadata, entity_groups,
// "Overall Result" and the document-level error and warning are SIBLINGS of it,
// at the root.
//
// This was inferred, and inferred wrong, until it was compared with a real
// capture. The capture was already in this repository: runners/dcgm-diag/
// diagjson_test.go holds `dcgm4GB10`, a genuine `dcgmi diag -j` document from
// this project's own GB10 fleet under DCGM 4.2.3, and its root keys are
// exactly ["DCGM Diagnostic", "entity_groups", "metadata"] with only
// "test_categories" inside the first.
//
// The consequence of the old shape was total rather than cosmetic: a consumer
// reading the real tree looks for metadata and entity_groups at the root, finds
// neither, and gets a document with no driver version, no serials and no entity
// inventory — while every key name in it is correct, so nothing looks wrong.
// TestTheDocumentTreeMatchesARealCapture pins it against the capture itself, so
// this cannot be re-inferred.
type Document struct {
	Diagnostic Diagnostic `json:"DCGM Diagnostic"`

	Metadata     *Metadata     `json:"metadata,omitempty"`
	EntityGroups []EntityGroup `json:"entity_groups,omitempty"`
	// OverallResult is the host's verdict. Kept distinct from a per-test status
	// so that a reader who only looks here is not misled.
	OverallResult string `json:"Overall Result,omitempty"`
	// RuntimeError is set when the diagnostic itself could not run. We use it
	// for a run that reached no verdict, which is a statement about the run and
	// not about the hardware.
	RuntimeError string `json:"runtime_error,omitempty"`
	// Warning is the document-level caveat: a baseline sweep, a partial record.
	Warning string   `json:"Warning,omitempty"`
	AuxData *AuxData `json:"aux_data,omitempty"`
}

// Diagnostic is one host's report. NVVS is single-host by construction —
// dcgmi runs on one machine — which is why a multi-node run renders one of
// these per node rather than a single document that misrepresents its shape.
type Diagnostic struct {
	Version string `json:"version,omitempty"`
	// Categories is the ONLY key that nests here. Everything else that used to
	// live in this struct is a root sibling; see Document.
	Categories []Category `json:"test_categories"`
}

// Metadata is the environment the diagnostic observed.
//
// Every field is omitempty and every field is genuine-or-absent. DriverVersion
// appears only when a fingerprint or an inventory carried one; DCGMVersion only
// when a real dcgmi artifact was passed through, because we have no DCGM of our
// own to name and inventing a version would be the exact fabrication this
// package refuses.
type Metadata struct {
	DCGMVersion   string   `json:"version,omitempty"`
	DriverVersion string   `json:"Driver Version Detected,omitempty"`
	GPUDeviceIDs  []string `json:"GPU Device IDs,omitempty"`
	GPUSerials    []string `json:"GPU Device Serials,omitempty"`
}

// EntityGroup collects the things results attach to.
//
// EntityGroup is the NAME ("GPU", "CPU"), which the vendor's own vocabulary
// fixes and we can state confidently. The numeric entity_group_id that NVVS
// also carries is deliberately OMITTED: its values come from a DCGM enum this
// project does not vendor, and a wrong number is worse than a missing one — a
// consumer keying on it would attach our results to the wrong kind of device.
type EntityGroup struct {
	EntityGroup string   `json:"entity_group"`
	Entities    []Entity `json:"entities"`
}

// Entity is one device.
type Entity struct {
	EntityID int32 `json:"entity_id"`
	// SerialNum and DeviceID are omitted when uncaptured. GB10 fingerprints
	// carry no serial, and this is the field an RMA turns on.
	SerialNum string `json:"serial_num,omitempty"`
	DeviceID  string `json:"device_id,omitempty"`
}

// Category groups tests the way NVVS does: Deployment, Integration, Hardware,
// Stress.
type Category struct {
	Category string `json:"category"`
	Tests    []Test `json:"tests"`
}

// Test is one test and its per-entity results.
type Test struct {
	Name string `json:"name"`
	// TestSummary is the test's overall status across its entities.
	TestSummary *Result  `json:"test_summary,omitempty"`
	Results     []Result `json:"results"`
}

// Result is one test's outcome for one entity.
type Result struct {
	Status string `json:"status"`
	// EntityGroup and EntityID say what this result is about. Omitted on a
	// summary, which is about the test rather than about a device.
	EntityGroup string `json:"entity_group,omitempty"`
	EntityID    *int32 `json:"entity_id,omitempty"`
	// Warnings carry failures and the reasons behind them.
	Warnings []Warning `json:"warnings,omitempty"`
	// Info carries everything measured and everything not evaluated. NVVS
	// treats it as free text, which is where our metrics and our
	// not-evaluated thresholds go without pretending to be vendor fields.
	Info []string `json:"info,omitempty"`
}

// Warning is one thing that went wrong.
//
// ErrorID is NOT emitted. NVVS carries NVIDIA's own DCGM_FR_* codes there, and
// a synthesised entry inventing one would let a consumer look up a vendor
// diagnosis for a condition the vendor never diagnosed. Category and severity
// are ours, drawn from the violation's own classification, which is information
// NVVS has no equivalent for and is strictly additive.
type Warning struct {
	Warning string `json:"warning"`
	// ErrorCategory carries the violation's Cause: whether the hardware fell
	// short, the evidence could not support a judgement, or the threshold is
	// broken. It is the field to read first and NVVS has nothing like it.
	ErrorCategory string `json:"error_category,omitempty"`
	ErrorSeverity string `json:"error_severity,omitempty"`
}

// AuxData is where this document says what it is.
//
// Mandatory on every document. A schema-compatible report with no provenance is
// indistinguishable from vendor output once it has been forwarded twice, and
// that is not a risk worth taking to save four fields.
type AuxData struct {
	Generator string `json:"generator"`
	Schema    string `json:"schema"`
	Notice    string `json:"notice"`

	RunUID      string `json:"run_uid,omitempty"`
	RunName     string `json:"run_name,omitempty"`
	Profile     string `json:"profile,omitempty"`
	DeliveryID  string `json:"delivery_id,omitempty"`
	Cluster     string `json:"cluster,omitempty"`
	GeneratedAt string `json:"generated_at,omitempty"`
	Node        string `json:"node,omitempty"`

	// Baseline says the run measured rather than certified. A thresholdless
	// sweep that reads as an acceptance is the fail-open this exists to close.
	Baseline bool `json:"baseline,omitempty"`
	// Warnings are what the report could not establish — a run with no verdict,
	// evidence that was too large to store. Surfaced, never swallowed.
	Warnings []string `json:"warnings,omitempty"`
}

// The notice every document carries.
const (
	schemaTag = "NVVS-compatible"
	notice    = "Schema-compatible with the DCGM diagnostic JSON format. " +
		"Generated by glimmer-burnin; NOT produced by, or endorsed by, NVIDIA. " +
		"Test names are this project's own and do not correspond to NVIDIA plugins."
)
