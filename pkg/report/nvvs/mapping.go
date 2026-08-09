package nvvs

// Phases a result can carry, and the violation causes that explain a failure.
//
// Restated here rather than imported because pkg/report deliberately holds no
// Kubernetes types and does not re-export the API's vocabulary either. The
// duplication is narrow and TestPhaseVocabularyMatchesTheAPI keeps it honest:
// a phase the API grows and this package has not been taught would silently
// render as "Not Run", which is the safe direction but still a wrong document.
const (
	phasePending   = "Pending"
	phaseRunning   = "Running"
	phasePassed    = "Passed"
	phaseFailed    = "Failed"
	phaseError     = "Error"
	phaseSkipped   = "Skipped"
	phaseCancelled = "Cancelled"
)

// Violation causes, as pkg/verdict classifies them and the envelope carries
// them. The cause is the field that says WHO SHOULD ACT.
const (
	// causeMeasurement — the hardware fell short. This one is about the part.
	causeMeasurement = "Measurement"
	// causeEvidence — the runner's report could not support a judgement. The
	// node is unjudged, NOT condemned.
	causeEvidence = "Evidence"
	// causeAuthoring — the threshold itself is broken. No hardware is
	// implicated and nobody should be sent to a rack.
	causeAuthoring = "Authoring"
)

// NVVS categories, as the vendor's own tool groups its plugins.
const (
	catDeployment  = "Deployment"
	catIntegration = "Integration"
	catHardware    = "Hardware"
	catStress      = "Stress"
)

// NVVS status values.
const (
	statusPass = "Pass"
	statusFail = "Fail"
	statusSkip = "Skip"
	statusWarn = "Warn"
	// statusNotRun is where an Error goes.
	//
	// This is the single most important mapping in the package. `Error` means
	// the machinery failed and the hardware was never judged; `Fail` means it
	// was judged and fell short. Collapsing the two would undo the distinction
	// the entire verdict engine is built to preserve, and it would do it at the
	// schema boundary, where nobody downstream can recover it.
	//
	// NVVS has its own vocabulary for unjudged, so the distinction survives the
	// crossing intact rather than being approximated.
	statusNotRun = "Not Run"
)

// categoryFor places a TestKind in the vendor's taxonomy.
//
// Judgement calls, not a lookup of anything authoritative: the load generators
// are Stress, the passive probes and single-shot correctness checks are
// Hardware, anything measuring a path BETWEEN things is Integration. An
// unrecognised kind lands in Hardware, which is the least surprising bucket for
// "something was measured about this machine".
//
// Categories are the vendor's; the test NAMES that go inside them stay ours.
func categoryFor(kind string) string {
	switch kind {
	case "gpu-burn", "thermal-soak", "clockprobe", "power-swing", "memory-bw", "gemm-sweep":
		return catStress
	case "ib-write-bw", "gpudirect-rdma", "nccl", "fabric-soak", "tcp-baseline":
		return catIntegration
	case "compute-smoke", "host-health", "memory-stress", "disk-io", "dcgm-diag", "rvs-diag", "xpum-diag":
		return catHardware
	default:
		return catHardware
	}
}

// statusFor maps a phase to the vendor's vocabulary.
//
// Pending and Running both become "Not Run", which is accurate: at the moment
// the document was rendered, that test had produced no verdict.
// statusUnder is statusFor, with the run-level baseline rule applied.
//
// A baseline run applied NO thresholds, so no result in it met a criterion —
// there were none to meet. Rendering such a result as "Pass" is how a
// thresholdless measurement sweep becomes indistinguishable from a
// certification to anything reading the status field, which is the fail-open
// the Baseline flag exists to close.
//
// It is separate from statusFor rather than folded into it because statusFor is
// the pure phase-to-vocabulary mapping, and TestPhaseVocabularyMatchesTheAPI
// reads the API's own constants through it to check this package has been taught
// every phase. Adding a second argument there would make that guard about two
// things at once.
func statusUnder(phase string, baseline bool) string {
	// ONLY A PASS. This said `if baseline` and mapped every phase to Skip,
	// which was wrong in the one direction that matters — issue #232.
	//
	// A baseline carries no thresholds, so a Failed cannot arrive from a gate:
	// refuseGatedBaseline refuses that combination at plan time. But it can
	// still arrive from the RUNNER. gpu-burn exits 1 on ECC errors,
	// memory-stress on miscompares, and completeAttempt settles those Failed
	// with no reference to baseline anywhere. So under a baseline the ONLY route
	// to Failed is a runner's own hardware verdict — precisely the signal this
	// was erasing.
	//
	// A GPU reporting 17 ECC errors reached the vendor-schema document as
	// "Skip": the weakest claim in the vocabulary, which consumers filter as
	// not-applicable noise. That is the Error/Fail collapse this package exists
	// to prevent, one step past Not Run, at the schema boundary where nothing
	// downstream can recover it — and it made this renderer and the JUnit one
	// disagree about the same record, which is the failure pkg/report was
	// created to end.
	//
	// The reasoning that produced the bug was sound as far as it went and is
	// unchanged below: a baseline PASS must not read as a certification, because
	// no criterion was applied for it to have met. It simply says nothing about
	// a phase that was not a pass.
	if baseline && phase == phasePassed {
		return statusSkip
	}
	return statusFor(phase)
}

func statusFor(phase string) string {
	switch phase {
	case phasePassed:
		return statusPass
	case phaseFailed:
		return statusFail
	case phaseSkipped:
		return statusSkip
	case phaseError, phasePending, phaseRunning, phaseCancelled:
		return statusNotRun
	default:
		// An unrecognised phase is not assumed to be a pass. Fail-closed is the
		// house rule and it applies to rendering too.
		return statusNotRun
	}
}

// severityFor turns a violation's Cause into an NVVS severity.
//
// The cause vocabulary is richer than a severity, so ErrorCategory carries the
// cause verbatim and this is only a coarse ranking for consumers that sort by
// severity. Evidence and Authoring are deliberately NOT "error": neither
// implicates hardware, and a consumer triaging by severity should not be sent
// to a rack over a broken threshold.
func severityFor(cause string) string {
	switch cause {
	case causeMeasurement:
		return "error"
	case causeEvidence:
		return "warning"
	case causeAuthoring:
		return "info"
	default:
		return "warning"
	}
}

// hostScoped reports whether a kind measures the machine rather than its
// accelerators, which decides whether its results attach to a CPU entity or to
// the GPU entities.
func hostScoped(kind string) bool {
	switch kind {
	case "host-health", "memory-stress", "disk-io", "tcp-baseline":
		return true
	default:
		return false
	}
}
