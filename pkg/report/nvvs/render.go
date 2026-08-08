package nvvs

import (
	"encoding/json"
	"fmt"
	"sort"
	"strconv"
	"strings"
	"time"

	"github.com/baldwinSPC/glimmer-burnin/pkg/contract"
	"github.com/baldwinSPC/glimmer-burnin/pkg/report"
)

// Renderer emits one NVVS-compatible document per node.
type Renderer struct{}

// New returns the renderer.
func New() Renderer { return Renderer{} }

// Name is what a user types after -o.
func (Renderer) Name() string { return "nvvs-json" }

// ContentType is the MIME type of the documents.
func (Renderer) ContentType() string { return "application/json" }

// Render produces one document per node that appears in the run's results.
//
// One per node because NVVS is single-host: dcgmi runs on one machine, and its
// schema has no way to say "these results are from eight machines". Emitting one
// document that pretended otherwise would be a lie in the vendor's own grammar.
func (r Renderer) Render(in report.Input) ([]report.Output, error) {
	verdict, final := in.Verdict()
	if verdict == nil {
		return nil, fmt.Errorf("nvvs: no delivery to render")
	}

	nodes := nodesIn(verdict.Results)
	if len(nodes) == 0 {
		// A run whose results name no node still deserves a document — it is
		// evidence that a run happened and judged nothing.
		nodes = []string{""}
	}

	outputs := make([]report.Output, 0, len(nodes))
	for _, node := range nodes {
		doc := r.document(in, verdict, final, node)
		data, err := json.MarshalIndent(doc, "", "  ")
		if err != nil {
			return nil, fmt.Errorf("nvvs: encoding the document for %q: %w", node, err)
		}
		outputs = append(outputs, report.Output{
			Filename: filename(node),
			Data:     append(data, '\n'),
		})
	}
	return outputs, nil
}

func filename(node string) string {
	if node == "" {
		return "diag.json"
	}
	// Node names are DNS labels, so they are already filename-safe; the
	// replacement is belt and braces against a caller with looser names.
	return "diag-" + strings.NewReplacer("/", "_", string('\\'), "_", "..", "_").Replace(node) + ".json"
}

// document builds one host's report.
func (r Renderer) document(in report.Input, verdict *contract.Envelope, final bool, node string) Document {
	fp := in.Fingerprint()
	info, _ := in.NodeByName(node)

	doc := Document{
		Metadata:     metadataFor(info, fp, node, in.Artifacts),
		EntityGroups: entityGroupsFor(info),
		AuxData:      auxFor(in, verdict, final, node),
	}
	d := Diagnostic{Version: schemaTag}

	if !final {
		// A run that never reached a verdict renders, because a soak reporting
		// at hour six is useful. It must not read as a verdict, and NVVS's own
		// field for "the diagnostic did not complete" is the honest place to
		// say so.
		doc.RuntimeError = fmt.Sprintf(
			"this run had not finished when the report was generated (phase %s); it records progress, not a verdict",
			verdict.Phase)
	}
	if verdict.Baseline {
		doc.Warning = "BASELINE RUN: this profile carried no thresholds. " +
			"It measured the hardware and certified nothing."
	}

	byCategory := map[string][]Test{}
	for _, res := range verdict.Results {
		if !touches(res, node) {
			continue
		}
		cat := categoryFor(res.Kind)
		byCategory[cat] = append(byCategory[cat], r.test(res, info, node, verdict.Baseline))
	}

	for _, cat := range []string{catDeployment, catIntegration, catHardware, catStress} {
		tests := byCategory[cat]
		if len(tests) == 0 {
			continue
		}
		sort.SliceStable(tests, func(i, j int) bool { return tests[i].Name < tests[j].Name })
		d.Categories = append(d.Categories, Category{Category: cat, Tests: tests})
	}
	if d.Categories == nil {
		d.Categories = []Category{}
	}

	doc.OverallResult = overall(verdict.Results, node, verdict.Baseline)
	doc.Diagnostic = d
	return doc
}

// test renders one result.
func (r Renderer) test(res contract.TestResult, info report.NodeInfo, node string, baseline bool) Test {
	status := statusUnder(res.Phase, baseline)
	link := len(res.Nodes) > 1

	t := Test{
		Name:        testName(res, link),
		TestSummary: &Result{Status: status},
	}

	base := Result{
		Status:   status,
		Warnings: warningsFor(res),
		Info:     infoFor(res, link, node),
	}

	switch {
	case link:
		// A LINK result is about the relationship, not about either endpoint.
		// NVVS has no vocabulary for that at all, so the document says it in
		// the only place it can — in words, in every copy — and the identical
		// result appears in BOTH endpoints' documents rather than being split.
		//
		// Splitting it is the failure this guards against: attributing a
		// fabric measurement to one machine sends an engineer to replace the
		// wrong part.
		t.Results = []Result{base}

	case hostScoped(res.Kind):
		// Measures the machine, so it attaches to the host rather than to any
		// accelerator.
		id := int32(0)
		base.EntityGroup = "CPU"
		base.EntityID = &id
		t.Results = []Result{base}

	case len(info.GPUs) == 1:
		// EXACTLY one device, so the binding is exact and is made.
		id := info.GPUs[0].Index
		base.EntityGroup = "GPU"
		base.EntityID = &id
		t.Results = []Result{base}

	case len(info.GPUs) > 1:
		// More than one device, so the verdict binds to NONE of them — issue
		// #206.
		//
		// This used to broadcast the node's verdict to every accelerator, with
		// an info line saying the per-device attribution had not been measured.
		// The line is honest and it does not reach the consumer that matters:
		// NVVS consumers COUNT results[] objects, which is the normal reason to
		// parse this format at all. Prose in info[] does not participate in
		// counting, so one failing thermal-soak on an eight-GPU node was eight
		// objects with status "Fail" and every tallying tool reported eight
		// failed GPUs — an engineer replacing seven healthy parts, or a fleet
		// manager quarantining eight devices.
		//
		// It is the split-Pair-verdict defect one level down, and it is not
		// recoverable downstream: once the document has been forwarded or
		// re-serialised, eight objects are eight results and nothing carries
		// the fact that they were one.
		//
		// It is also more wrong than "over-attribution" suggests. clockprobe,
		// thermal-soak and gpu-burn run on ONE CUDA device and report for that
		// device alone, so the verdict was never about all eight — it was about
		// one the ordinal cannot identify.
		//
		// An entity-less results[] entry is a shape DCGM itself produces (it is
		// exactly what test_summary is), so binding to nothing costs no schema
		// compatibility.
		//
		// The device the runner ACTUALLY measured is already in info[]:
		// infoFor emits every metric as key=value, so a runner that reported
		// pciBusId has said which part was under load. That is the only
		// unambiguous statement available — an ordinal is not, because MIG and
		// CUDA_VISIBLE_DEVICES both remap it — and duplicating it here would
		// print the same line twice.
		base.Info = append(append([]string{}, base.Info...), nodeScopeLine(info))
		t.Results = []Result{base}

	default:
		// No inventory: report the result without claiming which device it was
		// about, rather than inventing an entity id.
		t.Results = []Result{base}
	}

	return t
}

// nodeScopeLine names every device on the node, for a human.
//
// The verdict binds to no entity, so this is where a reader learns what hardware
// was in the machine. It is prose deliberately: a machine reading entity fields
// must find none, and a person reading the document must still see the fleet.
func nodeScopeLine(info report.NodeInfo) string {
	ids := make([]string, 0, len(info.GPUs))
	for _, g := range info.GPUs {
		ids = append(ids, strconv.Itoa(int(g.Index)))
	}
	return fmt.Sprintf(
		"scope=node gpus=%s (this verdict is about the node, not about any one device; "+
			"it is emitted ONCE and is not duplicated per device)", strings.Join(ids, ","))
}

// testName keeps our name and annotates a link so the category alone cannot
// mislead. Never an NVIDIA plugin name.
func testName(res contract.TestResult, link bool) string {
	if link {
		return res.Name + " (link)"
	}
	return res.Name
}

// warningsFor turns violations into NVVS warnings.
//
// Every violation becomes one, carrying its Cause verbatim in error_category —
// information NVVS has no equivalent for, and the field that says whether a
// human should walk to a rack or fix a profile. error_id is never emitted.
func warningsFor(res contract.TestResult) []Warning {
	var out []Warning
	for _, v := range res.Violations {
		text := v.Reason
		if text == "" {
			text = fmt.Sprintf("threshold %d on %s was not satisfied", v.Index, v.Metric)
		}
		out = append(out, Warning{
			Warning:       text,
			ErrorCategory: v.Cause,
			ErrorSeverity: severityFor(v.Cause),
		})
	}
	// A failing or errored test with no structured violations still owes the
	// reader its message, or the document would show a bare "Fail".
	if len(out) == 0 && res.Message != "" {
		switch res.Phase {
		case phaseFailed, phaseError:
			out = append(out, Warning{Warning: res.Message})
		}
	}
	return out
}

// infoFor is where measurements and unevaluated gates go.
//
// NVVS treats info as free text, which is exactly right for data its schema has
// no field for: our metrics keep their canonical names, and a not-evaluated
// threshold is stated as such rather than being silently absent — because an
// absent gate reads as a gate that passed.
func infoFor(res contract.TestResult, link bool, node string) []string {
	var out []string

	if link {
		peers := make([]string, 0, len(res.Nodes))
		for _, n := range res.Nodes {
			if n != node {
				peers = append(peers, n)
			}
		}
		out = append(out,
			"link="+strings.Join(res.Nodes, "<->"),
			"peer="+strings.Join(peers, ","),
			"NOTE: this verdict is about the link, not about this endpoint")
	}
	if res.Scope != "" {
		out = append(out, "scope="+res.Scope)
	}

	keys := make([]string, 0, len(res.Metrics))
	for k := range res.Metrics {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	for _, k := range keys {
		out = append(out, k+"="+res.Metrics[k])
	}

	for _, ne := range res.NotEvaluated {
		out = append(out, "NOT EVALUATED: "+ne.Metric+" — "+ne.Reason)
	}
	for _, m := range res.Unmeasurable {
		out = append(out, "unmeasurable on this hardware: "+m)
	}
	for _, a := range res.Artifacts {
		if a.Dropped != "" {
			// The operator records a truncation rather than discarding in
			// silence. A report that omits that undoes the honesty.
			out = append(out, "evidence not stored: "+a.Name+" — "+a.Dropped)
			continue
		}
		out = append(out, "evidence: "+a.Name+" ("+a.MediaType+")")
	}
	return out
}

// overall is the host's verdict, and it is an ADMISSION GATE rather than a
// diagnosis: both a Fail and a Not Run answer "do not admit this node", and the
// distinction between them lives in each result's own status, which is where a
// reader looking for a diagnosis goes.
//
// It is computed per DOCUMENT, so a node that passed does not report Fail
// because a different node failed. It is deliberately pessimistic — neither a
// Fail nor an Error may be hidden by a passing sibling — and the precedence
// between those two follows the engine that produced the verdict. See the
// switch.
//
// The BASELINE rule is not here, and that is deliberate. It lives in exactly one
// place, statusUnder, which this function computes its flags through: a baseline
// run's results are all Skip before they reach the switch below. A second
// baseline case here was written first and removed, because it could never fire
// — this function reaches the phases only via statusUnder — so it was
// unreachable code in the shape of a guard, which is worse than none. One rule,
// one site, one test.
func overall(results []contract.TestResult, node string, baseline bool) string {
	var sawError, sawFail, sawSkip, sawPass bool
	for _, res := range results {
		if !touches(res, node) {
			continue
		}
		switch statusUnder(res.Phase, baseline) {
		case statusNotRun:
			sawError = true
		case statusFail:
			sawFail = true
		case statusSkip:
			sawSkip = true
		case statusPass:
			sawPass = true
		}
	}
	switch {
	case sawFail:
		// FAIL BEATS ERROR, and the order matters more than it looks.
		//
		// This used to rank Error first, justified as "deliberately pessimistic
		// in the same order the engine uses". The engine uses the OPPOSITE
		// order and documents it: finalize() says "Precedence: Failed beats
		// Error beats Passed. A required test that FAILED is a hardware verdict
		// and wins outright", and implements it by setting RunError and then
		// overwriting it with RunFailed.
		//
		// So a run the operator settled `Failed` rendered this node's verdict as
		// "Not Run" — which in real dcgmi means a plugin that was not selected,
		// and which consumers filter as infrastructure noise. A GPU that missed
		// a sustained-clock gate was queued for a re-run instead of an RMA,
		// because an unrelated image pull failed on the same node. "Not Run" is
		// a strictly weaker claim than the run had established, and it is the
		// Error/Fail collapse this package exists to prevent, running in the
		// direction nobody checks.
		//
		// The Error-beats-Fail precedence that DOES exist here is the PAIR rule,
		// and it answers a different question: it combines the two ends of ONE
		// measurement, where a machinery failure at either end means the link
		// was never measured. Aggregating INDEPENDENT tests on a host is
		// finalize()'s case, not that one.
		return statusFail
	case sawError:
		return statusNotRun
	case sawPass:
		return statusPass
	case sawSkip:
		return statusSkip
	default:
		return statusNotRun
	}
}

// metadataFor fills only what was genuinely captured.
func metadataFor(info report.NodeInfo, fp map[string]string, node string, artifacts []report.Artifact) *Metadata {
	m := &Metadata{}
	for _, g := range info.GPUs {
		if g.DriverVer != "" && m.DriverVersion == "" {
			m.DriverVersion = g.DriverVer
		}
		if g.Model != "" {
			m.GPUDeviceIDs = append(m.GPUDeviceIDs, g.Model)
		}
		if g.Serial != "" {
			m.GPUSerials = append(m.GPUSerials, g.Serial)
		}
	}
	// A DCGM version is only claimable from a real dcgmi document. We ship no
	// DCGM and inventing a version would be the fabrication this package exists
	// to refuse, so it stays absent unless one was passed through.
	for _, a := range artifacts {
		if a.Node == node && strings.Contains(a.MediaType, "dcgm") {
			m.DCGMVersion = ""
			break
		}
	}
	if m.DriverVersion == "" && m.GPUDeviceIDs == nil && m.GPUSerials == nil && m.DCGMVersion == "" {
		return nil
	}
	return m
}

// entityGroupsFor declares the devices results can attach to.
func entityGroupsFor(info report.NodeInfo) []EntityGroup {
	if len(info.GPUs) == 0 {
		return nil
	}
	g := EntityGroup{EntityGroup: "GPU"}
	for _, gpu := range info.GPUs {
		g.Entities = append(g.Entities, Entity{
			EntityID:  gpu.Index,
			SerialNum: gpu.Serial,
			DeviceID:  gpu.Model,
		})
	}
	return []EntityGroup{g}
}

// auxFor stamps provenance. Mandatory.
func auxFor(in report.Input, verdict *contract.Envelope, final bool, node string) *AuxData {
	a := &AuxData{
		Generator:  strings.TrimSpace(in.Meta.Generator + " " + in.Meta.Version),
		Schema:     schemaTag,
		Notice:     notice,
		RunUID:     verdict.Run.UID,
		RunName:    verdict.Run.Name,
		Profile:    verdict.Run.Profile,
		DeliveryID: verdict.DeliveryID,
		Node:       node,
		Baseline:   verdict.Baseline,
	}
	if a.Generator == "" {
		a.Generator = "glimmer-burnin"
	}
	if verdict.Cluster != nil {
		a.Cluster = verdict.Cluster.Name
	}
	if !in.Meta.GeneratedAt.IsZero() {
		a.GeneratedAt = in.Meta.GeneratedAt.UTC().Format(time.RFC3339)
	}
	if !final {
		a.Warnings = append(a.Warnings, "no terminal delivery: this document records progress rather than a verdict")
	}
	for _, res := range verdict.Results {
		if !touches(res, node) {
			continue
		}
		for _, art := range res.Artifacts {
			if art.Dropped != "" {
				a.Warnings = append(a.Warnings,
					fmt.Sprintf("evidence %q from %q was not stored: %s", art.Name, res.Name, art.Dropped))
			}
		}
	}
	return a
}

// touches reports whether a result concerns this node. A result naming no node
// is run-scoped and appears in every document.
func touches(res contract.TestResult, node string) bool {
	if node == "" || len(res.Nodes) == 0 {
		return true
	}
	for _, n := range res.Nodes {
		if n == node {
			return true
		}
	}
	return false
}

// nodesIn returns every node named by any result, in stable order.
func nodesIn(results []contract.TestResult) []string {
	seen := map[string]bool{}
	var out []string
	for _, r := range results {
		for _, n := range r.Nodes {
			if n != "" && !seen[n] {
				seen[n] = true
				out = append(out, n)
			}
		}
	}
	sort.Strings(out)
	return out
}
