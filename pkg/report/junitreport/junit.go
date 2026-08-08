// Package junitreport renders a burn-in run as JUnit XML.
//
// JUnit is how a burn-in result enters a pipeline: a node-acceptance run gating
// a provisioning workflow, or a nightly re-verification that should turn a
// dashboard red, both want the format every CI system already parses.
//
// # The one rule that matters
//
//	Error   -> <error>
//	Fail    -> <failure>
//	Skipped -> <skipped>
//
// Collapsing Error into failure would erase the distinction this project's
// entire verdict engine exists to preserve. An unpullable image and a degraded
// GPU would look identical on a dashboard, and only one of them is a reason for
// anybody to touch hardware. JUnit has had a separate <error> element since
// Ant, precisely for "the test could not run", so nothing is being invented
// here — the mapping is the honest one and the lossy one is a choice.
//
// # A link is one test case
//
// A Pair or Group result appears ONCE, in a dedicated `links` suite naming every
// endpoint — never split across the endpoints' own suites. A point-to-point
// measurement is a property of the link, and attributing it to one end sends an
// engineer to replace the wrong part. The view model makes this structural: link
// results are not reachable through the per-node suites at all.
package junitreport

import (
	"encoding/xml"
	"fmt"
	"sort"
	"strings"

	"github.com/baldwinSPC/glimmer-burnin/pkg/contract"
	"github.com/baldwinSPC/glimmer-burnin/pkg/report"
)

// Renderer renders JUnit XML.
type Renderer struct{}

func (Renderer) Name() string        { return "junit" }
func (Renderer) ContentType() string { return "application/xml" }

// linksSuite is the suite name Pair and Group results land in.
const linksSuite = "links"

type testsuites struct {
	XMLName  xml.Name    `xml:"testsuites"`
	Name     string      `xml:"name,attr"`
	Tests    int         `xml:"tests,attr"`
	Failures int         `xml:"failures,attr"`
	Errors   int         `xml:"errors,attr"`
	Skipped  int         `xml:"skipped,attr"`
	Suites   []testsuite `xml:"testsuite"`
}

type testsuite struct {
	Name     string     `xml:"name,attr"`
	Tests    int        `xml:"tests,attr"`
	Failures int        `xml:"failures,attr"`
	Errors   int        `xml:"errors,attr"`
	Skipped  int        `xml:"skipped,attr"`
	Props    *props     `xml:"properties,omitempty"`
	Cases    []testcase `xml:"testcase"`
}

type props struct {
	Props []prop `xml:"property"`
}

type prop struct {
	Name  string `xml:"name,attr"`
	Value string `xml:"value,attr"`
}

type testcase struct {
	Name      string `xml:"name,attr"`
	Classname string `xml:"classname,attr"`
	// Time is omitted when the record has no timestamps. A duration of 0.000
	// would read as a test that ran instantly, which is a different claim from
	// a test whose duration nobody recorded.
	Time      string   `xml:"time,attr,omitempty"`
	Props     *props   `xml:"properties,omitempty"`
	Failure   *detail  `xml:"failure,omitempty"`
	Error     *detail  `xml:"error,omitempty"`
	Skipped   *skipped `xml:"skipped,omitempty"`
	SystemOut string   `xml:"system-out,omitempty"`
}

type detail struct {
	Message string `xml:"message,attr"`
	Type    string `xml:"type,attr,omitempty"`
	Body    string `xml:",chardata"`
}

type skipped struct {
	Message string `xml:"message,attr,omitempty"`
}

// Render produces one JUnit document for the whole run.
func (r Renderer) Render(in report.Input) ([]report.Output, error) {
	v := report.BuildView(in)
	if v.Run.UID == "" {
		return nil, fmt.Errorf("no run to render")
	}

	doc := testsuites{Name: suiteRunName(v)}
	for _, n := range v.Nodes {
		if len(n.Results) == 0 {
			continue
		}
		doc.Suites = append(doc.Suites, nodeSuite(n))
	}
	if len(v.Links) > 0 {
		doc.Suites = append(doc.Suites, linkSuite(v))
	}
	for _, s := range doc.Suites {
		doc.Tests += s.Tests
		doc.Failures += s.Failures
		doc.Errors += s.Errors
		doc.Skipped += s.Skipped
	}

	body, err := xml.MarshalIndent(doc, "", "  ")
	if err != nil {
		return nil, err
	}
	return []report.Output{{
		Filename: "junit.xml",
		Data:     append([]byte(xml.Header), append(body, '\n')...),
	}}, nil
}

// suiteRunName identifies the run. A baseline sweep says so IN THE NAME,
// because a CI dashboard shows the name and nothing else, and a green
// thresholdless sweep read as a certification is the failure #142 describes.
func suiteRunName(v report.View) string {
	name := v.Run.Namespace + "/" + v.Run.Name
	if v.Run.Baseline {
		name += " [baseline: measurement only, no thresholds applied]"
	}
	if !v.Run.Final {
		name += " [INCOMPLETE: run had not finished]"
	}
	return name
}

func nodeSuite(n report.NodeView) testsuite {
	s := testsuite{Name: n.Name, Props: nodeProps(n)}
	for _, res := range n.Results {
		s.Cases = append(s.Cases, caseFor(res, n.Name, res.Name))
	}
	tally(&s)
	return s
}

// linkSuite holds every Pair and Group result, each exactly once.
func linkSuite(v report.View) testsuite {
	s := testsuite{Name: linksSuite}
	for _, l := range v.Links {
		// The case name carries every endpoint, so a CI dashboard row is
		// self-describing: a reader who sees it red must be able to tell it is
		// about a link without opening anything.
		s.Cases = append(s.Cases, caseFor(l.Result, linksSuite,
			fmt.Sprintf("%s (%s: %s)", l.Result.Name, strings.ToLower(l.Scope), l.Label())))
	}
	tally(&s)
	return s
}

func tally(s *testsuite) {
	s.Tests = len(s.Cases)
	for _, c := range s.Cases {
		switch {
		case c.Error != nil:
			s.Errors++
		case c.Failure != nil:
			s.Failures++
		case c.Skipped != nil:
			s.Skipped++
		}
	}
}

func caseFor(res contract.TestResult, classname, name string) testcase {
	c := testcase{
		Name:      name,
		Classname: classname,
		Time:      durationSeconds(res),
		Props:     metricProps(res),
		SystemOut: evidence(res),
	}

	switch res.Phase {
	case "Error":
		// NEVER <failure>. An Error means the machinery failed and the hardware
		// was never judged; a dashboard that shows it as a failure sends
		// somebody to a node over an image pull.
		c.Error = &detail{
			Message: firstLine(res.Message),
			Type:    "Error",
			Body:    errorBody(res),
		}
	case "Failed":
		c.Failure = &detail{
			Message: firstLine(res.Message),
			Type:    "Failed",
			Body:    failureBody(res),
		}
	case "Skipped":
		c.Skipped = &skipped{Message: firstLine(res.Message)}
	case "Passed":
		// Nothing. A passing case carries only its metrics.
	default:
		// A phase this build does not recognise — an older or newer operator —
		// is an <error>, never a silent pass. A result nobody can classify has
		// not certified anything, and reporting it green is the one outcome
		// that lets unmeasured hardware into a fleet.
		c.Error = &detail{
			Message: fmt.Sprintf("unrecognised phase %q", res.Phase),
			Type:    "Unknown",
			Body: fmt.Sprintf("This build does not recognise phase %q, so it cannot say "+
				"whether the hardware was judged. Reported as an error rather than a "+
				"pass: an unrecognised verdict has certified nothing.\n%s",
				res.Phase, res.Message),
		}
	}
	return c
}

// failureBody spells out every gate the test missed, with its cause.
//
// Message names only the first violation and that field is frozen. A run that
// failed three gates for three different reasons would otherwise reach a
// dashboard as one sentence about one of them.
func failureBody(res contract.TestResult) string {
	var b strings.Builder
	if res.Message != "" {
		b.WriteString(res.Message + "\n")
	}
	for _, v := range res.Violations {
		fmt.Fprintf(&b, "\n[%s] threshold %d on %s: %s", causeAdvice(v.Cause), v.Index, v.Metric, v.Reason)
		if v.Kind != "" {
			fmt.Fprintf(&b, " (%s)", v.Kind)
		}
	}
	if s := notEvaluated(res); s != "" {
		b.WriteString("\n" + s)
	}
	return strings.TrimSpace(b.String())
}

func errorBody(res contract.TestResult) string {
	var b strings.Builder
	b.WriteString(res.Message)
	b.WriteString("\n\nThe hardware was NOT judged. An Error means the measurement did not " +
		"happen — an image that would not pull, a pod that would not schedule, a runner " +
		"that could not look. It is reported separately from a failure because only a " +
		"failure is a reason to touch a node.")
	if s := notEvaluated(res); s != "" {
		b.WriteString("\n\n" + s)
	}
	return strings.TrimSpace(b.String())
}

// causeAdvice turns the violation's Cause into who should act, because a CI log
// is read by whoever the build woke up and they should not have to look it up.
func causeAdvice(cause string) string {
	switch cause {
	case "Measurement":
		return "hardware fell short"
	case "Evidence":
		return "unjudged: the runner's report could not support a judgement"
	case "Authoring":
		return "the threshold itself is broken; no node should be touched"
	case "":
		return "cause not recorded"
	default:
		return cause
	}
}

// notEvaluated reports gates that never ran. A gate that did not run is not a
// gate that passed, and a consumer reading only the phase cannot tell.
func notEvaluated(res contract.TestResult) string {
	if len(res.NotEvaluated) == 0 {
		return ""
	}
	var b strings.Builder
	b.WriteString("Thresholds NOT EVALUATED (neither satisfied nor violated):")
	for _, n := range res.NotEvaluated {
		fmt.Fprintf(&b, "\n  %s: %s", n.Metric, n.Reason)
	}
	return b.String()
}

// evidence is the system-out block: what was measured, what could not be, and
// what raw evidence exists.
func evidence(res contract.TestResult) string {
	var lines []string
	if len(res.Unmeasurable) > 0 {
		// A claim about the PART, not about the run: "this part has no ECC" is
		// a different statement from "this part reported zero ECC errors".
		lines = append(lines, "unmeasurable on this hardware (runner declared): "+
			strings.Join(sorted(res.Unmeasurable), ", "))
	}
	for _, a := range res.Artifacts {
		if a.Dropped != "" {
			lines = append(lines, fmt.Sprintf("artifact %s NOT KEPT: %s", a.Name, a.Dropped))
			continue
		}
		lines = append(lines, fmt.Sprintf("artifact %s (%s, %d bytes) in configmap %s key %s",
			a.Name, a.MediaType, a.SizeBytes, a.ConfigMap, a.Key))
	}
	return strings.Join(lines, "\n")
}

// metricProps carries the measurements. Absent when none were taken — an empty
// <properties/> would suggest a test that measured nothing on purpose.
func metricProps(res contract.TestResult) *props {
	if len(res.Metrics) == 0 {
		return nil
	}
	p := &props{}
	for _, k := range sortedKeys(res.Metrics) {
		p.Props = append(p.Props, prop{Name: k, Value: res.Metrics[k]})
	}
	return p
}

// nodeProps carries the node's captured identity, and only what was captured.
func nodeProps(n report.NodeView) *props {
	if !n.Known {
		return nil
	}
	p := &props{}
	add := func(k, v string) {
		if strings.TrimSpace(v) != "" {
			p.Props = append(p.Props, prop{Name: k, Value: v})
		}
	}
	add("node.osImage", n.Info.OSImage)
	add("node.kernel", n.Info.Kernel)
	add("node.fingerprintDigest", n.Info.Digest)
	for _, g := range n.Info.GPUs {
		add(fmt.Sprintf("gpu.%d.model", g.Index), g.Model)
		add(fmt.Sprintf("gpu.%d.arch", g.Index), g.Arch)
		add(fmt.Sprintf("gpu.%d.driverVersion", g.Index), g.DriverVer)
		// Omitted when uncaptured. An empty serial in an RMA is a claim that
		// the part has none.
		add(fmt.Sprintf("gpu.%d.serial", g.Index), g.Serial)
		add(fmt.Sprintf("gpu.%d.pciBusId", g.Index), g.PCIBusID)
	}
	for _, nic := range n.Info.NICs {
		add("nic."+nic.Name+".model", nic.Model)
		add("nic."+nic.Name+".rdmaDevice", nic.RDMADevice)
		add("nic."+nic.Name+".linkLayer", nic.LinkLayer)
	}
	if len(p.Props) == 0 {
		return nil
	}
	return p
}

// durationSeconds is how long the execution took, or empty when the record does
// not say. It is never zero-filled: 0.000 reads as a test that ran instantly.
func durationSeconds(res contract.TestResult) string {
	if res.StartedAt == nil || res.FinishedAt == nil {
		return ""
	}
	d := res.FinishedAt.Sub(*res.StartedAt).Seconds()
	if d < 0 {
		return ""
	}
	return fmt.Sprintf("%.3f", d)
}

func firstLine(s string) string {
	if i := strings.IndexByte(s, '\n'); i >= 0 {
		return strings.TrimSpace(s[:i])
	}
	return strings.TrimSpace(s)
}

func sorted(in []string) []string {
	out := append([]string(nil), in...)
	sort.Strings(out)
	return out
}

func sortedKeys(m map[string]string) []string {
	out := make([]string, 0, len(m))
	for k := range m {
		out = append(out, k)
	}
	sort.Strings(out)
	return out
}
