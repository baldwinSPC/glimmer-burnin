// Package htmlreport renders a burn-in run as ONE self-contained HTML file, and
// as markdown from the same view model.
//
// # One file, no network
//
// That is the constraint everything else follows from. A report that fetches a
// stylesheet is useless in a secure facility, useless as an email attachment,
// and useless in six months when the CDN path has moved — and a burn-in report's
// whole job is to still be readable long after the run, in a room that may have
// no route to the internet at all. So: inline CSS, inline SVG, no JavaScript, no
// web fonts, no external anything. TestNoExternalRequests asserts it on the
// rendered bytes rather than on this comment.
//
// # Error and Fail are not two shades of red
//
// They mean different things and lead to different actions: a Fail is a hardware
// verdict and a reason to touch a node, an Error means the measurement never
// happened and the node is unjudged. The report distinguishes them in TEXT as
// well as colour, which is an accessibility requirement and a correctness one at
// once — a reader who cannot perceive the colour difference, or who printed the
// page in greyscale, must still be able to tell a condemned node from an
// unmeasured one.
package htmlreport

import (
	"bytes"
	_ "embed"
	"fmt"
	"html/template"
	"sort"
	"strings"

	"github.com/baldwinSPC/glimmer-burnin/pkg/contract"
	"github.com/baldwinSPC/glimmer-burnin/pkg/report"
)

//go:embed report.html.tmpl
var htmlTemplate string

// Renderer renders the single-file HTML report.
type Renderer struct{}

func (Renderer) Name() string        { return "html" }
func (Renderer) ContentType() string { return "text/html; charset=utf-8" }

func (Renderer) Render(in report.Input) ([]report.Output, error) {
	page, err := buildPage(in)
	if err != nil {
		return nil, err
	}
	tmpl, err := template.New("report").Funcs(funcs()).Parse(htmlTemplate)
	if err != nil {
		return nil, err
	}
	var buf bytes.Buffer
	if err := tmpl.Execute(&buf, page); err != nil {
		return nil, err
	}
	return []report.Output{{Filename: "burnin-report.html", Data: buf.Bytes()}}, nil
}

// page is the template's view: the shared model plus presentation-only
// derivations. Nothing here reaches back to the envelopes, so what the template
// can render is exactly what the record said.
type page struct {
	report.View
	// Statuses are the four phases in a fixed order, so a legend and a matrix
	// agree without either re-deriving the list.
	Statuses []statusInfo
	// Rows is the node × test matrix, with tests as columns.
	Columns []string
	Rows    []matrixRow
	// Details is every Node-scope result with its node, for the per-test
	// section. Link results are NOT here — they are in View.Links, once each.
	Details []detailView
	// Limitations are things this document CANNOT say, stated in the document.
	// A report that silently omits what it does not know invites the reader to
	// assume it knew.
	Limitations []string
}

type statusInfo struct {
	Phase string
	// Symbol is a text glyph, so status survives greyscale printing and a
	// reader who cannot perceive the colour difference.
	Symbol string
	Class  string
	Means  string
}

type matrixRow struct {
	Node  string
	Cells []matrixCell
}

type matrixCell struct {
	Test   string
	Phase  string
	Symbol string
	Class  string
	// Present is false when this node ran no such test. Absence is rendered as
	// absence, never as a pass.
	Present bool
}

// detailView is one result with everything a reader needs to act on it.
type detailView struct {
	Node   string
	Result contract.TestResult
	Status statusInfo
	// Causes translates each violation's Cause into who should act. It is
	// computed here rather than in the template so the HTML and markdown
	// renderers cannot word it differently.
	Causes []violationView
}

type violationView struct {
	contract.Violation
	// Advice is the plain-language statement of who should act.
	Advice string
}

// causeAdvice turns a violation's Cause into an instruction.
//
// The three causes lead to three different actions and only one of them
// involves hardware, which is the whole reason Cause exists: a Failed test can
// mix a hardware shortfall with a broken threshold, and sending somebody to a
// rack over the latter is the outcome this wording exists to prevent.
func causeAdvice(cause string) string {
	switch cause {
	case "Measurement":
		return "The hardware was measured and fell short. This is a verdict about the part."
	case "Evidence":
		return "The runner's report could not support a judgement. The node is UNJUDGED, not condemned — do not replace anything on this basis."
	case "Authoring":
		return "The threshold itself is broken. No node should be touched; fix the profile."
	case "":
		return "The cause was not recorded, so this document cannot say who should act."
	default:
		return cause
	}
}

func funcs() template.FuncMap {
	return template.FuncMap{
		"lower":    strings.ToLower,
		"statusOf": statusFor,
	}
}

func buildPage(in report.Input) (*page, error) {
	v := report.BuildView(in)
	if v.Run.UID == "" {
		return nil, fmt.Errorf("no run to render")
	}
	p := &page{View: v, Statuses: statuses()}

	// Columns are every Node-scope test name in the run, so a node that ran a
	// test another node did not still gets a column — and the gap is visible
	// rather than implied.
	seen := map[string]bool{}
	for _, n := range v.Nodes {
		for _, r := range n.Results {
			if !seen[r.Name] {
				seen[r.Name] = true
				p.Columns = append(p.Columns, r.Name)
			}
		}
	}
	sort.Strings(p.Columns)

	for _, n := range v.Nodes {
		row := matrixRow{Node: n.Name}
		byName := map[string]contract.TestResult{}
		for _, r := range n.Results {
			byName[r.Name] = r
		}
		for _, col := range p.Columns {
			r, ok := byName[col]
			if !ok {
				row.Cells = append(row.Cells, matrixCell{Test: col, Symbol: "·", Class: "absent"})
				continue
			}
			si := statusFor(r.Phase)
			row.Cells = append(row.Cells, matrixCell{
				Test: col, Phase: r.Phase, Symbol: si.Symbol, Class: si.Class, Present: true,
			})
		}
		p.Rows = append(p.Rows, row)
	}

	for _, n := range v.Nodes {
		for _, r := range n.Results {
			p.Details = append(p.Details, detailFor(n.Name, r))
		}
	}

	p.Limitations = limitations(v)
	return p, nil
}

func detailFor(node string, r contract.TestResult) detailView {
	d := detailView{Node: node, Result: r, Status: statusFor(r.Phase)}
	for _, v := range r.Violations {
		d.Causes = append(d.Causes, violationView{Violation: v, Advice: causeAdvice(v.Cause)})
	}
	return d
}

// statuses is the legend, and the single source of the symbol and class each
// phase gets. A matrix cell and a legend entry that disagreed would be worse
// than no legend.
func statuses() []statusInfo {
	return []statusInfo{
		{Phase: "Passed", Symbol: "PASS", Class: "pass",
			Means: "the measurement was taken and satisfied every threshold applied to it"},
		{Phase: "Failed", Symbol: "FAIL", Class: "fail",
			Means: "the hardware was measured and fell short — this is a verdict about the part"},
		{Phase: "Error", Symbol: "ERR", Class: "error",
			Means: "the measurement did not happen, so the hardware is UNJUDGED — not a verdict about the part"},
		{Phase: "Skipped", Symbol: "SKIP", Class: "skip",
			Means: "the runner looked and reported the test does not apply to this hardware"},
	}
}

func statusFor(phase string) statusInfo {
	for _, s := range statuses() {
		if s.Phase == phase {
			return s
		}
	}
	// A phase this build does not recognise is shown AS unrecognised. Folding
	// it into "Passed" would let unmeasured hardware read as certified.
	return statusInfo{Phase: phase, Symbol: "?", Class: "unknown",
		Means: "this build does not recognise the phase, so it cannot say whether the hardware was judged"}
}

// limitations states, in the document, what the document cannot say.
//
// A report that silently omits what it does not know invites the reader to
// assume it knew. Each entry here is a real gap in the record, not a disclaimer.
func limitations(v report.View) []string {
	var out []string
	if !v.Run.Final {
		out = append(out, "This run had not finished when the report was generated. "+
			"The phase shown is progress, not a verdict.")
	}
	if v.Run.Baseline {
		out = append(out, "This was a BASELINE run: it applied no thresholds and gated "+
			"nothing. The measurements below are evidence, not an acceptance.")
	}
	var anyInventory bool
	for _, n := range v.Nodes {
		if n.Known {
			anyInventory = true
		}
	}
	if !anyInventory && len(v.Nodes) > 0 {
		out = append(out, "No structured hardware inventory was supplied, so the "+
			"inventory section shows only what the run itself captured. A field that "+
			"is absent was not measured — it is not a claim that the hardware lacks it.")
	}
	// The envelope carries the thresholds that were VIOLATED and those that were
	// NOT EVALUATED, but not the full set that was applied. Saying so is the
	// difference between a reader inferring "only these gates existed" and
	// knowing the document cannot tell them.
	out = append(out, "Thresholds are site-authored. This document lists the gates that "+
		"were violated and the gates that could not be evaluated; the complete set "+
		"applied lives in the profile the run named.")
	return out
}
