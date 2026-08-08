// Package htmlreport renders a run as one self-contained HTML file.
//
// # One file, no network
//
// The constraint that shapes everything here: the output must open with no
// network access, forever. A report that fetches a stylesheet is useless in a
// secure facility, useless as an email attachment, and useless in eighteen
// months when the CDN path has moved and the document that was meant to be
// evidence renders as unstyled text.
//
// So the CSS is inline, there is no JavaScript, there are no fonts to fetch and
// no images to miss. TestTheDocumentMakesNoExternalRequests asserts it rather
// than trusting it.
//
// # What the document is for
//
// Someone deciding whether to accept a delivery, or assembling evidence for a
// warranty claim. That reader needs three things the machine-readable formats do
// not have to provide: the verdict at a glance, the reason behind a failure in
// words, and enough provenance that the document still means something after it
// has been forwarded twice.
//
// Two presentation rules follow from the project's own invariants and are not
// stylistic:
//
//   - Error and Fail are given different hues and different words, never two
//     shades of red. They mean different things — one says the hardware fell
//     short, the other says it was never judged — and a reader who cannot tell
//     them apart cannot tell whether to send an engineer.
//   - Status is conveyed in text as well as colour, so the document survives
//     greyscale printing and colour-blind readers. An RMA attachment gets
//     printed.
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

//go:embed report.gohtml
var tmplSource string

var tmpl = template.Must(template.New("report").Parse(tmplSource))

// Renderer emits one HTML document for the whole run.
//
// One document, unlike the NVVS renderer's one-per-node: a human reading this
// is deciding about a delivery, and splitting the run across eight files would
// hide exactly the cross-node comparison they are here to make.
type Renderer struct{}

// New returns the renderer.
func New() Renderer { return Renderer{} }

// Name is what a user types after -o.
func (Renderer) Name() string { return "html" }

// ContentType is the MIME type of the document.
func (Renderer) ContentType() string { return "text/html; charset=utf-8" }

// Render produces the document.
func (Renderer) Render(in report.Input) ([]report.Output, error) {
	p := build(report.BuildView(in))

	var buf bytes.Buffer
	if err := tmpl.Execute(&buf, p); err != nil {
		return nil, fmt.Errorf("htmlreport: rendering: %w", err)
	}
	return []report.Output{{Filename: "burn-in-report.html", Data: buf.Bytes()}}, nil
}

// page is the fully-prepared model the template renders.
//
// Everything the template needs is computed here so the template itself stays
// dumb: logic in a template is logic that cannot be unit-tested, and this
// document is evidence.
type page struct {
	Run       runBlock
	Summary   []counter
	Matrix    matrix
	Links     []linkBlock
	Nodes     []nodeBlock
	Inventory []invRow
	Warnings  []string
	Meta      metaBlock
}

type runBlock struct {
	Name       string
	Namespace  string
	UID        string
	Profile    string
	Cluster    string
	Phase      string
	PhaseClass string
	PhaseWord  string
	Final      bool
	Baseline   bool
	Cancelled  bool
	CancelWhy  string
	StartedAt  string
	FinishedAt string
}

type counter struct {
	Label string
	Count int32
	Class string
}

// matrix is the node × test grid. Tests are columns so a reader scanning down a
// column sees one test across the fleet, which is the comparison that finds a
// single bad node.
type matrix struct {
	Tests []string
	Rows  []matrixRow
	Empty bool
}

type matrixRow struct {
	Node  string
	Cells []cell
}

type cell struct {
	Word  string
	Class string
	Title string
}

type linkBlock struct {
	Name       string
	Scope      string
	Label      string
	Endpoints  []string
	Status     status
	Metrics    []kv
	Violations []violation
	Notes      []string
}

type nodeBlock struct {
	Name  string
	Tests []testBlock
}

type testBlock struct {
	Name       string
	Kind       string
	Status     status
	Message    string
	Metrics    []kv
	Violations []violation
	Notes      []string
}

type status struct {
	Word  string
	Class string
}

type kv struct{ Key, Value string }

type violation struct {
	Metric   string
	Reason   string
	Cause    string
	CauseWhy string
	Class    string
}

type invRow struct {
	Node    string
	Fields  []kv
	Unknown bool
}

type metaBlock struct {
	Generator   string
	GeneratedAt string
	Notice      string
}

// build turns the shared view into the prepared page.
func build(v report.View) page {
	p := page{
		Run: runBlock{
			Name:       v.Run.Name,
			Namespace:  v.Run.Namespace,
			UID:        v.Run.UID,
			Profile:    v.Run.Profile,
			Phase:      v.Run.Phase,
			PhaseClass: classFor(v.Run.Phase),
			PhaseWord:  wordFor(v.Run.Phase),
			Final:      v.Run.Final,
			Baseline:   v.Run.Baseline,
			Cancelled:  v.Run.Phase == "Cancelled",
			CancelWhy:  v.Run.CancelReason,
			StartedAt:  v.Run.StartedAt,
			FinishedAt: v.Run.FinishedAt,
		},
		Meta: metaBlock{
			Generator:   strings.TrimSpace(v.Meta.Generator + " " + v.Meta.Version),
			GeneratedAt: formatTime(v),
			Notice:      "Thresholds in this report are authored by the site that ran it. A pass is a statement about the gates that were applied, not a warranty.",
		},
	}
	if v.Run.Cluster != nil {
		p.Run.Cluster = v.Run.Cluster.Name
	}
	if p.Meta.Generator == "" {
		p.Meta.Generator = "glimmer-burnin"
	}

	s := v.Run.Summary
	p.Summary = []counter{
		{Label: "Passed", Count: s.Passed, Class: "pass"},
		{Label: "Failed", Count: s.Failed, Class: "fail"},
		{Label: "Errored", Count: s.Errored, Class: "error"},
		{Label: "Skipped", Count: s.Skipped, Class: "skip"},
	}

	p.Matrix = buildMatrix(v)
	p.Links = buildLinks(v)
	p.Nodes = buildNodes(v)
	p.Inventory = buildInventory(v)
	p.Warnings = buildWarnings(v)
	return p
}

func formatTime(v report.View) string {
	if !v.Meta.GeneratedAt.IsZero() {
		return v.Meta.GeneratedAt.UTC().Format("2006-01-02 15:04:05 UTC")
	}
	return ""
}

// buildMatrix is the at-a-glance grid.
func buildMatrix(v report.View) matrix {
	names := map[string]bool{}
	for _, n := range v.Nodes {
		for _, r := range n.Results {
			names[r.Name] = true
		}
	}
	tests := make([]string, 0, len(names))
	for n := range names {
		tests = append(tests, n)
	}
	sort.Strings(tests)

	m := matrix{Tests: tests, Empty: len(tests) == 0 || len(v.Nodes) == 0}
	for _, n := range v.Nodes {
		row := matrixRow{Node: n.Name}
		byName := map[string]contract.TestResult{}
		for _, r := range n.Results {
			byName[r.Name] = r
		}
		for _, t := range tests {
			r, ok := byName[t]
			if !ok {
				// Not run on this node. Blank rather than a status, because an
				// absent test is not a passing one.
				row.Cells = append(row.Cells, cell{Word: "—", Class: "absent", Title: t + ": not run on this node"})
				continue
			}
			row.Cells = append(row.Cells, cell{
				Word:  wordFor(r.Phase),
				Class: classFor(r.Phase),
				Title: t + ": " + r.Phase,
			})
		}
		m.Rows = append(m.Rows, row)
	}
	return m
}

func buildLinks(v report.View) []linkBlock {
	out := make([]linkBlock, 0, len(v.Links))
	for _, l := range v.Links {
		out = append(out, linkBlock{
			Name:       l.Result.Name,
			Scope:      l.Scope,
			Label:      l.Label(),
			Endpoints:  l.Endpoints,
			Status:     status{Word: wordFor(l.Result.Phase), Class: classFor(l.Result.Phase)},
			Metrics:    metricsOf(l.Result),
			Violations: violationsOf(l.Result),
			Notes:      notesOf(l.Result),
		})
	}
	return out
}

func buildNodes(v report.View) []nodeBlock {
	out := make([]nodeBlock, 0, len(v.Nodes))
	for _, n := range v.Nodes {
		b := nodeBlock{Name: n.Name}
		for _, r := range n.Results {
			// A test is usually named after its kind, and printing both then
			// says the same word twice. Show the kind only when it adds
			// something — which is exactly when a profile named a test
			// something other than its kind.
			kind := r.Kind
			if kind == r.Name {
				kind = ""
			}
			b.Tests = append(b.Tests, testBlock{
				Name:       r.Name,
				Kind:       kind,
				Status:     status{Word: wordFor(r.Phase), Class: classFor(r.Phase)},
				Message:    r.Message,
				Metrics:    metricsOf(r),
				Violations: violationsOf(r),
				Notes:      notesOf(r),
			})
		}
		out = append(out, b)
	}
	return out
}

// buildInventory shows what the run knew about the hardware.
//
// Known distinguishes "no inventory was collected" from "the hardware reported
// nothing", which are different facts and only one of them is about the machine.
func buildInventory(v report.View) []invRow {
	out := make([]invRow, 0, len(v.Nodes))
	for _, n := range v.Nodes {
		row := invRow{Node: n.Name, Unknown: !n.Known}
		if n.Known {
			i := n.Info
			row.Fields = appendIf(row.Fields, "OS", i.OSImage)
			row.Fields = appendIf(row.Fields, "Kernel", i.Kernel)
			for _, g := range i.GPUs {
				label := fmt.Sprintf("GPU %d", g.Index)
				parts := nonEmpty(g.Model, g.Arch, g.DriverVer)
				if g.Serial != "" {
					parts = append(parts, "serial "+g.Serial)
				}
				row.Fields = appendIf(row.Fields, label, strings.Join(parts, " · "))
			}
			for _, nic := range i.NICs {
				parts := nonEmpty(nic.Model, nic.LinkLayer, nic.RDMADevice)
				if nic.SpeedMbps > 0 {
					parts = append(parts, fmt.Sprintf("%d Mb/s", nic.SpeedMbps))
				}
				row.Fields = appendIf(row.Fields, "NIC "+nic.Name, strings.Join(parts, " · "))
			}
		}
		// Whatever the run itself captured, even with no inventory.
		if raw, ok := report.FingerprintValue(v.Run.Fingerprint, n.Name); ok {
			row.Fields = append(row.Fields, kv{Key: "Captured", Value: raw})
		}
		out = append(out, row)
	}
	return out
}

// buildWarnings surfaces what the report could not establish.
//
// A document that quietly omits evidence is worse than one that says the
// evidence was too large: the first is indistinguishable from a clean run.
func buildWarnings(v report.View) []string {
	var out []string
	if !v.Run.Final {
		out = append(out, "This run had not finished when the report was generated. It records progress, not a verdict.")
	}
	if v.Run.Baseline {
		out = append(out, "This was a BASELINE run: the profile carried no thresholds, so nothing was certified. The numbers are measurements to set gates from.")
	}
	seen := map[string]bool{}
	add := func(s string) {
		if !seen[s] {
			seen[s] = true
			out = append(out, s)
		}
	}
	for _, n := range v.Nodes {
		for _, r := range n.Results {
			for _, a := range r.Artifacts {
				if a.Dropped != "" {
					add(fmt.Sprintf("Evidence %q from %s was not stored: %s", a.Name, r.Name, a.Dropped))
				}
			}
		}
	}
	for _, l := range v.Links {
		for _, a := range l.Result.Artifacts {
			if a.Dropped != "" {
				add(fmt.Sprintf("Evidence %q from %s was not stored: %s", a.Name, l.Result.Name, a.Dropped))
			}
		}
	}
	return out
}

func metricsOf(r contract.TestResult) []kv {
	keys := make([]string, 0, len(r.Metrics))
	for k := range r.Metrics {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	out := make([]kv, 0, len(keys))
	for _, k := range keys {
		out = append(out, kv{Key: k, Value: r.Metrics[k]})
	}
	return out
}

// violationsOf renders every failed gate with its cause in plain language.
//
// The cause is the most decision-relevant thing in the document — it is the
// difference between sending someone to a rack and fixing a profile — so it is
// spelled out rather than left as a one-word enum a reader has to look up.
func violationsOf(r contract.TestResult) []violation {
	out := make([]violation, 0, len(r.Violations))
	for _, v := range r.Violations {
		out = append(out, violation{
			Metric:   v.Metric,
			Reason:   v.Reason,
			Cause:    v.Cause,
			CauseWhy: causeSentence(v.Cause),
			Class:    causeClass(v.Cause),
		})
	}
	return out
}

func causeSentence(cause string) string {
	switch cause {
	case "Measurement":
		return "The hardware fell short of the gate. This one is about the part."
	case "Evidence":
		return "The runner's report could not support a judgement, so this node is unjudged — not condemned."
	case "Authoring":
		return "The threshold itself is broken. No hardware is implicated; the profile needs fixing."
	default:
		return ""
	}
}

func causeClass(cause string) string {
	switch cause {
	case "Measurement":
		return "fail"
	case "Evidence":
		return "error"
	case "Authoring":
		return "authoring"
	default:
		return "skip"
	}
}

// notesOf carries the states that are neither a metric nor a violation, and
// that a reader would otherwise never learn about.
func notesOf(r contract.TestResult) []string {
	var out []string
	for _, ne := range r.NotEvaluated {
		// An unevaluated gate is invisible unless it is said out loud, and an
		// invisible gate reads as a gate that passed.
		out = append(out, "Not evaluated: "+ne.Metric+" — "+ne.Reason)
	}
	for _, m := range r.Unmeasurable {
		out = append(out, "Unmeasurable on this hardware: "+m)
	}
	for _, a := range r.Artifacts {
		if a.Dropped == "" {
			out = append(out, "Evidence attached: "+a.Name)
		}
	}
	return out
}

// wordFor is the status as a reader sees it. Text, so the document survives
// greyscale printing and colour-blindness.
func wordFor(phase string) string {
	switch phase {
	case "Passed":
		return "PASS"
	case "Failed":
		return "FAIL"
	case "Error":
		return "ERROR"
	case "Skipped":
		return "SKIP"
	case "Cancelled":
		return "CANCELLED"
	case "Running":
		return "RUNNING"
	case "Pending":
		return "PENDING"
	default:
		return strings.ToUpper(phase)
	}
}

// classFor picks the visual treatment. Error and Fail get DIFFERENT hues, not
// two shades of one: they lead to different actions.
func classFor(phase string) string {
	switch phase {
	case "Passed":
		return "pass"
	case "Failed":
		return "fail"
	case "Error":
		return "error"
	case "Skipped":
		return "skip"
	case "Cancelled":
		return "cancelled"
	default:
		return "running"
	}
}

func appendIf(dst []kv, key, value string) []kv {
	if strings.TrimSpace(value) == "" {
		return dst
	}
	return append(dst, kv{Key: key, Value: value})
}

func nonEmpty(vals ...string) []string {
	out := make([]string, 0, len(vals))
	for _, v := range vals {
		if strings.TrimSpace(v) != "" {
			out = append(out, v)
		}
	}
	return out
}
