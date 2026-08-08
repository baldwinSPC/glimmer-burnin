package htmlreport

import (
	"fmt"
	"strings"

	"github.com/baldwinSPC/glimmer-burnin/pkg/report"
)

// The markdown renderer, the second half of #146.
//
// # Why it lives beside the HTML one rather than in its own package
//
// Because it shares build(), and that sharing is the entire point rather than a
// convenience. The two formats describe the same run to the same people — one in
// a browser, one in a pull-request comment, a terminal, or a chat message — and
// if they derived "which node failed", or worded "who should act", separately
// they would eventually disagree. Two formats of one run disagreeing is the
// problem pkg/report was created to end, and it is not a problem a code review
// catches: the divergence appears months later, in one format only, in front of
// somebody deciding whether to accept a delivery.
//
// So every phrase that carries meaning here — the status words, the cause
// sentences, the warnings, the inventory rows — comes from the shared page, and
// this file only decides how to lay it out. If you find yourself writing a
// sentence in this file that is not already in build(), it belongs in build().
//
// # What markdown is for, that HTML is not
//
// The HTML report is one self-contained file to open or print. This one is for
// the places a burn-in verdict actually gets discussed: pasted into an issue,
// posted by CI, read over SSH. It therefore stays plain — no HTML fallbacks, no
// collapsible sections, nothing that renders as literal angle brackets in a
// terminal.
type MarkdownRenderer struct{}

// NewMarkdown returns the markdown renderer.
func NewMarkdown() MarkdownRenderer { return MarkdownRenderer{} }

func (MarkdownRenderer) Name() string        { return "markdown" }
func (MarkdownRenderer) ContentType() string { return "text/markdown; charset=utf-8" }

func (MarkdownRenderer) Render(in report.Input) ([]report.Output, error) {
	p := build(report.BuildView(in))
	var b strings.Builder

	fmt.Fprintf(&b, "# Burn-in report — `%s/%s`\n\n", p.Run.Namespace, p.Run.Name)

	// The banner. Every qualifier that changes what the phase MEANS goes on the
	// same line as the phase, because a reader who takes only the first line
	// must not take away a verdict the run did not reach.
	fmt.Fprintf(&b, "**%s**", p.Run.PhaseWord)
	if !p.Run.Final {
		b.WriteString(" · `INCOMPLETE — the run had not finished; this is progress, not a verdict`")
	}
	if p.Run.Baseline {
		b.WriteString(" · `BASELINE — no thresholds were applied; this certifies nothing`")
	}
	if p.Run.Cancelled {
		fmt.Fprintf(&b, " · `CANCELLED — %s`", oneLine(p.Run.CancelWhy))
	}
	b.WriteString("\n\n")

	rows := [][2]string{
		{"Run UID", "`" + p.Run.UID + "`"},
		{"Profile", codeOrDash(p.Run.Profile)},
		{"Cluster", codeOrDash(p.Run.Cluster)},
		{"Started", orDash(p.Run.StartedAt)},
		{"Finished", orDash(p.Run.FinishedAt)},
	}
	b.WriteString("| | |\n|---|---|\n")
	for _, r := range rows {
		fmt.Fprintf(&b, "| %s | %s |\n", r[0], r[1])
	}
	if len(p.Summary) > 0 {
		var parts []string
		for _, c := range p.Summary {
			parts = append(parts, fmt.Sprintf("%d %s", c.Count, c.Label))
		}
		fmt.Fprintf(&b, "| Results | %s |\n", strings.Join(parts, ", "))
	}
	b.WriteString("\n")

	// What the document cannot say, before what it can. A reader who stops
	// early must meet the caveats, not miss them.
	if len(p.Warnings) > 0 {
		b.WriteString("> **What this report cannot tell you**\n")
		for _, w := range p.Warnings {
			fmt.Fprintf(&b, "> - %s\n", oneLine(w))
		}
		b.WriteString("\n")
	}

	if !p.Matrix.Empty && len(p.Matrix.Rows) > 0 {
		b.WriteString("## Results by node\n\n")
		b.WriteString("| Node | " + strings.Join(escapeAll(p.Matrix.Tests), " | ") + " |\n")
		b.WriteString("|---" + strings.Repeat("|---", len(p.Matrix.Tests)) + "|\n")
		for _, row := range p.Matrix.Rows {
			fmt.Fprintf(&b, "| `%s` |", escape(row.Node))
			for _, c := range row.Cells {
				// The word, never a colour. A terminal has no colour to lose and
				// a pasted table keeps none, so the status has to be readable as
				// text or it is not readable at all.
				fmt.Fprintf(&b, " %s |", escape(c.Word))
			}
			b.WriteString("\n")
		}
		b.WriteString("\n")
	}

	if len(p.Links) > 0 {
		b.WriteString("## Links\n\n")
		b.WriteString("> A link verdict describes the **link or the collective**, never " +
			"either endpoint. Attributing it to one end sends an engineer to replace the " +
			"wrong part.\n\n")
		for _, l := range p.Links {
			fmt.Fprintf(&b, "### `%s` — %s\n\n", escape(l.Name), l.Status.Word)
			fmt.Fprintf(&b, "%s across `%s`\n\n", l.Scope, strings.Join(escapeAll(l.Endpoints), "`, `"))
			writeDetail(&b, l.Metrics, l.Violations, l.Notes)
		}
	}

	if len(p.Nodes) > 0 {
		b.WriteString("## Test detail\n\n")
		for _, n := range p.Nodes {
			fmt.Fprintf(&b, "### `%s`\n\n", escape(n.Name))
			if len(n.Tests) == 0 {
				b.WriteString("This node produced no results.\n\n")
				continue
			}
			for _, tst := range n.Tests {
				fmt.Fprintf(&b, "#### `%s` — %s\n\n", escape(tst.Name), tst.Status.Word)
				if tst.Message != "" {
					fmt.Fprintf(&b, "%s\n\n", oneLine(tst.Message))
				}
				writeDetail(&b, tst.Metrics, tst.Violations, tst.Notes)
			}
		}
	}

	if len(p.Inventory) > 0 {
		b.WriteString("## Hardware inventory\n\n")
		for _, row := range p.Inventory {
			fmt.Fprintf(&b, "### `%s`\n\n", escape(row.Node))
			if row.Unknown || len(row.Fields) == 0 {
				b.WriteString("No inventory was captured for this node. An absent field " +
					"was not measured; it is not a claim that the hardware lacks it.\n\n")
				continue
			}
			for _, f := range row.Fields {
				fmt.Fprintf(&b, "- %s: `%s`\n", escape(f.Key), escape(f.Value))
			}
			b.WriteString("\n")
		}
	}

	b.WriteString("---\n\n")
	fmt.Fprintf(&b, "Generated by %s", p.Meta.Generator)
	if p.Meta.GeneratedAt != "" {
		fmt.Fprintf(&b, " at %s", p.Meta.GeneratedAt)
	}
	fmt.Fprintf(&b, ". %s\n", p.Meta.Notice)

	return []report.Output{{Filename: "burnin-report.md", Data: []byte(b.String())}}, nil
}

// writeDetail lays out the parts every test and every link share.
//
// One function, so a link's detail and a node test's detail can never drift into
// showing different things about the same kind of finding.
func writeDetail(b *strings.Builder, metrics []kv, violations []violation, notes []string) {
	for _, v := range violations {
		// The cause sentence comes from build(); this file must not reword it.
		// Three causes lead to three different actions and only one involves
		// hardware, which is the whole reason Cause exists.
		fmt.Fprintf(b, "- **%s** — %s\n  _%s_\n", escape(v.Metric), oneLine(v.Reason), oneLine(v.CauseWhy))
	}
	if len(violations) > 0 {
		b.WriteString("\n")
	}
	for _, n := range notes {
		fmt.Fprintf(b, "> %s\n", oneLine(n))
	}
	if len(notes) > 0 {
		b.WriteString("\n")
	}
	if len(metrics) > 0 {
		b.WriteString("| Measured | Value |\n|---|---|\n")
		for _, m := range metrics {
			fmt.Fprintf(b, "| `%s` | `%s` |\n", escape(m.Key), escape(m.Value))
		}
		b.WriteString("\n")
	}
}

// escape neutralises the one character that can break a markdown table.
//
// A runner's stdout reaches this document — a test name, a metric name, a
// threshold reason — and a pipe in any of them silently splits the row it is in,
// so a reader sees a shifted table rather than the values that were measured.
// Everything else markdown treats specially degrades to odd emphasis at worst;
// this one loses data.
//
// BACKTICKS DO NOT PROTECT IT. A table row is split into cells before code spans
// are parsed, so `a|b` in a cell is still two cells. That is why escape() is
// applied to values this file already wraps in backticks, which looks redundant
// and is not — it was caught by a test, having been written the redundant-looking
// way first and been wrong.
func escape(s string) string { return strings.ReplaceAll(s, "|", `\|`) }

func escapeAll(in []string) []string {
	out := make([]string, 0, len(in))
	for _, s := range in {
		out = append(out, escape(s))
	}
	return out
}

// oneLine flattens a message and escapes it. A runner's message is arbitrary
// text and a newline in the middle of a table row ends the table.
func oneLine(s string) string { return escape(strings.Join(strings.Fields(s), " ")) }

func orDash(s string) string {
	if strings.TrimSpace(s) == "" {
		// An em dash, not an empty cell: "not recorded" and "recorded as
		// nothing" must not look the same.
		return "—"
	}
	return escape(s)
}

func codeOrDash(s string) string {
	if strings.TrimSpace(s) == "" {
		return "—"
	}
	return "`" + escape(s) + "`"
}
