package htmlreport

import (
	"fmt"
	"sort"
	"strings"

	"github.com/baldwinSPC/glimmer-burnin/pkg/report"
)

// MarkdownRenderer renders the same report as markdown.
//
// It shares buildPage with the HTML renderer, and that sharing is the point
// rather than a convenience. The two formats describe the same run to the same
// people — one in a browser, one in a pull-request comment or a terminal — and
// if they derived "which node failed" or "who should act" separately they would
// eventually disagree, which is the failure this whole package exists to end.
// Every wording that carries meaning (the status symbols, the cause advice, the
// limitations) comes from the shared model, not from either template.
type MarkdownRenderer struct{}

func (MarkdownRenderer) Name() string        { return "markdown" }
func (MarkdownRenderer) ContentType() string { return "text/markdown; charset=utf-8" }

func (MarkdownRenderer) Render(in report.Input) ([]report.Output, error) {
	p, err := buildPage(in)
	if err != nil {
		return nil, err
	}
	var b strings.Builder

	fmt.Fprintf(&b, "# Burn-in report — `%s/%s`\n\n", p.Run.Namespace, p.Run.Name)
	fmt.Fprintf(&b, "**%s**", p.Run.Phase)
	if !p.Run.Final {
		b.WriteString("  `INCOMPLETE — the run had not finished`")
	}
	if p.Run.Baseline {
		b.WriteString("  `BASELINE — measurement only, no thresholds applied`")
	}
	b.WriteString("\n\n")

	fmt.Fprintf(&b, "| | |\n|---|---|\n")
	fmt.Fprintf(&b, "| Run UID | `%s` |\n", p.Run.UID)
	if p.Run.Profile != "" {
		fmt.Fprintf(&b, "| Profile | `%s` |\n", p.Run.Profile)
	}
	if p.Run.Cluster != nil {
		fmt.Fprintf(&b, "| Cluster | `%s` |\n", p.Run.Cluster.Name)
	}
	if p.Run.StartedAt != "" {
		fmt.Fprintf(&b, "| Started | %s |\n", p.Run.StartedAt)
	}
	if p.Run.FinishedAt != "" {
		fmt.Fprintf(&b, "| Finished | %s |\n", p.Run.FinishedAt)
	}
	if p.Run.CancelReason != "" {
		fmt.Fprintf(&b, "| Cancelled | %s |\n", p.Run.CancelReason)
	}
	fmt.Fprintf(&b, "| Results | %d passed, %d failed, %d errored (unjudged), %d skipped |\n\n",
		p.Run.Summary.Passed, p.Run.Summary.Failed, p.Run.Summary.Errored, p.Run.Summary.Skipped)

	if len(p.Limitations) > 0 {
		b.WriteString("> **What this document cannot tell you**\n")
		for _, l := range p.Limitations {
			fmt.Fprintf(&b, "> - %s\n", l)
		}
		b.WriteString("\n")
	}

	if len(p.Rows) > 0 {
		b.WriteString("## Results by node\n\n| Node | " + strings.Join(p.Columns, " | ") + " |\n")
		b.WriteString("|---|" + strings.Repeat("---|", len(p.Columns)) + "\n")
		for _, row := range p.Rows {
			fmt.Fprintf(&b, "| `%s` |", row.Node)
			for _, c := range row.Cells {
				fmt.Fprintf(&b, " %s |", c.Symbol)
			}
			b.WriteString("\n")
		}
		b.WriteString("\n")
		for _, s := range p.Statuses {
			fmt.Fprintf(&b, "- **%s** — %s\n", s.Symbol, s.Means)
		}
		b.WriteString("- **·** — this node did not run this test\n\n")
	}

	if len(p.Links) > 0 {
		b.WriteString("## Links\n\n")
		b.WriteString("> A link verdict is a statement about the **link or the collective**, " +
			"never about either endpoint. Attributing it to one end sends an engineer " +
			"to replace the wrong part.\n\n")
		b.WriteString("| Test | Scope | Endpoints | Status | Detail |\n|---|---|---|---|---|\n")
		for _, l := range p.Links {
			fmt.Fprintf(&b, "| `%s` | %s | `%s` | %s | %s |\n",
				l.Result.Name, l.Scope, l.Label(), statusFor(l.Result.Phase).Symbol,
				oneLine(l.Result.Message))
		}
		b.WriteString("\n")
	}

	if len(p.Details) > 0 {
		b.WriteString("## Test detail\n\n")
		for _, d := range p.Details {
			fmt.Fprintf(&b, "### `%s` on `%s` — %s\n\n", d.Result.Name, d.Node, d.Status.Symbol)
			if d.Result.Message != "" {
				fmt.Fprintf(&b, "%s\n\n", d.Result.Message)
			}
			if len(d.Causes) > 0 {
				b.WriteString("| Metric | Why it did not pass | Who should act |\n|---|---|---|\n")
				for _, c := range d.Causes {
					fmt.Fprintf(&b, "| `%s` | %s | %s |\n", c.Metric, oneLine(c.Reason), c.Advice)
				}
				b.WriteString("\n")
			}
			if len(d.Result.NotEvaluated) > 0 {
				b.WriteString("> **Thresholds NOT evaluated** — a gate that did not run is " +
					"not a gate that passed.\n")
				for _, n := range d.Result.NotEvaluated {
					fmt.Fprintf(&b, "> - `%s`: %s\n", n.Metric, oneLine(n.Reason))
				}
				b.WriteString("\n")
			}
			if len(d.Result.Unmeasurable) > 0 {
				b.WriteString("> The runner declared these **unmeasurable on this hardware**. " +
					"That is a claim about the part — not a measurement of zero.\n")
				for _, m := range sortedStrings(d.Result.Unmeasurable) {
					fmt.Fprintf(&b, "> - `%s`\n", m)
				}
				b.WriteString("\n")
			}
			if len(d.Result.Metrics) > 0 {
				b.WriteString("| Measured | Value |\n|---|---|\n")
				for _, k := range sortedKeys(d.Result.Metrics) {
					fmt.Fprintf(&b, "| `%s` | `%s` |\n", k, d.Result.Metrics[k])
				}
				b.WriteString("\n")
			}
			for _, a := range d.Result.Artifacts {
				if a.Dropped != "" {
					fmt.Fprintf(&b, "- Evidence `%s` **NOT KEPT**: %s\n", a.Name, a.Dropped)
					continue
				}
				fmt.Fprintf(&b, "- Evidence `%s` (%s, %d bytes) — configmap `%s` key `%s`\n",
					a.Name, a.MediaType, a.SizeBytes, a.ConfigMap, a.Key)
			}
			b.WriteString("\n")
		}
	}

	b.WriteString("## Hardware inventory\n\n")
	for _, n := range p.Nodes {
		fmt.Fprintf(&b, "### `%s`\n\n", n.Name)
		if !n.Known {
			b.WriteString("No structured inventory was collected for this node. An absent " +
				"field was not measured; it is not a claim that the hardware lacks it.\n\n")
			continue
		}
		if n.Info.OSImage != "" {
			fmt.Fprintf(&b, "- OS: %s\n", n.Info.OSImage)
		}
		if n.Info.Kernel != "" {
			fmt.Fprintf(&b, "- Kernel: `%s`\n", n.Info.Kernel)
		}
		if n.Info.Digest != "" {
			fmt.Fprintf(&b, "- Fingerprint: `%s`\n", n.Info.Digest)
		}
		for _, g := range n.Info.GPUs {
			fmt.Fprintf(&b, "- GPU %d: %s", g.Index, g.Model)
			// Only what was captured. An empty serial in an RMA is a claim that
			// the part has none.
			for _, kv := range [][2]string{
				{"arch", g.Arch}, {"driver", g.DriverVer}, {"serial", g.Serial}, {"pci", g.PCIBusID},
			} {
				if strings.TrimSpace(kv[1]) != "" {
					fmt.Fprintf(&b, ", %s `%s`", kv[0], kv[1])
				}
			}
			b.WriteString("\n")
		}
		for _, nic := range n.Info.NICs {
			fmt.Fprintf(&b, "- NIC `%s`: %s", nic.Name, nic.Model)
			if nic.RDMADevice != "" {
				fmt.Fprintf(&b, ", rdma `%s`", nic.RDMADevice)
			}
			if nic.LinkLayer != "" {
				fmt.Fprintf(&b, ", %s", nic.LinkLayer)
			}
			b.WriteString("\n")
		}
		b.WriteString("\n")
	}

	fmt.Fprintf(&b, "---\n\nGenerated by %s %s", p.Meta.Generator, p.Meta.Version)
	if !p.Meta.GeneratedAt.IsZero() {
		fmt.Fprintf(&b, " at %s", p.Meta.GeneratedAt.UTC().Format("2006-01-02 15:04:05 UTC"))
	}
	fmt.Fprintf(&b, ". Run UID `%s`. Thresholds are site-authored: a pass means the "+
		"measurements satisfied the gates this site wrote, not that the hardware is fit "+
		"for any particular purpose.\n", p.Run.UID)

	return []report.Output{{Filename: "burnin-report.md", Data: []byte(b.String())}}, nil
}

// oneLine flattens a message for a table cell, and escapes the pipe that would
// otherwise break the row it is in.
func oneLine(s string) string {
	s = strings.Join(strings.Fields(s), " ")
	return strings.ReplaceAll(s, "|", `\|`)
}

func sortedStrings(in []string) []string {
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
