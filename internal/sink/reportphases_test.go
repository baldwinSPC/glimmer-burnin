package sink

import (
	"go/ast"
	"go/parser"
	"go/token"
	"testing"

	burninv1alpha1 "github.com/baldwinSPC/glimmer-burnin/api/v1alpha1"
	"github.com/baldwinSPC/glimmer-burnin/pkg/report"
)

// pkg/report pins the terminal phase strings, and this is what stops them
// drifting — issue #144.
//
// pkg/report deliberately does not import api/v1alpha1: its audience is CI jobs
// and third-party ingest, which must not pull in apimachinery to format a
// table. So it spells the phases out, which is honest — the strings are what a
// stored envelope actually contains, and an envelope written years ago must
// still render — but it means nothing in that package can notice a new phase.
//
// This test lives here because internal/sink is a place that CAN see both
// sides. The failure it prevents is quiet and bad: a terminal phase the report
// package did not know about would make every run in it render as UNFINISHED
// forever, so a failed acceptance would be reported as "still running" rather
// than as a failure — and a consumer polling for a verdict would poll until the
// heat death of the fleet.
func TestReportKnowsEveryTerminalPhase(t *testing.T) {
	declared := runPhaseConstants(t)
	if len(declared) < 5 {
		t.Fatalf("found %d RunPhase constants, which cannot be right — this guard is "+
			"not reading the API and would pass whatever it said", len(declared))
	}

	for name, value := range declared {
		apiTerminal := burninv1alpha1.RunPhase(value).IsTerminal()
		reportTerminal := report.IsTerminalPhase(value)
		if apiTerminal == reportTerminal {
			continue
		}
		if apiTerminal {
			t.Errorf("%s (%q) is terminal in the API and not in pkg/report — every run "+
				"that ends in it renders as unfinished forever, so a settled verdict is "+
				"reported as still running", name, value)
			continue
		}
		t.Errorf("%s (%q) is terminal in pkg/report and not in the API — a run still "+
			"executing would be rendered as a verdict of record", name, value)
	}
}

// runPhaseConstants reads the API's own source, because a constant nobody
// references leaves nothing to reflect on at runtime — the same reason
// pkg/contract's TestReasonsIsComplete parses its own file.
func runPhaseConstants(t *testing.T) map[string]string {
	t.Helper()
	fset := token.NewFileSet()
	file, err := parser.ParseFile(fset, "../../api/v1alpha1/burninrun_types.go", nil, 0)
	if err != nil {
		t.Fatal(err)
	}
	out := map[string]string{}
	ast.Inspect(file, func(n ast.Node) bool {
		spec, ok := n.(*ast.ValueSpec)
		if !ok {
			return true
		}
		id, ok := spec.Type.(*ast.Ident)
		if !ok || id.Name != "RunPhase" {
			return true
		}
		for i, name := range spec.Names {
			if i >= len(spec.Values) {
				continue
			}
			lit, ok := spec.Values[i].(*ast.BasicLit)
			if !ok || lit.Kind != token.STRING {
				continue
			}
			out[name.Name] = lit.Value[1 : len(lit.Value)-1]
		}
		return true
	})
	return out
}
