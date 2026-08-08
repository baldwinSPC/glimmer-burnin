package contract

import (
	"go/ast"
	"go/parser"
	"go/token"
	"testing"
)

// Reasons must name every Reason constant this package declares.
//
// The list exists so tests can be TOTAL over the delivery reasons. That only
// holds while it is complete, and nothing in Go can check completeness at
// runtime — a constant that is never referenced leaves no trace to reflect on.
// So this reads the source, which is the same technique runners/pins_test.go
// uses for the same reason: the property is about what was WRITTEN.
//
// A forgotten entry is not a cosmetic slip. Every test that ranges over Reasons
// would keep passing while silently no longer covering the new delivery path —
// and a new delivery path is exactly where a partial summary or an envelope that
// fails validation would be easiest to introduce.
func TestReasonsIsComplete(t *testing.T) {
	fset := token.NewFileSet()
	file, err := parser.ParseFile(fset, "envelope.go", nil, 0)
	if err != nil {
		t.Fatal(err)
	}

	declared := map[string]bool{}
	ast.Inspect(file, func(n ast.Node) bool {
		spec, ok := n.(*ast.ValueSpec)
		if !ok {
			return true
		}
		id, ok := spec.Type.(*ast.Ident)
		if !ok || id.Name != "Reason" {
			return true
		}
		for _, name := range spec.Names {
			declared[name.Name] = true
		}
		return true
	})
	if len(declared) == 0 {
		t.Fatal("found no Reason constants in envelope.go — this guard is not reading " +
			"what it thinks it is, and would pass no matter what the source said")
	}

	listed := map[string]bool{}
	ast.Inspect(file, func(n ast.Node) bool {
		spec, ok := n.(*ast.ValueSpec)
		if !ok || len(spec.Names) != 1 || spec.Names[0].Name != "Reasons" {
			return true
		}
		lit, ok := spec.Values[0].(*ast.CompositeLit)
		if !ok {
			return true
		}
		for _, el := range lit.Elts {
			if id, ok := el.(*ast.Ident); ok {
				listed[id.Name] = true
			}
		}
		return true
	})

	for name := range declared {
		if !listed[name] {
			t.Errorf("%s is declared but missing from Reasons — every test that ranges "+
				"over Reasons has silently stopped covering it", name)
		}
	}
	for name := range listed {
		if !declared[name] {
			t.Errorf("Reasons names %s, which is not a Reason constant", name)
		}
	}
	if len(listed) != len(Reasons) {
		t.Errorf("the source lists %d reasons but the value has %d — the guard is reading "+
			"a different expression than the one in use", len(listed), len(Reasons))
	}
}
