package invariants

import (
	"os/exec"
	"strings"
	"testing"
)

// The publish workflows' "is this a Go binary?" test must use `go version -m`.
//
// `go version <file>` on a non-Go executable writes "not a Go executable" to
// STDERR and EXITS 0. `go version -m <file>` exits 1. Both exit 0 on a real Go
// binary, so only the second form discriminates — and the bare form reads like
// it works, which is why it survived review and a release.
//
// The cost was #386. publish-runner.yml carried exactly this guard, written for
// the six C++/CUDA runners that have no Go binary, and it could never fire:
// every such runner fell through to `govulncheck -mode=binary`, which died with
// "unrecognized binary format" AFTER the tag had been pushed. Publishing
// gemm-sweep:v0.6.4 reported FAILURE over a tag that was correct, signed, and a
// proper two-platform index.
//
// A red publish run must mean a bad publish. It must not also mean "this runner
// is not written in Go".
func TestThePublishScansUseGoVersionDashM(t *testing.T) {
	for _, wf := range []string{
		".github/workflows/publish-runner.yml",
		".github/workflows/publish-operator.yml",
	} {
		body := read(t, wf)

		// Every `go version` invocation that is being TESTED — i.e. whose exit
		// status is consumed by an `if` — must carry -m. An unchecked `go
		// version -m ... ` inside $( ) for display is fine and is not matched
		// here, because it is not deciding anything.
		for _, line := range strings.Split(body, "\n") {
			l := strings.TrimSpace(line)
			if !strings.HasPrefix(l, "if ! go version") && !strings.HasPrefix(l, "if go version") {
				continue
			}
			if !strings.Contains(l, "go version -m") {
				t.Errorf("%s decides on a bare `go version`:\n    %s\n"+
					"That exits 0 for a non-Go binary, so the branch can never be taken. "+
					"Use `go version -m`, which exits 1. See #386.", wf, l)
			}
		}
	}
}

// The behaviour the guard above depends on, asserted against the real toolchain
// rather than trusted from documentation.
//
// If a future Go release makes `go version` exit non-zero on a non-Go file,
// this test starts failing and the comment above stops being true — which is
// the moment to revisit it, rather than discovering it from a publish that
// silently changed shape.
func TestGoVersionDashMIsWhatDiscriminatesANonGoBinary(t *testing.T) {
	if _, err := exec.LookPath("go"); err != nil {
		t.Skip("no go toolchain on PATH")
	}
	// /bin/ls is a native executable on every platform this runs on, and is
	// emphatically not a Go binary.
	const nonGo = "/bin/ls"

	bare := exec.Command("go", "version", nonGo).Run()
	withM := exec.Command("go", "version", "-m", nonGo).Run()

	if bare != nil {
		t.Errorf("`go version %s` returned %v, want success — the whole reason this project "+
			"cannot use the bare form to detect a non-Go binary is that it SUCCEEDS here. "+
			"If that has changed, the guard in the publish workflows can be simplified.", nonGo, bare)
	}
	if withM == nil {
		t.Errorf("`go version -m %s` succeeded, want failure — this is the discriminator the "+
			"publish workflows rely on to skip scanning a C++/CUDA runner. If it no longer "+
			"fails, those workflows will try to govulncheck a non-Go binary again (#386).", nonGo)
	}
}
