// SPDX-License-Identifier: Apache-2.0
// Copyright the Glimmer authors.

package runners

import (
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
)

// TestCxxUnitTests compiles and runs every `*_test.cc` under runners/.
//
// Some runners are C++ or CUDA, and the part of one that is worth unit-testing
// is never the part that needs a GPU: it is the parsing, the classification,
// the decision that quietly returns a wrong answer when an upstream's output
// shifts or a build arg changes. Those tests are written as plain C++ programs
// with no framework, and until this existed nothing ran them —
// memory-bw/README.md records exactly that ("`scan_test.cc` is still not run by
// CI"), because `make test` covers Go packages and a C++ file is not one.
//
// A guard nobody runs is not a guard. This is the seam package, a directory
// listing is the only thing that knows how many such tests there are, and
// running them needs no Docker, no CUDA toolchain and no GPU — which is why the
// headers they cover are kept free of CUDA in the first place.
//
// It deliberately does NOT live in each runner's own directory. `go build`
// refuses a package directory containing C++ sources unless cgo is in use, so a
// Go file next to a `.cc` file breaks the build of the whole module.
func TestCxxUnitTests(t *testing.T) {
	tests := findCxxTests(t)

	// Asserted rather than assumed: if a rename or a moved file empties this
	// list, the sweep would otherwise pass while testing nothing at all.
	if len(tests) == 0 {
		t.Fatal("found no *_test.cc under runners/, which cannot be right — " +
			"if the last one was genuinely removed, remove this guard with it")
	}

	cxx := compiler(t)
	for _, src := range tests {
		t.Run(src, func(t *testing.T) {
			dir, file := filepath.Split(src)
			bin := filepath.Join(t.TempDir(), strings.TrimSuffix(file, ".cc"))

			// The same invocation each file documents in its own header, so a
			// developer running it by hand sees what CI saw.
			build := exec.Command(cxx, "-std=c++17", "-O1", "-Wall", "-Wextra", "-o", bin, file)
			build.Dir = dir
			if out, err := build.CombinedOutput(); err != nil {
				// A compile error fails as loudly as a wrong answer: a header
				// that no longer builds standalone has lost the property that
				// makes it testable without hardware.
				t.Fatalf("compiling %s with %s failed: %v\n%s", src, cxx, err, out)
			}

			run := exec.Command(bin)
			run.Dir = dir
			out, err := run.CombinedOutput()
			if err != nil {
				t.Fatalf("%s failed: %v\n%s", src, err, out)
			}
			t.Logf("%s", out)
		})
	}
}

// findCxxTests lists every C++ unit test in the tree, relative to this package.
func findCxxTests(t *testing.T) []string {
	t.Helper()
	var found []string
	err := filepath.WalkDir(".", func(path string, d os.DirEntry, err error) error {
		if err != nil {
			return err
		}
		if !d.IsDir() && strings.HasSuffix(path, "_test.cc") {
			found = append(found, path)
		}
		return nil
	})
	if err != nil {
		t.Fatalf("walking runners/: %v", err)
	}
	return found
}

// compiler finds a C++ compiler, honouring CXX so a cross-compiling or
// container build can point at its own.
//
// A missing compiler skips rather than fails: this is a repository of Go and
// CUDA, and `go test ./...` should not demand a C++ toolchain of someone
// working on the reconciler. Both GitHub runner images ship g++, so the guard is
// real where it matters — and the skip names itself rather than passing quietly.
func compiler(t *testing.T) string {
	t.Helper()
	for _, c := range []string{os.Getenv("CXX"), "c++", "g++", "clang++"} {
		if c == "" {
			continue
		}
		if path, err := exec.LookPath(c); err == nil {
			return path
		}
	}
	t.Skip("no C++ compiler found (tried $CXX, c++, g++, clang++); no *_test.cc was run")
	return ""
}
