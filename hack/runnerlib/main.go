// SPDX-License-Identifier: Apache-2.0
// Copyright the Glimmer authors.

// Command runnerlib stamps hack/runnerlib/runnerlib.go.src into every Go
// runner directory as zz_runnerlib.go, byte-identical.
//
// Run it from the repository root:
//
//	go run ./hack/runnerlib
//
// runners/runnerlib_test.go is the guard: a drifted or missing copy fails CI
// with instructions to run this. See the .src header for why this is a stamp
// rather than a package (the runner image build cannot import one).
package main

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"
)

func main() {
	src, err := os.ReadFile(filepath.Join("hack", "runnerlib", "runnerlib.go.src"))
	if err != nil {
		fmt.Fprintln(os.Stderr, "run this from the repository root:", err)
		os.Exit(1)
	}
	entries, err := os.ReadDir("runners")
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	for _, e := range entries {
		if !e.IsDir() {
			continue
		}
		dir := filepath.Join("runners", e.Name())
		if !hasGoSources(dir) {
			continue
		}
		dst := filepath.Join(dir, "zz_runnerlib.go")
		if cur, err := os.ReadFile(dst); err == nil && string(cur) == string(src) {
			continue
		}
		if err := os.WriteFile(dst, src, 0o644); err != nil {
			fmt.Fprintln(os.Stderr, err)
			os.Exit(1)
		}
		fmt.Println("stamped", dst)
	}
}

// hasGoSources reports whether the directory holds non-test Go files — the
// definition of "a Go runner" the guard test shares.
func hasGoSources(dir string) bool {
	names, _ := filepath.Glob(filepath.Join(dir, "*.go"))
	for _, n := range names {
		base := filepath.Base(n)
		if !strings.HasSuffix(base, "_test.go") && base != "zz_runnerlib.go" {
			return true
		}
	}
	return false
}
