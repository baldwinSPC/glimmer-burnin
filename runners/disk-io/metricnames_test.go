// SPDX-License-Identifier: Apache-2.0
// Copyright the Glimmer authors.

package main

import (
	"go/parser"
	"go/token"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/baldwinSPC/glimmer-burnin/pkg/contract"
	"github.com/baldwinSPC/glimmer-burnin/pkg/runner"
)

// This file is the seam between the runner and the operator.
//
// The runner prints its own key names; pkg/runner maps them to the canonical
// names a threshold is written against, and pkg/contract owns the grammar those
// names must obey. Nothing at runtime reconciles a disagreement — a key that
// normalises to an illegal name is silently unthresholdable, and the failure
// shows up as a node nobody can accept for reasons nobody can see.
//
// These imports are test-only. The image build removes *_test.go, which is what
// lets the non-test sources stay standard-library-only.

func TestEmittedMetricNamesObeyTheContract(t *testing.T) {
	// The exact set report() can emit, plus the evidence label from run().
	emitted := []string{
		"writeBandwidthMBs=1234.567",
		"readBandwidthMBs=2345.678",
		"ioLatencyUs=812.400",
		"p99LatencyUs=9100.000",
		"ioErrors=0",
		"diskIoPath=/mnt/scratch",
	}

	res := runner.Parse("disk-io", strings.Join(emitted, "\n"), 0)
	if len(res.Metrics) != len(emitted) {
		t.Fatalf("parsed %d metrics from %d lines: %v", len(res.Metrics), len(emitted), res.Metrics)
	}

	for name := range res.Metrics {
		if _, known := contract.Lookup(name); !known {
			t.Errorf("%q is not in the metric registry — a threshold written against it can never "+
				"be evaluated, and nothing at runtime would say so", name)
		}
		if err := contract.ValidateMetricName(name); err != nil {
			t.Errorf("%q does not obey the name grammar: %v", name, err)
		}
	}
}

func TestZeroIoErrorsSurvivesParsingWhileAnUnmeasuredBandwidthIsAbsent(t *testing.T) {
	// The asymmetry that matters. `ioErrors=0` is a real measurement — the run
	// reached the device and counted none — and a gate of `ioErrors Equal 0`
	// needs to see it to pass a healthy node. A bandwidth of zero would be a
	// measurement nobody took, so report() omits the key entirely instead.
	res := runner.Parse("disk-io", "ioErrors=0\ndiskIoPath=/mnt/scratch", 0)
	if res.Metrics["ioErrors"] != "0" {
		t.Errorf("ioErrors did not survive as a measurement: %v", res.Metrics)
	}
	if _, present := res.Metrics["writeBandwidthMBs"]; present {
		t.Error("a bandwidth appeared from nowhere")
	}
}

// TestTheRunnerImportsNothingOutsideTheStandardLibrary is the licence guard.
//
// fio, IOR and elbencho are GPL, which is why this runner exists as its own Go
// program rather than a wrapper around one. That property has to be enforced
// rather than remembered: the Dockerfile builds with GOPROXY=off so a
// dependency fails the image build, and this catches it one step earlier, in
// `make test`, with an error that says why.
func TestTheRunnerImportsNothingOutsideTheStandardLibrary(t *testing.T) {
	entries, err := os.ReadDir(".")
	if err != nil {
		t.Fatal(err)
	}

	checked := 0
	for _, e := range entries {
		name := e.Name()
		if !strings.HasSuffix(name, ".go") || strings.HasSuffix(name, "_test.go") {
			continue
		}
		checked++

		f, err := parser.ParseFile(token.NewFileSet(), filepath.Join(".", name), nil, parser.ImportsOnly)
		if err != nil {
			t.Fatalf("parsing %s: %v", name, err)
		}
		for _, imp := range f.Imports {
			path := strings.Trim(imp.Path.Value, `"`)
			// A standard-library import path has no dot in its first segment.
			// That is the same rule the toolchain itself uses.
			if strings.Contains(strings.Split(path, "/")[0], ".") {
				t.Errorf("%s imports %q. This runner must stay standard-library-only: the tools it "+
					"replaces are GPL, and a copyleft dependency would make this project unpublishable",
					name, path)
			}
		}
	}

	// Asserted rather than assumed: a rename that emptied this list would leave
	// the guard passing while checking nothing.
	if checked == 0 {
		t.Fatal("found no non-test .go files to check, which cannot be right")
	}
}
