// SPDX-License-Identifier: Apache-2.0
// Copyright the Glimmer authors.

package runners

import (
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strings"
	"testing"

	"github.com/baldwinSPC/glimmer-burnin/pkg/contract"
	"github.com/baldwinSPC/glimmer-burnin/pkg/localrun"
	"github.com/baldwinSPC/glimmer-burnin/pkg/runner"
)

// A Node verdict describes EVERY accelerator on the node, gated on the worst;
// a single-device measurement on a multi-device node is a bug
// (docs/dev/multi-device.md). These guards are what keep that true in code
// rather than in a design note.

// deviceFoldHeader is the shared, CUDA-free header. It is byte-identical in
// every runner that carries it — each is its own build context — and
// sharedsource_test.go refuses drift; this file needs one copy to read.
const deviceFoldHeader = "gpu-burn/device_fold.h"

// TestEveryAcceleratorRunnerIteratesDevicesOrSaysWhyNot is the TOTAL table.
//
// Every runner directory that touches an accelerator is in exactly one of
// three states, and a new one that is in none fails the build:
//
//   - CONVERTED: its source uses the shared iteration helper, so it measures
//     every device it was allocated and folds them.
//   - EXEMPT, with the reason: the iteration happens somewhere this header
//     cannot reach (nvbandwidth iterates; dcgmi enumerates; NVML is already
//     per GPU), or the kind's measurement is one endpoint per pod by design.
//   - PENDING, with the issue that converts it: today's honest state for
//     every CUDA/HIP kernel in the tree. A PENDING runner that starts using
//     the helper fails until it is moved to CONVERTED — a table that is
//     allowed to be stale is a table nobody reads.
//
// The runners that touch no accelerator at all are exempt by construction
// (they are the noAccelerator set in TestDriverInjectionIsDeclaredWhereItIsNeeded)
// and are listed here too so the table is total over runnerDirs.
func TestEveryAcceleratorRunnerIteratesDevicesOrSaysWhyNot(t *testing.T) {
	// The claim is the INCLUDE, not a call spelling: a runner that converts with
	// `using namespace burnin::devices;` calls planIteration unqualified, and a
	// substring on the qualified name would read it as still pending. The
	// header is only ever included by a runner that iterates through it.
	const helper = `#include "device_fold.h"`

	converted := map[string]bool{
		"gpu-burn":           true,
		"thermal-soak":       true,
		"power-swing":        true,
		"gpu-burn-rocm":      true,
		"thermal-soak-rocm":  true,
		"clockprobe":         true,
		"clockprobe-rocm":    true,
		"compute-smoke":      true,
		"compute-smoke-rocm": true,
		"gemm-sweep":         true,
		// memory-bw's conversion is the allocation budget check around
		// nvbandwidth (#431) — see noFoldTable below for why it carries no
		// kDeviceFold table despite being CONVERTED. memory-bw-rocm's HIP loop
		// got the full per-device conversion, mirroring clockprobe-rocm.
		"memory-bw":      true,
		"memory-bw-rocm": true,
	}

	exempt := map[string]string{
		"dcgm-diag":         "a node-wide, read-only diagnostic: dcgmi enumerates and diagnoses every device the driver exposes, loads nothing, and reports devices_visible rather than claiming deviceCount",
		"host-health":       "a node-wide, read-only probe: NVML per GPU already, loads nothing, reports devices_visible rather than claiming deviceCount",
		"fingerprint-probe": "READS ABOUT accelerators over sysfs and never opens one",
		"nccl":              "one rank per pod is the measurement at Pair and Group scope; Node scope over all visible devices is its own design (P0.2)",
		"nccl-rocm":         "the ROCm sibling of nccl; same reason",
		"gpudirect-rdma":    "a Pair kind: one endpoint per pod IS the measurement",
		"ib-write-bw":       "no accelerator; RDMA verbs over the wire",
		"fabric-soak":       "no accelerator; iterates ib_write_bw over the wire",
		"memory-stress":     "no accelerator; host RAM",
		"disk-io":           "no accelerator; storage",
		"tcp-baseline":      "no accelerator; the kernel TCP stack",
	}

	// PENDING: every CUDA/HIP kernel that today takes device 0. Each names the
	// delivery step in docs/dev/multi-device.md that converts it; the issue
	// number is the issue that converts it.
	pending := map[string]string{}

	for _, d := range runnerDirs(t) {
		uses := sourceMentions(t, d, helper)
		switch {
		case converted[d]:
			if !uses {
				t.Errorf("%s is listed as CONVERTED but its source never calls %s; it is back to measuring one device and nothing says so", d, helper)
			}
		case exempt[d] != "":
			if uses {
				t.Errorf("%s is EXEMPT (%s) but its source calls %s; either the exemption is stale or the helper is being used where the design says it does not apply", d, exempt[d], helper)
			}
		case pending[d] != "":
			if uses {
				t.Errorf("%s is listed as PENDING (%s) but already calls %s; move it to CONVERTED so the table stays true", d, pending[d], helper)
			}
		default:
			t.Errorf("runners/%s is in none of CONVERTED, EXEMPT or PENDING in TestEveryAcceleratorRunnerIteratesDevicesOrSaysWhyNot. "+
				"An accelerator runner that measures one device on a multi-device node certifies hardware nobody measured; "+
				"either use the shared iteration helper (device_fold.h), or say why this runner does not need to", d)
		}
	}

	// A CONVERTED runner must also carry the header — the helper is header-only.
	for d := range converted {
		if _, err := os.Stat(filepath.Join(d, "device_fold.h")); err != nil {
			t.Errorf("%s is CONVERTED but carries no device_fold.h: %v", d, err)
		}
	}
}

// sourceMentions reports whether any SOURCE file (never a README) in a runner
// directory contains needle.
func sourceMentions(t *testing.T, dir, needle string) bool {
	t.Helper()
	entries, err := os.ReadDir(dir)
	if err != nil {
		t.Fatalf("reading %s: %v", dir, err)
	}
	for _, e := range entries {
		if e.IsDir() {
			continue
		}
		name := e.Name()
		switch filepath.Ext(name) {
		case ".cu", ".cuh", ".cc", ".h", ".go":
		default:
			continue
		}
		if strings.HasSuffix(name, "_test.cc") || strings.HasSuffix(name, "_test.go") || name == "device_fold.h" {
			continue // the header and its test are not a runner including it
		}
		raw, err := os.ReadFile(filepath.Join(dir, name))
		if err != nil {
			t.Fatalf("reading %s: %v", filepath.Join(dir, name), err)
		}
		if strings.Contains(string(raw), needle) {
			return true
		}
	}
	return false
}

// TestDeviceFoldAllowSetsAgreeWithTheCLI holds the runners' vendor allow-sets
// (device_fold.h: nvidiaResources / amdResources) to pkg/localrun's
// vendorResources table.
//
// The bare-metal CLI reads the same profile a cluster does and maps a resource
// name to a vendor to decide which device flags to pass; a runner reads the
// same names, injected as BURNIN_RESOURCE_LIMITS, to decide how many devices it
// was allocated. A name the CLI treats as an accelerator that a runner refuses
// as unrecognised — or the reverse — is one profile meaning two things across
// the two dispatchers, which is precisely the failure the shared vocabulary
// exists to prevent. Read from the source because the header is C++.
func TestDeviceFoldAllowSetsAgreeWithTheCLI(t *testing.T) {
	raw, err := os.ReadFile(deviceFoldHeader)
	if err != nil {
		t.Fatalf("reading %s: %v", deviceFoldHeader, err)
	}
	// inline VendorResources nvidiaResources() { return {"nvidia.com/", {"nvidia.com/gpu"}, {"nvidia.com/mig-"}}; }
	re := regexp.MustCompile(`inline VendorResources \w+Resources\(\) \{ return \{"([^"]+)", \{([^}]*)\}, \{([^}]*)\}\}; \}`)
	matches := re.FindAllStringSubmatch(string(raw), -1)
	if len(matches) == 0 {
		t.Fatalf("found no vendor allow-set in %s; the pattern this guard reads has moved", deviceFoldHeader)
	}
	var headerExact, headerPrefixes []string
	quoted := regexp.MustCompile(`"([^"]+)"`)
	for _, m := range matches {
		domain := m[1]
		for _, q := range quoted.FindAllStringSubmatch(m[2], -1) {
			if !strings.HasPrefix(q[1], domain) {
				t.Errorf("%s: exact name %q is not under its own vendor domain %q", deviceFoldHeader, q[1], domain)
			}
			headerExact = append(headerExact, q[1])
		}
		for _, q := range quoted.FindAllStringSubmatch(m[3], -1) {
			if !strings.HasPrefix(q[1], domain) {
				t.Errorf("%s: prefix %q is not under its own vendor domain %q", deviceFoldHeader, q[1], domain)
			}
			headerPrefixes = append(headerPrefixes, q[1])
		}
	}
	sort.Strings(headerExact)
	sort.Strings(headerPrefixes)
	if cli := localrun.VendorResourceNames(); strings.Join(headerExact, ",") != strings.Join(cli, ",") {
		t.Errorf("device_fold.h exact allow-set %v and pkg/localrun.VendorResourceNames %v differ; "+
			"the two dispatchers would read one profile's accelerator request as two different things", headerExact, cli)
	}
	// Prefixes too: a MIG-sliced profile (nvidia.com/mig-1g.5gb: 7) that the
	// runner counts as seven instances must not be a profile the CLI reads as
	// "no accelerator" and runs with no device at all.
	if cli := localrun.VendorResourcePrefixes(); strings.Join(headerPrefixes, ",") != strings.Join(cli, ",") {
		t.Errorf("device_fold.h prefix allow-set %v and pkg/localrun.VendorResourcePrefixes %v differ; "+
			"a MIG profile would mean instances to the runner and nothing to the CLI", headerPrefixes, cli)
	}
}

// TestDeviceFoldTablesAgreeWithTheRegistry reads every runner's fold table and
// asserts each Min/Max/Sum entry agrees with the registry's Aggregation for
// the key's canonical name — so a runner cannot decide, privately, that a floor
// is a ceiling.
//
// Two dimensions, deliberately: Aggregation is ACROSS WINDOWS, the table is
// ACROSS DEVICES, and they agree exactly where Aggregation names a direction
// (Min, Max, Sum). A registered Last metric is the class where Aggregation
// says nothing about direction — a lifetime counter such as eccErrors is Last
// across windows and SUMS across devices; a label is Last and is emitted Once
// — so a Last metric is not checked against the table's direction, and an
// unregistered key is not checked at all (nothing declared it). Once entries
// are never checked.
//
// The table is `static const burnin::devices::FoldRule kDeviceFold[] = { {"raw_key", Fold::Min}, … };`
// in the runner's own source. A runner that includes the header must declare
// one that this guard can read — a converted runner whose table yields no
// entries would pass vacuously, and a floor could be declared a ceiling in
// private.
func TestDeviceFoldTablesAgreeWithTheRegistry(t *testing.T) {
	entry := regexp.MustCompile(`\{"([a-z0-9_]+)",\s*(?:burnin::)?(?:devices::)?Fold::(Min|Max|Sum|Once)\}`)

	// noFoldTable holds the one CONVERTED runner that carries device_fold.h and
	// calls it purely for the allocation BUDGET CHECK, never for a per-device
	// fold: nvbandwidth already iterates every visible device on its own, so
	// memory-bw's conversion is parseBudget/planIteration around the
	// invocation, and there is nothing here for kDeviceFold to declare
	// (docs/dev/multi-device.md, #431). Every other CONVERTED runner must still
	// declare a checkable table.
	noFoldTable := map[string]bool{
		"memory-bw": true,
	}

	found := 0
	for _, d := range runnerDirs(t) {
		if _, err := os.Stat(filepath.Join(d, "device_fold.h")); err != nil {
			continue
		}
		includes := sourceMentions(t, d, `#include "device_fold.h"`)
		perDir := 0
		entries, err := os.ReadDir(d)
		if err != nil {
			t.Fatal(err)
		}
		for _, e := range entries {
			name := e.Name()
			if e.IsDir() || strings.HasSuffix(name, "_test.cc") || name == "device_fold.h" {
				continue
			}
			switch filepath.Ext(name) {
			case ".cu", ".cuh", ".cc", ".h":
			default:
				continue
			}
			raw, err := os.ReadFile(filepath.Join(d, name))
			if err != nil {
				t.Fatal(err)
			}
			body := string(raw)
			if !strings.Contains(body, "kDeviceFold") {
				continue
			}
			kind, _ := kindForDir(d)
			for _, m := range entry.FindAllStringSubmatch(body, -1) {
				found++
				perDir++
				// Through the real parser for the runner's kind, so an alias
				// table entry is honoured the way it is at harvest time.
				parsed := runner.Parse(string(kind), m[1]+"=1\n", 0)
				canonical := ""
				for k := range parsed.Metrics {
					canonical = k
				}
				if canonical == "" {
					t.Errorf("%s/%s: fold-table key %q did not parse to a metric name", d, name, m[1])
					continue
				}
				want := map[string]contract.Aggregation{
					"Min": contract.AggMin, "Max": contract.AggMax, "Sum": contract.AggSum,
				}[m[2]]
				if m[2] == "Once" {
					continue
				}
				reg, registered := contract.Lookup(canonical)
				if !registered || reg.Aggregation == contract.AggLast {
					continue // no direction to agree with; see the doc comment
				}
				got := reg.Aggregation
				if got != want {
					t.Errorf("%s/%s folds %q (%s) as %s across devices but the registry aggregates it %s across windows; "+
						"the direction is the metric's, declared once beside its name — fix the table or the registry, not both",
						d, name, m[1], canonical, m[2], got)
				}
			}
		}
		if includes && perDir == 0 && !noFoldTable[d] {
			t.Errorf("%s includes device_fold.h but declares no kDeviceFold table this guard can read "+
				"(`{\"raw_key\", Fold::Min}` entries); a converted runner's fold direction must be checkable here, "+
				"or a floor can be declared a ceiling in private", d)
		}
	}
	t.Logf("checked %d fold-table entries", found)
}
