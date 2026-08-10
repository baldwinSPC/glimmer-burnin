// SPDX-License-Identifier: Apache-2.0
// Copyright the Glimmer authors.

package runners

import (
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"testing"
)

// Several runners carry byte-identical copies of the same source file.
//
// That duplication is STRUCTURAL and cannot be factored out: each runner is its
// own Docker build context, `COPY` cannot reach outside it, and each image
// builds a throwaway Go module from the files in its own directory. It is the
// same constraint that puts cxxtests_test.go and the memory-bw metric-name test
// in this package rather than beside their runners.
//
// pins_test.go already cites this duplication as the precedent for accepting
// other duplication — and until now NOTHING CHECKED IT. Four files were copied
// across three fabric runners with no guard, so a fix to `rdma.go` in one runner
// would leave the identical bug in the other two, and the only way to notice
// would be for someone to hit it twice.
//
// "Duplication that cannot be removed is duplication that has to be checked" is
// the rule this file enforces, and it is derived rather than listed: any
// filename appearing in more than one runner directory must either be identical
// everywhere, or appear in divergent below with the reason it differs. A new
// shared file is picked up automatically; a new divergence has to be declared.

// perRunner are filenames that are each runner's OWN, not a shared source that
// happens to share a name. Every runner has a main.go and no two are alike;
// treating that as drift would bury the real signal under nine entries.
var perRunner = map[string]bool{
	"main.go": true,
}

// divergent are filenames whose copies deliberately differ, with why.
//
// A divergence is a real design decision every time it appears here. It is NOT
// a place to park drift: the entry has to say what makes the copies different,
// so a reader can tell "these two runners answer a question differently" from
// "somebody edited one and forgot the other".
var divergent = map[string]string{
	"rendezvous.go": "nccl speaks the GROUP contract (BURNIN_RANK/NRANKS/ROOT_HOST); " +
		"ib-write-bw and gpudirect-rdma speak the PAIR contract (BURNIN_ROLE/PEER_HOST). " +
		"Different rendezvous, so different code — not drift.",
	"memlock.go": "nccl REFUSES when RLIMIT_MEMLOCK is too small; ib-write-bw and " +
		"gpudirect-rdma SIZE THEMSELVES to fit it. Two defensible policies for the " +
		"same limit, and the runners chose differently on purpose.",

	// This one is a SUBSET rather than a different answer, and it is the entry
	// most worth revisiting: clockprobe's copy omits the ECC symbols
	// (nvmlDeviceGetEccMode, nvmlDeviceGetTotalEccErrors) that gpu-burn and
	// thermal-soak load, because clockprobe does not read ECC. The risk is that
	// a fix to the parts they DO share — the dlopen fallback chain, the
	// _v2-then-plain symbol resolution — lands in one copy and not the other,
	// and a subset diverging that way looks exactly like a subset that is
	// meant to. Unifying them would cost clockprobe two unused symbols.
	"nvml_dynamic.h": "clockprobe carries a SUBSET: no ECC symbols, because it does not " +
		"read ECC. gpu-burn and thermal-soak load the full set and are identical to " +
		"each other.",
}

func TestSharedRunnerSourcesDoNotDrift(t *testing.T) {
	dirs := runnerDirs(t)

	// filename -> digest -> the runners carrying that exact content.
	byName := map[string]map[string][]string{}

	for _, d := range dirs {
		entries, err := os.ReadDir(d)
		if err != nil {
			t.Fatalf("reading %s: %v", d, err)
		}
		for _, e := range entries {
			name := e.Name()
			if e.IsDir() || perRunner[name] ||
				strings.HasSuffix(name, "_test.go") || strings.HasSuffix(name, "_test.cc") {
				continue
			}
			// Only source files can be shared; a Dockerfile or README is
			// per-runner by nature and differs legitimately everywhere.
			if !strings.HasSuffix(name, ".go") && !strings.HasSuffix(name, ".cuh") &&
				!strings.HasSuffix(name, ".h") {
				continue
			}

			body, err := os.ReadFile(filepath.Join(d, name))
			if err != nil {
				t.Fatalf("reading %s/%s: %v", d, name, err)
			}
			sum := sha256.Sum256(body)
			digest := hex.EncodeToString(sum[:8])

			if byName[name] == nil {
				byName[name] = map[string][]string{}
			}
			byName[name][digest] = append(byName[name][digest], d)
		}
	}

	shared := 0
	for _, name := range sortedNames(byName) {
		groups := byName[name]

		// How many runners carry this filename at all.
		carriers := 0
		for _, dirs := range groups {
			carriers += len(dirs)
		}
		if carriers < 2 {
			continue // not shared; nothing to keep in step
		}
		shared++

		if len(groups) == 1 {
			continue // one digest, every copy identical — the good case
		}

		why, declared := divergent[name]
		if !declared {
			t.Errorf("%s exists in %d runners with %d DIFFERENT contents:\n%s\n"+
				"Either bring the copies back into line, or add %q to the divergent map in this file "+
				"with the reason they differ. A silent divergence means a fix landing in one runner and "+
				"not the others, which is invisible until somebody hits the same bug twice.",
				name, carriers, len(groups), describe(groups), name)
			continue
		}
		t.Logf("%s diverges deliberately across %d runners: %s", name, carriers, why)
	}

	// Asserted rather than assumed: a refactor that stopped sharing anything
	// would empty this sweep and leave it passing while checking nothing.
	if shared == 0 {
		t.Fatal("found no source file shared between runners, which cannot be right — " +
			"if the duplication genuinely ended, remove this guard with it")
	}
}

// TestEveryDeclaredDivergenceIsReal keeps the divergent map honest in the other
// direction.
//
// An entry that no longer describes anything is worse than no entry: it reads
// as a considered decision about the current code while describing code that
// has since been unified or deleted, and the next person to touch those files
// trusts it.
func TestEveryDeclaredDivergenceIsReal(t *testing.T) {
	dirs := runnerDirs(t)

	for name, why := range divergent {
		digests := map[string]bool{}
		carriers := 0
		for _, d := range dirs {
			body, err := os.ReadFile(filepath.Join(d, name))
			if err != nil {
				continue
			}
			carriers++
			sum := sha256.Sum256(body)
			digests[hex.EncodeToString(sum[:8])] = true
		}

		switch {
		case carriers == 0:
			t.Errorf("divergent lists %q, which no runner carries any more. Remove the entry: "+
				"a stale reason reads as a decision about code that is gone.\n  was: %s", name, why)
		case carriers == 1:
			t.Errorf("divergent lists %q, which only one runner carries now. Nothing can diverge "+
				"from itself; remove the entry.\n  was: %s", name, why)
		case len(digests) == 1:
			t.Errorf("divergent lists %q, but every copy is now IDENTICAL. If they were unified "+
				"deliberately, remove the entry so the guard starts holding them in step.\n  was: %s",
				name, why)
		}
	}
}

func describe(groups map[string][]string) string {
	var lines []string
	for digest, dirs := range groups {
		sort.Strings(dirs)
		lines = append(lines, fmt.Sprintf("  %s  %s", digest, strings.Join(dirs, ", ")))
	}
	sort.Strings(lines)
	return strings.Join(lines, "\n")
}

func sortedNames(m map[string]map[string][]string) []string {
	out := make([]string, 0, len(m))
	for k := range m {
		out = append(out, k)
	}
	sort.Strings(out)
	return out
}
