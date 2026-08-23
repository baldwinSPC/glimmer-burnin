// Command checkpins is `go run ./hack/checkpins`: for every kind
// pkg/runnerimages pins to a published image, checks the image's OWN recorded
// build commit against runners/<kind>/'s current source — and fails, naming
// the kind, when the directory has moved since the pin was built (#379).
//
// # The gap this closes
//
// A pin equal to the newest tag is not the same claim as a pin whose image was
// built from the source in this tree, and only the second one means anything
// to a node that pulls it. That gap opened twice in one day (dcgm-diag and
// host-health both drifted from their pins, #379) and was caught only because
// a human happened to diff a runner directory against the commit its image
// was built from while doing something else.
//
// # How it answers the question
//
// publish-runner.yml already stamps every image it builds with
// org.opencontainers.image.revision=${{ github.sha }} — the commit the image
// was actually built from. This program reads that label back via `crane
// config` (the same tool, and the same invocation shape, publish-runner.yml's
// own binary-provenance step already uses) and asks git a direct question:
// has anything under runners/<kind>/ changed between that commit and HEAD? If
// so, the pin is stale — the published image does not reflect the source this
// tree now carries for that kind.
//
// # Why this is a separate program and not part of `make test`
//
// It makes a network call to GHCR for every pinned kind, and a network call in
// the unit suite means a suite that flakes on registry latency — worse than a
// suite that never asked the question. It runs as its own CI step instead
// (the supply-chain job already makes network calls of this shape:
// govulncheck, the Helm chart lint, golangci-lint's linter downloads).
//
// It also needs full git history to resolve an arbitrarily old pinned commit,
// which a shallow checkout does not carry — the CI step that runs this must
// check out with fetch-depth: 0.
package main

import (
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
	"strings"

	"github.com/baldwinSPC/glimmer-burnin/pkg/runnerimages"
)

// craneModule pins the exact version already used by publish-operator.yml and
// publish-runner.yml's own binary-provenance step, so this program and those
// workflows agree about which crane they are running — never a moving target.
const craneModule = "github.com/google/go-containerregistry/cmd/crane@v0.21.9"

// craneConfig is `crane config <ref>` — the same tool and the same
// invocation shape publish-runner.yml's binary-provenance step already uses,
// so a failure mode this program has not seen before is one the publish
// workflow has not seen either.
func craneConfig(ref string) (map[string]any, error) {
	cmd := exec.Command("go", "run", craneModule, "config", ref)
	out, err := cmd.Output()
	if err != nil {
		detail := err.Error()
		if ee, ok := err.(*exec.ExitError); ok {
			detail = strings.TrimSpace(string(ee.Stderr))
		}
		return nil, fmt.Errorf("crane config %s: %s", ref, detail)
	}
	var cfg map[string]any
	if err := json.Unmarshal(out, &cfg); err != nil {
		return nil, fmt.Errorf("crane config %s: not valid JSON: %w", ref, err)
	}
	return cfg, nil
}

// buildRevision extracts org.opencontainers.image.revision from a crane
// config response. Absent (never published through this project's own
// workflow, or published before the label existed) is reported rather than
// treated as clean — an image this program cannot date is not one it can
// vouch for.
func buildRevision(cfg map[string]any) (string, bool) {
	config, _ := cfg["config"].(map[string]any)
	labels, _ := config["Labels"].(map[string]any)
	rev, ok := labels["org.opencontainers.image.revision"].(string)
	return rev, ok && rev != ""
}

// sourceMovedSince reports whether runners/<kind>/ has any change between rev
// and HEAD. Uses `git diff --quiet`, whose exit code IS the answer: 0 means no
// difference, 1 means a difference, anything else is a git failure this
// program must not read as either.
func sourceMovedSince(rev, dir string) (bool, error) {
	cmd := exec.Command("git", "diff", "--quiet", rev, "HEAD", "--", dir)
	err := cmd.Run()
	if err == nil {
		return false, nil
	}
	if ee, ok := err.(*exec.ExitError); ok && ee.ExitCode() == 1 {
		return true, nil
	}
	return false, fmt.Errorf("git diff %s..HEAD -- %s: %w", rev, dir, err)
}

func main() {
	var stale []string
	var problems []string

	for kind, ref := range runnerimages.All() {
		dir := "runners/" + string(kind)
		if _, err := os.Stat(dir); err != nil {
			// Not every pinned kind's directory name matches the repo layout
			// 1:1 forever — this program must not invent a mapping table
			// that can drift from runners/pins_test.go's own kindForDir, so
			// an unresolvable directory is reported and skipped rather than
			// guessed at.
			problems = append(problems, fmt.Sprintf("%s: pinned image %s, but runners/%s does not exist", kind, ref, kind))
			continue
		}

		cfg, err := craneConfig(ref)
		if err != nil {
			problems = append(problems, fmt.Sprintf("%s: %v", kind, err))
			continue
		}
		rev, ok := buildRevision(cfg)
		if !ok {
			problems = append(problems, fmt.Sprintf(
				"%s: pinned image %s carries no org.opencontainers.image.revision label — this program cannot "+
					"date it, so it cannot vouch for it either", kind, ref))
			continue
		}

		moved, err := sourceMovedSince(rev, dir)
		if err != nil {
			problems = append(problems, fmt.Sprintf("%s: %v", kind, err))
			continue
		}
		if moved {
			stale = append(stale, fmt.Sprintf(
				"%s: pinned %s was built from %s, but %s has changed since — republish before trusting this pin",
				kind, ref, rev[:12], dir))
		}
	}

	if len(problems) > 0 {
		fmt.Println("Could not check every pin (treated as failures, not passes — an unresolvable pin is not a clean one):")
		for _, p := range problems {
			fmt.Println("  - " + p)
		}
	}
	if len(stale) > 0 {
		fmt.Println("The following pinned images were built from a commit older than their runner's current source:")
		for _, s := range stale {
			fmt.Println("  - " + s)
		}
	}
	if len(problems) > 0 || len(stale) > 0 {
		os.Exit(1)
	}
	fmt.Println("Every pinned runner image matches its runner's current source.")
}
