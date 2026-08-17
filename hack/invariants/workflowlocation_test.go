package invariants

import (
	"io/fs"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// GitHub reads .github/ at the REPOSITORY ROOT and nowhere else, so a workflow
// filed anywhere but .github/workflows/ is a file that looks live and is not.
//
// This is the same failure as the CLAUDE.md / invariants.md pair above, reached
// by a different route: two copies that drift are two different rules, and the
// one somebody follows is whichever they found first. Here it had already
// happened. A tracked .github/.github/ held eleven files including copies of all
// three workflows, 27, 58 and 98 lines behind the real ones (#367).
//
// The cost was not hypothetical. Grepping the publish path — the thing that
// mints immutable public tags — returned two hits per match at different line
// numbers, with nothing in either path to say which one CI actually runs.
//
// Kept in hack/invariants rather than beside the workflows because there is no
// Go package under .github/ to hang it on, and this is where the repository
// already asserts facts about its own layout.
func TestNoWorkflowLivesOutsideDotGithubWorkflows(t *testing.T) {
	root := filepath.Join("..", "..")

	var stray []string
	err := filepath.WalkDir(root, func(path string, d fs.DirEntry, err error) error {
		if err != nil {
			return err
		}
		rel, relErr := filepath.Rel(root, path)
		if relErr != nil {
			return relErr
		}
		rel = filepath.ToSlash(rel)

		if d.IsDir() {
			// .git holds packed copies of everything; worktrees and vendored
			// trees are not this repository's own layout.
			base := d.Name()
			if base == ".git" || base == "vendor" || base == "node_modules" {
				return filepath.SkipDir
			}
			// A .github nested anywhere other than the root is dead by
			// construction, whatever it contains.
			if base == ".github" && rel != ".github" {
				stray = append(stray, rel+"/ (a .github that is not the repository root's)")
				return filepath.SkipDir
			}
			return nil
		}

		ext := filepath.Ext(rel)
		if ext != ".yml" && ext != ".yaml" {
			return nil
		}
		if !strings.Contains(rel, "workflows/") {
			return nil
		}
		if strings.HasPrefix(rel, ".github/workflows/") {
			return nil
		}
		stray = append(stray, rel)
		return nil
	})
	if err != nil {
		t.Fatalf("walking the repository: %v", err)
	}

	if len(stray) > 0 {
		t.Errorf("these look like GitHub workflow files but GitHub will never run them:\n  %s\n\n"+
			"Only .github/workflows/ at the repository root is live. A copy anywhere else is a\n"+
			"file that reads like CI and is not, and it drifts silently — see #367, where the\n"+
			"nested publish-runner.yml had fallen 98 lines behind the one that actually runs.",
			strings.Join(stray, "\n  "))
	}
}

// The root .github/ still holds the things GitHub reads there.
//
// The deletion in #367 removed a nested duplicate, and the failure mode of that
// edit is removing one level too many. This is cheap and says which files the
// repository is claiming to have.
func TestTheRootGithubDirectoryIsIntact(t *testing.T) {
	for _, rel := range []string{
		".github/workflows/ci.yml",
		".github/workflows/publish-operator.yml",
		".github/workflows/publish-runner.yml",
		".github/PULL_REQUEST_TEMPLATE.md",
		".github/dependabot.yml",
	} {
		if _, err := os.Stat(filepath.Join("..", "..", rel)); err != nil {
			t.Errorf("%s is missing: %v", rel, err)
		}
	}
}
