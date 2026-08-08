package community

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// The files a stranger looks for must exist, and must say the things that make
// them useful rather than merely present — issue #176.
//
// This is a guard against a specific and unglamorous failure: these files are
// added once, in a burst, and then a rename or a tidy-up removes one and nobody
// notices for a year, because nothing builds them and nothing reads them in CI.
// The repository shipped public for several releases with none of them at all.
//
// It checks CONTENT and not just existence, because an empty SECURITY.md is
// worse than none: it answers the question "is there a private reporting path"
// with a yes that turns out to be no.
func TestTheCommunityFilesExistAndSayWhatTheyMustSay(t *testing.T) {
	cases := map[string][]string{
		"SECURITY.md": {
			// The private path. A public issue is the wrong first move for an
			// operator that cordons nodes and runs semi-trusted images.
			"security/advisories/new",
			// The scope items a reader would not guess, and which are the whole
			// reason this project's security surface is unusual.
			"runner image",
			"published tag",
			// The immutability rule, which decides what a fix looks like.
			"never re-publish",
		},
		"SUPPORT.md":         {"SECURITY.md", "issue"},
		"MAINTAINERS.md":     {"baldwinSPC"},
		"CONTRIBUTING.md":    {"SECURITY.md", "Signed-off-by", "immutable"},
		"CODE_OF_CONDUCT.md": {"Contributor Covenant"},
		".github/CODEOWNERS": {"@baldwinSPC"},
		".github/PULL_REQUEST_TEMPLATE.md": {
			// The two things CONTRIBUTING.md demands and that otherwise depend
			// on memory.
			"licence", "watched failing",
		},
		".github/ISSUE_TEMPLATE/config.yml": {"security/advisories/new"},
		".github/ISSUE_TEMPLATE/bug_report.yml": {
			// Without the exact tag a verdict cannot be attributed to what
			// produced it.
			"Runner image and tag",
		},
		".github/ISSUE_TEMPLATE/new-runner.yml": {
			// The question that decides whether a verdict can condemn a node.
			"FAIL from an ERROR",
			// The three that are always reconstructed later if not asked now.
			"_SKIP", "aggregation", "permissive only", "dual-licensed",
		},
		".github/ISSUE_TEMPLATE/new-vendor.yml":      {"MEAN the same thing"},
		".github/ISSUE_TEMPLATE/feature_request.yml": {"vendor branch in the reconciler"},
	}

	for name, wants := range cases {
		t.Run(name, func(t *testing.T) {
			path := filepath.Join("..", "..", name)
			data, err := os.ReadFile(path)
			if err != nil {
				t.Fatalf("%s is missing. A stranger arriving at this repository cannot "+
					"answer the question it exists to answer: %v", name, err)
			}
			body := string(data)
			if len(strings.TrimSpace(body)) < 80 {
				t.Fatalf("%s is effectively empty, which is worse than absent — it "+
					"answers its question with a yes that turns out to be no", name)
			}
			for _, want := range wants {
				if !strings.Contains(body, want) {
					t.Errorf("%s does not mention %q", name, want)
				}
			}
		})
	}
}

// Nothing in the repository may point a security report at the public issue
// tracker.
func TestNoDocumentSendsASecurityReportToAPublicIssue(t *testing.T) {
	for _, name := range []string{"SUPPORT.md", "CONTRIBUTING.md", "README.md"} {
		data, err := os.ReadFile(filepath.Join("..", "..", name))
		if err != nil {
			continue
		}
		for _, line := range strings.Split(string(data), "\n") {
			lower := strings.ToLower(line)
			if !strings.Contains(lower, "security") && !strings.Contains(lower, "vulnerab") {
				continue
			}
			// The line is about security. It must not be the line that tells
			// somebody to open an issue.
			if strings.Contains(lower, "open an issue") || strings.Contains(lower, "file an issue") {
				t.Errorf("%s: %q sends a vulnerability report to the public tracker",
					name, strings.TrimSpace(line))
			}
		}
	}
}
