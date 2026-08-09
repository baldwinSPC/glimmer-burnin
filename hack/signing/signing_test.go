package signing

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// Everything this project publishes is signed — issue #175.
//
// A runner image is executed by a node's readiness gate, on hardware a site is
// deciding whether to accept. An unsigned image in that position decides
// acceptance. These guards are about the publish workflows, which run rarely and
// by hand — exactly the code least likely to be noticed when it breaks.

func read(t *testing.T, rel string) string {
	t.Helper()
	data, err := os.ReadFile(filepath.Join("..", "..", rel))
	if err != nil {
		t.Fatalf("%s: %v", rel, err)
	}
	return string(data)
}

var publishWorkflows = []string{
	".github/workflows/publish-operator.yml",
	".github/workflows/publish-runner.yml",
}

// Both workflows sign, and both can.
//
// `id-token: write` is the half that is easy to lose: a permissions block edited
// for another reason drops it, and cosign then fails at publish time — on the
// rarely-run workflow, by hand, with a release half done.
func TestBothPublishWorkflowsSign(t *testing.T) {
	for _, wf := range publishWorkflows {
		body := read(t, wf)
		if !strings.Contains(body, "id-token: write") {
			t.Errorf("%s does not request id-token: write, so keyless signing cannot "+
				"work — and it fails at publish time, by hand, with a release half done", wf)
		}
		if !strings.Contains(body, "cosign-installer") {
			t.Errorf("%s does not install cosign", wf)
		}
		if !strings.Contains(body, "cosign sign") {
			t.Errorf("%s signs nothing", wf)
		}
	}
}

// The signature covers the DIGEST, never the tag.
//
// A signature over a tag would still verify after the tag moved, which is
// exactly the situation a signature exists to detect. Published tags are
// immutable by policy here — but a policy is not a cryptographic guarantee, and
// the signature is supposed to be the guarantee.
func TestSignaturesCoverDigestsNotTags(t *testing.T) {
	for _, wf := range publishWorkflows {
		for _, line := range strings.Split(read(t, wf), "\n") {
			if !strings.Contains(line, "cosign sign") {
				continue
			}
			if !strings.Contains(line, "@") {
				t.Errorf("%s: %q signs a reference with no digest. A signature over a "+
					"mutable tag still verifies after the tag moves.", wf, strings.TrimSpace(line))
			}
			// A bare `${REF}` with a tag in it is the mistake this catches.
			if strings.Contains(line, "\"${REF}\"") {
				t.Errorf("%s: %q signs the tag reference directly", wf, strings.TrimSpace(line))
			}
		}
	}
}

// The runner workflow signs the manifest list AND both children.
//
// The children are separately pullable by digest: a mirror, a registry proxy, or
// anything resolving a platform before it pulls will reference one directly, and
// a signature on the list alone says nothing about those.
func TestTheRunnerWorkflowSignsTheChildrenToo(t *testing.T) {
	body := read(t, ".github/workflows/publish-runner.yml")
	if strings.Count(body, "cosign sign") < 2 {
		t.Fatalf("the runner workflow has %d cosign sign invocations; it needs one for "+
			"the manifest list and one covering the per-architecture children, which are "+
			"separately pullable", strings.Count(body, "cosign sign"))
	}
	if !strings.Contains(body, "/tmp/digests") {
		t.Error("the children are not signed from the digest files the build jobs left")
	}
	// The list digest is READ BACK from the registry, not assumed: imagetools
	// create does not report it, and computing it locally would sign what we
	// think we pushed rather than what is there.
	//
	// Matched on the ASSIGNMENT, not on "imagetools inspect" alone — the
	// manifest-assert step further down uses that too, so the looser check
	// passed with the signing step's resolution deleted entirely. A second
	// occurrence of a string is a second route past the test.
	if !strings.Contains(body, "LIST_DIGEST=$(docker buildx imagetools inspect") {
		t.Error("the manifest list digest is not resolved from the registry before " +
			"signing; signing a locally-computed digest signs what we think we pushed " +
			"rather than what is there")
	}
}

// Signing refuses to proceed on an empty digest.
//
// `cosign sign "repo@"` is not obviously an error to a shell, and a publish that
// silently signed nothing would leave a tag everybody believes is signed.
func TestSigningRefusesAnEmptyDigest(t *testing.T) {
	for _, wf := range publishWorkflows {
		body := read(t, wf)
		if !strings.Contains(body, "refusing to sign nothing") {
			t.Errorf("%s does not guard against an empty digest. A publish that signed "+
				"nothing leaves a tag everybody believes is signed.", wf)
		}
	}
}

// The verification command is documented, and documented CORRECTLY.
//
// An unverifiable signature is worth very little. A verification command that
// does not pin the issuer is worth almost nothing: without
// --certificate-oidc-issuer, a signature from any OIDC provider satisfies it.
func TestVerificationIsDocumentedAndPinsTheIdentity(t *testing.T) {
	doc := read(t, "docs/verifying-images.md")

	for what, must := range map[string]string{
		"the verify command":    "cosign verify",
		"the identity":          "--certificate-identity-regexp",
		"the issuer":            "--certificate-oidc-issuer",
		"GitHub's issuer URL":   "token.actions.githubusercontent.com",
		"this repository":       "baldwinSPC/glimmer-burnin",
		"what is signed":        "Digests, never tags",
		"the children":          "separately pullable",
		"what to do on failure": "Do not deploy",
	} {
		if !strings.Contains(doc, must) {
			t.Errorf("docs/verifying-images.md does not cover %s (looked for %q)", what, must)
		}
	}

	// The documented identity must actually match the workflow filenames it
	// claims to accept, or a site following this document gates on a pattern
	// that rejects every real image.
	for _, wf := range []string{"publish-operator.yml", "publish-runner.yml"} {
		if !strings.Contains(doc, "publish-") {
			t.Fatalf("the documented identity regexp cannot match %s", wf)
		}
	}
}
