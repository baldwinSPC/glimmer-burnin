package chart

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// The chart must install what config/ installs — issue #174.
//
// Two install paths for one operator is two things that can drift, and the one
// that drifts is the one nobody runs in CI. config/ is applied by the e2e job on
// every PR; the chart is not. So the chart's copy of anything is checked against
// config/'s, here, rather than trusted.

func read(t *testing.T, rel string) string {
	t.Helper()
	data, err := os.ReadFile(filepath.Join("..", "..", rel))
	if err != nil {
		t.Fatalf("%s: %v", rel, err)
	}
	return string(data)
}

// The chart's RBAC rules are byte-identical to the generated ClusterRole's.
//
// The ClusterRole is generated from kubebuilder markers by `make manifests`, and
// CI already blocks on drift between the markers and config/rbac. This closes
// the third side: a chart whose rules lag behind grants too little, and the
// operator then fails at RUNTIME as permission errors that read like bugs —
// which is a far more expensive way to find out than a failing test.
func TestTheChartRBACMatchesTheGeneratedRole(t *testing.T) {
	generated := read(t, "config/rbac/role.yaml")
	i := strings.Index(generated, "rules:")
	if i < 0 {
		t.Fatal("config/rbac/role.yaml has no rules: block, so this guard is reading " +
			"the wrong file and would pass whatever the chart contained")
	}
	want := strings.TrimSpace(generated[i+len("rules:"):])
	got := strings.TrimSpace(read(t, "deploy/charts/glimmer-burnin/rbac/role-rules.yaml"))

	if got != want {
		t.Errorf("the chart's RBAC rules differ from the generated ClusterRole.\n\n"+
			"Run `make manifests`, then copy config/rbac/role.yaml's rules: block into\n"+
			"deploy/charts/glimmer-burnin/rbac/role-rules.yaml.\n\n"+
			"chart has %d bytes, generated has %d", len(got), len(want))
	}
	if len(want) < 200 {
		t.Fatalf("the generated rules are only %d bytes; this comparison is not "+
			"exercising anything", len(want))
	}
}

// The chart ships the CRDs, and all of them.
//
// A chart missing one installs an operator whose reconciler watches a kind the
// apiserver does not know, which surfaces as the operator doing nothing at all
// for that kind — with no error anybody sees.
func TestTheChartShipsEveryCRD(t *testing.T) {
	shipped := map[string]bool{}
	entries, err := os.ReadDir(filepath.Join("..", "..", "deploy/charts/glimmer-burnin/crds"))
	if err != nil {
		t.Fatal(err)
	}
	for _, e := range entries {
		shipped[e.Name()] = true
	}

	generated, err := os.ReadDir(filepath.Join("..", "..", "config/crd"))
	if err != nil {
		t.Fatal(err)
	}
	var count int
	for _, e := range generated {
		if !strings.HasSuffix(e.Name(), ".yaml") {
			continue
		}
		count++
		if !shipped[e.Name()] {
			t.Errorf("config/crd has %s and the chart does not. The operator would watch "+
				"a kind the apiserver does not know, and do nothing at all for it with no "+
				"error anybody sees.", e.Name())
		}
	}
	if count < 5 {
		t.Fatalf("found only %d generated CRDs; this guard is reading the wrong "+
			"directory", count)
	}
	// And the CRDs themselves must match, not merely be present.
	for _, e := range generated {
		if !strings.HasSuffix(e.Name(), ".yaml") {
			continue
		}
		a := read(t, filepath.Join("config/crd", e.Name()))
		b := read(t, filepath.Join("deploy/charts/glimmer-burnin/crds", e.Name()))
		if a != b {
			t.Errorf("%s differs between config/crd and the chart. A stale CRD in the "+
				"chart silently drops the fields it does not know about — the apiserver "+
				"accepts the object and the run then executes a spec nobody wrote.", e.Name())
		}
	}
}

// The chart pins the same image the shipped manifests do.
func TestTheChartAndTheManifestsAgreeOnTheImage(t *testing.T) {
	manifest := read(t, "config/manager/manager.yaml")
	const prefix = "image: ghcr.io/baldwinspc/glimmer-burnin:"
	i := strings.Index(manifest, prefix)
	if i < 0 {
		t.Fatal("config/manager/manager.yaml does not pin an image in the expected " +
			"form; this guard is not reading what it thinks it is")
	}
	tag := strings.TrimSpace(manifest[i+len(prefix) : i+len(prefix)+20])
	tag = strings.SplitN(tag, "\n", 2)[0]

	chart := read(t, "deploy/charts/glimmer-burnin/Chart.yaml")
	want := "appVersion: \"" + strings.TrimPrefix(tag, "v") + "\""
	if !strings.Contains(chart, want) {
		t.Errorf("config/manager pins %s but Chart.yaml does not carry the matching "+
			"appVersion (%s). \"Which chart\" and \"which operator\" must not be two "+
			"questions.", tag, want)
	}
}

// The chart declares no cert-manager dependency, and must not acquire one.
//
// The operator has no webhooks. That is one of the few things that makes it
// cheap to adopt, and a dependency added without noticing would take it away.
func TestTheChartNeedsNoCertManager(t *testing.T) {
	// Checked line by line, skipping COMMENTS. A first version exempted any file
	// containing the phrase "no cert-manager" — which both of these say, in the
	// comment explaining why there is no dependency — so adding a real
	// `certManager: enabled: true` block changed nothing. An exemption broad
	// enough to cover the explanation is broad enough to cover the thing it
	// explains.
	for _, f := range []string{
		"deploy/charts/glimmer-burnin/Chart.yaml",
		"deploy/charts/glimmer-burnin/values.yaml",
	} {
		for _, line := range strings.Split(read(t, f), "\n") {
			trimmed := strings.TrimSpace(line)
			if trimmed == "" || strings.HasPrefix(trimmed, "#") {
				continue
			}
			if strings.Contains(strings.ToLower(trimmed), "cert-manager") ||
				strings.Contains(strings.ToLower(trimmed), "certmanager") {
				t.Errorf("%s declares cert-manager in %q. The operator has no webhooks, "+
					"and not needing cert-manager is a real adoption advantage.", f, trimmed)
			}
		}
	}
	// And nothing templates a webhook configuration.
	entries, err := os.ReadDir(filepath.Join("..", "..", "deploy/charts/glimmer-burnin/templates"))
	if err != nil {
		t.Fatal(err)
	}
	for _, e := range entries {
		body := read(t, filepath.Join("deploy/charts/glimmer-burnin/templates", e.Name()))
		if strings.Contains(body, "WebhookConfiguration") {
			t.Errorf("%s templates a webhook configuration", e.Name())
		}
	}
}

// The Makefile offers the non-Helm path the issue asks for.
func TestMakeDeployExists(t *testing.T) {
	mk := read(t, "Makefile")
	for _, target := range []string{"deploy:", "undeploy:", "helm-lint:"} {
		if !strings.Contains(mk, target) {
			t.Errorf("the Makefile has no %s target", target)
		}
	}
	// undeploy must NOT delete the CRDs: that deletes every BurnInRun in the
	// cluster, and a verdict is worth more than a tidy uninstall.
	i := strings.Index(mk, "undeploy:")
	j := strings.Index(mk[i:], "\n.PHONY")
	if j < 0 {
		j = len(mk) - i
	}
	if strings.Contains(mk[i:i+j], "config/crd") {
		t.Error("make undeploy deletes the CRDs, which deletes every BurnInRun in the " +
			"cluster along with them. A verdict is worth more than a tidy uninstall.")
	}
}
