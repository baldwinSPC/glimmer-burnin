package main

import (
	"encoding/json"
	"errors"
	"os"
	"path/filepath"
	"strconv"
	"testing"

	"github.com/baldwinSPC/glimmer-burnin/pkg/compare"
	"github.com/baldwinSPC/glimmer-burnin/pkg/contract"
)

func writeRunDir(t *testing.T, node string, bw float64, i int) string {
	t.Helper()
	dir := filepath.Join(t.TempDir(), node+strconv.Itoa(i))
	env := contract.Envelope{
		Version: contract.Version, Reason: contract.ReasonPhaseChanged, Phase: "Passed",
		DeliveryID:  "d" + strconv.Itoa(i),
		Run:         contract.RunRef{Namespace: "local", Name: node, UID: node + "-" + strconv.Itoa(i)},
		Fingerprint: map[string]string{node: "kernel=6.11.0 os=Ubuntu 24.04.3 LTS arch=arm64"},
		Results: []contract.TestResult{{Name: "mem", Kind: "memory-bw", Phase: "Passed",
			Metrics: map[string]string{"hostToDeviceBandwidthGBs": strconv.FormatFloat(bw, 'f', 2, 64)}}},
	}
	if err := writeJSON(filepath.Join(dir, "envelopes", "001-RunPhaseChanged.json"), env); err != nil {
		t.Fatal(err)
	}
	return dir
}

func TestPromoteThenCompareFromTheCommandLine(t *testing.T) {
	args := []string{"promote"}
	for i, v := range []float64{51.31, 51.44, 51.55, 51.32, 51.52} {
		args = append(args, "--results-dir", writeRunDir(t, "spark-043a", v, i))
	}
	out := filepath.Join(t.TempDir(), "b.json")
	if err := runBaseline(append(args, "--as", "gb10-h2d", "--out", out)); err != nil {
		t.Fatalf("promote: %v", err)
	}
	var b compare.Baseline
	data, _ := os.ReadFile(out)
	if err := json.Unmarshal(data, &b); err != nil || !b.Metrics["mem/hostToDeviceBandwidthGBs"].Enforceable {
		t.Fatalf("baseline = %+v, %v", b, err)
	}
	if err := runBaseline(append(args, "--as", "again", "--out", out)); err == nil {
		t.Error("a baseline file was overwritten")
	}

	ok := writeRunDir(t, "spark-043a", 51.40, 9)
	if err := runCompare([]string{"--baseline", out, "--results-dir", ok, "--enforce"}); err != nil {
		t.Errorf("a normal run failed an enforced compare: %v", err)
	}
	bad := writeRunDir(t, "spark-043a", 44.0, 10)
	err := runCompare([]string{"--baseline", out, "--results-dir", bad, "--enforce"})
	var ex *exitErr
	if err == nil || !errors.As(err, &ex) || ex.code != exitFail {
		t.Errorf("a 14%% drop under --enforce: err=%v, want exit %d", err, exitFail)
	}
	if err := runCompare([]string{"--baseline", out, "--results-dir", bad}); err != nil {
		t.Errorf("without --enforce a regression is reported, not failed: %v", err)
	}
}

func TestCompareRefusesASanitisedResult(t *testing.T) {
	dir := writeRunDir(t, "spark-043a", 51.4, 1)
	if err := os.WriteFile(filepath.Join(dir, sanitizedMarker), []byte("{}"), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := runCompare([]string{"--results-dir", dir, "--peer-results-dir", dir}); err == nil {
		t.Error("compared a sanitised result")
	}
}
