package main

import (
	"encoding/json"
	"fmt"
	"sort"
	"strings"
)

// status is a DCGM subtest outcome, normalised.
type status int

// Ordered by severity. Aggregating a test's per-GPU results takes the MAX, so
// the ordering is the policy: one failing GPU fails the test, and one GPU whose
// result never arrived leaves the test unjudged rather than passed.
const (
	statusSkip status = iota
	statusPass
	statusNotRun
	statusWarn
	statusFail
)

func (s status) String() string {
	switch s {
	case statusSkip:
		return "skip"
	case statusPass:
		return "pass"
	case statusNotRun:
		return "not run"
	case statusWarn:
		return "warn"
	case statusFail:
		return "fail"
	}
	return "unknown"
}

// parseStatus maps DCGM's spellings onto ours.
//
// An unrecognised status becomes statusNotRun, never statusPass. A word this
// runner has not seen before is not evidence that hardware is healthy, and a
// new DCGM release inventing a spelling must not silently turn into a pass.
func parseStatus(raw string) status {
	switch strings.ToLower(strings.Join(strings.Fields(raw), " ")) {
	case "pass", "passed", "ok":
		return statusPass
	case "fail", "failed", "error":
		return statusFail
	case "warn", "warning":
		return statusWarn
	case "skip", "skipped", "not applicable", "n/a":
		return statusSkip
	default:
		// "not run", "notrun", "incomplete", and anything unrecognised.
		return statusNotRun
	}
}

// diagResults is what a `dcgmi diag -j` document told us.
//
// The document is walked structurally rather than unmarshalled into a fixed
// schema on purpose. DCGM has changed its JSON shape between major versions
// (3.x nests everything under "DCGM GPU Diagnostic" > "test_categories"; 4.x
// hoists "tests" to the top and describes results by entity), and a runner
// image pinned into a fleet's readiness gate outlives the DCGM release it was
// built against. A struct that matched only one shape would report "no results"
// on the other — which, being indistinguishable from "the diagnostic produced
// nothing", is the shape of a false verdict.
type diagResults struct {
	Version       string
	DriverVersion string

	// tests is aggregated by test name: DCGM reports one result per GPU, and
	// acceptance is a statement about the node.
	tests map[string]status
	order []string

	// skipReasons and failReasons carry whatever prose DCGM attached, so a
	// verdict can say WHY rather than only that it happened.
	skipReasons []string
	failReasons []string

	versionDepth int
	driverDepth  int
}

type diagCounts struct {
	Total    int
	Executed int // pass + warn + fail: the subtests that actually judged hardware
	Passed   int
	Failed   int
	Warned   int
	NotRun   int
	Skipped  int

	FailedNames []string
}

// parseDiagJSON extracts the diagnostic document from dcgmi's stdout.
func parseDiagJSON(stdout string) (*diagResults, error) {
	body, err := extractJSON(stdout)
	if err != nil {
		return nil, err
	}
	var root any
	if err := json.Unmarshal([]byte(body), &root); err != nil {
		return nil, fmt.Errorf("dcgmi output is not valid JSON: %w", err)
	}
	r := &diagResults{tests: map[string]status{}, versionDepth: -1, driverDepth: -1}
	r.walk(root, "", 0)
	return r, nil
}

// extractJSON finds the JSON document inside dcgmi's stdout. dcgmi prints a
// banner and, on some paths, warnings before the document; taking the outermost
// braces is more durable than assuming the document starts at byte zero.
func extractJSON(s string) (string, error) {
	start := strings.Index(s, "{")
	end := strings.LastIndex(s, "}")
	if start < 0 || end <= start {
		return "", fmt.Errorf("no JSON document in dcgmi output")
	}
	return s[start : end+1], nil
}

func (r *diagResults) walk(node any, inheritedName string, depth int) {
	switch n := node.(type) {
	case map[string]any:
		if name := stringField(n, "name"); name != "" {
			inheritedName = name
		}
		// Shallowest wins: Go map iteration is unordered, so "the first one we
		// happened to see" would not be reproducible across runs.
		if v := stringField(n, "version"); v != "" && (r.versionDepth < 0 || depth < r.versionDepth) {
			r.Version, r.versionDepth = v, depth
		}
		if v := driverVersionField(n); v != "" && (r.driverDepth < 0 || depth < r.driverDepth) {
			r.DriverVersion, r.driverDepth = v, depth
		}
		if raw, ok := n["status"].(string); ok {
			r.record(inheritedName, parseStatus(raw), n)
		}
		for _, v := range n {
			r.walk(v, inheritedName, depth+1)
		}
	case []any:
		for _, v := range n {
			r.walk(v, inheritedName, depth)
		}
	}
}

func (r *diagResults) record(name string, st status, node map[string]any) {
	if name == "" {
		name = "unnamed"
	}
	prev, seen := r.tests[name]
	if !seen {
		r.order = append(r.order, name)
	}
	if !seen || st > prev {
		r.tests[name] = st
	}
	switch st {
	case statusSkip:
		r.skipReasons = appendReasons(r.skipReasons, name, node)
	case statusFail, statusWarn:
		r.failReasons = appendReasons(r.failReasons, name, node)
	}
}

// reasonKeys are the fields DCGM has used to explain a result. Collecting from
// all of them, rather than one, is the cheap way to survive a schema change:
// losing the prose costs a legible message, not a verdict.
var reasonKeys = []string{"info", "warning", "warnings", "error", "errors", "message", "error_message"}

func appendReasons(dst []string, name string, node map[string]any) []string {
	var found []string
	for _, key := range reasonKeys {
		collectStrings(node[key], &found)
	}
	for _, f := range found {
		f = strings.TrimSpace(f)
		if f == "" {
			continue
		}
		dst = append(dst, name+": "+f)
		if len(dst) >= 8 {
			return dst
		}
	}
	return dst
}

func collectStrings(node any, out *[]string) {
	if len(*out) >= 8 {
		return
	}
	switch n := node.(type) {
	case string:
		*out = append(*out, n)
	case []any:
		for _, v := range n {
			collectStrings(v, out)
		}
	case map[string]any:
		for _, v := range n {
			if s, ok := v.(string); ok {
				*out = append(*out, s)
			}
		}
	}
}

func stringField(m map[string]any, key string) string {
	if v, ok := m[key].(string); ok {
		return strings.TrimSpace(v)
	}
	return ""
}

// driverVersionNames are the spellings DCGM has used for the driver version,
// reduced to letters only.
//
// Matching is done on the reduced form because DCGM's own spelling is not
// stable and is not even self-consistent: 4.2.3 emits the human-readable
// "Driver Version Detected" in its metadata object while other paths use the
// snake_case "driver_version". An exact-match list gets one of them, silently
// drops driverVersion from the report on the other, and nothing about the
// output says a field went missing.
// Most specific first; the map is scanned in sorted key order within each, so
// the answer does not depend on Go's randomised map iteration.
var driverVersionNames = []string{"driverversiondetected", "driverversion"}

// driverVersionField finds the driver version under any of those spellings.
func driverVersionField(m map[string]any) string {
	keys := make([]string, 0, len(m))
	for k := range m {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	for _, want := range driverVersionNames {
		for _, k := range keys {
			if reduceKey(k) != want {
				continue
			}
			if v, ok := m[k].(string); ok {
				if v = strings.TrimSpace(v); v != "" {
					return v
				}
			}
		}
	}
	return ""
}

// reduceKey lowercases a JSON key and drops everything that is not a letter, so
// "Driver Version Detected", "driver_version_detected" and "DriverVersion" all
// reduce to the same token.
func reduceKey(s string) string {
	var b strings.Builder
	for _, r := range strings.ToLower(s) {
		if r >= 'a' && r <= 'z' {
			b.WriteRune(r)
		}
	}
	return b.String()
}

// Counts tallies the aggregated per-test statuses.
func (r *diagResults) Counts() diagCounts {
	var c diagCounts
	for _, name := range r.order {
		c.Total++
		switch r.tests[name] {
		case statusPass:
			c.Passed++
		case statusFail:
			c.Failed++
			c.FailedNames = append(c.FailedNames, name)
		case statusWarn:
			c.Warned++
		case statusSkip:
			c.Skipped++
		default:
			c.NotRun++
		}
	}
	c.Executed = c.Passed + c.Failed + c.Warned
	return c
}

// unsupportedSignatures are the phrases DCGM uses when it will not run against
// a part at all.
//
// They are matched ONLY against a run that produced no subtest results (see
// run()). That guard is the whole reason this is safe: "not supported" also
// appears in per-test prose on perfectly healthy hardware ("NVLink is not
// supported on this GPU"), and matching it there would turn a real pass into a
// skip and quietly excuse a node from acceptance.
var unsupportedSignatures = []string{
	"unsupported gpu",
	"is not supported by dcgm",
	"not supported by dcgm",
	"no supported gpus",
	"no supported devices",
	"unable to run diagnostic on unsupported",
	"gpu is not supported",
	"does not support the diagnostic",
	"unsupported chip",
}

// unsupportedSignature returns the matched phrase, with a little surrounding
// context, or "" if the text does not look like an unsupported-part refusal.
func unsupportedSignature(text string) string {
	lower := strings.ToLower(text)
	for _, sig := range unsupportedSignatures {
		if i := strings.Index(lower, sig); i >= 0 {
			return excerpt(text, i)
		}
	}
	return ""
}

// excerpt returns the line the match landed on, so the reason names DCGM's own
// words rather than ours.
func excerpt(text string, index int) string {
	start := strings.LastIndexByte(text[:index], '\n') + 1
	end := strings.IndexByte(text[index:], '\n')
	if end < 0 {
		end = len(text)
	} else {
		end += index
	}
	return strings.TrimSpace(text[start:end])
}
