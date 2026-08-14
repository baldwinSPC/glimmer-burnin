package main

import (
	"encoding/json"
	"fmt"
	"sort"
	"strconv"
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
	// findings carries the same failures WITH whatever metadata DCGM attached.
	// failReasons stays as it was because it is what builds the human message;
	// this is what the verdict classifies against. Keyed by subtest name.
	findings map[string][]finding

	versionDepth int
	driverDepth  int

	// SawResultStructure records that the document contained a non-empty
	// tests / results / test_categories array, whether or not any status was
	// read out of it.
	//
	// It exists so a verdict can distinguish "DCGM produced no results" from
	// "we recorded no results", which are not the same statement and had the
	// same representation (#322). The first may legitimately become a Skip;
	// the second is a parser that did not understand the document, and a Skip
	// there takes a node out of scope on the strength of our own blindness.
	SawResultStructure bool
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

	// SkippedNames are subtests DCGM declined to run. Named rather than only
	// counted: "4 skipped" does not tell an operator whether the memory test or
	// the PCIe test is the one that never ran.
	SkippedNames []string
	// ExcusedNames are subtests DCGM failed whose findings were all
	// non-hardware, so they were counted as NotRun instead. See #304.
	ExcusedNames []string
	// ConfigFindings are node-setting problems: real, reportable, and not a
	// statement about the silicon.
	ConfigFindings []string
	// BlockingFinding is the first finding that kept a failed subtest from
	// being excused — the one that is actually about the hardware.
	BlockingFinding string
	// UnreadableFindings are fields DCGM could not read. Reported as
	// unmeasurable rather than as a value, which is the rule everywhere in this
	// project — a counter nobody read is not a counter that read zero.
	UnreadableFindings []string
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
	r := &diagResults{tests: map[string]status{}, findings: map[string][]finding{},
		versionDepth: -1, driverDepth: -1}
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
		// Did DCGM report subtest STRUCTURE, whether or not we could read a
		// status out of it? This is what lets the verdict tell "DCGM ran
		// nothing" apart from "DCGM ran things and we did not understand the
		// document" — two states that were indistinguishable, and only the
		// first of which may become a Skip. See #322.
		for _, container := range []string{"tests", "results", "test_categories"} {
			if items, ok := n[container].([]any); ok && len(items) > 0 {
				r.SawResultStructure = true
			}
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
		// findings only from results that actually FAILED, while failReasons
		// takes both. record() is called once per entity, so a subtest can hold
		// a Warn on one GPU and a Fail on another; collecting findings from the
		// Warn too let a MONITOR-severity observation on one GPU become the
		// blocking finding for another GPU's excusable failure. The excusal
		// decision is about why the subtest failed, so it reads the failures.
		//
		// The consequence is that a subtest whose every result is a Warn
		// collects no findings — which is honest, because nothing downstream
		// classifies them: Counts() handles statusWarn as a counter and
		// verdict() exits pass. Making a DCGM warning actionable is a separate
		// question, tracked rather than half-answered here.
		if st == statusFail {
			r.findings[name] = append(r.findings[name], findingsFrom(node)...)
		}
		r.failReasons = appendReasons(r.failReasons, name, node)
	}
}

// reasonKeys are the fields DCGM has used to explain a result. Collecting from
// all of them, rather than one, is the cheap way to survive a schema change:
// losing the prose costs a legible message, not a verdict.
var reasonKeys = []string{"info", "warning", "warnings", "error", "errors", "message", "error_message"}

// findingsFrom extracts findings WITH their DCGM metadata.
//
// A `warnings` array holds objects carrying error_severity / error_category /
// error_id beside the prose; `info` and friends hold bare strings. Both are
// collected, and only the first kind gets HasMeta — so classify() knows when it
// is reading DCGM's own answer and when it is reading a sentence.
//
// NOTHING HERE MAY DROP A FINDING, and that rule is load-bearing in a way it
// was not before this function existed. failReasons is prose for a human, so
// losing an entry cost legibility; findings is what the VERDICT is computed
// from, so losing an entry means excusedFindings never sees it, and a subtest
// whose remaining findings are all excusable is excused. Review found three
// separate shapes that were being dropped, each of which turned a Fail into a
// non-verdict on a document DCGM really emits. Hence: every element shape is
// handled, an unreadable message does not discard the metadata that came with
// it, and the collection is uncapped.
func findingsFrom(node map[string]any) []finding {
	var out []finding

	// The metadata-bearing keys. These carry objects in DCGM 3.1+, and bare
	// strings in older documents — and the runner is explicitly expected to
	// outlive the DCGM release it was built against, so both are handled.
	for _, key := range []string{"warnings", "errors"} {
		out = appendFindingEntry(out, node[key])
	}

	// The bare-string shapes. DCGM attaches no enums to these.
	for _, key := range []string{"info", "warning", "error", "message", "error_message"} {
		var found []string
		collectAllStrings(node[key], &found)
		for _, t := range found {
			if t = strings.TrimSpace(t); t != "" {
				out = append(out, finding{Text: t})
			}
		}
	}
	return out
}

// appendFindingEntry turns one warnings/errors value into findings, whatever
// shape it arrives in: an array, a single object, or a bare string.
func appendFindingEntry(out []finding, entry any) []finding {
	switch v := entry.(type) {
	case []any:
		for _, inner := range v {
			out = appendFindingEntry(out, inner)
		}
	case string:
		if s := strings.TrimSpace(v); s != "" {
			out = append(out, finding{Text: s})
		}
	case map[string]any:
		f := finding{Text: strings.TrimSpace(firstString(v,
			"warning", "error", "message", "error_message", "detail", "text", "info"))}
		sev, sevOK := intField(v, "error_severity")
		cat, catOK := intField(v, "error_category")
		id, _ := intField(v, "error_id")
		f.Severity, f.Category, f.ErrorID = sev, cat, id
		// Both, not either: a severity with no category cannot be composed
		// with the hardware-category veto, and half the classification is
		// worse than none because it looks authoritative.
		f.HasMeta = sevOK && catOK
		if f.Text == "" {
			// No prose under any key this runner knows. If DCGM still told us
			// what it was, that classification is the whole finding and must
			// survive — dropping it would discard precisely the ISOLATE /
			// HARDWARE_MEMORY answer this change exists to start trusting.
			if !f.HasMeta {
				return out
			}
			f.Text = fmt.Sprintf(
				"DCGM reported error_id %d (severity %d, category %d) with no message this runner could read",
				id, sev, cat)
		}
		out = append(out, f)
	}
	return out
}

func firstString(obj map[string]any, keys ...string) string {
	for _, k := range keys {
		if s, ok := obj[k].(string); ok && strings.TrimSpace(s) != "" {
			return s
		}
	}
	return ""
}

// intField reads a JSON number. encoding/json gives float64 for every number,
// so that case is the one that matters for a real dcgmi document.
//
// The string case is not defensive clutter. A quoted enum ("error_severity":
// "2") makes intField report "absent", which clears HasMeta and sends the
// finding to the TEXT path — and the text path has no hardware-category veto,
// so a finding DCGM marked ISOLATE / HARDWARE_MEMORY whose prose happens to
// read "Could not read the row remapping state" is excused as unreadable. The
// failure is silent and it is in the fail-open direction, which is the one
// direction this runner may never fail in.
func intField(obj map[string]any, key string) (int, bool) {
	switch v := obj[key].(type) {
	case float64:
		return int(v), true
	case int:
		return v, true
	case string:
		if n, err := strconv.Atoi(strings.TrimSpace(v)); err == nil {
			return n, true
		}
	}
	return 0, false
}

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

// collectAllStrings is collectStrings for the VERDICT rather than for a message.
//
// Two differences, and both exist because the consumer is different. It does not
// stop at 8: that cap keeps a human-readable reason short, and applying it here
// meant a subtest whose first eight findings were configuration noise and whose
// ninth was the hardware got excused — which is exactly the arithmetic of an
// 8-GPU node with persistence mode off. And it recurses into nested maps in
// sorted key order, so a finding nested one level deeper is still seen, and the
// finding reported as the blocking one does not vary between runs on a
// byte-identical document.
func collectAllStrings(node any, out *[]string) {
	switch n := node.(type) {
	case string:
		*out = append(*out, n)
	case []any:
		for _, v := range n {
			collectAllStrings(v, out)
		}
	case map[string]any:
		keys := make([]string, 0, len(n))
		for k := range n {
			keys = append(keys, k)
		}
		sort.Strings(keys)
		for _, k := range keys {
			collectAllStrings(n[k], out)
		}
	}
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
			// A failure whose every finding is a node setting or a failed read
			// is not a verdict about the part — issue #304. It becomes NOT RUN,
			// which is already this runner's word for "DCGM did not establish a
			// result here", and which run() reports as Error: unjudged, rather
			// than a hardware Fail that is never retried.
			//
			// Reusing NotRun rather than inventing a fourth outcome keeps one
			// meaning per phase. The findings themselves are not discarded —
			// they are carried out for the report, because a node whose
			// persistence mode is off should say so.
			cfg, unread, blocking, excused := excusedFindings(r.findings[name])
			if !excused && blocking != "" && c.BlockingFinding == "" {
				c.BlockingFinding = blocking
			}
			if excused {
				c.NotRun++
				c.ExcusedNames = append(c.ExcusedNames, name)
				c.ConfigFindings = append(c.ConfigFindings, cfg...)
				c.UnreadableFindings = append(c.UnreadableFindings, unread...)
				continue
			}
			c.Failed++
			c.FailedNames = append(c.FailedNames, name)
		case statusWarn:
			c.Warned++
		case statusSkip:
			c.Skipped++
			c.SkippedNames = append(c.SkippedNames, name)
		default:
			c.NotRun++
		}
	}
	// Excused subtests COUNT AS EXECUTED (#304). DCGM ran them and produced
	// findings; what changed is that the findings were not about the part. If
	// they were left out, a run whose only subtest was excused would reach
	// Executed == 0 and be reported as SKIPPED — "not applicable to this
	// hardware" — which is a worse claim than the Fail this excusal exists to
	// avoid, because it looks benign while verifying nothing. verdict_test.go
	// catches exactly that.
	c.Executed = c.Passed + c.Failed + c.Warned + len(c.ExcusedNames)
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
