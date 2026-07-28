// Package runner parses what a burn-in runner container reports.
//
// A runner's contract is deliberately small: an exit code, and key=value pairs
// on stdout. That is all a runner has to implement in any language, which is
// what keeps the vendor seam at the image boundary rather than in the
// reconciler.
//
// Like pkg/verdict, this is public and free of Kubernetes types. Glimmer's
// pre-Kubernetes burn-in path runs the SAME runner images, and if the two
// dispatchers derived different metrics from identical output they would reach
// different verdicts about the same hardware. One brain, two dispatchers.
package runner

import (
	"strconv"
	"strings"
	"unicode"

	"github.com/baldwinSPC/glimmer-burnin/pkg/contract"
)

// Verdict is what a runner's exit code means.
type Verdict string

const (
	// VerdictPass is exit 0: the test ran and the hardware passed.
	VerdictPass Verdict = "Pass"
	// VerdictFail is exit 1: the test ran and the hardware failed.
	VerdictFail Verdict = "Fail"
	// VerdictSkip is exit 2: the test does not apply to this hardware.
	//
	// Skip must stay distinct from Fail. A node that cannot run a test has not
	// failed it, and collapsing the two is the false-negative that made healthy
	// Sparks look broken.
	VerdictSkip Verdict = "Skip"
	// VerdictError is any other exit code: the runner itself malfunctioned, so
	// the hardware is unjudged. Distinct from Fail, which is a real verdict.
	VerdictError Verdict = "Error"
)

// VerdictFor maps an exit code to its meaning.
func VerdictFor(exitCode int) Verdict {
	switch exitCode {
	case 0:
		return VerdictPass
	case 1:
		return VerdictFail
	case 2:
		return VerdictSkip
	default:
		return VerdictError
	}
}

// Result is one runner execution, parsed.
type Result struct {
	Verdict  Verdict
	ExitCode int
	// Metrics are keyed by CANONICAL name (see pkg/contract), not by whatever
	// the runner printed.
	Metrics map[string]string
	// Message is the last line that was not a key=value pair — usually a
	// marker or an error string. Kept because a failing runner's most useful
	// output is rarely a metric.
	Message string
	// InvalidNames lists canonical names that failed contract validation. They
	// are reported rather than silently dropped: a runner emitting an unusable
	// name should be visible, not invisible.
	InvalidNames []string
}

// aliases maps a runner's own key to the canonical metric name, for the cases
// generic normalisation cannot reach. Keyed by TestKind.
//
// Generic snake_case → lowerCamelCase handles nearly everything
// ("nonfinite_count" → "nonfiniteCount", "elapsed_ms" → "elapsedMs"). An entry
// is only needed where the runner's key is not merely a different spelling of
// the canonical name — "tflops" is a bare unit that names no measurand, so it
// cannot be derived, only mapped.
var aliases = map[string]map[string]string{
	"compute-smoke": {
		"tflops": "throughputTflops",
	},
}

// Parse turns a runner's stdout and exit code into a Result.
//
// kind selects the alias table. An unknown or custom kind still gets the
// generic key=value scan — that scan IS the runner contract, and a custom
// runner honouring it should not be punished — but gets no kind-specific
// mapping, since only the kind's author knows what its keys mean.
func Parse(kind, stdout string, exitCode int) Result {
	res := Result{
		Verdict:  VerdictFor(exitCode),
		ExitCode: exitCode,
		Metrics:  map[string]string{},
	}
	kindAliases := aliases[kind]

	for _, line := range strings.Split(stdout, "\n") {
		line = strings.TrimSpace(line)
		if line == "" {
			continue
		}
		// Split on the FIRST "=" so a value may itself contain one.
		eq := strings.Index(line, "=")
		if eq <= 0 {
			res.Message = line
			continue
		}
		rawKey := strings.TrimSpace(line[:eq])
		value := strings.TrimSpace(line[eq+1:])
		if rawKey == "" || strings.ContainsAny(rawKey, " \t") {
			// "some prose = with an equals" is not a metric.
			res.Message = line
			continue
		}

		name := canonicalName(rawKey, kindAliases)
		if err := contract.ValidateMetricName(name); err != nil {
			res.InvalidNames = append(res.InvalidNames, name)
			continue
		}
		// Last occurrence wins: runners may report progressively, and the final
		// value is the settled one.
		res.Metrics[name] = value
	}
	return res
}

// canonicalName maps a runner's key to its canonical form.
func canonicalName(rawKey string, kindAliases map[string]string) string {
	if canonical, ok := kindAliases[rawKey]; ok {
		return canonical
	}
	return snakeToLowerCamel(rawKey)
}

// snakeToLowerCamel converts "max_rel_error" to "maxRelError". A key that is
// already lowerCamelCase passes through unchanged, so a runner may emit either
// spelling.
func snakeToLowerCamel(s string) string {
	if !strings.ContainsAny(s, "_-") {
		return s
	}
	parts := strings.FieldsFunc(s, func(r rune) bool { return r == '_' || r == '-' })
	if len(parts) == 0 {
		return s
	}
	var b strings.Builder
	b.WriteString(strings.ToLower(parts[0]))
	for _, p := range parts[1:] {
		if p == "" {
			continue
		}
		r := []rune(strings.ToLower(p))
		r[0] = unicode.ToUpper(r[0])
		b.WriteString(string(r))
	}
	return b.String()
}

// Numeric reports a metric's value as a float, which is what a threshold is
// evaluated against. Non-numeric metrics (a GPU's name, say) are legitimate
// evidence but are not comparable.
func (r Result) Numeric(name string) (float64, bool) {
	raw, ok := r.Metrics[name]
	if !ok {
		return 0, false
	}
	f, err := strconv.ParseFloat(raw, 64)
	if err != nil {
		return 0, false
	}
	return f, true
}
