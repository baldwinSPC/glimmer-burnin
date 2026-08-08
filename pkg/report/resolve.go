package report

import (
	"errors"
	"fmt"
	"regexp"
	"sort"
	"strings"
	"time"

	"github.com/baldwinSPC/glimmer-burnin/pkg/contract"
)

// Resolved is the single view of one run that renderers consume.
//
// It exists so that the precedence rules below are applied ONCE. Three renderers
// deriving "which delivery was authoritative" independently would eventually
// disagree, and the first anyone would know of it is two documents about one run
// that do not match — in the artifact most likely to be forwarded outside the
// building.
type Resolved struct {
	Run       RunView
	Results   []ResultView
	Nodes     []NodeInfo
	Artifacts []Artifact
	Meta      Meta

	// Warnings record what this view could NOT establish: a run that had not
	// finished, results known only from a partial delivery, hardware detail that
	// was never supplied. Renderers are expected to surface these rather than
	// drop them — the whole point of the package is that a document says what it
	// does not know.
	Warnings []string
}

// RunView identifies the run and carries its verdict.
type RunView struct {
	Namespace string
	Name      string
	UID       string
	Profile   string
	Cluster   *contract.ClusterRef

	// Phase is the run's phase as of the authoritative delivery.
	Phase string
	// Terminal says whether that phase is final. A report rendered from a
	// still-running run is legitimate — a soak reporting at hour six is useful —
	// but it is not a verdict, and a renderer must not present it as one.
	Terminal bool
	// Baseline says the run MEASURED rather than certified. A renderer must make
	// this visible: a thresholdless sweep that reads as an acceptance is exactly
	// the fail-open the flag was added to close.
	Baseline bool

	Fingerprint map[string]string
	Summary     contract.Summary

	// StartedAt and FinishedAt are derived from the results, not from delivery
	// timestamps — when the work happened, not when someone was told about it.
	StartedAt  *time.Time
	FinishedAt *time.Time

	// SentAt is the authoritative delivery's timestamp, kept so a reader can
	// tell how current the document is.
	SentAt time.Time
	// DeliveryID of the authoritative delivery, for tracing a document back to
	// the delivery that produced it.
	DeliveryID string
}

// ResultView is one test's outcome, plus what a renderer needs to present it.
type ResultView struct {
	contract.TestResult

	// Link is true when this result is about a relationship between machines
	// rather than about one of them — Pair and Group scope.
	//
	// It is the single most misrenderable thing in the whole model. A pair
	// verdict attributed to one endpoint sends an engineer to replace the wrong
	// part, and NVVS has no vocabulary for a link at all, so the renderer that
	// most needs this flag is the one whose schema cannot express it.
	Link bool

	// Partial marks a result taken from a non-authoritative delivery because the
	// authoritative one did not carry it.
	Partial bool
}

// ErrNoEnvelopes is returned when there is nothing to render.
var ErrNoEnvelopes = errors.New("report: no envelopes")

// Resolve reduces a delivery stream to one view.
//
// The precedence rules, in order:
//
//  1. Every envelope must be a version this package understands, and must
//     describe the same run. Both are refusals rather than best-effort merges:
//     guessing at an unrecognised schema, or rendering two runs as one document,
//     produces a confident document that is wrong.
//  2. The authoritative delivery is the latest terminal RunPhaseChanged. Failing
//     that, the latest delivery of any kind — with a warning, because the run had
//     not reached a verdict.
//  3. Results come from the authoritative delivery. A test it does not carry, but
//     an earlier TestCompleted does, is included and marked Partial.
func Resolve(in Input) (Resolved, error) {
	if len(in.Envelopes) == 0 {
		return Resolved{}, ErrNoEnvelopes
	}

	var warnings []string

	// Rule 1: one run, understood versions.
	uid := ""
	for i, e := range in.Envelopes {
		if e == nil {
			return Resolved{}, fmt.Errorf("report: envelope %d is nil", i)
		}
		if e.Version != contract.Version {
			return Resolved{}, fmt.Errorf(
				"report: envelope %d has version %q, want %q — refusing to guess at an unrecognised schema",
				i, e.Version, contract.Version)
		}
		if uid == "" {
			uid = e.Run.UID
		} else if e.Run.UID != uid {
			return Resolved{}, fmt.Errorf(
				"report: envelopes describe different runs (%q and %q) — a document covering two runs would be a false record",
				uid, e.Run.UID)
		}
	}

	// Rule 2: pick the authoritative delivery.
	ordered := make([]*contract.Envelope, len(in.Envelopes))
	copy(ordered, in.Envelopes)
	sort.SliceStable(ordered, func(i, j int) bool {
		if !ordered[i].SentAt.Equal(ordered[j].SentAt) {
			return ordered[i].SentAt.Before(ordered[j].SentAt)
		}
		// A deterministic tie-break so a report is byte-stable when two
		// deliveries share a timestamp.
		return ordered[i].DeliveryID < ordered[j].DeliveryID
	})

	var auth *contract.Envelope
	for i := len(ordered) - 1; i >= 0; i-- {
		if ordered[i].Reason == contract.ReasonPhaseChanged && IsTerminal(ordered[i].Phase) {
			auth = ordered[i]
			break
		}
	}
	if auth == nil {
		auth = ordered[len(ordered)-1]
		warnings = append(warnings, fmt.Sprintf(
			"no terminal delivery: this run was in phase %q when the report was generated, so it records progress rather than a verdict",
			auth.Phase))
	}

	// Rule 3: results from the authoritative delivery, topped up from earlier
	// ones. Keyed on name plus nodes because one test name legitimately appears
	// once per node at Node scope.
	type key struct{ name, nodes string }
	seen := map[key]bool{}
	results := make([]ResultView, 0, len(auth.Results))
	for _, r := range auth.Results {
		k := key{r.Name, strings.Join(r.Nodes, "\x00")}
		seen[k] = true
		results = append(results, ResultView{TestResult: r, Link: isLink(r)})
	}
	for _, e := range ordered {
		if e == auth {
			continue
		}
		for _, r := range e.Results {
			k := key{r.Name, strings.Join(r.Nodes, "\x00")}
			if seen[k] {
				continue
			}
			seen[k] = true
			results = append(results, ResultView{TestResult: r, Link: isLink(r), Partial: true})
			warnings = append(warnings, fmt.Sprintf(
				"test %q is reported from an earlier delivery: the authoritative one did not carry it",
				r.Name))
		}
	}

	sort.SliceStable(results, func(i, j int) bool {
		if results[i].Name != results[j].Name {
			return results[i].Name < results[j].Name
		}
		return strings.Join(results[i].Nodes, ",") < strings.Join(results[j].Nodes, ",")
	})

	started, finished := span(results)

	nodes := in.Nodes
	if len(nodes) == 0 && len(auth.Fingerprint) > 0 {
		nodes = nodesFromFingerprint(auth.Fingerprint)
		warnings = append(warnings, "no structured hardware inventory was supplied: node detail is limited to the run's fingerprint")
	}
	sort.SliceStable(nodes, func(i, j int) bool { return nodes[i].Name < nodes[j].Name })

	// Surface dropped artifacts. The operator records a truncation rather than
	// discarding silently, and a report that swallows that is worse than one
	// that says evidence was too large to carry.
	for _, r := range results {
		for _, a := range r.Artifacts {
			if a.Dropped != "" {
				warnings = append(warnings, fmt.Sprintf(
					"artifact %q on test %q was not stored: %s", a.Name, r.Name, a.Dropped))
			}
		}
	}

	return Resolved{
		Run: RunView{
			Namespace:   auth.Run.Namespace,
			Name:        auth.Run.Name,
			UID:         auth.Run.UID,
			Profile:     auth.Run.Profile,
			Cluster:     auth.Cluster,
			Phase:       auth.Phase,
			Terminal:    IsTerminal(auth.Phase),
			Baseline:    auth.Baseline,
			Fingerprint: auth.Fingerprint,
			Summary:     auth.Summary,
			StartedAt:   started,
			FinishedAt:  finished,
			SentAt:      auth.SentAt,
			DeliveryID:  auth.DeliveryID,
		},
		Results:   results,
		Nodes:     nodes,
		Artifacts: in.Artifacts,
		Meta:      in.Meta,
		Warnings:  warnings,
	}, nil
}

// isLink reports whether a result is about a relationship rather than a machine.
//
// Derived from the node count rather than from the scope string, because the
// node count is what actually makes a verdict unattributable to one endpoint —
// and because a result carrying two nodes is a link whether or not the scope
// field survived the trip.
func isLink(r contract.TestResult) bool { return len(r.Nodes) > 1 }

// span returns the earliest start and latest finish across results, which is
// when the work happened. Nil when nothing reported a time.
func span(results []ResultView) (start, end *time.Time) {
	for _, r := range results {
		if r.StartedAt != nil && (start == nil || r.StartedAt.Before(*start)) {
			start = r.StartedAt
		}
		if r.FinishedAt != nil && (end == nil || r.FinishedAt.After(*end)) {
			end = r.FinishedAt
		}
	}
	return start, end
}

// fingerprintKey matches the start of a `key=` field in a fingerprint
// descriptor: either at the beginning or after a space. Keys may be bare words
// (kernel, os, arch) or label names (nvidia.com/gpu.product).
var fingerprintKey = regexp.MustCompile(`(^|\s)([A-Za-z0-9][A-Za-z0-9._/-]*)=`)

// nodesFromFingerprint recovers what it can when no inventory was supplied.
//
// The operator builds the descriptor as space-joined `key=value` pairs — and
// VALUES CONTAIN SPACES: an OSImage is routinely "Ubuntu 24.04.1 LTS". Splitting
// on whitespace therefore yields "Ubuntu", which is not a shorter answer but a
// wrong one, and a wrong OS in a report someone is using to accept hardware is
// precisely the fabrication this package forbids.
//
// So fields are delimited by the NEXT key rather than by whitespace: find every
// `key=` boundary and take the value as everything up to the following one.
//
// Anything whose key this package does not recognise is left out of the
// structured fields rather than guessed at. Nothing is lost by that — the raw
// descriptor still reaches renderers through RunView.Fingerprint, which is where
// a reader should look for accelerator labels and anything else the operator
// chose to record.
func nodesFromFingerprint(fp map[string]string) []NodeInfo {
	out := make([]NodeInfo, 0, len(fp))
	for name, desc := range fp {
		n := NodeInfo{Name: name}
		bounds := fingerprintKey.FindAllStringSubmatchIndex(desc, -1)
		for i, b := range bounds {
			// b[4]:b[5] is the key; the value runs to the next key's start.
			key := desc[b[4]:b[5]]
			valueEnd := len(desc)
			if i+1 < len(bounds) {
				valueEnd = bounds[i+1][0]
			}
			value := strings.TrimSpace(desc[b[1]:valueEnd])
			if value == "" {
				continue
			}
			switch key {
			case "kernel":
				n.Kernel = value
			case "os":
				n.OSImage = value
			case "arch":
				n.Arch = value
			}
		}
		out = append(out, n)
	}
	return out
}
