package report

import (
	"encoding/json"
	"fmt"
	"io/fs"
	"os"
	"path/filepath"
	"sort"
	"strings"

	"github.com/baldwinSPC/glimmer-burnin/pkg/contract"
)

// Loading stored envelopes back into an Input.
//
// A ConfigMap sink writes one key per delivery and a CLI run writes one file per
// delivery, so what a reader actually has is a PILE: several envelopes about one
// run, in no order, some of them superseded. Everything in this file is about
// turning that pile into a single coherent view without inventing anything.

// LoadDir reads every envelope under dir into an Input.
//
// It descends into subdirectories, because a ConfigMap dumped with `kubectl get
// -o json` and a CLI results directory nest differently and a reader should not
// have to care. Files that are not JSON are skipped silently — a results
// directory legitimately holds artifacts, logs and a README beside the
// envelopes.
//
// A file that IS JSON but is not an envelope is an ERROR, not a skip. Silently
// ignoring it is how a report ends up describing three of a run's four
// deliveries with nothing saying so, and a verdict assembled from part of the
// record is the one failure mode this package must not have.
func LoadDir(dir string) (Input, error) {
	var envelopes []*contract.Envelope
	err := filepath.WalkDir(dir, func(path string, d fs.DirEntry, err error) error {
		if err != nil {
			return err
		}
		if d.IsDir() || !strings.EqualFold(filepath.Ext(path), ".json") {
			return nil
		}
		data, err := os.ReadFile(path)
		if err != nil {
			return err
		}
		found, err := decodeEnvelopes(data)
		if err != nil {
			return fmt.Errorf("%s: %w", filepath.Base(path), err)
		}
		envelopes = append(envelopes, found...)
		return nil
	})
	if err != nil {
		return Input{}, err
	}
	if len(envelopes) == 0 {
		return Input{}, fmt.Errorf("no envelopes found under %s", dir)
	}
	return Assemble(envelopes)
}

// LoadFile reads one file, which may hold a single envelope or a JSON array of
// them.
func LoadFile(path string) (Input, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return Input{}, err
	}
	envelopes, err := decodeEnvelopes(data)
	if err != nil {
		return Input{}, fmt.Errorf("%s: %w", filepath.Base(path), err)
	}
	return Assemble(envelopes)
}

// decodeEnvelopes accepts one envelope or an array of them.
//
// Both shapes exist in the wild: a webhook sink posts one per request, while a
// ConfigMap sink's contents are usually collected into a list before they are
// handed anywhere. Requiring the caller to know which it has would push the
// guesswork out to every consumer, which is what this package exists to stop.
func decodeEnvelopes(data []byte) ([]*contract.Envelope, error) {
	trimmed := strings.TrimLeft(string(data), " \t\r\n")
	if strings.HasPrefix(trimmed, "[") {
		var list []*contract.Envelope
		if err := strictUnmarshal([]byte(trimmed), &list); err != nil {
			return nil, err
		}
		for i, e := range list {
			if e == nil {
				return nil, fmt.Errorf("entry %d is null", i)
			}
		}
		return list, nil
	}
	var one contract.Envelope
	if err := strictUnmarshal(data, &one); err != nil {
		return nil, err
	}
	return []*contract.Envelope{&one}, nil
}

// strictUnmarshal refuses unknown fields.
//
// A consumer rendering a document produced by a NEWER operator must be told it
// does not understand the whole record, rather than quietly dropping the field
// that mattered — a `baseline` flag it never learned about would turn a
// measurement sweep into a certification, silently, which is the exact bug #142
// was filed for one layer up.
func strictUnmarshal(data []byte, v any) error {
	dec := json.NewDecoder(strings.NewReader(string(data)))
	dec.DisallowUnknownFields()
	if err := dec.Decode(v); err != nil {
		return fmt.Errorf("not a burn-in envelope: %w", err)
	}
	return nil
}

// Assemble orders and validates a pile of envelopes into an Input.
//
// The ordering rule: the terminal RunPhaseChanged delivery is the verdict of
// record and sorts LAST, so Verdict() can take it without re-deriving anything.
// Checkpoints order by sequence, which is what makes a long soak's progress
// readable — they are numbered precisely so a retry keeps its identity, and a
// consumer receiving them out of order can still put the record back together.
//
// Envelopes are deduplicated by DeliveryID, because at-least-once delivery
// means a results directory legitimately holds the same delivery twice and a
// report that counted a retry as a second checkpoint would show a soak
// progressing twice as fast as it did.
func Assemble(envelopes []*contract.Envelope) (Input, error) {
	if len(envelopes) == 0 {
		return Input{}, fmt.Errorf("no envelopes: there is nothing to report on")
	}

	seen := make(map[string]bool, len(envelopes))
	out := make([]*contract.Envelope, 0, len(envelopes))
	var runUID string
	for _, e := range envelopes {
		if e == nil {
			continue
		}
		if err := e.Validate(); err != nil {
			return Input{}, fmt.Errorf("envelope %s: %w", e.DeliveryID, err)
		}
		// One run per report. Rendering two runs into one document would
		// produce a verdict about hardware that was never tested together, and
		// the UID is the identity that survives a name being reused.
		if runUID == "" {
			runUID = e.Run.UID
		} else if e.Run.UID != runUID {
			return Input{}, fmt.Errorf(
				"envelopes describe two different runs (%s and %s); a report covers one run",
				runUID, e.Run.UID)
		}
		if seen[e.DeliveryID] {
			continue
		}
		seen[e.DeliveryID] = true
		out = append(out, e)
	}
	if len(out) == 0 {
		return Input{}, fmt.Errorf("no usable envelopes")
	}

	sort.SliceStable(out, func(i, j int) bool { return deliveryOrder(out[i]) < deliveryOrder(out[j]) })
	return Input{Envelopes: out}, nil
}

// deliveryOrder ranks a delivery within a run's record.
//
// Terminal phase changes sort last so the verdict of record is the final
// element. Checkpoints sort by their own sequence. Everything else keeps its
// arrival order, which sort.SliceStable preserves within a rank.
func deliveryOrder(e *contract.Envelope) int64 {
	switch {
	case e.Reason == contract.ReasonPhaseChanged && IsTerminalPhase(e.Phase):
		return 1 << 40
	case e.Reason == contract.ReasonCheckpoint:
		// Offset so a checkpoint never sorts ahead of the run's own start.
		return 1<<20 + int64(e.CheckpointSequence)
	default:
		return 1 << 10
	}
}

// TerminalPhases are the run phases that mean the run is finished.
//
// They are spelled out rather than derived from api/v1alpha1's constants, which
// this package deliberately does not import. The strings are the CONTRACT — they
// are what a stored envelope actually contains, and an envelope written years
// ago must still render — so pinning them here is more honest than importing a
// type whose constants could be renamed without the wire format changing.
//
// The risk of pinning them is drift, and internal/sink.TestReportKnowsEveryTerminalPhase
// closes it from the side that CAN see both: a new terminal phase this list
// missed would make its runs render as unfinished forever, so a terminal Failed
// run would be reported as "still running" rather than as a failure.
var TerminalPhases = []string{"Passed", "Failed", "Error", "Cancelled", "Skipped"}

// IsTerminalPhase reports whether a run in this phase is finished.
func IsTerminalPhase(phase string) bool {
	for _, p := range TerminalPhases {
		if phase == p {
			return true
		}
	}
	return false
}

// Verdict is the delivery of record: the terminal phase change if the run
// finished, otherwise the most recent delivery.
//
// A report on an unfinished run is legitimate — that is what a checkpoint is for
// — so this returns the best available rather than an error. What it must never
// do is present a checkpoint AS a verdict, which is why Final reports whether
// the run had actually finished.
func (in Input) Verdict() (e *contract.Envelope, final bool) {
	if len(in.Envelopes) == 0 {
		return nil, false
	}
	last := in.Envelopes[len(in.Envelopes)-1]
	return last, last.Reason == contract.ReasonPhaseChanged && IsTerminalPhase(last.Phase)
}

// Checkpoints returns the run's progress deliveries in sequence order.
func (in Input) Checkpoints() []*contract.Envelope {
	var out []*contract.Envelope
	for _, e := range in.Envelopes {
		if e.Reason == contract.ReasonCheckpoint {
			out = append(out, e)
		}
	}
	return out
}

// NodeByName returns the structured inventory for a node, if the caller
// supplied one.
//
// The second return is whether anything is known at all, so a renderer can tell
// "no inventory" from "an inventory with every field empty" — the first is a
// caller that did not collect fingerprints, the second is hardware that reported
// nothing, and a report must not render them the same way.
func (in Input) NodeByName(name string) (NodeInfo, bool) {
	for _, n := range in.Nodes {
		if n.Name == name {
			return n, true
		}
	}
	return NodeInfo{}, false
}

// Fingerprint returns what the RUN captured about a node, which is the fallback
// when no structured inventory was supplied.
//
// It is a flat map of whatever was recorded at run start. Renderers read it with
// FingerprintValue rather than indexing it directly, so a missing key is
// unambiguously absent.
func (in Input) Fingerprint() map[string]string {
	v, _ := in.Verdict()
	if v == nil {
		return nil
	}
	return v.Fingerprint
}

// FingerprintValue reads one captured value, reporting whether it was captured
// at all.
//
// The bool is the whole point. A renderer that wrote fp["serial"] straight into
// a document would print an empty serial number for a node whose serial was
// never read, and an empty serial in an RMA is a claim that the part has none.
func FingerprintValue(fp map[string]string, key string) (string, bool) {
	v, ok := fp[key]
	if !ok || strings.TrimSpace(v) == "" {
		return "", false
	}
	return v, true
}
