package report

import (
	"encoding/json"
	"fmt"
	"io/fs"
	"path"
	"sort"

	"github.com/baldwinSPC/glimmer-burnin/pkg/contract"
)

// LoadEnvelopes reads every envelope in a directory.
//
// It takes an fs.FS rather than a path so that a caller can hand it a
// ConfigMap's contents, an embedded fixture, or a real directory without this
// package growing an opinion about where results live. The bare-metal CLI writes
// a results directory; the kubectl plugin reads a ConfigMap; both arrive here.
//
// Files are read in name order and every .json file at any depth is considered,
// because the CLI's layout is results/<run>/envelopes/NNN-<reason>-<key>.json
// and a caller should be able to point at either level.
//
// A file that is not a valid envelope is an error rather than a skip. A report
// assembled from "the ones that happened to parse" is a report whose omissions
// nobody knows about, and the omission most likely to be silently dropped is a
// malformed terminal delivery — the one that carries the verdict.
func LoadEnvelopes(fsys fs.FS, dir string) ([]*contract.Envelope, error) {
	if dir == "" {
		dir = "."
	}

	var names []string
	err := fs.WalkDir(fsys, dir, func(p string, d fs.DirEntry, err error) error {
		if err != nil {
			return err
		}
		if d.IsDir() || path.Ext(p) != ".json" {
			return nil
		}
		names = append(names, p)
		return nil
	})
	if err != nil {
		return nil, fmt.Errorf("report: walking %s: %w", dir, err)
	}
	sort.Strings(names)

	envelopes := make([]*contract.Envelope, 0, len(names))
	for _, name := range names {
		b, err := fs.ReadFile(fsys, name)
		if err != nil {
			return nil, fmt.Errorf("report: reading %s: %w", name, err)
		}
		e, err := ParseEnvelope(b)
		if err != nil {
			return nil, fmt.Errorf("report: %s: %w", name, err)
		}
		envelopes = append(envelopes, e)
	}
	if len(envelopes) == 0 {
		return nil, fmt.Errorf("report: no envelopes found under %s: %w", dir, ErrNoEnvelopes)
	}
	return envelopes, nil
}

// ParseEnvelope decodes one envelope and validates it.
//
// Validation is the contract's own, so this package cannot drift into accepting
// a document the rest of the system would reject.
func ParseEnvelope(b []byte) (*contract.Envelope, error) {
	var e contract.Envelope
	if err := json.Unmarshal(b, &e); err != nil {
		return nil, fmt.Errorf("decoding envelope: %w", err)
	}
	if err := e.Validate(); err != nil {
		return nil, fmt.Errorf("invalid envelope: %w", err)
	}
	return &e, nil
}
