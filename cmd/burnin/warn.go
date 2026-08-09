package main

import (
	"encoding/json"
	"fmt"
	"os"

	"github.com/baldwinSPC/glimmer-burnin/pkg/contract"
)

// warn reports something the command worked around.
//
// It goes to stderr so it cannot contaminate a document written to stdout, and
// it is never silent: a report assembled from part of a record must say which
// part it could not reach.
func warn(format string, args ...any) {
	fmt.Fprintf(os.Stderr, "burnin: warning: "+format+"\n", args...)
}

// decodeOne parses a stored envelope and holds it to the contract's own
// validation, so this command cannot accept a document the rest of the system
// would reject.
func decodeOne(data []byte) (*contract.Envelope, error) {
	var e contract.Envelope
	if err := json.Unmarshal(data, &e); err != nil {
		return nil, fmt.Errorf("decoding envelope: %w", err)
	}
	if err := e.Validate(); err != nil {
		return nil, fmt.Errorf("invalid envelope: %w", err)
	}
	return &e, nil
}
