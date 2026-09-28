package main

import (
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"unicode/utf8"

	"github.com/baldwinSPC/glimmer-burnin/pkg/sanitize"
)

// sanitizedMarker marks a directory as sanitised. Its presence makes the
// result TERMINAL: its machine identity is a pseudonym, so nothing that asks
// "is this the same hardware?" may accept it, and it is never sanitised twice.
const sanitizedMarker = "SANITIZED.json"

func isSanitized(dir string) bool {
	_, err := os.Stat(filepath.Join(dir, sanitizedMarker))
	return err == nil
}

const sanitizeUsage = `burnin sanitize — a copy of a result that can leave the site

USAGE
  burnin sanitize --results-dir IN --out OUT [--host NAME]... [--user NAME]...

  Writes a copy of a sealed results directory with machine identity replaced:
  node names, UUIDs, MACs, RDMA GUIDs, serials and hostnames become
  pseudonyms, IP addresses move into documentation ranges, and secrets and
  environment dumps are masked. Every measurement is kept exactly as written.

  Pseudonyms are stable within one output and unlinkable across outputs: the
  salt is drawn fresh for each run of this command and never written down.

  The input must be sealed and unchanged (see "burnin verify"). The output is
  sealed too, and is TERMINAL: nothing compares a sanitised result, because
  its identity is a pseudonym.

  It fails closed. Every output file is re-scanned, and on any finding no
  output is written at all.

FLAGS
  --results-dir  the sealed directory to sanitise
  --out          where to write the copy (must not exist)
  --host         a name to treat as a host, beyond the node names the result
                 already records (repeatable)
  --user         a name to treat as a user (repeatable)
`

func runSanitize(args []string) error {
	fs := flag.NewFlagSet("sanitize", flag.ContinueOnError)
	fs.SetOutput(os.Stderr)
	fs.Usage = func() { fmt.Fprint(os.Stderr, sanitizeUsage) }
	var in, out string
	var hosts, users multiFlag
	fs.StringVar(&in, "results-dir", "", "the sealed directory to sanitise")
	fs.StringVar(&out, "out", "", "where to write the copy")
	fs.Var(&hosts, "host", "a host name to replace (repeatable)")
	fs.Var(&users, "user", "a user name to replace (repeatable)")
	if err := fs.Parse(args); err != nil {
		if errors.Is(err, flag.ErrHelp) {
			return nil
		}
		return exitWith(exitError, err)
	}
	if in == "" || out == "" {
		fs.Usage()
		return exitWith(exitError, fmt.Errorf("both --results-dir and --out are required"))
	}
	n, err := sanitizeDir(in, out, hosts, users)
	if err != nil {
		return exitWith(exitError, err)
	}
	fmt.Printf("sanitised %s -> %s (%d files); check it with: burnin verify %s\n", in, out, n, out)
	return nil
}

func sanitizeDir(in, out string, hosts, users []string) (int, error) {
	if isSanitized(in) {
		return 0, fmt.Errorf("%s is already sanitised; a sanitised result is terminal", in)
	}
	problems, err := verifyDir(in)
	if err != nil {
		return 0, err
	}
	if len(problems) > 0 {
		return 0, fmt.Errorf("%s is not sealed and unchanged, so there is no knowing what is being shared: %s",
			in, strings.Join(problems, "; "))
	}
	if _, err := os.Lstat(out); err == nil {
		return 0, fmt.Errorf("%s already exists; refusing to write into it", out)
	}

	files, err := listRegular(in)
	if err != nil {
		return 0, err
	}
	s, err := sanitize.New(sanitize.Options{Users: users})
	if err != nil {
		return 0, err
	}
	for _, h := range append(inventory(in, files), hosts...) {
		s.AddHost(h)
	}

	type outFile struct {
		rel  string
		data []byte
	}
	var outs []outFile
	seen := map[string]string{}
	for _, rel := range files {
		data, err := os.ReadFile(filepath.Join(in, filepath.FromSlash(rel)))
		if err != nil {
			return 0, err
		}
		nrel, err := s.Path(rel)
		if err != nil {
			return 0, err
		}
		if prev, dup := seen[nrel]; dup {
			return 0, fmt.Errorf("%s and %s would both be written as %s", prev, rel, nrel)
		}
		seen[nrel] = rel

		var clean []byte
		var findings []string
		switch {
		case strings.HasSuffix(rel, ".json"):
			if clean, err = s.JSON(data); err != nil {
				return 0, fmt.Errorf("%s: %w", rel, err)
			}
			if findings, err = s.ResidualJSON(clean); err != nil {
				return 0, fmt.Errorf("%s: %w", rel, err)
			}
		case utf8.Valid(data):
			text, err := s.Text(string(data))
			if err != nil {
				return 0, fmt.Errorf("%s: %w", rel, err)
			}
			clean = []byte(text)
			findings = s.Residual(text)
		default:
			return 0, fmt.Errorf("%s is not text, and a binary file cannot be scanned; nothing was written", rel)
		}
		if len(findings) > 0 {
			return 0, fmt.Errorf("the residual scan still finds identifying values in %s after sanitising (%s); "+
				"nothing was written", nrel, strings.Join(findings, ", "))
		}
		outs = append(outs, outFile{nrel, clean})
	}

	staging := out + ".sanitize-staging"
	if err := os.RemoveAll(staging); err != nil {
		return 0, err
	}
	fail := func(err error) (int, error) {
		os.RemoveAll(staging)
		return 0, err
	}
	for _, f := range outs {
		p := filepath.Join(staging, filepath.FromSlash(f.rel))
		if err := os.MkdirAll(filepath.Dir(p), 0o755); err != nil {
			return fail(err)
		}
		if err := os.WriteFile(p, f.data, 0o644); err != nil {
			return fail(err)
		}
	}
	counts := map[string]int{}
	for k, v := range s.Counts() {
		counts[string(k)] = v
	}
	marker := map[string]any{
		"sanitized":  true,
		"replaced":   counts,
		"producer":   cliProducer(),
		"terminal":   "machine identity in this directory is pseudonymous; it cannot be compared against another result",
		"unlinkable": "pseudonyms were made under a salt that was never written down, so no other output shares them",
	}
	if err := writeJSON(filepath.Join(staging, sanitizedMarker), marker); err != nil {
		return fail(err)
	}
	if err := sealDir(staging); err != nil {
		return fail(err)
	}
	if err := os.Rename(staging, out); err != nil {
		return fail(err)
	}
	return len(outs), nil
}

// inventory collects the names a result already records: node names in run
// records, envelope fingerprints and node lists, a Pair server's peer, and the
// cluster name. These are single-label names no content pattern can
// recognise, so without this they would be shared verbatim.
func inventory(dir string, files []string) []string {
	set := map[string]bool{}
	add := func(v any) {
		if s, ok := v.(string); ok && s != "" {
			set[s] = true
		}
	}
	for _, rel := range files {
		if !strings.HasSuffix(rel, ".json") {
			continue
		}
		b, err := os.ReadFile(filepath.Join(dir, filepath.FromSlash(rel)))
		if err != nil {
			continue
		}
		var m map[string]any
		if json.Unmarshal(b, &m) != nil {
			continue
		}
		add(m["node"])
		add(m["peerNode"])
		if run, ok := m["run"].(map[string]any); ok && run["namespace"] == "local" {
			// A bare-metal run is named after its node.
			add(run["name"])
		}
		if c, ok := m["cluster"].(map[string]any); ok {
			add(c["name"])
		}
		if fp, ok := m["fingerprint"].(map[string]any); ok {
			for k := range fp {
				set[k] = true
			}
		}
		results, _ := m["results"].([]any)
		for _, r := range results {
			if rm, ok := r.(map[string]any); ok {
				nodes, _ := rm["nodes"].([]any)
				for _, n := range nodes {
					add(n)
				}
				add(rm["node"])
			}
		}
	}
	out := make([]string, 0, len(set))
	for k := range set {
		out = append(out, k)
	}
	sort.Strings(out)
	return out
}
