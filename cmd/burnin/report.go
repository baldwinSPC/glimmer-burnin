package main

import (
	"flag"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"time"

	"github.com/baldwinSPC/glimmer-burnin/pkg/report"
	"github.com/baldwinSPC/glimmer-burnin/pkg/report/htmlreport"
	"github.com/baldwinSPC/glimmer-burnin/pkg/report/junitreport"
	"github.com/baldwinSPC/glimmer-burnin/pkg/report/nvvs"
)

// renderers is every format a user can ask for.
//
// A map rather than a switch so `-o` can list what it accepts when it is given
// something it does not recognise. A user who mistypes a format should be told
// the four that exist, not just that theirs is wrong.
func renderers() map[string]report.Renderer {
	return map[string]report.Renderer{
		nvvs.New().Name():               nvvs.New(),
		htmlreport.New().Name():         htmlreport.New(),
		htmlreport.NewMarkdown().Name(): htmlreport.NewMarkdown(),
		junitreport.Renderer{}.Name():   junitreport.Renderer{},
	}
}

func formatNames() []string {
	var out []string
	for name := range renderers() {
		out = append(out, name)
	}
	sort.Strings(out)
	return out
}

const reportUsage = `burnin report — render a run's results

USAGE
  burnin report --from-file <file>    [-o FORMAT] [--out DIR]
  burnin report --results-dir <dir>   [-o FORMAT] [--out DIR]
  burnin report --run <ns>/<name>     [-o FORMAT] [--out DIR] [--kubeconfig PATH]

INPUTS (exactly one)
  --from-file     a stored envelope, or a JSON array of them
  --results-dir   a directory of envelope JSON — a results directory, or a
                  ConfigMap sink's contents written to disk
  --run           read a BurnInRun from a cluster, along with its ConfigMap
                  sink and the NodeFingerprints for the nodes it touched

OUTPUT
  -o        format: %s (default html)
  --out     directory to write into; "-" writes a single document to stdout

Rendering the same run through two formats produces two views of one record —
they cannot disagree, because they read the same assembled input.
`

type reportFlags struct {
	fromFile   string
	resultsDir string
	run        string
	kubeconfig string
	format     string
	out        string
}

func runReport(args []string) error {
	var f reportFlags
	fs := flag.NewFlagSet("report", flag.ContinueOnError)
	fs.SetOutput(os.Stderr)
	fs.Usage = func() { fmt.Fprintf(os.Stderr, reportUsage, strings.Join(formatNames(), ", ")) }

	fs.StringVar(&f.fromFile, "from-file", "", "a stored envelope, or a JSON array of them")
	fs.StringVar(&f.resultsDir, "results-dir", "", "a directory of envelope JSON")
	fs.StringVar(&f.run, "run", "", "a BurnInRun as <namespace>/<name>")
	fs.StringVar(&f.kubeconfig, "kubeconfig", "", "path to a kubeconfig (default: in-cluster, then $KUBECONFIG, then ~/.kube/config)")
	fs.StringVar(&f.format, "o", "html", "output format")
	fs.StringVar(&f.out, "out", ".", `directory to write into, or "-" for stdout`)

	if err := fs.Parse(args); err != nil {
		return err
	}

	renderer, ok := renderers()[f.format]
	if !ok {
		return fmt.Errorf("unknown format %q; available: %s", f.format, strings.Join(formatNames(), ", "))
	}

	in, err := loadInput(f)
	if err != nil {
		return err
	}

	// Stamped here rather than inside a renderer so that every format of one
	// run agrees on when the document was made, and so a test can fix it.
	in.Meta = report.Meta{
		Generator:   "glimmer-burnin",
		Version:     version,
		GeneratedAt: time.Now().UTC(),
	}

	outputs, err := renderer.Render(in)
	if err != nil {
		return fmt.Errorf("rendering %s: %w", f.format, err)
	}
	return write(outputs, f.out, os.Stdout)
}

// loadInput resolves exactly one input mode.
//
// Exactly one, refused rather than prioritised: a user who passes two has a
// mistaken belief about which one is being read, and silently picking a winner
// leaves them holding a report about the wrong run.
func loadInput(f reportFlags) (report.Input, error) {
	var chosen []string
	if f.fromFile != "" {
		chosen = append(chosen, "--from-file")
	}
	if f.resultsDir != "" {
		chosen = append(chosen, "--results-dir")
	}
	if f.run != "" {
		chosen = append(chosen, "--run")
	}

	switch len(chosen) {
	case 0:
		return report.Input{}, fmt.Errorf("nothing to render: pass one of --from-file, --results-dir or --run")
	case 1:
	default:
		return report.Input{}, fmt.Errorf("pass exactly one input; got %s", strings.Join(chosen, " and "))
	}

	switch {
	case f.fromFile != "":
		return report.LoadFile(f.fromFile)
	case f.resultsDir != "":
		return report.LoadDir(f.resultsDir)
	default:
		ns, name, ok := strings.Cut(f.run, "/")
		if !ok || ns == "" || name == "" {
			return report.Input{}, fmt.Errorf("--run wants <namespace>/<name>, got %q", f.run)
		}
		return loadFromCluster(ns, name, f.kubeconfig)
	}
}

// write puts the documents where the user asked.
//
// Renderers return a base filename with no directory component, and this joins
// it to the destination — a renderer that chose its own path could write
// outside the directory it was given.
func write(outputs []report.Output, dest string, stdout io.Writer) error {
	if len(outputs) == 0 {
		return fmt.Errorf("the renderer produced no documents")
	}

	if dest == "-" {
		if len(outputs) > 1 {
			// Concatenating N documents into one stream would produce a file
			// that is not valid in its own format. The NVVS renderer emits one
			// per node precisely because the schema is single-host.
			names := make([]string, 0, len(outputs))
			for _, o := range outputs {
				names = append(names, o.Filename)
			}
			return fmt.Errorf(
				"this format produced %d documents (%s) and they cannot be concatenated onto stdout; use --out DIR",
				len(outputs), strings.Join(names, ", "))
		}
		_, err := stdout.Write(outputs[0].Data)
		return err
	}

	if err := os.MkdirAll(dest, 0o755); err != nil {
		return fmt.Errorf("creating %s: %w", dest, err)
	}
	for _, o := range outputs {
		// Defence in depth: a renderer is documented to return a base name, and
		// this makes a renderer that forgets unable to escape the directory.
		name := filepath.Base(o.Filename)
		if name == "." || name == string(filepath.Separator) {
			return fmt.Errorf("renderer produced an unusable filename %q", o.Filename)
		}
		path := filepath.Join(dest, name)
		if err := os.WriteFile(path, o.Data, 0o644); err != nil {
			return fmt.Errorf("writing %s: %w", path, err)
		}
		fmt.Fprintln(os.Stderr, "wrote", path)
	}
	return nil
}
