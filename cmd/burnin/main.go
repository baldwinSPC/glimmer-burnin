// Command burnin is the user-facing tool for burn-in results.
//
// Today it renders reports. It is deliberately its own binary rather than a
// subcommand of the manager: the manager is a controller that runs in a cluster
// under a service account, and this is a thing a person runs on a laptop
// against files. Fusing them would put client-go's flags and the manager's
// leader-election machinery in front of someone who wants to turn a JSON file
// into an HTML page.
//
// It is also the binary the rest of the CLI grows into. The bare-metal
// dispatcher (plan, run, fingerprint) fills in the same command tree, which is
// why the dispatch below is a switch over subcommands rather than a single
// flag set.
//
// Installed as kubectl-burnin on PATH, `kubectl burnin report --run ns/name`
// works with no further plumbing — that is just how kubectl discovers plugins.
package main

import (
	"errors"
	"fmt"
	"os"
)

// version is stamped at build time with -ldflags "-X main.version=v0.6.0".
// Unset, it says so rather than claiming a version it does not have.
var version = "(devel)"

const usage = `burnin — burn-in results

USAGE
  burnin <command> [flags]

COMMANDS
  plan       resolve a suite and print what would run
  run        execute a profile on this machine
  merge      fold every rank's record into one collective verdict
  report     render a run's results as a document
  version    print the version

Run "burnin <command> -h" for a command's flags.
`

func main() {
	if len(os.Args) < 2 {
		fmt.Fprint(os.Stderr, usage)
		os.Exit(2)
	}

	var err error
	switch os.Args[1] {
	case "run":
		err = runRun(os.Args[2:])
	case "plan":
		// A dry run under its own name, because "show me what this would do" is
		// a question people ask before they trust a tool with their hardware.
		err = runRun(append([]string{"--dry-run"}, os.Args[2:]...))
	case "merge":
		err = runMerge(os.Args[2:])
	case "report":
		err = runReport(os.Args[2:])
	case "version", "--version", "-v":
		fmt.Println("burnin", version)
	case "help", "--help", "-h":
		fmt.Print(usage)
	default:
		fmt.Fprintf(os.Stderr, "burnin: unknown command %q\n\n%s", os.Args[1], usage)
		os.Exit(2)
	}

	if err != nil {
		// A run reports its verdict through the exit code, and those codes are
		// load-bearing: a CI job has to be able to tell a hardware failure from
		// a run that never happened.
		var ex *exitErr
		if errors.As(err, &ex) {
			if msg := ex.Error(); msg != "" {
				fmt.Fprintln(os.Stderr, "burnin:", msg)
			}
			os.Exit(ex.code)
		}
		fmt.Fprintln(os.Stderr, "burnin:", err)
		os.Exit(1)
	}
}
