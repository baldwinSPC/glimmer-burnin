package contract

import "runtime/debug"

// Producer names the build that produced an envelope (#537).
//
// A stored verdict is kept for years, and without this it cannot say which
// operator or CLI produced it. When a runner or verdict defect is found later,
// the question every consumer asks is "which of our stored results came from
// the affected build", and only this field answers it.
type Producer struct {
	// Name is the program: "glimmer-burnin-operator" or "burnin".
	Name string `json:"name"`
	// Version is the release tag the binary was stamped with at build time,
	// or "(devel)" when it was not stamped. Never guessed.
	Version string `json:"version,omitempty"`
	// Commit is the source revision, with "-dirty" appended when the build
	// tree had uncommitted changes. Empty when the build recorded none.
	Commit string `json:"commit,omitempty"`
}

// ProducerFromBuild describes the running binary.
//
// stampedVersion and stampedCommit are what the build passed through -ldflags;
// either may be empty. Where the build did not stamp a commit, the Go
// toolchain's own VCS stamping is used, which a plain `go build` inside a git
// checkout provides.
func ProducerFromBuild(name, stampedVersion, stampedCommit string) *Producer {
	p := &Producer{Name: name, Version: stampedVersion, Commit: stampedCommit}
	bi, ok := debug.ReadBuildInfo()
	if ok {
		if p.Version == "" || p.Version == "(devel)" {
			if v := bi.Main.Version; v != "" {
				p.Version = v
			}
		}
		if p.Commit == "" {
			var rev string
			dirty := false
			for _, s := range bi.Settings {
				switch s.Key {
				case "vcs.revision":
					rev = s.Value
				case "vcs.modified":
					dirty = s.Value == "true"
				}
			}
			if rev != "" && dirty {
				rev += "-dirty"
			}
			p.Commit = rev
		}
	}
	if p.Version == "" {
		p.Version = "(devel)"
	}
	return p
}
