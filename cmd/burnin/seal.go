package main

import (
	"bufio"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"flag"
	"fmt"
	"io"
	"io/fs"
	"os"
	"path"
	"path/filepath"
	"sort"
	"strings"
)

// A results directory is SEALED by a SHA256SUMS file written as the very last
// step of `burnin run` and `burnin merge` (#537).
//
// What it proves is narrow and worth stating exactly. It proves the directory
// holds the files the run wrote, byte for byte, and nothing else: a file edited,
// removed or added after the run fails `burnin verify`. It does NOT prove who
// produced them — anyone who can edit a file can also rewrite SHA256SUMS. That
// is what signing is for (#175). A checksum defends evidence against accident
// and careless handling, which is most of what happens to evidence.
//
// Sealing is last on purpose. A run interrupted before it finished filing has
// no SHA256SUMS, and `burnin verify` reports it as unsealed rather than
// verifying a half-written directory.
const sumsFile = "SHA256SUMS"

// sealDir writes SHA256SUMS over every regular file under dir.
//
// It refuses rather than seals a tree holding a symlink, device, FIFO or
// socket. A symlink is the dangerous one: a sealed directory that is later
// copied with a tool that follows links would carry whatever the link points
// at, under a checksum that verified clean.
func sealDir(dir string) error {
	files, err := listRegular(dir)
	if err != nil {
		return err
	}
	var b strings.Builder
	for _, rel := range files {
		sum, err := hashFile(filepath.Join(dir, filepath.FromSlash(rel)))
		if err != nil {
			return err
		}
		fmt.Fprintf(&b, "%s  %s\n", sum, rel)
	}

	tmp, err := os.CreateTemp(dir, ".SHA256SUMS-*")
	if err != nil {
		return err
	}
	defer os.Remove(tmp.Name())
	if _, err := tmp.WriteString(b.String()); err != nil {
		tmp.Close()
		return err
	}
	if err := tmp.Sync(); err != nil {
		tmp.Close()
		return err
	}
	if err := tmp.Close(); err != nil {
		return err
	}
	if err := os.Chmod(tmp.Name(), 0o644); err != nil {
		return err
	}
	if err := os.Rename(tmp.Name(), filepath.Join(dir, sumsFile)); err != nil {
		return err
	}
	if d, err := os.Open(dir); err == nil {
		_ = d.Sync()
		d.Close()
	}
	return nil
}

// listRegular walks dir without following links and returns every regular
// file except SHA256SUMS itself (and a sealing temp file), as sorted
// slash-separated paths relative to dir.
func listRegular(dir string) ([]string, error) {
	var out []string
	err := filepath.WalkDir(dir, func(p string, d fs.DirEntry, err error) error {
		if err != nil {
			return err
		}
		rel, err := filepath.Rel(dir, p)
		if err != nil {
			return err
		}
		rel = filepath.ToSlash(rel)
		switch t := d.Type(); {
		case t.IsDir():
			return nil
		case t.IsRegular():
			if rel == sumsFile || strings.HasPrefix(rel, ".SHA256SUMS-") {
				return nil
			}
			out = append(out, rel)
			return nil
		default:
			return fmt.Errorf("%s is not a regular file (%s); a results directory holds only files the run wrote", rel, t)
		}
	})
	sort.Strings(out)
	return out, err
}

func hashFile(p string) (string, error) {
	f, err := os.Open(p)
	if err != nil {
		return "", err
	}
	defer f.Close()
	h := sha256.New()
	if _, err := io.Copy(h, f); err != nil {
		return "", err
	}
	return hex.EncodeToString(h.Sum(nil)), nil
}

// verifyDir checks a sealed directory and returns every problem it finds,
// not just the first: someone deciding whether to trust evidence needs the
// whole list.
func verifyDir(dir string) ([]string, error) {
	b, err := os.ReadFile(filepath.Join(dir, sumsFile))
	if errors.Is(err, fs.ErrNotExist) {
		return []string{"not sealed: no SHA256SUMS. A run interrupted before it finished filing leaves none"}, nil
	}
	if err != nil {
		return nil, err
	}

	var problems []string
	listed := map[string]bool{}
	sc := bufio.NewScanner(strings.NewReader(string(b)))
	for n := 1; sc.Scan(); n++ {
		line := sc.Text()
		sum, rel, ok := strings.Cut(line, "  ")
		if !ok || len(sum) != 64 {
			problems = append(problems, fmt.Sprintf("SHA256SUMS line %d is malformed", n))
			continue
		}
		if clean := path.Clean(rel); clean != rel || path.IsAbs(rel) || rel == ".." || strings.HasPrefix(rel, "../") {
			problems = append(problems, fmt.Sprintf("SHA256SUMS names %q, which is outside the directory or not a clean path", rel))
			continue
		}
		listed[rel] = true
		if p := checkOrdinaryPath(dir, rel); p != "" {
			problems = append(problems, p)
			continue
		}
		got, err := hashFile(filepath.Join(dir, filepath.FromSlash(rel)))
		if err != nil {
			problems = append(problems, fmt.Sprintf("%s: %v", rel, err))
			continue
		}
		if got != sum {
			problems = append(problems, fmt.Sprintf("%s has changed since the run sealed it", rel))
		}
	}

	present, err := listRegular(dir)
	if err != nil {
		problems = append(problems, err.Error())
	}
	for _, rel := range present {
		if !listed[rel] {
			problems = append(problems, fmt.Sprintf("%s was not written by the run: it is not in SHA256SUMS", rel))
		}
	}
	return problems, nil
}

// checkOrdinaryPath lstat's every component of rel, so a listed file reached
// through a symlinked directory is refused rather than hashed.
func checkOrdinaryPath(dir, rel string) string {
	parts := strings.Split(rel, "/")
	cur := dir
	for i, part := range parts {
		cur = filepath.Join(cur, part)
		fi, err := os.Lstat(cur)
		if err != nil {
			return fmt.Sprintf("%s is missing", rel)
		}
		last := i == len(parts)-1
		if last && !fi.Mode().IsRegular() {
			return fmt.Sprintf("%s is not a regular file", rel)
		}
		if !last && !fi.IsDir() {
			return fmt.Sprintf("%s is reached through %s, which is not a directory", rel, part)
		}
	}
	return ""
}

const verifyUsage = `burnin verify — check a sealed results directory

USAGE
  burnin verify DIR

  Checks that DIR holds exactly the files the run wrote, unchanged: every file
  in SHA256SUMS re-hashed, nothing reached through a symlink, and no file
  present that SHA256SUMS does not list.

  It proves integrity, not origin: anyone able to edit a file can also rewrite
  SHA256SUMS.

EXIT
  0  sealed and intact
  1  not sealed, or changed since it was sealed
  3  could not read the directory
`

func runVerify(args []string) error {
	fs := flag.NewFlagSet("verify", flag.ContinueOnError)
	fs.SetOutput(os.Stderr)
	fs.Usage = func() { fmt.Fprint(os.Stderr, verifyUsage) }
	if err := fs.Parse(args); err != nil {
		if errors.Is(err, flag.ErrHelp) {
			return nil
		}
		return exitWith(exitError, err)
	}
	if fs.NArg() != 1 {
		fs.Usage()
		return exitWith(exitError, fmt.Errorf("verify takes exactly one directory"))
	}
	dir := fs.Arg(0)
	problems, err := verifyDir(dir)
	if err != nil {
		return exitWith(exitError, err)
	}
	if len(problems) > 0 {
		for _, p := range problems {
			fmt.Println("FAIL", p)
		}
		return exitWith(exitFail, fmt.Errorf("%s: %d problem(s)", dir, len(problems)))
	}
	fmt.Println("OK", dir, "is sealed and unchanged")
	return nil
}
