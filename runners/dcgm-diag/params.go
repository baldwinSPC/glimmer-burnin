// SPDX-License-Identifier: Apache-2.0
// Copyright the Glimmer authors.

package main

import (
	"fmt"
	"regexp"
	"strconv"
	"strings"
	"time"
)

// runSpec is the scope of one diagnostic: what goes after dcgmi's -r, and where
// that choice came from.
type runSpec struct {
	arg    string // the -r argument, verbatim
	level  int    // 0 when named tests were asked for
	tests  string // the normalised test list, "" when a level was asked for
	source string // explicit, derived, or named
}

// A DCGM parameter clause: dotted name, then "=", then a value.
//
// Two or more segments, because DCGM scopes some parameters to a subtest of a
// plugin (pcie.h2d_d2h_single_pinned.min_bandwidth) as well as to the plugin
// itself (memory.is_allowed). Requiring exactly one dot rejected those.
var paramClause = regexp.MustCompile(`^[A-Za-z][A-Za-z0-9_]*(\.[A-Za-z][A-Za-z0-9_]*)+=[^;]+$`)

// A test name acceptable to -r. Digits alone are a run LEVEL, and are rejected
// by resolveRun rather than here.
var testName = regexp.MustCompile(`^[A-Za-z][A-Za-z0-9_ -]*$`)

// resolveRun decides what to ask dcgmi to run.
//
// BURNIN_DCGM_LEVEL and BURNIN_DCGM_TESTS are mutually exclusive and setting
// both is an error. They describe different scopes — "everything up to this
// depth" versus "exactly these plugins" — and picking one would repeat the
// substitution resolveLevel already refuses to make: a profile that asked for
// both and got the other's scope would certify a node against a suite it never
// ran, with nothing in the output admitting it.
func resolveRun(level, tests string, duration time.Duration) (runSpec, error) {
	tests = strings.TrimSpace(tests)
	if tests == "" {
		lvl, source, err := resolveLevel(level, duration)
		if err != nil {
			return runSpec{}, err
		}
		return runSpec{arg: strconv.Itoa(lvl), level: lvl, source: source}, nil
	}
	if strings.TrimSpace(level) != "" {
		return runSpec{}, fmt.Errorf(
			"BURNIN_DCGM_LEVEL=%q and BURNIN_DCGM_TESTS=%q are both set, and they ask for "+
				"different scopes: set one or the other", level, tests)
	}

	names := make([]string, 0, 4)
	for _, raw := range strings.Split(tests, ",") {
		name := strings.TrimSpace(raw)
		if name == "" {
			return runSpec{}, fmt.Errorf(
				"BURNIN_DCGM_TESTS=%q has an empty entry: use a comma-separated list of "+
					"DCGM test names, e.g. memory,pcie", tests)
		}
		if _, err := strconv.Atoi(name); err == nil {
			return runSpec{}, fmt.Errorf(
				"BURNIN_DCGM_TESTS=%q names %q, which is a DCGM run level rather than a test: "+
					"use BURNIN_DCGM_LEVEL for a level", tests, name)
		}
		if !testName.MatchString(name) {
			return runSpec{}, fmt.Errorf(
				"BURNIN_DCGM_TESTS=%q contains %q, which is not a DCGM test name", tests, name)
		}
		names = append(names, name)
	}
	joined := strings.Join(names, ",")
	return runSpec{arg: joined, tests: joined, source: "named"}, nil
}

// resolveParams builds dcgmi's -p argument from the two surfaces that feed it,
// and returns "" when neither was set.
//
// BURNIN_DCGM_ALLOW is the ergonomic half: each name becomes
// "<name>.is_allowed=true". That expansion is uniform, so the runner holds no
// table of plugin names and no per-SKU knowledge — which is the point. DCGM
// owns the allowlist, and a profile's job is to say what its hardware supports.
//
// BURNIN_DCGM_PARAMS is the raw half, in DCGM's own vocabulary, for everything
// is_allowed does not cover.
//
// A parameter set by both is an error rather than a precedence rule: DCGM
// documents no order for a repeated key, so a runner that picked one would be
// guessing at which of two contradictory instructions the operator meant.
func resolveParams(allow, raw string) (string, error) {
	clauses := make([]string, 0, 8)
	seen := map[string]string{}

	add := func(clause, from string) error {
		if !paramClause.MatchString(clause) {
			return fmt.Errorf(
				"%s produced %q, which is not a DCGM parameter: write "+
					"<test>.<variable>=<value>, separated by \";\"", from, clause)
		}
		key := clause[:strings.Index(clause, "=")]
		if prev, dup := seen[key]; dup {
			return fmt.Errorf(
				"%q is set twice (%q and %q) and DCGM documents no precedence between them, "+
					"so the run would be ambiguous: set it once", key, prev, clause)
		}
		seen[key] = clause
		clauses = append(clauses, clause)
		return nil
	}

	for _, entry := range strings.Split(allow, ",") {
		name := strings.TrimSpace(entry)
		if name == "" {
			continue
		}
		if !testName.MatchString(name) {
			return "", fmt.Errorf(
				"BURNIN_DCGM_ALLOW names %q, which is not a DCGM test name", name)
		}
		if err := add(name+".is_allowed=true", "BURNIN_DCGM_ALLOW"); err != nil {
			return "", err
		}
	}
	for _, clause := range strings.Split(raw, ";") {
		clause = strings.TrimSpace(clause)
		if clause == "" {
			continue
		}
		if err := add(clause, "BURNIN_DCGM_PARAMS"); err != nil {
			return "", err
		}
	}
	return strings.Join(clauses, ";"), nil
}

// diagTimeoutHeadroom is how far inside the runner's own deadline dcgmi's -t is
// placed when it is derived.
//
// It has to be enough for dcgmi to stop its plugins and print the JSON document
// the verdict is read from. Whether DCGM does print one on its own timeout is
// unverified on GB10; if it does not, the run lands on the same "no readable
// JSON result" Error it reaches today, so the derivation cannot make the
// outcome worse than leaving -t unset.
const diagTimeoutHeadroom = 15 * time.Second

// minDerivedDiagTimeout keeps a derived -t from being so short that it ends a
// diagnostic which had time to finish.
const minDerivedDiagTimeout = 30 * time.Second

// resolveDiagTimeout decides the value for dcgmi's own -t.
//
// Left unset, DCGM's timeout is UNLIMITED, so the only thing bounding the
// diagnostic is this runner's context — and that expires as a SIGKILL, which
// takes dcgmi's JSON with it and leaves the node unjudged with no record of
// which subtests had already run. Placing DCGM's timeout inside the runner's
// deadline means the tool ends its own run and reports what it got.
//
// An explicit value is honoured even when it sits outside the runner's budget,
// on the same rule as an explicit level: the operator asked for it. It is
// logged, because a -t beyond the budget can never fire and silently restores
// the SIGKILL this exists to avoid.
func resolveDiagTimeout(explicit time.Duration, budget time.Duration) time.Duration {
	if explicit > 0 {
		if explicit >= budget {
			logf("BURNIN_DCGM_TIMEOUT_SECONDS is %s, at or beyond this run's %s budget, so "+
				"dcgmi's own timeout cannot fire and the run will end as a SIGKILL with no "+
				"diagnostic document", explicit, budget)
		}
		return explicit
	}
	derived := budget - diagTimeoutHeadroom
	if derived < minDerivedDiagTimeout {
		return 0
	}
	return derived
}
