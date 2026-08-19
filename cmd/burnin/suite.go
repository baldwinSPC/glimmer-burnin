package main

import (
	"bufio"
	"bytes"
	"fmt"
	"io"
	"os"
	"sort"
	"strings"

	"k8s.io/apimachinery/pkg/util/yaml"

	api "github.com/baldwinSPC/glimmer-burnin/api/v1alpha1"
	"github.com/baldwinSPC/glimmer-burnin/pkg/hostinfo"
	"github.com/baldwinSPC/glimmer-burnin/pkg/localrun"
	"github.com/baldwinSPC/glimmer-burnin/pkg/plan"
)

// A suite file is multi-document YAML of the SAME objects the CRDs use.
//
// Not a CLI-native schema, deliberately. The apimachinery dependency costs this
// binary some size, and what it buys is that a profile is copy-paste identical
// between `kubectl apply` and a bare host: one schema to document, one linter to
// run, and a site that has tuned an acceptance profile in-cluster can run it on
// a fresh box without translating anything.
//
// A slimmer parallel schema would be a third contract that can drift, which is
// the disease this whole design exists to cure.

// suite is what a file declares.
type suite struct {
	Profiles map[string]*api.BurnInProfile
	Tests    map[string]*api.BurnInTest
	// Order records the order profiles appeared, so "the only profile" and "the
	// first profile" are answerable without a map iteration.
	Order []string
}

// typeMeta is the minimum needed to route a document to its type.
type typeMeta struct {
	Kind       string `json:"kind"`
	APIVersion string `json:"apiVersion"`
}

// loadSuite reads one or more YAML files.
func loadSuite(paths []string) (*suite, error) {
	s := &suite{
		Profiles: map[string]*api.BurnInProfile{},
		Tests:    map[string]*api.BurnInTest{},
	}
	for _, p := range paths {
		f, err := os.Open(p)
		if err != nil {
			return nil, fmt.Errorf("opening %s: %w", p, err)
		}
		err = s.readAll(f, p)
		f.Close()
		if err != nil {
			return nil, err
		}
	}
	if len(s.Profiles) == 0 {
		return nil, fmt.Errorf("no BurnInProfile found — a suite needs one to say what runs and in what order")
	}
	return s, nil
}

func (s *suite) readAll(r io.Reader, name string) error {
	// A YAMLReader splits on the document separator and hands back raw bytes,
	// which is what this needs: each document is routed by its own kind before
	// anything tries to give it a type.
	rdr := yaml.NewYAMLReader(bufio.NewReader(r))
	for i := 0; ; i++ {
		raw, err := rdr.Read()
		if err != nil {
			if err == io.EOF {
				return nil
			}
			return fmt.Errorf("%s: document %d: %w", name, i+1, err)
		}
		if len(bytes.TrimSpace(raw)) == 0 {
			continue
		}

		var tm typeMeta
		if err := yaml.Unmarshal(raw, &tm); err != nil {
			return fmt.Errorf("%s: document %d: %w", name, i+1, err)
		}

		switch tm.Kind {
		case "BurnInProfile":
			var p api.BurnInProfile
			if err := yaml.Unmarshal(raw, &p); err != nil {
				return fmt.Errorf("%s: BurnInProfile: %w", name, err)
			}
			if _, dup := s.Profiles[p.Name]; dup {
				return fmt.Errorf("%s: two profiles named %q", name, p.Name)
			}
			s.Profiles[p.Name] = &p
			s.Order = append(s.Order, p.Name)

		case "BurnInTest":
			var t api.BurnInTest
			if err := yaml.Unmarshal(raw, &t); err != nil {
				return fmt.Errorf("%s: BurnInTest: %w", name, err)
			}
			if _, dup := s.Tests[t.Name]; dup {
				return fmt.Errorf("%s: two tests named %q", name, t.Name)
			}
			s.Tests[t.Name] = &t

		case "":
			return fmt.Errorf("%s: document %d has no kind — is it a Kubernetes object?", name, i+1)

		default:
			// A suite file may sit beside sinks, schedules and runs that only
			// mean something in a cluster. Ignoring them lets one file serve
			// both paths, which is the entire reason for reusing the schema.
			continue
		}
	}
}

// buildPlan resolves a profile into something localrun can execute.
func (s *suite) buildPlan(profileName, node string, retries int32, rz *localrun.Rendezvous) (localrun.Plan, []string, error) {
	profile, err := s.profile(profileName)
	if err != nil {
		return localrun.Plan{}, nil, err
	}

	// Both host facts are probed ONCE, here, and pinned into the plan. The
	// alternative — probing inside Translate — would make translation depend on
	// the machine it happens to run on, and would re-answer the same question
	// per test.
	h := hostinfo.Probe(hostinfo.Options{})
	p := localrun.Plan{
		Node:              node,
		Vendor:            vendorOf(h),
		HostIP:            h.PrimaryIP,
		FailFast:          profile.Spec.FailFast,
		RetryOnErrorLimit: retries,
		Rendezvous:        rz,
	}

	var warnings []string
	seen := map[string]bool{}

	for i, pt := range profile.Spec.Tests {
		spec, name, err := s.resolveTest(profile.Name, i, pt)
		if err != nil {
			return localrun.Plan{}, nil, err
		}

		// Variants expand HERE, before anything downstream, exactly as they do
		// in the operator — and through the same function, so the two
		// dispatchers cannot plan a different number of executions from one
		// profile. Until this call existed, spec.tests[].variants was read,
		// parsed, validated by the CRD schema, and then silently discarded: a
		// four-cell precision sweep ran once, under the parent's name, with
		// none of the cell's thresholds and no BURNIN_VARIANT_* reaching the
		// runner. Nothing in the output said a cell was missing.
		//
		// Every check below (the duplicate-name check, the cluster-only
		// warnings) then applies PER CELL, which is what makes a cell's name a
		// result identity here as well.
		for _, cell := range plan.ExpandVariants(name, spec, pt.Required == nil || *pt.Required, pt.Variants) {
			// Test name is result identity, and the operator enforces
			// uniqueness at plan time. Enforced here too, or two results would
			// collide in the report and one would silently win. After expansion
			// this also catches two variants of one test sharing a name, which
			// is the same failure by a different route.
			if seen[cell.Name] {
				return localrun.Plan{}, nil, fmt.Errorf(
					"profile %s names the test %q twice — a test name is a result's identity", profile.Name, cell.Name)
			}
			seen[cell.Name] = true

			warnings = append(warnings, clusterOnlyFields(cell.Name, cell.Spec, rz)...)
			warnings = append(warnings, unresolvableEnvWarnings(p, cell.Name, cell.Spec)...)
			p.Tests = append(p.Tests, cell)
		}
	}

	if len(p.Tests) == 0 {
		return localrun.Plan{}, nil, fmt.Errorf("profile %s lists no tests", profile.Name)
	}

	// The same refusal the operator makes, for the same reason and with the
	// same message: an axis that cannot become an environment variable would
	// run the cell's DEFAULT configuration while the run reported it under a
	// distinct label. Refused before a single container starts, because a run
	// that quietly produces a meaningless sweep is a much worse report than one
	// that refuses at start and says which axis and why.
	if err := plan.RefuseUnreachableAxes(p.Tests); err != nil {
		return localrun.Plan{}, nil, err
	}
	return p, warnings, nil
}

// resolveTest turns a profile entry into a concrete spec.
func (s *suite) resolveTest(profile string, i int, pt api.ProfileTest) (api.BurnInTestSpec, string, error) {
	switch {
	case pt.Inline != nil && pt.TestRef != "":
		return api.BurnInTestSpec{}, "", fmt.Errorf(
			"profile %s test %d sets both testRef and inline — they are mutually exclusive", profile, i+1)

	case pt.Inline != nil:
		name := pt.TestRef
		if name == "" {
			name = fmt.Sprintf("%s-%d", pt.Inline.Kind, i+1)
		}
		return *pt.Inline, name, nil

	case pt.TestRef != "":
		t, ok := s.Tests[pt.TestRef]
		if !ok {
			return api.BurnInTestSpec{}, "", fmt.Errorf(
				"profile %s references the test %q, which is not in this suite — "+
					"add its BurnInTest document, or pass the file that has it", profile, pt.TestRef)
		}
		return t.Spec, t.Name, nil

	default:
		return api.BurnInTestSpec{}, "", fmt.Errorf("profile %s test %d names nothing", profile, i+1)
	}
}

// profile picks the profile to run.
func (s *suite) profile(name string) (*api.BurnInProfile, error) {
	if name != "" {
		p, ok := s.Profiles[name]
		if !ok {
			return nil, fmt.Errorf("no profile %q in this suite; found: %s", name, strings.Join(s.profileNames(), ", "))
		}
		return p, nil
	}
	if len(s.Profiles) == 1 {
		return s.Profiles[s.Order[0]], nil
	}
	// Refused rather than defaulting to the first. Picking one silently means a
	// user gets a run they did not ask for, against hardware they care about.
	return nil, fmt.Errorf("this suite declares %d profiles; name one with --profile: %s",
		len(s.Profiles), strings.Join(s.profileNames(), ", "))
}

func (s *suite) profileNames() []string {
	out := append([]string(nil), s.Order...)
	sort.Strings(out)
	return out
}

// clusterOnlyFields reports the parts of a spec that only mean something under a
// scheduler.
//
// Warned about rather than silently ignored: a user whose profile sets a
// nodeSelector has a belief about where this runs, and letting that pass without
// a word would leave the belief intact and wrong.
func clusterOnlyFields(name string, spec api.BurnInTestSpec, rz *localrun.Rendezvous) []string {
	var out []string

	switch spec.Scope {
	case api.ScopePair:
		if rz == nil {
			out = append(out, fmt.Sprintf(
				"%s is Pair-scope and no --role was given: it will run with BURNIN_ROLE unset, "+
					"which a runner treats as not-applicable and skips — pass --role server here and "+
					"--role client --peer <ip> on the other machine to actually measure the link", name))
		}
	case api.ScopeGroup:
		if rz == nil || rz.Rank == nil {
			out = append(out, fmt.Sprintf(
				"%s is Group-scope and no --rank was given: it will run with BURNIN_RANK unset, which a "+
					"fabric runner treats as not-applicable and skips — pass --rank i --nranks n on each "+
					"machine (--root <ip> on every rank but 0), then `burnin merge` the records", name))
		}
	}

	if spec.Runner != nil && spec.Runner.ReadinessProbe != nil && spec.Scope != api.ScopePair {
		// A probe on a Pair test is honoured here — the server end watches it to
		// announce its listener. Anywhere else there is nothing to gate.
		out = append(out, fmt.Sprintf(
			"%s declares a readinessProbe, which nothing here acts on outside a Pair-scope server", name))
	}
	return out
}

// unresolvableEnvWarnings names the variables a test declares that will NOT be
// set on this machine.
//
// Warned about for the same reason clusterOnlyFields warns about a
// nodeSelector: the author has a belief about what the runner will receive, and
// letting it pass without a word would leave the belief intact and wrong. The
// variable is left UNSET rather than set to "" — see localrun.ResolveEnv — so
// this warning is the only thing standing between a profile that reads a Secret
// and a runner that quietly behaves as if the Secret were blank.
func unresolvableEnvWarnings(p localrun.Plan, name string, spec api.BurnInTestSpec) []string {
	var out []string
	for _, v := range localrun.UnresolvableEnv(p, spec) {
		out = append(out, fmt.Sprintf(
			"%s declares env %s, which has no meaning outside a cluster: it will be UNSET rather than "+
				"empty, so a runner can tell it apart from a variable the cluster set to nothing", name, v))
	}
	return out
}

// vendorOf is this machine's accelerator vendor, read off the PCI bus.
//
// It decides which image a test resolves to, so the operator and this path must
// answer the same question from equally authoritative sources: the operator asks
// a NodeFingerprint, which derives the vendor from the DNS domain of a device
// plugin's label or resource; here there is no apiserver, so it is read from the
// hardware itself.
//
// EMPTY IS THE HONEST ANSWER when nothing is found, and it is not a mismatch —
// pkg/runnerimages.Resolve treats an unknown vendor exactly as it did before
// vendors existed. Only the FIRST accelerator is consulted, matching the
// operator's own vendorOf; a host with two vendors' cards in it is a case
// neither side handles yet and neither should pretend to.
//
// It takes an ALREADY-PROBED host so buildPlan can take the vendor and the
// primary IP from one probe rather than two.
func vendorOf(h hostinfo.Host) string {
	for _, a := range h.Accelerators {
		if a.Vendor != "" {
			return a.Vendor
		}
	}
	return ""
}
