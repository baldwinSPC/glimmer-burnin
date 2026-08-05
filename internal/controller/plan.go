package controller

import (
	"encoding/json"
	"fmt"
	"strings"
	"time"

	burninv1alpha1 "github.com/baldwinSPC/glimmer-burnin/api/v1alpha1"
	"github.com/baldwinSPC/glimmer-burnin/pkg/verdict"
)

// planAnnotation pins the fully-resolved execution plan onto the run at start.
//
// Everything downstream — pod identity, result bookkeeping, the final verdict,
// sink routing — derives from the plan, never from a live re-read of the
// profile or the cluster. A run must be hermetic: editing a profile, deleting
// a BurnInTest, or a node dropping out of a label selector mid-run must not
// change what an in-flight run believes it is executing, which tests count as
// required, or which nodes it owes results for. Re-resolving on every pass
// allowed all of those to silently rewrite history (up to and including a
// required failure being forgotten because its node left the selector).
const planAnnotation = "burnin.glimmer.ai/plan"

// pendingDeliveryAnnotation records that the terminal envelope has not yet
// been accepted by every sink. The terminal transition is the one delivery
// that has no "later transition" to piggyback on, so it gets its own retry
// loop; the annotation is cleared once every sink has taken it.
const pendingDeliveryAnnotation = "burnin.glimmer.ai/pending-terminal-delivery"

// cancellingAnnotation records that the controller has OBSERVED spec.cancel and
// has begun stopping the run. Its value is the reason the cancel carried.
//
// It exists because cancellation is one-way and a graceful cancel takes several
// passes: the run stops launching work and waits for the executions already in
// flight to finish. Without a durable mark, clearing spec.cancel during that
// drain would resume a run that had already been told to stop — and a run that
// can un-cancel races its own cleanup, up to and including resuming after its
// cordons have been released, which would put test load back onto nodes the
// scheduler has already been given back.
const cancellingAnnotation = "burnin.glimmer.ai/cancelling"

// maxPlanBytes bounds the serialized plan. Annotations share a 256KiB budget
// per object; refusing an oversized plan at start beats an unwritable run.
const maxPlanBytes = 128 * 1024

// plan is the pinned execution plan.
type plan struct {
	// Version guards future shape changes.
	Version int `json:"version"`
	// Targets is the resolved node list. Pinned so a node dropping out of a
	// selector mid-run still owes its results.
	Targets []string `json:"targets"`
	// Sinks is the profile's sink list at start.
	Sinks []string `json:"sinks,omitempty"`
	// FailFast is the profile's policy at start.
	FailFast bool `json:"failFast,omitempty"`
	// Tests are the materialised test specs, in execution order.
	Tests []plannedTest `json:"tests"`
}

type plannedTest struct {
	Name     string                        `json:"name"`
	Required bool                          `json:"required"`
	Spec     burninv1alpha1.BurnInTestSpec `json:"spec"`
}

// buildPlan materialises a profile against resolved targets, validating the
// invariants the rest of the controller depends on.
//
// nodeCap is the run's resolved MaxConcurrentNodes. It is validated here even
// though it is read live elsewhere, because a Pair-scope test under a cap of 1
// can never launch, and a run that quietly waits out its deadline to say so is
// a much worse report than one that refuses at start with the reason.
func buildPlan(profile *burninv1alpha1.BurnInProfile, tests []resolvedTest, targets []string, nodeCap int) (*plan, error) {
	if err := refuseUnsatisfiableThresholds(tests); err != nil {
		return nil, err
	}
	seen := map[string]bool{}
	p := &plan{
		Version:  1,
		Targets:  targets,
		Sinks:    profile.Spec.Sinks,
		FailFast: profile.Spec.FailFast,
	}
	for _, t := range tests {
		// Result identity is (test name, node). A duplicate name would make
		// two different tests share results — the second would never run, and
		// its required flag could be silently overwritten by the first's.
		if seen[t.name] {
			return nil, fmt.Errorf("profile lists test %q twice — test names are result identity and must be unique", t.name)
		}
		seen[t.name] = true
		if err := validateTopology(t, targets, nodeCap); err != nil {
			return nil, err
		}
		if err := validateHostPaths(t); err != nil {
			return nil, err
		}
		p.Tests = append(p.Tests, plannedTest{Name: t.name, Required: t.required, Spec: t.spec})
	}
	return p, nil
}

// validateTopology enforces what a multi-node test needs from the run.
//
// Every rule refuses at START rather than mid-flight, and every one produces an
// Error naming what was actually found. A Pair or Group test that ran against
// the wrong topology would not fail — it would produce a plausible-looking
// number for a link or a collective nobody asked about, which is the one outcome
// worse than no number.
func validateTopology(t resolvedTest, targets []string, nodeCap int) error {
	switch t.spec.Scope {
	case burninv1alpha1.ScopePair:
		return validatePairTopology(t, targets, nodeCap)
	case burninv1alpha1.ScopeGroup:
		return validateGroupTopology(t, targets, nodeCap)
	}
	return nil
}

// validateGroupTopology enforces what a Group-scope test needs from the run.
//
// A Group runs on EVERY target, one rank each, so there is no target-count rule
// beyond needing enough nodes to be a collective at all — and there is no
// upper bound, because the target list IS the group. What there is instead is
// the interlock, which bites much harder here than at Pair scope: a group holds
// all N of its nodes for its whole duration, so it needs N of the cap's slots,
// and at any smaller cap it can never start.
func validateGroupTopology(t resolvedTest, targets []string, nodeCap int) error {
	if len(targets) < 2 {
		return fmt.Errorf(
			"test %q is Group scope and needs at least two target nodes, but this run's target resolved to %d (%v) — "+
				"a collective has no meaning with fewer; name more nodes in spec.target.nodeNames, or use a nodeSelector "+
				"that matches them",
			t.name, len(targets), targets)
	}
	if dup := firstDuplicate(targets); dup != "" {
		return fmt.Errorf(
			"test %q is Group scope but node %q appears in the target list more than once — every rank must be a "+
				"DISTINCT node, or two ranks would contend for the same hardware and the collective would measure "+
				"that contention instead of the fabric",
			t.name, dup)
	}
	// NO SHIPPED RUNNER IMAGE SPEAKS THE GROUP CONTRACT YET (#118), and a Group
	// test that falls back to a default image does not fail — it SKIPS, which is
	// the worst available outcome.
	//
	// Every fabric runner this project publishes branches on BURNIN_ROLE and
	// treats its absence as "this is a Node-scope run, and a link test with one
	// node has no meaning": exit 2, with a declared _SKIP marker. At Group scope
	// the operator sets BURNIN_RANK/BURNIN_NRANKS and deliberately NOT
	// BURNIN_ROLE, so all three runners take that branch. The result is recorded
	// as Skip — "acceptance does not apply to this hardware" — and the run
	// settles Passed around a collective that never ran.
	//
	// Newer images can and will refuse honestly, but they cannot help here:
	// published tags are IMMUTABLE, so every v0.3.0 image already in the field
	// will behave this way forever. The only thing that protects a fleet running
	// those is the operator declining to dispatch them, which is what this does.
	//
	// It refuses ONLY the default. An explicit spec.runner.image is the author
	// saying they have a runner that speaks the contract, and this operator has
	// no business second-guessing that — it is also the documented way to run
	// Group scope today.
	if t.spec.Runner == nil || t.spec.Runner.Image == "" {
		return fmt.Errorf(
			"test %q is Group scope and names no spec.runner.image, so it would fall back to the "+
				"default image for kind %q — and no runner image this project has published speaks the "+
				"Group rendezvous contract (issue #118). Every published fabric runner branches on "+
				"BURNIN_ROLE, which is deliberately unset at Group scope, and treats its absence as a "+
				"Node-scope run: it would exit 2 with a skip marker, this run would record "+
				"\"acceptance does not apply to this hardware\", and it would settle Passed around a "+
				"collective that never executed. Set spec.runner.image to a runner that reads "+
				"BURNIN_RANK, BURNIN_NRANKS and BURNIN_ROOT_HOST",
			t.name, t.spec.Kind)
	}
	if nodeCap < len(targets) {
		return fmt.Errorf(
			"test %q is Group scope across %d nodes but spec.maxConcurrentNodes is %d — a group is one indivisible "+
				"unit of load that holds EVERY one of its nodes for the whole test, so it needs %d of the cap's slots "+
				"and can never start at %d; set spec.maxConcurrentNodes to at least %d, having checked that %d nodes "+
				"at full load is within the room's power and cooling headroom",
			t.name, len(targets), nodeCap, len(targets), nodeCap, len(targets), len(targets))
	}
	return nil
}

// firstDuplicate returns the first node name that appears twice, or "".
func firstDuplicate(targets []string) string {
	seen := map[string]bool{}
	for _, n := range targets {
		if seen[n] {
			return n
		}
		seen[n] = true
	}
	return ""
}

// validatePairTopology enforces what a Pair-scope test needs from the run.
func validatePairTopology(t resolvedTest, targets []string, nodeCap int) error {
	if t.spec.Scope != burninv1alpha1.ScopePair {
		return nil
	}
	if len(targets) != 2 {
		return fmt.Errorf(
			"test %q is Pair scope and needs exactly two target nodes, but this run's target resolved to %d (%v) — "+
				"a point-to-point fabric test measures the link between two endpoints and has no meaning with %d; "+
				"name exactly two nodes in spec.target.nodeNames, or use a nodeSelector that matches exactly two",
			t.name, len(targets), targets, len(targets))
	}
	if targets[0] == targets[1] {
		return fmt.Errorf(
			"test %q is Pair scope but both target nodes are %q — a link test needs two DISTINCT endpoints; "+
				"a node paired with itself would measure a loopback, not the fabric",
			t.name, targets[0])
	}
	if nodeCap < 2 {
		return fmt.Errorf(
			"test %q is Pair scope but spec.maxConcurrentNodes is %d — a pair is one indivisible unit of load that "+
				"holds BOTH of its nodes for the whole test, so it needs two of the cap's slots and can never start at %d; "+
				"set spec.maxConcurrentNodes to at least 2, having checked that two nodes at full load is within the "+
				"room's power and cooling headroom",
			t.name, nodeCap, nodeCap)
	}
	return nil
}

// validateHostPaths refuses a host mount that would produce a pod the apiserver
// rejects.
//
// The CRD already enforces most of this — MountPath is the list's map key, and
// both paths carry an absolute-path pattern — so on a real cluster this is a
// backstop. It matters anyway because the failure it prevents has no other exit:
// an invalid pod spec is rejected on Create, the reconcile returns an error, and
// the run retries the same rejection forever while holding a cordon. Refusing at
// START instead turns that into one legible sentence at run start, which is the
// same trade validatePairTopology makes.
func validateHostPaths(t resolvedTest) error {
	if t.spec.Runner == nil {
		return nil
	}
	byMountPath := map[string]bool{}
	for _, m := range t.spec.Runner.HostPaths {
		switch {
		case m.Path == "" || m.MountPath == "":
			return fmt.Errorf(
				"test %q declares a host mount with an empty path (host %q -> container %q) — "+
					"both ends of a host mount have to be written down before this operator will grant it",
				t.name, m.Path, m.MountPath)
		case !strings.HasPrefix(m.Path, "/") || !strings.HasPrefix(m.MountPath, "/"):
			return fmt.Errorf(
				"test %q declares a host mount with a relative path (host %q -> container %q) — "+
					"both must be absolute; a relative host path names nothing in particular on a node",
				t.name, m.Path, m.MountPath)
		case byMountPath[m.MountPath]:
			return fmt.Errorf(
				"test %q declares two host mounts at container path %q — a duplicate mount point is an invalid "+
					"pod spec, so every attempt would be refused by the apiserver while the run held the node",
				t.name, m.MountPath)
		}
		byMountPath[m.MountPath] = true
	}
	return nil
}

// ─── The threshold linter's surface ───────────────────────────────────────────
//
// pkg/verdict.ValidateThresholds has existed since the epsilon question was
// settled, and the answer to that question — no tolerance, the guidance is
// delivered at authoring time instead — rests on an author actually being
// warned. Until this ran, nothing called it. The two severities land in
// different places here, and the split is the point:
//
//   - MALFORMED refuses the run, as a config Error, before a node is cordoned.
//     A threshold that cannot be evaluated at all fails closed on every node
//     forever, and a fleet-wide permanent failure reported as a hardware verdict
//     is the exact confusion `Error` is kept distinct from `Fail` to prevent. It
//     is a broken profile; the honest phase is the one that says our input was
//     wrong, not the one that says the hardware was.
//   - UNSOUND records a condition and runs anyway. It is a suspicion, not a
//     proof — the gate evaluates, and it may well be doing what its author
//     wanted. Refusing on suspicion would let this operator veto profiles it
//     merely does not understand, which is a worse failure than a noisy note.

// maxAdvisedThresholds bounds how many gates a report about thresholds spells
// out — the refusal and the condition both. The count is always exact even when
// the details are elided: a profile gating forty metrics must not produce a
// status write the apiserver refuses, and a truncated list that lied about how
// much it truncated would be worse than no list.
const maxAdvisedThresholds = 5

// refuseUnsatisfiableThresholds rejects a profile carrying a gate that no run
// could satisfy.
//
// It reports EVERY malformed threshold across every test rather than the first,
// because the caller is about to be told to go and fix a file: rediscovering the
// second typo on the next run costs another cordon, another schedule, and
// another wait.
func refuseUnsatisfiableThresholds(tests []resolvedTest) error {
	var bad []string
	for _, t := range tests {
		for _, p := range verdict.ValidateThresholdsForKind(t.spec.Kind, t.spec.Thresholds) {
			if p.Severity != verdict.SeverityMalformed {
				continue
			}
			bad = append(bad, fmt.Sprintf("test %q %s", t.name, p.Error()))
		}
	}
	if len(bad) == 0 {
		return nil
	}
	shown := bad
	if len(bad) > maxAdvisedThresholds {
		shown = append(append([]string{}, bad[:maxAdvisedThresholds]...),
			fmt.Sprintf("and %d more", len(bad)-maxAdvisedThresholds))
	}
	return fmt.Errorf(
		"this profile carries %d threshold(s) that can never be satisfied, so the run is refused before any node is touched — "+
			"every one of them would otherwise fail on every node, forever, and be reported in the same shape as a hardware "+
			"verdict: %s",
		len(bad), strings.Join(shown, "; "))
}

// thresholdAdvice is one unsound gate, carrying the test it was found in so an
// operator reading a run's status knows which file to open.
type thresholdAdvice struct {
	test    string
	problem verdict.Problem
}

// unsoundThresholds lints the PINNED plan.
//
// Pinned, not the live profile, for the same reason everything else downstream
// reads the plan: the advisory has to describe the thresholds that actually
// decided this run's verdict. It is also what lets the answer be recomputed
// identically on the recovery path, after a crash between pinning the plan and
// writing the run's status.
func (p *plan) unsoundThresholds() []thresholdAdvice {
	var out []thresholdAdvice
	for _, t := range p.Tests {
		for _, problem := range verdict.ValidateThresholdsForKind(t.Spec.Kind, t.Spec.Thresholds) {
			if problem.Severity != verdict.SeverityUnsound {
				continue
			}
			out = append(out, thresholdAdvice{test: t.Name, problem: problem})
		}
	}
	return out
}

// summariseThresholdAdvice renders the condition message.
func summariseThresholdAdvice(advice []thresholdAdvice) string {
	shown := advice
	if len(shown) > maxAdvisedThresholds {
		shown = shown[:maxAdvisedThresholds]
	}
	parts := make([]string, 0, len(shown)+1)
	for _, a := range shown {
		parts = append(parts, fmt.Sprintf("test %q %s", a.test, a.problem.Error()))
	}
	if elided := len(advice) - len(shown); elided > 0 {
		parts = append(parts, fmt.Sprintf("and %d more", elided))
	}

	gates := "gates"
	if len(advice) == 1 {
		gates = "gate"
	}
	return fmt.Sprintf(
		"%d %s evaluate but may not mean what they say; the run proceeds and these do not affect its verdict: %s",
		len(advice), gates, strings.Join(parts, "; "))
}

// pinPlan serialises the plan into the run's annotations (caller persists).
func pinPlan(run *burninv1alpha1.BurnInRun, p *plan) error {
	raw, err := json.Marshal(p)
	if err != nil {
		return fmt.Errorf("marshal plan: %w", err)
	}
	if len(raw) > maxPlanBytes {
		return fmt.Errorf("resolved plan is %d bytes, over the %d limit — split the profile", len(raw), maxPlanBytes)
	}
	if run.Annotations == nil {
		run.Annotations = map[string]string{}
	}
	run.Annotations[planAnnotation] = string(raw)
	return nil
}

// loadPlan reads the pinned plan. Absent means the run has not started.
func loadPlan(run *burninv1alpha1.BurnInRun) (*plan, bool, error) {
	raw, ok := run.Annotations[planAnnotation]
	if !ok {
		return nil, false, nil
	}
	var p plan
	if err := json.Unmarshal([]byte(raw), &p); err != nil {
		return nil, true, fmt.Errorf("pinned plan does not decode: %w", err)
	}
	if p.Version != 1 {
		return nil, true, fmt.Errorf("pinned plan has unknown version %d", p.Version)
	}
	return &p, true, nil
}

// requiredByPlan maps test name to its pinned required flag.
func (p *plan) requiredByName() map[string]bool {
	m := make(map[string]bool, len(p.Tests))
	for _, t := range p.Tests {
		m[t.Name] = t.Required
	}
	return m
}

// checkpointInterval is the run-level checkpoint cadence: the shortest positive
// interval any pinned test asks for, or zero when none does.
//
// One cadence for the whole run, rather than one per test, because a checkpoint
// delivery is a CUMULATIVE snapshot of the run — it carries every result
// recorded so far, not just the test that happened to be due. Taking the
// shortest interval means the test with the tightest evidence requirement gets
// the resolution it asked for, and every other test is merely sampled more
// often than it demanded, which costs a log read and can never lose data.
//
// It reads from the PINNED plan, so the cadence a run checkpoints at cannot be
// changed underneath it by editing a BurnInTest mid-flight.
func (p *plan) checkpointInterval() time.Duration {
	var out time.Duration
	for i := range p.Tests {
		d := checkpointInterval(&p.Tests[i].Spec)
		if d > 0 && (out == 0 || d < out) {
			out = d
		}
	}
	return out
}

// ─── Resolved knobs ───────────────────────────────────────────────────────────
//
// Each of these re-applies the default declared by the CRD's kubebuilder
// marker. The reconciler must not assume apiserver defaulting has happened: an
// object created before the field existed, or one built directly in a test,
// still has to be executed under the policy the field documentation promises.
// For MaxConcurrentNodes in particular, a nil that fell through as "no cap"
// would turn the facility interlock off exactly where it is least visible.

// repeatsRequired is how many passing executions a test owes on each node.
func repeatsRequired(spec *burninv1alpha1.BurnInTestSpec) int32 {
	if spec.RepeatCount == nil || *spec.RepeatCount < 1 {
		return 1
	}
	return *spec.RepeatCount
}

// checkpointInterval is how often one test publishes its in-progress metrics.
// Zero disables checkpointing for that test.
func checkpointInterval(spec *burninv1alpha1.BurnInTestSpec) time.Duration {
	if spec.CheckpointIntervalSeconds == nil || *spec.CheckpointIntervalSeconds <= 0 {
		return 0
	}
	return time.Duration(*spec.CheckpointIntervalSeconds) * time.Second
}

// maxConcurrentNodes is the facility power/cooling interlock: how many nodes
// this run may hold under test load at once.
//
// It is read LIVE from the run's spec on every pass, and deliberately not
// pinned into the plan the way the profile is. The plan is pinned so that a
// verdict describes a fixed set of tests against a fixed set of nodes; this is
// not part of that description, it is a standing instruction about the room. An
// operator lowering it during a power event needs it to take effect on the next
// node the run would have started, not on the next run.
//
// It never widens on its own: a nil or out-of-range value is 1.
func maxConcurrentNodes(run *burninv1alpha1.BurnInRun) int {
	if run.Spec.MaxConcurrentNodes == nil || *run.Spec.MaxConcurrentNodes < 1 {
		return 1
	}
	return int(*run.Spec.MaxConcurrentNodes)
}

// retryOnErrorLimit is how many extra attempts an ERRORED execution gets.
//
// Read live for the same reason as the cap, and floored at zero. It is applied
// only ever to the Error phase; see completeAttempt, which is where the rule
// that a Failed attempt is never re-run actually lives.
func retryOnErrorLimit(run *burninv1alpha1.BurnInRun) int32 {
	if run.Spec.RetryOnErrorLimit == nil || *run.Spec.RetryOnErrorLimit < 0 {
		return 0
	}
	return *run.Spec.RetryOnErrorLimit
}

// runDeadline is the whole-run time bound, measured from status.startedAt.
// Zero means unbounded.
func runDeadline(run *burninv1alpha1.BurnInRun) time.Duration {
	if run.Spec.DeadlineSeconds == nil || *run.Spec.DeadlineSeconds <= 0 {
		return 0
	}
	return time.Duration(*run.Spec.DeadlineSeconds) * time.Second
}

// suspended reports whether the run is paused. A suspended run launches nothing
// new, reaches no terminal phase, and produces no verdict.
func suspended(run *burninv1alpha1.BurnInRun) bool {
	return run.Spec.Suspend != nil && *run.Spec.Suspend
}

// cancelRequested reports whether spec.cancel is set.
func cancelRequested(run *burninv1alpha1.BurnInRun) bool {
	return run.Spec.Cancel != nil && *run.Spec.Cancel
}
