package controller

import (
	"bytes"
	"compress/gzip"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"io"
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
	// Baseline records that this run MEASURES rather than certifies. Pinned like
	// everything else the verdict is read against, so a consumer reading a stored
	// run months later cannot be misled by a spec that has since been edited.
	Baseline bool `json:"baseline,omitempty"`
	// ProfileEntries is how many tests the profile LISTED, before variant
	// expansion. Pinned so the plan can say how much it grew without re-reading
	// a profile that may have changed since.
	ProfileEntries int `json:"profileEntries,omitempty"`
	// Tests are the materialised test specs, in execution order.
	Tests []plannedTest `json:"tests"`
}

// variantSummary is what the profile ASKED FOR versus what the plan holds.
//
// Reported because the arithmetic is invisible in the profile: one entry with
// four precision variants is four executions, and across eight nodes at a
// concurrency cap of one it is thirty-two sequential ones. The usual way to
// discover that is to wait for it.
type variantSummary struct {
	// Entries is how many tests the profile lists.
	Entries int
	// Executions is how many the plan holds after expansion.
	Executions int
	// Expanded names the entries that became more than one, and how many.
	Expanded []string
}

func (p *plan) variantSummary() variantSummary {
	sum := variantSummary{Entries: p.ProfileEntries, Executions: len(p.Tests)}
	counts := map[string]int{}
	var order []string
	for _, t := range p.Tests {
		// Keyed on Parent, which expansion recorded. Not on Axes: a variant may
		// legitimately declare none and still be a cell, and not on the name,
		// because splitting it on a hyphen is wrong the moment a test is called
		// "gemm-sweep".
		if t.Parent == "" {
			continue
		}
		if counts[t.Parent] == 0 {
			order = append(order, t.Parent)
		}
		counts[t.Parent]++
	}
	for _, name := range order {
		sum.Expanded = append(sum.Expanded, fmt.Sprintf("%s (%d cells)", name, counts[name]))
	}
	return sum
}

type plannedTest struct {
	Name     string                        `json:"name"`
	Required bool                          `json:"required"`
	Spec     burninv1alpha1.BurnInTestSpec `json:"spec"`
	// Axes are the variant labels this execution came from, pinned like the rest
	// of the spec so editing the profile mid-run cannot change what an in-flight
	// cell reports it was.
	Axes map[string]string `json:"axes,omitempty"`
	// Parent is the profile entry this cell was expanded from, empty when the
	// entry was not expanded. Recorded rather than recovered from Name; see
	// resolvedTest.parent.
	Parent string `json:"parent,omitempty"`
}

// buildPlan materialises a profile against resolved targets, validating the
// invariants the rest of the controller depends on.
//
// nodeCap is the run's resolved MaxConcurrentNodes. It is validated here even
// though it is read live elsewhere, because a Pair-scope test under a cap of 1
// can never launch, and a run that quietly waits out its deadline to say so is
// a much worse report than one that refuses at start with the reason.
func buildPlan(profile *burninv1alpha1.BurnInProfile, tests []resolvedTest, targets []string, nodeCap int, baseline bool) (*plan, error) {
	if err := refuseUnsatisfiableThresholds(tests); err != nil {
		return nil, err
	}
	if err := refuseGatedBaseline(tests, baseline); err != nil {
		return nil, err
	}
	seen := map[string]bool{}
	p := &plan{
		Version:  1,
		Targets:  targets,
		Sinks:    profile.Spec.Sinks,
		FailFast: profile.Spec.FailFast,
		Baseline: baseline,
		// Pinned like everything else here: the advisory it feeds describes the
		// plan this run is executing, not whatever the profile says later.
		ProfileEntries: len(profile.Spec.Tests),
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
		p.Tests = append(p.Tests, plannedTest{
			Name: t.name, Required: t.required, Spec: t.spec,
			// COPIED, not shared. The plan is pinned precisely so that editing
			// the profile mid-run changes nothing in flight, and a map handed
			// over by reference would leave a live pointer back into the thing
			// the pinning exists to isolate it from.
			Axes:   copyAxes(t.axes),
			Parent: t.parent,
		})
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
	if (t.spec.Runner == nil || t.spec.Runner.Image == "") && !groupCapableKinds[t.spec.Kind] {
		return fmt.Errorf(
			"test %q is Group scope and names no spec.runner.image, so it would fall back to the "+
				"default image for kind %q — and that image does not speak the Group rendezvous "+
				"contract (issue #118). It branches on BURNIN_ROLE, which is deliberately unset at "+
				"Group scope, and reads its absence as a Node-scope run: it would exit 2 with a skip "+
				"marker, this run would record \"acceptance does not apply to this hardware\", and it "+
				"would settle Passed around a collective that never executed. Set spec.runner.image "+
				"to a runner that reads BURNIN_RANK, BURNIN_NRANKS and BURNIN_ROOT_HOST — or use "+
				"kind %q, whose shipped runner does",
			t.name, t.spec.Kind, burninv1alpha1.KindNCCL)
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

// groupCapableKinds are the kinds whose SHIPPED runner image speaks the Group
// rendezvous — BURNIN_RANK, BURNIN_NRANKS, BURNIN_ROOT_HOST — rather than
// branching on BURNIN_ROLE and reading its absence as a Node-scope run.
//
// It is a property of the IMAGE and not of the kind, which is why this list is
// short and why adding to it is a claim rather than bookkeeping. Every other
// published fabric runner refuses Group scope outright, and correctly: a
// point-to-point RDMA write and a GPUDirect peer-memory check are measurements
// of a LINK, and have no N-rank form to implement.
//
// THE ENTRY IS ONLY TRUE FROM THE RELEASE THAT SHIPS IT. Published tags are
// immutable, so a profile pinning an older nccl image gets the old behaviour —
// exit 2 with a skip marker, a collective certified as inapplicable. That is the
// case an explicit spec.runner.image exists for, and the reason this check
// refuses the DEFAULT rather than every unpinned test: the default moves with
// the operator, and the operator and its default images ship together.
var groupCapableKinds = map[burninv1alpha1.TestKind]bool{
	burninv1alpha1.KindNCCL: true,
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

// maxDecompressedPlanBytes caps what a compressed annotation may expand to.
//
// maxPlanBytes bounds the STORED bytes, which is the real annotation budget.
// This bounds the other end: gzip expands, and a malformed or hostile annotation
// that decompresses without bound would take the manager down for every run in
// the cluster, not just the one that carries it. 4 MiB is far above any real
// plan — a hundred variants of a large spec is a few hundred KiB — and far below
// anything that matters to a process.
const maxDecompressedPlanBytes = 4 * 1024 * 1024

// pinPlan serialises the plan into the run's annotations (caller persists).
//
// # Why this may compress
//
// The pinned plan is what makes a run hermetic: it is written in the same write
// that adds the finalizer, and everything afterwards reads the pinned copy, so
// editing a BurnInTest mid-run cannot change what an in-flight attempt does.
// That property is load-bearing and is not traded away here.
//
// It is also a full copy of every test's spec, in an annotation sharing a 256
// KiB budget with the run's own churn. Variant expansion multiplies that by the
// number of cells, and the copies are NEARLY IDENTICAL — which is exactly the
// input a compressor is best at, and why compression is the right first answer
// rather than a delta format.
//
// A small plan is still written raw, so `kubectl get -o yaml` shows something a
// human can read. Only a plan that would not otherwise fit is compressed.
func pinPlan(run *burninv1alpha1.BurnInRun, p *plan) error {
	raw, err := json.Marshal(p)
	if err != nil {
		return fmt.Errorf("marshal plan: %w", err)
	}

	stored := string(raw)
	if len(raw) > maxPlanBytes {
		packed, err := compressPlan(raw)
		if err != nil {
			return fmt.Errorf("compress plan: %w", err)
		}
		if len(packed) > maxPlanBytes {
			return fmt.Errorf(
				"resolved plan is %d bytes and %d compressed, over the %d limit — split the "+
					"profile, or reduce the number of variants", len(raw), len(packed), maxPlanBytes)
		}
		stored = packed
	}

	if run.Annotations == nil {
		run.Annotations = map[string]string{}
	}
	run.Annotations[planAnnotation] = stored
	return nil
}

// loadPlan reads the pinned plan. Absent means the run has not started.
//
// The format is SNIFFED rather than versioned, and deliberately: a value
// starting with '{' is raw JSON and is parsed exactly as it always was, so a run
// pinned by an older controller and reconciled by this one is read correctly
// with no migration and no flag day.
//
// The reverse direction does not work, and that is the right way round to fail:
// a run pinned by THIS controller and then reconciled by an OLDER one gets a
// JSON parse error, which finalizes the run as Error rather than misreading the
// plan. A downgrade hazard worth writing down, not a silent one.
func loadPlan(run *burninv1alpha1.BurnInRun) (*plan, bool, error) {
	raw, ok := run.Annotations[planAnnotation]
	if !ok {
		return nil, false, nil
	}

	data := []byte(raw)
	if !isRawJSONPlan(raw) {
		var err error
		data, err = decompressPlan(raw)
		if err != nil {
			return nil, true, fmt.Errorf("pinned plan does not decode: %w", err)
		}
	}

	var p plan
	if err := json.Unmarshal(data, &p); err != nil {
		return nil, true, fmt.Errorf("pinned plan does not decode: %w", err)
	}
	if p.Version != 1 {
		return nil, true, fmt.Errorf("pinned plan has unknown version %d", p.Version)
	}
	return &p, true, nil
}

// isRawJSONPlan is the format sniff: legacy plans are a JSON object, and the
// compressed form is base64, which never begins with '{'.
func isRawJSONPlan(s string) bool {
	return strings.HasPrefix(strings.TrimLeft(s, " \t\r\n"), "{")
}

func compressPlan(raw []byte) (string, error) {
	var buf bytes.Buffer
	// BestCompression, not Default: this runs once per run, and the byte it
	// saves is a byte of a hard annotation budget.
	zw, err := gzip.NewWriterLevel(&buf, gzip.BestCompression)
	if err != nil {
		return "", err
	}
	if _, err := zw.Write(raw); err != nil {
		return "", err
	}
	if err := zw.Close(); err != nil {
		return "", err
	}
	return base64.StdEncoding.EncodeToString(buf.Bytes()), nil
}

func decompressPlan(s string) ([]byte, error) {
	packed, err := base64.StdEncoding.DecodeString(strings.TrimSpace(s))
	if err != nil {
		return nil, fmt.Errorf("not raw JSON and not base64: %w", err)
	}
	zr, err := gzip.NewReader(bytes.NewReader(packed))
	if err != nil {
		return nil, fmt.Errorf("not gzip: %w", err)
	}
	defer func() { _ = zr.Close() }()

	// LimitReader with ONE byte of headroom, so exceeding the cap is detectable
	// rather than silently truncating a plan into something that parses.
	out, err := io.ReadAll(io.LimitReader(zr, maxDecompressedPlanBytes+1))
	if err != nil {
		return nil, err
	}
	if len(out) > maxDecompressedPlanBytes {
		return nil, fmt.Errorf(
			"decompresses to more than %d bytes; refusing to expand it further",
			maxDecompressedPlanBytes)
	}
	return out, nil
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

// isBaseline reports whether this run measures rather than certifies. Read from
// the spec at start and then PINNED into the plan, because it is part of what a
// verdict means and must not change under a run that is already executing.
func isBaseline(run *burninv1alpha1.BurnInRun) bool {
	return run.Spec.Baseline != nil && *run.Spec.Baseline
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

// refuseGatedBaseline rejects a baseline run whose profile carries a threshold.
//
// A REFUSAL AND NOT A SUPPRESSION, and the distinction is the whole point. Making
// spec.baseline suppress thresholds would turn it into a threshold-laundering
// switch: a way to convert a failing acceptance run into a passing measurement
// run by flipping a boolean, after the fact, with the profile unchanged. This
// operator's premise is that a verdict cannot be edited into existence, and a
// flag that can do it is worse than no flag.
//
// So the two are kept mutually exclusive at the point where it costs nothing —
// before a node is cordoned — and the error names the test, because "your
// profile has gates somewhere" is not actionable on a twelve-test suite.
func refuseGatedBaseline(tests []resolvedTest, baseline bool) error {
	if !baseline {
		return nil
	}
	var gated []string
	for _, t := range tests {
		if len(t.spec.Thresholds) > 0 {
			gated = append(gated, fmt.Sprintf("%q (%d threshold(s))", t.name, len(t.spec.Thresholds)))
		}
	}
	if len(gated) == 0 {
		return nil
	}
	shown := gated
	if len(shown) > maxAdvisedThresholds {
		shown = append(append([]string{}, gated[:maxAdvisedThresholds]...),
			fmt.Sprintf("and %d more", len(gated)-maxAdvisedThresholds))
	}
	return fmt.Errorf(
		"this run is marked spec.baseline but its profile carries thresholds on %d test(s): %s. "+
			"A baseline MEASURES and an acceptance run CERTIFIES, and the flag does not suppress "+
			"gates — that would be a way to turn a failing run into a passing one by editing a "+
			"boolean. Either clear spec.baseline and let the gates decide, or point this run at a "+
			"thresholdless profile",
		len(gated), strings.Join(shown, "; "))
}
