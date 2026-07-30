package controller

import (
	"encoding/json"
	"fmt"

	burninv1alpha1 "github.com/baldwinSPC/glimmer-burnin/api/v1alpha1"
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
func buildPlan(profile *burninv1alpha1.BurnInProfile, tests []resolvedTest, targets []string) (*plan, error) {
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
		p.Tests = append(p.Tests, plannedTest{Name: t.name, Required: t.required, Spec: t.spec})
	}
	return p, nil
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
