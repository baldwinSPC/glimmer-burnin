package v1alpha1

// Cordon safety.
//
// Burn-in makes a node unusable while it runs: the tests saturate the
// accelerators, the memory and the fabric on purpose, so any workload
// scheduled alongside them gets starved, and — worse — corrupts the
// measurement it is competing with. So the operator cordons a node before
// testing it and uncordons it after.
//
// That is a destructive operation on an object the operator does not own, and
// it has exactly two ways to go wrong. Both are guarded here rather than in
// controller-local strings, because the controller and its tests have to agree
// on the values byte for byte: a constant that only the controller knows is a
// constant no test can prove the controller honours.
//
//  1. UNCORDONING A NODE SOMEBODY ELSE CORDONED. An administrator draining a
//     node for a firmware update, or another controller holding it out of
//     service, must not have that undone because a burn-in happened to finish.
//     Returning a node to the scheduler that a human deliberately removed is
//     how a node under maintenance receives production traffic mid-update.
//
//  2. LEAVING A NODE CORDONED FOREVER. A run that is deleted, cancelled, or
//     dies with its controller between cordon and cleanup silently removes
//     capacity from the fleet with no record of why. Nothing ever puts it
//     back, because nothing knows it was taken.
//
// The scheme below addresses both: an ownership stamp so only the run that
// cordoned a node may uncordon it, a record of the node's prior state so the
// uncordon restores rather than assumes, and a finalizer so cleanup runs even
// when the run is deleted.
const (
	// AnnotationCordonOwner is written on a NODE, naming the BurnInRun that
	// cordoned it, as "<namespace>/<name>/<uid>".
	//
	// Uncordon is permitted only when this annotation is present AND names the
	// run doing the uncordoning. Absent means the node was cordoned by
	// somebody else and must be left alone. The UID is part of the value
	// because names are reusable: a deleted-and-recreated run of the same name
	// is a different run, and must not inherit authority over a node the
	// previous one cordoned.
	//
	// The annotation is written BEFORE the node is cordoned, never after. The
	// window between cordoning and stamping is the window in which a crash
	// leaves a cordoned node with no owner and nothing able to claim it; in
	// the other order the worst case is a harmless stamp on an uncordoned node.
	AnnotationCordonOwner = "burnin.glimmer.ai/cordon-owner"

	// AnnotationPriorUnschedulable is written on a NODE, recording
	// spec.unschedulable as it was found BEFORE the run touched it.
	//
	// It exists because "uncordon" is not the inverse of "cordon" when the
	// node was already cordoned: setting unschedulable=false unconditionally
	// would return an administratively drained node to service on the way out
	// of a burn-in that never should have owned its schedulability at all.
	// With this recorded, cleanup restores the prior value — "true" means
	// leave it cordoned.
	//
	// The value is COPIED from the run's own start-time record
	// (BurnInRunStatus.PriorUnschedulable) and is never derived from the node
	// at the moment of cordoning. A node is cordoned and released once per
	// wave, so deriving it here would re-ask "was this already cordoned?" after
	// the run had a footprint of its own, and any cordon the run could not
	// attribute to itself would be latched as pre-existing and made permanent
	// at teardown — a stranded cordon that looks deliberate. The annotation
	// remains the account a human reads off the node, and the fallback for a
	// run that started under an operator version with no record of its own.
	//
	// Values are the literal strings "true" and "false"; see
	// PriorUnschedulableTrue and PriorUnschedulableFalse.
	AnnotationPriorUnschedulable = "burnin.glimmer.ai/prior-unschedulable"

	// AnnotationHeatExpected is written on a NODE, declaring that a burn-in is
	// putting deliberate thermal load on it right now, as
	// "<namespace>/<name>/<uid>/<RFC3339 expiry>".
	//
	// It exists because the two safety systems on this fleet were fighting each
	// other. A separate fleet agent runs a thermal watchdog that drains a node
	// when it passes its trip point, and a thermal soak drives the part past that
	// point BY DESIGN — so every soak was drained, its runner pods SIGKILLed,
	// and the hardware recorded as "Error — hardware unjudged". The watchdog was
	// working; it simply had no way to tell a soak from a cooling failure. This
	// annotation is that way: the operator declares the heat, the agent gates
	// its drain on the declaration. See issue #280.
	//
	// THE VALUE CARRIES AN ABSOLUTE EXPIRY, AND THAT IS WHAT MAKES IT SAFE TO
	// SHIP. The reader is on the node and knows nothing about whether the run is
	// still alive. A bare marker left behind by an operator that crashed, lost
	// its lease, or was deleted mid-flight would disable thermal protection on
	// that node forever, and the failure would be silent until the part cooked.
	// With an expiry the worst case is bounded by one pod's window: the reader
	// refuses to honour a marker whose expiry has passed, so a marker nothing
	// ever removes stops mattering on its own.
	//
	// It is written, refreshed and removed on exactly the cordon's lifecycle,
	// in the same updates, for the same reasons and under the same ownership
	// rule — the UID is here for the reason it is on AnnotationCordonOwner, so
	// a run that never placed this marker can never remove it. A run whose pods
	// outlive one window re-stamps a later expiry as it goes; the marker never
	// grants more than the window it can currently justify.
	//
	// THE KEY MUST MATCH THE AGENT'S READER BYTE FOR BYTE. A mismatch fails
	// OPEN — the drain still happens, the pod still dies — and looks exactly
	// like the bug this fixes.
	AnnotationHeatExpected = "burnin.glimmer.ai/heat-expected"

	// PriorUnschedulableTrue means the node was already unschedulable when the
	// run found it, and must be left unschedulable on cleanup.
	PriorUnschedulableTrue = "true"

	// PriorUnschedulableFalse means the node was schedulable when the run found
	// it, so cleanup must return it to the scheduler.
	PriorUnschedulableFalse = "false"

	// FinalizerCordonCleanup is placed on a BURNINRUN so that deleting the run
	// cannot strand the nodes it cordoned.
	//
	// Without it, `kubectl delete burninrun` — the most natural reaction to a
	// run behaving badly, and therefore the likeliest moment for this to
	// happen — removes the only object that knows which nodes are being held,
	// and the fleet quietly loses them. The finalizer makes cleanup part of
	// deletion instead of a thing that happens beforehand if you remember.
	//
	// It must be added before the first node is cordoned and removed only
	// after the last one is released, including on the cancel path. Cleanup
	// has to be idempotent and has to tolerate a node that no longer exists:
	// a run holding a finalizer for a deleted node would be undeletable, which
	// trades a stuck node for a stuck object.
	FinalizerCordonCleanup = "burnin.glimmer.ai/cordon-cleanup"
)
