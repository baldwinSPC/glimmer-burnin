package v1alpha1

import (
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

// ScheduleConcurrencyPolicy decides what happens when a schedule fires while a
// run it created is still going.
//
// There is deliberately no Replace option, unlike CronJob. Replace means "kill
// the in-flight one and start fresh", and for burn-in that is wrong twice
// over: it destroys expensive evidence (a soak four hours in has already cost
// the fleet those four hours) and it kills a run mid-cordon, which is exactly
// the moment the cordon-cleanup path is most likely to be interrupted. A
// burn-in is not a job that can be safely restarted from the top on a whim.
// +kubebuilder:validation:Enum=Forbid;Allow
type ScheduleConcurrencyPolicy string

const (
	// ConcurrencyForbid skips the tick when a run from this schedule is still
	// active. THIS IS THE DEFAULT AND IT IS THE NODE-BUSY GUARD.
	//
	// Burn-in is not idempotent with respect to the hardware: two runs
	// targeting the same node would fight for the accelerators, the memory
	// bandwidth and the fabric that each of them is trying to measure. The
	// result is not merely a slower run, it is a WRONG run — both report
	// degraded bandwidth and depressed clocks, and a healthy node gets a
	// failing verdict because it was tested twice at once. Then, on top of the
	// bad measurement, the two runs disagree about the node's cordon and one
	// of them can uncordon a node the other is still driving at full power.
	//
	// A skipped tick is recorded, never silently dropped, so a schedule that
	// never actually runs because its runs overtake its period is visible as
	// the misconfiguration it is rather than as a quiet absence of results.
	ConcurrencyForbid ScheduleConcurrencyPolicy = "Forbid"

	// ConcurrencyAllow lets a new run start while an earlier one is active.
	//
	// Only safe when successive ticks cannot land on the same node — disjoint
	// targets, or a fleet large enough that the selector never overlaps. It
	// does NOT make concurrent testing of one node safe; nothing does. It also
	// stacks power draw the way MaxConcurrentNodes does, so the facility
	// interlock argument on that field applies here too, multiplied by the
	// number of runs in flight.
	ConcurrencyAllow ScheduleConcurrencyPolicy = "Allow"
)

// BurnInRunTemplate is the run-shaped subset a schedule stamps onto every
// BurnInRun it creates. Each field means exactly what the identically named
// field on BurnInRunSpec means.
type BurnInRunTemplate struct {
	// TTLSecondsAfterFinished garbage-collects a created run this long after it
	// reaches a terminal phase. 0 or nil keeps it, and the history limits below
	// are then the only thing bounding accumulation.
	//
	// Prefer the history limits for retention policy and use this only for
	// hard expiry: a TTL deletes by age, so a schedule that has been failing
	// for a week can delete the failures that explain why.
	// +kubebuilder:validation:Minimum=0
	TTLSecondsAfterFinished *int32 `json:"ttlSecondsAfterFinished,omitempty"`

	// DeadlineSeconds bounds each created run. A periodic schedule especially
	// wants this: without it a wedged run can still be active when the next
	// tick arrives, and under the default Forbid policy that one stuck run
	// stops the schedule from ever firing again.
	// +kubebuilder:validation:Minimum=0
	DeadlineSeconds *int32 `json:"deadlineSeconds,omitempty"`

	// MaxConcurrentNodes is the facility power and cooling SAFETY INTERLOCK,
	// stamped onto each created run. Default 1. Read the field documentation on
	// BurnInRunSpec before changing it — an unattended schedule is a worse
	// place to raise it than a hand-launched run, because nobody is watching
	// the room at 03:00.
	// +kubebuilder:default=1
	// +kubebuilder:validation:Minimum=1
	MaxConcurrentNodes *int32 `json:"maxConcurrentNodes,omitempty"`

	// RetryOnErrorLimit re-runs ERRORED tests only, never Failed ones.
	// +kubebuilder:default=0
	// +kubebuilder:validation:Minimum=0
	RetryOnErrorLimit *int32 `json:"retryOnErrorLimit,omitempty"`
}

// BurnInScheduleSpec runs a BurnInProfile against a target on a cron schedule.
//
// Periodic burn-in exists because acceptance at install time only proves the
// hardware was good on the day it arrived. Fans clog, thermal paste degrades,
// PD supplies and cables age, DIMMs develop faults, links start retraining.
// The clock-probe failure mode in particular is invisible to every liveness
// check a cluster already runs, so nothing but a scheduled re-test will ever
// find it — the node keeps reporting healthy and simply produces less work.
type BurnInScheduleSpec struct {
	// Schedule is a cron expression in standard 5-field form
	// ("0 3 * * 0" — 03:00 every Sunday), or a "@every 168h" style interval.
	//
	// Pick a window when the fleet can afford to lose the targeted nodes:
	// every tick cordons them and drives them to their power and thermal
	// limits for the duration of the profile.
	// +kubebuilder:validation:Required
	// +kubebuilder:validation:MinLength=1
	Schedule string `json:"schedule"`

	// TimeZone is an IANA name ("America/Los_Angeles") the schedule is
	// interpreted in. Empty means UTC.
	//
	// It matters here more than it does for an ordinary job: maintenance
	// windows and the facility staff who can respond to a power event are
	// local, so a schedule that drifts by an hour twice a year can put a
	// full-power sweep outside the window it was agreed for.
	TimeZone *string `json:"timeZone,omitempty"`

	// ProfileRef names the BurnInProfile each tick runs (same namespace).
	// +kubebuilder:validation:Required
	ProfileRef string `json:"profileRef"`

	// Target selects the node(s) each created run tests. It is re-resolved at
	// each tick, so a label selector naturally covers nodes added since the
	// schedule was written — which is the point of scheduling by selector
	// rather than by name.
	// +kubebuilder:validation:Required
	Target TargetSelector `json:"target"`

	// Suspend stops new runs from being created. Runs already in flight are
	// left to finish and record their results.
	//
	// This is the intended way to stop a schedule during a facility event or a
	// maintenance freeze: deleting the object would lose the history and the
	// intent, and the freeze is temporary. Suspending does not disturb the
	// cordon state of any node — an active run still owns and releases its own.
	// +kubebuilder:default=false
	Suspend *bool `json:"suspend,omitempty"`

	// ConcurrencyPolicy decides what a tick does when a previous run is still
	// active. Defaults to Forbid, the node-busy guard.
	// +kubebuilder:default=Forbid
	ConcurrencyPolicy ScheduleConcurrencyPolicy `json:"concurrencyPolicy,omitempty"`

	// StartingDeadlineSeconds is how late a missed tick may still start. A tick
	// older than this is recorded as missed and skipped, rather than firing
	// whenever the controller comes back.
	//
	// Without it, a controller that was down over a weekend would launch a
	// full-power sweep the moment it recovers, at whatever hour that turns out
	// to be — outside the agreed maintenance window and with nobody expecting
	// the load. A burn-in is not worth catching up on: the next tick is soon
	// enough, and running it at the right time matters more than running it.
	// +kubebuilder:validation:Minimum=0
	StartingDeadlineSeconds *int32 `json:"startingDeadlineSeconds,omitempty"`

	// SuccessfulRunsHistoryLimit is how many Passed runs to keep. nil means the
	// default; 0 keeps none.
	// +kubebuilder:default=3
	// +kubebuilder:validation:Minimum=0
	SuccessfulRunsHistoryLimit *int32 `json:"successfulRunsHistoryLimit,omitempty"`

	// FailedRunsHistoryLimit is how many non-passing runs — Failed, Error and
	// Cancelled — to keep. nil means the default; 0 keeps none.
	//
	// It defaults HIGHER than the successful limit, which is the opposite of
	// CronJob's bias, and deliberately: the successful runs are the boring
	// ones, and the failing runs are the entire reason the schedule exists.
	// Diagnosing degradation means comparing the failure against the ones
	// before it — was the bandwidth sagging for three weeks, or did it fall
	// off a cliff on Tuesday? — and that comparison is impossible if the
	// evidence was garbage-collected down to the latest failure.
	// +kubebuilder:default=10
	// +kubebuilder:validation:Minimum=0
	FailedRunsHistoryLimit *int32 `json:"failedRunsHistoryLimit,omitempty"`

	// RunTemplate carries the run-shaped settings stamped onto each created run.
	RunTemplate BurnInRunTemplate `json:"runTemplate,omitempty"`
}

// BurnInScheduleStatus tracks what the schedule has launched.
type BurnInScheduleStatus struct {
	// LastScheduleTime is the tick most recently acted on. It is the cursor
	// the controller resumes from, so it must be persisted before the run is
	// created rather than after: the failure that duplicates a full-power
	// sweep across a fleet is worse than the one that misses a tick.
	LastScheduleTime *metav1.Time `json:"lastScheduleTime,omitempty"`

	// LastSuccessfulTime is when a created run last reached Passed. A schedule
	// firing on time whose runs never pass looks healthy by LastScheduleTime
	// alone; this is the field that distinguishes them.
	LastSuccessfulTime *metav1.Time `json:"lastSuccessfulTime,omitempty"`

	// Active are the runs this schedule created that have not reached a
	// terminal phase. It is what the Forbid policy consults, and it is also
	// the list that says which nodes are currently cordoned by this schedule.
	Active []corev1.ObjectReference `json:"active,omitempty"`

	// LastRunName is the most recently created BurnInRun.
	LastRunName string `json:"lastRunName,omitempty"`

	// LastRunPhase is that run's phase. Passed, Failed, Error and Cancelled are
	// four different states here and none of them may be collapsed into
	// another: Failed is a hardware verdict, Error means the schedule produced
	// no verdict at all, and Cancelled means somebody stopped it. A schedule
	// summarised as "not passing" hides which of the three is happening, and
	// they call for completely different responses.
	LastRunPhase RunPhase `json:"lastRunPhase,omitempty"`

	// MissedSchedules counts ticks skipped because a previous run was still
	// active or because the starting deadline had passed. A steadily rising
	// count means the schedule is not doing what its author thinks it is.
	// +kubebuilder:validation:Minimum=0
	MissedSchedules int32 `json:"missedSchedules,omitempty"`

	ObservedGeneration int64              `json:"observedGeneration,omitempty"`
	Conditions         []metav1.Condition `json:"conditions,omitempty"`
}

// +kubebuilder:object:root=true
// +kubebuilder:subresource:status
// +kubebuilder:resource:shortName=bisc,categories=burnin
// +kubebuilder:printcolumn:name="Schedule",type=string,JSONPath=`.spec.schedule`
// +kubebuilder:printcolumn:name="Profile",type=string,JSONPath=`.spec.profileRef`
// +kubebuilder:printcolumn:name="Suspend",type=boolean,JSONPath=`.spec.suspend`
// +kubebuilder:printcolumn:name="Last Schedule",type=date,JSONPath=`.status.lastScheduleTime`
// +kubebuilder:printcolumn:name="Last Phase",type=string,JSONPath=`.status.lastRunPhase`
// +kubebuilder:printcolumn:name="Age",type=date,JSONPath=`.metadata.creationTimestamp`

// BurnInSchedule runs a BurnInProfile against a target periodically.
//
// It is how a fleet moves from one-time acceptance to continuous verification:
// hardware degrades in service, and a node that passed on install day is not
// evidence about the node today.
type BurnInSchedule struct {
	metav1.TypeMeta   `json:",inline"`
	metav1.ObjectMeta `json:"metadata,omitempty"`
	Spec              BurnInScheduleSpec   `json:"spec,omitempty"`
	Status            BurnInScheduleStatus `json:"status,omitempty"`
}

// +kubebuilder:object:root=true

// BurnInScheduleList contains a list of BurnInSchedule.
type BurnInScheduleList struct {
	metav1.TypeMeta `json:",inline"`
	metav1.ListMeta `json:"metadata,omitempty"`
	Items           []BurnInSchedule `json:"items"`
}
