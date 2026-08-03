package controller

import (
	"context"
	"fmt"
	"strings"
	"testing"
	"time"

	"github.com/robfig/cron/v3"
	corev1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/client/fake"
	"sigs.k8s.io/controller-runtime/pkg/client/interceptor"

	burninv1alpha1 "github.com/baldwinSPC/glimmer-burnin/api/v1alpha1"
)

// Sunday 2026-08-02 00:00:00 UTC. The weekly fixtures below use "0 3 * * 0",
// so the next tick from here is 03:00 the same day.
var schedBase = time.Date(2026, 8, 2, 0, 0, 0, 0, time.UTC)

type schedHarness struct {
	t      *testing.T
	r      *BurnInScheduleReconciler
	c      client.Client
	nowVal time.Time
}

func newSchedHarness(t *testing.T, objs ...client.Object) *schedHarness {
	t.Helper()
	return newSchedHarnessWith(t, nil, objs...)
}

func newSchedHarnessWith(t *testing.T, funcs *interceptor.Funcs, objs ...client.Object) *schedHarness {
	t.Helper()
	scheme := runtime.NewScheme()
	if err := corev1.AddToScheme(scheme); err != nil {
		t.Fatal(err)
	}
	if err := burninv1alpha1.AddToScheme(scheme); err != nil {
		t.Fatal(err)
	}
	b := fake.NewClientBuilder().
		WithScheme(scheme).
		WithObjects(objs...).
		WithStatusSubresource(&burninv1alpha1.BurnInSchedule{}, &burninv1alpha1.BurnInRun{})
	if funcs != nil {
		b = b.WithInterceptorFuncs(*funcs)
	}
	c := b.Build()

	h := &schedHarness{t: t, c: c, nowVal: schedBase}
	h.r = &BurnInScheduleReconciler{
		Client: c,
		Scheme: scheme,
		Now:    func() time.Time { return h.nowVal },
	}
	return h
}

func (h *schedHarness) at(t time.Time) *schedHarness {
	h.nowVal = t
	return h
}

func (h *schedHarness) reconcile(name string) ctrl.Result {
	h.t.Helper()
	res, err := h.r.Reconcile(context.Background(),
		ctrl.Request{NamespacedName: types.NamespacedName{Namespace: "burnin", Name: name}})
	if err != nil {
		h.t.Fatalf("Reconcile: %v", err)
	}
	return res
}

func (h *schedHarness) reconcileErr(name string) error {
	h.t.Helper()
	_, err := h.r.Reconcile(context.Background(),
		ctrl.Request{NamespacedName: types.NamespacedName{Namespace: "burnin", Name: name}})
	return err
}

func (h *schedHarness) schedule(name string) *burninv1alpha1.BurnInSchedule {
	h.t.Helper()
	var s burninv1alpha1.BurnInSchedule
	if err := h.c.Get(context.Background(),
		types.NamespacedName{Namespace: "burnin", Name: name}, &s); err != nil {
		h.t.Fatalf("get schedule: %v", err)
	}
	return &s
}

// runs returns every BurnInRun owned by the named schedule.
func (h *schedHarness) runs(scheduleName string) []burninv1alpha1.BurnInRun {
	h.t.Helper()
	var list burninv1alpha1.BurnInRunList
	if err := h.c.List(context.Background(), &list, client.InNamespace("burnin")); err != nil {
		h.t.Fatalf("list runs: %v", err)
	}
	sched := h.schedule(scheduleName)
	var out []burninv1alpha1.BurnInRun
	for i := range list.Items {
		if metav1.IsControlledBy(&list.Items[i], sched) {
			out = append(out, list.Items[i])
		}
	}
	sortRunsNewestFirst(out)
	return out
}

// setRunPhase drives an owned run to a phase, as the run reconciler would.
func (h *schedHarness) setRunPhase(run *burninv1alpha1.BurnInRun, phase burninv1alpha1.RunPhase) {
	h.t.Helper()
	var fresh burninv1alpha1.BurnInRun
	if err := h.c.Get(context.Background(),
		types.NamespacedName{Namespace: run.Namespace, Name: run.Name}, &fresh); err != nil {
		h.t.Fatalf("get run: %v", err)
	}
	fresh.Status.Phase = phase
	fresh.Status.FinishedAt = &metav1.Time{Time: h.nowVal}
	if err := h.c.Status().Update(context.Background(), &fresh); err != nil {
		h.t.Fatalf("update run status: %v", err)
	}
}

type schedOpt func(*burninv1alpha1.BurnInSchedule)

func newSchedule(name, cronSpec string, opts ...schedOpt) *burninv1alpha1.BurnInSchedule {
	s := &burninv1alpha1.BurnInSchedule{
		ObjectMeta: metav1.ObjectMeta{
			Name:              name,
			Namespace:         "burnin",
			UID:               types.UID("uid-" + name),
			Generation:        1,
			CreationTimestamp: metav1.Time{Time: schedBase},
		},
		Spec: burninv1alpha1.BurnInScheduleSpec{
			Schedule:   cronSpec,
			ProfileRef: "acceptance",
			Target: burninv1alpha1.TargetSelector{
				NodeSelector: map[string]string{"glimmer.ai/accelerator": "gb10"},
			},
		},
	}
	for _, o := range opts {
		o(s)
	}
	return s
}

func withSuspend(v bool) schedOpt {
	return func(s *burninv1alpha1.BurnInSchedule) { s.Spec.Suspend = &v }
}

func withTimeZone(tz string) schedOpt {
	return func(s *burninv1alpha1.BurnInSchedule) { s.Spec.TimeZone = &tz }
}

func withConcurrency(p burninv1alpha1.ScheduleConcurrencyPolicy) schedOpt {
	return func(s *burninv1alpha1.BurnInSchedule) { s.Spec.ConcurrencyPolicy = p }
}

func withStartingDeadline(sec int32) schedOpt {
	return func(s *burninv1alpha1.BurnInSchedule) { s.Spec.StartingDeadlineSeconds = &sec }
}

func withHistoryLimits(successful, failed int32) schedOpt {
	return func(s *burninv1alpha1.BurnInSchedule) {
		s.Spec.SuccessfulRunsHistoryLimit = &successful
		s.Spec.FailedRunsHistoryLimit = &failed
	}
}

func withLastScheduleTime(t time.Time) schedOpt {
	return func(s *burninv1alpha1.BurnInSchedule) {
		s.Status.LastScheduleTime = &metav1.Time{Time: t}
	}
}

func withRunTemplate(tpl burninv1alpha1.BurnInRunTemplate) schedOpt {
	return func(s *burninv1alpha1.BurnInSchedule) { s.Spec.RunTemplate = tpl }
}

// ownedRun builds a pre-existing terminal (or active) run attributed to a
// schedule, as the reconciler itself would have created it.
func ownedRun(sched *burninv1alpha1.BurnInSchedule, name string, created time.Time, phase burninv1alpha1.RunPhase) *burninv1alpha1.BurnInRun {
	yes := true
	return &burninv1alpha1.BurnInRun{
		ObjectMeta: metav1.ObjectMeta{
			Name:              name,
			Namespace:         sched.Namespace,
			UID:               types.UID("uid-" + name),
			CreationTimestamp: metav1.Time{Time: created},
			Labels:            map[string]string{labelSchedule: sched.Name},
			OwnerReferences: []metav1.OwnerReference{{
				APIVersion: burninv1alpha1.GroupVersion.String(),
				Kind:       "BurnInSchedule",
				Name:       sched.Name,
				UID:        sched.UID,
				Controller: &yes,
			}},
		},
		Spec:   burninv1alpha1.BurnInRunSpec{ProfileRef: sched.Spec.ProfileRef},
		Status: burninv1alpha1.BurnInRunStatus{Phase: phase},
	}
}

// --- due-time computation -------------------------------------------------

func TestSchedule_FiresAtTheDueTick(t *testing.T) {
	s := newSchedule("weekly", "0 3 * * 0")
	h := newSchedHarness(t, s)

	// 02:59 — not yet due.
	res := h.at(schedBase.Add(2*time.Hour + 59*time.Minute)).reconcile("weekly")
	if got := len(h.runs("weekly")); got != 0 {
		t.Fatalf("run created before the tick was due: %d runs", got)
	}
	// Requeue must land on the tick, not poll.
	if want := time.Minute; res.RequeueAfter != want {
		t.Errorf("RequeueAfter = %v, want %v (requeue precisely at the next due time)", res.RequeueAfter, want)
	}

	// 03:00 — due.
	h.at(schedBase.Add(3 * time.Hour)).reconcile("weekly")
	runs := h.runs("weekly")
	if len(runs) != 1 {
		t.Fatalf("expected exactly 1 run at the due tick, got %d", len(runs))
	}

	got := h.schedule("weekly")
	if got.Status.LastScheduleTime == nil || !got.Status.LastScheduleTime.Time.Equal(schedBase.Add(3*time.Hour)) {
		t.Errorf("LastScheduleTime = %v, want %v", got.Status.LastScheduleTime, schedBase.Add(3*time.Hour))
	}
	if got.Status.LastRunName != runs[0].Name {
		t.Errorf("LastRunName = %q, want %q", got.Status.LastRunName, runs[0].Name)
	}
	if len(got.Status.Active) != 1 || got.Status.Active[0].Name != runs[0].Name {
		t.Errorf("Active = %+v, want the one created run", got.Status.Active)
	}

	// The next requeue is a week out, not a poll interval.
	res = h.at(schedBase.Add(3 * time.Hour)).reconcile("weekly")
	if want := 7 * 24 * time.Hour; res.RequeueAfter != want {
		t.Errorf("RequeueAfter after firing = %v, want %v", res.RequeueAfter, want)
	}
}

func TestSchedule_TimeZoneShiftsTheDueTick(t *testing.T) {
	// 03:00 America/Los_Angeles on 2026-08-02 is 10:00 UTC (PDT, UTC-7).
	s := newSchedule("weekly-la", "0 3 * * 0", withTimeZone("America/Los_Angeles"))
	h := newSchedHarness(t, s)

	// 03:00 UTC would be due for a UTC schedule; for this one it is not.
	h.at(schedBase.Add(3 * time.Hour)).reconcile("weekly-la")
	if got := len(h.runs("weekly-la")); got != 0 {
		t.Fatalf("LA schedule fired at 03:00 UTC: %d runs (timezone ignored)", got)
	}

	h.at(schedBase.Add(10 * time.Hour)).reconcile("weekly-la")
	if got := len(h.runs("weekly-la")); got != 1 {
		t.Fatalf("LA schedule did not fire at 10:00 UTC (= 03:00 PDT): %d runs", got)
	}
}

func TestSchedule_EmptyTimeZoneIsUTCNotProcessLocal(t *testing.T) {
	// The upstream parser defaults to time.Local, which would make the same
	// object fire at different wall-clock times on two clusters.
	s := newSchedule("weekly", "0 3 * * 0")
	cronSchedule, err := parseSchedule(s)
	if err != nil {
		t.Fatalf("parseSchedule: %v", err)
	}
	next := cronSchedule.Next(schedBase)
	if want := schedBase.Add(3 * time.Hour); !next.Equal(want) {
		t.Errorf("next tick = %v, want %v (UTC)", next.UTC(), want)
	}
}

func TestSchedule_NeverBurstsAfterAnOutage(t *testing.T) {
	// Cursor a week stale on an hourly schedule: 168 ticks elapsed.
	s := newSchedule("hourly", "0 * * * *", withLastScheduleTime(schedBase.Add(-7*24*time.Hour)))
	h := newSchedHarness(t, s)

	h.at(schedBase).reconcile("hourly")

	runs := h.runs("hourly")
	if len(runs) != 1 {
		t.Fatalf("outage recovery created %d runs; a burn-in backlog must never be replayed as a burst", len(runs))
	}
	// The one run must be for the most recent tick, not the oldest.
	if got := runs[0].Annotations[annotationScheduledAt]; got != schedBase.Format(time.RFC3339) {
		t.Errorf("scheduled-at = %q, want the most recent tick %q", got, schedBase.Format(time.RFC3339))
	}
	if got := h.schedule("hourly").Status.MissedSchedules; got != 167 {
		t.Errorf("MissedSchedules = %d, want 167 (the skipped ticks must be recorded, not silently dropped)", got)
	}
}

func TestSchedule_StartingDeadlineDropsStaleTicks(t *testing.T) {
	// Hourly, cursor 5h stale, deadline 90m: only ticks inside the last 90
	// minutes may run.
	s := newSchedule("hourly", "0 * * * *",
		withLastScheduleTime(schedBase.Add(-5*time.Hour)),
		withStartingDeadline(90*60))
	h := newSchedHarness(t, s)

	h.at(schedBase).reconcile("hourly")

	runs := h.runs("hourly")
	if len(runs) != 1 {
		t.Fatalf("expected 1 run, got %d", len(runs))
	}
	if got := runs[0].Annotations[annotationScheduledAt]; got != schedBase.Format(time.RFC3339) {
		t.Errorf("scheduled-at = %q, want %q", got, schedBase.Format(time.RFC3339))
	}
	// 5 ticks elapsed (-4h..0h); 4 dropped, 1 fired.
	if got := h.schedule("hourly").Status.MissedSchedules; got != 4 {
		t.Errorf("MissedSchedules = %d, want 4", got)
	}
}

func TestSchedule_StartingDeadlineCanDropEveryTick(t *testing.T) {
	// The last tick is 40 minutes old but the deadline is 10 minutes: nothing
	// may start. A burn-in outside its agreed window is worse than none.
	s := newSchedule("hourly", "0 * * * *",
		withLastScheduleTime(schedBase.Add(-90*time.Minute)),
		withStartingDeadline(10*60))
	h := newSchedHarness(t, s)

	h.at(schedBase.Add(40 * time.Minute)).reconcile("hourly")

	if got := len(h.runs("hourly")); got != 0 {
		t.Fatalf("fired %d runs past the starting deadline; want 0", got)
	}
	if got := h.schedule("hourly").Status.MissedSchedules; got != 2 {
		t.Errorf("MissedSchedules = %d, want 2", got)
	}
}

func TestSchedule_HugeBacklogResynchronisesInsteadOfScanning(t *testing.T) {
	// A minutely schedule with a cursor a year stale is far past the scan cap.
	s := newSchedule("minutely", "@every 1m", withLastScheduleTime(schedBase.Add(-365*24*time.Hour)))
	h := newSchedHarness(t, s)

	h.at(schedBase).reconcile("minutely")

	if got := len(h.runs("minutely")); got != 0 {
		t.Fatalf("resync fired %d runs; want 0", got)
	}
	got := h.schedule("minutely")
	if got.Status.LastScheduleTime == nil || !got.Status.LastScheduleTime.Time.Equal(schedBase) {
		t.Errorf("LastScheduleTime = %v, want the cursor moved to now", got.Status.LastScheduleTime)
	}
	if got.Status.MissedSchedules == 0 {
		t.Error("MissedSchedules = 0; a resync must still record that ticks were dropped")
	}

	// And the next tick fires normally.
	h.at(schedBase.Add(time.Minute)).reconcile("minutely")
	if n := len(h.runs("minutely")); n != 1 {
		t.Fatalf("after resync expected 1 run at the next tick, got %d", n)
	}
}

// --- Forbid prevents overlap ---------------------------------------------

func TestSchedule_ForbidSkipsTickWhileRunActive(t *testing.T) {
	s := newSchedule("hourly", "0 * * * *")
	h := newSchedHarness(t, s)

	h.at(schedBase.Add(time.Hour)).reconcile("hourly")
	if n := len(h.runs("hourly")); n != 1 {
		t.Fatalf("expected 1 run, got %d", n)
	}

	// Next tick, previous run still Running.
	h.setRunPhase(&h.runs("hourly")[0], burninv1alpha1.RunRunning)
	h.at(schedBase.Add(2 * time.Hour)).reconcile("hourly")

	if n := len(h.runs("hourly")); n != 1 {
		t.Fatalf("Forbid created a second run beside an active one: %d runs", n)
	}
	got := h.schedule("hourly")
	if got.Status.MissedSchedules != 1 {
		t.Errorf("MissedSchedules = %d, want 1 (a skipped tick is recorded, never dropped)", got.Status.MissedSchedules)
	}
	// The cursor must advance, or the stale tick fires the moment the run ends.
	if got.Status.LastScheduleTime == nil || !got.Status.LastScheduleTime.Time.Equal(schedBase.Add(2*time.Hour)) {
		t.Errorf("LastScheduleTime = %v, want the skipped tick %v", got.Status.LastScheduleTime, schedBase.Add(2*time.Hour))
	}

	// Once it finishes, the NEXT tick fires — not the skipped one.
	h.setRunPhase(&h.runs("hourly")[0], burninv1alpha1.RunPassed)
	h.at(schedBase.Add(3 * time.Hour)).reconcile("hourly")
	runs := h.runs("hourly")
	if len(runs) != 2 {
		t.Fatalf("expected 2 runs after the active one finished, got %d", len(runs))
	}
	if got := runs[0].Annotations[annotationScheduledAt]; got != schedBase.Add(3*time.Hour).Format(time.RFC3339) {
		t.Errorf("scheduled-at = %q, want the current tick, not a replayed one", got)
	}
}

// A run created but not yet reconciled has an EMPTY phase. If that did not
// count as active, the Forbid guard would fail in exactly the window it
// matters most: the moments right after creation.
func TestSchedule_ForbidCountsFreshlyCreatedRunAsActive(t *testing.T) {
	s := newSchedule("hourly", "0 * * * *")
	h := newSchedHarness(t, s)

	h.at(schedBase.Add(time.Hour)).reconcile("hourly")
	// Immediately reconcile again at the same instant, phase still "".
	h.at(schedBase.Add(time.Hour)).reconcile("hourly")

	if n := len(h.runs("hourly")); n != 1 {
		t.Fatalf("a run with an unset phase was not treated as active: %d runs created", n)
	}
}

func TestSchedule_ForbidIsTheDefaultForAnUnsetPolicy(t *testing.T) {
	s := newSchedule("hourly", "0 * * * *")
	if s.Spec.ConcurrencyPolicy != "" {
		t.Fatal("fixture should leave the policy unset")
	}
	if got := concurrencyPolicy(s); got != burninv1alpha1.ConcurrencyForbid {
		t.Fatalf("unset ConcurrencyPolicy resolved to %q; an unset policy must never fall through to Allow", got)
	}
	// An unrecognised value must also fail closed.
	s.Spec.ConcurrencyPolicy = burninv1alpha1.ScheduleConcurrencyPolicy("Replace")
	if got := concurrencyPolicy(s); got != burninv1alpha1.ConcurrencyForbid {
		t.Fatalf("unrecognised ConcurrencyPolicy resolved to %q, want Forbid", got)
	}
}

func TestSchedule_AllowStartsRunBesideActiveOne(t *testing.T) {
	s := newSchedule("hourly", "0 * * * *", withConcurrency(burninv1alpha1.ConcurrencyAllow))
	h := newSchedHarness(t, s)

	h.at(schedBase.Add(time.Hour)).reconcile("hourly")
	h.setRunPhase(&h.runs("hourly")[0], burninv1alpha1.RunRunning)
	h.at(schedBase.Add(2 * time.Hour)).reconcile("hourly")

	if n := len(h.runs("hourly")); n != 2 {
		t.Fatalf("Allow produced %d runs, want 2", n)
	}
	if got := h.schedule("hourly").Status.MissedSchedules; got != 0 {
		t.Errorf("MissedSchedules = %d, want 0 under Allow", got)
	}
}

// --- history GC -----------------------------------------------------------

func TestSchedule_HistoryLimitsAreCountedSeparately(t *testing.T) {
	s := newSchedule("weekly", "0 3 * * 0", withHistoryLimits(2, 3))

	var objs []client.Object
	objs = append(objs, s)
	// 4 passed and 5 non-passing, oldest first.
	for i := 0; i < 4; i++ {
		objs = append(objs, ownedRun(s, fmt.Sprintf("pass-%d", i),
			schedBase.Add(time.Duration(i)*time.Minute), burninv1alpha1.RunPassed))
	}
	phases := []burninv1alpha1.RunPhase{
		burninv1alpha1.RunFailed, burninv1alpha1.RunFailed, burninv1alpha1.RunError,
		burninv1alpha1.RunCancelled, burninv1alpha1.RunFailed,
	}
	for i, p := range phases {
		objs = append(objs, ownedRun(s, fmt.Sprintf("bad-%d", i),
			schedBase.Add(time.Duration(i)*time.Minute), p))
	}

	h := newSchedHarness(t, objs...)
	h.at(schedBase.Add(time.Hour)).reconcile("weekly")

	var passed, notPassed []string
	for _, r := range h.runs("weekly") {
		if r.Status.Phase == burninv1alpha1.RunPassed {
			passed = append(passed, r.Name)
		} else {
			notPassed = append(notPassed, r.Name)
		}
	}
	if len(passed) != 2 {
		t.Errorf("kept %d passed runs %v, want 2", len(passed), passed)
	}
	if len(notPassed) != 3 {
		t.Errorf("kept %d non-passing runs %v, want 3", len(notPassed), notPassed)
	}
	// The NEWEST are kept: comparing a failure against the ones before it is
	// the reason the history exists.
	if len(passed) == 2 && (passed[0] != "pass-3" || passed[1] != "pass-2") {
		t.Errorf("kept %v, want the newest passed runs [pass-3 pass-2]", passed)
	}
	if len(notPassed) == 3 && notPassed[0] != "bad-4" {
		t.Errorf("newest kept non-passing run is %q, want bad-4", notPassed[0])
	}
}

// Error and Cancelled share the failure budget rather than the success one:
// neither is a passing verdict.
func TestSchedule_ErrorAndCancelledCountAgainstTheFailureBudget(t *testing.T) {
	s := newSchedule("weekly", "0 3 * * 0", withHistoryLimits(5, 1))
	objs := []client.Object{
		s,
		ownedRun(s, "a-error", schedBase, burninv1alpha1.RunError),
		ownedRun(s, "b-cancelled", schedBase.Add(time.Minute), burninv1alpha1.RunCancelled),
	}
	h := newSchedHarness(t, objs...)
	h.at(schedBase.Add(time.Hour)).reconcile("weekly")

	runs := h.runs("weekly")
	if len(runs) != 1 {
		t.Fatalf("kept %d runs, want 1", len(runs))
	}
	if runs[0].Name != "b-cancelled" {
		t.Errorf("kept %q, want the newest non-passing run b-cancelled", runs[0].Name)
	}
}

// GC must never touch a run that is still going: deleting it would kill an
// in-flight burn-in and tear down the object that owes its nodes an uncordon.
func TestSchedule_HistoryGCNeverDeletesAnActiveRun(t *testing.T) {
	s := newSchedule("weekly", "0 3 * * 0",
		withHistoryLimits(0, 0),
		withConcurrency(burninv1alpha1.ConcurrencyAllow))
	objs := []client.Object{
		s,
		ownedRun(s, "old-passed", schedBase, burninv1alpha1.RunPassed),
		ownedRun(s, "running", schedBase.Add(time.Minute), burninv1alpha1.RunRunning),
		ownedRun(s, "pending", schedBase.Add(2*time.Minute), burninv1alpha1.RunPending),
	}
	h := newSchedHarness(t, objs...)
	h.at(schedBase.Add(time.Hour)).reconcile("weekly")

	names := map[string]bool{}
	for _, r := range h.runs("weekly") {
		names[r.Name] = true
	}
	if names["old-passed"] {
		t.Error("terminal run survived a zero history limit")
	}
	if !names["running"] || !names["pending"] {
		t.Errorf("history GC deleted an active run; survivors = %v", names)
	}
}

func TestSchedule_UnsetHistoryLimitsUseTheDocumentedDefaults(t *testing.T) {
	s := newSchedule("weekly", "0 3 * * 0")
	if s.Spec.SuccessfulRunsHistoryLimit != nil || s.Spec.FailedRunsHistoryLimit != nil {
		t.Fatal("fixture should leave both limits unset")
	}
	objs := []client.Object{s}
	for i := 0; i < 6; i++ {
		objs = append(objs, ownedRun(s, fmt.Sprintf("pass-%d", i),
			schedBase.Add(time.Duration(i)*time.Minute), burninv1alpha1.RunPassed))
	}
	for i := 0; i < 12; i++ {
		objs = append(objs, ownedRun(s, fmt.Sprintf("fail-%02d", i),
			schedBase.Add(time.Duration(i)*time.Minute), burninv1alpha1.RunFailed))
	}
	h := newSchedHarness(t, objs...)
	h.at(schedBase.Add(time.Hour)).reconcile("weekly")

	var passed, failed int
	for _, r := range h.runs("weekly") {
		if r.Status.Phase == burninv1alpha1.RunPassed {
			passed++
		} else {
			failed++
		}
	}
	if passed != int(defaultSuccessfulRunsHistoryLimit) {
		t.Errorf("kept %d passed runs, want the default %d", passed, defaultSuccessfulRunsHistoryLimit)
	}
	// The failure budget is deliberately the larger of the two.
	if failed != int(defaultFailedRunsHistoryLimit) {
		t.Errorf("kept %d failed runs, want the default %d", failed, defaultFailedRunsHistoryLimit)
	}
	if defaultFailedRunsHistoryLimit <= defaultSuccessfulRunsHistoryLimit {
		t.Error("the failure history budget must exceed the success budget")
	}
}

// --- suspend ---------------------------------------------------------------

func TestSchedule_SuspendStopsNewRuns(t *testing.T) {
	s := newSchedule("hourly", "0 * * * *", withSuspend(true))
	h := newSchedHarness(t, s)

	res := h.at(schedBase.Add(time.Hour)).reconcile("hourly")
	if n := len(h.runs("hourly")); n != 0 {
		t.Fatalf("suspended schedule created %d runs", n)
	}
	if res.RequeueAfter != 0 {
		t.Errorf("RequeueAfter = %v, want 0: a suspended schedule has nothing to poll for", res.RequeueAfter)
	}
	if got := h.schedule("hourly").Status.LastScheduleTime; got != nil {
		t.Errorf("LastScheduleTime = %v, want nil while suspended", got)
	}
}

// Suspending pauses creation. It is not an instruction to stop observing the
// runs already in flight, nor to retain them differently.
func TestSchedule_SuspendStillTracksActiveRunsAndCollectsHistory(t *testing.T) {
	s := newSchedule("hourly", "0 * * * *", withSuspend(true), withHistoryLimits(1, 1))
	objs := []client.Object{
		s,
		ownedRun(s, "running", schedBase.Add(time.Minute), burninv1alpha1.RunRunning),
		ownedRun(s, "old-pass", schedBase, burninv1alpha1.RunPassed),
		ownedRun(s, "new-pass", schedBase.Add(2*time.Minute), burninv1alpha1.RunPassed),
	}
	h := newSchedHarness(t, objs...)
	h.at(schedBase.Add(time.Hour)).reconcile("hourly")

	got := h.schedule("hourly")
	if len(got.Status.Active) != 1 || got.Status.Active[0].Name != "running" {
		t.Errorf("Active = %+v, want the in-flight run while suspended", got.Status.Active)
	}
	names := map[string]bool{}
	for _, r := range h.runs("hourly") {
		names[r.Name] = true
	}
	if names["old-pass"] {
		t.Error("history was not collected while suspended")
	}
	if !names["running"] || !names["new-pass"] {
		t.Errorf("unexpected deletions while suspended; survivors = %v", names)
	}
}

func TestSchedule_UnsuspendDoesNotReplayTheSuspendedWindow(t *testing.T) {
	s := newSchedule("hourly", "0 * * * *", withSuspend(true))
	h := newSchedHarness(t, s)
	h.at(schedBase.Add(5 * time.Hour)).reconcile("hourly")

	// Unsuspend five hours later.
	live := h.schedule("hourly")
	no := false
	live.Spec.Suspend = &no
	if err := h.c.Update(context.Background(), live); err != nil {
		t.Fatalf("unsuspend: %v", err)
	}
	h.at(schedBase.Add(5 * time.Hour)).reconcile("hourly")

	if n := len(h.runs("hourly")); n != 1 {
		t.Fatalf("unsuspend created %d runs; the suspended window must not be replayed as a burst", n)
	}
}

// --- run template ---------------------------------------------------------

func TestSchedule_StampsTemplateOntoCreatedRun(t *testing.T) {
	ttl := int32(3600)
	deadline := int32(7200)
	maxNodes := int32(2)
	retries := int32(1)
	s := newSchedule("weekly", "0 3 * * 0", withRunTemplate(burninv1alpha1.BurnInRunTemplate{
		TTLSecondsAfterFinished: &ttl,
		DeadlineSeconds:         &deadline,
		MaxConcurrentNodes:      &maxNodes,
		RetryOnErrorLimit:       &retries,
	}))
	h := newSchedHarness(t, s)
	h.at(schedBase.Add(3 * time.Hour)).reconcile("weekly")

	runs := h.runs("weekly")
	if len(runs) != 1 {
		t.Fatalf("expected 1 run, got %d", len(runs))
	}
	run := runs[0]

	if !strings.HasPrefix(run.Name, "weekly-") {
		t.Errorf("run name %q does not use the schedule's generateName prefix", run.Name)
	}
	if run.Spec.ProfileRef != "acceptance" {
		t.Errorf("ProfileRef = %q, want acceptance", run.Spec.ProfileRef)
	}
	if got := run.Spec.Target.NodeSelector["glimmer.ai/accelerator"]; got != "gb10" {
		t.Errorf("target not carried onto the run: %+v", run.Spec.Target)
	}
	if int32OrDefault(run.Spec.TTLSecondsAfterFinished, -1) != ttl {
		t.Errorf("TTLSecondsAfterFinished = %v, want %d", run.Spec.TTLSecondsAfterFinished, ttl)
	}
	if int32OrDefault(run.Spec.DeadlineSeconds, -1) != deadline {
		t.Errorf("DeadlineSeconds = %v, want %d", run.Spec.DeadlineSeconds, deadline)
	}
	if int32OrDefault(run.Spec.MaxConcurrentNodes, -1) != maxNodes {
		t.Errorf("MaxConcurrentNodes = %v, want %d", run.Spec.MaxConcurrentNodes, maxNodes)
	}
	if int32OrDefault(run.Spec.RetryOnErrorLimit, -1) != retries {
		t.Errorf("RetryOnErrorLimit = %v, want %d", run.Spec.RetryOnErrorLimit, retries)
	}
	if run.Labels[labelSchedule] != "weekly" {
		t.Errorf("schedule label = %q, want weekly", run.Labels[labelSchedule])
	}
	if !metav1.IsControlledBy(&run, h.schedule("weekly")) {
		t.Error("created run is not controller-owned by the schedule")
	}
}

// The facility interlock must be written explicitly, not left to apiserver
// defaulting, and it must never be inferred from anything.
func TestSchedule_UnsetTemplateGetsTheSafeFloor(t *testing.T) {
	s := newSchedule("weekly", "0 3 * * 0")
	h := newSchedHarness(t, s)
	h.at(schedBase.Add(3 * time.Hour)).reconcile("weekly")

	run := h.runs("weekly")[0]
	if run.Spec.MaxConcurrentNodes == nil {
		t.Fatal("MaxConcurrentNodes left nil; the safety interlock must be explicit on the run")
	}
	if *run.Spec.MaxConcurrentNodes != 1 {
		t.Errorf("MaxConcurrentNodes = %d, want the safe floor of 1", *run.Spec.MaxConcurrentNodes)
	}
	if run.Spec.RetryOnErrorLimit == nil || *run.Spec.RetryOnErrorLimit != 0 {
		t.Errorf("RetryOnErrorLimit = %v, want an explicit 0", run.Spec.RetryOnErrorLimit)
	}
}

// The schedule comes out of a shared informer cache; the created run must not
// alias its slices and maps.
func TestSchedule_CreatedRunDoesNotAliasTheScheduleTarget(t *testing.T) {
	s := newSchedule("weekly", "0 3 * * 0")
	s.Spec.Target.NodeNames = []string{"node-a"}
	h := newSchedHarness(t, s)

	run, err := h.r.runForTick(s, schedBase)
	if err != nil {
		t.Fatalf("runForTick: %v", err)
	}
	run.Spec.Target.NodeNames[0] = "mutated"
	run.Spec.Target.NodeSelector["glimmer.ai/accelerator"] = "mutated"

	if s.Spec.Target.NodeNames[0] != "node-a" {
		t.Error("mutating the run's target reached back into the schedule's NodeNames")
	}
	if s.Spec.Target.NodeSelector["glimmer.ai/accelerator"] != "gb10" {
		t.Error("mutating the run's target reached back into the schedule's NodeSelector")
	}
}

// --- status ---------------------------------------------------------------

func TestSchedule_TracksLastRunPhaseAndSuccessTime(t *testing.T) {
	s := newSchedule("hourly", "0 * * * *")
	h := newSchedHarness(t, s)

	h.at(schedBase.Add(time.Hour)).reconcile("hourly")
	first := h.runs("hourly")[0]
	h.at(schedBase.Add(90 * time.Minute))
	h.setRunPhase(&first, burninv1alpha1.RunPassed)
	h.reconcile("hourly")

	got := h.schedule("hourly")
	if got.Status.LastRunPhase != burninv1alpha1.RunPassed {
		t.Errorf("LastRunPhase = %q, want Passed", got.Status.LastRunPhase)
	}
	if got.Status.LastSuccessfulTime == nil {
		t.Fatal("LastSuccessfulTime not recorded for a passing run")
	}
	if len(got.Status.Active) != 0 {
		t.Errorf("Active = %+v, want empty once the run is terminal", got.Status.Active)
	}

	// A later failing run must not clear the record that it once succeeded,
	// and the failure must be reported as Failed rather than smoothed over.
	h.at(schedBase.Add(2 * time.Hour)).reconcile("hourly")
	second := h.runs("hourly")[0]
	h.at(schedBase.Add(150 * time.Minute))
	h.setRunPhase(&second, burninv1alpha1.RunFailed)
	h.reconcile("hourly")

	got = h.schedule("hourly")
	if got.Status.LastRunPhase != burninv1alpha1.RunFailed {
		t.Errorf("LastRunPhase = %q, want Failed", got.Status.LastRunPhase)
	}
	if got.Status.LastSuccessfulTime == nil {
		t.Error("LastSuccessfulTime was cleared by a later failure")
	}
}

// History GC eventually deletes the passing run LastSuccessfulTime came from;
// retention must not erase the fact that the schedule once succeeded.
func TestSchedule_LastSuccessfulTimeSurvivesHistoryGC(t *testing.T) {
	s := newSchedule("weekly", "0 3 * * 0", withHistoryLimits(0, 10))
	pass := ownedRun(s, "old-pass", schedBase, burninv1alpha1.RunPassed)
	finished := schedBase.Add(30 * time.Minute)
	pass.Status.FinishedAt = &metav1.Time{Time: finished}

	h := newSchedHarness(t, s, pass)

	// First pass records the success and GCs the run it came from (limit 0).
	h.at(schedBase.Add(time.Hour)).reconcile("weekly")
	if n := len(h.runs("weekly")); n != 0 {
		t.Fatalf("expected the passing run to be garbage-collected, %d left", n)
	}
	got := h.schedule("weekly")
	if got.Status.LastSuccessfulTime == nil || !got.Status.LastSuccessfulTime.Time.Equal(finished) {
		t.Fatalf("LastSuccessfulTime = %v, want %v", got.Status.LastSuccessfulTime, finished)
	}

	// Second pass sees no passing run at all. Retention must not erase the
	// fact that the schedule once succeeded.
	h.at(schedBase.Add(2 * time.Hour)).reconcile("weekly")
	got = h.schedule("weekly")
	if got.Status.LastSuccessfulTime == nil || !got.Status.LastSuccessfulTime.Time.Equal(finished) {
		t.Errorf("LastSuccessfulTime = %v after GC, want it preserved at %v", got.Status.LastSuccessfulTime, finished)
	}
}

func TestSchedule_IgnoresRunsItDoesNotOwn(t *testing.T) {
	s := newSchedule("hourly", "0 * * * *")
	other := newSchedule("other", "0 * * * *")
	other.UID = types.UID("uid-other")

	// An active run belonging to a different schedule, plus a stray run
	// wearing our label but owned by nobody.
	foreign := ownedRun(other, "foreign", schedBase, burninv1alpha1.RunRunning)
	stray := ownedRun(s, "stray", schedBase, burninv1alpha1.RunRunning)
	stray.OwnerReferences = nil
	stray.Labels = map[string]string{labelSchedule: "hourly"}

	h := newSchedHarness(t, s, other, foreign, stray)
	h.at(schedBase.Add(time.Hour)).reconcile("hourly")

	// Neither may satisfy the Forbid guard, so the tick still fires.
	if n := len(h.runs("hourly")); n != 1 {
		t.Fatalf("expected 1 owned run, got %d (a foreign or unowned run was mistaken for ours)", n)
	}
	got := h.schedule("hourly")
	for _, ref := range got.Status.Active {
		if ref.Name == "foreign" || ref.Name == "stray" {
			t.Errorf("Active contains a run this schedule does not own: %q", ref.Name)
		}
	}
}

// --- invalid specs --------------------------------------------------------

func TestSchedule_InvalidScheduleIsSurfacedAndCreatesNothing(t *testing.T) {
	for _, tc := range []struct {
		name string
		opts []schedOpt
		spec string
	}{
		{name: "garbage", spec: "not a cron spec"},
		{name: "empty", spec: "   "},
		{name: "bad timezone", spec: "0 3 * * 0", opts: []schedOpt{withTimeZone("Mars/Olympus_Mons")}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			s := newSchedule("broken", tc.spec, tc.opts...)
			h := newSchedHarness(t, s)
			res := h.at(schedBase.Add(3 * time.Hour)).reconcile("broken")

			if n := len(h.runs("broken")); n != 0 {
				t.Fatalf("invalid schedule created %d runs", n)
			}
			if res.RequeueAfter != 0 {
				t.Errorf("RequeueAfter = %v; an unparseable spec must wait for an edit, not poll", res.RequeueAfter)
			}
			got := h.schedule("broken")
			var found *metav1.Condition
			for i := range got.Status.Conditions {
				if got.Status.Conditions[i].Type == conditionScheduleValid {
					found = &got.Status.Conditions[i]
				}
			}
			if found == nil {
				t.Fatal("no ScheduleValid condition; a schedule that never fires must not look healthy")
			}
			if found.Status != metav1.ConditionFalse {
				t.Errorf("ScheduleValid = %q, want False", found.Status)
			}
		})
	}
}

// cron.ParseStandard PANICS on a bare "TZ=..." with no following space: it
// slices on strings.Index(spec, " ") without checking for -1. spec.schedule is
// user-supplied, so passing it through unguarded would let anyone who can
// create a BurnInSchedule crash the manager — taking down every other
// reconciler in the process, including runs that owe nodes an uncordon.
func TestSchedule_TimeZonePrefixIsRejectedNotPassedToTheParser(t *testing.T) {
	for _, spec := range []string{"TZ=Foo", "CRON_TZ=Foo", "TZ=America/Los_Angeles 0 3 * * 0"} {
		t.Run(spec, func(t *testing.T) {
			s := newSchedule("broken", spec)
			h := newSchedHarness(t, s)

			// Must return an error value, never panic.
			if _, err := parseSchedule(s); err == nil {
				t.Fatal("a TZ prefix in spec.schedule was accepted; use spec.timeZone")
			}
			h.at(schedBase.Add(3 * time.Hour)).reconcile("broken")
			if n := len(h.runs("broken")); n != 0 {
				t.Fatalf("created %d runs for an invalid schedule", n)
			}
		})
	}
}

func TestSchedule_ValidScheduleClearsTheCondition(t *testing.T) {
	s := newSchedule("weekly", "0 3 * * 0")
	h := newSchedHarness(t, s)
	h.at(schedBase).reconcile("weekly")

	got := h.schedule("weekly")
	for _, c := range got.Status.Conditions {
		if c.Type == conditionScheduleValid && c.Status != metav1.ConditionTrue {
			t.Errorf("ScheduleValid = %q, want True", c.Status)
		}
	}
	if got.Status.ObservedGeneration != s.Generation {
		t.Errorf("ObservedGeneration = %d, want %d", got.Status.ObservedGeneration, s.Generation)
	}
}

// --- crash-safety ordering ------------------------------------------------

// The cursor is persisted BEFORE the run is created. Losing a tick is a gap in
// the record; duplicating one is a second full-power sweep launched across the
// fleet against nodes a live run already holds cordoned.
func TestSchedule_CursorIsPersistedBeforeTheRunIsCreated(t *testing.T) {
	s := newSchedule("hourly", "0 * * * *")
	failCreate := true
	h := newSchedHarnessWith(t, &interceptor.Funcs{
		Create: func(ctx context.Context, c client.WithWatch, obj client.Object, opts ...client.CreateOption) error {
			if _, ok := obj.(*burninv1alpha1.BurnInRun); ok && failCreate {
				return apierrors.NewInternalError(fmt.Errorf("apiserver is down"))
			}
			return c.Create(ctx, obj, opts...)
		},
	}, s)

	h.at(schedBase.Add(time.Hour))
	if err := h.reconcileErr("hourly"); err == nil {
		t.Fatal("a failed run creation must be reported, not swallowed")
	}

	// The cursor advanced even though nothing was created.
	got := h.schedule("hourly")
	if got.Status.LastScheduleTime == nil || !got.Status.LastScheduleTime.Time.Equal(schedBase.Add(time.Hour)) {
		t.Fatalf("LastScheduleTime = %v, want the cursor persisted at %v before the create",
			got.Status.LastScheduleTime, schedBase.Add(time.Hour))
	}
	if n := len(h.runs("hourly")); n != 0 {
		t.Fatalf("expected 0 runs after a failed create, got %d", n)
	}

	// The spent tick is NOT retried once creation works again: at the same
	// instant nothing new is due, so no run appears.
	failCreate = false
	h.at(schedBase.Add(time.Hour)).reconcile("hourly")
	if n := len(h.runs("hourly")); n != 0 {
		t.Fatalf("the consumed tick was replayed: %d runs", n)
	}

	// The next tick fires normally.
	h.at(schedBase.Add(2 * time.Hour)).reconcile("hourly")
	if n := len(h.runs("hourly")); n != 1 {
		t.Fatalf("expected 1 run at the next tick, got %d", n)
	}
}

// --- idempotence ----------------------------------------------------------

func TestSchedule_ReconcileIsIdempotent(t *testing.T) {
	s := newSchedule("weekly", "0 3 * * 0")
	h := newSchedHarness(t, s)
	h.at(schedBase.Add(3 * time.Hour))

	for i := 0; i < 5; i++ {
		h.reconcile("weekly")
	}
	if n := len(h.runs("weekly")); n != 1 {
		t.Fatalf("repeated reconciles at one instant produced %d runs, want 1", n)
	}
	if got := h.schedule("weekly").Status.MissedSchedules; got != 0 {
		t.Errorf("MissedSchedules = %d, want 0", got)
	}
}

// A settled schedule must not rewrite its own status on every pass: each write
// wakes the watch, which would wake the reconciler, which would write again.
func TestSchedule_SettledReconcileDoesNotRewriteStatus(t *testing.T) {
	s := newSchedule("weekly", "0 3 * * 0")
	h := newSchedHarness(t, s)
	h.at(schedBase.Add(3 * time.Hour)).reconcile("weekly")

	rv := h.schedule("weekly").ResourceVersion
	for i := 0; i < 3; i++ {
		h.reconcile("weekly")
	}
	if got := h.schedule("weekly").ResourceVersion; got != rv {
		t.Errorf("resourceVersion moved from %s to %s with nothing to change", rv, got)
	}
}

func TestSchedule_MissingScheduleIsNotAnError(t *testing.T) {
	h := newSchedHarness(t)
	if err := h.reconcileErr("gone"); err != nil {
		t.Fatalf("reconciling a deleted schedule returned %v, want nil", err)
	}
}

// --- unit-level helpers ---------------------------------------------------

func TestDueTick_ReturnsLatestAndCountsTheRest(t *testing.T) {
	cronSchedule, err := cronFor("0 * * * *")
	if err != nil {
		t.Fatal(err)
	}
	from := schedBase.Add(-3 * time.Hour)
	tick, missed, resync := dueTick(cronSchedule, from, schedBase, nil)

	if resync {
		t.Fatal("unexpected resync for a 3-tick backlog")
	}
	if !tick.Equal(schedBase) {
		t.Errorf("tick = %v, want the most recent %v", tick, schedBase)
	}
	if missed != 2 {
		t.Errorf("missed = %d, want 2", missed)
	}
}

func TestUntilNext_NeverReturnsNonPositive(t *testing.T) {
	cronSchedule, err := cronFor("@every 1s")
	if err != nil {
		t.Fatal(err)
	}
	// A sub-second gap must not round down to zero — controller-runtime reads
	// a non-positive RequeueAfter as "do not requeue", stranding the schedule.
	if got := untilNext(cronSchedule, schedBase); got < scheduleMinRequeue {
		t.Errorf("untilNext = %v, want at least %v", got, scheduleMinRequeue)
	}
}

func TestAddMissed_SaturatesInsteadOfWrapping(t *testing.T) {
	const maxInt32 = int32(1<<31 - 1)
	if got := addMissed(maxInt32-1, 10); got != maxInt32 {
		t.Errorf("addMissed = %d, want saturation at %d", got, maxInt32)
	}
	if got := addMissed(5, 0); got != 5 {
		t.Errorf("addMissed with no delta = %d, want 5", got)
	}
}

// cronFor is a test shim so helper tests do not need a whole schedule object.
func cronFor(spec string) (cron.Schedule, error) {
	s := &burninv1alpha1.BurnInSchedule{Spec: burninv1alpha1.BurnInScheduleSpec{Schedule: spec}}
	return parseSchedule(s)
}
