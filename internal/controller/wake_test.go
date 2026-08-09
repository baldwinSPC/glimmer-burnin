package controller

import (
	"testing"
	"time"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"

	burninv1alpha1 "github.com/baldwinSPC/glimmer-burnin/api/v1alpha1"
)

// The wake cadence comes from what is IN FLIGHT — issue #158.
//
// It used to floor at 30 seconds for a run's entire duration, so a seven-day
// soak reconciled roughly 20,000 times, and every pass listed pods, possibly
// fetched logs, and possibly wrote status. For a run that is by design doing
// nothing observable for hours at a stretch, that is a lot of apiserver traffic
// to learn nothing.

func wakeRun() *burninv1alpha1.BurnInRun { return newRun("r", "p", "spark-a") }

// Once every pod has started and nothing checkpoints sooner, the run settles to
// the long backstop.
func TestASettledRunUsesTheLongBackstop(t *testing.T) {
	h := newHarness(t)
	got := h.r.nextWake(wakeRun(), passResult{running: true})
	if got != settledPollInterval {
		t.Fatalf("wake = %s, want %s. A day-long run at the old 30-second floor is "+
			"~2,880 reconciles; at this one it is ~288.", got, settledPollInterval)
	}
}

// A pod that has not started keeps the fast cadence. That is the window where
// scheduling problems appear and a fast reaction is worth the traffic.
func TestAnUnstartedPodKeepsTheFastCadence(t *testing.T) {
	h := newHarness(t)
	got := h.r.nextWake(wakeRun(), passResult{running: true, awaitingStart: true})
	if got != waitingPollInterval {
		t.Fatalf("wake = %s, want %s — an unschedulable pod or an ImagePullBackOff "+
			"would go unnoticed for five minutes", got, waitingPollInterval)
	}
}

// A checkpoint due sooner than the backstop still wins.
func TestAnImminentCheckpointBeatsTheBackstop(t *testing.T) {
	h := newHarness(t)
	got := h.r.nextWake(wakeRun(), passResult{running: true, checkpointEvery: 60 * time.Second})
	if got != 60*time.Second {
		t.Fatalf("wake = %s, want 1m — a checkpoint interval shorter than the backstop "+
			"must still be honoured", got)
	}
}

// A checkpoint interval LONGER than the backstop does not stretch it.
//
// The backstop exists for the pod that never emits an event; an hourly
// checkpoint must not mean the run goes an hour without noticing one.
func TestALongCheckpointIntervalDoesNotStretchTheBackstop(t *testing.T) {
	h := newHarness(t)
	got := h.r.nextWake(wakeRun(), passResult{running: true, checkpointEvery: time.Hour})
	if got != settledPollInterval {
		t.Fatalf("wake = %s, want %s", got, settledPollInterval)
	}
}

// THE FIX ITSELF: only the tests actually in flight get a vote.
//
// checkpointEvery is populated from executions that returned advanceRunning, so
// a profile mixing a 60-second-checkpoint test with an hourly one wakes fast
// only while the fast test is running. The plan-wide minimum used to set the
// cadence for the whole run, including while the hourly test was the only thing
// executing.
func TestOnlyTestsInFlightSetTheCadence(t *testing.T) {
	h := newHarness(t)

	fast := h.r.nextWake(wakeRun(), passResult{running: true, checkpointEvery: 60 * time.Second})
	if fast != 60*time.Second {
		t.Errorf("while the fast test is in flight, wake = %s, want 1m", fast)
	}
	// The same run, later, with only the hourly test executing: checkpointEvery
	// now carries the hourly interval and nothing votes for 60 seconds.
	slow := h.r.nextWake(wakeRun(), passResult{running: true, checkpointEvery: time.Hour})
	if slow == 60*time.Second {
		t.Fatal("the cadence is still 1m while only the hourly test is running — one " +
			"test's interval is setting the wake for a profile it is not part of")
	}
	if slow != settledPollInterval {
		t.Errorf("wake = %s, want %s", slow, settledPollInterval)
	}
}

// The run deadline still clamps, so nothing overshoots it.
func TestTheDeadlineStillClamps(t *testing.T) {
	h := newHarness(t)
	run := wakeRun()
	run.Spec.DeadlineSeconds = int32p(120)
	started := metav1.NewTime(h.r.now().Add(-90 * time.Second))
	run.Status.StartedAt = &started

	got := h.r.nextWake(run, passResult{running: true})
	if got > 30*time.Second {
		t.Fatalf("wake = %s with 30s left on the deadline; the run would notice it "+
			"%s late", got, got-30*time.Second)
	}
	if got <= 0 {
		t.Fatalf("wake = %s, which would spin", got)
	}
}

// The wake is always positive, whatever it is given.
//
// A non-positive RequeueAfter busy-loops the manager against the apiserver.
// nextWake has no zero guard because every assignment in it is positive-guarded
// and one was provably unreachable — so the property is pinned here, over the
// inputs, rather than defended by a branch that can never run.
func TestTheWakeIsAlwaysPositive(t *testing.T) {
	h := newHarness(t)
	for _, deadline := range []int32{0, 1, 60, 3600} {
		for _, age := range []time.Duration{0, time.Second, time.Hour, 10 * 24 * time.Hour} {
			for _, cp := range []time.Duration{0, -time.Second, time.Second, time.Hour} {
				for _, awaiting := range []bool{false, true} {
					run := wakeRun()
					if deadline > 0 {
						run.Spec.DeadlineSeconds = int32p(deadline)
					}
					started := metav1.NewTime(h.r.now().Add(-age))
					run.Status.StartedAt = &started

					got := h.r.nextWake(run,
						passResult{running: true, checkpointEvery: cp, awaitingStart: awaiting})
					if got <= 0 {
						t.Fatalf("wake = %s for deadline=%d age=%s checkpointEvery=%s "+
							"awaitingStart=%v — the manager would busy-loop against the "+
							"apiserver", got, deadline, age, cp, awaiting)
					}
				}
			}
		}
	}
}

// The cadence through a REAL reconcile, not by calling nextWake directly.
//
// The tests above pass passResult by hand, so they cannot see whether execute
// actually populates it — and mutating away both the awaitingStart derivation
// and the in-flight checkpoint vote left every one of them green. These drive
// the reconciler.
func TestTheCadenceIsDerivedByARealPass(t *testing.T) {
	t.Run("an unstarted pod keeps the fast cadence", func(t *testing.T) {
		h := newHarness(t,
			gb10Node("spark-a"),
			smokeTest("fp4"),
			profile("acceptance", nil, false, testRef("fp4")),
			newRun("run1", "acceptance", "spark-a"),
		)
		h.reconcile("run1") // Pending -> Running
		h.reconcile("run1") // creates the pod
		if len(h.pods("run1")) != 1 {
			t.Fatal("no pod was created")
		}
		// The pod exists and has NOT been started by the fake kubelet.
		got := h.reconcile("run1").RequeueAfter
		if got != waitingPollInterval {
			t.Fatalf("RequeueAfter = %s with a pod that has not started, want %s. "+
				"An unschedulable pod or an ImagePullBackOff would go unnoticed for "+
				"five minutes.", got, waitingPollInterval)
		}
	})

	t.Run("a started pod settles to the backstop", func(t *testing.T) {
		h := newHarness(t,
			gb10Node("spark-a"),
			smokeTest("fp4"),
			profile("acceptance", nil, false, testRef("fp4")),
			newRun("run1", "acceptance", "spark-a"),
		)
		h.reconcile("run1")
		h.reconcile("run1")
		h.startPod(h.pods("run1")["spark-a"])

		got := h.reconcile("run1").RequeueAfter
		if got != settledPollInterval {
			t.Fatalf("RequeueAfter = %s once the pod is running, want %s", got, settledPollInterval)
		}
	})

	t.Run("an in-flight test's own checkpoint interval wins", func(t *testing.T) {
		test := smokeTest("soak")
		test.Spec.CheckpointIntervalSeconds = int32p(45)
		h := newHarness(t,
			gb10Node("spark-a"),
			test,
			profile("acceptance", nil, false, testRef("soak")),
			newRun("run1", "acceptance", "spark-a"),
		)
		h.reconcile("run1")
		h.reconcile("run1")
		h.startPod(h.pods("run1")["spark-a"])

		got := h.reconcile("run1").RequeueAfter
		if got != 45*time.Second {
			t.Fatalf("RequeueAfter = %s for a test checkpointing every 45s, want 45s — "+
				"the in-flight test's own interval is not reaching the wake", got)
		}
	})
}

// A suspended run gets the settled backstop: nothing is in flight, and the thing
// it is waiting for is a spec edit, which arrives as an event.
func TestASuspendedRunSettles(t *testing.T) {
	h := newHarness(t)
	if got := h.r.nextWake(wakeRun(), passResult{}); got != settledPollInterval {
		t.Errorf("wake = %s, want %s", got, settledPollInterval)
	}
}

// The traffic claim, made concrete. A future change that reintroduces a
// 30-second floor for a settled run is caught here rather than on a fleet's
// apiserver.
func TestALongRunDoesNotWakeTwentyThousandTimes(t *testing.T) {
	h := newHarness(t)
	const week = 7 * 24 * time.Hour

	settled := h.r.nextWake(wakeRun(), passResult{running: true})
	wakes := int(week / settled)
	if wakes > 3000 {
		t.Fatalf("a seven-day soak with everything started wakes %d times at a %s "+
			"cadence. Every pass lists pods, may fetch logs and may write status — "+
			"that is a lot of apiserver traffic to learn nothing.", wakes, settled)
	}
}
