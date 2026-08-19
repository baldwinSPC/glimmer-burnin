package controller

import (
	"context"
	"strings"
	"testing"
	"time"

	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"

	burninv1alpha1 "github.com/baldwinSPC/glimmer-burnin/api/v1alpha1"
)

// The operator's two halves of docs/dev/multi-device.md. Everything else in
// that note is the runner's: the reconciler learns nothing about devices.

// A runner can only tell the devices it was ALLOCATED from the devices it can
// SEE if somebody tells it what it asked for. The pod cannot read its own
// spec, so the operator copies its limits in — verbatim, sorted, uninterpreted.
// Which name is the accelerator is vendor knowledge, and it stays in the image.
func TestPod_CarriesItsOwnResourceLimitsForTheRunner(t *testing.T) {
	test := smokeTest("fp4")
	test.Spec.Resources = corev1.ResourceRequirements{
		Limits: corev1.ResourceList{
			corev1.ResourceName("rdma/hca"):       resource.MustParse("1"),
			corev1.ResourceName("nvidia.com/gpu"): resource.MustParse("8"),
			corev1.ResourceMemory:                 resource.MustParse("2Gi"),
		},
		// Requests alone are never consulted: Kubernetes refuses a pod that
		// requests an extended resource without an equal limit, so limits are
		// the one place a count is always present.
		Requests: corev1.ResourceList{
			corev1.ResourceName("nvidia.com/gpu"): resource.MustParse("8"),
		},
	}
	h := newHarness(t,
		gb10Node("spark-a"),
		test,
		profile("acceptance", nil, false, testRef("fp4")),
		newRun("run1", "acceptance", "spark-a"),
	)
	h.reconcile("run1")
	h.reconcile("run1")

	pod := h.pods("run1")["spark-a"]
	got := envOf(pod, "BURNIN_RESOURCE_LIMITS")
	// Sorted by name so two identical plans make two identical pods; the
	// quantities are the canonical strings so a runner parses what an author
	// wrote.
	want := "memory=2Gi,nvidia.com/gpu=8,rdma/hca=1"
	if got != want {
		t.Errorf("BURNIN_RESOURCE_LIMITS = %q, want %q", got, want)
	}

	// No limits, no variable: absent is a state the runner reads ("iterate
	// every visible device and say so" — safe on bare metal, where all IS the
	// allocation), and an empty string would be a third state nobody defined.
	h2 := newHarness(t,
		gb10Node("spark-a"),
		smokeTest("fp4"),
		profile("acceptance", nil, false, testRef("fp4")),
		newRun("run1", "acceptance", "spark-a"),
	)
	h2.reconcile("run1")
	h2.reconcile("run1")
	for _, e := range h2.pods("run1")["spark-a"].Spec.Containers[0].Env {
		if e.Name == "BURNIN_RESOURCE_LIMITS" {
			t.Errorf("a test with no limits still got BURNIN_RESOURCE_LIMITS=%q; absent must stay absent", e.Value)
		}
	}
}

// A BurnInTest that requests more accelerators than the node holds is a pod
// the scheduler never places. That is already an Error after the pod's window
// — but the generic sentence names only the class ("unschedulable target or
// stuck image pull"), and the scheduler's own account, "Insufficient
// nvidia.com/gpu", is the one that names the fix. It rides in the way the
// kubelet's message does for a container that never started (#52): as detail,
// clamped, never as the reason.
func TestRun_AnUnscheduledPodsErrorNamesTheSchedulersReason(t *testing.T) {
	h := newHarness(t,
		gb10Node("spark-a"),
		smokeTest("fp4"),
		profile("acceptance", nil, false, testRef("fp4")),
		withNodeCap(newRun("run1", "acceptance", "spark-a"), 1),
	)
	h.reconcile("run1")
	h.reconcile("run1")
	pod := h.pods("run1")["spark-a"]
	if pod == nil {
		t.Fatal("setup: no pod")
	}

	// The scheduler's verdict, as kube-scheduler writes it on a pod it cannot
	// place. The pod stays Pending with no StartTime — it never ran anywhere.
	pod.Status.Phase = corev1.PodPending
	pod.Status.Conditions = []corev1.PodCondition{{
		Type:    corev1.PodScheduled,
		Status:  corev1.ConditionFalse,
		Reason:  corev1.PodReasonUnschedulable,
		Message: "0/3 nodes are available: 3 Insufficient nvidia.com/gpu.\npreemption: 0/3 nodes are available: 3 No preemption victims found for incoming pod.",
	}}
	if err := h.c.Status().Update(context.Background(), pod); err != nil {
		t.Fatal(err)
	}
	backdate(t, h, pod, h.nowVal)

	// Past duration + deadline grace + scheduling grace: podOverdue fires and
	// the attempt is settled Error.
	h.nowVal = h.nowVal.Add(24 * time.Hour)
	h.reconcile("run1")
	h.reconcile("run1")

	run := h.run("run1")
	if len(run.Status.Results) != 1 {
		t.Fatalf("results = %+v, want one", run.Status.Results)
	}
	res := run.Status.Results[0]
	if len(res.Attempts) == 0 || res.Attempts[0].Phase != burninv1alpha1.RunError {
		t.Fatalf("attempt = %+v, want an Error for a pod that never landed", res.Attempts)
	}
	msg := res.Attempts[0].Message
	if !strings.Contains(msg, "Insufficient nvidia.com/gpu") {
		t.Errorf("message = %q, want the scheduler's own reason in it — that is what names a wrong accelerator count", msg)
	}
	if strings.Contains(msg, "\n") {
		t.Errorf("message = %q carries a newline; the scheduler's text is flattened like the kubelet's", msg)
	}
	if !strings.HasPrefix(msg, "pod never completed within its window") {
		t.Errorf("message = %q, want the generic reason to stay first — the detail is appended, never substituted", msg)
	}
	// And the reason is a DETAIL: nothing else about the outcome moved.
	if run.Status.Phase == burninv1alpha1.RunPassed {
		t.Errorf("run passed on a pod that never scheduled")
	}
}
