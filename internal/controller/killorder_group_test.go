package controller

import (
	"testing"
	"time"

	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/client/interceptor"
)

// The remaining Gap B paths from issue #372 — Group scope.
//
// killorder_test.go covers the Node-scope overdue pod and two of the four Pair
// paths; killorder_linger_test.go covers the third. This file covers the two
// Group paths that can actually be observed at a kill: the root ready-gate
// timeout and a hung collective. Both go through podOverdue/anyOverdue exactly
// like the paths already covered, and reach a pass.kill list exactly like them,
// so the SAME killOrderWatcher applies unchanged.
//
// The third row in #372's table — "root exited before its workers" — is not
// here, deliberately. gateWorkersOnRoot's early-terminate branch (the root
// finishing before any worker exists) records the attempt and returns a kill
// list of pods that are ALL either nil (a worker never created) or already
// terminated (the root itself) — so killPods has nothing live to delete and no
// Delete call is ever made. There is no ordering to assert: the same is true of
// Pair's analogous branch in gateClientOnServer (the server finishing before the
// client exists), which killorder_test.go correctly does not test for the same
// reason.

// TestKillOrder_AGroupRootThatNeverBecameReadyIsRecordedBeforeItIsDestroyed.
//
// The Group ready-gate timeout: the root started but never reported Ready, so
// no worker was ever created and the collective was never measured. The Error
// naming that has to be durable before the root pod goes — Group's mirror of
// the Pair ready-gate test above.
func TestKillOrder_AGroupRootThatNeverBecameReadyIsRecordedBeforeItIsDestroyed(t *testing.T) {
	nodes := []string{"spark-a", "spark-b", "spark-c"}
	w := &killOrderWatcher{}
	h := newHarnessWithInterceptor(t, func(inner client.Client) interceptor.Funcs { return w.funcs(inner) },
		gb10Node(nodes[0]), gb10Node(nodes[1]), gb10Node(nodes[2]),
		groupTest("nccl-group"),
		profile("acceptance", nil, false, testRef("nccl-group")),
		groupRun("run1", "acceptance", nodes...),
	)

	h.reconcile("run1")
	h.reconcile("run1")
	root := h.rankPod("run1", groupRootRank)
	if root == nil {
		t.Fatal("setup: the root pod was never created")
	}
	// STARTED but never READY, exactly the Pair gate's shape: a container that
	// is up has not necessarily bound the rendezvous socket.
	h.startPod(root)
	backdate(t, h, root, h.nowVal)

	h.nowVal = h.nowVal.Add(24 * time.Hour)
	for i := 0; i < 5; i++ {
		h.reconcile("run1")
	}

	if w.deletes == 0 {
		t.Fatal("no pod was deleted, so the Group ready-gate timeout never fired and this proved nothing")
	}
	for _, v := range w.violations {
		t.Errorf("issue #247 on the Group ready-gate path: %s", v)
	}
}

// TestKillOrder_AHungCollectiveRecordsBeforeTakingDownTheRanksStillRunning.
//
// Mirrors TestGroup_DeadlineNamesTheRanksThatHung, but watches the kill order
// rather than the verdict message: two ranks finish cleanly, one hangs past its
// window, and the Error naming the hung rank has to be durable before ANY rank
// is destroyed — including the two that already reported, which harvestGroup's
// kill list carries alongside the hung one.
func TestKillOrder_AHungCollectiveRecordsBeforeTakingDownTheRanksStillRunning(t *testing.T) {
	nodes := []string{"spark-a", "spark-b", "spark-c"}
	w := &killOrderWatcher{}
	h := newHarnessWithInterceptor(t, func(inner client.Client) interceptor.Funcs { return w.funcs(inner) },
		gb10Node(nodes[0]), gb10Node(nodes[1]), gb10Node(nodes[2]),
		groupTest("nccl-group"),
		profile("acceptance", nil, false, testRef("nccl-group")),
		groupRun("run1", "acceptance", nodes...),
	)
	h.startGroup("run1", len(nodes))

	// Ranks 0 and 1 finish; rank 2 never does, and the window expires.
	h.finishPod(h.rankPod("run1", 0), 0, ncclRootStdout, "Completed")
	h.finishPod(h.rankPod("run1", 1), 0, ncclWorkerStdout, "Completed")
	h.agePods("run1")
	h.nowVal = h.nowVal.Add(time.Duration(defaultDurationSeconds+deadlineGraceSeconds)*time.Second +
		schedulingGracePeriod + time.Minute)
	for i := 0; i < 5; i++ {
		h.reconcile("run1")
	}

	if w.deletes == 0 {
		t.Fatal("no pod was deleted, so the hung-collective path never fired and this proved nothing")
	}
	for _, v := range w.violations {
		t.Errorf("issue #247 on the Group hung-collective path: %s", v)
	}
}
