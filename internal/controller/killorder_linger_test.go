package controller

import (
	"testing"

	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/client/interceptor"
)

// TestKillOrder_ALingeringServerIsRecordedBeforeItIsDestroyed — the fourth Pair
// path, and the one that was actually BROKEN rather than merely unasserted.
//
// A server may legitimately linger after its client has finished; the pair
// settles on the client, which is the measuring side. harvestPair deleted that
// server INLINE, before combinePair and completeAttempt and therefore before
// the status write — the only Pair site that did not defer through pass.kill,
// and the only one with no comment about the ordering.
//
// This test failed against main when it was written, which is how the defect
// was found: the test infrastructure for #372 caught a real #247 violation
// rather than only documenting coverage.
func TestKillOrder_ALingeringServerIsRecordedBeforeItIsDestroyed(t *testing.T) {
	w := &killOrderWatcher{}
	h := newHarnessWithInterceptor(t, func(inner client.Client) interceptor.Funcs { return w.funcs(inner) },
		gb10Node("spark-a"), gb10Node("spark-b"),
		pairTest("link"),
		profile("acceptance", nil, false, testRef("link")),
		withNodeCap(newRun("run1", "acceptance", "spark-a", "spark-b"), 2),
	)
	h.reconcile("run1")
	h.reconcile("run1")
	server := h.pairPod("run1", pairRoleServer)
	if server == nil {
		t.Fatal("no server pod")
	}
	h.readyPod(server)
	h.reconcile("run1")
	clientPod := h.pairPod("run1", pairRoleClient)
	if clientPod == nil {
		t.Fatal("no client pod")
	}
	h.startPod(clientPod)
	// The CLIENT finishes cleanly; the SERVER is left running. That is the
	// lingering-server case the pair logic documents as legitimate.
	h.finishPod(clientPod, 0, "bandwidthGbps=99.6\n", "Completed")
	for i := 0; i < 5; i++ {
		h.reconcile("run1")
	}
	if w.deletes == 0 {
		t.Fatal("the lingering server was never deleted; path not reached")
	}
	for _, v := range w.violations {
		t.Errorf("issue #247 on the LINGERING SERVER path: %s", v)
	}
}
