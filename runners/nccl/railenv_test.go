// SPDX-License-Identifier: Apache-2.0
// Copyright the Glimmer authors.

package main

import (
	"strings"
	"testing"
)

// TestRailEnv_SingleRailMatchesThePreviousSingleDeviceShape guards backward
// compatibility: a one-element rail list (the single-rail SKU's fallback
// path, or any point-to-point selectPort answer) must produce exactly the
// same NCCL_IB_HCA=<device>:<port> shape ncclEnv always emitted before #489.
func TestRailEnv_SingleRailMatchesThePreviousSingleDeviceShape(t *testing.T) {
	env := railEnv([]rdmaPort{{Device: "roceP2p1s0f1", Port: 1, NetDev: "enP2p1s0f1np1", GIDIndex: 3}})
	want := map[string]string{
		"NCCL_IB_HCA":        "roceP2p1s0f1:1",
		"NCCL_SOCKET_IFNAME": "enP2p1s0f1np1",
		"NCCL_IB_GID_INDEX":  "3",
	}
	assertEnv(t, env, want)
}

// TestRailEnv_MultiRailJoinsHCAsWithCommas is the actual point of #489: two
// rails must produce ONE comma-joined NCCL_IB_HCA naming both, which is the
// syntax NCCL's own multi-rail striping reads.
func TestRailEnv_MultiRailJoinsHCAsWithCommas(t *testing.T) {
	env := railEnv([]rdmaPort{
		{Device: "roceA", Port: 1, NetDev: "roceAnp0", GIDIndex: 3},
		{Device: "roceB", Port: 1, NetDev: "roceBnp0", GIDIndex: 3},
	})
	hca := lookupEnv(t, env, "NCCL_IB_HCA")
	if hca != "roceA:1,roceB:1" {
		t.Errorf("NCCL_IB_HCA = %q, want %q", hca, "roceA:1,roceB:1")
	}
	// The bootstrap ring names exactly one interface — the first rail's — not
	// a list. It is a small control connection, not the striped data path.
	if ifname := lookupEnv(t, env, "NCCL_SOCKET_IFNAME"); ifname != "roceAnp0" {
		t.Errorf("NCCL_SOCKET_IFNAME = %q, want the first rail's netdev %q", ifname, "roceAnp0")
	}
}

// TestRailEnv_NoGIDIndexWhenUnresolved asserts railEnv leaves
// NCCL_IB_GID_INDEX unset when the (already-agreed, by classifyRails' own
// contract) index is -1 — the "nothing to pin" case — rather than emitting a
// nonsensical NCCL_IB_GID_INDEX=-1.
func TestRailEnv_NoGIDIndexWhenUnresolved(t *testing.T) {
	env := railEnv([]rdmaPort{{Device: "roceA", Port: 1, NetDev: "roceAnp0", GIDIndex: -1}})
	for _, e := range env {
		if strings.HasPrefix(e, "NCCL_IB_GID_INDEX=") {
			t.Errorf("NCCL_IB_GID_INDEX set to %q when no GID resolved", e)
		}
	}
}

func lookupEnv(t *testing.T, env []string, key string) string {
	t.Helper()
	for _, e := range env {
		if v, ok := strings.CutPrefix(e, key+"="); ok {
			return v
		}
	}
	t.Fatalf("no %s in %v", key, env)
	return ""
}

func assertEnv(t *testing.T, env []string, want map[string]string) {
	t.Helper()
	for k, v := range want {
		if got := lookupEnv(t, env, k); got != v {
			t.Errorf("%s = %q, want %q", k, got, v)
		}
	}
}
