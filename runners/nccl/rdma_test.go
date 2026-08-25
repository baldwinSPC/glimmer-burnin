// SPDX-License-Identifier: Apache-2.0
// Copyright the Glimmer authors.

package main

import (
	"net"
	"testing"
)

func rail(device string, gidIndex int, addr string, maskBits int) railPort {
	return railPort{
		port: rdmaPort{
			Device:    device,
			Port:      1,
			NetDev:    device + "np" + device, // arbitrary, unused by classifyRails
			LinkLayer: linkLayerEthernet,
			State:     "ACTIVE",
			GIDIndex:  gidIndex,
		},
		addr: net.ParseIP(addr),
		mask: net.CIDRMask(maskBits, 32),
	}
}

// TestClassifyRails_PicksTheLargestMultiMemberSubnet is the core regression
// test for #489: two rails on a shared /29 (the collective) plus one lone
// address on a separate /31 (the cluster/etcd-peer cable, addressed but never
// shared with a second local RDMA netdev) must select ONLY the two-member
// group — never the lone one, and never all three.
func TestClassifyRails_PicksTheLargestMultiMemberSubnet(t *testing.T) {
	candidates := []railPort{
		rail("roceA", 3, "10.252.217.104", 29),
		rail("roceB", 3, "10.252.217.105", 29),
		rail("roceC", 3, "10.252.145.134", 31), // cluster/etcd-peer cable, alone on its subnet
	}
	got, why, err := classifyRails(candidates)
	if err != nil {
		t.Fatalf("classifyRails: %v", err)
	}
	if len(got) != 2 {
		t.Fatalf("selected %d rails, want 2: %+v", len(got), got)
	}
	names := map[string]bool{got[0].Device: true, got[1].Device: true}
	if !names["roceA"] || !names["roceB"] {
		t.Errorf("selected %v, want exactly roceA and roceB (the shared /29) — not the lone /31", names)
	}
	if why == "" {
		t.Error("no explanation returned for a real selection")
	}
}

// TestClassifyRails_NoMultiMemberSubnetFallsThrough asserts the single-rail
// SKU case changes nothing: every candidate alone on its own subnet must
// return (nil, "", nil) so the caller falls back to selectPort's existing
// single-device answer.
func TestClassifyRails_NoMultiMemberSubnetFallsThrough(t *testing.T) {
	candidates := []railPort{
		rail("roceA", 3, "10.252.217.104", 30),
		rail("roceB", 3, "10.252.145.134", 31),
	}
	got, why, err := classifyRails(candidates)
	if err != nil {
		t.Fatalf("classifyRails: %v", err)
	}
	if got != nil {
		t.Errorf("selected %v on a node with no multi-member subnet, want nil", got)
	}
	if why != "" {
		t.Errorf("non-empty explanation %q for a nil selection", why)
	}
}

// TestClassifyRails_EmptyInputFallsThrough is the degenerate case: no
// candidates at all (a node with no active Ethernet RDMA port).
func TestClassifyRails_EmptyInputFallsThrough(t *testing.T) {
	got, _, err := classifyRails(nil)
	if err != nil || got != nil {
		t.Errorf("classifyRails(nil) = %v, %v, want nil, nil", got, err)
	}
}

// TestClassifyRails_RefusesOnDisagreeingGIDIndex is the safety property this
// whole design leans on: NCCL_IB_GID_INDEX is one global value, so a rail
// group whose members resolved DIFFERENT indices must be refused entirely —
// not pinned at one of the two and hoping — falling back to selectPort
// exactly as if there were no qualifying subnet at all.
func TestClassifyRails_RefusesOnDisagreeingGIDIndex(t *testing.T) {
	candidates := []railPort{
		rail("roceA", 3, "10.252.217.104", 29),
		rail("roceB", 5, "10.252.217.105", 29), // different GID index
	}
	got, why, err := classifyRails(candidates)
	if err != nil {
		t.Fatalf("classifyRails: %v", err)
	}
	if got != nil || why != "" {
		t.Errorf("classifyRails did not refuse a GID-index disagreement: got=%v why=%q", got, why)
	}
}

// TestClassifyRails_ProceedsWhenNeitherRailHasAResolvedGID covers pure
// InfiniBand-shaped Ethernet ports (no RoCE v2 GID resolvable at all): with
// nothing to disagree about, the group is still valid — NCCL_IB_GID_INDEX
// simply will not be set, exactly as the single-device path already does
// when no GID resolves.
func TestClassifyRails_ProceedsWhenNeitherRailHasAResolvedGID(t *testing.T) {
	candidates := []railPort{
		rail("roceA", -1, "10.252.217.104", 29),
		rail("roceB", -1, "10.252.217.105", 29),
	}
	got, _, err := classifyRails(candidates)
	if err != nil {
		t.Fatalf("classifyRails: %v", err)
	}
	if len(got) != 2 {
		t.Fatalf("selected %d rails, want 2 (no GID resolved is not a disagreement): %+v", len(got), got)
	}
}

// TestClassifyRails_PicksTheLargestGroupWhenMultipleQualify covers a
// theoretical 4-rail node: two separate collective subnets, sizes 2 and 3.
// The larger group wins, deterministically (subnet keys are sorted before
// comparison, so this is not a map-iteration-order accident).
func TestClassifyRails_PicksTheLargestGroupWhenMultipleQualify(t *testing.T) {
	candidates := []railPort{
		rail("roceA", 3, "10.0.0.1", 30),
		rail("roceB", 3, "10.0.0.2", 30),
		rail("roceC", 3, "10.1.0.1", 29),
		rail("roceD", 3, "10.1.0.2", 29),
		rail("roceE", 3, "10.1.0.3", 29),
	}
	got, _, err := classifyRails(candidates)
	if err != nil {
		t.Fatalf("classifyRails: %v", err)
	}
	if len(got) != 3 {
		t.Fatalf("selected %d rails, want 3 (the larger group): %+v", len(got), got)
	}
}

func TestAgreeingGIDIndex(t *testing.T) {
	tests := []struct {
		name    string
		indices []int
		wantOK  bool
		wantIdx int
	}{
		{"none resolved", []int{-1, -1}, true, -1},
		{"all agree", []int{3, 3, 3}, true, 3},
		{"two disagree", []int{3, 5}, false, -1},
		{"mixed resolved and unresolved", []int{3, -1}, false, -1},
		{"single port, resolved", []int{3}, true, 3},
		{"single port, unresolved", []int{-1}, true, -1},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			var ports []rdmaPort
			for _, idx := range tc.indices {
				ports = append(ports, rdmaPort{GIDIndex: idx})
			}
			idx, ok := agreeingGIDIndex(ports)
			if ok != tc.wantOK {
				t.Errorf("ok = %v, want %v", ok, tc.wantOK)
			}
			if ok && idx != tc.wantIdx {
				t.Errorf("idx = %d, want %d", idx, tc.wantIdx)
			}
		})
	}
}
