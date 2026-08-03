// SPDX-License-Identifier: Apache-2.0
// Copyright the Glimmer authors.

package main

import (
	"net"
	"os"
	"testing"
)

func TestIPv4OfGID(t *testing.T) {
	for _, tc := range []struct {
		name string
		gid  string
		want string
	}{
		// The RoCE v2 IPv4 GID, which is the only kind this runner may use.
		{"ipv4 mapped", "0000:0000:0000:0000:0000:ffff:0afc:a1d1", "10.252.161.209"},
		{"ipv4 mapped second node", "0000:0000:0000:0000:0000:ffff:0afc:a1d0", "10.252.161.208"},
		// Index 0/1 on every RoCE port. Selecting one of these is the mistake
		// that makes a healthy link refuse to connect, so they must be rejected
		// by CONTENT rather than by assuming an index.
		{"link local v6", "fe80:0000:0000:0000:4ebb:47ff:fe2d:85af", ""},
		{"garbage", "not-a-gid", ""},
		{"empty", "", ""},
	} {
		t.Run(tc.name, func(t *testing.T) {
			got := ipv4OfGID(tc.gid)
			if tc.want == "" {
				if got != nil {
					t.Fatalf("ipv4OfGID(%q) = %v, want nil", tc.gid, got)
				}
				return
			}
			if got == nil || !got.Equal(net.ParseIP(tc.want)) {
				t.Fatalf("ipv4OfGID(%q) = %v, want %s", tc.gid, got, tc.want)
			}
		})
	}
}

func TestPortActive(t *testing.T) {
	if !(rdmaPort{State: "ACTIVE"}).active() {
		t.Error("ACTIVE should be active")
	}
	if (rdmaPort{State: "DOWN"}).active() {
		t.Error("DOWN should not be active")
	}
	if (rdmaPort{State: "INIT"}).active() {
		t.Error("INIT should not be active")
	}
}

// sysfs reports the port state as "4: ACTIVE". The enum number is a kernel
// detail; only the name is stable, so readTrimmed strips it.
func TestReadTrimmedStripsTheSysfsEnum(t *testing.T) {
	dir := t.TempDir()
	write := func(name, content string) string {
		p := dir + "/" + name
		if err := os.WriteFile(p, []byte(content), 0o600); err != nil {
			t.Fatal(err)
		}
		return p
	}
	if got := readTrimmed(write("state", "4: ACTIVE\n")); got != "ACTIVE" {
		t.Errorf("state = %q, want ACTIVE", got)
	}
	if got := readTrimmed(write("link_layer", "Ethernet\n")); got != "Ethernet" {
		t.Errorf("link_layer = %q, want Ethernet", got)
	}
	if got := readTrimmed(dir + "/absent"); got != "" {
		t.Errorf("absent file = %q, want empty", got)
	}
}

// selectPort must never return a DOWN port. A node whose fabric is unplugged is
// a verdict about the fabric, and quietly measuring some other port instead
// would report a pass about a link nobody asked about.
func TestSelectPortRefusesWhenNothingIsActive(t *testing.T) {
	ports := []rdmaPort{
		{Device: "rocep1s0f0", Port: 1, State: "DOWN", LinkLayer: linkLayerEthernet, GIDIndex: -1},
		{Device: "rocep1s0f1", Port: 1, State: "INIT", LinkLayer: linkLayerEthernet, GIDIndex: -1},
	}
	if _, _, err := selectPort(ports, nil); err == nil {
		t.Fatal("selectPort accepted a set of ports with none ACTIVE")
	}
}

func TestSelectPortWithNoDevicesIsSkip(t *testing.T) {
	_, _, err := selectPort(nil, nil)
	if err != errNoRDMA {
		t.Fatalf("err = %v, want errNoRDMA (the SKIP condition)", err)
	}
}

// An InfiniBand port has no GID index to find and must still be selectable; the
// Ethernet fallback path requires a GID, the InfiniBand one must not.
func TestSelectPortAcceptsInfiniBandWithoutAGID(t *testing.T) {
	ports := []rdmaPort{{Device: "mlx5_0", Port: 1, State: "ACTIVE", LinkLayer: "InfiniBand", GIDIndex: -1}}
	got, _, err := selectPort(ports, nil)
	if err != nil {
		t.Fatalf("selectPort: %v", err)
	}
	if got.Device != "mlx5_0" || got.GIDIndex != -1 {
		t.Errorf("got %v, want mlx5_0 with no GID index", got)
	}
}
