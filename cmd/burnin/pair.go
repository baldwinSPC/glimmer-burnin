package main

import (
	"context"
	"fmt"
	"io"
	"net"
	"time"

	api "github.com/baldwinSPC/glimmer-burnin/api/v1alpha1"
	"github.com/baldwinSPC/glimmer-burnin/pkg/localrun"
)

// A link test needs two machines, and at enrollment time neither is a cluster
// member — which makes a fabric fault the defect most worth catching before a
// node joins, and the one Path B structurally cannot catch early.
//
// This is plumbing, not protocol. The runners already carry their own TCP
// control channel: the server never resolves its peer (it takes
// conn.RemoteAddr()) and the client dials whatever address it is handed. The
// operator supplies a cluster DNS name because it has one, not because the
// runner requires it, so substituting a plain IP is the whole change.

// pairFlags are the two-host flags.
type pairFlags struct {
	role     string
	peer     string
	peerNode string
}

// rendezvous validates the flags and builds what the plan needs.
//
// Validation is strict rather than forgiving because every mistake it catches
// would otherwise surface as a connection error, and a connection error in a
// fabric test reads as a bad link. Refusing at the command line keeps a typo
// from being recorded as hardware evidence.
func (f pairFlags) rendezvous() (*localrun.Rendezvous, error) {
	switch f.role {
	case "":
		if f.peer != "" || f.peerNode != "" {
			return nil, fmt.Errorf("--peer needs --role: which end of the link is this machine?")
		}
		return nil, nil

	case localrun.RoleServer:
		if f.peer != "" {
			return nil, fmt.Errorf(
				"--role server takes no --peer: the server learns the client's address when the client connects")
		}

	case localrun.RoleClient:
		if f.peer == "" {
			return nil, fmt.Errorf("--role client needs --peer <ip|host>: the address of the server end")
		}

	default:
		return nil, fmt.Errorf("--role is %q; it is %s or %s", f.role, localrun.RoleServer, localrun.RoleClient)
	}

	return &localrun.Rendezvous{
		Role:     f.role,
		PeerHost: f.peer,
		PeerNode: f.peerNode,
	}, nil
}

// deciding reports whether this end produces the link's verdict.
//
// The client is the deciding side, here as at Pair scope in the operator: it is
// where perftest and nccl-tests report. A server that also emitted a verdict
// would be a second claim about one measurement, and the two would be compared.
// A GROUP RANK NEVER DECIDES. rank 0 holds the bandwidth figure, but "did every
// rank take part" is a fact no rank can see — so a collective's verdict is
// rendered by `burnin merge`, which is the only thing that can count to N. See
// groupDeciding.
func deciding(rz *localrun.Rendezvous) bool {
	if rz == nil {
		return true
	}
	if rz.Rank != nil {
		return groupDeciding(rz)
	}
	return rz.Role == localrun.RoleClient
}

// announceServerReady watches for the server's listener and says when it opens.
//
// Start ordering is the caller's on bare metal — there is no scheduler to gate
// the client on the server's readiness. So the server end prints a line a script
// can wait for, from the SAME readinessProbe the operator uses in-cluster, which
// makes that field mean something here rather than being ignored.
//
// It is an affordance, not the gate. The runners' connect-with-retry is the real
// one, exactly as in a cluster: a client that starts first must retry into
// success, never report a fabric fault.
func announceServerReady(ctx context.Context, w io.Writer, plan localrun.Plan) {
	port := probePort(plan)
	if port == 0 {
		// No probe to watch. Said plainly rather than silently skipped: the
		// caller is about to start a client by hand and deserves to know that
		// nothing here will tell them when.
		fmt.Fprintf(w, "burnin: no readinessProbe on the pair test — nothing can report when the server is listening;\n"+
			"        add a tcpSocket probe on the runner's port, or start the client and let it retry\n")
		return
	}

	go func() {
		addr := net.JoinHostPort("127.0.0.1", fmt.Sprint(port))
		for ctx.Err() == nil {
			conn, err := net.DialTimeout("tcp", addr, time.Second)
			if err == nil {
				conn.Close()
				fmt.Fprintf(w, "burnin: server is listening on :%d — start the client now\n", port)
				return
			}
			select {
			case <-ctx.Done():
				return
			case <-time.After(500 * time.Millisecond):
			}
		}
	}()
}

// probePort finds the tcpSocket port of the first Pair-scope test that declares
// one. Numeric ports only: a named port is resolved against a container's ports
// by a kubelet, and there is no kubelet here to do it.
func probePort(plan localrun.Plan) int32 {
	for _, t := range plan.Tests {
		if t.Spec.Scope != api.ScopePair || t.Spec.Runner == nil {
			continue
		}
		p := t.Spec.Runner.ReadinessProbe
		if p == nil || p.TCPSocket == nil {
			continue
		}
		if v := p.TCPSocket.Port.IntValue(); v > 0 {
			return int32(v)
		}
	}
	return 0
}

// pairScoped reports whether the profile has anything a rendezvous applies to.
func pairScoped(plan localrun.Plan) bool {
	for _, t := range plan.Tests {
		if t.Spec.Scope == api.ScopePair || t.Spec.Scope == api.ScopeGroup {
			return true
		}
	}
	return false
}
