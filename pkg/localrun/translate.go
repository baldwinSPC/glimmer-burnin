package localrun

import (
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"time"

	corev1 "k8s.io/api/core/v1"

	api "github.com/baldwinSPC/glimmer-burnin/api/v1alpha1"
	"github.com/baldwinSPC/glimmer-burnin/pkg/runnerimages"
)

// RunSpec is one container invocation, described in terms a runtime adapter can
// turn into flags.
//
// Deliberately not a docker command line. Keeping it structured means the
// podman and nerdctl adapters differ only where the runtimes genuinely differ,
// and it makes the translation testable without running anything.
type RunSpec struct {
	Image string
	// Command and Args override the image's entrypoint, when the test says so.
	Command []string
	Args    []string
	Env     map[string]string
	Mounts  []Mount
	Devices []string
	// HostNetwork puts the container in the host's network namespace. Required
	// for anything reading /sys/class/net, and it removes port mapping from the
	// picture for a fabric test.
	HostNetwork bool
	Privileged  bool
	// GPUAccess says which accelerator the container needs, if any.
	GPUAccess GPUAccess
	// UnlimitedMemlock raises RLIMIT_MEMLOCK. RDMA registration needs it, and
	// unlike Kubernetes — where there is no PodSpec field for it at all — a
	// container runtime can simply be told.
	UnlimitedMemlock bool
	// Timeout kills the container. Mirrors activeDeadlineSeconds: the test's
	// duration plus a grace period, so a runner that hangs cannot hang the run.
	Timeout time.Duration
}

// GPUAccess is which accelerators to expose.
type GPUAccess string

const (
	// GPUNone requests no accelerator. A CPU-only test gets no device access at
	// all, which is the same posture as the operator's: nothing is granted that
	// was not asked for.
	GPUNone GPUAccess = ""
	// GPUNvidia requests NVIDIA devices, via the container toolkit or CDI.
	GPUNvidia GPUAccess = "nvidia"
	// GPUAMD requests AMD devices, which is a pair of device nodes rather than
	// a runtime integration.
	GPUAMD GPUAccess = "amd"
)

// Mount is a host path made visible to the container.
type Mount struct {
	Source   string
	Target   string
	ReadOnly bool
}

// vendorResources maps an extended-resource name to the access it implies.
//
// A table rather than a switch on the test kind, for the reason the operator
// keeps vendor behaviour out of its reconciler: a new accelerator should be a
// new row, not a new branch. The resource name is what a profile already writes
// to request a device in-cluster, so the same profile means the same thing here.
var vendorResources = map[corev1.ResourceName]GPUAccess{
	"nvidia.com/gpu": GPUNvidia,
	"amd.com/gpu":    GPUAMD,
}

// vendorResourcePrefixes are count-shaped resource FAMILIES: a MIG profile is
// requested as nvidia.com/mig-1g.5gb, and there is one name per profile. A
// profile written for a MIG-sliced cluster must mean the same thing here as
// there — the runner's own allow-set (device_fold.h, nvidiaResources) counts
// these as instances, and runners/devicefold_test.go holds the two tables to
// each other — so they are recognised by prefix rather than listed.
var vendorResourcePrefixes = map[string]GPUAccess{
	"nvidia.com/mig-": GPUNvidia,
}

// accessFor maps one resource name to the accelerator access it implies, or
// GPUNone.
func accessFor(name corev1.ResourceName) GPUAccess {
	if access, ok := vendorResources[name]; ok {
		return access
	}
	for prefix, access := range vendorResourcePrefixes {
		if strings.HasPrefix(string(name), prefix) {
			return access
		}
	}
	return GPUNone
}

// memlockTriggers are host paths whose presence means the test does RDMA.
//
// Data-driven rather than a list of kinds: any test that reaches the verbs
// devices needs the limit raised, including one this package has never heard of.
// Keying on the declared mount is how that stays true for a third-party runner.
var memlockTriggers = []string{"/dev/infiniband"}

// Translate turns a planned test into a container invocation.
//
// It mirrors podForTest, which is the one place the operator builds a runner
// pod, and the mirroring is the point: a test that gets /dev/kmsg read-only in
// a cluster must get /dev/kmsg read-only here, or the two dispatchers are
// measuring different things.
func Translate(p Plan, t PlannedTest) (RunSpec, error) {
	image, err := resolveImage(t.Spec, p.Vendor)
	if err != nil {
		return RunSpec{}, err
	}

	duration := durationSeconds(t.Spec)
	spec := RunSpec{
		Image:       image,
		HostNetwork: t.Spec.HostNetwork,
		Env: map[string]string{
			"BURNIN_DURATION_SECONDS": fmt.Sprint(duration),
		},
		// The same grace the operator gives a pod beyond its declared duration,
		// so a runner that would have been killed in-cluster is killed here too.
		Timeout: time.Duration(duration+deadlineGraceSeconds) * time.Second,
	}
	if t.Spec.Runner != nil {
		spec.Command = t.Spec.Runner.Command
		spec.Args = t.Spec.Runner.Args
		spec.Privileged = t.Spec.Runner.Privileged

		for _, m := range t.Spec.Runner.HostPaths {
			mount := Mount{
				Source:   m.Path,
				Target:   m.MountPath,
				ReadOnly: m.ReadOnly == nil || *m.ReadOnly, // defaults to true
			}
			spec.Mounts = append(spec.Mounts, mount)

			// A character device has to be granted through the runtime's device
			// cgroup, not only bind-mounted, or the container sees the node and
			// cannot open it.
			if m.Type != nil && *m.Type == corev1.HostPathCharDev {
				spec.Devices = append(spec.Devices, m.Path)
			}
			// AND SO DOES EVERY CHARACTER DEVICE INSIDE A MOUNTED DIRECTORY.
			//
			// The rule above keys on the declared type, which is right as far as
			// it goes and covers /dev/kmsg. But the case this project actually
			// ships is a DIRECTORY of devices: pair-network-acceptance.yaml
			// declares /dev/infiniband as `type: Directory`, because uverbs
			// numbering is per-node and not even symmetric across a cable, so no
			// fixed list of nodes is right everywhere.
			//
			// Bind-mounting that directory makes uverbs0-3 visible and leaves
			// them unopenable — `Operation not permitted` — so perftest failed
			// with "Couldn't get context for the device" and no fabric test could
			// run on bare metal. In-cluster the same BurnInTest works, which is
			// how the documented, shipped configuration came to be broken on one
			// dispatcher only (#289).
			//
			// Only the immediate children are granted. Recursing would sweep in
			// by-ibdev/ and by-path/, which hold symlinks to the same nodes, and
			// granting a path twice is noise at best.
			spec.Devices = append(spec.Devices, charDevicesIn(m.Path)...)
			for _, trigger := range memlockTriggers {
				if strings.HasPrefix(m.Path, trigger) {
					spec.UnlimitedMemlock = true
				}
			}
		}

		// The test's own environment is applied LAST so it wins, matching the
		// operator's ordering — except that the operator's injected variables
		// win there. Kept identical here: a profile cannot override the
		// contract's own variables in either dispatcher.
		for _, e := range t.Spec.Runner.Env {
			if _, reserved := reservedEnv[e.Name]; reserved {
				continue
			}
			spec.Env[e.Name] = e.Value
		}
	}

	// Rendezvous, for a test whose scope actually has one.
	//
	// Gated on SCOPE, not merely on the flag being present: a Node-scope test in
	// a profile run with --role must not receive BURNIN_ROLE. A runner reading
	// that variable concludes it is one end of a link, and a single-machine test
	// told it is half of a pair will wait for a peer that does not exist.
	if p.Rendezvous != nil {
		switch t.Spec.Scope {
		case api.ScopePair:
			spec.Env["BURNIN_ROLE"] = p.Rendezvous.Role
			setIf(spec.Env, "BURNIN_PEER_HOST", p.Rendezvous.PeerHost)
			setIf(spec.Env, "BURNIN_PEER_NODE", p.Rendezvous.PeerNode)
		case api.ScopeGroup:
			if p.Rendezvous.Rank != nil {
				spec.Env["BURNIN_RANK"] = fmt.Sprint(*p.Rendezvous.Rank)
			}
			if p.Rendezvous.NRanks > 0 {
				spec.Env["BURNIN_NRANKS"] = fmt.Sprint(p.Rendezvous.NRanks)
			}
			setIf(spec.Env, "BURNIN_ROOT_HOST", p.Rendezvous.RootHost)
			setIf(spec.Env, "BURNIN_ROOT_NODE", p.Rendezvous.RootNode)
		}

		// hostNetwork for a fabric test on bare metal, unless the test already
		// asked for it. The RDMA runners want it, and it takes port mapping out
		// of the picture entirely — a NAT'd control channel is a connection
		// error that reads as a bad link.
		if t.Spec.Scope == api.ScopePair || t.Spec.Scope == api.ScopeGroup {
			spec.HostNetwork = true
		}
	}

	// Accelerator access from the resource request, so a profile written for the
	// cluster asks for a device the same way here.
	for name := range t.Spec.Resources.Limits {
		if access := accessFor(name); access != GPUNone {
			spec.GPUAccess = access
			break
		}
	}
	if spec.GPUAccess == GPUNone {
		for name := range t.Spec.Resources.Requests {
			if access := accessFor(name); access != GPUNone {
				spec.GPUAccess = access
				break
			}
		}
	}

	sort.Slice(spec.Mounts, func(i, j int) bool { return spec.Mounts[i].Target < spec.Mounts[j].Target })
	sort.Strings(spec.Devices)
	return spec, nil
}

// reservedEnv are the variables the contract owns. A test cannot set them.
//
// Not a stylistic rule: BURNIN_ROLE decides which end of a link a runner is, and
// a profile that could set it would be able to make both ends the client.

// charDevicesIn lists the character devices directly inside a directory.
//
// Reads the host filesystem, which makes Translate environment-dependent — so it
// is a seam (statDir) that tests replace, and it returns nothing rather than an
// error for a path that is not a directory or cannot be read. A missing device
// is the runtime's to report: refusing to translate here would turn "this node
// has no RDMA" into a CLI error instead of the runner's own honest Skip.
var statDir = func(dir string) ([]os.DirEntry, error) { return os.ReadDir(dir) }

func charDevicesIn(dir string) []string {
	entries, err := statDir(dir)
	if err != nil {
		return nil
	}
	var out []string
	for _, e := range entries {
		info, err := e.Info()
		if err != nil || info.Mode()&os.ModeCharDevice == 0 {
			continue
		}
		out = append(out, filepath.Join(dir, e.Name()))
	}
	return out
}

var reservedEnv = map[string]struct{}{
	"BURNIN_DURATION_SECONDS": {},
	"BURNIN_ATTEMPT":          {},
	"BURNIN_ROLE":             {},
	"BURNIN_PEER_HOST":        {},
	"BURNIN_PEER_NODE":        {},
	"BURNIN_RANK":             {},
	"BURNIN_NRANKS":           {},
	"BURNIN_ROOT_HOST":        {},
	"BURNIN_ROOT_NODE":        {},
	// The pod's own limits, which a runner reads to tell allocated devices from
	// visible ones. A profile that could set it would tell a runner it owns a
	// board it was never handed — the spoof the variable exists to prevent.
	// The CLI does not inject it (on bare metal `--gpus all` IS the
	// allocation, and absent is the safe reading there); it only refuses it.
	"BURNIN_RESOURCE_LIMITS": {},
}

// deadlineGraceSeconds matches the operator's own grace beyond the declared
// duration, so a hung runner is killed on the same schedule either way.
const deadlineGraceSeconds = 120

// defaultDurationSeconds is what a test that declares none gets, matching the
// operator's default.
const defaultDurationSeconds = 600

// durationSeconds is the WHOLE duration, and it deliberately ignores spec.soak.
//
// The operator divides a soak into segments so that an eviction, a drain or a
// kubelet restart costs one window instead of the week — and every one of those
// is a thing a cluster does. There is no cluster here: this path runs one
// container per test, on a box somebody is sitting in front of, so segmenting it
// would buy nothing and cost a pod-shaped restart the runtime never needed.
//
// The two dispatchers still reach the same verdict from the same evidence, which
// is the promise pkg/localrun exists under, and the aggregation rules are exactly
// what make that true: elapsedS SUMS, so 672 windows of 900 seconds is the same
// 604,800 as one; a floor takes the worst window and a ceiling the highest, which
// is what a single full-length runner reports anyway; and a lifetime total takes
// the last reading either way. A segmented soak and one long execution are the
// same measurement, which is the property that made segmenting safe in the first
// place.
func durationSeconds(spec api.BurnInTestSpec) int32 {
	if spec.DurationSeconds > 0 {
		return spec.DurationSeconds
	}
	return defaultDurationSeconds
}

// resolveImage picks the image, in the operator's own order.
// resolveImage picks this test's image for THIS HOST's accelerator vendor.
//
// It calls the same pkg/runnerimages.Resolve the operator calls, with the same
// arguments, because the alternative is what shipped: this function ignored
// spec.runner.imagesByVendor entirely, so a host the operator would have sent to
// a ROCm image ran the NVIDIA default instead. One brain, two dispatchers — and
// image selection is part of the brain.
func resolveImage(spec api.BurnInTestSpec, vendor string) (string, error) {
	return runnerimages.Resolve(spec.Kind, spec.Runner, vendor)
}

// setIf sets a variable only when there is a value.
//
// An empty BURNIN_PEER_HOST is worse than none: a runner that checks presence
// rather than content would dial the empty string.
func setIf(env map[string]string, k, v string) {
	if v != "" {
		env[k] = v
	}
}

// VendorResourceNames lists the extended-resource names this package treats as
// an accelerator request, sorted. Exported for the guard that holds this table
// and the runners' own allow-sets (device_fold.h, nvidiaResources /
// amdResources) to the same names: a resource the CLI maps to a vendor that a
// runner then refuses as unrecognised — or the reverse — is one profile meaning
// two things across the two dispatchers.
func VendorResourceNames() []string {
	names := make([]string, 0, len(vendorResources))
	for n := range vendorResources {
		names = append(names, string(n))
	}
	sort.Strings(names)
	return names
}

// VendorResourcePrefixes lists the count-shaped resource-name prefixes this
// package treats as an accelerator request, sorted. Same guard, same reason.
func VendorResourcePrefixes() []string {
	out := make([]string, 0, len(vendorResourcePrefixes))
	for p := range vendorResourcePrefixes {
		out = append(out, p)
	}
	sort.Strings(out)
	return out
}
