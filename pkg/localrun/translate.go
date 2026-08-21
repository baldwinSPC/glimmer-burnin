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
	"github.com/baldwinSPC/glimmer-burnin/pkg/plan"
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

	// Variant axes, injected BEFORE the test's own env so an explicit setting
	// still wins — the same ordering the operator uses, where variantEnv is
	// appended before spec.Runner.Env and Kubernetes takes the last value.
	//
	// This is the half of variant support that reaches the RUNNER. A cell that
	// planned correctly but never received BURNIN_VARIANT_PRECISION would run
	// the default configuration and be reported as the fp4 cell — which is
	// exactly the failure plan.RefuseUnreachableAxes exists to prevent, arrived
	// at by forgetting to inject rather than by an unusable name.
	//
	// Which axes reach the runner, and under what name, is pkg/plan's decision,
	// not this file's. An axis whose name cannot become an environment variable
	// is skipped here as the backstop; cmd/burnin refuses such a plan outright.
	for _, e := range plan.VariantEnv(t.Axes) {
		spec.Env[e.Name] = e.Value
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
			v, ok, err := ResolveEnv(p, e)
			if err != nil {
				return RunSpec{}, err
			}
			if !ok {
				// Not resolvable here, and therefore NOT SET — never set to "".
				// cmd/burnin warns about exactly these, from the same function,
				// so the omission is announced rather than discovered by a
				// runner. See ResolveEnv.
				continue
			}
			spec.Env[e.Name] = v
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

// Downward-API field paths this dispatcher can answer for itself.
//
// Deliberately the two that name the MACHINE, and no more. Everything else a
// fieldRef can select — metadata.name, metadata.namespace, status.podIP,
// spec.serviceAccountName — is a fact about a POD, and there is no pod here.
// Inventing values for them would be inventing a cluster.
const (
	fieldHostIP   = "status.hostIP"
	fieldNodeName = "spec.nodeName"
)

// ResolveEnv answers what one of a test's declared environment variables is
// worth on this machine.
//
// It returns (value, true, nil) for a variable this dispatcher can set,
// (_, false, nil) for one it cannot, and an error for one it SHOULD be able to
// set and could not.
//
// # The bug this exists to end
//
// A BurnInTest may set spec.runner.env with a valueFrom rather than a value —
// `status.hostIP` through the Downward API is the documented way for a runner
// to learn the address its peers reach it on, and it survives podForTest
// verbatim because Kubernetes resolves it. This dispatcher copied e.Value, and
// e.Value on a valueFrom entry is the EMPTY STRING. So the variable was set,
// and set to nothing: a runner that checked `if [ -n "$HOST_IP" ]` took the
// wrong branch, and one that used it built an address of ":9000". The same
// BurnInTest worked in a cluster, which is the shape of failure this project's
// parity ledger exists to catch.
//
// # Why absence rather than empty
//
// A variable this cannot resolve is NOT SET AT ALL, and cmd/burnin warns about
// it by name. That is the same rule the runners emit metrics under: absence is
// not a declaration, and a value nobody established must never be presented as
// one. A runner meeting an unset variable can fail loudly or skip honestly; a
// runner meeting an empty one cannot tell the difference between "the cluster
// says this is blank" and "nobody ever asked".
//
// # Why status.hostIP is an ERROR when unknown
//
// The two paths below are questions about the machine, and this dispatcher is
// standing on the machine. If it cannot answer status.hostIP, the host has no
// routable address at all, and a test that asked for it cannot do what it was
// written to do. That is refused at plan time, naming the variable — not
// silently omitted, which would leave the same hole this function closes.
func ResolveEnv(p Plan, e corev1.EnvVar) (string, bool, error) {
	if e.ValueFrom == nil {
		return e.Value, true, nil
	}
	ref := e.ValueFrom.FieldRef
	if ref == nil {
		// secretKeyRef, configMapKeyRef, resourceFieldRef: each names an object
		// in a namespace, and there is no apiserver here to hold one.
		return "", false, nil
	}

	switch ref.FieldPath {
	case fieldHostIP:
		if p.HostIP == "" {
			return "", false, fmt.Errorf(
				"env %q asks for %s and this machine has no routable address to answer with — "+
					"pkg/hostinfo found neither a default route nor a global unicast address. "+
					"Set one, or give the variable a literal value for this run",
				e.Name, fieldHostIP)
		}
		return p.HostIP, true, nil

	case fieldNodeName:
		if p.Node == "" {
			return "", false, fmt.Errorf(
				"env %q asks for %s and this run has no node name; pass --node", e.Name, fieldNodeName)
		}
		return p.Node, true, nil
	}
	return "", false, nil
}

// UnresolvableEnv names the variables a test declares that this dispatcher will
// not set, each with the reason, so cmd/burnin can warn before anything runs.
//
// It asks ResolveEnv rather than restating its rules, so a path that becomes
// resolvable stops being warned about in the same commit — a warning that
// outlived its cause would send someone looking for a problem that is fixed.
// An ERROR from ResolveEnv is not listed: that is a refusal, and Translate
// raises it as one.
func UnresolvableEnv(p Plan, spec api.BurnInTestSpec) []string {
	if spec.Runner == nil {
		return nil
	}
	var out []string
	for _, e := range spec.Runner.Env {
		if e.ValueFrom == nil {
			continue
		}
		if _, ok, err := ResolveEnv(p, e); ok || err != nil {
			continue
		}
		out = append(out, fmt.Sprintf("%s (%s)", e.Name, describeEnvSource(e.ValueFrom)))
	}
	return out
}

// describeEnvSource names what a valueFrom selects, in the words the profile
// used, so the warning points at the line the author wrote.
func describeEnvSource(src *corev1.EnvVarSource) string {
	switch {
	case src.FieldRef != nil:
		return "fieldRef " + src.FieldRef.FieldPath
	case src.SecretKeyRef != nil:
		return "secretKeyRef " + src.SecretKeyRef.Name + "/" + src.SecretKeyRef.Key
	case src.ConfigMapKeyRef != nil:
		return "configMapKeyRef " + src.ConfigMapKeyRef.Name + "/" + src.ConfigMapKeyRef.Key
	case src.ResourceFieldRef != nil:
		return "resourceFieldRef " + src.ResourceFieldRef.Resource
	}
	return "valueFrom"
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
