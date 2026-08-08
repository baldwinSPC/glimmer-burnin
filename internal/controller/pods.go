package controller

import (
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"strconv"
	"strings"
	"time"

	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"

	burninv1alpha1 "github.com/baldwinSPC/glimmer-burnin/api/v1alpha1"
)

// Pod labels used to find a run's pods and attribute them to a test/node/attempt.
const (
	labelRun  = "burnin.glimmer.ai/run"
	labelTest = "burnin.glimmer.ai/test"
	labelNode = "burnin.glimmer.ai/node"
	// labelAttempt carries the 1-based attempt index. It exists because a test
	// no longer maps to one pod: RepeatCount and RetryOnErrorLimit both mint
	// further executions on the same (test, node), and each one is its own pod
	// with its own exit code and its own log. Without the label, `kubectl get
	// pods -l burnin.glimmer.ai/test=soak` returns a pile of pods with no way
	// to tell the third repeat from the retry that followed an error.
	labelAttempt = "burnin.glimmer.ai/attempt"
	// labelPairRole is "server" or "client" on the two pods of a Pair-scope
	// execution, and is absent at Node scope. It is also the headless Service's
	// selector discriminator, so it is what makes the rendezvous addressable.
	labelPairRole = "burnin.glimmer.ai/pair-role"
	// labelRank carries the 0-based rank on the pods of a Group-scope execution,
	// and is absent at every other scope. It is not a selector — the rendezvous
	// Service matches on (run, test, attempt) and addresses individual pods by
	// hostname — it exists so that `kubectl get pods -l burnin.glimmer.ai/rank=0`
	// finds the root of a collective without anybody decoding a name hash.
	labelRank = "burnin.glimmer.ai/rank"
)

// defaultRunnerImages maps a TestKind to its default runner image. A kind
// without an entry requires spec.runner.image — scheduling a pod with no image
// would produce an ImagePullBackOff blamed on the hardware.
//
// Images are pinned by version, never :latest: a readiness/acceptance verdict
// must be reproducible, and a floating tag changes the test under every
// existing profile with no audit trail.
//
// PUBLICATION STATUS — every entry below is published, public, and immutable.
//
// All eleven images are v0.5.0, published to GHCR, public, and anonymously
// pullable.
//
// THESE TAGS WERE NOT VERIFIED ON REAL HARDWARE BEFORE PUBLICATION, which this
// file's own rule below ("publish only after the kernel has been verified on
// real hardware") requires. The GPU fleet was unavailable and the release was
// cut anyway, deliberately and with the trade understood. What DOES stand behind
// them: the unit, envtest and kind-based e2e tiers, including a real three-node
// rendezvous for Group scope, and four rounds of adversarial review that found
// twenty defects in this release's own code. What does not: no GB10 has run
// these exact tags, NO 3-RANK COLLECTIVE HAS EVER EXECUTED through the N-rank
// path on any hardware, and CI builds runner images by CHANGED DIRECTORY — so
// the runners whose sources did not move are republished from sources CI did not
// rebuild for this release.
//
// The v0.3.0 tags remain published and immutable, and the measurements behind
// THEM were taken on a two-node DGX Spark cluster with the v0.2.0 build of the
// same sources: the Node-scope suite passed 10/10 across both nodes, and the
// Pair-scope fabric suite passed with ib-write-bw at 99.61 Gb/s, nccl at
// 12.02 GB/s bus bandwidth, and gpudirect-rdma correctly reporting exit 2 on
// hardware whose unified memory has no peer-memory provider. Pin v0.3.0 if
// hardware-verified images matter more to you than the v0.4.0 fixes — but read
// the release notes first: two of those fixes are false negatives that were
// certifying hardware nobody measured.
//
// ARCHITECTURE: linux/amd64 AND linux/arm64. A node pulls only its own
// platform's layers, so one tag serves an x86 rack and a Grace rack alike, and
// x86 users no longer need to set spec.runner.image to run at all.
//
// That is the HOST axis and it is not the GPU axis. A tag resolving on a host
// says nothing about whether the image contains code for the accelerator in it —
// compute-smoke carries only the CC 12.x kernel and reports a B200 as an ERROR,
// hardware unjudged — not a Skip, because a B200 does do NVFP4 and the test
// applies to it — and nccl ships one gencode that an unlisted fleet must
// rebuild. Each runner's README states its
// GPU coverage; conflating the two is how an operator ends up reporting Error on
// hardware it simply was not built for.
//
// A missing entry fails fast and legibly at plan time ("set spec.runner.image");
// a present-but-unpullable entry fails slowly, per node, and looks like an
// infrastructure fault rather than a configuration one. That asymmetry is why
// nothing is listed here before it is actually published.
//
// The names follow the publish workflow's own construction —
// ghcr.io/<owner lowercased>/glimmer-burnin-<kind>:<version> — so an entry here
// and a workflow_dispatch of publish-runner with the same version agree by
// construction rather than by anyone remembering to keep them in step.
var defaultRunnerImages = map[burninv1alpha1.TestKind]string{
	// compute-smoke moves off v0.1.0 deliberately. That tag is public and
	// immutable and stays exactly as it is, but it reports "no usable CUDA
	// device", a wrong-arch image, and every CUDA runtime error as exit 1 —
	// a hardware FAILURE verdict against the node. v0.2.0 was the first build
	// where those are exit 3, Error, hardware unjudged. Anyone still pinning
	// v0.1.0 keeps the old behaviour, which is why those are new tags and not
	// repushed ones.
	burninv1alpha1.KindComputeSmoke: "ghcr.io/baldwinspc/glimmer-burnin-compute-smoke:v0.5.0",

	burninv1alpha1.KindClockProbe:   "ghcr.io/baldwinspc/glimmer-burnin-clockprobe:v0.5.0",
	burninv1alpha1.KindDCGMDiag:     "ghcr.io/baldwinspc/glimmer-burnin-dcgm-diag:v0.5.0",
	burninv1alpha1.KindHostHealth:   "ghcr.io/baldwinspc/glimmer-burnin-host-health:v0.5.0",
	burninv1alpha1.KindMemoryBW:     "ghcr.io/baldwinspc/glimmer-burnin-memory-bw:v0.5.0",
	burninv1alpha1.KindMemoryStress: "ghcr.io/baldwinspc/glimmer-burnin-memory-stress:v0.5.0",
	burninv1alpha1.KindThermalSoak:  "ghcr.io/baldwinspc/glimmer-burnin-thermal-soak:v0.5.0",
	burninv1alpha1.KindGPUBurn:      "ghcr.io/baldwinspc/glimmer-burnin-gpu-burn:v0.5.0",
	burninv1alpha1.KindIBWriteBW:    "ghcr.io/baldwinspc/glimmer-burnin-ib-write-bw:v0.5.0",
	burninv1alpha1.KindNCCL:         "ghcr.io/baldwinspc/glimmer-burnin-nccl:v0.5.0",
	burninv1alpha1.KindGPUDirect:    "ghcr.io/baldwinspc/glimmer-burnin-gpudirect-rdma:v0.5.0",

	// KindCustom has no default by definition: it exists so a user can point
	// any image at the contract, and inventing a default would defeat it.
}

// defaultDurationSeconds bounds a test whose spec does not: an unbounded
// acceptance pod that hangs would wedge the whole run.
const defaultDurationSeconds int32 = 600

// deadlineGraceSeconds is added on top of the test's duration before the
// kubelet kills the pod: image pull and container start are not the test's
// fault and must not eat its budget.
const deadlineGraceSeconds int32 = 120

// rendezvousGraceSeconds is the extra kubelet deadline a pod gets for time it
// spends waiting for peers THIS OPERATOR has not created yet.
//
// EVERY rank of a group waits, which is why this is not scoped to the root. A
// collective makes no progress until the last rank joins, so rank 1 sits idle
// while ranks 2..N-1 are scheduled and pull their image, exactly as the root
// sits idle while all of them do. The root simply waits longest, because it is
// created first and the workers are not created until it reports Ready.
//
// Every second of that was charged against an activeDeadlineSeconds sized for
// ONE pod's start. On a cold 8-node cluster the root is killed part-way through
// a test that had barely begun, reported as "test exceeded its deadline and was
// killed" — a finding about hardware that was fine (issue #122).
//
// THE NUMBER IS NOT INVENTED, and that matters here more than its size: it is
// schedulingGracePeriod, which is already how long this operator waits for a pod
// to start before it gives up on it (podOverdue). The argument is exactly that
// symmetry — the kubelet must not kill the root for waiting out a window the
// operator itself considers a reasonable wait. Anything larger would be a
// tolerance nobody measured, which is the mistake #54 and #61 are about.
//
// The ordering it produces is deliberate. activeDeadlineSeconds runs from the
// pod's StartTime and podOverdue from its CreationTimestamp, and StartTime is
// never earlier, so with equal windows the OPERATOR gives up first — and its
// message names the ranks that did not finish, where the kubelet's says only
// that a deadline passed.
//
// A Pair server has the same shape with ONE peer, and is deliberately left
// alone: 120 s has been sufficient for it on real hardware (the Pair suite
// passed on two Sparks), so there is evidence it does not need this and none
// that it does. Widen it when that evidence exists, not by symmetry.
func rendezvousGraceSeconds(rv *rendezvous) int32 {
	if rv == nil || rv.scope != burninv1alpha1.ScopeGroup {
		return 0
	}
	return int32(schedulingGracePeriod / time.Second)
}

// podName derives the deterministic name for one ATTEMPT of a test on a node.
//
// Determinism is what makes reconciliation idempotent: a crashed controller
// finds the pod it already created instead of starting the test twice.
//
// Three things are part of the identity, and each one prevents a specific way
// of attributing an execution to the wrong thing:
//
//   - the run's UID, because delete-and-recreate of a same-name run must NOT
//     adopt the previous run's pods, or the new run would record the old
//     execution's evidence as its own verdict without touching the hardware;
//   - the test index and node, the coordinates of the execution;
//   - the attempt, because repeats and error-retries run the same test on the
//     same node again. Without it the second attempt would find the first
//     attempt's terminated pod, re-harvest its exit code, and "confirm" a
//     result the hardware never produced a second time.
//
// The hash keeps the name valid regardless of test/node name length.
func podName(run *burninv1alpha1.BurnInRun, testIndex int, node string, attempt int32) string {
	return podNameForRole(run, testIndex, node, attempt, "")
}

// podNameForRole is podName with a Pair-scope role folded in.
//
// A Pair execution is two pods, and they already differ in the hash because the
// node is part of it — but a name that differs only in eight hex characters
// tells an operator staring at `kubectl get pods` nothing about which end of
// the link they are looking at. The role goes into the READABLE part for that
// reason, and into the hash so the two identities are independent by
// construction rather than by the accident that the nodes happened to differ.
//
// An empty role reproduces the Node-scope name byte for byte, deliberately: a
// run that is in flight when the operator is upgraded must keep finding the
// pods it already created, not decide they are missing and start the test again
// on hardware that is already being burned.
func podNameForRole(run *burninv1alpha1.BurnInRun, testIndex int, node string, attempt int32, role string) string {
	seed := string(run.UID) + "\x00" + run.Name + "\x00" +
		strconv.Itoa(testIndex) + "\x00" + node + "\x00" + strconv.Itoa(int(attempt))
	readable := ""
	if role != "" {
		seed += "\x00" + role
		readable = "-" + role
	}
	h := sha256.Sum256([]byte(seed))
	return fmt.Sprintf("%s-t%d-a%d%s-%s", truncate(run.Name, 36), testIndex, attempt, readable, hex.EncodeToString(h[:4]))
}

// rendezvousServiceName is the headless Service one multi-pod attempt
// rendezvous through, and therefore also the DNS subdomain of all its pods.
//
// It is per (run, test, attempt), not per test: a retry mints new pods, and a
// Service whose selector could still match the previous attempt's pod would let
// a member resolve a peer that is already dead. It carries the run UID in its
// hash for the same reason pod names do — a deleted-and-recreated run of the
// same name must not adopt the previous incarnation's rendezvous.
//
// The "pair" literal in the seed is HISTORICAL and must not be tidied: changing
// it changes every name this function produces, and a Pair run in flight across
// an operator upgrade would stop finding the Service its two pods are already
// rendezvousing through. Group scope reuses it unchanged — the test index is
// part of the seed, so a Pair test and a Group test in one profile cannot
// collide.
//
// The "bp-" prefix is not decoration: a Service name must be a DNS-1035 LABEL,
// which has to start with a letter, and a BurnInRun name is a DNS-1123 subdomain
// that may legally start with a digit.
func rendezvousServiceName(run *burninv1alpha1.BurnInRun, testIndex int, attempt int32) string {
	h := sha256.Sum256([]byte(
		string(run.UID) + "\x00" + run.Name + "\x00" +
			strconv.Itoa(testIndex) + "\x00" + strconv.Itoa(int(attempt)) + "\x00pair"))
	return fmt.Sprintf("bp-%s-t%d-a%d-%s", truncate(run.Name, 26), testIndex, attempt, hex.EncodeToString(h[:4]))
}

// peerHost is the DNS name one member of a multi-pod execution uses to reach
// another: the pair's opposite endpoint, or the collective's root.
//
// It is deliberately NOT fully qualified to cluster.local. A cluster's DNS
// domain is configurable, and hard-coding the default would break the
// rendezvous — silently, as a connection error that looks like a bad link — on
// every cluster that changed it. "<pod>.<service>.<namespace>.svc" resolves
// through the pod's own search path under any cluster domain.
func peerHost(service, namespace, role string) string {
	return role + "." + service + "." + namespace + ".svc"
}

// rendezvous carries the facts one pod of a MULTI-POD execution needs in order
// to find the others. Nil at Node scope, which is what keeps a Node pod's shape
// exactly what it was.
//
// One struct serves Pair and Group because the topology they need is the same
// shape — "which member am I, and where do the others answer" — and because
// podForTest must stay the single place a runner pod is built. Two constructors
// would be two chances for the two scopes to drift in host mounts, tolerations
// or deadlines, and a link or a collective measured under different pod shapes
// at each end is not measured.
//
// The env it produces is scope-specific and deliberately minimal: everything
// topological stays in the operator, so one image is a server on one node and a
// client on another, or rank 4 of eleven, without the image learning anything
// about Kubernetes.
type rendezvous struct {
	// scope selects which environment contract this pod gets.
	scope burninv1alpha1.TestScope
	// role is this pod's own identity within the unit, and it is load-bearing
	// twice over: it becomes the pod's HOSTNAME, which is what gives it its own
	// A record under the headless Service, and at Pair scope it is also
	// BURNIN_ROLE. "server"/"client" at Pair scope, "rank-N" at Group scope.
	role string
	// service is the headless Service, and therefore the pod's subdomain.
	service string

	// ── Pair only ──
	// peerRole is the OTHER endpoint's role, used to build BURNIN_PEER_HOST.
	peerRole string
	// peerNode is the node the other endpoint runs on. It is recorded for the
	// message a human reads, never used for addressing.
	peerNode string

	// ── Group only ──
	// rank is this pod's 0-based rank, and rootRole names rank 0.
	rank   int
	nranks int
	// rootNode is the node rank 0 runs on: for messages, never for addressing.
	rootNode string
}

// Group rendezvous role names. Rank 0 is the root — the rank that publishes
// whatever bootstrap handle the collective needs, and the one every other rank
// is gated on.
const (
	groupRootRank = 0
	groupRolePfx  = "rank-"
)

func groupRole(rank int) string { return groupRolePfx + strconv.Itoa(rank) }

// env is the rendezvous contract as the runner sees it.
//
// Pair gets BURNIN_ROLE / BURNIN_PEER_HOST / BURNIN_PEER_NODE, byte for byte
// what it always got — the fabric runners are published, immutable images and
// this must not move under them.
//
// Group gets BURNIN_RANK / BURNIN_NRANKS / BURNIN_ROOT_HOST / BURNIN_ROOT_NODE.
// That is the whole contract: which member of the collective this is, how many
// there are, and where the root answers. It is deliberately NOT a peer list —
// every collective bootstrap in practice has one rank publish a handle that the
// rest fetch, which is what our own nccl runner already does over one TCP
// connection, and an N-entry list would be a topology the operator has to keep
// correct rather than a name the runner resolves.
//
// BURNIN_ROLE is deliberately ABSENT at Group scope. A runner that keys off
// "server"/"client" must not silently treat rank 4 of eleven as a client; making
// it read an unset variable is what turns a wrong assumption into a loud one.
func (r *rendezvous) env(namespace string) []corev1.EnvVar {
	if r.scope == burninv1alpha1.ScopeGroup {
		return []corev1.EnvVar{
			{Name: "BURNIN_RANK", Value: strconv.Itoa(r.rank)},
			{Name: "BURNIN_NRANKS", Value: strconv.Itoa(r.nranks)},
			{Name: "BURNIN_ROOT_HOST", Value: peerHost(r.service, namespace, groupRole(groupRootRank))},
			{Name: "BURNIN_ROOT_NODE", Value: r.rootNode},
		}
	}
	return []corev1.EnvVar{
		{Name: "BURNIN_ROLE", Value: r.role},
		{Name: "BURNIN_PEER_HOST", Value: peerHost(r.service, namespace, r.peerRole)},
		{Name: "BURNIN_PEER_NODE", Value: r.peerNode},
	}
}

func truncate(s string, n int) string {
	if len(s) <= n {
		return s
	}
	return s[:n]
}

// runnerTolerations adds the toleration without which this operator deadlocks
// against itself.
//
// The run cordons its target, which the node controller expresses as the taint
// node.kubernetes.io/unschedulable:NoSchedule. A pod that does not tolerate it
// is refused by the scheduler — "0/2 nodes are available: 2 node(s) were
// unschedulable" — so the run would hold a node out of service and then wait
// for a pod that can never land on it, until its deadline turned the whole
// thing into an Error. Every test, every node, always.
//
// A burn-in runner is precisely the workload that SHOULD run on a node this
// operator has cordoned: the cordon exists to clear the node FOR it. So the
// toleration is added unconditionally rather than being left to the profile
// author, who would be configuring their way around a self-inflicted deadlock.
//
// It is scoped to that one taint by key. It is emphatically not a blanket
// toleration: a node tainted NotReady, or under an administrator's own taint,
// is still a node the runner should stay off unless the target's own
// tolerations say otherwise.
func runnerTolerations(fromTarget []corev1.Toleration) []corev1.Toleration {
	out := make([]corev1.Toleration, 0, len(fromTarget)+1)
	out = append(out, fromTarget...)
	return append(out, corev1.Toleration{
		Key:      corev1.TaintNodeUnschedulable,
		Operator: corev1.TolerationOpExists,
		Effect:   corev1.TaintEffectNoSchedule,
	})
}

// hostPathVolumes turns a runner's declared host mounts into the pod-level
// volumes and the container-level mounts that go with them.
//
// It returns nil, nil when nothing is declared, and that is load-bearing: a test
// that asks for no host access must produce a pod with NO volumes at all. There
// is no implicit mount anywhere in this operator — not a default /dev, not a
// convenience /sys — because a host mount nobody wrote down is a host mount
// nobody reviewed.
//
// Both slices are built in declaration order so a pod's shape is a function of
// the pinned spec alone, which is what lets a controller restart rebuild the same
// pod rather than decide the existing one is wrong.
func hostPathVolumes(mounts []burninv1alpha1.HostPathMount) ([]corev1.Volume, []corev1.VolumeMount) {
	var volumes []corev1.Volume
	var volumeMounts []corev1.VolumeMount
	for i, m := range mounts {
		name := hostPathVolumeName(i, m.MountPath)
		volumes = append(volumes, corev1.Volume{
			Name: name,
			VolumeSource: corev1.VolumeSource{
				HostPath: &corev1.HostPathVolumeSource{Path: m.Path, Type: m.Type},
			},
		})
		volumeMounts = append(volumeMounts, corev1.VolumeMount{
			Name:      name,
			MountPath: m.MountPath,
			ReadOnly:  hostPathReadOnly(m),
		})
	}
	return volumes, volumeMounts
}

// hostPathReadOnly re-applies the CRD's `readOnly` default of true.
//
// The reconciler must not assume apiserver defaulting has happened — an object
// built directly in a test, or one written before the field existed, still has to
// be executed under the policy the field documentation promises. For this field
// in particular a nil that fell through as "false" would hand out write access to
// a host path because nobody mentioned it, which is exactly backwards for a
// privilege grant.
func hostPathReadOnly(m burninv1alpha1.HostPathMount) bool {
	return m.ReadOnly == nil || *m.ReadOnly
}

// hostPathVolumeName derives a pod volume name from a mount's position and its
// mount path.
//
// The index alone would be enough for uniqueness — MountPath is the list's map
// key, so the API already rejects duplicates — but a name like "host-0-dev-kmsg"
// tells whoever is reading `kubectl describe pod` on a node that is refusing to
// start what the pod was actually asking the kubelet for. The index stays in
// front so that two paths which sanitise to the same label still get distinct
// names.
//
// The result is a DNS-1123 label: lowercase alphanumerics and dashes, no leading
// or trailing dash, at most 63 characters.
func hostPathVolumeName(index int, mountPath string) string {
	sanitized := strings.Map(func(r rune) rune {
		switch {
		case r >= 'a' && r <= 'z', r >= '0' && r <= '9':
			return r
		case r >= 'A' && r <= 'Z':
			return r + ('a' - 'A')
		default:
			return '-'
		}
	}, mountPath)
	name := fmt.Sprintf("host-%d-%s", index, sanitized)
	if len(name) > 63 {
		name = name[:63]
	}
	return strings.Trim(name, "-")
}

// runnerImage resolves the container image for a test.
func runnerImage(spec *burninv1alpha1.BurnInTestSpec) (string, error) {
	if spec.Runner != nil && spec.Runner.Image != "" {
		return spec.Runner.Image, nil
	}
	if img, ok := defaultRunnerImages[spec.Kind]; ok {
		return img, nil
	}
	return "", fmt.Errorf("no default runner image for kind %q — set spec.runner.image", spec.Kind)
}

// podForTest builds the pod that executes one attempt of one test on one node.
//
// rv is nil for a Node-scope execution and names this pod's place in the
// rendezvous for a Pair- or Group-scope one.
func podForTest(
	run *burninv1alpha1.BurnInRun,
	testIndex int,
	attempt int32,
	testName string,
	spec *burninv1alpha1.BurnInTestSpec,
	node string,
	target burninv1alpha1.TargetSelector,
	rv *rendezvous,
) (*corev1.Pod, error) {
	image, err := runnerImage(spec)
	if err != nil {
		return nil, err
	}

	duration := spec.DurationSeconds
	if duration <= 0 {
		duration = defaultDurationSeconds
	}
	deadline := int64(duration + deadlineGraceSeconds + rendezvousGraceSeconds(rv))

	container := corev1.Container{
		Name:      "runner",
		Image:     image,
		Resources: spec.Resources,
		// Runners that honour a duration read it from the environment; ones
		// that do not are still bounded by the pod's activeDeadlineSeconds.
		Env: []corev1.EnvVar{{
			Name:  "BURNIN_DURATION_SECONDS",
			Value: strconv.Itoa(int(duration)),
		}, {
			// Which pass this is. A runner may use it for seeding or logging;
			// nothing in the contract depends on it, and a runner that ignores
			// it is still correct.
			Name:  "BURNIN_ATTEMPT",
			Value: strconv.Itoa(int(attempt)),
		}},
	}

	role := ""
	if rv != nil {
		role = rv.role
		// The rendezvous contract, built in one place for both multi-pod scopes.
		// See rendezvous.env for what each scope gets and why Group does not get
		// BURNIN_ROLE.
		container.Env = append(container.Env, rv.env(run.Namespace)...)
	}

	var volumes []corev1.Volume
	if spec.Runner != nil {
		container.Command = spec.Runner.Command
		container.Args = spec.Runner.Args
		// Appended LAST so an operator's explicit env wins over the injected
		// defaults, which is the existing behaviour for BURNIN_DURATION_SECONDS
		// and stays true of the rendezvous variables.
		container.Env = append(container.Env, spec.Runner.Env...)
		container.ReadinessProbe = spec.Runner.ReadinessProbe
		if spec.Runner.Privileged || spec.Runner.RunAsUser != nil {
			container.SecurityContext = &corev1.SecurityContext{}
			if spec.Runner.Privileged {
				t := true
				container.SecurityContext.Privileged = &t
			}
			// Separate from Privileged, because Linux drops the effective
			// capability set when a process switches away from root: a privileged
			// container running as a non-root uid does NOT hold CAP_SYSLOG, which
			// is what reading /dev/kmsg needs wherever kernel.dmesg_restrict=1.
			// Granting one and assuming the other is issue #134.
			if spec.Runner.RunAsUser != nil {
				uid := *spec.Runner.RunAsUser
				container.SecurityContext.RunAsUser = &uid
				// RunAsNonRoot defaults on some clusters via PodSecurity or a
				// mutating admission policy, and it REFUSES a pod whose uid is 0
				// — with an error about the image, not about this field. Saying
				// so explicitly makes the intent legible and the failure legible
				// if a policy still refuses it.
				nonRoot := uid != 0
				container.SecurityContext.RunAsNonRoot = &nonRoot
			}
		}
		// Host mounts. This runs for EVERY scope and, at Pair scope, for both
		// roles: podForTest is the one place a runner pod is built, so the server
		// and the client of a pair get identical host access by construction
		// rather than by two code paths agreeing. A fabric test that reached the
		// verbs devices from only one end would measure nothing.
		//
		// spec is the PINNED plan's copy of the test spec, so what a running test
		// mounts is fixed at run start: editing the BurnInTest mid-run cannot
		// change which host paths an in-flight attempt is handed.
		volumes, container.VolumeMounts = hostPathVolumes(spec.Runner.HostPaths)
	}

	labels := map[string]string{
		labelRun:     run.Name,
		labelTest:    testName,
		labelNode:    node,
		labelAttempt: strconv.Itoa(int(attempt)),
	}
	if rv != nil {
		switch rv.scope {
		case burninv1alpha1.ScopeGroup:
			labels[labelRank] = strconv.Itoa(rv.rank)
		default:
			labels[labelPairRole] = rv.role
		}
	}

	pod := &corev1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:      podNameForRole(run, testIndex, node, attempt, role),
			Namespace: run.Namespace,
			Labels:    labels,
		},
		Spec: corev1.PodSpec{
			// Never restart: the exit code IS the verdict, and a restarted
			// container would overwrite the evidence.
			RestartPolicy: corev1.RestartPolicyNever,
			// Pin via the scheduler (nodeSelector on the hostname label), not
			// spec.nodeName: direct assignment bypasses scheduler accounting
			// for the extended resources (nvidia.com/gpu) the test requests.
			NodeSelector: map[string]string{corev1.LabelHostname: node},
			// The target's own tolerations let the runner land on a node that
			// carries taints of its own; runnerTolerations adds the one this
			// operator cannot do without.
			Tolerations:           runnerTolerations(target.Tolerations),
			HostNetwork:           spec.HostNetwork,
			ActiveDeadlineSeconds: &deadline,
			Containers:            []corev1.Container{container},
			// Nil unless the test declared host mounts, so a test that asked for
			// no host access gets a pod with no volumes whatsoever.
			Volumes: volumes,
		},
	}

	if rv != nil {
		// hostname + subdomain are what give this pod its own A record under
		// the headless Service, which is how the other members address it.
		// Without both, the Service resolves to the set of endpoints and no
		// member can name another.
		pod.Spec.Hostname = rv.role
		pod.Spec.Subdomain = rv.service
		if spec.HostNetwork {
			// A host-network pod defaults to the node's resolv.conf, which
			// knows nothing about cluster DNS — the rendezvous name would not
			// resolve at all. Fabric runners want host networking (it is how
			// the RDMA device is reached), so the two have to be reconciled,
			// and this is the supported way to do it.
			//
			// Only rendezvous pods get this. A Node-scope host-network runner
			// has no rendezvous to perform and may well depend on host DNS.
			pod.Spec.DNSPolicy = corev1.DNSClusterFirstWithHostNet
		}
	}
	return pod, nil
}

// headlessServiceForRendezvous is the rendezvous, for Pair and Group alike.
//
// It is headless (ClusterIP None) because nothing here wants load balancing:
// the point is per-pod DNS, so that "server" and "client" are names the two
// runners can resolve to each other's addresses. A headless Service is also the
// one Service shape Kubernetes permits with no ports, which matters because the
// operator does not know — and must not need to know — which port a runner
// listens on. Ports are the runner's business; naming is the operator's.
//
// PublishNotReadyAddresses is deliberately true. Readiness is enforced by the
// CONTROLLER, which does not create the client (or the worker ranks) until the
// server (or the root) is Ready; leaving it to DNS instead would add an
// endpoint-propagation race to their very first lookup, and a failed lookup on a
// fabric test is precisely the failure that gets misread as a bad link.
// Publishing every address unconditionally also lets the first pod name its
// peers before they are up, which a runner that wants to whitelist them needs.
func headlessServiceForRendezvous(run *burninv1alpha1.BurnInRun, testIndex int, attempt int32, testName string) *corev1.Service {
	return &corev1.Service{
		ObjectMeta: metav1.ObjectMeta{
			Name:      rendezvousServiceName(run, testIndex, attempt),
			Namespace: run.Namespace,
			Labels: map[string]string{
				labelRun:     run.Name,
				labelTest:    testName,
				labelAttempt: strconv.Itoa(int(attempt)),
			},
		},
		Spec: corev1.ServiceSpec{
			ClusterIP:                corev1.ClusterIPNone,
			PublishNotReadyAddresses: true,
			Selector: map[string]string{
				labelRun:     run.Name,
				labelTest:    testName,
				labelAttempt: strconv.Itoa(int(attempt)),
			},
		},
	}
}

// podReady reports whether the kubelet considers the pod able to serve.
//
// This is the client's start gate, and the distinction it draws is the whole
// reason the gate exists: podStarted says a process is running, this says the
// pod has reported itself able to answer. With a readinessProbe on the server
// runner (RunnerSpec.ReadinessProbe) that is a real statement about a listening
// socket; without one it degrades to "the containers started", which is the
// weakest honest gate available and still strictly better than launching both
// pods at once.
func podReady(pod *corev1.Pod) bool {
	if pod.Status.Phase != corev1.PodRunning {
		return false
	}
	for _, c := range pod.Status.Conditions {
		if c.Type == corev1.PodReady {
			return c.Status == corev1.ConditionTrue
		}
	}
	return false
}

// podOutcome reads a terminated pod's exit code. The second return is false
// while the pod is still running or pending.
// The fourth return is the kubelet's own DETAIL, and dropping it is what made a
// cluster-wide outage unreadable (#52). For a StartError the kubelet puts the
// whole createContainer hook failure in Terminated.Message — the literal string
// "No help topic for 'disable-device-node-modification'", which names the broken
// CDI hook and therefore the fix. Discarding it left the stored result saying
// only `runner terminated abnormally (exit 128, reason "StartError")`, which is
// true of every unstartable container and identifies nothing.
//
// It is DETAIL and not a reason: the reason stays the short machine token the
// phase logic keys off, and the message is appended where a human reads it.
func podOutcome(pod *corev1.Pod) (exitCode int, terminated bool, reason, detail string) {
	switch pod.Status.Phase {
	case corev1.PodSucceeded, corev1.PodFailed:
	default:
		return 0, false, "", ""
	}
	// A pod-level failure reason outranks the container's own: when the
	// kubelet kills a pod at its activeDeadlineSeconds it sets
	// pod.Status.Reason="DeadlineExceeded" while the container records
	// whatever its process did on SIGTERM — including a clean exit 0 from a
	// signal-trapping entrypoint. Reading only the container reason would let
	// a deadline-killed test masquerade as a completed one.
	podReason := ""
	if pod.Status.Phase == corev1.PodFailed {
		podReason = pod.Status.Reason
	}
	for _, cs := range pod.Status.ContainerStatuses {
		if cs.Name != "runner" {
			continue
		}
		if term := cs.State.Terminated; term != nil {
			if podReason != "" {
				return int(term.ExitCode), true, podReason, term.Message
			}
			return int(term.ExitCode), true, term.Reason, term.Message
		}
	}
	// PodFailed with no terminated runner container: the pod never started
	// (unpullable image, deadline before start, evicted). The machinery
	// failed, not the hardware — surface whatever reason the pod carries.
	if pod.Status.Phase == corev1.PodFailed {
		reason, detail := firstPodFailure(pod)
		return -1, true, reason, detail
	}
	return 0, false, "", ""
}

func firstPodFailure(pod *corev1.Pod) (reason, detail string) {
	// A waiting container's Message is where an ImagePullBackOff records WHICH
	// image and WHY — "failed to resolve reference ...: not found" — which is the
	// difference between a typo in spec.runner.image and a registry outage.
	for _, cs := range pod.Status.ContainerStatuses {
		if w := cs.State.Waiting; w != nil && w.Reason != "" {
			if pod.Status.Reason != "" {
				return pod.Status.Reason, w.Message
			}
			return w.Reason, w.Message
		}
	}
	if pod.Status.Reason != "" {
		return pod.Status.Reason, pod.Status.Message
	}
	return "pod failed before the runner container terminated", pod.Status.Message
}

// podLive reports whether a pod is still occupying its node.
//
// This is the unit MaxConcurrentNodes is enforced in, so it must answer the
// facility's question — "is this node under test load right now?" — and not the
// bookkeeping one. A Pending pod counts: it has claimed the slot and will start
// without anybody asking again. A Succeeded or Failed pod does not, even before
// its result has been harvested: the container is gone and the node is drawing
// idle power, so making the next node wait for a status write would serialise
// the run on this controller's write latency rather than on the hardware.
func podLive(pod *corev1.Pod) bool {
	switch pod.Status.Phase {
	case corev1.PodSucceeded, corev1.PodFailed:
		return false
	default:
		return true
	}
}

// podStarted reports whether the kubelet has actually begun executing the pod.
//
// It is the gate for TestResult.StartedAt, which must mean "the hardware began
// being tested" and never "we asked for a pod". A pod that sat unschedulable
// for an hour tested nothing for that hour, and counting the wait as test time
// turns a stuck run into what looks like a slow one — the exact reading that
// sends somebody hunting for a hardware fault that is really a missing
// toleration.
//
// A terminated pod counts as started even without a StartTime: it plainly ran,
// and a controller that only ever observes the pod after it finished must still
// record that the execution happened.
func podStarted(pod *corev1.Pod) bool {
	if pod.Status.StartTime != nil {
		return true
	}
	switch pod.Status.Phase {
	case corev1.PodRunning, corev1.PodSucceeded, corev1.PodFailed:
		return true
	default:
		return false
	}
}

// attemptStart is when an execution began: the kubelet's own StartTime when it
// is available, and otherwise the moment the controller first saw the pod
// running. The fallback is deliberately the observation time and not the pod's
// creation time — see podStarted.
func attemptStart(pod *corev1.Pod, fallback time.Time) metav1.Time {
	if pod != nil && pod.Status.StartTime != nil {
		return *pod.Status.StartTime
	}
	return metav1.NewTime(fallback)
}

// ownedBy reports whether the pod belongs to this exact run object.
//
// Name is not enough. The run label carries a name, and names are reusable, so
// a delete-and-recreate of the same run would otherwise count the previous
// incarnation's pods towards its concurrency cap and delete them on its way
// out. The controller reference carries the UID, which is the identity that
// cannot be recycled.
func ownedBy(pod *corev1.Pod, run *burninv1alpha1.BurnInRun) bool {
	for _, or := range pod.OwnerReferences {
		if or.UID == run.UID {
			return true
		}
	}
	return false
}
