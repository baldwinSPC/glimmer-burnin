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
)

// defaultRunnerImages maps a TestKind to its default runner image. A kind
// without an entry requires spec.runner.image — scheduling a pod with no image
// would produce an ImagePullBackOff blamed on the hardware.
//
// Images are pinned by version, never :latest: a readiness/acceptance verdict
// must be reproducible, and a floating tag changes the test under every
// existing profile with no audit trail.
//
// PUBLICATION STATUS — read this before relying on any entry below.
//
// Only compute-smoke has actually been built, verified on real hardware, and
// pushed by the manual publish-runner workflow. Every other entry names the tag
// its runner WILL be published under and that tag does not exist in the
// registry yet. Until it is pushed, a test of that kind with no explicit
// spec.runner.image will pull-fail, and the run records an infrastructure Error
// for that node after the scheduling grace period — never a hardware verdict.
// That distinction is what makes listing them tolerable: an unpublished tag
// costs a run one grace period and reports honestly that nothing was judged.
//
// The entry is still not free. A missing entry fails fast and legibly at plan
// time ("set spec.runner.image"), whereas an unpublished entry fails slowly, per
// node, and looks like an infrastructure fault rather than a configuration one.
// Anyone adding a kind here ahead of its publish is trading the fast error for
// the slow one deliberately.
//
// The names follow the publish workflow's own construction —
// ghcr.io/<owner lowercased>/glimmer-burnin-<kind>:<version> — so an entry here
// and a workflow_dispatch of publish-runner with the same version agree by
// construction rather than by anyone remembering to keep them in step.
var defaultRunnerImages = map[burninv1alpha1.TestKind]string{
	// Published and verified on real hardware (GB10 / DGX Spark).
	burninv1alpha1.KindComputeSmoke: "ghcr.io/baldwinspc/glimmer-burnin-compute-smoke:v0.1.0",

	// Source landed, image NOT yet published. See PUBLICATION STATUS above.
	burninv1alpha1.KindClockProbe:   "ghcr.io/baldwinspc/glimmer-burnin-clockprobe:v0.1.0",
	burninv1alpha1.KindDCGMDiag:     "ghcr.io/baldwinspc/glimmer-burnin-dcgm-diag:v0.1.0",
	burninv1alpha1.KindHostHealth:   "ghcr.io/baldwinspc/glimmer-burnin-host-health:v0.1.0",
	burninv1alpha1.KindMemoryBW:     "ghcr.io/baldwinspc/glimmer-burnin-memory-bw:v0.1.0",
	burninv1alpha1.KindMemoryStress: "ghcr.io/baldwinspc/glimmer-burnin-memory-stress:v0.1.0",

	// Deliberately absent: gpu-burn, thermal-soak, nccl, ib-write-bw and
	// gpudirect-rdma have alias tables in pkg/runner but no runner source in
	// this repo, so there is no image for a tag to point at. They require an
	// explicit spec.runner.image and fail fast at plan time saying so.
}

// defaultDurationSeconds bounds a test whose spec does not: an unbounded
// acceptance pod that hangs would wedge the whole run.
const defaultDurationSeconds int32 = 600

// deadlineGraceSeconds is added on top of the test's duration before the
// kubelet kills the pod: image pull and container start are not the test's
// fault and must not eat its budget.
const deadlineGraceSeconds int32 = 120

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

// pairServiceName is the headless Service one Pair attempt rendezvous through,
// and therefore also the DNS subdomain of both its pods.
//
// It is per (run, test, attempt), not per test: a retry mints new pods, and a
// Service whose selector could still match the previous attempt's pod would let
// a client resolve a server that is already dead. It carries the run UID in its
// hash for the same reason pod names do — a deleted-and-recreated run of the
// same name must not adopt the previous incarnation's rendezvous.
//
// The "bp-" prefix is not decoration: a Service name must be a DNS-1035 LABEL,
// which has to start with a letter, and a BurnInRun name is a DNS-1123 subdomain
// that may legally start with a digit.
func pairServiceName(run *burninv1alpha1.BurnInRun, testIndex int, attempt int32) string {
	h := sha256.Sum256([]byte(
		string(run.UID) + "\x00" + run.Name + "\x00" +
			strconv.Itoa(testIndex) + "\x00" + strconv.Itoa(int(attempt)) + "\x00pair"))
	return fmt.Sprintf("bp-%s-t%d-a%d-%s", truncate(run.Name, 26), testIndex, attempt, hex.EncodeToString(h[:4]))
}

// pairPeerHost is the DNS name one endpoint of a pair uses to reach the other.
//
// It is deliberately NOT fully qualified to cluster.local. A cluster's DNS
// domain is configurable, and hard-coding the default would break the
// rendezvous — silently, as a connection error that looks like a bad link — on
// every cluster that changed it. "<pod>.<service>.<namespace>.svc" resolves
// through the pod's own search path under any cluster domain.
func pairPeerHost(service, namespace, peerRole string) string {
	return peerRole + "." + service + "." + namespace + ".svc"
}

// pairing carries the rendezvous facts one Pair-scope pod needs. Nil at Node
// scope, which is what keeps a Node pod's shape exactly what it was.
type pairing struct {
	// role is this pod's own role: pairRoleServer or pairRoleClient. It becomes
	// BURNIN_ROLE and the pod's hostname.
	role string
	// service is the headless Service, and therefore the pod's subdomain.
	service string
	// peerRole is the OTHER endpoint's role, used to build BURNIN_PEER_HOST.
	peerRole string
	// peerNode is the node the other endpoint runs on. It is recorded for the
	// message a human reads, never used for addressing.
	peerNode string
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
// pair is nil for a Node-scope execution and names this pod's end of the
// rendezvous for a Pair-scope one.
func podForTest(
	run *burninv1alpha1.BurnInRun,
	testIndex int,
	attempt int32,
	testName string,
	spec *burninv1alpha1.BurnInTestSpec,
	node string,
	target burninv1alpha1.TargetSelector,
	pair *pairing,
) (*corev1.Pod, error) {
	image, err := runnerImage(spec)
	if err != nil {
		return nil, err
	}

	duration := spec.DurationSeconds
	if duration <= 0 {
		duration = defaultDurationSeconds
	}
	deadline := int64(duration + deadlineGraceSeconds)

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
	if pair != nil {
		role = pair.role
		// The Pair rendezvous contract. Two variables, and a runner needs
		// nothing else to find its peer: which end of the link it is, and where
		// the other end answers. Everything topological stays in the operator,
		// so the same image is a server on one node and a client on the other
		// without the image knowing anything about Kubernetes.
		container.Env = append(container.Env,
			corev1.EnvVar{Name: "BURNIN_ROLE", Value: pair.role},
			corev1.EnvVar{Name: "BURNIN_PEER_HOST", Value: pairPeerHost(pair.service, run.Namespace, pair.peerRole)},
			corev1.EnvVar{Name: "BURNIN_PEER_NODE", Value: pair.peerNode},
		)
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
		if spec.Runner.Privileged {
			t := true
			container.SecurityContext = &corev1.SecurityContext{Privileged: &t}
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
	if role != "" {
		labels[labelPairRole] = role
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

	if pair != nil {
		// hostname + subdomain are what give this pod its own A record under
		// the headless Service, which is how the peer addresses it. Without
		// both, the Service resolves to the set of endpoints and neither end
		// can name the other.
		pod.Spec.Hostname = pair.role
		pod.Spec.Subdomain = pair.service
		if spec.HostNetwork {
			// A host-network pod defaults to the node's resolv.conf, which
			// knows nothing about cluster DNS — the rendezvous name would not
			// resolve at all. Fabric runners want host networking (it is how
			// the RDMA device is reached), so the two have to be reconciled,
			// and this is the supported way to do it.
			//
			// Only Pair pods get this. A Node-scope host-network runner has no
			// rendezvous to perform and may well depend on host DNS.
			pod.Spec.DNSPolicy = corev1.DNSClusterFirstWithHostNet
		}
	}
	return pod, nil
}

// headlessServiceForPair is the rendezvous.
//
// It is headless (ClusterIP None) because nothing here wants load balancing:
// the point is per-pod DNS, so that "server" and "client" are names the two
// runners can resolve to each other's addresses. A headless Service is also the
// one Service shape Kubernetes permits with no ports, which matters because the
// operator does not know — and must not need to know — which port a runner
// listens on. Ports are the runner's business; naming is the operator's.
//
// PublishNotReadyAddresses is deliberately true. Readiness is enforced by the
// CONTROLLER, which does not create the client pod until the server pod is
// Ready; leaving it to DNS instead would add an endpoint-propagation race to the
// client's very first lookup, and a failed lookup on a fabric test is precisely
// the failure that gets misread as a bad link. Publishing both addresses
// unconditionally also lets the server name its peer before the client is up,
// which a runner that wants to whitelist its peer needs.
func headlessServiceForPair(run *burninv1alpha1.BurnInRun, testIndex int, attempt int32, testName string) *corev1.Service {
	return &corev1.Service{
		ObjectMeta: metav1.ObjectMeta{
			Name:      pairServiceName(run, testIndex, attempt),
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
func podOutcome(pod *corev1.Pod) (exitCode int, terminated bool, reason string) {
	switch pod.Status.Phase {
	case corev1.PodSucceeded, corev1.PodFailed:
	default:
		return 0, false, ""
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
				return int(term.ExitCode), true, podReason
			}
			return int(term.ExitCode), true, term.Reason
		}
	}
	// PodFailed with no terminated runner container: the pod never started
	// (unpullable image, deadline before start, evicted). The machinery
	// failed, not the hardware — surface whatever reason the pod carries.
	if pod.Status.Phase == corev1.PodFailed {
		return -1, true, firstPodFailureReason(pod)
	}
	return 0, false, ""
}

func firstPodFailureReason(pod *corev1.Pod) string {
	if pod.Status.Reason != "" {
		return pod.Status.Reason
	}
	for _, cs := range pod.Status.ContainerStatuses {
		if w := cs.State.Waiting; w != nil && w.Reason != "" {
			return w.Reason
		}
	}
	return "pod failed before the runner container terminated"
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
