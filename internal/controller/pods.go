package controller

import (
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"strconv"

	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"

	burninv1alpha1 "github.com/baldwinSPC/glimmer-burnin/api/v1alpha1"
)

// Pod labels used to find a run's pods and attribute them to a test/node.
const (
	labelRun  = "burnin.glimmer.ai/run"
	labelTest = "burnin.glimmer.ai/test"
	labelNode = "burnin.glimmer.ai/node"
)

// defaultRunnerImages maps a TestKind to its published default runner image.
// A kind without an entry requires spec.runner.image — scheduling a pod with
// no image would produce an ImagePullBackOff blamed on the hardware.
//
// Images are pinned by version, never :latest: a readiness/acceptance verdict
// must be reproducible, and a floating tag changes the test under every
// existing profile with no audit trail.
var defaultRunnerImages = map[burninv1alpha1.TestKind]string{
	burninv1alpha1.KindComputeSmoke: "ghcr.io/baldwinspc/glimmer-burnin-compute-smoke:v0.1.0",
}

// defaultDurationSeconds bounds a test whose spec does not: an unbounded
// acceptance pod that hangs would wedge the whole run.
const defaultDurationSeconds int32 = 600

// deadlineGraceSeconds is added on top of the test's duration before the
// kubelet kills the pod: image pull and container start are not the test's
// fault and must not eat its budget.
const deadlineGraceSeconds int32 = 120

// podName derives the deterministic name for a test's pod on a node.
// Determinism is what makes reconciliation idempotent: a crashed controller
// finds the pod it already created instead of starting the test twice.
// The run's UID is part of the identity: delete-and-recreate of a same-name
// run must NOT adopt the previous run's pods, or the new run would record the
// old execution's evidence as its own verdict without touching the hardware.
// The hash keeps the name valid regardless of test/node name length.
func podName(run *burninv1alpha1.BurnInRun, testIndex int, node string) string {
	h := sha256.Sum256([]byte(string(run.UID) + "\x00" + run.Name + "\x00" + strconv.Itoa(testIndex) + "\x00" + node))
	return fmt.Sprintf("%s-t%d-%s", truncate(run.Name, 40), testIndex, hex.EncodeToString(h[:4]))
}

func truncate(s string, n int) string {
	if len(s) <= n {
		return s
	}
	return s[:n]
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

// podForTest builds the pod that executes one test on one node.
func podForTest(run *burninv1alpha1.BurnInRun, testIndex int, testName string, spec *burninv1alpha1.BurnInTestSpec, node string, target burninv1alpha1.TargetSelector) (*corev1.Pod, error) {
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
		}},
	}
	if spec.Runner != nil {
		container.Command = spec.Runner.Command
		container.Args = spec.Runner.Args
		container.Env = append(container.Env, spec.Runner.Env...)
		if spec.Runner.Privileged {
			t := true
			container.SecurityContext = &corev1.SecurityContext{Privileged: &t}
		}
	}

	pod := &corev1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:      podName(run, testIndex, node),
			Namespace: run.Namespace,
			Labels: map[string]string{
				labelRun:  run.Name,
				labelTest: testName,
				labelNode: node,
			},
		},
		Spec: corev1.PodSpec{
			// Never restart: the exit code IS the verdict, and a restarted
			// container would overwrite the evidence.
			RestartPolicy: corev1.RestartPolicyNever,
			// Pin via the scheduler (nodeSelector on the hostname label), not
			// spec.nodeName: direct assignment bypasses scheduler accounting
			// for the extended resources (nvidia.com/gpu) the test requests.
			NodeSelector:          map[string]string{corev1.LabelHostname: node},
			Tolerations:           target.Tolerations,
			HostNetwork:           spec.HostNetwork,
			ActiveDeadlineSeconds: &deadline,
			Containers:            []corev1.Container{container},
		},
	}
	return pod, nil
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
