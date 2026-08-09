package localrun

import (
	"strings"
	"testing"
	"time"

	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"

	api "github.com/baldwinSPC/glimmer-burnin/api/v1alpha1"
)

// These assert the mirroring of podForTest. A test that gets /dev/kmsg
// read-only in a cluster must get /dev/kmsg read-only here, or the two
// dispatchers are measuring different things.

func ptrBool(b bool) *bool { return &b }

func hostPathType(t corev1.HostPathType) *corev1.HostPathType { return &t }

func TestAMountDefaultsToReadOnly(t *testing.T) {
	// readOnly is a pointer in the API precisely so "unset" is distinguishable
	// from "false", and a privilege grant's default has to fall towards the
	// harmless form.
	spec := api.BurnInTestSpec{
		Kind:   api.KindCustom,
		Runner: &api.RunnerSpec{Image: "x:1", HostPaths: []api.HostPathMount{{Path: "/dev/kmsg", MountPath: "/dev/kmsg"}}},
	}
	got, err := Translate(Plan{Node: "n1"}, PlannedTest{Name: "t", Spec: spec})
	if err != nil {
		t.Fatalf("Translate: %v", err)
	}
	if len(got.Mounts) != 1 || !got.Mounts[0].ReadOnly {
		t.Fatalf("mount = %+v, want read-only by default", got.Mounts)
	}

	// Explicit false is honoured, because that is a decision someone made.
	spec.Runner.HostPaths[0].ReadOnly = ptrBool(false)
	got, _ = Translate(Plan{Node: "n1"}, PlannedTest{Name: "t", Spec: spec})
	if got.Mounts[0].ReadOnly {
		t.Error("an explicit readOnly:false was overridden")
	}
}

func TestACharDeviceIsGrantedNotJustMounted(t *testing.T) {
	// Bind-mounting a device node is not enough: without the device cgroup
	// grant the container sees it and cannot open it, which reads as broken
	// hardware rather than a missing permission.
	spec := api.BurnInTestSpec{
		Kind: api.KindCustom,
		Runner: &api.RunnerSpec{Image: "x:1", HostPaths: []api.HostPathMount{
			{Path: "/dev/kmsg", MountPath: "/dev/kmsg", Type: hostPathType(corev1.HostPathCharDev)},
			{Path: "/var/log", MountPath: "/var/log", Type: hostPathType(corev1.HostPathDirectory)},
		}},
	}
	got, err := Translate(Plan{Node: "n1"}, PlannedTest{Name: "t", Spec: spec})
	if err != nil {
		t.Fatalf("Translate: %v", err)
	}
	if len(got.Devices) != 1 || got.Devices[0] != "/dev/kmsg" {
		t.Errorf("devices = %v, want only the character device", got.Devices)
	}
	if len(got.Mounts) != 2 {
		t.Errorf("both paths should still be mounted, got %d", len(got.Mounts))
	}
}

func TestAnRDMAMountRaisesMemlock(t *testing.T) {
	// There is no PodSpec field for RLIMIT_MEMLOCK at all, which is why the
	// fabric runners need a node-level containerd change in-cluster. Here it is
	// one flag — and the trigger is the declared mount rather than a list of
	// kinds, so a third-party fabric runner gets it too.
	spec := api.BurnInTestSpec{
		Kind:   api.KindCustom,
		Runner: &api.RunnerSpec{Image: "x:1", HostPaths: []api.HostPathMount{{Path: "/dev/infiniband", MountPath: "/dev/infiniband"}}},
	}
	got, _ := Translate(Plan{Node: "n1"}, PlannedTest{Name: "t", Spec: spec})
	if !got.UnlimitedMemlock {
		t.Error("a test declaring /dev/infiniband did not get memlock raised")
	}

	// A test that does not touch the verbs devices does not get the privilege.
	spec.Runner.HostPaths[0].Path = "/var/log"
	spec.Runner.HostPaths[0].MountPath = "/var/log"
	got, _ = Translate(Plan{Node: "n1"}, PlannedTest{Name: "t", Spec: spec})
	if got.UnlimitedMemlock {
		t.Error("memlock was raised for a test with no RDMA mount")
	}
}

func TestAcceleratorAccessComesFromTheResourceRequest(t *testing.T) {
	// The same thing a profile already writes to request a device in-cluster,
	// so one profile means the same thing on both paths.
	rows := []struct {
		resource corev1.ResourceName
		want     GPUAccess
	}{
		{"nvidia.com/gpu", GPUNvidia},
		{"amd.com/gpu", GPUAMD},
	}
	for _, r := range rows {
		spec := api.BurnInTestSpec{
			Kind:   api.KindCustom,
			Runner: &api.RunnerSpec{Image: "x:1"},
			Resources: corev1.ResourceRequirements{
				Limits: corev1.ResourceList{r.resource: resource.MustParse("1")},
			},
		}
		got, _ := Translate(Plan{Node: "n1"}, PlannedTest{Name: "t", Spec: spec})
		if got.GPUAccess != r.want {
			t.Errorf("%s: GPUAccess = %q, want %q", r.resource, got.GPUAccess, r.want)
		}
	}

	// A CPU-only test gets no device access at all — nothing is granted that
	// was not asked for.
	spec := api.BurnInTestSpec{Kind: api.KindCustom, Runner: &api.RunnerSpec{Image: "x:1"}}
	got, _ := Translate(Plan{Node: "n1"}, PlannedTest{Name: "t", Spec: spec})
	if got.GPUAccess != GPUNone {
		t.Errorf("a test requesting no device got %q", got.GPUAccess)
	}
}

func TestAProfileCannotOverrideTheContractsOwnVariables(t *testing.T) {
	// BURNIN_ROLE decides which end of a link a runner is. A profile able to
	// set it could make both ends the client, and the rendezvous would hang in
	// a way that reads as a fabric fault.
	spec := api.BurnInTestSpec{
		Kind:            api.KindCustom,
		DurationSeconds: 60,
		Runner: &api.RunnerSpec{Image: "x:1", Env: []corev1.EnvVar{
			{Name: "BURNIN_DURATION_SECONDS", Value: "1"},
			{Name: "BURNIN_ROLE", Value: "client"},
			{Name: "MY_OWN_KNOB", Value: "fine"},
		}},
	}
	got, _ := Translate(Plan{Node: "n1"}, PlannedTest{Name: "t", Spec: spec})

	if got.Env["BURNIN_DURATION_SECONDS"] != "60" {
		t.Errorf("a profile overrode the contract's duration: %q", got.Env["BURNIN_DURATION_SECONDS"])
	}
	if _, set := got.Env["BURNIN_ROLE"]; set {
		t.Error("a profile set BURNIN_ROLE")
	}
	if got.Env["MY_OWN_KNOB"] != "fine" {
		t.Error("a runner's own variable was dropped")
	}
}

func TestTheDeadlineMatchesTheOperatorsGrace(t *testing.T) {
	// A runner that would have been killed in-cluster is killed here on the
	// same schedule.
	spec := api.BurnInTestSpec{Kind: api.KindCustom, DurationSeconds: 300, Runner: &api.RunnerSpec{Image: "x:1"}}
	got, _ := Translate(Plan{Node: "n1"}, PlannedTest{Name: "t", Spec: spec})

	if want := (300 + deadlineGraceSeconds) * time.Second; got.Timeout != want {
		t.Errorf("Timeout = %s, want %s (duration + the operator's grace)", got.Timeout, want)
	}

	// A test that declares no duration gets the operator's default.
	spec.DurationSeconds = 0
	got, _ = Translate(Plan{Node: "n1"}, PlannedTest{Name: "t", Spec: spec})
	if got.Env["BURNIN_DURATION_SECONDS"] != "600" {
		t.Errorf("default duration = %q, want 600", got.Env["BURNIN_DURATION_SECONDS"])
	}
}

// A segmented soak runs here as ONE execution of the whole duration, and that is
// a decision rather than an oversight.
//
// spec.soak exists so a cluster's own housekeeping — an eviction, a drain, a
// kubelet restart — costs one window instead of the week. None of that exists on
// a bare box, so segmenting here would buy nothing and add a restart the runtime
// never needed. The verdict stays comparable because the aggregation rules make
// N windows and one long window the same measurement: elapsedS sums, a floor
// keeps the worst reading, a ceiling the highest, and a lifetime total the last.
//
// It is asserted rather than assumed because the operator's own podForTest now
// sizes its pod from the SEGMENT, and a reader comparing the two dispatchers
// would otherwise have to guess whether this one had simply been forgotten.
func TestASegmentedSoakRunsAsOneLocalExecution(t *testing.T) {
	spec := api.BurnInTestSpec{
		Kind:            api.KindCustom,
		DurationSeconds: 3600,
		Soak:            &api.SoakSpec{SegmentSeconds: 900},
		Runner:          &api.RunnerSpec{Image: "x:1"},
	}
	got, err := Translate(Plan{Node: "n1"}, PlannedTest{Name: "t", Spec: spec})
	if err != nil {
		t.Fatalf("Translate: %v", err)
	}
	if got.Env["BURNIN_DURATION_SECONDS"] != "3600" {
		t.Errorf("BURNIN_DURATION_SECONDS = %q, want the whole soak — there is no cluster "+
			"here to survive", got.Env["BURNIN_DURATION_SECONDS"])
	}
	if want := (3600 + deadlineGraceSeconds) * time.Second; got.Timeout != want {
		t.Errorf("Timeout = %s, want %s", got.Timeout, want)
	}
}

func TestImageResolutionFollowsTheOperatorsOrder(t *testing.T) {
	// Explicit image beats the default table, and a kind with neither is a
	// configuration error naming the field to set.
	spec := api.BurnInTestSpec{Kind: api.KindHostHealth, Runner: &api.RunnerSpec{Image: "mine:1"}}
	got, err := Translate(Plan{Node: "n1"}, PlannedTest{Name: "t", Spec: spec})
	if err != nil || got.Image != "mine:1" {
		t.Errorf("explicit image = %q (err %v), want mine:1", got.Image, err)
	}

	spec.Runner = nil
	got, err = Translate(Plan{Node: "n1"}, PlannedTest{Name: "t", Spec: spec})
	if err != nil || !strings.Contains(got.Image, "host-health") {
		t.Errorf("default image = %q (err %v), want the table's entry", got.Image, err)
	}

	_, err = Translate(Plan{Node: "n1"}, PlannedTest{Name: "t", Spec: api.BurnInTestSpec{Kind: api.TestKind("nope")}})
	if err == nil || !strings.Contains(err.Error(), "spec.runner.image") {
		t.Errorf("an unresolvable kind should say what to set, got %v", err)
	}
}

func TestTheCommandLineIsDeterministicAndCorrectPerRuntime(t *testing.T) {
	spec := RunSpec{
		Image:            "img:1",
		Env:              map[string]string{"B": "2", "A": "1"},
		Mounts:           []Mount{{Source: "/dev/kmsg", Target: "/dev/kmsg", ReadOnly: true}},
		Devices:          []string{"/dev/kmsg"},
		HostNetwork:      true,
		UnlimitedMemlock: true,
		GPUAccess:        GPUNvidia,
	}

	docker := (&CLIRuntime{Binary: "docker", GPUFlagStyle: GPUFlagDockerNative}).args(spec)
	joined := strings.Join(docker, " ")
	for _, want := range []string{
		"run --rm", "--network host", "--ulimit memlock=-1:-1",
		"--gpus all", "-v /dev/kmsg:/dev/kmsg:ro", "--device /dev/kmsg",
		"-e A=1 -e B=2", // sorted, so the command is stable and readable
	} {
		if !strings.Contains(joined, want) {
			t.Errorf("docker args missing %q\ngot: %s", want, joined)
		}
	}

	// podman and nerdctl want the CDI form for accelerators.
	cdi := strings.Join((&CLIRuntime{Binary: "podman", GPUFlagStyle: GPUFlagCDI}).args(spec), " ")
	if !strings.Contains(cdi, "--device nvidia.com/gpu=all") {
		t.Errorf("CDI runtime did not use the CDI device form: %s", cdi)
	}
	if strings.Contains(cdi, "--gpus") {
		t.Error("CDI runtime used docker's --gpus flag")
	}

	// The image is last before the runner's own arguments, or the runtime would
	// read a flag as the image name.
	if docker[len(docker)-1] != "img:1" {
		t.Errorf("image is not the final argument: %v", docker[len(docker)-3:])
	}
}

func TestAnAMDTestGetsBothDeviceNodes(t *testing.T) {
	args := strings.Join((&CLIRuntime{Binary: "docker"}).args(RunSpec{Image: "i:1", GPUAccess: GPUAMD}), " ")
	if !strings.Contains(args, "--device /dev/kfd") || !strings.Contains(args, "--device /dev/dri") {
		t.Errorf("AMD access needs both device nodes: %s", args)
	}
}
