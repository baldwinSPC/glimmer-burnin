package envtest

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"strings"
	"testing"

	corev1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/client-go/kubernetes"

	burninv1alpha1 "github.com/baldwinSPC/glimmer-burnin/api/v1alpha1"
)

// What a runner's stdout can do to a ConfigMap write, asked of a REAL apiserver
// — issue #143.
//
// This lives at the envtest tier because the property is enforced by the
// apiserver and by nothing else. controller-runtime's fake client is a map with
// a type checker in front of it: it stores whatever bytes it is handed, so a
// ConfigMap the real apiserver rejects outright round-trips through the unit
// suite without a murmur.
//
// The consequence is not "one artifact is lost". A rejected write loses the
// WHOLE object, which is every artifact of every test in the run — one runner's
// stdout discarding another test's evidence, on a path the unit suite reports
// green.
//
// It drives the shipped reconciler rather than re-deriving what it would write.
// A test that assembled its own ConfigMap the way the code does would be
// asserting on a copy, and a copy is exactly what stops tracking the original.

func artifactFence(name, mediaType, payload string) string {
	return "-----BEGIN BURNIN ARTIFACT " + name + " " + mediaType + "-----\n" +
		payload + "\n-----END BURNIN ARTIFACT-----\n"
}

func artifactCM(t *testing.T, ns, runName string) *corev1.ConfigMap {
	t.Helper()
	var cm corev1.ConfigMap
	err := admin.Get(context.Background(),
		types.NamespacedName{Namespace: ns, Name: runName + "-artifacts"}, &cm)
	if apierrors.IsNotFound(err) {
		return nil
	}
	if err != nil {
		t.Fatal(err)
	}
	return &cm
}

// A payload that is not valid UTF-8 must reach a JSON consumer BYTE-EXACT.
//
// The obvious guess — that the apiserver rejects invalid UTF-8 in Data — is
// wrong, and this test was written against that guess before being corrected by
// what the apiserver actually did. Measured here: the write is accepted and the
// bytes round-trip intact to a protobuf client. But the same value read over
// JSON, which is what kubectl and every REST consumer uses, comes back with each
// invalid byte replaced by U+FFFD, silently. The evidence still looks present;
// only the ArtifactRef's digest would reveal it, and a consumer that verifies
// the digest then concludes the evidence was tampered with.
//
// So the assertion is made through a JSON client on purpose. A protobuf-only
// check would pass against code that puts the payload straight into Data, which
// is the bug.
func TestArtifactThatIsNotUTF8SurvivesARealApiserver(t *testing.T) {
	ns := newNamespace(t)
	node := nodeName(t, "binary")
	newNode(t, node)

	create(t,
		customTest(ns, "probe", nil),
		profile(ns, "acceptance", nil, burninv1alpha1.ProfileTest{TestRef: "probe"}),
		runFor(ns, "run", "acceptance", []string{node}, nil),
	)

	// A lone continuation byte: not valid UTF-8, and the shape a truncated
	// multi-byte rune actually takes.
	const binary = "captured\xffbinary\x80evidence"
	d := newDriver(t, ns)
	d.reconcilerOver(admin)
	d.script("", script{stdout: "nonfiniteCount=0\n" +
		artifactFence("dump.bin", "application/octet-stream", binary) +
		artifactFence("dcgmi.json", "application/json", `{"overall":"Pass"}`) + "DONE\n"})

	run := d.run("run", 40)
	if run.Status.Phase != burninv1alpha1.RunPassed {
		t.Fatalf("run phase = %q, want Passed: %s", run.Status.Phase, resultMessages(run))
	}
	refs := run.Status.Results[0].Artifacts
	if len(refs) != 2 {
		t.Fatalf("got %d artifact refs, want 2: %+v", len(refs), refs)
	}
	for _, ref := range refs {
		if ref.Dropped != "" {
			t.Fatalf("%s was dropped by a REAL apiserver: %s", ref.Name, ref.Dropped)
		}
	}

	// Read it the way a consumer does.
	jcfg := *cfg
	jcfg.ContentType = "application/json"
	cs, err := kubernetes.NewForConfig(&jcfg)
	if err != nil {
		t.Fatal(err)
	}
	cm, err := cs.CoreV1().ConfigMaps(ns).Get(context.Background(), "run-artifacts", metav1.GetOptions{})
	if err != nil {
		t.Fatalf("no artifact ConfigMap a JSON consumer can read: %v", err)
	}

	for _, ref := range refs {
		// The channel is stdout, so a payload is a sequence of lines and always
		// ends in one newline. Everything before it must be exact.
		var want string
		switch ref.Name {
		case "dump.bin":
			want = binary + "\n"
		default:
			want = `{"overall":"Pass"}` + "\n"
		}
		got, ok := cm.Data[ref.Key]
		if !ok {
			got = string(cm.BinaryData[ref.Key])
		}
		if got != want {
			t.Errorf("%s read back over JSON as %q, want %q\n\n"+
				"The apiserver did not refuse this — it stored the bytes and mangled them "+
				"on the way out, one U+FFFD per invalid byte. The digest in the ref no "+
				"longer matches, so a consumer that verifies it concludes the evidence "+
				"was tampered with.", ref.Name, got, want)
		}
		if !verifiesAgainst(ref.Digest, got) {
			t.Errorf("%s does not match its recorded digest %s — the ref promises "+
				"evidence the store cannot produce", ref.Name, ref.Digest)
		}
	}
	// A text payload stays in Data, so `kubectl get cm -o yaml` shows a dcgmi
	// document as a document rather than as base64.
	var textKey string
	for _, ref := range refs {
		if ref.Name == "dcgmi.json" {
			textKey = ref.Key
		}
	}
	if _, ok := cm.Data[textKey]; !ok {
		t.Errorf("a valid-UTF-8 payload was hidden in BinaryData as base64")
	}
}

// verifiesAgainst is what a consumer does with the digest it was handed.
func verifiesAgainst(digest, payload string) bool {
	sum := sha256.Sum256([]byte(payload))
	return digest == "sha256:"+hex.EncodeToString(sum[:])
}

// The key grammar is the APISERVER'S, not ours.
//
// A name a runner printed becomes a ConfigMap key, and the apiserver validates
// keys against [-._a-zA-Z0-9]+. A name with a '/' in it does not store badly —
// it makes the entire object unwritable. So the operator must refuse the name
// and keep the rest of the write, which is what this asserts end to end.
func TestAnUnusableArtifactNameCannotPoisonTheWholeWrite(t *testing.T) {
	ns := newNamespace(t)
	node := nodeName(t, "badname")
	newNode(t, node)

	create(t,
		customTest(ns, "probe", nil),
		profile(ns, "acceptance", nil, burninv1alpha1.ProfileTest{TestRef: "probe"}),
		runFor(ns, "run", "acceptance", []string{node}, nil),
	)

	d := newDriver(t, ns)
	d.reconcilerOver(admin)
	d.script("", script{stdout: "nonfiniteCount=0\n" +
		artifactFence("../../etc/passwd", "text/plain", "escaped") +
		artifactFence("has space.json", "application/json", "{}") +
		artifactFence("keeper.json", "application/json", `{"kept":true}`) + "DONE\n"})

	run := d.run("run", 40)
	if run.Status.Phase != burninv1alpha1.RunPassed {
		t.Fatalf("a runner's malformed artifact name changed the run's verdict to %q: %s",
			run.Status.Phase, resultMessages(run))
	}

	refs := run.Status.Results[0].Artifacts
	if len(refs) != 3 {
		t.Fatalf("got %d refs, want 3 (two refused, one kept): %+v", len(refs), refs)
	}
	var kept int
	for _, ref := range refs {
		switch ref.Name {
		case "keeper.json":
			if ref.Dropped != "" {
				t.Fatalf("the well-named artifact was lost alongside the bad ones: %s", ref.Dropped)
			}
			kept++
		default:
			if ref.Dropped == "" {
				t.Errorf("%q was accepted as a ConfigMap key; the apiserver will not have it",
					ref.Name)
			}
		}
	}
	if kept != 1 {
		t.Fatalf("expected exactly one kept artifact, got %d", kept)
	}

	cm := artifactCM(t, ns, "run")
	if cm == nil {
		t.Fatal("the apiserver refused the whole ConfigMap — one runner's malformed " +
			"name discarded every other artifact in the run")
	}
	if len(cm.Data) != 1 {
		t.Errorf("ConfigMap holds %d entries, want only the well-named one: %v",
			len(cm.Data), keysOf(cm.Data))
	}
}

func keysOf(m map[string]string) []string {
	out := make([]string, 0, len(m))
	for k := range m {
		out = append(out, k)
	}
	return out
}

func resultMessages(run *burninv1alpha1.BurnInRun) string {
	var b strings.Builder
	for _, r := range run.Status.Results {
		b.WriteString(r.Name + ": " + string(r.Phase) + " — " + r.Message + "; ")
	}
	return b.String()
}
