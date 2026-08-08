// Command manager runs the Glimmer Burn-In Operator: it reconciles BurnInRun
// objects into test pods across the targeted nodes and exports verdicts to the
// configured sinks. Standalone — no Glimmer control-plane dependency.
package main

import (
	"context"
	"crypto/tls"
	"flag"
	"io"
	"os"

	"k8s.io/apimachinery/pkg/runtime"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	"k8s.io/client-go/kubernetes"
	clientgoscheme "k8s.io/client-go/kubernetes/scheme"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/healthz"
	"sigs.k8s.io/controller-runtime/pkg/log/zap"
	metricsserver "sigs.k8s.io/controller-runtime/pkg/metrics/server"

	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/types"

	burninv1alpha1 "github.com/baldwinSPC/glimmer-burnin/api/v1alpha1"
	"github.com/baldwinSPC/glimmer-burnin/internal/controller"
	"github.com/baldwinSPC/glimmer-burnin/pkg/contract"
)

var (
	scheme   = runtime.NewScheme()
	setupLog = ctrl.Log.WithName("setup")
)

func init() {
	utilruntime.Must(clientgoscheme.AddToScheme(scheme))
	utilruntime.Must(burninv1alpha1.AddToScheme(scheme))
}

// leaderElectionID is the name of the Lease the managers elect through. It is
// not a display string: two managers that disagree about it are two managers
// that both believe they are the leader.
const leaderElectionID = "burnin.glimmer.ai"

// managerOptions builds the manager's configuration.
//
// It is a function rather than an inline literal so it can be tested. Each of
// the three settings below is load-bearing somewhere the rest of the suite
// cannot see:
//
//   - the Secret cache exclusion is what keeps the operator's ClusterRole free
//     of list+watch on every secret in the cluster;
//   - the leader-election ID is the Lease name, and config/manager grants
//     access to leases in ONE namespace;
//   - the leader-election NAMESPACE is passed explicitly rather than inferred.
//     controller-runtime falls back to reading the in-cluster service-account
//     file, which works in the shipped Deployment and fails outside it, and
//     everything else in this operator that needs a namespace refuses to guess
//     (see the NodeFingerprint wiring below). An empty value keeps the old
//     inference, so nothing regresses for a pod that does not project it.
func managerOptions(metricsAddr, probeAddr string, enableLeaderElection bool, podNamespace string) ctrl.Options {
	return ctrl.Options{
		Scheme: scheme,
		// Secrets are read once per delivery to resolve a sink's bearer token.
		// Reading them through the cache would lazily start a cluster-wide
		// Secret informer, which needs list+watch RBAC on every secret — a
		// needlessly broad grant. Bypass the cache so a plain namespaced get
		// suffices.
		Client: client.Options{
			Cache: &client.CacheOptions{DisableFor: []client.Object{&corev1.Secret{}}},
		},
		Metrics: metricsserver.Options{
			BindAddress: metricsAddr,
			TLSOpts:     []func(*tls.Config){func(c *tls.Config) { c.MinVersion = tls.VersionTLS12 }},
		},
		HealthProbeBindAddress:  probeAddr,
		LeaderElection:          enableLeaderElection,
		LeaderElectionID:        leaderElectionID,
		LeaderElectionNamespace: podNamespace,
	}
}

func main() {
	var metricsAddr, probeAddr, clusterName string
	var enableLeaderElection bool
	flag.StringVar(&metricsAddr, "metrics-bind-address", ":8080", "The address the metrics endpoint binds to.")
	flag.StringVar(&probeAddr, "health-probe-bind-address", ":8081", "The address the probe endpoint binds to.")
	flag.StringVar(&clusterName, "cluster-name", "",
		"Name of this cluster, echoed into every delivered envelope so a stored verdict is "+
			"portable evidence on its own. NEVER guessed: unset emits no cluster name, because "+
			"a verdict attributed to the wrong fleet is worse than one attributed to none.")
	flag.BoolVar(&enableLeaderElection, "leader-elect", false,
		"Enable leader election for controller manager. Ensures only one active manager.")
	opts := zap.Options{Development: true}
	opts.BindFlags(flag.CommandLine)
	flag.Parse()

	ctrl.SetLogger(zap.New(zap.UseFlagOptions(&opts)))

	// Projected by config/manager through the downward API. Two things need it:
	// leader election above, and the NodeFingerprint capture below.
	podNamespace := os.Getenv("POD_NAMESPACE")

	mgr, err := ctrl.NewManager(ctrl.GetConfigOrDie(),
		managerOptions(metricsAddr, probeAddr, enableLeaderElection, podNamespace))
	if err != nil {
		setupLog.Error(err, "unable to start manager")
		os.Exit(1)
	}

	// A plain clientset for pod logs: controller-runtime's client does not
	// serve subresource log streams, and the runner's stdout IS the metrics
	// channel — without it every verdict would be exit-code-only.
	clientset, err := kubernetes.NewForConfig(mgr.GetConfig())
	if err != nil {
		setupLog.Error(err, "unable to build clientset for pod logs")
		os.Exit(1)
	}

	if err := (&controller.BurnInRunReconciler{
		Client:  mgr.GetClient(),
		Scheme:  mgr.GetScheme(),
		Cluster: clusterRef(context.Background(), mgr.GetAPIReader(), clusterName),
		// The uncached reader, for the two node reads whose staleness costs the
		// fleet a node rather than a reconcile. See BurnInRunReconciler.APIReader.
		APIReader: mgr.GetAPIReader(),
		PodLogs: func(ctx context.Context, namespace, name string) (string, error) {
			// Keep the TAIL, not the head: runners report progressively and
			// the parser's contract is last-occurrence-wins, so the settled
			// metric lines are the final ones. Head-truncating a chatty
			// soak's log would silently drop exactly the lines that matter.
			tail := int64(2000)
			limit := int64(1 << 20)
			req := clientset.CoreV1().Pods(namespace).GetLogs(name, &corev1.PodLogOptions{
				Container:  "runner",
				TailLines:  &tail,
				LimitBytes: &limit,
			})
			rc, err := req.Stream(ctx)
			if err != nil {
				return "", err
			}
			defer func() { _ = rc.Close() }()
			b, err := io.ReadAll(rc)
			if err != nil {
				// A partial read must not masquerade as the full log: the
				// caller treats log absence as "metrics unavailable", which
				// is the honest state here.
				return "", err
			}
			return string(b), nil
		},
	}).SetupWithManager(mgr); err != nil {
		setupLog.Error(err, "unable to create controller", "controller", "BurnInRun")
		os.Exit(1)
	}

	if err := (&controller.BurnInScheduleReconciler{
		Client: mgr.GetClient(),
		Scheme: mgr.GetScheme(),
	}).SetupWithManager(mgr); err != nil {
		setupLog.Error(err, "unable to create controller", "controller", "BurnInSchedule")
		os.Exit(1)
	}

	// NodeFingerprint capture: record what each node is made of, and flag it
	// when that changes. NodeFingerprint is namespaced and Node is not, so the
	// namespace has to be supplied; POD_NAMESPACE (downward API) is preferred
	// and SetupWithManager falls back to the in-cluster service-account file.
	// It refuses to guess, and a fingerprint written where nobody looks is a
	// verdict nobody can audit — so an unresolvable namespace is fatal here
	// rather than silently disabling the capture.
	//
	// It also carries the orphaned-cordon reaper, which is why it gets the
	// uncached reader: the reaper removes a cordon stamp on the strength of its
	// owning run being absent, and a cache miss is not an absence.
	if err := (&controller.NodeFingerprintReconciler{
		Client:    mgr.GetClient(),
		Scheme:    mgr.GetScheme(),
		APIReader: mgr.GetAPIReader(),
		Recorder:  mgr.GetEventRecorderFor("nodefingerprint-controller"),
		Namespace: podNamespace,
	}).SetupWithManager(mgr); err != nil {
		setupLog.Error(err, "unable to create controller", "controller", "NodeFingerprint")
		os.Exit(1)
	}

	if err := mgr.AddHealthzCheck("healthz", healthz.Ping); err != nil {
		setupLog.Error(err, "unable to set up health check")
		os.Exit(1)
	}
	if err := mgr.AddReadyzCheck("readyz", healthz.Ping); err != nil {
		setupLog.Error(err, "unable to set up ready check")
		os.Exit(1)
	}

	setupLog.Info("starting manager")
	if err := mgr.Start(ctrl.SetupSignalHandler()); err != nil {
		setupLog.Error(err, "problem running manager")
		os.Exit(1)
	}
}

// clusterRef assembles the cluster identity carried in every envelope.
//
// The NAME comes from a flag and is never inferred. There is no reliable way to
// learn what an operator calls a cluster from inside it, and a guess that is
// wrong attributes a hardware verdict to the wrong fleet — which is worse than
// carrying no name at all, because a consumer cannot tell a guess from a fact.
//
// The UID is the kube-system namespace UID, the closest thing Kubernetes has to
// a stable cluster identity: it survives renames and node replacement, and it
// differs between two clusters an operator happened to give the same name. It is
// read ONCE, here, through the uncached reader — the manager's cache is not
// started yet at this point, and a value that cannot change is not worth a watch.
//
// Either half may be absent and the envelope stays valid. An operator with
// neither configured emits no cluster block at all.
func clusterRef(ctx context.Context, reader client.Reader, name string) *contract.ClusterRef {
	ref := &contract.ClusterRef{Name: name}

	var ns corev1.Namespace
	if err := reader.Get(ctx, types.NamespacedName{Name: "kube-system"}, &ns); err != nil {
		// Not fatal, and deliberately quiet at info level: an RBAC that does not
		// grant this read is a legitimate deployment, and the envelope simply
		// carries less. Failing to start over an optional identity field would
		// trade hardware acceptance for provenance.
		setupLog.Info("could not read the kube-system namespace UID; envelopes will carry no cluster UID",
			"error", err.Error())
	} else {
		ref.UID = string(ns.UID)
	}

	if ref.Name == "" && ref.UID == "" {
		return nil
	}
	return ref
}
