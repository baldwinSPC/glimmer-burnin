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

	burninv1alpha1 "github.com/baldwinSPC/glimmer-burnin/api/v1alpha1"
	"github.com/baldwinSPC/glimmer-burnin/internal/controller"
)

var (
	scheme   = runtime.NewScheme()
	setupLog = ctrl.Log.WithName("setup")
)

func init() {
	utilruntime.Must(clientgoscheme.AddToScheme(scheme))
	utilruntime.Must(burninv1alpha1.AddToScheme(scheme))
}

func main() {
	var metricsAddr, probeAddr string
	var enableLeaderElection bool
	flag.StringVar(&metricsAddr, "metrics-bind-address", ":8080", "The address the metrics endpoint binds to.")
	flag.StringVar(&probeAddr, "health-probe-bind-address", ":8081", "The address the probe endpoint binds to.")
	flag.BoolVar(&enableLeaderElection, "leader-elect", false,
		"Enable leader election for controller manager. Ensures only one active manager.")
	opts := zap.Options{Development: true}
	opts.BindFlags(flag.CommandLine)
	flag.Parse()

	ctrl.SetLogger(zap.New(zap.UseFlagOptions(&opts)))

	mgr, err := ctrl.NewManager(ctrl.GetConfigOrDie(), ctrl.Options{
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
		HealthProbeBindAddress: probeAddr,
		LeaderElection:         enableLeaderElection,
		LeaderElectionID:       "burnin.glimmer.ai",
	})
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
		Client: mgr.GetClient(),
		Scheme: mgr.GetScheme(),
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
	if err := (&controller.NodeFingerprintReconciler{
		Client:    mgr.GetClient(),
		Scheme:    mgr.GetScheme(),
		Recorder:  mgr.GetEventRecorderFor("nodefingerprint-controller"),
		Namespace: os.Getenv("POD_NAMESPACE"),
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
