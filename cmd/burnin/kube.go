package main

import (
	"context"
	"fmt"
	"sort"
	"time"

	corev1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/client-go/kubernetes/scheme"
	"k8s.io/client-go/rest"
	"k8s.io/client-go/tools/clientcmd"
	"sigs.k8s.io/controller-runtime/pkg/client"

	burninv1alpha1 "github.com/baldwinSPC/glimmer-burnin/api/v1alpha1"
	"github.com/baldwinSPC/glimmer-burnin/pkg/contract"
	"github.com/baldwinSPC/glimmer-burnin/pkg/report"
)

// clusterReadTimeout bounds the whole read. A person is waiting at a terminal.
const clusterReadTimeout = 30 * time.Second

// loadFromCluster assembles an Input from a live cluster.
//
// Three reads, and only the first is required:
//
//   - the BurnInRun, for the run's identity and the nodes it touched;
//   - its profile's ConfigMap sinks, for the delivered envelopes;
//   - a NodeFingerprint per node, for the hardware inventory.
//
// The envelopes are the point. The operator already assembled them, already
// derived their idempotency keys, and already decided what a verdict was — so
// this reads the record rather than reconstructing one from CRD status. A second
// reconstruction is a second opinion, and two opinions about one run is the
// failure this whole package exists to prevent.
func loadFromCluster(namespace, name, kubeconfig string) (report.Input, error) {
	ctx, cancel := context.WithTimeout(context.Background(), clusterReadTimeout)
	defer cancel()

	c, err := newClient(kubeconfig)
	if err != nil {
		return report.Input{}, err
	}

	var run burninv1alpha1.BurnInRun
	if err := c.Get(ctx, client.ObjectKey{Namespace: namespace, Name: name}, &run); err != nil {
		if apierrors.IsNotFound(err) {
			return report.Input{}, fmt.Errorf("no BurnInRun %s/%s", namespace, name)
		}
		return report.Input{}, fmt.Errorf("reading BurnInRun %s/%s: %w", namespace, name, err)
	}

	envelopes, err := envelopesFor(ctx, c, &run)
	if err != nil {
		return report.Input{}, err
	}
	if len(envelopes) == 0 {
		return report.Input{}, fmt.Errorf(
			"run %s/%s has delivered no envelopes to a ConfigMap sink, so there is no record to render.\n"+
				"A report is built from what was DELIVERED, not from the run's status — add a ConfigMap sink to the profile, "+
				"or render a stored envelope with --from-file", namespace, name)
	}

	in, err := report.Assemble(envelopes)
	if err != nil {
		return report.Input{}, err
	}
	in.Nodes = fingerprintsFor(ctx, c, &run)
	return in, nil
}

// newClient prefers in-cluster credentials, then the usual kubeconfig search.
func newClient(kubeconfig string) (client.Client, error) {
	s := scheme.Scheme
	if err := burninv1alpha1.AddToScheme(s); err != nil {
		return nil, fmt.Errorf("registering the burn-in types: %w", err)
	}

	cfg, err := restConfig(kubeconfig)
	if err != nil {
		return nil, err
	}
	c, err := client.New(cfg, client.Options{Scheme: s})
	if err != nil {
		return nil, fmt.Errorf("connecting to the cluster: %w", err)
	}
	return c, nil
}

func restConfig(kubeconfig string) (*rest.Config, error) {
	if kubeconfig != "" {
		cfg, err := clientcmd.BuildConfigFromFlags("", kubeconfig)
		if err != nil {
			return nil, fmt.Errorf("reading kubeconfig %s: %w", kubeconfig, err)
		}
		return cfg, nil
	}
	if cfg, err := rest.InClusterConfig(); err == nil {
		return cfg, nil
	}
	rules := clientcmd.NewDefaultClientConfigLoadingRules()
	cfg, err := clientcmd.NewNonInteractiveDeferredLoadingClientConfig(rules, &clientcmd.ConfigOverrides{}).ClientConfig()
	if err != nil {
		return nil, fmt.Errorf("no cluster credentials: %w", err)
	}
	return cfg, nil
}

// envelopesFor collects the deliveries from every ConfigMap sink the run's
// profile named.
//
// A sink that cannot be read is skipped with a note on stderr rather than
// failing the whole command: one unreadable sink among several should not stop
// a report the others can support, and the user is told what was missed rather
// than handed a quietly thinner document.
func envelopesFor(ctx context.Context, c client.Client, run *burninv1alpha1.BurnInRun) ([]*contract.Envelope, error) {
	var profile burninv1alpha1.BurnInProfile
	if err := c.Get(ctx, client.ObjectKey{Namespace: run.Namespace, Name: run.Spec.ProfileRef}, &profile); err != nil {
		return nil, fmt.Errorf("reading profile %s: %w", run.Spec.ProfileRef, err)
	}

	var out []*contract.Envelope
	for _, sinkName := range profile.Spec.Sinks {
		var sink burninv1alpha1.BurnInSink
		if err := c.Get(ctx, client.ObjectKey{Namespace: run.Namespace, Name: sinkName}, &sink); err != nil {
			warn("sink %s: %v", sinkName, err)
			continue
		}
		if sink.Spec.Type != burninv1alpha1.SinkConfigMap || sink.Spec.ConfigMap == nil {
			continue
		}

		var cm corev1.ConfigMap
		if err := c.Get(ctx, client.ObjectKey{Namespace: run.Namespace, Name: sink.Spec.ConfigMap.Name}, &cm); err != nil {
			warn("ConfigMap %s (sink %s): %v", sink.Spec.ConfigMap.Name, sinkName, err)
			continue
		}

		// Keys are DeliveryIDs. Sorting makes the read deterministic; Assemble
		// orders the deliveries properly afterwards.
		keys := make([]string, 0, len(cm.Data))
		for k := range cm.Data {
			keys = append(keys, k)
		}
		sort.Strings(keys)

		for _, k := range keys {
			e, err := decodeOne([]byte(cm.Data[k]))
			if err != nil {
				// This entry is part of the record. Dropping it silently would
				// produce a report describing part of a run with nothing saying
				// so, which is the one failure the report packages must not have.
				return nil, fmt.Errorf("ConfigMap %s key %s: %w", cm.Name, k, err)
			}
			if e.Run.UID != string(run.UID) {
				// A ConfigMap sink accumulates across runs. Only this one's
				// deliveries belong in this document.
				continue
			}
			out = append(out, e)
		}
	}
	return out, nil
}

// fingerprintsFor reads the inventory for the nodes the run touched.
//
// Best effort by design: a missing NodeFingerprint means the report says less
// about the hardware, which is honest, and is not a reason to refuse to render
// the verdict. `Known` in the view already distinguishes "no inventory" from
// "the hardware reported nothing".
func fingerprintsFor(ctx context.Context, c client.Client, run *burninv1alpha1.BurnInRun) []report.NodeInfo {
	seen := map[string]bool{}
	var names []string
	for _, r := range run.Status.Results {
		for _, n := range r.Nodes {
			if n != "" && !seen[n] {
				seen[n] = true
				names = append(names, n)
			}
		}
	}
	sort.Strings(names)

	var out []report.NodeInfo
	for _, node := range names {
		var fp burninv1alpha1.NodeFingerprint
		if err := c.Get(ctx, client.ObjectKey{Name: node}, &fp); err != nil {
			if !apierrors.IsNotFound(err) {
				warn("NodeFingerprint %s: %v", node, err)
			}
			continue
		}
		out = append(out, toNodeInfo(node, fp.Status))
	}
	return out
}

// toNodeInfo converts the CRD status into the report package's plain mirror.
//
// Field for field, with nothing invented: a value the fingerprint did not
// capture stays at its zero value, and the renderers already treat that as
// absent rather than as a measurement.
func toNodeInfo(node string, s burninv1alpha1.NodeFingerprintStatus) report.NodeInfo {
	info := report.NodeInfo{
		Name:    node,
		OSImage: s.OSImage,
		Kernel:  s.Kernel,
	}
	if s.CapturedAt != nil {
		t := s.CapturedAt.Time
		info.CapturedAt = &t
	}
	for _, g := range s.GPUs {
		info.GPUs = append(info.GPUs, report.GPUInfo{
			Index:       g.Index,
			Vendor:      g.Vendor,
			Model:       g.Model,
			Arch:        g.Arch,
			MemoryBytes: g.MemoryBytes,
			DriverVer:   g.DriverVer,
		})
	}
	for _, n := range s.NICs {
		info.NICs = append(info.NICs, report.NICInfo{
			Name:       n.Name,
			Role:       n.Role,
			PCIVendor:  n.PCIVendor,
			Model:      n.Model,
			RDMADevice: n.RDMADevice,
			LinkLayer:  n.LinkLayer,
			SpeedMbps:  n.SpeedMbps,
			MTU:        n.MTU,
		})
	}
	return info
}
