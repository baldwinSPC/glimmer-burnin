// Package v1alpha1 contains the API types for the Glimmer Burn-In Operator —
// a vendor-neutral, Kubernetes-native hardware acceptance-testing controller.
//
// It is deliberately standalone: nothing here imports its control-plane consumer.
// Results are exported through a BurnInSink (a generic contract), so the operator
// runs on any cluster and integrates with a control plane — or anything else —
// without a code dependency. CI enforces the no-import rule.
//
// +kubebuilder:object:generate=true
// +groupName=burnin.glimmer.ai
package v1alpha1

import (
	"k8s.io/apimachinery/pkg/runtime/schema"
	"sigs.k8s.io/controller-runtime/pkg/scheme"
)

var (
	// GroupVersion is the group/version used to register these objects.
	GroupVersion = schema.GroupVersion{Group: "burnin.glimmer.ai", Version: "v1alpha1"}

	// SchemeBuilder registers the API types with a runtime.Scheme.
	SchemeBuilder = &scheme.Builder{GroupVersion: GroupVersion}

	// AddToScheme adds the types in this group-version to the given scheme.
	AddToScheme = SchemeBuilder.AddToScheme
)

func init() {
	SchemeBuilder.Register(
		&BurnInTest{}, &BurnInTestList{},
		&BurnInProfile{}, &BurnInProfileList{},
		&BurnInRun{}, &BurnInRunList{},
		&BurnInSchedule{}, &BurnInScheduleList{},
		&NodeFingerprint{}, &NodeFingerprintList{},
		&BurnInSink{}, &BurnInSinkList{},
	)
}
