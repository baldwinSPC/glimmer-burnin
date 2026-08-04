// Package envtest holds the operator's envtest suite: controller invariants
// exercised against a REAL kube-apiserver and etcd instead of a fake client.
//
// # Why this exists
//
// The controller's unit tests all run against controller-runtime's fake
// client. That client is a map with a type checker in front of it: it does not
// enforce RBAC, it does not validate against the CRD schema, it does not apply
// defaults, it does not separate the status subresource from the spec, and its
// resourceVersion conflicts are not the ones a real apiserver produces. Five
// shipped bugs went through a green fake-client suite for exactly those
// reasons. Adding more fake-client tests cannot close that gap, because the gap
// IS the fake client.
//
// What a real apiserver buys, concretely:
//
//   - RBAC is enforced, so the generated ClusterRole in config/rbac is TESTED
//     rather than assumed. A run driven through an impersonated service account
//     fails the moment a verb is missing — which is how a Pair-scope test that
//     creates a headless Service shipped against a ClusterRole with no
//     `services` rule.
//   - The CRD schema is applied, so defaulting, enums, patterns and listType
//     keys are real. A field present in the Go type and absent from the
//     manifest is silently dropped by the apiserver, and this is where that
//     shows up.
//   - status is a subresource, so a spec write cannot carry status with it and
//     a status write cannot carry spec.
//   - resourceVersion conflicts are real, which is the only way to express the
//     invariant this package exists for: A POD MAY ONLY BE DESTROYED ONCE THE
//     STATUS JUSTIFYING ITS DESTRUCTION IS READABLE FROM THE APISERVER.
//
// # What it deliberately does NOT cover
//
// envtest is an apiserver and etcd — there is no scheduler, no kubelet and no
// controller-manager. Pods are never scheduled and never run; the suite writes
// their status itself. Anything whose failure mode is "the scheduler refused to
// place this pod" (the cordon/toleration deadlock) is therefore invisible here
// and belongs to the kind e2e in test/e2e.
//
// # Running it
//
// The suite needs the envtest control-plane binaries and SKIPS ITSELF when they
// are absent, so `go test ./...` stays green on a laptop that has not installed
// them:
//
//	make envtest-assets   # download kube-apiserver + etcd
//	make test-envtest     # run this suite
//
// CI sets BURNIN_ENVTEST=required, which turns that skip into a hard failure —
// a suite that silently skips in CI is worse than no suite at all.
package envtest
