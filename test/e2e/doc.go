// Package e2e is the operator's end-to-end suite: the SHIPPED manifests, the
// SHIPPED image, and a real cluster with a real scheduler and a real kubelet.
//
// # Why this exists, and what only it can catch
//
// envtest (test/envtest) gives a real apiserver, which is enough for RBAC,
// schema validation, status subresources and resourceVersion conflicts. What it
// does not give is a SCHEDULER or a KUBELET, and two of the five bugs that
// reached production live exactly there:
//
//   - The cordon deadlock. The operator cordoned the node it was about to test
//     and then created a runner pod that did not tolerate that cordon. Every
//     test, on every node, forever. Nothing without a scheduler can see it: the
//     fake client places pods by assignment, and envtest never places them at
//     all.
//   - The manifests themselves. config/manager was swallowed by an unanchored
//     `manager` line in .gitignore and a whole release shipped with no way to
//     deploy the operator. No Go test can catch that, because the Go code was
//     fine.
//
// SO THE MOST VALUABLE THING IN THIS DIRECTORY IS NOT A TEST. It is the CI step
// that applies config/crd, config/rbac and config/manager to a cluster and
// waits for the Deployment to become Available. That single step is worth more
// than any assertion below, it runs before all of them, and it is the reason
// the e2e job exists at all.
//
// # Running it
//
// The suite is behind the `e2e` build tag so `go test ./...` can never pick it
// up, and it expects a cluster already carrying the operator:
//
//	kind create cluster --config test/e2e/kind.yaml
//	docker build -t glimmer-burnin:e2e .
//	kind load docker-image glimmer-burnin:e2e
//	kubectl apply -f config/crd -f config/rbac -f config/manager
//	kubectl -n glimmer-burnin-system set image deploy/burnin-controller-manager manager=glimmer-burnin:e2e
//	make test-e2e
//
// # The runner
//
// Every test here uses a `custom` runner on a plain CPU image, because every
// bug this suite is aimed at is ORCHESTRATION and not GPU: scheduling,
// cordoning, RBAC, rendezvous, delivery, recovery. A GPU would add nothing to
// any assertion in this package and would make the suite unrunnable on the
// hosted runners CI actually has.
package e2e
