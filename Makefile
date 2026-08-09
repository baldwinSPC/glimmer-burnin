# Glimmer Burn-In Operator
CONTROLLER_GEN := go run sigs.k8s.io/controller-tools/cmd/controller-gen@v0.17.2
IMG ?= ghcr.io/baldwinspc/glimmer-burnin:dev
PLATFORMS ?= linux/arm64,linux/amd64

# setup-envtest is versioned by BRANCH rather than by tag, so the pin is the
# release branch matching the controller-runtime in go.mod. Bump both together:
# an envtest control plane from a different minor is not the apiserver this
# operator will run against.
SETUP_ENVTEST := go run sigs.k8s.io/controller-runtime/tools/setup-envtest@release-0.24
# The apiserver + etcd the envtest suite runs. Kept close to what a real cluster
# serves; setup-envtest downloads it on demand.
ENVTEST_K8S_VERSION ?= 1.36.2

# The e2e cluster and the image it runs. E2E_IMG is loaded into kind rather than
# pulled, so the tag never has to exist in a registry.
KIND_CLUSTER ?= glimmer-burnin-e2e
E2E_IMG ?= glimmer-burnin:e2e
# The CPU-only image every e2e runner executes. Nothing here needs a GPU: every
# bug the e2e exists for is orchestration.
E2E_RUNNER_IMAGE ?= busybox:1.37.0

.PHONY: all
all: generate manifests fmt vet test build

.PHONY: generate
generate: ## Generate deepcopy methods.
	$(CONTROLLER_GEN) object:headerFile=hack/boilerplate.go.txt paths=./api/...

.PHONY: manifests
manifests: ## Generate CRDs + RBAC.
	$(CONTROLLER_GEN) crd rbac:roleName=burnin-manager-role webhook paths=./... \
		output:crd:artifacts:config=config/crd output:rbac:artifacts:config=config/rbac

.PHONY: fmt
fmt:
	gofmt -w .

.PHONY: vet
vet:
	go vet ./...

.PHONY: test
test:
	go test ./... -count=1

# ── envtest: controller invariants against a real apiserver ───────────────────
#
# test/envtest is part of `make test` and SKIPS ITSELF when the control-plane
# binaries are absent, so a laptop that has never run these still gets a green
# `go test ./...`. CI sets BURNIN_ENVTEST=required, which turns that skip into a
# hard failure — a suite that silently skips in CI has negative value.

.PHONY: deploy
deploy: manifests ## Deploy the operator into the current kubectl context, from config/.
	kubectl apply -f config/crd
	kubectl apply -f config/rbac
	kubectl apply -f config/manager

.PHONY: undeploy
undeploy: ## Remove the operator. CRDs are left ALONE: deleting them deletes every
	## BurnInRun in the cluster, and a verdict is worth more than a tidy uninstall.
	kubectl delete -f config/manager --ignore-not-found
	kubectl delete -f config/rbac --ignore-not-found

.PHONY: helm-lint
helm-lint: ## Lint and render the chart.
	helm lint deploy/charts/glimmer-burnin
	helm template burnin deploy/charts/glimmer-burnin --namespace glimmer-burnin-system >/dev/null
# There must be exactly ONE chart. A stray copy — deploy/deploy/charts, say —
# lints clean, because linting names the good path and never looks elsewhere, so
# nothing would report it. What it does instead is rot: a fix lands in one copy,
# somebody installs the other, and the difference surfaces as an operator that
# behaves unlike its manifests.
	@found=$$(find deploy -name Chart.yaml | wc -l | tr -d ' '); \
	if [ "$$found" != "1" ]; then \
		echo "expected exactly one Chart.yaml under deploy/, found $$found:"; \
		find deploy -name Chart.yaml; \
		exit 1; \
	fi

.PHONY: licenses
licenses: ## Enforce the licence policy over the whole Go module graph.
	go run ./hack/licensecheck

.PHONY: vulncheck
vulncheck: ## Report vulnerabilities in code paths this project actually reaches.
	go run golang.org/x/vuln/cmd/govulncheck@v1.1.4 ./...

.PHONY: supply-chain
supply-chain: licenses vulncheck ## Everything the supply-chain CI job runs, minus the linter binary.

.PHONY: envtest-assets
envtest-assets: ## Download the envtest kube-apiserver + etcd binaries.
	@$(SETUP_ENVTEST) use $(ENVTEST_K8S_VERSION) -p path

.PHONY: test-envtest
test-envtest: ## Run the envtest suite (real apiserver + etcd).
	KUBEBUILDER_ASSETS="$$($(SETUP_ENVTEST) use $(ENVTEST_K8S_VERSION) -p path)" \
	BURNIN_ENVTEST=required go test ./test/envtest/... -count=1

# ── e2e: the shipped manifests on a real cluster ──────────────────────────────
#
# `make e2e` is the whole loop. The step that matters most is the plain
# `kubectl apply -f config/...` inside e2e-deploy: a release once shipped with
# no deployment manifest at all, and no Go test could have caught it.

.PHONY: e2e
e2e: e2e-cluster e2e-deploy test-e2e ## Create a kind cluster, deploy the operator, run the e2e suite.

.PHONY: e2e-cluster
e2e-cluster: ## Create the e2e kind cluster (1 control-plane + 3 workers).
	kind create cluster --name $(KIND_CLUSTER) --config test/e2e/kind.yaml
	docker pull $(E2E_RUNNER_IMAGE)
	kind load docker-image $(E2E_RUNNER_IMAGE) --name $(KIND_CLUSTER)

.PHONY: e2e-deploy
e2e-deploy: ## Apply config/ and point the Deployment at a locally built image.
	kubectl apply -f config/crd -f config/rbac -f config/manager
	docker build -t $(E2E_IMG) .
	kind load docker-image $(E2E_IMG) --name $(KIND_CLUSTER)
	kubectl -n glimmer-burnin-system set image deployment/burnin-controller-manager manager=$(E2E_IMG)
	kubectl -n glimmer-burnin-system rollout status deployment/burnin-controller-manager --timeout=5m

.PHONY: test-e2e
test-e2e: ## Run the e2e suite against the current kubectl context.
	E2E_RUNNER_IMAGE=$(E2E_RUNNER_IMAGE) go test -tags e2e ./test/e2e/... -count=1 -v -timeout 30m

.PHONY: e2e-clean
e2e-clean:
	kind delete cluster --name $(KIND_CLUSTER)

.PHONY: build
build:
	go build -o bin/manager ./cmd

# The user-facing CLI, built separately from the manager on purpose: this is a
# thing a person runs on a laptop against files, not a controller that runs in
# a cluster under a service account.
#
# It is also installed as kubectl-burnin, which is all `kubectl burnin report`
# requires — that is how kubectl discovers plugins.
CLI_VERSION ?= $(shell git describe --tags --always --dirty 2>/dev/null || echo "(devel)")

.PHONY: build-cli
build-cli: ## Build the burnin CLI (also as kubectl-burnin).
	go build -ldflags="-X main.version=$(CLI_VERSION)" -o bin/burnin ./cmd/burnin
	cp bin/burnin bin/kubectl-burnin

.PHONY: run
run: manifests generate
	go run ./cmd

.PHONY: install
install: manifests ## Install CRDs into the current cluster.
	kubectl apply -f config/crd

.PHONY: uninstall
uninstall:
	kubectl delete -f config/crd

.PHONY: docker-build
docker-build: ## Multi-arch build + push (set IMG).
	docker buildx build --platform $(PLATFORMS) -t $(IMG) --push .

.PHONY: lint
lint: fmt vet
	@test -z "$$(gofmt -l .)" || (echo "gofmt needed" && exit 1)
