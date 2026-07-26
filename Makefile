# Glimmer Burn-In Operator
CONTROLLER_GEN := go run sigs.k8s.io/controller-tools/cmd/controller-gen@v0.17.2
IMG ?= ghcr.io/baldwinspc/glimmer-burnin:dev
PLATFORMS ?= linux/arm64,linux/amd64

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

.PHONY: build
build:
	go build -o bin/manager ./cmd

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
