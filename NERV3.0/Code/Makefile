
# ============================================================================
# NERV — the unified build system
# ============================================================================
# Everything from one command: `make all` builds every target, runs every
# test, and produces every artifact. Individual targets for day-to-day work.
# ============================================================================

SHELL := /bin/bash
.DEFAULT_GOAL := help
TARGET_DIR := target/release

.PHONY: help
help: ## Show this help
    @echo "NERV Build System"
    @echo "=================="
    @echo ""
    @echo "  make build          Build all release binaries"
    @echo "  make test           Run all workspace tests"
    @echo "  make fast-test      Run core tests only (no STARK proving)"
    @echo "  make firewall       Run the Axiom 3 firewall test"
    @echo "  make conformance    Run the conformance suite"
    @echo "  make testkit        Run the TLA+ schedule corpus"
    @echo "  make wasm           Build the web wallet (WASM/PWA)"
    @echo "  make tui            Build the terminal wallet"
    @echo "  make gui            Build the desktop wallet"
    @echo "  make docker         Build Docker images"
    @echo "  make all            Everything: build + test + wasm + docker"
    @echo "  make clean          Clean all build artifacts"

# --- Building -------------------------------------------------------------

.PHONY: build
build: ## Build all release binaries
    cargo build --release --workspace

.PHONY: tui
tui: ## Build the terminal wallet
    cargo build --release -p nerv-tui

.PHONY: gui
gui: ## Build the desktop wallet (native egui)
    cargo build --release -p nerv-gui

.PHONY: wasm
wasm: ## Build the web wallet (WASM/PWA)
    cd apps/nerv-gui && bash build_web.sh

# --- Testing ----------------------------------------------------------------

.PHONY: test
test: ## Run all workspace tests
    cargo test --workspace

.PHONY: fast-test
fast-test: ## Run core tests (skip the slow STARK tests)
    cargo test -p nerv-core -p nerv-crypto -p nerv-custody \
               -p nerv-codec -p nerv-seal -p nerv-state \
               -p nerv-economy -p nerv-knowledge

.PHONY: firewall
firewall: ## Run the Axiom 3 firewall test
    cargo test -p nerv-knowledge the_firewall_is_a_build_rule -- --nocapture

.PHONY: firewall-full
firewall-full: ## Run the full delete test (minutes; separate build)
    cargo test -p nerv-knowledge --release -- --ignored \
        the_workspace_builds_without_nerv_knowledge --nocapture

.PHONY: conformance
conformance: ## Run the conformance suite
    cargo test -p nerv-conformance

.PHONY: testkit
testkit: ## Run the TLA+ schedule corpus (T5–T7)
    cargo test -p nerv-testkit

.PHONY: wallet-core-test
wallet-core-test: ## Run the wallet state machine tests
    cargo test -p nerv-wallet-core

.PHONY: fuzz
fuzz: ## Run the fuzz targets (requires cargo-fuzz)
    cd fuzz && cargo fuzz run codec -- -max_total_time=60

# --- Mobile -----------------------------------------------------------------

.PHONY: ios
ios: ## Build the iOS app (requires Xcode)
    bash apps/nerv-gui/mobile/build_ios.sh

.PHONY: android
android: ## Build the Android app (requires Android SDK)
    bash apps/nerv-gui/mobile/build_android.sh

# --- Docker -------------------------------------------------------------------

.PHONY: docker
docker: ## Build all Docker images
    docker build -t nerv-node -f docker/Dockerfile.node .
    docker build -t nerv-relay -f docker/Dockerfile.relay .
    docker build -t nerv-aggregator -f docker/Dockerfile.aggregator .

.PHONY: docker-up
docker-up: docker ## Start a local network with docker-compose
    cd docker && docker compose up -d

.PHONY: docker-down
docker-down: ## Stop the local network
    cd docker && docker compose down

# --- Everything -----------------------------------------------------------------

.PHONY: all
all: build test firewall conformance testkit wasm ## Build, test, and package everything
    @echo ""
    @echo "✓ All targets built and tested."
    @echo "  Binaries in $(TARGET_DIR)/"
    @echo "  Web wallet in apps/nerv-gui/dist/"

.PHONY: clean
clean: ## Clean all build artifacts
    cargo clean
    rm -rf apps/nerv-gui/pkg apps/nerv-gui/dist
    docker system prune -f 2>/dev/null || true
