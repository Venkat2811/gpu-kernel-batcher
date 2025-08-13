# CUDA Kernel Launch Optimizer - Makefile
# Multi-process SHM queue implementation

# Configuration
CARGO = cargo
RUSTFMT = rustfmt
CLIPPY = cargo clippy
TARGET_DIR = target/release
RESULTS_DIR = results/runs
TIMESTAMP := $(shell date +%Y%m%d_%H%M%S)
RUN_DIR = $(RESULTS_DIR)/$(TIMESTAMP)

# Binary paths
ORCHESTRATOR = $(TARGET_DIR)/orchestrator
PRODUCER = $(PRODUCER)
CONSUMER = $(CONSUMER)

# CUDA configuration
CUDA_VISIBLE_DEVICES ?= 0
CUDA_DEVICE_MAX_CONNECTIONS ?= 1

# Timing configuration
USE_CUDA_EVENTS ?= 0
CUDA_EVENTS_FLAG = $(if $(filter 1,$(USE_CUDA_EVENTS)),--cuda-events,)

# Default target
.PHONY: all
all: build

# Build targets
.PHONY: build
build:
	@echo "Building producer and consumer binaries..."
	$(CARGO) build --release --bins

.PHONY: build-debug
build-debug:
	@echo "Building debug binaries..."
	$(CARGO) build --bins

# Development utilities
.PHONY: clean
clean:
	@echo "Cleaning build artifacts..."
	$(CARGO) clean
	rm -rf $(RESULTS_DIR)/*

.PHONY: fmt
fmt:
	@echo "Formatting code..."
	$(CARGO) fmt --all

.PHONY: clippy
clippy:
	@echo "Running clippy lints..."
	$(CLIPPY) --all-targets --all-features -- -D warnings

.PHONY: docs
docs:
	@echo "Generating documentation..."
	$(CARGO) doc --no-deps --open

# Testing
.PHONY: test
test:
	@echo "Running all tests..."
	$(CARGO) test --all


.PHONY: test-orchestrator
test-orchestrator: build
	@echo "Testing with orchestrator..."
	@mkdir -p $(RUN_DIR)/orchestrated
	$(ORCHESTRATOR) \
		--shm-name /cklo_test_$(TIMESTAMP) \
		--capacity 4096 \
		--workload micro \
		--count 10000 \
		--mode shape-batched \
		--max-batch 128 \
		$(CUDA_EVENTS_FLAG) \
		--out-dir $(RUN_DIR)/orchestrated

# Benchmarking Suite
.PHONY: benchmark
benchmark: benchmark-both

.PHONY: benchmark-fp32
benchmark-fp32: build-release benchmark-gemv-fp32 benchmark-decode-fp32 benchmark-prefill-fp32 analyze

.PHONY: benchmark-fp16
benchmark-fp16: build-release benchmark-gemv-fp16 benchmark-decode-fp16 benchmark-prefill-fp16 analyze

.PHONY: benchmark-both
benchmark-both: benchmark-fp32 benchmark-fp16
	@$(MAKE) analyze


.PHONY: build-release
build-release:
	@echo "🔨 Building release binaries with optimizations..."
	$(CARGO) build --release --bins

.PHONY: benchmark-gemv-fp32
benchmark-gemv-fp32: build-release
	@echo "🔬 Running GEMV-micro benchmark with FP32 (launch-dominated)..."
	@mkdir -p $(RUN_DIR)
	$(ORCHESTRATOR) $(CUDA_EVENTS_FLAG) --shm-name /cklo_gemv_baseline_fp32 --workload gemv-micro --m 64 --k 8192 --count 2000000 --mode none --out-dir $(RUN_DIR)/gemv_baseline_fp32
	$(ORCHESTRATOR) $(CUDA_EVENTS_FLAG) --shm-name /cklo_gemv_batched_fp32 --workload gemv-micro --m 64 --k 8192 --count 2000000 --mode shape-batched --max-batch 512 --idle-us 150 --out-dir $(RUN_DIR)/gemv_batched_fp32

.PHONY: benchmark-gemv-fp16
benchmark-gemv-fp16: build-release
	@echo "🔬 Running GEMV-micro benchmark with FP16 (launch-dominated)..."
	@mkdir -p $(RUN_DIR)
	$(ORCHESTRATOR) $(CUDA_EVENTS_FLAG) --shm-name /cklo_gemv_baseline_fp16 --workload gemv-micro --m 64 --k 8192 --count 2000000 --mode none --dtype fp16 --out-dir $(RUN_DIR)/gemv_baseline_fp16
	$(ORCHESTRATOR) $(CUDA_EVENTS_FLAG) --shm-name /cklo_gemv_batched_fp16 --workload gemv-micro --m 64 --k 8192 --count 2000000 --mode shape-batched --max-batch 512 --idle-us 150 --dtype fp16 --out-dir $(RUN_DIR)/gemv_batched_fp16

.PHONY: benchmark-decode-fp32
benchmark-decode-fp32: build-release
	@echo "🔬 Running Decode benchmark with FP32 (realistic inference)..."
	@mkdir -p $(RUN_DIR)
	$(ORCHESTRATOR) $(CUDA_EVENTS_FLAG) --shm-name /cklo_decode_baseline_fp32 --workload decode --count 400000 --layers 8 --hidden 1024 --batch 1 --mode none --out-dir $(RUN_DIR)/decode_baseline_fp32
	$(ORCHESTRATOR) $(CUDA_EVENTS_FLAG) --shm-name /cklo_decode_batched_fp32 --workload decode --count 400000 --layers 8 --hidden 1024 --batch 1 --mode shape-batched --max-batch 16 --idle-us 300 --out-dir $(RUN_DIR)/decode_batched_fp32

.PHONY: benchmark-decode-fp16
benchmark-decode-fp16: build-release
	@echo "🔬 Running Decode benchmark with FP16 (realistic inference)..."
	@mkdir -p $(RUN_DIR)
	$(ORCHESTRATOR) $(CUDA_EVENTS_FLAG) --shm-name /cklo_decode_baseline_fp16 --workload decode --count 400000 --layers 8 --hidden 1024 --batch 1 --mode none --dtype fp16 --out-dir $(RUN_DIR)/decode_baseline_fp16
	$(ORCHESTRATOR) $(CUDA_EVENTS_FLAG) --shm-name /cklo_decode_batched_fp16 --workload decode --count 400000 --layers 8 --hidden 1024 --batch 1 --mode shape-batched --max-batch 16 --idle-us 300 --dtype fp16 --out-dir $(RUN_DIR)/decode_batched_fp16

.PHONY: benchmark-prefill-fp32
benchmark-prefill-fp32: build-release 
	@echo "🔬 Running Prefill benchmark with FP32 (compute-bound)..."
	@mkdir -p $(RUN_DIR)
	$(ORCHESTRATOR) $(CUDA_EVENTS_FLAG) --shm-name /cklo_prefill_baseline_fp32 --workload prefill --batch 1 --seq-len 256 --hidden 1024 --layers 64 --mode none --out-dir $(RUN_DIR)/prefill_baseline_fp32
	$(ORCHESTRATOR) $(CUDA_EVENTS_FLAG) --shm-name /cklo_prefill_batched_fp32 --workload prefill --batch 1 --seq-len 256 --hidden 1024 --layers 64 --mode shape-batched --max-batch 8 --idle-us 200 --out-dir $(RUN_DIR)/prefill_batched_fp32

.PHONY: benchmark-prefill-fp16
benchmark-prefill-fp16: build-release
	@echo "🔬 Running Prefill benchmark with FP16 (compute-bound)..."
	@mkdir -p $(RUN_DIR)
	$(ORCHESTRATOR) $(CUDA_EVENTS_FLAG) --shm-name /cklo_prefill_baseline_fp16 --workload prefill --batch 1 --seq-len 256 --hidden 1024 --layers 64 --mode none --dtype fp16 --out-dir $(RUN_DIR)/prefill_baseline_fp16
	$(ORCHESTRATOR) $(CUDA_EVENTS_FLAG) --shm-name /cklo_prefill_batched_fp16 --workload prefill --batch 1 --seq-len 256 --hidden 1024 --layers 64 --mode shape-batched --max-batch 8 --idle-us 200 --dtype fp16 --out-dir $(RUN_DIR)/prefill_batched_fp16

.PHONY: analyze
analyze:
	@echo "📊 Analyzing latest benchmark results..."
	@latest_dir=$$(ls -1t $(RESULTS_DIR) 2>/dev/null | head -1); \
	if [ -n "$$latest_dir" ]; then \
		echo "Using latest run: $$latest_dir"; \
		/usr/bin/python3 scripts/analyze_results.py $(RESULTS_DIR)/$$latest_dir; \
	else \
		echo "No timestamped runs found. Use 'make reorganize' first."; \
	fi

.PHONY: analyze-all
analyze-all:
	@echo "📊 Analyzing all benchmark results..."
	@for run_dir in $$(ls -1t $(RESULTS_DIR) 2>/dev/null); do \
		echo ""; \
		echo "=== Results from $$run_dir ==="; \
		python3 scripts/analyze_results.py $(RESULTS_DIR)/$$run_dir; \
	done


# Development helpers
.PHONY: check
check: fmt clippy test
	@echo "All checks passed!"

.PHONY: run-producer
run-producer: build
	@echo "Running producer standalone..."
	$(PRODUCER) --shm-name /cklo_test --capacity 65536 --workload micro --count 1000

.PHONY: run-consumer
run-consumer: build
	@echo "Running consumer standalone..."
	CUDA_VISIBLE_DEVICES=$(CUDA_VISIBLE_DEVICES) \
	$(CONSUMER) --shm-name /cklo_test --capacity 65536 --mode none --gpu-id 0

# Installation check
.PHONY: check-tools
check-tools:
	@echo "Checking required tools..."
	@which cargo >/dev/null || (echo "ERROR: cargo not found"; exit 1)
	@which rustc >/dev/null || (echo "ERROR: rustc not found"; exit 1)
	@which nvidia-smi >/dev/null || (echo "WARNING: nvidia-smi not found")
	@echo "Tool check complete"

# Help target
.PHONY: help
help:
	@echo "CUDA Kernel Launch Optimizer - Make targets"
	@echo ""
	@echo "Build & Development:"
	@echo "  all/build    - Build release binaries"
	@echo "  build-debug  - Build debug binaries"
	@echo "  clean        - Remove build artifacts"
	@echo "  fmt          - Format code"
	@echo "  clippy       - Run lints"
	@echo "  docs         - Generate documentation"
	@echo "  check        - Run fmt, clippy, and tests"
	@echo ""
	@echo "Testing:"
	@echo "  test         - Run all tests"
	@echo ""
	@echo "Benchmarking:"
	@echo "  benchmark         - Run complete benchmark suite (FP32 + FP16)"
	@echo "  benchmark-fp32    - Run all FP32 benchmarks"
	@echo "  benchmark-fp16    - Run all FP16 benchmarks"
	@echo "  benchmark-both    - Run both FP32 and FP16 benchmarks"
	@echo "  benchmark-gemv-fp32   - Run GEMV-micro FP32 benchmark" 
	@echo "  benchmark-gemv-fp16   - Run GEMV-micro FP16 benchmark"
	@echo "  benchmark-decode-fp32 - Run Decode FP32 benchmark"
	@echo "  benchmark-decode-fp16 - Run Decode FP16 benchmark"
	@echo "  benchmark-prefill-fp32- Run Prefill FP32 benchmark"
	@echo "  benchmark-prefill-fp16- Run Prefill FP16 benchmark"
	@echo "  analyze           - Analyze latest benchmark results"
	@echo "  analyze-all       - Analyze all benchmark results"
	@echo ""
	@echo "Features:"
	@echo "  - Fast wall-clock timing by default"
	@echo "  - Optional GPU event timing: USE_CUDA_EVENTS=1 make benchmark"
	@echo "  - Dual metrics: Reports both GPU execution time and wall-clock time"
	@echo ""
	@echo "Environment variables:"
	@echo "  CUDA_VISIBLE_DEVICES (default: 0)"
	@echo "  USE_CUDA_EVENTS (default: 0) - Set to 1 for accurate GPU timing"
	@echo "  CUDA_DEVICE_MAX_CONNECTIONS (default: 1)"