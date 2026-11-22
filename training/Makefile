<<<<<<< HEAD
CC = xcrun clang
CFLAGS = -O2 -Wall -Wno-deprecated-declarations -DACCELERATE_NEW_LAPACK -fobjc-arc
CC_C = xcrun clang
CFLAGS_C = -O2 -Wall -Wextra -Werror -std=c11
FRAMEWORKS = -framework Foundation -framework CoreML -framework IOSurface
LDFLAGS = $(FRAMEWORKS) -ldl

HEADERS_LARGE = stories_config.h stories_io.h stories_mil.h stories_cpu_ops.h data_validation.h
=======
CC = xcrun clang
CC_C = xcrun clang

# The private ANE APIs (_ANEInMemoryModelDescriptor etc.) are reached through objc_msgSend and trigger deprecation
# warnings on newer macOS SDKs; suppressing them is intentional. `make check-deprecated` shows what is hidden.
ANE_COMPAT = -Wno-deprecated-declarations

# Security hardening flags (docs/reports/security-audit-2026-03-02.md, LOW-01).
# -D_FORTIFY_SOURCE=2 is implicitly active at -O2 with Apple's LLVM toolchain.
SEC_FLAGS = -fstack-protector-strong -Wformat-security

CFLAGS = -O2 -Wall -DACCELERATE_NEW_LAPACK $(ANE_COMPAT) -fobjc-arc $(SEC_FLAGS)
CFLAGS_C = -O2 -Wall -Wextra -Werror -std=c11
CFLAGS_DEBUG = -O0 -g -Wall $(ANE_COMPAT) -fobjc-arc -fsanitize=address,undefined
FRAMEWORKS = -framework Foundation -framework CoreML -framework IOSurface
LDFLAGS = $(FRAMEWORKS) -ldl

HEADERS_LARGE = stories_config.h stories_io.h stories_mil.h stories_cpu_ops.h data_validation.h
HEADERS_OPT = $(HEADERS_LARGE) stories_cpu_ops_opt.h
>>>>>>> middle/agglomerate

HEADERS_ANE = $(HEADERS_LARGE) ane_rmsnorm_bwd.h ane_classifier.h

train: train.m ane_runtime.h ane_mil_gen.h model.h forward.h backward.h
	$(CC) $(CFLAGS) -o $@ train.m $(LDFLAGS) -framework Accelerate

train_large: train_large.m $(HEADERS_LARGE)
	$(CC) $(CFLAGS) -o $@ train_large.m $(LDFLAGS) -framework Accelerate

train_large_ane: train_large_ane.m $(HEADERS_ANE)
	$(CC) $(CFLAGS) -o $@ train_large_ane.m $(LDFLAGS) -framework Accelerate

train_opt: train_opt.m $(HEADERS_OPT)
	$(CC) $(CFLAGS) -o $@ train_opt.m $(LDFLAGS) -framework Accelerate -framework Metal -framework MetalPerformanceShaders

train_double_buffer: train_double_buffer.m $(HEADERS_LARGE)
	$(CC) $(CFLAGS) -o $@ train_double_buffer.m $(LDFLAGS) -framework Accelerate

PROBES = test_weight_reload test_perf_stats test_qos_sweep test_ane_advanced test_sdk_model

test_rmsnorm_bwd: test_rmsnorm_bwd.m $(HEADERS_ANE)
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS) -framework Accelerate

test_classifier: test_classifier.m $(HEADERS_ANE)
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS) -framework Accelerate

test_weight_reload: test_weight_reload.m
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

test_perf_stats: test_perf_stats.m
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

test_qos_sweep: test_qos_sweep.m
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

test_ane_advanced: test_ane_advanced.m
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

# ANE private API research (chaining, E5 runtime, custom MIL)
test_chaining: test_chaining.m
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

test_chaining_v2: test_chaining_v2.m
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

test_bench_paths: test_bench_paths.m ane_runtime.h
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

test_ane_model: test_ane_model.m
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS) -framework Metal

test_throughput_ceiling: test_throughput_ceiling.m ane_runtime.h
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

test_coreml_chaining: test_coreml_chaining.m
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS) -framework Metal

test_e5_validate: test_e5_validate.m
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS) -framework Metal

test_mil_custom: test_mil_custom.m
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS) -framework Accelerate

# ANE SDK (layers/)
benchmark_ane: benchmark_ane.m
	$(CC) $(CFLAGS) -o benchmark_ane benchmark_ane.m $(LDFLAGS)

test_sdk_layers: test_sdk_layers.m
	$(CC) $(CFLAGS) -O2 -o test_sdk_layers test_sdk_layers.m $(LDFLAGS) -framework Accelerate

test_sdk_model: test_sdk_model.m
	$(CC) $(CFLAGS) -O2 -o test_sdk_model test_sdk_model.m $(LDFLAGS) -framework Accelerate

regression:
	python3 regression_test.py

# M5 benchmark suites
m5_performance_suite: m5_performance_suite.m ane_runtime.h
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

m5_pipeline_suite: m5_pipeline_suite.m ane_runtime.h ane_mil_gen.h
	$(CC) $(CFLAGS) -Wno-unused-function -Wno-gnu-folding-constant -o $@ $< $(LDFLAGS)

# Training-data validation (plain C, strict flags)
test_data_validation: test_data_validation.c data_validation.h
	$(CC_C) $(CFLAGS_C) -o $@ $<

security-tests: test_data_validation

probes: $(PROBES)

tokenize:
	python3 tokenize.py

data: tokenize
	@bash download_data.sh

setup: data
	@echo "=== Setup complete ==="
	@echo "Data:  tinystories_data00.bin"
	@echo "To train: make train_large && ./train_large"
	@echo "Override paths: ANE_MODEL_PATH=... ANE_DATA_PATH=... ./train_large"

# Show the active build flags, to verify that the hardening flags are on (LOW-01)
verify-flags:
	@echo "=== Active CFLAGS ==="
	@echo "$(CFLAGS)"
	@echo "=== Compiler version ==="
	@xcrun clang --version

# Build train_large without the deprecation suppression to inspect what it hides (LOW-02); informational only,
# the warnings from the private ANE API usage are expected.
check-deprecated:
	@echo "=== Building without deprecation suppression (informational) ==="
	@$(CC) $(CFLAGS:-Wno-deprecated-declarations=-Wdeprecated-declarations) \
	    -o /dev/null train_large.m $(LDFLAGS) -framework Accelerate 2>&1 | \
	    grep -i "deprecated" | sort -u || echo "(no additional deprecation warnings)"

clean:
	rm -f train train_large train_large_ane train_opt train_double_buffer $(PROBES) test_rmsnorm_bwd test_classifier test_chaining test_chaining_v2 test_bench_paths test_ane_model test_throughput_ceiling test_coreml_chaining test_e5_validate test_mil_custom test_data_validation test_sdk_layers benchmark_ane m5_performance_suite m5_pipeline_suite

.PHONY: clean tokenize probes security-tests verify-flags check-deprecated data setup regression
