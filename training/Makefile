<<<<<<< HEAD
CC = xcrun clang
<<<<<<< HEAD
<<<<<<< HEAD
CFLAGS = -O2 -Wall -DACCELERATE_NEW_LAPACK -fobjc-arc
=======

ANE_COMPAT = -Wno-deprecated-declarations
SEC_FLAGS = -fstack-protector-strong -Wformat-security

CFLAGS = -O2 -Wall $(ANE_COMPAT) -fobjc-arc $(SEC_FLAGS)
CFLAGS_DEBUG = -O0 -g -Wall $(ANE_COMPAT) -fobjc-arc -fsanitize=address,undefined
>>>>>>> be96079bbf2511a7b4a5569a026cf03883ebfbaf
=======

# Required: Private ANE APIs (_ANEInMemoryModelDescriptor etc.) via objc_msgSend
# trigger deprecation warnings on newer macOS SDKs. This flag is intentional.
# Run 'make check-deprecated' to inspect which declarations are currently suppressed.
ANE_COMPAT = -Wno-deprecated-declarations

# Security hardening flags (ref: docs/reports/security-audit-2026-03-02.md LOW-01)
# Note: -D_FORTIFY_SOURCE=2 is implicitly active at -O2 on macOS (Apple LLVM toolchain).
SEC_FLAGS = -fstack-protector-strong -Wformat-security

CFLAGS = -O2 -Wall -DACCELERATE_NEW_LAPACK $(ANE_COMPAT) -fobjc-arc $(SEC_FLAGS)
CFLAGS_DEBUG = -O0 -g -Wall $(ANE_COMPAT) -fobjc-arc -fsanitize=address,undefined
>>>>>>> 1a7a05708bf5e34349150cb6e0805c16027b9fd7
FRAMEWORKS = -framework Foundation -framework CoreML -framework IOSurface
LDFLAGS = $(FRAMEWORKS) -ldl

HEADERS_LARGE = stories_config.h stories_io.h stories_mil.h stories_cpu_ops.h
HEADERS_OPT = $(HEADERS_LARGE) stories_cpu_ops_opt.h
=======
CC = xcrun clang
CFLAGS = -O2 -Wall -Wno-deprecated-declarations -fobjc-arc
CC_C = xcrun clang
CFLAGS_C = -O2 -Wall -Wextra -Werror -std=c11
FRAMEWORKS = -framework Foundation -framework CoreML -framework IOSurface
LDFLAGS = $(FRAMEWORKS) -ldl

HEADERS_LARGE = stories_config.h stories_io.h stories_mil.h stories_cpu_ops.h data_validation.h
>>>>>>> 60b0512be32bf88f096481ac105f5558c0afa631

HEADERS_ANE = $(HEADERS_LARGE) ane_rmsnorm_bwd.h ane_classifier.h

train: train.m ane_runtime.h ane_mil_gen.h model.h forward.h backward.h
	$(CC) $(CFLAGS) -o $@ train.m $(LDFLAGS) -framework Accelerate

train_large: train_large.m $(HEADERS_LARGE)
	$(CC) $(CFLAGS) -o $@ train_large.m $(LDFLAGS) -framework Accelerate

<<<<<<< HEAD
train_large_ane: train_large_ane.m $(HEADERS_ANE)
	$(CC) $(CFLAGS) -o $@ train_large_ane.m $(LDFLAGS) -framework Accelerate

<<<<<<< HEAD
train_opt: train_opt.m $(HEADERS_OPT)
	$(CC) $(CFLAGS) -o $@ train_opt.m $(LDFLAGS) -framework Accelerate -framework Metal -framework MetalPerformanceShaders

=======
HEADERS_OPT = $(HEADERS_LARGE) stories_cpu_ops_opt.h

train_opt: train_opt.m $(HEADERS_OPT)
	$(CC) $(CFLAGS) -o $@ train_opt.m $(LDFLAGS) -framework Accelerate -framework Metal -framework MetalPerformanceShaders

train_double_buffer: train_double_buffer.m $(HEADERS_LARGE)
	$(CC) $(CFLAGS) -o $@ train_double_buffer.m $(LDFLAGS) -framework Accelerate

>>>>>>> be96079bbf2511a7b4a5569a026cf03883ebfbaf
PROBES = test_weight_reload test_perf_stats test_qos_sweep test_ane_advanced
=======
benchmark_ane: benchmark_ane.m
	$(CC) $(CFLAGS) -o benchmark_ane benchmark_ane.m $(LDFLAGS)

test_sdk_layers: test_sdk_layers.m
	$(CC) $(CFLAGS) -O2 -o test_sdk_layers test_sdk_layers.m $(LDFLAGS) -framework Accelerate

test_sdk_model: test_sdk_model.m
	$(CC) $(CFLAGS) -O2 -o test_sdk_model test_sdk_model.m $(LDFLAGS) -framework Accelerate

regression:
	python3 regression_test.py

PROBES = test_weight_reload test_perf_stats test_qos_sweep test_ane_advanced test_sdk_model
>>>>>>> e113fae6832b497b79e1b32f0861b1797e007f53

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

<<<<<<< HEAD
test_ane_advanced: test_ane_advanced.m
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

<<<<<<< HEAD
test_chaining: test_chaining.m
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

=======
m5_performance_suite: m5_performance_suite.m ane_runtime.h
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

m5_pipeline_suite: m5_pipeline_suite.m ane_runtime.h ane_mil_gen.h
	$(CC) $(CFLAGS) -Wno-unused-function -Wno-gnu-folding-constant -o $@ $< $(LDFLAGS)

>>>>>>> 6f398781d734ae1cc53139358829082c0f83000c
probes: $(PROBES)

data: tokenize
	@bash download_data.sh

tokenize:
	python3 tokenize.py

<<<<<<< HEAD
setup: data
	@echo "=== Setup complete ==="
	@echo "Data:  tinystories_data00.bin"
	@echo "To train: make train_large && ./train_large"
	@echo "Override paths: ANE_MODEL_PATH=... ANE_DATA_PATH=... ./train_large"

=======
# Show active build flags — verify security hardening is active (LOW-01)
>>>>>>> 1a7a05708bf5e34349150cb6e0805c16027b9fd7
verify-flags:
	@echo "=== Active CFLAGS ==="
	@echo "$(CFLAGS)"
	@echo "=== Compiler version ==="
	@xcrun clang --version

<<<<<<< HEAD
=======
# Build train_large without deprecation suppression to inspect hidden warnings (LOW-02)
# Informational only — deprecated warnings from private ANE API usage are expected.
check-deprecated:
	@echo "=== Building without deprecation suppression (informational) ==="
	@$(CC) $(CFLAGS:-Wno-deprecated-declarations=-Wdeprecated-declarations) \
	    -o /dev/null train_large.m $(LDFLAGS) -framework Accelerate 2>&1 | \
	    grep -i "deprecated" | sort -u || echo "(keine zusaetzlichen deprecated-Warnungen)"

>>>>>>> 1a7a05708bf5e34349150cb6e0805c16027b9fd7
clean:
<<<<<<< HEAD
<<<<<<< HEAD
	rm -f train train_large train_large_ane train_opt $(PROBES) test_rmsnorm_bwd test_classifier
=======
	rm -f train train_large train_large_ane $(PROBES) test_rmsnorm_bwd test_classifier m5_performance_suite m5_pipeline_suite
>>>>>>> 6f398781d734ae1cc53139358829082c0f83000c

<<<<<<< HEAD
.PHONY: clean tokenize probes
=======
test_ane_advanced: test_ane_advanced.m
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

test_data_validation: test_data_validation.c data_validation.h
	$(CC_C) $(CFLAGS_C) -o $@ $<

probes: $(PROBES)

security-tests: test_data_validation

tokenize:
	python3 tokenize.py

clean:
	rm -f train train_large train_large_ane $(PROBES) test_rmsnorm_bwd test_classifier test_data_validation

.PHONY: clean tokenize probes security-tests
>>>>>>> 60b0512be32bf88f096481ac105f5558c0afa631
=======
	rm -f train train_large train_large_ane train_opt train_double_buffer $(PROBES) test_rmsnorm_bwd test_classifier

.PHONY: clean tokenize probes verify-flags data setup
>>>>>>> be96079bbf2511a7b4a5569a026cf03883ebfbaf

=======
.PHONY: clean tokenize probes verify-flags check-deprecated
>>>>>>> 1a7a05708bf5e34349150cb6e0805c16027b9fd7
