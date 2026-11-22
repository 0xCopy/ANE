<<<<<<< HEAD
CC = xcrun clang
CFLAGS = -O2 -Wall -DACCELERATE_NEW_LAPACK -fobjc-arc
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

train_opt: train_opt.m $(HEADERS_OPT)
	$(CC) $(CFLAGS) -o $@ train_opt.m $(LDFLAGS) -framework Accelerate -framework Metal -framework MetalPerformanceShaders

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

probes: $(PROBES)

tokenize:
	python3 tokenize.py

clean:
	rm -f train train_large train_large_ane train_opt $(PROBES) test_rmsnorm_bwd test_classifier

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

