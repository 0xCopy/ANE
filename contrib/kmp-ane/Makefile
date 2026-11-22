# Makefile — Top-level build for kmp-ane MNIST trainer
#
# KMP build (Kotlin → cinterop → ObjC bridge → ANE):
#   make bridge       — compile ObjC bridge into build/libanebridge.a
#   make data         — download MNIST dataset via Python
#   make kmp          — build Kotlin/Native MNIST binary
#   make run          — run the trainer
#   make all          — bridge + data + kmp
#   make clean        — remove build artifacts
#
# Stack: Kotlin (KotlinGrad AD) → cinterop → ANEBridge.m (ObjC) → ANE private API

SDKROOT     ?= $(shell xcrun --show-sdk-path)
CC          := xcrun clang
AR          := xcrun ar
CFLAGS      := -O2 -fobjc-arc -isysroot $(SDKROOT)
OBJC_FLAGS  := -x objective-c

BUILD_DIR   := build

.PHONY: all clean bridge data kmp kmp-kg run run-kg test

all: bridge data kmp

# ── ObjC bridge static library ──────────────────────────────────────────────
BRIDGE_LIB  := $(BUILD_DIR)/libanebridge.a
BRIDGE_SRCS := objc-bridge/ANEBridge.m
BRIDGE_OBJS := $(BUILD_DIR)/ANEBridge.o

bridge: $(BRIDGE_LIB)

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

$(BUILD_DIR)/ANEBridge.o: objc-bridge/ANEBridge.m objc-bridge/ANEBridge.h | $(BUILD_DIR)
	$(CC) $(CFLAGS) $(OBJC_FLAGS) -framework Foundation -framework IOSurface -c $< -o $@

$(BRIDGE_LIB): $(BRIDGE_OBJS)
	$(AR) rcs $@ $(BRIDGE_OBJS)
	@echo "Built $(BRIDGE_LIB)"

# ── MNIST data ──────────────────────────────────────────────────────────────
data:
	python3 prepare_mnist.py

# ── KMP Kotlin build ────────────────────────────────────────────────────────
KMP_BIN := build/bin/macosArm64/mnistTrainerDebugExecutable/mnistTrainer.kexe
KG_BIN  := build/bin/macosArm64/mnistKotlinGradDebugExecutable/mnistKotlinGrad.kexe

kmp: bridge
	./gradlew linkMnistTrainerDebugExecutableMacosArm64

kmp-kg: bridge
	./gradlew linkMnistKotlinGradDebugExecutableMacosArm64

run: kmp data
	@echo "═══ Running MNIST trainer (Autograd + ANE) ═══"
	$(KMP_BIN)

run-kg: kmp-kg data
	@echo "═══ Running MNIST trainer (KotlinGrad + ANE) ═══"
	$(KG_BIN)

# ── Smoke test (C bridge only) ──────────────────────────────────────────────
test: $(BRIDGE_LIB)
	@echo "=== Smoke test ==="
	$(CC) -O2 -fobjc-arc -isysroot $(SDKROOT) \
		-framework Foundation -framework IOSurface -ldl -lobjc \
		-Iobjc-bridge -L$(BUILD_DIR) -lanebridge \
		-x objective-c -o $(BUILD_DIR)/smoke_test - <<'EOF'
	#import "ANEBridge.h"
	#include <stdio.h>
	int main() {
	    ANEBridgeStatus s = ane_bridge_init();
	    printf("ane_bridge_init() = %d\n", s);
	    printf("available = %d\n", ane_bridge_is_available());
	    char *mil = ane_bridge_mil_gen_conv(4, 8, 2);
	    if (mil) { printf("MIL gen OK (%zu bytes)\n", strlen(mil)); free(mil); }
	    else     { printf("MIL gen returned NULL\n"); }
	    return 0;
	}
	EOF
	$(BUILD_DIR)/smoke_test
	@echo "=== Smoke test passed ==="

clean:
	rm -rf $(BUILD_DIR)
	rm -rf .gradle
