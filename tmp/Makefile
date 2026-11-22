# tmp/ sandbox — Self-contained MNIST on ANE demo
#
# Quick start:
#   make data   — download MNIST
#   make        — compile mnist_demo
#   make run    — download + compile + run
#   make clean  — remove artifacts
#
# Fully self-contained: no includes from ../training/

CLANG     := xcrun clang
SDKROOT   ?= $(shell xcrun --show-sdk-path)
CFLAGS    := -O2 -fobjc-arc -isysroot $(SDKROOT)
LDFLAGS   := -framework Foundation -framework IOSurface -ldl -lobjc

TARGET    := mnist_demo

.PHONY: all run data clean leakcheck

all: $(TARGET)

$(TARGET): mnist_demo.m
	$(CLANG) $(CFLAGS) $(LDFLAGS) -o $@ $<
	@echo "Built: $(TARGET)"

data:
	@python3 fetch_mnist.py

run: data $(TARGET)
	@echo ""
	./$(TARGET)

leakcheck: $(TARGET)
	leaks --atExit -- ./$(TARGET) 2>&1 | tail -20

clean:
	rm -f $(TARGET)
	rm -rf data/
