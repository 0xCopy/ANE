# tmp/ sandbox — Self-contained MNIST-on-ANE demo
#
# Quick start:
#   python3 fetch_data.py
#   make
#   ./mnist_demo
#
# Optional Swift variant (same network, no ObjC):
#   make swift

SDKROOT ?= $(shell xcrun --show-sdk-path)
CC      := xcrun clang
SWIFTC  := xcrun swiftc
CFLAGS  := -O2 -fobjc-arc -isysroot $(SDKROOT)
LIBS    := -framework Foundation -framework IOSurface -ldl -lobjc

.PHONY: all clean swift data

all: mnist_demo

mnist_demo: mnist_demo.m
	$(CC) $(CFLAGS) $(LIBS) -o $@ $<
	@echo "Built mnist_demo (ObjC, ARC-safe)"

swift: mnist_demo_swift
mnist_demo_swift: mnist_demo.swift
	$(SWIFTC) -O -import-objc-header bridge.h -o $@ $< $(LIBS)
	@echo "Built mnist_demo_swift"

data:
	python3 fetch_data.py

clean:
	rm -f mnist_demo mnist_demo_swift
	rm -rf data/
