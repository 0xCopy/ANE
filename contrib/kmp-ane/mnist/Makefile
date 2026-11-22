CLANG  = xcrun clang
CFLAGS = -O2 -fobjc-arc -framework Foundation -framework IOSurface -ldl -lobjc

.PHONY: all mnist data run clean

all: mnist

mnist: mnist_ane.m
	$(CLANG) $(CFLAGS) -I../../../training -o mnist_ane mnist_ane.m

data:
	python3 fetch_mnist.py

run: mnist data
	./mnist_ane

clean:
	rm -f mnist_ane
	rm -rf data/
