CLANG  = xcrun clang
CFLAGS = -O2 -fobjc-arc -I../../../training
ARFLAGS = rcs

BUILD_DIR = build

.PHONY: all lib clean

all: lib

lib: $(BUILD_DIR)/libane_mnist.a

$(BUILD_DIR)/libane_mnist.a: ANEMnist.m ANEMnist.h
	@mkdir -p $(BUILD_DIR)
	$(CLANG) $(CFLAGS) -c ANEMnist.m -o $(BUILD_DIR)/ANEMnist.o
	ar $(ARFLAGS) $(BUILD_DIR)/libane_mnist.a $(BUILD_DIR)/ANEMnist.o
	@echo "  Built $(BUILD_DIR)/libane_mnist.a"

clean:
	rm -rf $(BUILD_DIR)
