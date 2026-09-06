# Build file for the perf bench driver.  NOT part of the submission.
# Same pinned flags as the main Makefile so the measured code is identical.
#   make -C bench        ->  bench/bench

CXX      := g++
CXXFLAGS := -std=c++17 -O2 -fno-tree-vectorize -mavx2 -mfma -I../include -Wall -Wextra
SRC      := bench.cpp conv_widths.cpp conv_combo.cpp ../src/conv_naive.cpp \
            ../src/conv_reorder.cpp ../src/conv_unroll.cpp \
            ../src/conv_tile.cpp ../src/conv_simd.cpp \
            ../src/conv_optimized.cpp

.PHONY: all clean
all: bench
bench: $(SRC)
	$(CXX) $(CXXFLAGS) $^ -o $@
clean:
	rm -f bench sweep.csv
