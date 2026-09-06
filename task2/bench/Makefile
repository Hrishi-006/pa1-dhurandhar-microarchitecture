# Build file for the perf bench drivers.  NOT part of the submission.
# Same pinned flags as the main Makefile so the measured code is identical.
#   make -C bench        ->  bench/bench_matmul, bench/bench_widths

CXX      := g++
CXXFLAGS := -std=c++17 -O2 -fno-tree-vectorize -mavx2 -mfma -I../include -Wall -Wextra
SRC      := bench_matmul.cpp ../src/matmul_naive.cpp ../src/matmul_simd.cpp \
            ../src/matmul_prefetch.cpp ../src/matmul_optimized.cpp
WIDTHS_SRC := bench_widths.cpp matmul_widths.cpp ../src/matmul_naive.cpp

.PHONY: all clean
all: bench_matmul bench_widths
bench_matmul: $(SRC)
	$(CXX) $(CXXFLAGS) $^ -o $@
bench_widths: $(WIDTHS_SRC)
	$(CXX) $(CXXFLAGS) $^ -o $@
clean:
	rm -f bench_matmul bench_widths *.csv
