# CS683 PA-1: Hardware-Conscious Performance Engineering

This assignment consists of two parts. Both start from a correct-but-naive C++ kernel and ask you
to improve its performance by *thinking about the hardware*: the cache hierarchy, instruction-level
parallelism, and the SIMD units of a modern x86 core.

See the assignment [document](https://docs.google.com/document/d/1suURtM3WaensRvABVxlHZmhbbXe-g2v-sNi8gUfuojg/edit?usp=sharing)
for the full problem statement.

## Tasks

| Task | Workload | Techniques | Report |
|------|----------|------------|--------|
| 1 | 2D convolution | Loop reordering, loop unrolling, cache tiling, SIMD (AVX2) | [task1/task1_report.pdf](task1/task1_report.pdf) |
| 2 | Matrix multiplication (SGEMM), injected into llama.cpp | SIMD (AVX2), cache blocking + software prefetching, combining everything | [task2/task2_report.pdf](task2/task2_report.pdf) |

## Layout

```
task1/src/   conv_reorder.cpp, conv_unroll.cpp, conv_tile.cpp, conv_simd.cpp, conv_optimized.cpp
task2/src/   matmul_simd.cpp, matmul_prefetch.cpp, matmul_optimized.cpp
plots/       supporting figures referenced by the reports
```

Each report documents the motivation, implementation details, and speedup analysis for its task.
