// matmul_widths.h  SIMD width study (128 / 256 / 512 bit).  NOT part of the submission.
//
// Three implementations of the SAME algorithm and the SAME 4x4 register tile as
// src/matmul_optimized.cpp's micro-kernel (un-blocked -- no M/N cache tiling, no
// prefetch), differing only in vector width, so a speedup comparison isolates width
// and nothing else. Handles arbitrary M, N, K via a scalar-cleanup fallback, same as
// the graded kernels.
#ifndef CS683_PA1_MATMUL_WIDTHS_H
#define CS683_PA1_MATMUL_WIDTHS_H

#include "matmul.h"

void matmul_simd128(const float* A, const float* B, float* C, int M, int N, int K, int lda,
                    int ldb, int ldc);
void matmul_simd256(const float* A, const float* B, float* C, int M, int N, int K, int lda,
                    int ldb, int ldc);
void matmul_simd512(const float* A, const float* B, float* C, int M, int N, int K, int lda,
                    int ldb, int ldc);

// Runtime CPU checks: AVX-512 is absent on most client parts (including any hybrid
// P/E-core Intel design, which drops AVX-512 entirely), so the bench must ask before
// calling matmul_simd512 rather than crashing with SIGILL.
bool cpu_has_avx512f();
bool cpu_has_avx2();
bool cpu_has_fma();
bool cpu_has_sse42();

#endif  // CS683_PA1_MATMUL_WIDTHS_H
