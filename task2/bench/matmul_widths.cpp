// matmul_widths.cpp  SIMD width study.  NOT part of the submission.
//
// Each kernel carries a GCC/Clang `target` attribute, so all three widths compile into
// one binary even though the build line only says -mavx2 -mfma. Nothing here runs
// unless the bench has checked the CPU supports it first (see cpu_has_avx512f() etc).
//
// All three are the SAME 4x4 register-tiled dot-product micro-kernel as
// src/matmul_optimized.cpp, un-blocked (no M/N tiling, no prefetch) so the comparison
// isolates vector width and nothing else. Only the width (4 / 8 / 16 floats per
// instruction) and the corresponding register/intrinsic types differ.

#include <immintrin.h>

#include "matmul_widths.h"

bool cpu_has_avx512f() { return __builtin_cpu_supports("avx512f"); }
bool cpu_has_avx2()    { return __builtin_cpu_supports("avx2"); }
bool cpu_has_fma()     { return __builtin_cpu_supports("fma"); }
bool cpu_has_sse42()   { return __builtin_cpu_supports("sse4.2"); }

namespace {

// Plain scalar dot product, used only for the M%4 / N%4 edge tiles (a vanishing
// fraction of the work for any M, N worth this comparison). No target attribute
// needed, so it's safe to call from any of the three width-specific functions below.
inline float dot_scalar(const float* a, const float* b, int K) {
    float acc = 0.0f;
    for (int p = 0; p < K; ++p) acc += a[p] * b[p];
    return acc;
}

inline void edge_tile(const float* A, const float* B, float* C, int i0, int j0, int mr,
                      int nr, int K, int lda, int ldb, int ldc) {
    for (int ii = 0; ii < mr; ++ii)
        for (int jj = 0; jj < nr; ++jj)
            C[static_cast<long>(i0 + ii) * ldc + (j0 + jj)] =
                dot_scalar(A + static_cast<long>(i0 + ii) * lda,
                          B + static_cast<long>(j0 + jj) * ldb, K);
}

}  // namespace

// ---------------------------------------------------------------- 128-bit (SSE4.2)
__attribute__((target("sse4.2")))
void matmul_simd128(const float* A, const float* B, float* C, int M, int N, int K, int lda,
                    int ldb, int ldc) {
    constexpr int MR = 4, NR = 4, LANES = 4;
    int i = 0;
    for (; i + MR <= M; i += MR) {
        int j = 0;
        for (; j + NR <= N; j += NR) {
            __m128 c00 = _mm_setzero_ps(), c01 = _mm_setzero_ps();
            __m128 c02 = _mm_setzero_ps(), c03 = _mm_setzero_ps();
            __m128 c10 = _mm_setzero_ps(), c11 = _mm_setzero_ps();
            __m128 c12 = _mm_setzero_ps(), c13 = _mm_setzero_ps();
            __m128 c20 = _mm_setzero_ps(), c21 = _mm_setzero_ps();
            __m128 c22 = _mm_setzero_ps(), c23 = _mm_setzero_ps();
            __m128 c30 = _mm_setzero_ps(), c31 = _mm_setzero_ps();
            __m128 c32 = _mm_setzero_ps(), c33 = _mm_setzero_ps();

            const float* a0 = A + static_cast<long>(i + 0) * lda;
            const float* a1 = A + static_cast<long>(i + 1) * lda;
            const float* a2 = A + static_cast<long>(i + 2) * lda;
            const float* a3 = A + static_cast<long>(i + 3) * lda;
            const float* b0 = B + static_cast<long>(j + 0) * ldb;
            const float* b1 = B + static_cast<long>(j + 1) * ldb;
            const float* b2 = B + static_cast<long>(j + 2) * ldb;
            const float* b3 = B + static_cast<long>(j + 3) * ldb;

            int p = 0;
            for (; p + LANES <= K; p += LANES) {
                const __m128 av0 = _mm_loadu_ps(a0 + p), av1 = _mm_loadu_ps(a1 + p);
                const __m128 av2 = _mm_loadu_ps(a2 + p), av3 = _mm_loadu_ps(a3 + p);
                const __m128 bv0 = _mm_loadu_ps(b0 + p), bv1 = _mm_loadu_ps(b1 + p);
                const __m128 bv2 = _mm_loadu_ps(b2 + p), bv3 = _mm_loadu_ps(b3 + p);
                c00 = _mm_add_ps(c00, _mm_mul_ps(av0, bv0));
                c01 = _mm_add_ps(c01, _mm_mul_ps(av0, bv1));
                c02 = _mm_add_ps(c02, _mm_mul_ps(av0, bv2));
                c03 = _mm_add_ps(c03, _mm_mul_ps(av0, bv3));
                c10 = _mm_add_ps(c10, _mm_mul_ps(av1, bv0));
                c11 = _mm_add_ps(c11, _mm_mul_ps(av1, bv1));
                c12 = _mm_add_ps(c12, _mm_mul_ps(av1, bv2));
                c13 = _mm_add_ps(c13, _mm_mul_ps(av1, bv3));
                c20 = _mm_add_ps(c20, _mm_mul_ps(av2, bv0));
                c21 = _mm_add_ps(c21, _mm_mul_ps(av2, bv1));
                c22 = _mm_add_ps(c22, _mm_mul_ps(av2, bv2));
                c23 = _mm_add_ps(c23, _mm_mul_ps(av2, bv3));
                c30 = _mm_add_ps(c30, _mm_mul_ps(av3, bv0));
                c31 = _mm_add_ps(c31, _mm_mul_ps(av3, bv1));
                c32 = _mm_add_ps(c32, _mm_mul_ps(av3, bv2));
                c33 = _mm_add_ps(c33, _mm_mul_ps(av3, bv3));
            }

            float t00 = 0, t01 = 0, t02 = 0, t03 = 0, t10 = 0, t11 = 0, t12 = 0, t13 = 0;
            float t20 = 0, t21 = 0, t22 = 0, t23 = 0, t30 = 0, t31 = 0, t32 = 0, t33 = 0;
            for (; p < K; ++p) {
                const float x0 = a0[p], x1 = a1[p], x2 = a2[p], x3 = a3[p];
                const float y0 = b0[p], y1 = b1[p], y2 = b2[p], y3 = b3[p];
                t00 += x0 * y0; t01 += x0 * y1; t02 += x0 * y2; t03 += x0 * y3;
                t10 += x1 * y0; t11 += x1 * y1; t12 += x1 * y2; t13 += x1 * y3;
                t20 += x2 * y0; t21 += x2 * y1; t22 += x2 * y2; t23 += x2 * y3;
                t30 += x3 * y0; t31 += x3 * y1; t32 += x3 * y2; t33 += x3 * y3;
            }

            alignas(16) float buf[4];
            auto hsum = [&](__m128 v) {
                _mm_storeu_ps(buf, v);
                return buf[0] + buf[1] + buf[2] + buf[3];
            };
            float* c0 = C + static_cast<long>(i + 0) * ldc + j;
            float* c1 = C + static_cast<long>(i + 1) * ldc + j;
            float* c2 = C + static_cast<long>(i + 2) * ldc + j;
            float* c3 = C + static_cast<long>(i + 3) * ldc + j;
            c0[0] = hsum(c00) + t00; c0[1] = hsum(c01) + t01;
            c0[2] = hsum(c02) + t02; c0[3] = hsum(c03) + t03;
            c1[0] = hsum(c10) + t10; c1[1] = hsum(c11) + t11;
            c1[2] = hsum(c12) + t12; c1[3] = hsum(c13) + t13;
            c2[0] = hsum(c20) + t20; c2[1] = hsum(c21) + t21;
            c2[2] = hsum(c22) + t22; c2[3] = hsum(c23) + t23;
            c3[0] = hsum(c30) + t30; c3[1] = hsum(c31) + t31;
            c3[2] = hsum(c32) + t32; c3[3] = hsum(c33) + t33;
        }
        if (j < N) edge_tile(A, B, C, i, j, MR, N - j, K, lda, ldb, ldc);
    }
    if (i < M) edge_tile(A, B, C, i, 0, M - i, N, K, lda, ldb, ldc);
}

// ---------------------------------------------------------------- 256-bit (AVX2)
__attribute__((target("avx2,fma")))
void matmul_simd256(const float* A, const float* B, float* C, int M, int N, int K, int lda,
                    int ldb, int ldc) {
    constexpr int MR = 4, NR = 4, LANES = 8;
    int i = 0;
    for (; i + MR <= M; i += MR) {
        int j = 0;
        for (; j + NR <= N; j += NR) {
            __m256 c00 = _mm256_setzero_ps(), c01 = _mm256_setzero_ps();
            __m256 c02 = _mm256_setzero_ps(), c03 = _mm256_setzero_ps();
            __m256 c10 = _mm256_setzero_ps(), c11 = _mm256_setzero_ps();
            __m256 c12 = _mm256_setzero_ps(), c13 = _mm256_setzero_ps();
            __m256 c20 = _mm256_setzero_ps(), c21 = _mm256_setzero_ps();
            __m256 c22 = _mm256_setzero_ps(), c23 = _mm256_setzero_ps();
            __m256 c30 = _mm256_setzero_ps(), c31 = _mm256_setzero_ps();
            __m256 c32 = _mm256_setzero_ps(), c33 = _mm256_setzero_ps();

            const float* a0 = A + static_cast<long>(i + 0) * lda;
            const float* a1 = A + static_cast<long>(i + 1) * lda;
            const float* a2 = A + static_cast<long>(i + 2) * lda;
            const float* a3 = A + static_cast<long>(i + 3) * lda;
            const float* b0 = B + static_cast<long>(j + 0) * ldb;
            const float* b1 = B + static_cast<long>(j + 1) * ldb;
            const float* b2 = B + static_cast<long>(j + 2) * ldb;
            const float* b3 = B + static_cast<long>(j + 3) * ldb;

            int p = 0;
            for (; p + LANES <= K; p += LANES) {
                const __m256 av0 = _mm256_loadu_ps(a0 + p), av1 = _mm256_loadu_ps(a1 + p);
                const __m256 av2 = _mm256_loadu_ps(a2 + p), av3 = _mm256_loadu_ps(a3 + p);
                const __m256 bv0 = _mm256_loadu_ps(b0 + p), bv1 = _mm256_loadu_ps(b1 + p);
                const __m256 bv2 = _mm256_loadu_ps(b2 + p), bv3 = _mm256_loadu_ps(b3 + p);
                c00 = _mm256_fmadd_ps(av0, bv0, c00); c01 = _mm256_fmadd_ps(av0, bv1, c01);
                c02 = _mm256_fmadd_ps(av0, bv2, c02); c03 = _mm256_fmadd_ps(av0, bv3, c03);
                c10 = _mm256_fmadd_ps(av1, bv0, c10); c11 = _mm256_fmadd_ps(av1, bv1, c11);
                c12 = _mm256_fmadd_ps(av1, bv2, c12); c13 = _mm256_fmadd_ps(av1, bv3, c13);
                c20 = _mm256_fmadd_ps(av2, bv0, c20); c21 = _mm256_fmadd_ps(av2, bv1, c21);
                c22 = _mm256_fmadd_ps(av2, bv2, c22); c23 = _mm256_fmadd_ps(av2, bv3, c23);
                c30 = _mm256_fmadd_ps(av3, bv0, c30); c31 = _mm256_fmadd_ps(av3, bv1, c31);
                c32 = _mm256_fmadd_ps(av3, bv2, c32); c33 = _mm256_fmadd_ps(av3, bv3, c33);
            }

            float t00 = 0, t01 = 0, t02 = 0, t03 = 0, t10 = 0, t11 = 0, t12 = 0, t13 = 0;
            float t20 = 0, t21 = 0, t22 = 0, t23 = 0, t30 = 0, t31 = 0, t32 = 0, t33 = 0;
            for (; p < K; ++p) {
                const float x0 = a0[p], x1 = a1[p], x2 = a2[p], x3 = a3[p];
                const float y0 = b0[p], y1 = b1[p], y2 = b2[p], y3 = b3[p];
                t00 += x0 * y0; t01 += x0 * y1; t02 += x0 * y2; t03 += x0 * y3;
                t10 += x1 * y0; t11 += x1 * y1; t12 += x1 * y2; t13 += x1 * y3;
                t20 += x2 * y0; t21 += x2 * y1; t22 += x2 * y2; t23 += x2 * y3;
                t30 += x3 * y0; t31 += x3 * y1; t32 += x3 * y2; t33 += x3 * y3;
            }

            auto hsum = [](__m256 v) {
                __m128 lo = _mm256_castps256_ps128(v);
                __m128 hi = _mm256_extractf128_ps(v, 1);
                lo = _mm_add_ps(lo, hi);
                __m128 shuf = _mm_movehdup_ps(lo);
                __m128 sums = _mm_add_ps(lo, shuf);
                shuf = _mm_movehl_ps(shuf, sums);
                sums = _mm_add_ss(sums, shuf);
                return _mm_cvtss_f32(sums);
            };
            float* c0 = C + static_cast<long>(i + 0) * ldc + j;
            float* c1 = C + static_cast<long>(i + 1) * ldc + j;
            float* c2 = C + static_cast<long>(i + 2) * ldc + j;
            float* c3 = C + static_cast<long>(i + 3) * ldc + j;
            c0[0] = hsum(c00) + t00; c0[1] = hsum(c01) + t01;
            c0[2] = hsum(c02) + t02; c0[3] = hsum(c03) + t03;
            c1[0] = hsum(c10) + t10; c1[1] = hsum(c11) + t11;
            c1[2] = hsum(c12) + t12; c1[3] = hsum(c13) + t13;
            c2[0] = hsum(c20) + t20; c2[1] = hsum(c21) + t21;
            c2[2] = hsum(c22) + t22; c2[3] = hsum(c23) + t23;
            c3[0] = hsum(c30) + t30; c3[1] = hsum(c31) + t31;
            c3[2] = hsum(c32) + t32; c3[3] = hsum(c33) + t33;
        }
        if (j < N) edge_tile(A, B, C, i, j, MR, N - j, K, lda, ldb, ldc);
    }
    if (i < M) edge_tile(A, B, C, i, 0, M - i, N, K, lda, ldb, ldc);
}

// ---------------------------------------------------------------- 512-bit (AVX-512F)
namespace {
__attribute__((target("avx512f")))
inline float hsum512(__m512 v) { return _mm512_reduce_add_ps(v); }
}  // namespace

__attribute__((target("avx512f")))
void matmul_simd512(const float* A, const float* B, float* C, int M, int N, int K, int lda,
                    int ldb, int ldc) {
    constexpr int MR = 4, NR = 4, LANES = 16;
    int i = 0;
    for (; i + MR <= M; i += MR) {
        int j = 0;
        for (; j + NR <= N; j += NR) {
            __m512 c00 = _mm512_setzero_ps(), c01 = _mm512_setzero_ps();
            __m512 c02 = _mm512_setzero_ps(), c03 = _mm512_setzero_ps();
            __m512 c10 = _mm512_setzero_ps(), c11 = _mm512_setzero_ps();
            __m512 c12 = _mm512_setzero_ps(), c13 = _mm512_setzero_ps();
            __m512 c20 = _mm512_setzero_ps(), c21 = _mm512_setzero_ps();
            __m512 c22 = _mm512_setzero_ps(), c23 = _mm512_setzero_ps();
            __m512 c30 = _mm512_setzero_ps(), c31 = _mm512_setzero_ps();
            __m512 c32 = _mm512_setzero_ps(), c33 = _mm512_setzero_ps();

            const float* a0 = A + static_cast<long>(i + 0) * lda;
            const float* a1 = A + static_cast<long>(i + 1) * lda;
            const float* a2 = A + static_cast<long>(i + 2) * lda;
            const float* a3 = A + static_cast<long>(i + 3) * lda;
            const float* b0 = B + static_cast<long>(j + 0) * ldb;
            const float* b1 = B + static_cast<long>(j + 1) * ldb;
            const float* b2 = B + static_cast<long>(j + 2) * ldb;
            const float* b3 = B + static_cast<long>(j + 3) * ldb;

            int p = 0;
            for (; p + LANES <= K; p += LANES) {
                const __m512 av0 = _mm512_loadu_ps(a0 + p), av1 = _mm512_loadu_ps(a1 + p);
                const __m512 av2 = _mm512_loadu_ps(a2 + p), av3 = _mm512_loadu_ps(a3 + p);
                const __m512 bv0 = _mm512_loadu_ps(b0 + p), bv1 = _mm512_loadu_ps(b1 + p);
                const __m512 bv2 = _mm512_loadu_ps(b2 + p), bv3 = _mm512_loadu_ps(b3 + p);
                c00 = _mm512_fmadd_ps(av0, bv0, c00); c01 = _mm512_fmadd_ps(av0, bv1, c01);
                c02 = _mm512_fmadd_ps(av0, bv2, c02); c03 = _mm512_fmadd_ps(av0, bv3, c03);
                c10 = _mm512_fmadd_ps(av1, bv0, c10); c11 = _mm512_fmadd_ps(av1, bv1, c11);
                c12 = _mm512_fmadd_ps(av1, bv2, c12); c13 = _mm512_fmadd_ps(av1, bv3, c13);
                c20 = _mm512_fmadd_ps(av2, bv0, c20); c21 = _mm512_fmadd_ps(av2, bv1, c21);
                c22 = _mm512_fmadd_ps(av2, bv2, c22); c23 = _mm512_fmadd_ps(av2, bv3, c23);
                c30 = _mm512_fmadd_ps(av3, bv0, c30); c31 = _mm512_fmadd_ps(av3, bv1, c31);
                c32 = _mm512_fmadd_ps(av3, bv2, c32); c33 = _mm512_fmadd_ps(av3, bv3, c33);
            }

            float t00 = 0, t01 = 0, t02 = 0, t03 = 0, t10 = 0, t11 = 0, t12 = 0, t13 = 0;
            float t20 = 0, t21 = 0, t22 = 0, t23 = 0, t30 = 0, t31 = 0, t32 = 0, t33 = 0;
            for (; p < K; ++p) {
                const float x0 = a0[p], x1 = a1[p], x2 = a2[p], x3 = a3[p];
                const float y0 = b0[p], y1 = b1[p], y2 = b2[p], y3 = b3[p];
                t00 += x0 * y0; t01 += x0 * y1; t02 += x0 * y2; t03 += x0 * y3;
                t10 += x1 * y0; t11 += x1 * y1; t12 += x1 * y2; t13 += x1 * y3;
                t20 += x2 * y0; t21 += x2 * y1; t22 += x2 * y2; t23 += x2 * y3;
                t30 += x3 * y0; t31 += x3 * y1; t32 += x3 * y2; t33 += x3 * y3;
            }

            float* c0 = C + static_cast<long>(i + 0) * ldc + j;
            float* c1 = C + static_cast<long>(i + 1) * ldc + j;
            float* c2 = C + static_cast<long>(i + 2) * ldc + j;
            float* c3 = C + static_cast<long>(i + 3) * ldc + j;
            c0[0] = hsum512(c00) + t00; c0[1] = hsum512(c01) + t01;
            c0[2] = hsum512(c02) + t02; c0[3] = hsum512(c03) + t03;
            c1[0] = hsum512(c10) + t10; c1[1] = hsum512(c11) + t11;
            c1[2] = hsum512(c12) + t12; c1[3] = hsum512(c13) + t13;
            c2[0] = hsum512(c20) + t20; c2[1] = hsum512(c21) + t21;
            c2[2] = hsum512(c22) + t22; c2[3] = hsum512(c23) + t23;
            c3[0] = hsum512(c30) + t30; c3[1] = hsum512(c31) + t31;
            c3[2] = hsum512(c32) + t32; c3[3] = hsum512(c33) + t33;
        }
        if (j < N) edge_tile(A, B, C, i, j, MR, N - j, K, lda, ldb, ldc);
    }
    if (i < M) edge_tile(A, B, C, i, 0, M - i, N, K, lda, ldb, ldc);
}
