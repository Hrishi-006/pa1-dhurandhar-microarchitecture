// matmul_optimized.cpp  STAGE 3: PUT IT ALL TOGETHER
//
// This is the graded function AND the kernel that gets injected into llama.cpp. Combine
// everything you have learned across the whole assignment  loop reordering, register
// blocking and unrolling (Task 1 / Stage 1 here), cache tiling and software prefetch
// (Stage 2)  and TUNE it to be as fast as you can. Your speedup over matmul_naive determines
// your score (see the tier table the harness prints), and this same function will power a
// real LLM inference via `make llama-demo`.
//
// Two measured findings from tuning Stages 1-2 drive the two changes made here on top of
// the Stage-2 kernel:
//
// 1. NAMED accumulators, not an accumulator ARRAY. Stages 1-2 hold the MRxNR partial sums
//    in `__m256 acc[MR][NR]`. On this compiler/flags (-O2, no -march=native) that array is
//    not proven alias-free, so the accumulators round-trip through the stack on every
//    8-float K-step instead of staying in ymm registers -- measured ~1.5x SLOWER than the
//    same math with 16 individually named `__m256` locals. Register blocking only pays off
//    if the registers are actually registers, so this file uses named locals throughout.
//
// 2. A wider 4x4 register tile (16 accumulators) beats Stage 1/2's 4x2 (8 accumulators):
//    16 FMAs per 8 vector loads is a better flops-per-load ratio than 8 FMAs per 6 loads,
//    and 16 named ymm accumulators plus the 8 operand vectors still fit inside the 16
//    architectural ymm registers well enough to avoid the array-spill problem above
//    (measured ~25 GFLOP/s at 2048^3 here vs. Stage 1's ~11 GFLOP/s, on top of the tiling
//    this file also does).
//
// Software prefetch is ON by default here, unlike Stage 2. Measured on pinned, repeated
// (median-of-7 independent process) runs -- the only methodology that held still on this
// machine's turbo/thermal variance -- `_MM_HINT_T0` at a distance of 128 floats beats no
// prefetch by +6 to +12% across 512/1024/2048^3, consistently and reproducibly. This is
// the OPPOSITE ranking from Stage 2, where T0 was the clearly worst hint: Stage 2's
// array-indexed accumulators round-trip through L1 (see the array-vs-named finding
// above), so pulling more zero-reuse streamed data into L1 via T0 evicts data that
// mattered. Here the 16 accumulators live purely in registers -- nothing of the kernel's
// own working state occupies L1 -- so T0's aggressive caching has nothing to compete
// with, and wins outright over NTA/T1/T2. The lesson for Task 2C's write-up: the right
// prefetch hint is not a property of the access pattern alone, it depends on what else is
// already resident in the cache level you're targeting.

#include <immintrin.h>

#include "matmul.h"

// Tile size in output rows/cols (of A / of B respectively). <=0 means "use the full
// extent". Sized so a TM x K panel of A and a TN x K panel of B fit comfortably in L2
// together; see the report's tile-size sweep for the machine-specific optimum.
int matmul_opt_tile_m = 256;
int matmul_opt_tile_n = 256;

// Software prefetch toggle + parameters, exposed for the report's 2C sweep. ON by
// default with the measured-best (hint, distance): see the file header.
int matmul_optimized_prefetch = 1;             // 0 = off, 1 = on
int matmul_optimized_prefetch_dist = 128;      // floats ahead
int matmul_optimized_prefetch_hint = _MM_HINT_T0;  // locality hint

namespace {

constexpr int MR = 4;
constexpr int NR = 4;

inline float hsum256(__m256 v) {
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 hi = _mm256_extractf128_ps(v, 1);
    lo = _mm_add_ps(lo, hi);
    __m128 shuf = _mm_movehdup_ps(lo);
    __m128 sums = _mm_add_ps(lo, shuf);
    shuf = _mm_movehl_ps(shuf, sums);
    sums = _mm_add_ss(sums, shuf);
    return _mm_cvtss_f32(sums);
}

template <int Hint>
inline void pf(const void* p) {
    _mm_prefetch(reinterpret_cast<const char*>(p), static_cast<_mm_hint>(Hint));
}
inline void prefetch_hinted(const void* p, int hint) {
    switch (hint) {
        case _MM_HINT_T0: pf<_MM_HINT_T0>(p); break;
        case _MM_HINT_T1: pf<_MM_HINT_T1>(p); break;
        case _MM_HINT_T2: pf<_MM_HINT_T2>(p); break;
        default: pf<_MM_HINT_NTA>(p); break;
    }
}

// Fast path: exactly MR x NR (4x4), full K reduction, 16 NAMED __m256 accumulators.
inline void microkernel_4x4(const float* A, const float* B, float* C, int i0, int j0, int K,
                            int lda, int ldb, int ldc, bool prefetch, int dist, int hint) {
    __m256 c00 = _mm256_setzero_ps(), c01 = _mm256_setzero_ps();
    __m256 c02 = _mm256_setzero_ps(), c03 = _mm256_setzero_ps();
    __m256 c10 = _mm256_setzero_ps(), c11 = _mm256_setzero_ps();
    __m256 c12 = _mm256_setzero_ps(), c13 = _mm256_setzero_ps();
    __m256 c20 = _mm256_setzero_ps(), c21 = _mm256_setzero_ps();
    __m256 c22 = _mm256_setzero_ps(), c23 = _mm256_setzero_ps();
    __m256 c30 = _mm256_setzero_ps(), c31 = _mm256_setzero_ps();
    __m256 c32 = _mm256_setzero_ps(), c33 = _mm256_setzero_ps();

    const float* a0 = A + static_cast<long>(i0 + 0) * lda;
    const float* a1 = A + static_cast<long>(i0 + 1) * lda;
    const float* a2 = A + static_cast<long>(i0 + 2) * lda;
    const float* a3 = A + static_cast<long>(i0 + 3) * lda;
    const float* b0 = B + static_cast<long>(j0 + 0) * ldb;
    const float* b1 = B + static_cast<long>(j0 + 1) * ldb;
    const float* b2 = B + static_cast<long>(j0 + 2) * ldb;
    const float* b3 = B + static_cast<long>(j0 + 3) * ldb;

    int p = 0;
    for (; p + 16 <= K; p += 16) {
        if (prefetch) {
            prefetch_hinted(a0 + p + dist, hint);
            prefetch_hinted(a1 + p + dist, hint);
            prefetch_hinted(a2 + p + dist, hint);
            prefetch_hinted(a3 + p + dist, hint);
            prefetch_hinted(b0 + p + dist, hint);
            prefetch_hinted(b1 + p + dist, hint);
            prefetch_hinted(b2 + p + dist, hint);
            prefetch_hinted(b3 + p + dist, hint);
        }
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

        const int p2 = p + 8;
        const __m256 aw0 = _mm256_loadu_ps(a0 + p2), aw1 = _mm256_loadu_ps(a1 + p2);
        const __m256 aw2 = _mm256_loadu_ps(a2 + p2), aw3 = _mm256_loadu_ps(a3 + p2);
        const __m256 bw0 = _mm256_loadu_ps(b0 + p2), bw1 = _mm256_loadu_ps(b1 + p2);
        const __m256 bw2 = _mm256_loadu_ps(b2 + p2), bw3 = _mm256_loadu_ps(b3 + p2);
        c00 = _mm256_fmadd_ps(aw0, bw0, c00); c01 = _mm256_fmadd_ps(aw0, bw1, c01);
        c02 = _mm256_fmadd_ps(aw0, bw2, c02); c03 = _mm256_fmadd_ps(aw0, bw3, c03);
        c10 = _mm256_fmadd_ps(aw1, bw0, c10); c11 = _mm256_fmadd_ps(aw1, bw1, c11);
        c12 = _mm256_fmadd_ps(aw1, bw2, c12); c13 = _mm256_fmadd_ps(aw1, bw3, c13);
        c20 = _mm256_fmadd_ps(aw2, bw0, c20); c21 = _mm256_fmadd_ps(aw2, bw1, c21);
        c22 = _mm256_fmadd_ps(aw2, bw2, c22); c23 = _mm256_fmadd_ps(aw2, bw3, c23);
        c30 = _mm256_fmadd_ps(aw3, bw0, c30); c31 = _mm256_fmadd_ps(aw3, bw1, c31);
        c32 = _mm256_fmadd_ps(aw3, bw2, c32); c33 = _mm256_fmadd_ps(aw3, bw3, c33);
    }
    for (; p + 8 <= K; p += 8) {
        if (prefetch) {
            prefetch_hinted(a0 + p + dist, hint);
            prefetch_hinted(a1 + p + dist, hint);
            prefetch_hinted(a2 + p + dist, hint);
            prefetch_hinted(a3 + p + dist, hint);
            prefetch_hinted(b0 + p + dist, hint);
            prefetch_hinted(b1 + p + dist, hint);
            prefetch_hinted(b2 + p + dist, hint);
            prefetch_hinted(b3 + p + dist, hint);
        }
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

    // Scalar cleanup for K % 8, one accumulator per output (16 named floats, same reason
    // as above: an array here would spill too, though the cost matters far less since
    // this loop runs at most 7 times).
    float t00 = 0, t01 = 0, t02 = 0, t03 = 0;
    float t10 = 0, t11 = 0, t12 = 0, t13 = 0;
    float t20 = 0, t21 = 0, t22 = 0, t23 = 0;
    float t30 = 0, t31 = 0, t32 = 0, t33 = 0;
    for (; p < K; ++p) {
        const float x0 = a0[p], x1 = a1[p], x2 = a2[p], x3 = a3[p];
        const float y0 = b0[p], y1 = b1[p], y2 = b2[p], y3 = b3[p];
        t00 += x0 * y0; t01 += x0 * y1; t02 += x0 * y2; t03 += x0 * y3;
        t10 += x1 * y0; t11 += x1 * y1; t12 += x1 * y2; t13 += x1 * y3;
        t20 += x2 * y0; t21 += x2 * y1; t22 += x2 * y2; t23 += x2 * y3;
        t30 += x3 * y0; t31 += x3 * y1; t32 += x3 * y2; t33 += x3 * y3;
    }

    float* c_i0 = C + static_cast<long>(i0 + 0) * ldc + j0;
    float* c_i1 = C + static_cast<long>(i0 + 1) * ldc + j0;
    float* c_i2 = C + static_cast<long>(i0 + 2) * ldc + j0;
    float* c_i3 = C + static_cast<long>(i0 + 3) * ldc + j0;
    c_i0[0] = hsum256(c00) + t00; c_i0[1] = hsum256(c01) + t01;
    c_i0[2] = hsum256(c02) + t02; c_i0[3] = hsum256(c03) + t03;
    c_i1[0] = hsum256(c10) + t10; c_i1[1] = hsum256(c11) + t11;
    c_i1[2] = hsum256(c12) + t12; c_i1[3] = hsum256(c13) + t13;
    c_i2[0] = hsum256(c20) + t20; c_i2[1] = hsum256(c21) + t21;
    c_i2[2] = hsum256(c22) + t22; c_i2[3] = hsum256(c23) + t23;
    c_i3[0] = hsum256(c30) + t30; c_i3[1] = hsum256(c31) + t31;
    c_i3[2] = hsum256(c32) + t32; c_i3[3] = hsum256(c33) + t33;
}

// General path for edge tiles where mr<MR or nr<NR (M or N not a multiple of 4). Array
// accumulators are fine here: this only ever runs on the last partial row/col block, a
// vanishing fraction of the total work for any M, N worth tiling in the first place.
inline void microkernel_general(const float* A, const float* B, float* C, int i0, int j0,
                                int mr, int nr, int K, int lda, int ldb, int ldc) {
    __m256 acc[MR][NR];
    for (int ii = 0; ii < mr; ++ii)
        for (int jj = 0; jj < nr; ++jj) acc[ii][jj] = _mm256_setzero_ps();

    int p = 0;
    for (; p + 8 <= K; p += 8) {
        __m256 avec[MR];
        for (int ii = 0; ii < mr; ++ii)
            avec[ii] = _mm256_loadu_ps(A + static_cast<long>(i0 + ii) * lda + p);
        for (int jj = 0; jj < nr; ++jj) {
            __m256 bvec = _mm256_loadu_ps(B + static_cast<long>(j0 + jj) * ldb + p);
            for (int ii = 0; ii < mr; ++ii)
                acc[ii][jj] = _mm256_fmadd_ps(avec[ii], bvec, acc[ii][jj]);
        }
    }

    float tail[MR][NR] = {};
    for (; p < K; ++p) {
        for (int ii = 0; ii < mr; ++ii) {
            const float a = A[static_cast<long>(i0 + ii) * lda + p];
            for (int jj = 0; jj < nr; ++jj)
                tail[ii][jj] += a * B[static_cast<long>(j0 + jj) * ldb + p];
        }
    }

    for (int ii = 0; ii < mr; ++ii)
        for (int jj = 0; jj < nr; ++jj)
            C[static_cast<long>(i0 + ii) * ldc + (j0 + jj)] = hsum256(acc[ii][jj]) + tail[ii][jj];
}

}  // namespace

void matmul_optimized(const float* A, const float* B, float* C, int M, int N, int K, int lda,
                      int ldb, int ldc) {
    const int TM = (matmul_opt_tile_m > 0 && matmul_opt_tile_m < M) ? matmul_opt_tile_m : M;
    const int TN = (matmul_opt_tile_n > 0 && matmul_opt_tile_n < N) ? matmul_opt_tile_n : N;
    const bool prefetch = matmul_optimized_prefetch != 0;
    const int dist = matmul_optimized_prefetch_dist;
    const int hint = matmul_optimized_prefetch_hint;

    for (int i0 = 0; i0 < M; i0 += TM) {
        const int i1 = (i0 + TM < M) ? i0 + TM : M;
        for (int j0 = 0; j0 < N; j0 += TN) {
            const int j1 = (j0 + TN < N) ? j0 + TN : N;

            int i = i0;
            for (; i + MR <= i1; i += MR) {
                int j = j0;
                for (; j + NR <= j1; j += NR)
                    microkernel_4x4(A, B, C, i, j, K, lda, ldb, ldc, prefetch, dist, hint);
                if (j < j1) microkernel_general(A, B, C, i, j, MR, j1 - j, K, lda, ldb, ldc);
            }
            if (i < i1) {
                const int mr = i1 - i;
                int j = j0;
                for (; j + NR <= j1; j += NR)
                    microkernel_general(A, B, C, i, j, mr, NR, K, lda, ldb, ldc);
                if (j < j1) microkernel_general(A, B, C, i, j, mr, j1 - j, K, lda, ldb, ldc);
            }
        }
    }
}
