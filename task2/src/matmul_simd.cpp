// matmul_simd.cpp  STAGE 1: SIMD (AVX2) register-tiled micro-kernel
//
// Unlike Task 1's convolution (broadcast-weight * unit-stride-stream), this workload is a
// dot product over K for every (i, j): both A's row i and B's row j are the CONTIGUOUS
// operands, and the reduction is along the same axis for both. That means one FMA chain
// alone only builds ONE scalar output per K-loop, so the loop is latency-bound on the
// horizontal sum unless we widen the *output* it feeds: for a fixed chunk of K, load one
// A vector and reuse it against several B vectors (MRxNR register tile), so each 8-wide
// load does MR*NR work instead of 1.
//
// MR=4, NR=2 keeps the tile's accumulators (MR*NR = 8 ymm registers) plus the MR A-vectors
// and 1 B-vector comfortably inside the 16 architectural ymm registers, leaving headroom for
// the compiler's own bookkeeping -- a 4x4 tile (16 accumulators) has no room left and starts
// spilling on most compilers.
//
// K need not be a multiple of 8: the vector loop handles floor(K/8)*8 elements, and a
// scalar cleanup loop finishes the remainder into a separate accumulator that is added to
// the horizontally-reduced vector sum. M and N need not be multiples of MR/NR either --
// the outer loops fall back to a partial tile (mr<MR or nr<NR) at the bottom/right edge.

#include <immintrin.h>

#include "matmul.h"

namespace {

constexpr int MR = 4;  // rows of A per micro-kernel call
constexpr int NR = 2;  // rows of B per micro-kernel call

// Horizontal sum of one __m256 (8 lanes) down to a scalar.
inline float hsum256(__m256 v) {
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 hi = _mm256_extractf128_ps(v, 1);
    lo = _mm_add_ps(lo, hi);                  // 4 partial sums
    __m128 shuf = _mm_movehdup_ps(lo);        // [1,1,3,3]
    __m128 sums = _mm_add_ps(lo, shuf);       // [0+1, 1+1, 2+3, 3+3]
    shuf = _mm_movehl_ps(shuf, sums);         // high half of sums into low half of shuf
    sums = _mm_add_ss(sums, shuf);
    return _mm_cvtss_f32(sums);
}

// Computes the mr x nr tile C[i0..i0+mr, j0..j0+nr] over the full K reduction.
// mr <= MR, nr <= NR (partial tiles at the M/N edges).
inline void microkernel(const float* A, const float* B, float* C, int i0, int j0, int mr,
                        int nr, int K, int lda, int ldb, int ldc) {
    __m256 acc[MR][NR];
    for (int ii = 0; ii < mr; ++ii)
        for (int jj = 0; jj < nr; ++jj) acc[ii][jj] = _mm256_setzero_ps();

    int p = 0;
    for (; p + 8 <= K; p += 8) {
        __m256 avec[MR];
        for (int ii = 0; ii < mr; ++ii)
            avec[ii] = _mm256_loadu_ps(A + static_cast<long>(i0 + ii) * lda + p);
        // One B load is reused against all mr A-vectors before moving to the next j.
        for (int jj = 0; jj < nr; ++jj) {
            __m256 bvec = _mm256_loadu_ps(B + static_cast<long>(j0 + jj) * ldb + p);
            for (int ii = 0; ii < mr; ++ii)
                acc[ii][jj] = _mm256_fmadd_ps(avec[ii], bvec, acc[ii][jj]);
        }
    }

    // Scalar cleanup for K % 8, accumulated separately then folded in below.
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

void matmul_simd(const float* A, const float* B, float* C, int M, int N, int K, int lda,
                 int ldb, int ldc) {
    int i = 0;
    for (; i + MR <= M; i += MR) {
        int j = 0;
        for (; j + NR <= N; j += NR) microkernel(A, B, C, i, j, MR, NR, K, lda, ldb, ldc);
        if (j < N) microkernel(A, B, C, i, j, MR, N - j, K, lda, ldb, ldc);
    }
    if (i < M) {
        const int mr = M - i;
        int j = 0;
        for (; j + NR <= N; j += NR) microkernel(A, B, C, i, j, mr, NR, K, lda, ldb, ldc);
        if (j < N) microkernel(A, B, C, i, j, mr, N - j, K, lda, ldb, ldc);
    }
}
