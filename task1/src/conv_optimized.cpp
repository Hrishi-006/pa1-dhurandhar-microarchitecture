// conv_optimized.cpp  STAGE 5: EVERYTHING COMBINED
//
// reorder + tiling + AVX2 + unrolling, in that dependency order:
//
//   reorder  gives an inner loop with no dependence chain and unit stride, which is
//            the only shape the other three can build on.
//   tiling   keeps the K*K passes inside a cache-resident tile, so the kernel stops
//            waiting on DRAM.  This is the step that makes the rest matter.
//   AVX2     8 columns per instruction once the data is actually in cache.
//   unroll   2 vectors (16 columns) per iteration, so two independent FMA chains
//            cover the ~4 cycle FMA latency.
//
// Why the order matters: SIMD on its own is nearly free of benefit here, because an
// untiled kernel is bandwidth-bound and a narrower instruction stream does not make
// DRAM faster.  Tiling removes that bottleneck; SIMD then converts the freed headroom
// into real speedup.  That is the synergy.
//
// TILE_ROWS / TILE_COLS are the tuning knobs: retune them on the target machine
// (see bench/sweep.sh) and paste the winning pair here.

#include <immintrin.h>

#include "convolution.h"

namespace {

constexpr int TILE_ROWS = 64;   // output rows per tile
constexpr int TILE_COLS = 256;  // output cols per tile (multiple of 8)

// dst[0..n) = w * src[0..n)      (first tap: initializes the tile, no memset needed)
inline void row_store(float* __restrict dst, const float* __restrict src,
                      const __m256 wv, int n) {
    int i = 0;
    for (; i + 16 <= n; i += 16) {  // 2 vectors per iteration
        _mm256_storeu_ps(dst + i, _mm256_mul_ps(wv, _mm256_loadu_ps(src + i)));
        _mm256_storeu_ps(dst + i + 8, _mm256_mul_ps(wv, _mm256_loadu_ps(src + i + 8)));
    }
    for (; i + 8 <= n; i += 8)      // 8-column remainder (W is a multiple of 8)
        _mm256_storeu_ps(dst + i, _mm256_mul_ps(wv, _mm256_loadu_ps(src + i)));
}

// dst[0..n) += w * src[0..n)     (remaining taps accumulate)
inline void row_acc(float* __restrict dst, const float* __restrict src,
                    const __m256 wv, int n) {
    int i = 0;
    for (; i + 16 <= n; i += 16) {
        // Two independent accumulators: neither FMA waits on the other.
        __m256 d0 = _mm256_loadu_ps(dst + i);
        __m256 d1 = _mm256_loadu_ps(dst + i + 8);
        d0 = _mm256_fmadd_ps(wv, _mm256_loadu_ps(src + i), d0);
        d1 = _mm256_fmadd_ps(wv, _mm256_loadu_ps(src + i + 8), d1);
        _mm256_storeu_ps(dst + i, d0);
        _mm256_storeu_ps(dst + i + 8, d1);
    }
    for (; i + 8 <= n; i += 8) {
        __m256 d = _mm256_loadu_ps(dst + i);
        d = _mm256_fmadd_ps(wv, _mm256_loadu_ps(src + i), d);
        _mm256_storeu_ps(dst + i, d);
    }
}

}  // namespace

void conv_optimized(const float* in, float* out, const float* ker,
                    int H, int W, int K) {
    const float* __restrict in_p  = in;
    float* __restrict       out_p = out;

    const int p         = K / 2;
    const int in_stride = W + 2 * p;

    const int TH = (TILE_ROWS < H) ? TILE_ROWS : H;
    const int TW = (TILE_COLS < W) ? TILE_COLS : W;  // both are multiples of 8

    for (int oy0 = 0; oy0 < H; oy0 += TH) {
        const int oy1 = (oy0 + TH < H) ? oy0 + TH : H;

        for (int ox0 = 0; ox0 < W; ox0 += TW) {
            const int ox1 = (ox0 + TW < W) ? ox0 + TW : W;
            const int tw  = ox1 - ox0;

            // ---- all K*K taps for this tile, while the tile is hot in cache ----
            {
                const __m256 wv = _mm256_set1_ps(ker[0]);  // tap (0,0) stores
                for (int oy = oy0; oy < oy1; ++oy)
                    row_store(out_p + (long)oy * W + ox0,
                              in_p + (long)oy * in_stride + ox0, wv, tw);
            }

            for (int ky = 0; ky < K; ++ky) {
                for (int kx = 0; kx < K; ++kx) {
                    if (ky == 0 && kx == 0) continue;

                    // One broadcast per tap, hoisted out of both spatial loops.
                    const __m256 wv = _mm256_set1_ps(ker[ky * K + kx]);

                    for (int oy = oy0; oy < oy1; ++oy)
                        row_acc(out_p + (long)oy * W + ox0,
                                in_p + (long)(oy + ky) * in_stride + ox0 + kx, wv, tw);
                }
            }
        }
    }
}
