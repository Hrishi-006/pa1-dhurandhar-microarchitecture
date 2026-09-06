// conv_combo.cpp  tunable tile x width x unroll kernel.  NOT part of the submission.
//
// Structure is always the same (reorder + tiling); only the inner row loop changes
// with the width/unroll axes, so a speedup difference is attributable to the axis
// that moved and nothing else.
//
//   for each tile -> for each tap -> for each row -> [row_store | row_acc]
//
// Each width lives in its own function with a `target` attribute, so all of them
// compile into one binary regardless of the build flags.

#include <immintrin.h>

#include "conv_combo.h"

int combo_tile_rows = 64;
int combo_tile_cols = 256;
int combo_width     = 256;
int combo_unroll    = 2;

// ---------------------------------------------------------------- row kernels
// Each pair does: dst[0..n) = w*src[0..n)   (store, first tap)
//                 dst[0..n) += w*src[0..n)  (accumulate, later taps)
// U is the unroll factor in vectors.

namespace {

// ---- scalar ----
void row_scalar(float* __restrict dst, const float* __restrict src, float w, int n,
                bool store, int U) {
    const int step = 8 * U;
    int i = 0;
    if (store) {
        for (; i + step <= n; i += step)
            for (int j = 0; j < step; ++j) dst[i + j] = w * src[i + j];
        for (; i < n; ++i) dst[i] = w * src[i];
    } else {
        for (; i + step <= n; i += step)
            for (int j = 0; j < step; ++j) dst[i + j] += w * src[i + j];
        for (; i < n; ++i) dst[i] += w * src[i];
    }
}

// ---- 128-bit ----
__attribute__((target("avx,fma")))
void row_128(float* __restrict dst, const float* __restrict src, float w, int n,
             bool store, int U) {
    const __m128 wv = _mm_set1_ps(w);
    int i = 0;
    if (U == 2) {
        for (; i + 8 <= n; i += 8) {
            if (store) {
                _mm_storeu_ps(dst + i, _mm_mul_ps(wv, _mm_loadu_ps(src + i)));
                _mm_storeu_ps(dst + i + 4, _mm_mul_ps(wv, _mm_loadu_ps(src + i + 4)));
            } else {
                __m128 d0 = _mm_loadu_ps(dst + i), d1 = _mm_loadu_ps(dst + i + 4);
                d0 = _mm_fmadd_ps(wv, _mm_loadu_ps(src + i), d0);
                d1 = _mm_fmadd_ps(wv, _mm_loadu_ps(src + i + 4), d1);
                _mm_storeu_ps(dst + i, d0);
                _mm_storeu_ps(dst + i + 4, d1);
            }
        }
    }
    for (; i + 4 <= n; i += 4) {
        if (store) _mm_storeu_ps(dst + i, _mm_mul_ps(wv, _mm_loadu_ps(src + i)));
        else {
            __m128 d = _mm_loadu_ps(dst + i);
            d = _mm_fmadd_ps(wv, _mm_loadu_ps(src + i), d);
            _mm_storeu_ps(dst + i, d);
        }
    }
    for (; i < n; ++i) { if (store) dst[i] = w * src[i]; else dst[i] += w * src[i]; }
}

// ---- 256-bit ----
__attribute__((target("avx2,fma")))
void row_256(float* __restrict dst, const float* __restrict src, float w, int n,
             bool store, int U) {
    const __m256 wv = _mm256_set1_ps(w);
    int i = 0;
    if (U == 2) {
        for (; i + 16 <= n; i += 16) {
            if (store) {
                _mm256_storeu_ps(dst + i, _mm256_mul_ps(wv, _mm256_loadu_ps(src + i)));
                _mm256_storeu_ps(dst + i + 8,
                                 _mm256_mul_ps(wv, _mm256_loadu_ps(src + i + 8)));
            } else {
                __m256 d0 = _mm256_loadu_ps(dst + i), d1 = _mm256_loadu_ps(dst + i + 8);
                d0 = _mm256_fmadd_ps(wv, _mm256_loadu_ps(src + i), d0);
                d1 = _mm256_fmadd_ps(wv, _mm256_loadu_ps(src + i + 8), d1);
                _mm256_storeu_ps(dst + i, d0);
                _mm256_storeu_ps(dst + i + 8, d1);
            }
        }
    }
    for (; i + 8 <= n; i += 8) {
        if (store) _mm256_storeu_ps(dst + i, _mm256_mul_ps(wv, _mm256_loadu_ps(src + i)));
        else {
            __m256 d = _mm256_loadu_ps(dst + i);
            d = _mm256_fmadd_ps(wv, _mm256_loadu_ps(src + i), d);
            _mm256_storeu_ps(dst + i, d);
        }
    }
    for (; i < n; ++i) { if (store) dst[i] = w * src[i]; else dst[i] += w * src[i]; }
}

// ---- 512-bit ----
__attribute__((target("avx512f")))
void row_512(float* __restrict dst, const float* __restrict src, float w, int n,
             bool store, int U) {
    const __m512 wv = _mm512_set1_ps(w);
    int i = 0;
    if (U == 2) {
        for (; i + 32 <= n; i += 32) {
            if (store) {
                _mm512_storeu_ps(dst + i, _mm512_mul_ps(wv, _mm512_loadu_ps(src + i)));
                _mm512_storeu_ps(dst + i + 16,
                                 _mm512_mul_ps(wv, _mm512_loadu_ps(src + i + 16)));
            } else {
                __m512 d0 = _mm512_loadu_ps(dst + i), d1 = _mm512_loadu_ps(dst + i + 16);
                d0 = _mm512_fmadd_ps(wv, _mm512_loadu_ps(src + i), d0);
                d1 = _mm512_fmadd_ps(wv, _mm512_loadu_ps(src + i + 16), d1);
                _mm512_storeu_ps(dst + i, d0);
                _mm512_storeu_ps(dst + i + 16, d1);
            }
        }
    }
    for (; i + 16 <= n; i += 16) {
        if (store) _mm512_storeu_ps(dst + i, _mm512_mul_ps(wv, _mm512_loadu_ps(src + i)));
        else {
            __m512 d = _mm512_loadu_ps(dst + i);
            d = _mm512_fmadd_ps(wv, _mm512_loadu_ps(src + i), d);
            _mm512_storeu_ps(dst + i, d);
        }
    }
    for (; i < n; ++i) { if (store) dst[i] = w * src[i]; else dst[i] += w * src[i]; }
}

inline void row_dispatch(float* dst, const float* src, float w, int n, bool store) {
    switch (combo_width) {
        case 128: row_128(dst, src, w, n, store, combo_unroll); break;
        case 256: row_256(dst, src, w, n, store, combo_unroll); break;
        case 512: row_512(dst, src, w, n, store, combo_unroll); break;
        default:  row_scalar(dst, src, w, n, store, combo_unroll); break;
    }
}

}  // namespace

bool combo_supported() {
    if (combo_unroll != 1 && combo_unroll != 2) return false;
    switch (combo_width) {
        case 0:   return true;
        case 128: return __builtin_cpu_supports("avx") && __builtin_cpu_supports("fma");
        case 256: return __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma");
        case 512: return __builtin_cpu_supports("avx512f");
        default:  return false;
    }
}

void conv_combo(const float* in, float* out, const float* ker, int H, int W, int K) {
    const float* __restrict in_p  = in;
    float* __restrict       out_p = out;

    const int p         = K / 2;
    const int in_stride = W + 2 * p;

    // Snap tile width up to a multiple of 8 so tiles stay vector-friendly.
    int TH = (combo_tile_rows > 0) ? combo_tile_rows : H;
    int TW = (combo_tile_cols > 0) ? ((combo_tile_cols + 7) & ~7) : W;
    if (TH > H) TH = H;
    if (TW > W) TW = W;

    for (int oy0 = 0; oy0 < H; oy0 += TH) {
        const int oy1 = (oy0 + TH < H) ? oy0 + TH : H;

        for (int ox0 = 0; ox0 < W; ox0 += TW) {
            const int ox1 = (ox0 + TW < W) ? ox0 + TW : W;
            const int tw  = ox1 - ox0;

            // tap (0,0) stores, so the tile needs no zeroing pass
            for (int oy = oy0; oy < oy1; ++oy)
                row_dispatch(out_p + (long)oy * W + ox0,
                             in_p + (long)oy * in_stride + ox0, ker[0], tw, true);

            for (int ky = 0; ky < K; ++ky) {
                for (int kx = 0; kx < K; ++kx) {
                    if (ky == 0 && kx == 0) continue;
                    const float w = ker[ky * K + kx];
                    for (int oy = oy0; oy < oy1; ++oy)
                        row_dispatch(out_p + (long)oy * W + ox0,
                                     in_p + (long)(oy + ky) * in_stride + ox0 + kx, w,
                                     tw, false);
                }
            }
        }
    }
}
