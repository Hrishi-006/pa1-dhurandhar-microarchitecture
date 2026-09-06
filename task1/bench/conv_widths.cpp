// conv_widths.cpp  SIMD width study.  NOT part of the submission.
//
// Each kernel carries a GCC/Clang `target` attribute, so all three widths compile
// into one binary even though the build line only says -mavx2 -mfma.  Nothing here
// runs unless the bench has checked the CPU supports it first.
//
// All three are the stage-1 (reorder) loop nest with the inner ox loop replaced by
// one vector op: broadcast the tap weight once, then stream fmadd over the row.
// Only the width (4 / 8 / 16 floats per instruction) differs.

#include <immintrin.h>

#include "conv_widths.h"

bool cpu_has_avx512f() { return __builtin_cpu_supports("avx512f"); }
bool cpu_has_avx2()    { return __builtin_cpu_supports("avx2"); }
bool cpu_has_fma()     { return __builtin_cpu_supports("fma"); }

// ---------------------------------------------------------------- 128-bit (SSE/AVX)
__attribute__((target("avx,fma")))
void conv_simd128(const float* in, float* out, const float* ker, int H, int W, int K) {
    const float* __restrict in_p = in;
    float* __restrict out_p = out;
    const int p = K / 2, in_stride = W + 2 * p;

    {   // tap (0,0) stores
        const __m128 wv = _mm_set1_ps(ker[0]);
        for (int oy = 0; oy < H; ++oy) {
            const float* __restrict src = in_p + (long)oy * in_stride;
            float* __restrict dst = out_p + (long)oy * W;
            for (int ox = 0; ox < W; ox += 4)
                _mm_storeu_ps(dst + ox, _mm_mul_ps(wv, _mm_loadu_ps(src + ox)));
        }
    }
    for (int ky = 0; ky < K; ++ky) {
        for (int kx = 0; kx < K; ++kx) {
            if (ky == 0 && kx == 0) continue;
            const __m128 wv = _mm_set1_ps(ker[ky * K + kx]);
            for (int oy = 0; oy < H; ++oy) {
                const float* __restrict src = in_p + (long)(oy + ky) * in_stride + kx;
                float* __restrict dst = out_p + (long)oy * W;
                for (int ox = 0; ox < W; ox += 4) {
                    __m128 d = _mm_loadu_ps(dst + ox);
                    d = _mm_fmadd_ps(wv, _mm_loadu_ps(src + ox), d);
                    _mm_storeu_ps(dst + ox, d);
                }
            }
        }
    }
}

// ---------------------------------------------------------------- 256-bit (AVX2)
__attribute__((target("avx2,fma")))
void conv_simd256(const float* in, float* out, const float* ker, int H, int W, int K) {
    const float* __restrict in_p = in;
    float* __restrict out_p = out;
    const int p = K / 2, in_stride = W + 2 * p;

    {   // tap (0,0) stores
        const __m256 wv = _mm256_set1_ps(ker[0]);
        for (int oy = 0; oy < H; ++oy) {
            const float* __restrict src = in_p + (long)oy * in_stride;
            float* __restrict dst = out_p + (long)oy * W;
            for (int ox = 0; ox < W; ox += 8)
                _mm256_storeu_ps(dst + ox, _mm256_mul_ps(wv, _mm256_loadu_ps(src + ox)));
        }
    }
    for (int ky = 0; ky < K; ++ky) {
        for (int kx = 0; kx < K; ++kx) {
            if (ky == 0 && kx == 0) continue;
            const __m256 wv = _mm256_set1_ps(ker[ky * K + kx]);
            for (int oy = 0; oy < H; ++oy) {
                const float* __restrict src = in_p + (long)(oy + ky) * in_stride + kx;
                float* __restrict dst = out_p + (long)oy * W;
                for (int ox = 0; ox < W; ox += 8) {
                    __m256 d = _mm256_loadu_ps(dst + ox);
                    d = _mm256_fmadd_ps(wv, _mm256_loadu_ps(src + ox), d);
                    _mm256_storeu_ps(dst + ox, d);
                }
            }
        }
    }
}

// ---------------------------------------------------------------- 512-bit (AVX-512F)
// W is only guaranteed to be a multiple of 8, not 16, so this one needs a tail.
__attribute__((target("avx512f")))
void conv_simd512(const float* in, float* out, const float* ker, int H, int W, int K) {
    const float* __restrict in_p = in;
    float* __restrict out_p = out;
    const int p = K / 2, in_stride = W + 2 * p;
    const int Wv = W & ~15;  // largest multiple of 16

    {   // tap (0,0) stores
        const float w = ker[0];
        const __m512 wv = _mm512_set1_ps(w);
        for (int oy = 0; oy < H; ++oy) {
            const float* __restrict src = in_p + (long)oy * in_stride;
            float* __restrict dst = out_p + (long)oy * W;
            int ox = 0;
            for (; ox < Wv; ox += 16)
                _mm512_storeu_ps(dst + ox, _mm512_mul_ps(wv, _mm512_loadu_ps(src + ox)));
            for (; ox < W; ++ox) dst[ox] = w * src[ox];
        }
    }
    for (int ky = 0; ky < K; ++ky) {
        for (int kx = 0; kx < K; ++kx) {
            if (ky == 0 && kx == 0) continue;
            const float w = ker[ky * K + kx];
            const __m512 wv = _mm512_set1_ps(w);
            for (int oy = 0; oy < H; ++oy) {
                const float* __restrict src = in_p + (long)(oy + ky) * in_stride + kx;
                float* __restrict dst = out_p + (long)oy * W;
                int ox = 0;
                for (; ox < Wv; ox += 16) {
                    __m512 d = _mm512_loadu_ps(dst + ox);
                    d = _mm512_fmadd_ps(wv, _mm512_loadu_ps(src + ox), d);
                    _mm512_storeu_ps(dst + ox, d);
                }
                for (; ox < W; ++ox) dst[ox] += w * src[ox];
            }
        }
    }
}
