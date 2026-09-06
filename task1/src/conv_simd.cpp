// conv_simd.cpp  STAGE 4: SIMD (AVX2)
//
// Same (ky, kx, oy, ox) structure as stage 1.  Stage 1 already reshaped the inner
// loop into exactly the form a vector unit wants:
//
//     out[oy][ox] += w * in[oy+ky][ox+kx]      with w loop-invariant
//
// The weight is a scalar broadcast into all 8 lanes once per tap, and the two
// streams are unit-stride, so one _mm256_fmadd_ps replaces 8 scalar FMAs.
// W is a multiple of 8, so the loop never needs a scalar remainder.
//
// Loads/stores are the unaligned forms (loadu/storeu).  out[] is 64-byte aligned
// and W % 8 == 0, so dst is in fact always 32-byte aligned, but src is offset by
// kx floats and therefore is not.  On any AVX2-capable core an unaligned-form
// instruction on aligned data costs nothing extra, so using loadu everywhere is
// free and removes a whole class of alignment bugs.

#include <immintrin.h>

#include "convolution.h"

void conv_simd(const float* in, float* out, const float* ker,
               int H, int W, int K) {
    const float* __restrict in_p  = in;
    float* __restrict       out_p = out;

    const int p         = K / 2;
    const int in_stride = W + 2 * p;  // padded row stride

    // Tap (0,0) stores, which initializes out[] without a separate zeroing pass.
    {
        const __m256 wv = _mm256_set1_ps(ker[0]);
        for (int oy = 0; oy < H; ++oy) {
            const float* __restrict src = in_p + (long)oy * in_stride;
            float* __restrict       dst = out_p + (long)oy * W;
            for (int ox = 0; ox < W; ox += 8)
                _mm256_storeu_ps(dst + ox,
                                 _mm256_mul_ps(wv, _mm256_loadu_ps(src + ox)));
        }
    }

    // The remaining K*K-1 taps accumulate.
    for (int ky = 0; ky < K; ++ky) {
        for (int kx = 0; kx < K; ++kx) {
            if (ky == 0 && kx == 0) continue;

            // One broadcast per tap, hoisted out of both spatial loops.
            const __m256 wv = _mm256_set1_ps(ker[ky * K + kx]);

            for (int oy = 0; oy < H; ++oy) {
                const float* __restrict src = in_p + (long)(oy + ky) * in_stride + kx;
                float* __restrict       dst = out_p + (long)oy * W;

                for (int ox = 0; ox < W; ox += 8) {
                    __m256 d = _mm256_loadu_ps(dst + ox);
                    // fmadd: d = wv * src + d, one rounding, one instruction.
                    d = _mm256_fmadd_ps(wv, _mm256_loadu_ps(src + ox), d);
                    _mm256_storeu_ps(dst + ox, d);
                }
            }
        }
    }
}
