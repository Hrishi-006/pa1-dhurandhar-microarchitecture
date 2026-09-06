// conv_unroll.cpp  STAGE 2: LOOP UNROLLING
//
// Same (ky, kx, oy, ox) structure as stage 1, but the inner ox loop is unrolled
// 8x.  Two reasons for 8:
//   * An FMA has ~4 cycle latency and 2/cycle throughput, so you need ~8
//     independent FMAs in flight to keep both FP pipes busy.
//   * W is guaranteed to be a multiple of 8, so there is no remainder tail, and
//     8 floats is exactly one AVX2 register -- stage 4 is then a mechanical
//     rewrite of this same loop body.
//
// Unrolling also amortizes the loop counter/compare over 8 elements and lets the
// compiler use [base + constant] addressing instead of bumping a pointer each time.
//
// Expect this to help a lot when the image is cache-resident, and very little at
// 2048x2048: stage 1 already made the kernel DRAM-bandwidth-bound, and more ILP
// does not help a loop that is waiting on memory.  That is the point of the stage.

#include "convolution.h"

void conv_unroll(const float* in, float* out, const float* ker,
                 int H, int W, int K) {
    // See conv_reorder.cpp: __restrict tells the compiler the streams do not alias,
    // which is what allows the 8 loads/FMAs/stores below to be independent.
    const float* __restrict in_p  = in;
    float* __restrict       out_p = out;

    const int p         = K / 2;
    const int in_stride = W + 2 * p;  // padded row stride

    // Tap (0,0) stores, so out[] needs no separate zeroing pass.
    {
        const float w = ker[0];
        for (int oy = 0; oy < H; ++oy) {
            const float* __restrict src = in_p + oy * in_stride;
            float* __restrict       dst = out_p + oy * W;
            for (int ox = 0; ox < W; ox += 8) {
                dst[ox + 0] = w * src[ox + 0];
                dst[ox + 1] = w * src[ox + 1];
                dst[ox + 2] = w * src[ox + 2];
                dst[ox + 3] = w * src[ox + 3];
                dst[ox + 4] = w * src[ox + 4];
                dst[ox + 5] = w * src[ox + 5];
                dst[ox + 6] = w * src[ox + 6];
                dst[ox + 7] = w * src[ox + 7];
            }
        }
    }

    // The remaining K*K-1 taps accumulate on top.
    for (int ky = 0; ky < K; ++ky) {
        for (int kx = 0; kx < K; ++kx) {
            if (ky == 0 && kx == 0) continue;

            const float w = ker[ky * K + kx];  // loop-invariant: stays in a register

            for (int oy = 0; oy < H; ++oy) {
                const float* __restrict src = in_p + (oy + ky) * in_stride + kx;
                float* __restrict       dst = out_p + oy * W;

                for (int ox = 0; ox < W; ox += 8) {
                    // 8 independent accumulators: load all, FMA all, store all, so
                    // no FMA has to wait for the one before it.
                    float a0 = dst[ox + 0], a1 = dst[ox + 1];
                    float a2 = dst[ox + 2], a3 = dst[ox + 3];
                    float a4 = dst[ox + 4], a5 = dst[ox + 5];
                    float a6 = dst[ox + 6], a7 = dst[ox + 7];

                    a0 += w * src[ox + 0];  a1 += w * src[ox + 1];
                    a2 += w * src[ox + 2];  a3 += w * src[ox + 3];
                    a4 += w * src[ox + 4];  a5 += w * src[ox + 5];
                    a6 += w * src[ox + 6];  a7 += w * src[ox + 7];

                    dst[ox + 0] = a0;  dst[ox + 1] = a1;
                    dst[ox + 2] = a2;  dst[ox + 3] = a3;
                    dst[ox + 4] = a4;  dst[ox + 5] = a5;
                    dst[ox + 6] = a6;  dst[ox + 7] = a7;
                }
            }
        }
    }
}
