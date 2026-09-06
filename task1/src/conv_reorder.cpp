// conv_reorder.cpp  STAGE 1: LOOP REORDERING
// Hint: loops from outermost to innermost -> ky, kx, oy, ox.
//
// Naive order (oy, ox, ky, kx) accumulates K*K products into ONE register `acc`,
// so every FMA waits on the previous one: the loop runs at FMA latency (~4 cycles),
// not FMA throughput (2/cycle).
//
// Reordering to (ky, kx, oy, ox) turns the kernel into K*K independent "scale and
// add a row" passes.  For a fixed tap (ky,kx) the weight is loop-invariant (one
// register, no reload), and the inner loop is two unit-stride streams:
//
//     out[oy][ox] += w * in[oy+ky][ox+kx]
//
// Consecutive iterations touch different out[] elements, so there is no dependence
// chain left to stall on, and the address arithmetic collapses to pointer bumps.
//
// The price: this makes K*K full passes over the whole output image.  At
// 2048x2048 that is 16 MB per pass, far beyond L3, so the kernel trades a
// latency bottleneck for a DRAM-bandwidth one.  Cache tiling (stage 3) is what
// makes these passes cache-resident and finally pays the ILP win off.

#include "convolution.h"

void conv_reorder(const float* in, float* out, const float* ker,
                  int H, int W, int K) {
    // in and out are distinct buffers, but the compiler cannot prove it.  Without
    // __restrict it must assume the store to dst[ox] may alias src[ox+1] and reload
    // on every iteration, which re-serializes the loop.
    const float* __restrict in_p  = in;
    float* __restrict       out_p = out;

    const int p         = K / 2;
    const int in_stride = W + 2 * p;  // padded row stride

    // Tap (0,0) STORES instead of accumulating.  That initializes out[] as part of
    // a pass we have to make anyway, saving a separate memset over 16 MB.
    {
        const float w = ker[0];
        for (int oy = 0; oy < H; ++oy) {
            const float* __restrict src = in_p + oy * in_stride;
            float* __restrict       dst = out_p + oy * W;
            for (int ox = 0; ox < W; ++ox) dst[ox] = w * src[ox];
        }
    }

    // The remaining K*K-1 taps accumulate on top.
    for (int ky = 0; ky < K; ++ky) {
        for (int kx = 0; kx < K; ++kx) {
            if (ky == 0 && kx == 0) continue;  // already done above

            const float w = ker[ky * K + kx];  // loop-invariant: stays in a register

            for (int oy = 0; oy < H; ++oy) {
                // Row bases hoisted out of the inner loop: no multiply per element.
                const float* __restrict src = in_p + (oy + ky) * in_stride + kx;
                float* __restrict       dst = out_p + oy * W;

                for (int ox = 0; ox < W; ++ox) dst[ox] += w * src[ox];
            }
        }
    }
}
