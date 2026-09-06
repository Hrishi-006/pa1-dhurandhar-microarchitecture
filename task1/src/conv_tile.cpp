// conv_tile.cpp  STAGE 3: CACHE TILING
//
// Stage 1 made the inner loop fast but sweeps the WHOLE output image K*K times.
// At 2048x2048 that is 16 MB per sweep, so every pass re-reads out[] from DRAM and
// the kernel is bandwidth-bound.
//
// Tiling fixes the reuse distance.  Instead of "for each tap, sweep the whole
// image", we do "for each output tile, apply all K*K taps".  A tile is chosen small
// enough that the tile of out[] plus its (TH+K-1) x (TW+K-1) input halo stay resident
// in L1/L2, so the K*K passes hit cache and DRAM sees each byte roughly once.
//
// Loop nest:  oy0, ox0  (tiles)  ->  ky, kx  (taps)  ->  oy, ox  (within tile)
//
// Tile shape is a tuning knob, not a constant you can derive on paper: the useful
// size depends on associativity (a 2048-float row stride is 8 KB, so tile rows can
// alias in the same cache set), on the halo overhead, and on the hardware prefetcher.
// Sweep it on the target machine; see bench/ for the sweep harness.

#include "convolution.h"

// Tunable tile size.  The bench harness overrides these via `extern`; the graded
// run uses the defaults below.  <= 0 means "use the full extent" in that dimension.
int conv_tile_rows = 64;   // TH: output rows per tile
int conv_tile_cols = 256;  // TW: output cols per tile (snapped to a multiple of 8)

namespace {

// dst[0..n) = w * src[0..n)      (first tap: initializes the tile, so no memset)
inline void row_scale_store(float* __restrict dst, const float* __restrict src,
                            float w, int n) {
    int i = 0;
    for (; i + 8 <= n; i += 8) {  // 8-wide unroll: independent FMAs, no tail when n%8==0
        dst[i + 0] = w * src[i + 0];  dst[i + 1] = w * src[i + 1];
        dst[i + 2] = w * src[i + 2];  dst[i + 3] = w * src[i + 3];
        dst[i + 4] = w * src[i + 4];  dst[i + 5] = w * src[i + 5];
        dst[i + 6] = w * src[i + 6];  dst[i + 7] = w * src[i + 7];
    }
    for (; i < n; ++i) dst[i] = w * src[i];
}

// dst[0..n) += w * src[0..n)     (remaining taps accumulate on top)
inline void row_scale_acc(float* __restrict dst, const float* __restrict src,
                          float w, int n) {
    int i = 0;
    for (; i + 8 <= n; i += 8) {
        float a0 = dst[i + 0], a1 = dst[i + 1], a2 = dst[i + 2], a3 = dst[i + 3];
        float a4 = dst[i + 4], a5 = dst[i + 5], a6 = dst[i + 6], a7 = dst[i + 7];

        a0 += w * src[i + 0];  a1 += w * src[i + 1];
        a2 += w * src[i + 2];  a3 += w * src[i + 3];
        a4 += w * src[i + 4];  a5 += w * src[i + 5];
        a6 += w * src[i + 6];  a7 += w * src[i + 7];

        dst[i + 0] = a0;  dst[i + 1] = a1;  dst[i + 2] = a2;  dst[i + 3] = a3;
        dst[i + 4] = a4;  dst[i + 5] = a5;  dst[i + 6] = a6;  dst[i + 7] = a7;
    }
    for (; i < n; ++i) dst[i] += w * src[i];
}

}  // namespace

void conv_tile(const float* in, float* out, const float* ker,
               int H, int W, int K) {
    const float* __restrict in_p  = in;
    float* __restrict       out_p = out;

    const int p         = K / 2;
    const int in_stride = W + 2 * p;  // padded row stride

    // Resolve the tile shape.  Snapping TW up to a multiple of 8 keeps every tile
    // width a multiple of 8 (W is), so the unrolled loops never hit a scalar tail.
    int TH = (conv_tile_rows > 0) ? conv_tile_rows : H;
    int TW = (conv_tile_cols > 0) ? ((conv_tile_cols + 7) & ~7) : W;
    if (TH > H) TH = H;
    if (TW > W) TW = W;

    for (int oy0 = 0; oy0 < H; oy0 += TH) {
        const int oy1 = (oy0 + TH < H) ? oy0 + TH : H;

        for (int ox0 = 0; ox0 < W; ox0 += TW) {
            const int ox1 = (ox0 + TW < W) ? ox0 + TW : W;
            const int tw  = ox1 - ox0;  // tile width in floats

            // ---- all K*K taps for THIS tile, while it is still hot in cache ----

            // Tap (0,0) stores, which initializes the tile (no separate zeroing pass).
            {
                const float w = ker[0];
                for (int oy = oy0; oy < oy1; ++oy)
                    row_scale_store(out_p + (long)oy * W + ox0,
                                    in_p + (long)oy * in_stride + ox0, w, tw);
            }

            // The remaining K*K-1 taps accumulate.
            for (int ky = 0; ky < K; ++ky) {
                for (int kx = 0; kx < K; ++kx) {
                    if (ky == 0 && kx == 0) continue;

                    const float w = ker[ky * K + kx];  // loop-invariant: in a register

                    for (int oy = oy0; oy < oy1; ++oy)
                        row_scale_acc(out_p + (long)oy * W + ox0,
                                      in_p + (long)(oy + ky) * in_stride + ox0 + kx,
                                      w, tw);
                }
            }
        }
    }
}
