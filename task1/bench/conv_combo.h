// conv_combo.h  tunable tile x width x unroll kernel.  NOT part of the submission.
//
// One kernel whose three axes are set at runtime, so a sweep can find the best
// combination without a rebuild per config.  src/conv_optimized.cpp is this kernel
// with the winning parameters hardcoded.
#ifndef CS683_PA1_CONV_COMBO_H
#define CS683_PA1_CONV_COMBO_H

// Axis 1: tile shape.  <= 0 means "no tiling in that dimension" (full extent).
extern int combo_tile_rows;
extern int combo_tile_cols;
// Axis 2: vector width in bits: 0 = scalar, 128, 256, 512.
extern int combo_width;
// Axis 3: unroll factor, in vectors per iteration: 1 or 2.
extern int combo_unroll;

// Returns false if the requested config cannot run on this CPU (e.g. 512 without
// AVX-512F) or is not a valid combination; the caller should skip it.
bool combo_supported();

void conv_combo(const float* in, float* out, const float* ker, int H, int W, int K);

#endif  // CS683_PA1_CONV_COMBO_H
