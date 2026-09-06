// conv_widths.h  SIMD width study (128 / 256 / 512 bit).  NOT part of the submission.
//
// Three implementations of the SAME algorithm as src/conv_simd.cpp, differing only
// in vector width, so a speedup comparison isolates width and nothing else.
#ifndef CS683_PA1_CONV_WIDTHS_H
#define CS683_PA1_CONV_WIDTHS_H

void conv_simd128(const float* in, float* out, const float* ker, int H, int W, int K);
void conv_simd256(const float* in, float* out, const float* ker, int H, int W, int K);
void conv_simd512(const float* in, float* out, const float* ker, int H, int W, int K);

// Runtime CPU checks: AVX-512 is absent on most client parts, so the bench must ask
// before calling conv_simd512 rather than crashing with SIGILL.
bool cpu_has_avx512f();
bool cpu_has_avx2();
bool cpu_has_fma();

#endif  // CS683_PA1_CONV_WIDTHS_H
