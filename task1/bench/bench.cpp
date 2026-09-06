// bench.cpp  standalone driver for perf measurements.  NOT part of the submission.
//
// Runs exactly ONE kernel in a tight loop so `perf stat` attributes its counters to
// that kernel instead of to the harness.  Setup happens before the loop and is
// negligible once reps is a few tens.
//
//   ./bench <kernel> [H] [W] [K] [TH] [TW] [reps] [--check]
//
//   kernel : naive | reorder | unroll | tile
//   TH, TW : tile size, used by the `tile` kernel only (0 = full extent)
//   --check: verify against conv_naive and print the max abs error, then exit
//            (do NOT pass this under perf: it adds instructions to the count)
//
// Prints one CSV line on stdout so a sweep script can just append it.

#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "conv_combo.h"
#include "conv_widths.h"
#include "convolution.h"
#include "timer.h"
#include "utils.h"

// Tunable tile size, defined in src/conv_tile.cpp.
extern int conv_tile_rows;
extern int conv_tile_cols;

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr,
                     "usage: %s <kernel> [H] [W] [K] [TH] [TW] [reps] [--check]\n"
                     "  kernel: naive | reorder | unroll | tile | simd | optimized\n"
                     "        | simd128 | simd256 | simd512   (SIMD width study)\n"
                     "        | combo   (tile x width x unroll; set WIDTH/UNROLL env)\n",
                     argv[0]);
        return 1;
    }

    const char* name = argv[1];
    const int H    = (argc > 2) ? std::atoi(argv[2]) : 2048;
    const int W    = (argc > 3) ? std::atoi(argv[3]) : 2048;
    const int K    = (argc > 4) ? std::atoi(argv[4]) : 3;
    const int TH   = (argc > 5) ? std::atoi(argv[5]) : 64;
    const int TW   = (argc > 6) ? std::atoi(argv[6]) : 256;
    const int reps = (argc > 7) ? std::atoi(argv[7]) : 20;

    bool check = false;
    for (int i = 1; i < argc; ++i)
        if (std::strcmp(argv[i], "--check") == 0) check = true;

    if (W % 8 != 0 || K % 2 == 0 || K < 1 || H < 1) {
        std::fprintf(stderr, "bad workload: W must be a multiple of 8, K odd >= 1\n");
        return 1;
    }

    ConvFn fn = nullptr;
    if      (std::strcmp(name, "naive")   == 0) fn = conv_naive;
    else if (std::strcmp(name, "reorder") == 0) fn = conv_reorder;
    else if (std::strcmp(name, "unroll")  == 0) fn = conv_unroll;
    else if (std::strcmp(name, "tile")    == 0) fn = conv_tile;
    else if (std::strcmp(name, "simd")    == 0) fn = conv_simd;
    else if (std::strcmp(name, "simd128") == 0) fn = conv_simd128;
    else if (std::strcmp(name, "simd256") == 0) fn = conv_simd256;
    else if (std::strcmp(name, "simd512") == 0) fn = conv_simd512;
    else if (std::strcmp(name, "combo")   == 0) fn = conv_combo;
    else if (std::strcmp(name, "optimized") == 0) fn = conv_optimized;
    else { std::fprintf(stderr, "unknown kernel '%s'\n", name); return 1; }

    // These kernels are compiled with target attributes, so they exist in the binary
    // even on a CPU that cannot run them.  Check before calling, or it is a SIGILL.
    if (std::strcmp(name, "simd512") == 0 && !cpu_has_avx512f()) {
        std::fprintf(stderr, "skip: this CPU has no AVX-512F\n");
        return 77;  // distinct exit code so the sweep can skip rather than fail
    }
    if ((std::strncmp(name, "simd", 4) == 0) && !(cpu_has_avx2() && cpu_has_fma())) {
        std::fprintf(stderr, "skip: this CPU has no AVX2+FMA\n");
        return 77;
    }

    conv_tile_rows = TH;
    conv_tile_cols = TW;

    // combo takes its extra two axes from the environment, so the sweep can vary
    // them without adding positional arguments.
    combo_tile_rows = TH;
    combo_tile_cols = TW;
    if (const char* e = std::getenv("WIDTH"))  combo_width  = std::atoi(e);
    if (const char* e = std::getenv("UNROLL")) combo_unroll = std::atoi(e);
    if (std::strcmp(name, "combo") == 0 && !combo_supported()) {
        std::fprintf(stderr, "skip: combo width=%d unroll=%d unsupported here\n",
                     combo_width, combo_unroll);
        return 77;
    }

    // ---- setup (outside the measured loop) ----
    float* img = pa1::alloc_floats((std::size_t)H * W);
    float* ker = pa1::alloc_floats((std::size_t)K * K);
    float* out = pa1::alloc_floats((std::size_t)H * W);
    pa1::fill_random(img, (std::size_t)H * W, 1234u);
    pa1::fill_random(ker, (std::size_t)K * K, 1235u);
    float* in = pa1::make_padded(img, H, W, K);

    if (check) {
        float* ref = pa1::alloc_floats((std::size_t)H * W);
        conv_naive(in, ref, ker, H, W, K);
        fn(in, out, ker, H, W, K);
        const float err = pa1::max_abs_diff(out, ref, H, W);
        std::printf("%-8s H=%d W=%d K=%d TH=%d TW=%d w=%d u=%d  max_abs_err=%.3g  %s\n",
                    name, H, W, K, TH, TW, combo_width, combo_unroll, (double)err,
                    err <= 1e-3f ? "OK" : "FAIL");
        pa1::free_floats(ref);
        pa1::free_floats(in); pa1::free_floats(img);
        pa1::free_floats(ker); pa1::free_floats(out);
        return (err <= 1e-3f) ? 0 : 2;
    }

    // ---- the measured region: nothing here but the kernel ----
    auto run = [&]() { fn(in, out, ker, H, W, K); };
    const double ms = pa1::time_median_ms(run, 2, reps);

    const double gflops = pa1::conv_flops(H, W, K) / (ms * 1e6);
    // CSV: kernel,H,W,K,TH,TW,width,unroll,reps,ms,gflops
    const bool is_combo = (std::strcmp(name, "combo") == 0);
    std::printf("%s,%d,%d,%d,%d,%d,%d,%d,%d,%.4f,%.3f\n", name, H, W, K, TH, TW,
                is_combo ? combo_width : 0, is_combo ? combo_unroll : 0, reps, ms,
                gflops);

    pa1::free_floats(in); pa1::free_floats(img);
    pa1::free_floats(ker); pa1::free_floats(out);
    return 0;
}
