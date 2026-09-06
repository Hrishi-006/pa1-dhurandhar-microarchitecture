// bench_matmul.cpp  standalone driver for perf measurements.  NOT part of the submission.
//
// Runs exactly ONE kernel `reps` times in a tight loop so `perf stat` attributes its
// counters to that kernel instead of to setup/teardown. Prints one CSV line on stdout so
// a sweep script can just append it.
//
//   ./bench_matmul <kernel> [M] [N] [K] [dist] [hint] [enable] [tile] [reps] [--check]
//
//   kernel : naive | simd | prefetch | optimized
//   dist   : software prefetch distance, in floats (matmul_prefetch / matmul_optimized only)
//   hint   : 0=T0 1=T1 2=T2 3=NTA                  (matmul_prefetch / matmul_optimized only)
//   enable : 1/0 -- turn software prefetch on/off (matmul_prefetch / matmul_optimized only;
//            isolates the prefetch contribution from the tiling/SIMD it sits on top of)
//   tile   : M/N cache-tile size, 0 = full extent   (matmul_prefetch / matmul_optimized only)
//   --check: verify against matmul_naive and print the max abs error, then exit
//            (do NOT pass this under perf: it adds instructions to the count)

#include <immintrin.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "matmul.h"
#include "timer.h"
#include "utils.h"

// Tunables defined in src/matmul_prefetch.cpp / src/matmul_optimized.cpp.
extern int matmul_tile_m, matmul_tile_n;
extern int matmul_prefetch_dist, matmul_prefetch_hint, matmul_prefetch_enable;
extern int matmul_opt_tile_m, matmul_opt_tile_n;
extern int matmul_optimized_prefetch, matmul_optimized_prefetch_dist,
    matmul_optimized_prefetch_hint;

namespace {
// CLI convention (0=T0,1=T1,2=T2,3=NTA) does NOT match the real _MM_HINT_* values
// (T0=3,T1=2,T2=1,NTA=0 in immintrin.h) -- map explicitly rather than passing through.
int hint_arg_to_mm_hint(int h) {
    switch (h) {
        case 0: return _MM_HINT_T0;
        case 1: return _MM_HINT_T1;
        case 2: return _MM_HINT_T2;
        default: return _MM_HINT_NTA;
    }
}
}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr,
                     "usage: %s <kernel> [M] [N] [K] [dist] [hint] [enable] [tile] [reps] "
                     "[--check]\n"
                     "  kernel: naive | simd | prefetch | optimized\n",
                     argv[0]);
        return 1;
    }

    const char* name = argv[1];
    const int M      = (argc > 2) ? std::atoi(argv[2]) : 1024;
    const int N      = (argc > 3) ? std::atoi(argv[3]) : 1024;
    const int K      = (argc > 4) ? std::atoi(argv[4]) : 1024;
    const int dist   = (argc > 5) ? std::atoi(argv[5]) : 512;
    const int hint   = (argc > 6) ? std::atoi(argv[6]) : 3;  // NTA
    const int enable = (argc > 7) ? std::atoi(argv[7]) : 1;
    const int tile   = (argc > 8) ? std::atoi(argv[8]) : 256;
    const int reps   = (argc > 9) ? std::atoi(argv[9]) : 20;

    bool check = false;
    for (int i = 1; i < argc; ++i)
        if (std::strcmp(argv[i], "--check") == 0) check = true;

    MatMulFn fn = nullptr;
    if      (std::strcmp(name, "naive")     == 0) fn = matmul_naive;
    else if (std::strcmp(name, "simd")      == 0) fn = matmul_simd;
    else if (std::strcmp(name, "prefetch")  == 0) fn = matmul_prefetch;
    else if (std::strcmp(name, "optimized") == 0) fn = matmul_optimized;
    else { std::fprintf(stderr, "unknown kernel '%s'\n", name); return 1; }

    matmul_tile_m = tile; matmul_tile_n = tile;
    matmul_prefetch_dist = dist;
    matmul_prefetch_hint = hint_arg_to_mm_hint(hint);
    matmul_prefetch_enable = enable;

    matmul_opt_tile_m = tile; matmul_opt_tile_n = tile;
    matmul_optimized_prefetch = enable;
    matmul_optimized_prefetch_dist = dist;
    matmul_optimized_prefetch_hint = hint_arg_to_mm_hint(hint);

    const int lda = K, ldb = K, ldc = N;

    float* A = pa1::alloc_floats(static_cast<std::size_t>(M) * K);
    float* B = pa1::alloc_floats(static_cast<std::size_t>(N) * K);
    float* C = pa1::alloc_floats(static_cast<std::size_t>(M) * N);
    pa1::fill_random(A, static_cast<std::size_t>(M) * K, 1234u);
    pa1::fill_random(B, static_cast<std::size_t>(N) * K, 1235u);

    if (check) {
        float* ref = pa1::alloc_floats(static_cast<std::size_t>(M) * N);
        matmul_naive(A, B, ref, M, N, K, lda, ldb, ldc);
        fn(A, B, C, M, N, K, lda, ldb, ldc);
        const float err = pa1::max_abs_diff(C, ref, static_cast<std::size_t>(M) * N);
        const float ref_mag = pa1::max_abs(ref, static_cast<std::size_t>(M) * N);
        const float tol = 1e-4f * (ref_mag + 1e-30f);
        std::printf(
            "%-9s M=%d N=%d K=%d dist=%d hint=%d enable=%d tile=%d  max_abs_err=%.3g  %s\n",
            name, M, N, K, dist, hint, enable, tile, static_cast<double>(err),
            err <= tol ? "OK" : "FAIL");
        pa1::free_floats(ref);
        pa1::free_floats(A); pa1::free_floats(B); pa1::free_floats(C);
        return (err <= tol) ? 0 : 2;
    }

    // ---- the measured region: nothing here but the kernel ----
    auto run = [&]() { fn(A, B, C, M, N, K, lda, ldb, ldc); };
    const double ms = pa1::time_median_ms(run, 2, reps);
    const double gflops = pa1::matmul_flops(M, N, K) / (ms * 1e6);

    // CSV: kernel,M,N,K,dist,hint,enable,tile,reps,ms,gflops
    std::printf("%s,%d,%d,%d,%d,%d,%d,%d,%d,%.4f,%.3f\n", name, M, N, K, dist, hint, enable,
                tile, reps, ms, gflops);

    pa1::free_floats(A); pa1::free_floats(B); pa1::free_floats(C);
    return 0;
}
