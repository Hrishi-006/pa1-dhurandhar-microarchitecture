// bench_widths.cpp  standalone driver for the SIMD-width study (Task 2B).  NOT part of
// the submission.
//
//   ./bench_widths <width> [M] [N] [K] [reps] [--check]
//
//   width: 128 | 256 | 512
//
// Prints one CSV line: width,M,N,K,reps,ms,gflops
// Exit code 77 means "skip" (this CPU lacks the required ISA extension).

#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "matmul.h"
#include "matmul_widths.h"
#include "timer.h"
#include "utils.h"

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s <128|256|512> [M] [N] [K] [reps] [--check]\n",
                     argv[0]);
        return 1;
    }
    const int width = std::atoi(argv[1]);
    const int M     = (argc > 2) ? std::atoi(argv[2]) : 1024;
    const int N     = (argc > 3) ? std::atoi(argv[3]) : 1024;
    const int K     = (argc > 4) ? std::atoi(argv[4]) : 1024;
    const int reps  = (argc > 5) ? std::atoi(argv[5]) : 20;
    bool check = false;
    for (int i = 1; i < argc; ++i)
        if (std::strcmp(argv[i], "--check") == 0) check = true;

    MatMulFn fn = nullptr;
    if (width == 128) {
        if (!cpu_has_sse42()) { std::fprintf(stderr, "skip: no SSE4.2\n"); return 77; }
        fn = matmul_simd128;
    } else if (width == 256) {
        if (!(cpu_has_avx2() && cpu_has_fma())) {
            std::fprintf(stderr, "skip: no AVX2+FMA\n");
            return 77;
        }
        fn = matmul_simd256;
    } else if (width == 512) {
        if (!cpu_has_avx512f()) { std::fprintf(stderr, "skip: no AVX-512F\n"); return 77; }
        fn = matmul_simd512;
    } else {
        std::fprintf(stderr, "unknown width '%s' (want 128|256|512)\n", argv[1]);
        return 1;
    }

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
        std::printf("width=%-3d M=%d N=%d K=%d  max_abs_err=%.3g  %s\n", width, M, N, K,
                    static_cast<double>(err), err <= tol ? "OK" : "FAIL");
        pa1::free_floats(ref);
        pa1::free_floats(A); pa1::free_floats(B); pa1::free_floats(C);
        return (err <= tol) ? 0 : 2;
    }

    auto run = [&]() { fn(A, B, C, M, N, K, lda, ldb, ldc); };
    const double ms = pa1::time_median_ms(run, 2, reps);
    const double gflops = pa1::matmul_flops(M, N, K) / (ms * 1e6);
    std::printf("%d,%d,%d,%d,%d,%.4f,%.3f\n", width, M, N, K, reps, ms, gflops);

    pa1::free_floats(A); pa1::free_floats(B); pa1::free_floats(C);
    return 0;
}
