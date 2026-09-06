# Task 2 benchmark results (this machine: hybrid Intel P/E-core CPU)

Raw data for the Task 2 report (`task2/report/task2_report.pdf`). All measurements taken
with `bench/sweep_prefetch.sh`, `bench/sweep_optimized.sh`, `bench/sweep_widths.sh` (perf
pinned to a P-core, `cpu_core` PMU) and `bin/matmul` (the graded harness). AVX-512 is NOT
available on this CPU (avx2+fma only).

## Files
- `size_sweep.csv` — Task 2A: `matmul_naive` vs `matmul_prefetch` (software prefetch on
  and off), across matrix sizes 128-2048. Columns include time, GFLOP/s, instruction
  count, and L1D/L2/LLC miss counts plus `sw_prefetch_access`.
- `distance_sweep.csv` — Task 2A: `matmul_prefetch` at `N=2048`, sweeping the prefetch
  distance (8-1024 floats ahead), hint fixed at NTA.
- `hint_sweep.csv` — Task 2A: `matmul_prefetch` at `N=2048`, sweeping the locality hint
  (T0/T1/T2/NTA), distance fixed at 512.
- `width_sweep.csv` — Task 2B: 128-bit (SSE4.2) vs 256-bit (AVX2) un-blocked SIMD kernels
  across sizes, with instruction counts. 512-bit (AVX-512F) rows are absent by design —
  `bench_widths` detects the CPU lacks AVX-512F and skips it (exit code 77), not a
  measurement failure.
- `opt_size_sweep.csv` — Task 2C: repeats `size_sweep.csv` on `matmul_optimized` (the
  combined kernel), also including a `matmul_simd` (un-blocked SIMD only) reference row
  at each size for the final cross-technique comparison.
- `opt_distance_sweep.csv`, `opt_hint_sweep.csv` — Task 2C: repeats the 2A distance/hint
  sweeps on `matmul_optimized`.

## Data-quality notes (read before reusing these numbers)
- This machine shows substantial run-to-run timing variance (turbo/thermal state), up to
  30-40% for an identical configuration measured twice. Every `gflops`/`ms` value in
  these CSVs is the **median of 5 independent process invocations** (see
  `bench/sweep_*.sh`'s `measure()`), not a single timed run — a single run is NOT
  trustworthy for the fine-grained comparisons this data supports (prefetch distance,
  locality hint). Hardware counter columns (instructions, misses,
  `sw_prefetch_access`) are a single `perf stat`-wrapped sample per row; these are far
  less sensitive to the timing noise since they count retired events rather than
  depending on clock speed.
- Two real bugs were found and fixed while building this data, both worth knowing if
  regenerating it: (1) the bench harness's CLI hint-argument mapping did not match the
  real `_MM_HINT_*` enum values (T0=3, NTA=0 in `immintrin.h`, not 0/3) — early sweeps
  silently measured the wrong hint; (2) the `measure()` timing loop in the sweep scripts
  originally ran the bench binary unpinned (only the `perf`-wrapped counter call was
  pinned to the P-core) — this alone explained most of the apparent "noise" chased
  early on. Both are fixed in the committed `sweep_*.sh`.
- The hardware-prefetcher on/off comparison (`wrmsr -p 0 0x1a4`, MSR bits 0-3) is NOT in
  a CSV here — it needs root and was done as a one-off, time-bracketed manual
  measurement; see the report's Task 2A section for the numbers and the methodological
  note about why the *first* attempt at it was invalid (thermal drift across a large time
  gap between the two states, not a real prefetcher effect).

## Headline numbers (see the report for full discussion)
- **2A**: software prefetch is a net LOSS on `matmul_prefetch` at every size tested
  (6-24% slower than the same tiled kernel with prefetch calls removed). Best hint: NTA;
  worst: T0 (L1 pollution, since this kernel's accumulators round-trip through an
  array/L1 — see 2C). Best distance: ~128-256 floats.
- **2B**: 256-bit issues ~half the instructions of 128-bit but only gains
  ~1.3-1.5x wall-clock speedup (sub-linear returns to width). AVX-512 unavailable on
  this CPU.
- **2C**: the single biggest win in the whole task was NOT prefetching — it was fixing
  an accumulator-array register-spill bug in the micro-kernel (~1.5-2.5x on its own).
  On the fixed (register-resident) kernel, the hint ranking REVERSES: T0 becomes best,
  not worst, because nothing of the kernel's own state competes with it for L1 space
  anymore. Tuned config (T0, dist=128) beats no-prefetch by 6-12% across sizes and ships
  as the default in `src/matmul_optimized.cpp`.
