# Task 1 benchmark results (this machine: 13th Gen Intel i5-13420H, hybrid P+E core)

Raw data + derived summaries for writing the Task 1 report. All measurements taken
with `bench/sweep.sh` (perf pinned to a P-core, cpu_core PMU) and `bin/conv`
(the graded harness). AVX-512 is NOT available on this CPU (avx2+fma only).

## Files
- `sweep_full.csv` / `sweep_full.log` — the complete raw sweep: naive, reorder,
  unroll, tile (6 tile sizes), simd/simd128/simd256, optimized, and the
  tile x SIMD-width x unroll `combo` grid, for H=W in {512,1024,2048} and K in {3,5}.
- `1A_reorder_unroll.csv`, `1A_summary.txt` — Task 1A data: naive vs reorder vs
  unroll (time, GFLOP/s, instruction count, speedup).
- `1A_conv_all_*.txt`, `1A_multi_seed_check.txt` — correctness spot-checks from
  `bin/conv` across sizes and RNG seeds, for the "how effective/considerations" writeup.
- `1B_tiling_mpki.csv`, `1B_summary.txt` — Task 1B data: naive vs every tile size,
  L1-D MPKI and speedup per (matrix size, K). `1B_cpu_topology.txt` has the L1-D
  cache facts (48 KiB/12-way/64B lines on the P-core, confirmed via pinned getconf
  and /sys/devices/system/cpu/cpu0/cache).
- `1C_simd_instructions.csv`, `1C_summary.txt` — Task 1C data: naive vs simd
  (AVX2/256, the graded kernel) vs simd128/simd256 width study. Instruction counts
  and speedup per (matrix size, K).
- `1D_all_stages.csv`, `1D_combo_grid.csv`, `1D_summary.txt` — Task 1D data:
  every technique's best speedup vs naive side by side, plus the combo grid
  (tile x SIMD width x unroll) and a synergy check (optimized vs. naive-additive
  reference) per (matrix size, K).
- `1B_l2_llc_hierarchy.csv` — L1/L2/LLC miss counts (not just L1) for naive,
  reorder, unroll, tile (6 sizes), simd, optimized at the graded 2048x2048 K=3
  workload. This is the evidence for *why* tiling helps: see the note in
  `1B_summary.txt`.
- `graded_run.txt` — output of `./bin/conv` (no args): correctness, per-stage
  speedup table, and autograder score on this machine.

## Data-quality notes (read before writing the report)
- All correctness claims were re-verified with `--check` directly (not just
  inferred from `bin/conv`): naive/reorder/unroll/tile/simd/optimized AND the
  bench-only variants actually used to source these numbers (simd128, simd256,
  and several combo tile/width/unroll configs) all pass (max abs err <~2e-6)
  on this machine across multiple sizes and both K=3 and K=5.
- `sweep_full.csv`'s `instructions` column is the TOTAL for the whole `bench`
  process, which calls the kernel `warmup(2) + reps(20) = 22` times (setup is
  one-time and negligible). `1A_summary.txt` and `1C_summary.txt` divide by 22
  to report a true per-call instruction count; `1B`/`1D`/`sweep_full.csv` still
  carry the raw (x22) totals since only the *ratios* there are used. Any other
  script pulling absolute instruction counts from `sweep_full.csv` directly
  must apply the same /22 correction.
- conv_optimized.cpp's hardcoded 64x256 tile is NOT a leftover from the old
  laptop: re-tested head-to-head against several square tile sizes (16 through
  1024) with interleaved timing on this machine, and 64x256 wins (~7.27ms vs
  8.2-10.3ms at 128x128, the next best) at the graded 2048x2048 K=3 workload.
  Do not "fix" this in the report -- the smaller tile sizes that win in
  `1B_tiling_mpki.csv` are for the plain *scalar* tile kernel; once SIMD+unroll
  cut the per-element work, a bigger tile becomes optimal again.

## Headline numbers to build the report around
- Autograder score on this machine: 70/100 (40/40 correctness, 30/60 speedup
  points -- optimized reaches ~4.0-4.7x at K=3, autograder tiers need 6x/8x for
  more points).
- 1A: reorder/unroll roughly break even or LOSE to naive at 2048x2048 (0.66-0.72x)
  -- matches the intended lesson that an untiled K*K reorder is DRAM-bandwidth
  bound at that size; gains show up more at 512x512 (up to 1.39x).
- 1B: L1-D = 48 KiB/core. Naive's L1 MPKI is already very low (<1) because its
  working set is a 3-row sliding strip that fits L1 regardless of image size.
  L1 MPKI alone is misleading, though: L2/LLC counters (1B_l2_llc_hierarchy.csv)
  show untiled reorder pushes ~55M LLC misses at 2048x2048 vs naive's ~266K,
  while tiling at 16x16 brings LLC misses back down to ~341K (near-naive) by
  keeping the tile+halo resident in L2 instead of spilling to DRAM -- THAT is
  the real mechanism, not raw L1-miss count. Best plain-scalar tile sizes on
  this machine: 16x16 (512x512, 2048x2048 K=3) and 32x32 (1024x1024, K=5 cases).
- 1C: SIMD gives an 8.6-9.1x reduction in instruction count (K=3) vs naive but
  speedup shrinks from ~3.2x at 512x512 to ~1.1-1.2x at 2048x2048 -- the untiled
  SIMD kernel becomes memory-bound at scale, same story as 1A.
- 1D: at 2048x2048, no single technique alone beats ~1.24x, yet conv_optimized
  reaches 4.0-4.7x -- strongly superlinear vs. the naive-additive reference
  (0.69-0.86x), a clean synergy result for the "sabka saath sabka vikaas" writeup.
