#!/usr/bin/env bash
# sweep_optimized.sh  Task 2C (prefetch + SIMD combined) measurement sweep.  NOT part of
# the submission -- repeats 2A's prefetch analysis on matmul_optimized (the combined
# kernel: register-tiled AVX2 SIMD + M/N cache tiling + optional software prefetch),
# per the assignment's "repeat all the 2A/2B analyses on the combined kernel" requirement.
#
#   ./sweep_optimized.sh [outdir]
#
# Writes three CSVs into outdir (default: this directory):
#   opt_size_sweep.csv     matmul_optimized WITH vs WITHOUT software prefetch, across
#                           matrix sizes. Also includes matmul_naive and matmul_simd rows
#                           at each size as reference points for the final comparison plot.
#   opt_distance_sweep.csv fixed size, hint, tile; sweep prefetch distance
#   opt_hint_sweep.csv     fixed size, distance, tile; sweep locality hint (cache fill level)
#
# Columns (all files): kernel,M,N,K,dist,hint,enable,tile,reps,ms,gflops,
#                       instructions,l1_misses,l2_misses,llc_misses,sw_prefetch_access
#
# Each row is `perf stat` over one bench_matmul invocation, which itself runs the kernel
# `reps` times in a loop (see bench_matmul.cpp) so the counters are attributed to the
# kernel, not to process startup.
set -u
cd "$(dirname "$0")"
OUTDIR="${1:-.}"
mkdir -p "$OUTDIR"
REPS="${REPS:-5}"

# This machine shows large run-to-run variance (turbo/thermal state), up to 30-40% on an
# otherwise-identical config -- comparable to the effect sizes this sweep is trying to
# measure. A single perf-wrapped invocation is NOT trustworthy for timing here, so timing
# is instead the MEDIAN of TIMING_REPEATS independent (no-perf) process invocations; the
# perf-wrapped invocation is used only for the (much less noisy, event-count-based)
# hardware counters, not for its own wall-clock number.
TIMING_REPEATS="${TIMING_REPEATS:-5}"

median() {
    python3 -c "
import sys
v = sorted(float(x) for x in sys.argv[1:])
n = len(v)
print(v[n // 2] if n % 2 else (v[n // 2 - 1] + v[n // 2]) / 2)
" "$@"
}

[ -x ./bench_matmul ] || make -s || { echo "build failed" >&2; exit 1; }

command -v perf >/dev/null 2>&1 || {
    echo "ERROR: perf is not installed (try: sudo apt install linux-tools-$(uname -r))" >&2
    exit 1
}

# ---- hybrid-CPU aware event + core selection (same approach as task1/bench/sweep.sh) ---
PMU=""
PIN=""
CPU_CORE_SYS="${CPU_CORE_SYS:-/sys/devices/cpu_core}"
if [ -r "$CPU_CORE_SYS/cpus" ]; then
    PMU="cpu_core/"
    PCORES=$(cat "$CPU_CORE_SYS/cpus")
    PCORE0=${PCORES%%,*}; PCORE0=${PCORE0%%-*}
    command -v taskset >/dev/null 2>&1 && PIN="taskset -c $PCORE0"
    echo "hybrid CPU detected: using ${PMU} PMU, pinning with: ${PIN:-<none>}" >&2
fi
ev() { [ -n "$PMU" ] && echo "${PMU}$1/" || echo "$1"; }

have_event() {
    local v
    v=$($PIN perf stat -x, -e "$1" ./bench_matmul naive 128 128 128 0 0 0 0 3 2>&1 \
        | grep ",$1," | head -1 | cut -d, -f1)
    case "$v" in ''|*[!0-9]*) return 1 ;; *) return 0 ;; esac
}

EV_INSTR=$(ev instructions)
have_event "$EV_INSTR" || {
    echo "ERROR: could not count instructions (got nothing usable)." >&2
    echo "  paranoid level: $(cat /proc/sys/kernel/perf_event_paranoid)" >&2
    exit 1
}

EV_L1=""
for cand in "$(ev L1-dcache-load-misses)" "$(ev MEM_LOAD_RETIRED.L1_MISS)"; do
    have_event "$cand" && { EV_L1="$cand"; break; }
done
[ -n "$EV_L1" ] || { echo "ERROR: no usable L1-D miss event." >&2; exit 1; }

EV_L2=""
for cand in "$(ev l2_request.miss)" "$(ev l2_rqsts.miss)"; do
    have_event "$cand" && { EV_L2="$cand"; break; }
done
[ -n "$EV_L2" ] || echo "WARNING: no usable L2-miss event; column will be empty." >&2

EV_LLC=""
for cand in "$(ev llc-load-misses)" "$(ev LLC-load-misses)"; do
    have_event "$cand" && { EV_LLC="$cand"; break; }
done
[ -n "$EV_LLC" ] || echo "WARNING: no usable LLC-miss event; column will be empty." >&2

EV_SWPF=""
for cand in "$(ev sw_prefetch_access.any)"; do
    have_event "$cand" && { EV_SWPF="$cand"; break; }
done
[ -n "$EV_SWPF" ] || echo "WARNING: no usable sw_prefetch_access event; column will be empty." >&2

EVENTS="$EV_INSTR,$EV_L1"
[ -n "$EV_L2" ]   && EVENTS="$EVENTS,$EV_L2"
[ -n "$EV_LLC" ]  && EVENTS="$EVENTS,$EV_LLC"
[ -n "$EV_SWPF" ] && EVENTS="$EVENTS,$EV_SWPF"
echo "using perf events: $EVENTS" >&2

HEADER="kernel,M,N,K,dist,hint,enable,tile,reps,ms,gflops,instructions,l1_misses,l2_misses,llc_misses,sw_prefetch_access"

# $1 kernel  $2 M  $3 N  $4 K  $5 dist  $6 hint  $7 enable  $8 tile
measure() {
    # ---- timing: median of TIMING_REPEATS independent, perf-free invocations ----
    local ms_vals=() gf_vals=() out
    for _ in $(seq "$TIMING_REPEATS"); do
        out=$($PIN ./bench_matmul "$1" "$2" "$3" "$4" "$5" "$6" "$7" "$8" "$REPS" 2>/dev/null)
        [ -z "$out" ] && { echo "ERROR: bench_matmul produced no output for: $*" >&2; return 1; }
        ms_vals+=("$(echo "$out" | cut -d, -f10)")
        gf_vals+=("$(echo "$out" | cut -d, -f11)")
    done
    local ms gf
    ms=$(median "${ms_vals[@]}")
    gf=$(median "${gf_vals[@]}")

    # ---- counters: ONE perf-wrapped invocation (event counts are far less noisy than
    # wall-clock time on this machine, so a single sample is fine here) ----
    local perf_out bench_out
    perf_out=$(mktemp); bench_out=$(mktemp)
    $PIN perf stat -x, -o "$perf_out" -e "$EVENTS" \
        ./bench_matmul "$1" "$2" "$3" "$4" "$5" "$6" "$7" "$8" "$REPS" >"$bench_out" 2>/dev/null

    local instr l1 l2 llc swpf
    instr=$(grep ",$EV_INSTR,"  "$perf_out" | head -1 | cut -d, -f1)
    l1=$(grep ",$EV_L1,"        "$perf_out" | head -1 | cut -d, -f1)
    [ -n "$EV_L2" ]   && l2=$(grep ",$EV_L2,"    "$perf_out" | head -1 | cut -d, -f1)   || l2=""
    [ -n "$EV_LLC" ]  && llc=$(grep ",$EV_LLC,"  "$perf_out" | head -1 | cut -d, -f1)   || llc=""
    [ -n "$EV_SWPF" ] && swpf=$(grep ",$EV_SWPF," "$perf_out" | head -1 | cut -d, -f1)  || swpf=""

    case "$instr" in
        ''|*[!0-9]*)
            echo "ERROR: perf returned no instruction count for: $*" >&2
            rm -f "$perf_out" "$bench_out"; return 1
            ;;
    esac
    # CSV: kernel,M,N,K,dist,hint,enable,tile,reps,ms,gflops,instructions,l1,l2,llc,swpf
    echo "$1,$2,$3,$4,$5,$6,$7,$8,${REPS},${ms},${gf},${instr},${l1},${l2},${llc},${swpf}"
    rm -f "$perf_out" "$bench_out"
}

# ---- 1. size sweep: naive vs prefetch-off vs prefetch-on, across sizes -----------------
SIZES="${SIZES:-128 256 512 1024 1536 2048}"
BEST_DIST="${BEST_DIST:-512}"
BEST_HINT="${BEST_HINT:-3}"   # NTA
TILE="${TILE:-256}"

if [ "${SKIP_SIZE_SWEEP:-0}" = "1" ]; then
    echo "SKIP_SIZE_SWEEP=1: leaving existing size_sweep.csv alone" >&2
else
    OUT1="$OUTDIR/opt_size_sweep.csv"
    echo "$HEADER" >"$OUT1"
    for N in $SIZES; do
        echo "opt size sweep: N=$N naive"          >&2
        measure naive     "$N" "$N" "$N" 0 0 0 0            >>"$OUT1"
        echo "opt size sweep: N=$N simd (unblocked)" >&2
        measure simd      "$N" "$N" "$N" 0 0 0 0            >>"$OUT1"
        echo "opt size sweep: N=$N optimized, prefetch OFF" >&2
        measure optimized "$N" "$N" "$N" "$BEST_DIST" "$BEST_HINT" 0 "$TILE" >>"$OUT1"
        echo "opt size sweep: N=$N optimized, prefetch ON"  >&2
        measure optimized "$N" "$N" "$N" "$BEST_DIST" "$BEST_HINT" 1 "$TILE" >>"$OUT1"
    done
fi

# ---- 2. prefetch-distance sweep at a fixed size ----------------------------------------
DIST_N="${DIST_N:-2048}"
DISTS="${DISTS:-8 16 32 64 128 256 512 1024}"

OUT2="$OUTDIR/opt_distance_sweep.csv"
echo "$HEADER" >"$OUT2"
for D in $DISTS; do
    echo "opt distance sweep: dist=$D" >&2
    measure optimized "$DIST_N" "$DIST_N" "$DIST_N" "$D" "$BEST_HINT" 1 "$TILE" >>"$OUT2"
done

# ---- 3. locality-hint (cache fill level) sweep at a fixed size -------------------------
HINT_N="${HINT_N:-2048}"
HINTS="${HINTS:-0 1 2 3}"   # T0 T1 T2 NTA

OUT3="$OUTDIR/opt_hint_sweep.csv"
echo "$HEADER" >"$OUT3"
for H in $HINTS; do
    echo "opt hint sweep: hint=$H" >&2
    measure optimized "$HINT_N" "$HINT_N" "$HINT_N" "$BEST_DIST" "$H" 1 "$TILE" >>"$OUT3"
done

echo "done: $OUT1 $OUT2 $OUT3" >&2
