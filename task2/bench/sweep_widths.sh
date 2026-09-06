#!/usr/bin/env bash
# sweep_widths.sh  Task 2B (SIMD width) measurement sweep.  NOT part of the submission.
#
#   ./sweep_widths.sh [outdir]
#
# Writes outdir/width_sweep.csv:
#   width,M,N,K,reps,ms,gflops,instructions
#
# One row per (width, size). A width row is silently skipped if the CPU lacks the ISA
# extension (bench_widths exits 77) -- e.g. AVX-512F is absent on most hybrid P/E-core
# Intel parts, which is itself a result worth reporting for 2B's write-up.
set -u
cd "$(dirname "$0")"
OUTDIR="${1:-.}"
mkdir -p "$OUTDIR"
REPS="${REPS:-15}"

[ -x ./bench_widths ] || make -s bench_widths || { echo "build failed" >&2; exit 1; }
command -v perf >/dev/null 2>&1 || { echo "ERROR: perf not installed" >&2; exit 1; }

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
    v=$($PIN perf stat -x, -e "$1" ./bench_widths 256 128 128 128 3 2>&1 \
        | grep ",$1," | head -1 | cut -d, -f1)
    case "$v" in ''|*[!0-9]*) return 1 ;; *) return 0 ;; esac
}
EV_INSTR=$(ev instructions)
have_event "$EV_INSTR" || { echo "ERROR: cannot count instructions" >&2; exit 1; }

OUT="$OUTDIR/width_sweep.csv"
echo "width,M,N,K,reps,ms,gflops,instructions" >"$OUT"

SIZES="${SIZES:-128 256 512 1024 2048}"
WIDTHS="${WIDTHS:-128 256 512}"

for N in $SIZES; do
    for W in $WIDTHS; do
        echo "width sweep: width=$W N=$N" >&2
        perf_out=$(mktemp); bench_out=$(mktemp)
        $PIN perf stat -x, -o "$perf_out" -e "$EV_INSTR" \
            ./bench_widths "$W" "$N" "$N" "$N" "$REPS" >"$bench_out" 2>/dev/null
        rc=$?
        csv=$(cat "$bench_out")
        if [ -z "$csv" ]; then
            echo "  skipped (width $W unsupported or errored, rc=$rc)" >&2
            rm -f "$perf_out" "$bench_out"; continue
        fi
        instr=$(grep ",$EV_INSTR," "$perf_out" | head -1 | cut -d, -f1)
        echo "${csv},${instr}" >>"$OUT"
        rm -f "$perf_out" "$bench_out"
    done
done
echo "done: $OUT" >&2
