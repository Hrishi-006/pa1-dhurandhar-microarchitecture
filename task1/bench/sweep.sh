#!/usr/bin/env bash
# sweep.sh  run the tile-size / matrix-size sweep under perf and emit one CSV.
# NOT part of the submission.
#
#   ./sweep.sh [outfile.csv]
#
# Output columns: kernel,H,W,K,TH,TW,reps,ms,gflops,instructions,l1_loads,l1_misses,mpki
# MPKI = L1-D load misses per 1000 instructions.
set -u

cd "$(dirname "$0")"
OUT="${1:-sweep.csv}"
REPS="${REPS:-20}"

SIZES="${SIZES:-512 1024 2048}"          # square matrices (H=W)
KS="${KS:-3 5}"                          # kernel sizes
TILES="${TILES:-16 32 64 128 256 512}"   # square-ish tiles: TH=TW=t
KERNELS_ONLY="${KERNELS_ONLY:-}"         # set to skip the plain tile sweep
WIDTHS="${WIDTHS:-0 128 256 512}"        # combo axis 2: 0 = scalar
UNROLLS="${UNROLLS:-1 2}"                # combo axis 3: vectors per iteration

# ---- locate the binary -------------------------------------------------------
[ -x ./bench ] || make -s || { echo "build failed" >&2; exit 1; }

# ---- figure out which perf events actually exist on this CPU -----------------
have_event() { perf stat -e "$1" -x, true 2>&1 | grep -q "not supported" && return 1 || return 0; }

command -v perf >/dev/null 2>&1 || {
    echo "ERROR: perf is not installed (try: sudo apt install linux-tools-\$(uname -r))" >&2
    exit 1
}

# perf exits 0 even when it is denied access, so check that a real NUMBER came back
# rather than trusting the exit status.  Otherwise the whole sweep runs and silently
# writes empty counter columns.
probe=$(perf stat -x, -e instructions true 2>&1 | grep ",instructions" | cut -d, -f1)
case "$probe" in
    ''|*[!0-9]*)
        echo "ERROR: perf ran but returned no counter value (got: '${probe:-nothing}')." >&2
        echo "  If it said 'Access to performance monitoring operations is limited':" >&2
        echo "      sudo sysctl -w kernel.perf_event_paranoid=1" >&2
        echo "  To make that persist across reboots:" >&2
        echo "      echo 'kernel.perf_event_paranoid=1' | sudo tee /etc/sysctl.d/99-perf.conf" >&2
        echo "  If events read '<not supported>', this machine has no PMU access" >&2
        echo "  (VM/WSL) -- fall back to: valgrind --tool=cachegrind" >&2
        exit 1
        ;;
esac
echo "perf probe OK (instructions=$probe on a trivial process)" >&2

EV_INSTR="instructions"
EV_LOADS="L1-dcache-loads"
EV_MISS="L1-dcache-load-misses"
have_event "$EV_LOADS" || EV_LOADS=""
have_event "$EV_MISS"  || EV_MISS="MEM_LOAD_RETIRED.L1_MISS"
have_event "$EV_MISS"  || { echo "ERROR: no usable L1-D miss event; fall back to cachegrind." >&2; exit 1; }

EVENTS="$EV_INSTR,$EV_MISS"
[ -n "$EV_LOADS" ] && EVENTS="$EV_INSTR,$EV_LOADS,$EV_MISS"
echo "using perf events: $EVENTS" >&2

# ---- one measured run --------------------------------------------------------
# $1 kernel  $2 H  $3 W  $4 K  $5 TH  $6 TW
measure() {
    local perf_out bench_out
    perf_out=$(mktemp); bench_out=$(mktemp)

    perf stat -x, -o "$perf_out" -e "$EVENTS" \
        ./bench "$1" "$2" "$3" "$4" "$5" "$6" "$REPS" >"$bench_out" 2>/dev/null

    # perf -x, lines look like:  value,unit,event,run_time,pct
    local instr l1l l1m
    instr=$(grep ",$EV_INSTR," "$perf_out" | head -1 | cut -d, -f1)
    l1m=$(grep ",$EV_MISS,"   "$perf_out" | head -1 | cut -d, -f1)
    if [ -n "$EV_LOADS" ]; then
        l1l=$(grep ",$EV_LOADS," "$perf_out" | head -1 | cut -d, -f1)
    else
        l1l=""
    fi

    local csv mpki
    csv=$(cat "$bench_out")
    if [ -z "$csv" ]; then          # bench exited 77 (unsupported ISA) or errored
        rm -f "$perf_out" "$bench_out"
        return 1
    fi
    case "$instr" in                # never write a row with empty counters
        ''|*[!0-9]*)
            echo "ERROR: perf returned no instruction count for '$1' (got '${instr:-nothing}')." >&2
            echo "       aborting: counters are required. See the perf notes above." >&2
            rm -f "$perf_out" "$bench_out"
            exit 1
            ;;
    esac
    mpki=$(awk -v m="$l1m" -v i="$instr" 'BEGIN{ if (i+0>0) printf "%.4f", 1000*m/i; else printf "" }')
    echo "$csv,${instr},${l1l},${l1m},${mpki}"

    rm -f "$perf_out" "$bench_out"
}

# ---- the sweep ---------------------------------------------------------------
echo "kernel,H,W,K,TH,TW,width,unroll,reps,ms,gflops,instructions,l1_loads,l1_misses,mpki" >"$OUT"

for K in $KS; do
  for N in $SIZES; do
      echo "  naive   N=$N K=$K" >&2
      measure naive "$N" "$N" "$K" 0 0 >>"$OUT"

      for base in reorder unroll simd simd128 simd256 simd512 optimized; do
          echo "  $base N=$N K=$K" >&2
          measure "$base" "$N" "$N" "$K" 0 0 >>"$OUT" || echo "    (skipped)" >&2
      done

      # ---- combo grid: tile x width x unroll (the synergy study) ----
      for W_BITS in $WIDTHS; do
        for U in $UNROLLS; do
          for t in $TILES 0; do          # t=0 means "no tiling", for the baseline row
              [ "$t" -gt "$N" ] && continue
              echo "    combo N=$N K=$K T=$t w=$W_BITS u=$U" >&2
              export WIDTH="$W_BITS" UNROLL="$U"
              measure combo "$N" "$N" "$K" "$t" "$t" >>"$OUT" \
                  || echo "      (skipped)" >&2
              unset WIDTH UNROLL
          done
        done
      done

      [ -n "$KERNELS_ONLY" ] && continue
      for t in $TILES; do
          [ "$t" -gt "$N" ] && continue
          echo "  tile    N=$N K=$K T=$t" >&2
          measure tile "$N" "$N" "$K" "$t" "$t" >>"$OUT"
      done
  done
done

echo "wrote $OUT" >&2
