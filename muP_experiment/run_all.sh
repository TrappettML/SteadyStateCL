#!/usr/bin/env bash
# run_all.sh — every experiment script of muP_experiment, in the order of the note
# "μP Experiment with Learned Shared, Frozen Shared & Task-Specific Heads".
#
# Every parameter is written out below (nothing is left to a script default), so
# this file is the one place to check them.  Lines marked CHECK are where the note
# and the code disagree, or where the note leaves a value open.
#
# Usage (from anywhere; it runs in this folder, inside the jax env):
#   ./run_all.sh                          every step of the note, in order
#   ./run_all.sh szero stask              only these steps, in the order given
#   ./run_all.sh --dry-run                print the commands, run nothing
#   ./run_all.sh --list                   the step names
#   PLACE="--workers 48" ./run_all.sh     CPU workers instead of the GPUs ("" = in-process)
#   nohup ./run_all.sh > run_all.out 2>&1 &     detached, e.g. on cyclops
#
# Logs: results/logs/run_all_<date>_<time>/<NN>_<step>.log, and summary.txt with the
# git commit, exit code, wall time and full command of every step.  A failed step is
# reported and the next one still runs; the exit code is 1 if any step failed.

(( BASH_VERSINFO[0] >= 4 )) || { echo "run_all.sh needs bash ≥ 4" >&2; exit 1; }
set -u
set -f                                   # no globbing when the commands are word-split
cd "$(dirname "$0")" || exit 1

# ---------------------------------------------------------------------------
# Shared settings
# ---------------------------------------------------------------------------
PY=${PY:-python}
PLACE=${PLACE---gpus}                    # every card of the node, planned by mup.pool
LAM=4                                    # hidden-layer learning per task λ (configs/stage1.yaml: lam [4.0])
CLOCK=realized                           # task clock: realized = (ii), Λ_c = λ/ā²_c (write-up default); nominal = (i)
STAT=mean                                # half-life: mean = root of the seed-mean curve Σ(m) = ½ (write-up); seed = per-seed mean
SEEDS=20                                 # seeds 0 … SEEDS−1
N_GRID="10 20 50"            # widths of Nscaling, weight_norms, imbalance_comp (D = 4N)
KAPPAS="1 2 4"                           # κ of the same three; the note: "different kappa values (1,2,4)"

declare -A CMD

# ---------------------------------------------------------------------------
# 1. Making sure learning occurs                        scripts/run_training_verify.py
#    Note: "N=100, κ=2, δ0=1; N=50, δ0=0, κ=√N; N=20, δ=4, κ=0; N=200, κ=2, δ=1",
#    loss of the task being trained for the first two, middle two and last two tasks.
#    Cells are N,κ,δ₀.
#    CHECK: κ = 0 has no active units (pN = κ√N = 0), so nothing would train;
#           κ = 1 is used for the N = 20 cell.  The script's own defaults are
#           50,1,0 and 20,1,4; the N = 50 cell is sqrtN here, as in the note.
# ---------------------------------------------------------------------------
CMD[training_verify]="scripts/run_training_verify.py --cell 100,2,1 50,sqrtN,0 20,1,4 200,2,1
    --lam $LAM --seeds $SEEDS --clock $CLOCK"

# ---------------------------------------------------------------------------
# 2. Exact gauge projection                             scripts/run_gauge_comp.py
#    Note: Δ_i against steps (log x), row 1 without, row 2 with the projection;
#    "Make the N = 100 and the κ be 1".  δ₀ is not given in the note: 0.
#    Runs the config's stream, T_burn + T_meas = 1400 tasks at N = 100.
# ---------------------------------------------------------------------------
CMD[gauge_comp]="scripts/run_gauge_comp.py --N 100 --kappa 1 --delta0 0 --lam $LAM --seeds $SEEDS --clock $CLOCK"

# ---------------------------------------------------------------------------
# 3. Half-life of a signal                              scripts/run_szero.py
#    Note: S₀(t) against task t, "fixed N=100, and Δ0 = 0, and κ = 1".
# ---------------------------------------------------------------------------
CMD[szero]="scripts/run_szero.py --N 100 --kappa 1 --delta0 0 --lam $LAM --seeds $SEEDS --tasks 1400
    --clock $CLOCK --stat $STAT"

# ---------------------------------------------------------------------------
# 4. Burn-in vs steady state (sTask)                    scripts/run_stask.py
#    Note: "train for T=1400 ... then goes for τ = 1000 more tasks", same cell as szero.
# ---------------------------------------------------------------------------
CMD[stask]="scripts/run_stask.py --N 100 --kappa 1 --delta0 0 --lam $LAM --seeds $SEEDS --T 1400 --tau 1000
    --clock $CLOCK --stat $STAT"

# ---------------------------------------------------------------------------
# 5. Half-life vs N                                     scripts/run_Nscaling.py
#    Note: train T, follow the memory of task T for T more; m½ against N; dotted
#    theory, solid simulation; one panel per method, κ = 1, 2, 4 as colours.
#    --theory eq draws the half-life of each readout's equation of the write-up,
#    (S_heads), (S_frozen), (S_learned); nln2 draws N ln 2, nk2 N ln 2 / κ².
#    CHECK: the note gives no widths: N_GRID above.
# ---------------------------------------------------------------------------
CMD[Nscaling]="scripts/run_Nscaling.py --N $N_GRID --kappa $KAPPAS --delta0 0 --lam $LAM --seeds $SEEDS
    --theory eq --clock $CLOCK --stat $STAT"

# ---------------------------------------------------------------------------
# 6. Weight norms                                       scripts/run_weight_norms.py
#    Note: mean WWᵀ/D and mean a² (and |u_t|²/D) against task, rows; then against N.
#    CHECK: the note gives no widths or κ: N_GRID and KAPPAS above.  The GPU run of
#           2026-11-06 used N = 10 20 50 100 200, κ = 1 2 4 8 and 20 seeds.
# ---------------------------------------------------------------------------
CMD[weight_norms]="scripts/run_weight_norms.py --N $N_GRID --kappa $KAPPAS --delta0 0 --lam $LAM --seeds $SEEDS
    --clock $CLOCK"

# ---------------------------------------------------------------------------
# 7. Weight imbalance                                   scripts/run_imbalance_comp.py
#    Note: how δ₀ changes the weight norms and the half-life, including negative δ₀,
#    where the input norms exceed the output norms.  Allowed: δ₀ > −1
#    (configs/stage1.yaml uses −0.75 −0.5 0 1 4 16; the script default is 0 1 2 4).
# ---------------------------------------------------------------------------
CMD[imbalance_comp]="scripts/run_imbalance_comp.py --N $N_GRID --kappa $KAPPAS --delta0 -0.5 -0.25 0 1 2 4 --lam $LAM
    --seeds $SEEDS --clock $CLOCK --stat $STAT"

ORDER=(training_verify gauge_comp szero stask Nscaling weight_norms imbalance_comp)   # the note's order

# ---------------------------------------------------------------------------
DRY=0
SEL=()
for a in "$@"; do
    case $a in
        (-n|--dry-run) DRY=1 ;;
        (--list) printf '%s\n' "${ORDER[@]}"; exit 0 ;;
        (-h|--help) sed -n '2,19p' "$0"; exit 0 ;;
        (*) [[ -n ${CMD[$a]+x} ]] || { echo "unknown step: $a (see --list)" >&2; exit 2; }; SEL+=("$a") ;;
    esac
done
(( ${#SEL[@]} )) || SEL=("${ORDER[@]}")

LOGDIR=results/logs/run_all_$(date +%Y%m%d_%H%M%S)
if (( ! DRY )); then
    mkdir -p "$LOGDIR"
    { echo "run_all.sh  $(date)  host $(hostname)"
      echo "git $(git rev-parse --short HEAD 2>/dev/null)$(git diff --quiet HEAD -- . 2>/dev/null || echo ' + uncommitted changes')"
      echo "python $($PY -c 'import sys; print(sys.executable)')"; echo; } > "$LOGDIR/summary.txt"
fi

failed=()
i=0
for s in "${SEL[@]}"; do
    i=$((i + 1))
    cmd=$(echo $PY ${CMD[$s]} $PLACE)            # one line, single spaces
    printf '\n[%d/%d] %s\n  %s\n' "$i" "${#SEL[@]}" "$s" "$cmd"
    (( DRY )) && continue
    log=$(printf '%s/%02d_%s.log' "$LOGDIR" "$i" "$s")
    t0=$SECONDS
    $cmd 2>&1 | tee "$log"
    rc=${PIPESTATUS[0]}
    dt=$((SECONDS - t0))
    printf '%-16s rc=%-3d %4dm%02ds  %s\n' "$s" "$rc" $((dt / 60)) $((dt % 60)) "$cmd" >> "$LOGDIR/summary.txt"
    (( rc == 0 )) || failed+=("$s")
done

if (( ! DRY )); then
    echo; cat "$LOGDIR/summary.txt"
    if (( ${#failed[@]} )); then echo "FAILED: ${failed[*]}  (logs in $LOGDIR)"; exit 1; fi
    echo "all ${#SEL[@]} steps done; logs in $LOGDIR"
fi
