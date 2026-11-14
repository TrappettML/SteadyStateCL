#!/usr/bin/env bash
# run_all.sh — every experiment script of replay_experiment.
#
# Every parameter is written out below (nothing is left to a script default), so
# this file is the one place to check them.
#
# Usage (from anywhere; it runs in this folder, inside the jax env):
#   ./run_all.sh                          every step, in order
#   ./run_all.sh stask inter_vs_sleep     only these steps, in the order given
#   ./run_all.sh --dry-run                print the commands, run nothing
#   ./run_all.sh --list                   the step names
#   PLACE="--workers 48" ./run_all.sh     CPU workers instead of the GPUs ("" = in-process)
#   MODE=interleaved ./run_all.sh         interleaved instead of sleep replay
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
PLACE=${PLACE---gpus}                    # every card of the node, planned by sscl_core.pool
MODE=${MODE:-sleep}                      # replay schedule: sleep (default) | interleaved
LAM=4                                    # hidden-layer learning per task λ (configs/replay.yaml: lam [4.0])
STAT=mean                                # half-life: mean = first crossing of ½ by the seed-mean curve; seed = per-seed mean
SEEDS=8                                  # seeds 0 … SEEDS−1
N=100                                    # width of the single-cell scripts (D = 4N)
N_GRID="10 20 50 100"                    # widths of Nscaling and weight_norms
CAPS="0.5D 4D"                           # replay sizes C: one small, one large (multiples of D)
WRITE=0.125D                             # samples of each task stored after it
RATIO=1                                  # replayed samples per fresh sample
REPLAY="--capacity $CAPS --method none fifo random priority --write $WRITE --replay-ratio $RATIO"

declare -A CMD

# 1. Is every task learned under replay: loss of the task being trained, first / middle / last two tasks.
#    Cells are N,C,δ₀; the stream is the config's T_burn + T_meas = 14N tasks.
CMD[training_verify]="scripts/run_training_verify.py --cell $N,0.5D,0 $N,4D,0 --method none fifo random priority
    --write $WRITE --replay-ratio $RATIO --mode $MODE --lam $LAM --seeds $SEEDS"

# 2. Signal of the first task from a fresh start and an empty buffer, S₀(t), for 14N tasks.
CMD[szero]="scripts/run_szero.py --N $N $REPLAY --mode $MODE --delta0 0 --lam $LAM --seeds $SEEDS --tasks 1400 --stat $STAT"

# 3. Signal of a steady-state task, S_T(τ): train T = 14N tasks, follow task T for 200 more.
CMD[stask]="scripts/run_stask.py --N $N $REPLAY --mode $MODE --delta0 0 --lam $LAM --seeds $SEEDS --T 1400 --tau 200 --stat $STAT"

# 4. Steady-state half-life against N: train 14N tasks, follow for 2N (the buffer holds at most 32 tasks).
CMD[Nscaling]="scripts/run_Nscaling.py --N $N_GRID $REPLAY --mode $MODE --delta0 0 --lam $LAM --seeds $SEEDS
    --T 14 --tau 2 --stat $STAT"

# 5. Area under the current task's loss curve against task, 14N tasks.
CMD[loss_slope]="scripts/run_loss_slope.py --N $N $REPLAY --mode $MODE --delta0 0 --lam $LAM --seeds $SEEDS --tasks 1400"

# 6. Weight norms, end-of-task loss and buffer age against task and against N, 14N tasks.
CMD[weight_norms]="scripts/run_weight_norms.py --N $N_GRID $REPLAY --mode $MODE --delta0 0 --lam $LAM --seeds $SEEDS --tasks 14"

# 7. Interleaved against sleep replay: box plots of the per-seed half-life per method (runs both modes;
#    more seeds, since a box needs a distribution).
CMD[inter_vs_sleep]="scripts/run_inter_vs_sleep.py --N $N $REPLAY --delta0 0 --lam $LAM --seeds 20 --T 1400 --tau 200"

ORDER=(training_verify szero stask Nscaling loss_slope weight_norms inter_vs_sleep)

# ---------------------------------------------------------------------------
DRY=0
SEL=()
for a in "$@"; do
    case $a in
        (-n|--dry-run) DRY=1 ;;
        (--list) printf '%s\n' "${ORDER[@]}"; exit 0 ;;
        (-h|--help) sed -n '2,17p' "$0"; exit 0 ;;
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
