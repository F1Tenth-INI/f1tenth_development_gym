#!/bin/bash
# acados at the plot setting: seed 0 first (builds the mu 0.5 solver), then seeds 1-4 in parallel
cd "$(dirname "$0")"
export HW_GYM=/home/emilio/projects/bachelor/f1tenth_development_gym
python hw50_tune.py sweep acados_mu05.json acados_mu05 1 1000 1 > acados_mu05_s0.log 2>&1 &
until grep -aq "BENCH_LAP\|Traceback" runs/acados_mu05/acados/seed0/run.log 2>/dev/null; do sleep 5; done
HW_SEED_OFFSET=1 python hw50_tune.py sweep acados_mu05.json acados_mu05 4 1000 4 > acados_mu05.log 2>&1
wait
