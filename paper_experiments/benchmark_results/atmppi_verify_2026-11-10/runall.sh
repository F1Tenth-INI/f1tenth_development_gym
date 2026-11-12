#!/bin/bash
V=/tmp/claude-1000/-home-emilio-projects-bachelor/0207f9b3-5363-404f-bde5-16d43ddd5f7a/scratchpad/verify
PY=/home/emilio/miniconda3/envs/f1t/bin/python
SNAP=/home/emilio/projects/bachelor/benchmark_results/tune_noise_2026-11-08/code
run() { name=$1; shift
  (cd $SNAP && $PY $V/ref_trace.py "$@" $V/ref_$name.npz > $V/ref_$name.log 2>&1) &
  (cd /home/emilio/projects/bachelor/atmppi && $PY $V/my_trace.py "$@" $V/my_$name.npz > $V/my_$name.log 2>&1) &
}
run hwlong hw icra2022 raceline 0.5 off 1 12000 50
run hwrca hw RCA1 midline 0.9 on 3 7500 50
run hwhangar hw hangar3 raceline 0.9 on 4 7500 50
run mppi2000 mppi icra2022 raceline 0.5 off 1 2500 2000
run mppiipz mppi IPZ50 midline 0.9 on 2 5000 50
wait
for n in hwlong hwrca hwhangar mppi2000 mppiipz; do echo "=== $n"; $PY $V/compare.py $V/ref_$n.npz $V/my_$n.npz; done
