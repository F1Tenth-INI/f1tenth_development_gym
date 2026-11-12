# Notes for agents

- **Paper experiments (L4DC 2026, Auto-Tuning MPPI):** scripts, configs, results and how to regenerate the
  figures or run new simulations are in [paper_experiments/README.md](paper_experiments/README.md). Read it
  before touching anything in `paper_experiments/`.
- Auto-Tuning MPPI is the HirarchicalPlanner (`hw`): `TrainingLite/hwm/hirarchical_planner.py`, settings
  `HWM_*` / `MPC_*` in `utilities/Settings.py`. The MPPI baseline is `Control_Toolkit_ASF/Controllers/MPPILite/`.
- Single benchmark run: `python scripts/benchmark/bench_run.py --controller hw --rollouts 50 --map icra2022
  --wp raceline --mu 0.5 --noise off --seed 1 --sim-seconds 1000 --out <dir>` (writes `<dir>/result.json`).
