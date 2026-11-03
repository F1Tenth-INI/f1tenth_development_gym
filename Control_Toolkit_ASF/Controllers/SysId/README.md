# SysId

System identification for a small room:

- `bench_measurements.py`: bench measurement sheet (raw readings) → car parameter file
- `arena.py`: map free space → circles, figure eight, circle raceline + start pose
- `sysid_planner.py`: `--CONTROLLER sysid`, the automated driving program
- `fit_parameters.py`: driving recordings → fitted car parameter file

Full walkthrough for a new car, bench procedures, simulation test results and limitations:
`f1tenth_system/sysid/README.md` and `f1tenth_system/sysid/bench_measurements.md`
(outside this submodule). Measurement sheets live in `f1tenth_system/sysid/cars/`.
