"""Prediction error of one or more car parameter files on the real-car recordings.

Run from the repo root:
    python AnalyseData/DynamicsModelFit/evaluate.py yokomo1_car_parameters.yml
    python AnalyseData/DynamicsModelFit/evaluate.py old.yml new.yml --delay 0.08
    python AnalyseData/DynamicsModelFit/evaluate.py yokomo1_car_parameters.yml --set C_r=1.75 B_r=3.6
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from fitlib import *  # noqa: E402,F401

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("cars", nargs="+", help="parameter files in utilities/car_files")
ap.add_argument("--set", nargs="*", default=[], help="overrides name=value, e.g. mu=0.7 (see fitlib.PARAM_IDX)")
ap.add_argument("--recordings", default=DEFAULT_RECORDINGS)
ap.add_argument("--test-file", default="2026-10-26_09-31-34_.csv")
ap.add_argument("--delay", type=float, default=0.04)
ap.add_argument("--model", default=DEFAULT_MODEL)
args = ap.parse_args()

overrides = {k: float(v) for k, v in (s.split("=") for s in args.set)}
roll = build_rollout(load_model(args.model))
recs = load_recordings(args.recordings)
sets = [("validation (held-out blocks)", concat([make_windows(recs[k], args.delay, blocks=(3,))
                                                 for k in recs if k != args.test_file]))]
if args.test_file in recs:
    sets.append(("test (" + args.test_file + ")", make_windows(recs[args.test_file], args.delay)))
for car in args.cars:
    p = jnp.array(car_params(car, overrides))
    print(f"--- {car} {overrides or ''}  delay {args.delay}s")
    for label, W in sets:
        print(f"  loss {loss(roll, W, p):.3f}  " + report(roll, W, p, label))
