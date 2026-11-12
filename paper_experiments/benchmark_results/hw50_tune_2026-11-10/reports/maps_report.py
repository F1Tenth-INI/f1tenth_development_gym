"""Half noise, raceline mu 0.5, ATMPPI 50: H35 vs drive 25 after freeze (af25) vs optimizer + drive 25 (opt25)."""
import json, os, statistics as st, sys
BR = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, f"{BR}/grid_2026-11-05/thesis_icra2022")
import make_plots as mp
R = f"{BR}/hw50_tune_2026-11-10/runs"
LABEL = {"h35": "H35", "af25": "drive 25 after freeze", "opt25": "optimizer + drive 25"}
for m, d, seeds in (("icra2022", "hsweep", (1, 2, 3, 4)), ("RCA1", "hsweep_RCA1", (1, 2, 3)),
                    ("IPZ50", "hsweep_IPZ50", (1, 2, 3)), ("hangar3", "hsweep_hangar3", (1, 2, 3))):
    print(m)
    firsts = [json.load(open(f"{R}/{d}/{v}/seed{s}/result.json"))["laps"][0]["lap_time"]
              for v in ("h35", "af25", "opt25") for s in seeds if os.path.exists(f"{R}/{d}/{v}/seed{s}/result.json")]
    first_max = 1.3 * st.median(firsts) if firsts else 1e9  # broken first lap: much slower than the map's usual one
    for v in ("h35", "af25", "opt25"):
        cells, meds, c_af, n_af = [], [], 0, 0
        for s in seeds:
            p = f"{R}/{d}/{v}/seed{s}/result.json"
            if not os.path.exists(p):
                cells.append("  running   "); continue
            r = json.load(open(p)); lt = [l["lap_time"] for l in r["laps"]]
            if lt and (lt[0] > first_max or 1 in mp.collision_laps(r)):
                cells.append(" broken 1st "); continue
            if not r.get("freeze_after_lap"):
                cells.append(f" no freeze/{r['n_collisions']}"); continue
            af = r["after_freeze"]; md = st.median(af["lap_times"])
            meds.append(md); c_af += af["n_collisions"]; n_af += len(af["lap_times"])
            cells.append(f"{md:6.2f}/{af['n_collisions']:<2d}   ")
        agg = f"mean {st.mean(meds):.2f} s, {100 * c_af / max(n_af, 1):.1f} crashes/100 frozen laps ({c_af} in {n_af})" if meds else ""
        print(f"  {LABEL[v]:22s}: " + " ".join(cells) + "  " + agg)
