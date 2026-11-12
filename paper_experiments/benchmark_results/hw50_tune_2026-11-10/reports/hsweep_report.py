"""Half noise, icra2022 raceline mu 0.5. ATMPPI 50: drive horizon after the freeze (35 before, optimizer 35). MPPI 2000: horizon."""
import json, os, statistics as st, sys
BR = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, f"{BR}/grid_2026-11-05/thesis_icra2022")
import make_plots as mp
H, M = f"{BR}/hw50_tune_2026-11-10/runs/hsweep", f"{BR}/mppi_tune_2026-11-10/runs/hsweep"

print("ATMPPI 50 (drive horizon after freeze): per seed median after freeze / crashes after freeze")
for v in ("h35", "af30", "af25", "af20", "af17"):
    cells, meds, c_af, c_all, n_af = [], [], 0, 0, 0
    for s in (1, 2, 3, 4):
        p = f"{H}/{v}/seed{s}/result.json"
        if not os.path.exists(p):
            cells.append("  running   "); continue
        r = json.load(open(p)); lt = [l["lap_time"] for l in r["laps"]]
        if lt and (lt[0] > 30 or 1 in mp.collision_laps(r)):
            cells.append(" broken 1st "); continue
        if not r.get("freeze_after_lap"):
            cells.append(" no freeze  "); continue
        af = r["after_freeze"]; m = st.median(af["lap_times"])
        meds.append(m); c_af += af["n_collisions"]; c_all += r["n_collisions"]; n_af += len(af["lap_times"])
        cells.append(f"{m:6.2f}/{af['n_collisions']:<2d}   ")
    agg = f"mean {st.mean(meds):.2f} s, {100 * c_af / max(n_af, 1):.1f} crashes/100 frozen laps ({c_af} in {n_af}), {c_all} total" if meds else ""
    print(f"  H{v[-2:]:>2s}: " + " ".join(cells) + "  " + agg)

print("\nMPPI 2000 (horizon): per seed median from lap 2 / crashes")
for v in ("h50", "h35", "h30", "h25", "h20", "h17"):
    cells, meds, c = [], [], 0
    for s in (1, 2):
        p = f"{M}/{v}/seed{s}/result.json"
        if not os.path.exists(p):
            cells.append("  running   "); continue
        r = json.load(open(p)); lt = [l["lap_time"] for l in r["laps"]][1:]
        m = st.median(lt) if lt else float("nan"); meds.append(m); c += r["n_collisions"]
        cells.append(f"{m:6.2f}/{r['n_collisions']:<2d}   ")
    print(f"  H{v[-2:]:>2s}: " + " ".join(cells) + (f"  mean {st.mean(meds):.2f} s, {c} crashes" if meds else ""))
