"""Half noise (0.025), icra2022 raceline mu 0.5: H35 vs H17 drive-only vs H17 everywhere (ATMPPI), H35 vs H17 (MPPI 2000)."""
import json, os, statistics as st
BR = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
FIRST_LAP_MAX = 30.0  # [s] normal first laps take 22-26 s; slower ones went wrong from the start
H, M = f"{BR}/hw50_tune_2026-11-10/runs/hn", f"{BR}/mppi_tune_2026-11-10/runs/hn"

def line(path, hw):
    if not os.path.exists(f"{path}/result.json"):
        return "running"
    r = json.load(open(f"{path}/result.json"))
    lt = [l["lap_time"] for l in r["laps"]]
    import sys; sys.path.insert(0, f"{BR}/grid_2026-11-05/thesis_icra2022"); import make_plots as mp
    if hw and lt and (lt[0] > FIRST_LAP_MAX or 1 in mp.collision_laps(r)):
        return f"DISCARDED: broken first lap ({lt[0]:.1f} s, crashes in lap 1: {1 in mp.collision_laps(r)})"
    if hw and r.get("freeze_after_lap"):
        af = r["after_freeze"]
        return (f"{len(lt)} laps, frozen after lap {r['freeze_after_lap']} ({r['freeze_t']:.0f} s): median {st.median(af['lap_times']):.2f} s"
                f" ({len(af['lap_times'])} laps), crashes {r['n_collisions']} total, {af['n_collisions']} after freeze")
    tail = lt[1:] if not hw else lt[len(lt) // 2:]
    return (f"{len(lt)} laps, {'never froze, 2nd half' if hw else 'from lap 2'}: median {st.median(tail):.2f} s, "
            f"crashes {r['n_collisions']}" if tail else f"no laps, crashes {r['n_collisions']}")

for n in (50, 1000):
    for s in (1, 2):
        print(f"ATMPPI {n} seed{s}")
        for v, label in (("h35", "H35"), ("h17drive", "H17 drive"), ("h17af", "H17 frozen")):
            print(f"   {label:9s}: {line(f'{H}/{v}_n{n}/seed{s}', True)}")
for s in (1, 2):
    print(f"MPPI 2000 seed{s}")
    for v, label in (("h35", "H35"), ("h17", "H17")):
        print(f"   {label:9s}: {line(f'{M}/{v}/seed{s}', False)}")
