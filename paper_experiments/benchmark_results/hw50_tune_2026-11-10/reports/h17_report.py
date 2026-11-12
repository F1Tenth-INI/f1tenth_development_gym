"""Compare horizon-17 runs (still running: parsed from run.log) with horizon-35 runs over the same sim time."""
import glob, json, os, re, statistics as st
BR = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
H = f"{BR}/hw50_tune_2026-11-10/runs"
M = f"{BR}/mppi_tune_2026-11-10/runs"

def laps_log(d):
    txt = open(f"{d}/run.log").read()
    laps = [(float(t), float(l), f == "True") for t, l, f in re.findall(r"BENCH_LAP t=([\d.]+) lap_time=([\d.]+) frozen=(\w+)", txt)]
    coll = len(re.findall(r"BENCH_COLLISION|collision at|COLLISION", txt))
    return laps, coll

def laps_result(d, t_max):
    r = json.load(open(f"{d}/result.json"))
    out, t = [], 0.0
    fz = r.get("freeze_after_lap")
    for k, l in enumerate(r["laps"], 1):
        t += l["lap_time"]
        if t > t_max: break
        out.append((t, l["lap_time"], bool(fz) and k > fz))
    return out

def summary(laps):
    if not laps: return "no laps yet"
    lt = [l for _, l, _ in laps]
    fr = [l for _, l, f in laps if f]
    tail = lt[-5:]
    s = f"{len(lt)} laps in {laps[-1][0]:.0f} s, best {min(lt[1:] or lt):.2f}, median last 5 {st.median(tail):.2f}"
    if fr: s += f", frozen: median {st.median(fr):.2f} ({len(fr)} laps)"
    return s

def full(d):
    """Laps from the log; collisions and freeze from result.json once the run is done."""
    laps, _ = laps_log(d)
    extra = ""
    if os.path.exists(f"{d}/result.json"):
        r = json.load(open(f"{d}/result.json"))
        af = r.get("after_freeze") or {}
        extra = f", coll {r['n_collisions']}" + (f" ({af.get('n_collisions')} after freeze), freeze lap {r['freeze_after_lap']} ({r['freeze_t']:.0f} s)" if r.get("freeze_after_lap") else "")
    return laps, extra

rows = [("ATMPPI 50", f"{H}/r2/wm10_tsf103", f"{H}/h17/h17_n50", f"{H}/h17all/h17all_n50"),
        ("ATMPPI 100", f"{H}/r3_n100/wm10_tsf103", f"{H}/h17/h17_n100", f"{H}/h17all/h17all_n100"),
        ("ATMPPI 1000", f"{H}/r3_n1000/wm10_tsf103", f"{H}/h17/h17_n1000", f"{H}/h17all/h17all_n1000"),
        ("MPPI 2000", f"{M}/plotrun/ns_s02", f"{M}/h17/h17", None)]
for name, d35, d17, dall in rows:
    for s in (1, 2):
        variants = [("H17 drive" if dall else "H17", d17)] + ([("H17 all", dall)] if dall else [])
        got = []
        for label, d in variants:
            if os.path.exists(f"{d}/seed{s}/run.log"):
                laps, extra = full(f"{d}/seed{s}")
                got.append((label, laps, extra))
        t_max = max((l[-1][0] for _, l, _ in got if l), default=0)
        p35 = f"{d35}/seed{s}"
        print(f"{name} seed{s}")
        if os.path.exists(f"{p35}/result.json") and t_max:
            r = json.load(open(f"{p35}/result.json"))
            print(f"   H35       : {summary(laps_result(p35, t_max))}, coll {r['n_collisions']}" + (f", freeze lap {r['freeze_after_lap']} ({r['freeze_t']:.0f} s)" if r.get('freeze_after_lap') else ""))
        for label, laps, extra in got:
            print(f"   {label:10s}: {summary(laps)}{extra}")
