import sys, numpy as np
a, b = np.load(sys.argv[1]), np.load(sys.argv[2])
sa, sb = a["state"], b["state"]
n = min(len(sa), len(sb))
diff = np.abs(sa[:n] - sb[:n]).max(axis=1)
bad = np.nonzero(diff > 0)[0]
print("steps", len(sa), len(sb), "first differing step:", bad[0] if len(bad) else None, "max diff", diff.max())
if len(bad): print("diff at first/+10/+100:", [float(diff[min(bad[0]+k, n-1)]) for k in (0, 10, 100)])
print("laps ref", np.round(a["laps"], 3).tolist()); print("laps new", np.round(b["laps"], 3).tolist())
print("crash ref", a["crash"].tolist(), "new", b["crash"].tolist())
