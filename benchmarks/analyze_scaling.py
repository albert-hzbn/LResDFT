"""Summarise results/scaling_rigorous.json: medians, spreads, speed-ups, power-law fits, crossover.
Writes results/scaling_rigorous_summary.json and prints a table."""
import json
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
rows = json.load(open(os.path.join(ROOT, "results", "scaling_rigorous.json")))
out = {"sizes": {}}
Ns = sorted({r["N_e"] for r in rows})


def get(N, s, key="wall"):
    return np.array([r[key] for r in rows if r["N_e"] == N and r["solver"] == s])


for N in Ns:
    d = {}
    for s in ("KS", "LRes"):
        w = get(N, s)
        if len(w):
            d[s] = dict(n=len(w), median=float(np.median(w)), min=float(w.min()), max=float(w.max()),
                        iters=sorted({int(i) for i in get(N, s, "iters")}),
                        cpu_over_wall=float(np.median(get(N, s, "cpu") / w)))
    if "KS" in d and "LRes" in d:
        k, l = get(N, "KS"), get(N, "LRes")
        d["speedup_median"] = d["KS"]["median"] / d["LRes"]["median"]
        d["speedup_range"] = [float(k.min() / l.max()), float(k.max() / l.min())]   # extreme pairings
    out["sizes"][N] = d


def fit(s, Nmin):
    x = [N for N in Ns if N >= Nmin and s in out["sizes"][N]]
    y = [out["sizes"][N][s]["median"] for N in x]
    p, cov = np.polyfit(np.log(x), np.log(y), 1, cov=True)
    return float(p[0]), float(np.sqrt(cov[0, 0])), float(np.exp(p[1])), len(x)


for s in ("KS", "LRes"):
    out[f"fit_{s}"] = dict(zip(("exponent", "stderr", "prefactor", "npoints"), fit(s, 250)))
a, b = out["fit_KS"], out["fit_LRes"]
out["crossover_fit_N"] = float((b["prefactor"] / a["prefactor"]) ** (1 / (a["exponent"] - b["exponent"])))
json.dump(out, open(os.path.join(ROOT, "results", "scaling_rigorous_summary.json"), "w"), indent=1)
print(f"{'N':>5} {'KS med [min-max]':>24} {'LRes med [min-max]':>24} {'speedup':>8}  range")
for N in Ns:
    d = out["sizes"][N]
    f = lambda s: f"{d[s]['median']:8.1f} [{d[s]['min']:.1f}-{d[s]['max']:.1f}]" if s in d else "-"
    sp = f"{d['speedup_median']:.2f}  {d['speedup_range'][0]:.2f}-{d['speedup_range'][1]:.2f}" if "speedup_median" in d else ""
    print(f"{N:5d} {f('KS'):>24} {f('LRes'):>24} {sp}")
for s in ("KS", "LRes"):
    print(s, out[f"fit_{s}"])
print("crossover from fits (N):", round(out["crossover_fit_N"]))
