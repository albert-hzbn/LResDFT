"""Write every table body and inline number of main.tex / supplement.tex from results/*.json.
Missing data are written as '--' so that the document always compiles."""
import json
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
OUT = os.path.join(ROOT, "manuscript", "tables")
os.makedirs(OUT, exist_ok=True)


def load(name):
    p = os.path.join(RES, name)
    return json.load(open(p)) if os.path.exists(p) else {}


def sci(x, d=1):
    if x is None:
        return "--"
    if x == 0:
        return "$0$"
    m, e = f"{x:.{d}e}".split("e")
    return f"${m}\\times10^{{{int(e)}}}$"


def sci_plain(x, d=1):
    m, e = f"{x:.{d}e}".split("e")
    return f"{m}\\times10^{{{int(e)}}}"


def write(name, rows):
    open(os.path.join(OUT, name), "w").write("\n".join(rows) + "\n")


acc = load("accuracy.json")
t3 = load("three_d.json")
rc = load("referee_checks.json")
big = load("scf3d_large.json")
big1 = load("scf3d_large_v1.json")   # earlier complete run (includes 1024 electrons)
ex = load("extra_checks.json")
kd = load("ks_drivers.json")
s1 = load("scaling.json")

# ---------------- main text ----------------
write("exact.tex", [
    f"{r['system'].replace('H2 R=', 'H$_2$, $R=') + ('$' if r['system'].startswith('H2') else '')} & "
    f"{r['E_exact']:.5f} & {r['E_KS']:.5f} & {r['E_LRFT']:.5f} & {r['err_KS']:.4f} & "
    f"{sci(abs(r['LRFT_minus_KS']))} \\\\" for r in acc["exact"]])

names = {"(H2)_8 chain": "(H$_2$)$_8$ chain", "(H2)_32 chain": "(H$_2$)$_{32}$ chain",
         "H_20 uniform chain (small gap)": "H$_{20}$ chain (small gap)"}
write("ks1d.tex", [f"{names.get(r['system'], r['system'])} & {r['N']} & {r['E_KS']:.6f} & "
                   f"{sci(abs(r['dE_per_e']))} & {sci(r['density_L1_per_e'])} \\\\" for r in acc["ks"]])

rows = []
for r in rc.get("R10_pole_rules", []):
    case = {"grid": lambda r: f"grid, $\\Delta x={r['dx']}$",
            "gap": lambda r: f"H$_{{{r['N_atoms']}}}$ chain",
            "deep_core": lambda r: f"core, $Z={r['Z_heavy']}$"}[r["series"]](r)
    rows.append(f"{case} & {r['spectral_width']:.0f} & {r['gap']:.3f} & {r['occ_width']:.2f} & "
                f"{r['half_step_trapezoid']} & {r['gauss_legendre_circle']} & {r['zolotarev']} \\\\")
write("rules.tex", rows)

rows = []
for r in rc.get("R11_forces", []):
    rows.append(f"{r['M']} & {r['F_hf_grand']:.8f} & {sci(r['diff_grand_extrap'])} & {sci(r['diff_fixedN'])} \\\\")
write("forces.tex", rows)

rows = []
for r in rc.get("R2_gap_dependence", []):
    if r["M"] != 64:
        continue
    lab = f"$a={r['a']}$"
    rows.append(f"{lab} & {r['N_e']} & {r['gap']:.2f} & {r['R_w']} & {r['n_W']:.0f} & {sci(r['dn_per_e'])} & "
                f"{sci(r['dE_per_e'])} \\\\")
am = rc.get("R3_amorphous")
if am:
    for r in am["rows"]:
        if r["R_w"] in (8.5, 12.1):
            rows.append(f"amorph. & {am['N_e']} & {am['gap']:.2f} & {r['R_w']} & {r['n_W_mean']:.0f} & "
                        f"{sci(r['dn_per_e'])} & {sci(r['dE_per_e'])} \\\\")
write("compress.tex", rows)

rows = []
dc = rc.get("R1_dc_comparison")
if dc:
    for r in dc["rows"]:
        rows.append(f"{r['R_w']} & {r['n_W']:.0f} & {sci(r['LRFT_M64_dn'])} & {sci(r['DC_FD_kT0.005_dn'])} & "
                    f"{sci(r['DC_FD_kT0.01_dn'])} \\\\")
write("dc.tex", rows)

# SCF table (16-1024 electrons)
rows, dEs = [], []
for r in t3.get("scf", []):
    rows.append(f"{r['N_e']} & {r['E_KS']:.6f} & {sci(abs(r['dE_per_e']))} & {sci(r['dn_per_e'])} & "
                f"{r['it_KS']}/{r['it_LRFT']} & {r['t_scf_KS']:.1f} & {r['t_scf_LRFT']:.1f} & "
                f"{r['t_scf_KS'] / r['t_scf_LRFT']:.2f} \\\\")
    if r["N_e"] >= 128:
        dEs.append(abs(r["dE_per_e"]))
for k in ("n6", "n7", "n8"):
    if k in big:
        r = big[k]
        rows.append(f"{r['N_e']} & {r['E_KS']:.6f} & {sci(abs(r['dE_per_e']))} & {sci(r['dn_per_e'])} & "
                    f"{r['it_KS']}/{r['it_LRFT']} & {r['wall_KS']:.0f} & {r['wall_LRFT']:.0f} & "
                    f"{r['wall_KS'] / r['wall_LRFT']:.2f} \\\\")
        dEs.append(abs(r["dE_per_e"]))
write("scf3d_all.tex", rows)
open(os.path.join(OUT, "scf_range.tex"), "w").write(
    f"between ${sci_plain(min(dEs))}$ and ${sci_plain(max(dEs))}$" if dEs else "--")
if "n6" in big and "Rw_series" in big["n6"]:
    rs = {x["R_w"]: x for x in big["n6"]["Rw_series"]}
    vals = [abs(rs[8.5]["dE_per_e"]), abs(big["n6"]["dE_per_e"]), abs(rs[13.5]["dE_per_e"])]
    open(os.path.join(OUT, "scf_rw.tex"), "w").write(
        f"${sci_plain(vals[0])}$, ${sci_plain(vals[1])}$ and ${sci_plain(vals[2])}$~Ha per electron")
else:
    open(os.path.join(OUT, "scf_rw.tex"), "w").write("--")

# cost table and inline numbers
rows = []
cost = t3.get("cost", [])
for r in cost:
    ks = f"{r['t_ks']:.2f}" if "t_ks" in r else "--"
    rows.append(f"{r['N_e']} & {r['N_b']} & {ks} & {r['t_lrft']:.2f} & {r['t_lrft_cold']:.2f} & "
                f"{sci(r.get('dn_per_e'))} \\\\")
write("cost3d.tex", rows)


def fit(key):
    pts = [(r["N_e"], r[key]) for r in cost if key in r]
    x, y = np.log([p[0] for p in pts[-4:]]), np.log([p[1] for p in pts[-4:]])
    return np.polyfit(x, y, 1)[0], [p[0] for p in pts[-4:]]


if cost:
    pk, rk = fit("t_ks")
    pw, rw = fit("t_lrft")
    pc, rcold = fit("t_lrft_cold")
    open(os.path.join(OUT, "exponents.tex"), "w").write(
        f"$N^{{{pk:.2f}}}$ for KS ({rk[0]}--{rk[-1]} electrons), $N^{{{pw:.2f}}}$ for the LR-DFT warm step and "
        f"$N^{{{pc:.2f}}}$ for the cold step ({rw[0]}--{rw[-1]} electrons)")

    def cross(key):
        pts = [(r["N_e"], r["t_ks"] / r[key]) for r in cost if "t_ks" in r]
        for (n1, s1_), (n2, s2_) in zip(pts, pts[1:]):
            if s1_ < 1 <= s2_:
                t = np.log(1 / s1_) / np.log(s2_ / s1_)
                return float(np.exp(np.log(n1) + t * np.log(n2 / n1)))
        return None
    cw, cc = cross("t_lrft"), cross("t_lrft_cold")
    last = [r for r in cost if "t_ks" in r][-1]
    open(os.path.join(OUT, "crossover.tex"), "w").write(
        f"about {cw:.0f} electrons for the warm step and about {cc:.0f} electrons for the cold step; "
        f"at {last['N_e']} electrons LR-DFT is {last['t_ks'] / last['t_lrft']:.1f} (warm) and "
        f"{last['t_ks'] / last['t_lrft_cold']:.1f} (cold) times faster")

mem = rc.get("R8_memory", [])
mem_ok = [r for r in mem if "extra_MB" in r]
if mem_ok:
    def mfit(which):
        pts = [(r["N_e"], r["extra_MB"]) for r in mem_ok if r["which"] == which and r["extra_MB"] > 1]
        x, y = np.log([p[0] for p in pts[-3:]]), np.log([p[1] for p in pts[-3:]])
        return np.polyfit(x, y, 1)[0], pts[-1]
    pk, lk = mfit("ks")
    pl, ll = mfit("lrft")
    open(os.path.join(OUT, "memory.tex"), "w").write(
        f"$N^{{{pk:.1f}}}$ for KS and $N^{{{pl:.1f}}}$ for LR-DFT; at {lk[0]} electrons the solver step needs "
        f"{lk[1]:.0f}~MB (KS) and {ll[1]:.0f}~MB (LR-DFT) in addition to the shared basis and grid data")
else:
    open(os.path.join(OUT, "memory.tex"), "w").write("--")

# ---------------- Supplemental Material ----------------
rows = []
for r in rc.get("R4_basis", []):
    rows.append(f"{r['basis']} & {r['basis_err_per_e']:.4f} & {r['min_overlap_eig']:.4f} & {r['R_w']} & "
                f"{r['n_W']:.0f} & {sci(r['dn_per_e'])} & {sci(r['dE_per_e'])} \\\\")
write("sm_basis.tex", rows)

r5 = rc.get("R5_margin", {})
rows = []
for r in r5.get("rows", []):
    rows.append(f"{r['delta']} & {r['M']} & {r['rho']:.3f} & {sci(r['err_per_e'])} & {sci(r['bound_per_e'])} \\\\")
write("sm_margin.tex", rows)

rows = []
for r in rc.get("R7_timing_repeatability", []):
    rows.append(f"{r['N_e']} & {r['ks_mean']:.3f} & {r['ks_std']:.3f} & {r['lrft_mean']:.3f} & "
                f"{r['lrft_std']:.3f} \\\\")
write("sm_timing.tex", rows)

rows = []
for r in mem_ok:
    rows.append(f"{r['N_e']} & {r['N_b']} & {r['which'].replace('lrft', 'LR-DFT').replace('ks', 'KS')} & "
                f"{r['rss_before_MB']:.0f} & {r['extra_MB']:.0f} \\\\")
write("sm_memory.tex", rows)

rows = []
for r in rc.get("R9_closing_gap", []):
    rows.append(f"{r['N_atoms']} & {r['gap']:.3f} & {r['M_for_1e4']} & {r['M_for_1e6']} \\\\")
write("sm_gap.tex", rows)

rows = []
for name, d in rc.get("R12_window_diagnostics", {}).items():
    for Rw, x in d.items():
        rows.append(f"{name} & {Rw} & {x['homo']:.3f} & {x['lumo']:.3f} & {x['n_states_in_gap']} & "
                    f"{x['ks_min_eig']:.4f} & {x['min_window_eig']:.4f} \\\\")
write("sm_windows.tex", rows)

rows = []
for r in kd.get("three_d", []):
    rows.append(f"{r['N_e']} & {r['N_b']} & \\texttt{{{r['driver']}}} & {r['density']} & {r['t_eig']:.2f} & "
                f"{r['t_density']:.2f} & {r['t_total']:.2f} \\\\")
write("sm_ksdrivers.tex", rows)
rows = []
for r in kd.get("one_d", []):
    rows.append(f"{r['N_e']} & {r['N_g']} & {r['stebz']:.3f} & {r['stemr']:.3f} \\\\")
write("sm_ks1d.tex", rows)

rows = []
for r in ex.get("grid_refinement", []):
    rows.append(f"{r['dx']} & {r['N_g']} & {r['spectral_width']:.0f} & {r['gap']:.3f} & {r['M_needed']} & "
                f"{r['cheb_degree_needed']} \\\\")
write("sm_grid.tex", rows)

r13 = rc.get("R13_theorem1_bound", {})
rows = []
for r in r13.get("rows", []):
    rows.append(f"{r['M']} & {sci(r['err_L1_total'])} & {sci(r['err_per_e'])} & {sci(r['bound_total'])} \\\\")
write("sm_bound.tex", rows)

rows = []
for r in s1.get("rows", []):
    rows.append(f"{r['N_e']} & {r['N_g']} & {r['t_lrft']:.3f} & {r['t_lrft_single_mu']:.4f} & "
                f"{r.get('t_lrft_near', float('nan')):.3f} & "
                + (f"{r['t_ks_tridiag']:.3f}" if "t_ks_tridiag" in r else "--") + " & "
                + (f"{r['t_ks_dense']:.2f}" if "t_ks_dense" in r else "--") + " & "
                + (sci(r["near_err_per_e"]) if "near_err_per_e" in r else "--") + " \\\\")
write("sm_cost1d.tex", rows)
print("tables written to", OUT)

# ---------------- second-revision additions ----------------
def put(name, text):
    open(os.path.join(OUT, name), "w").write(text)


# forces: fixed mu (R11), fixed N rigid circle and fixed N moving edge (R17)
r11 = {r["M"]: r for r in rc.get("R11_forces", [])}
rows = []
for r in rc.get("R17_fixedN_forces", []):
    g = r11.get(r["M"], {})
    rows.append(f"{r['M']} & {sci(g.get('diff_grand_extrap'))} & {sci(r['diff_rigid'])} & {sci(r['diff_moving'])} \\\\")
write("forces.tex", rows)

# occupation range of the pole rules (R14)
r14 = rc.get("R14_occupation_range", [])
if r14:
    gl = max(max(-x["gauss_legendre"]["min"], x["gauss_legendre"]["max"] - 1) for x in r14)
    zo = max(max(-x["zolotarev"]["min"], x["zolotarev"]["max"] - 1) for x in r14)
    put("occrange.tex", f"${sci_plain(gl)}$ and ${sci_plain(zo)}$")
else:
    put("occrange.tex", "--")

# probing (R15)
r15 = rc.get("R15_probing", [])
if r15:
    a = {x["n_probes"]: x for x in r15}
    put("probing.tex", f"with 125 probes (spacing 3~bohr) the error of the probed diagonal was {a[125]['rel_error']:.1f} times "
        f"the density itself, and with 1000 probes it was still {100 * a[1000]['rel_error']:.0f}\\%")

# threads (R18)
r18 = rc.get("R18_threads", [])
if r18:
    t = {x["threads"]: x for x in r18}
    put("threads.tex", f"from {t[1]['lrft_warm']:.2f} to {t[8]['lrft_warm']:.2f}~s on 8 threads "
        f"({t[1]['lrft_warm'] / t[8]['lrft_warm']:.1f}$\\times$), while the threaded LAPACK of the KS step went from "
        f"{t[1]['ks']:.2f}~s to {min(x['ks'] for x in r18):.2f}~s at best")

# LOBPCG (R16)
r16 = rc.get("R16_lobpcg", {})
if r16:
    rr = [x for x in r16.get("scf", []) if x["tol"] == 1e-5]
    p = r16.get("n1024_partial") or rc.get("R16_lobpcg_1024")
    s_ = "; ".join(f"{x['N_e']} electrons: {x['wall_lobpcg']:.0f}~s versus {x['wall_dense']:.0f}~s" for x in rr)
    if p:
        s_ += (f"; at {p['N_e']} electrons a warm LOBPCG step took {np.mean(p['step_times'][1:]):.0f}~s "
               f"(up to {max(p['lobpcg_iters'])} iterations)")
    put("lobpcg.tex", s_)
else:
    put("lobpcg.tex", "--")

# SCF wall times: repeated, interleaved runs (benchmarks/scaling_rigorous.py, analyze_scaling.py)
rg = load("scaling_rigorous_summary.json")
big1 = load("scf3d_large_v1.json")
if rg:
    S = {int(k): v for k, v in rg["sizes"].items()}
    lines = []
    for N in sorted(S):
        d = S[N]
        k = f"{d['KS']['median']:.1f} ({d['KS']['min']:.1f}--{d['KS']['max']:.1f})" if "KS" in d else "--"
        l = (f"{d['LRes']['median']:.1f} ({d['LRes']['min']:.1f}--{d['LRes']['max']:.1f})" if d['LRes']['n'] > 1
             else f"{d['LRes']['median']:.1f} (1 run)")
        sp = f"{d['speedup_median']:.2f}" if "speedup_median" in d else "--"
        lines.append(f"{N} & {k} & {l} & {sp} \\\\")
    put("rigorous_rows.tex", "\n".join(lines))
    fk, fl = rg["fit_KS"], rg["fit_LRes"]
    put("rigorous_fit.tex", f"$N^{{{fk['exponent']:.2f}\\pm{fk['stderr']:.2f}}}$ for dense KS and "
        f"$N^{{{fl['exponent']:.2f}\\pm{fl['stderr']:.2f}}}$ for LRes-DFT")
    put("rigorous_cross.tex", f"{round(rg['crossover_fit_N'], -1):.0f}")
    sp = {N: S[N]["speedup_median"] for N in S if "speedup_median" in S[N]}
    put("scf_factor.tex", f"{sp[1024]:.1f}")
    put("scf_walls.tex",
        "Complete SCF wall times (single thread) were measured three times for each size, with the order of KS and "
        "LRes-DFT alternated, on an otherwise idle machine pinned to one core (Table~\\ref{tab:rigorous}). The repeats "
        f"agree within {max((S[N][s]['max'] - S[N][s]['min']) / S[N][s]['median'] for N in S for s in ('KS','LRes') if s in S[N] and S[N][s]['n'] > 1) * 100:.0f}\\%. "
        f"LRes-DFT is {1 / sp[432]:.1f} times slower than dense KS at 432 electrons, {sp[686]:.1f} times faster at 686 and "
        f"{sp[1024]:.1f} times faster at 1024 electrons")
    t = big1.get("n6", {}).get("Rw_series", [])
    B = load("scf3d_large.json").get("n6", {})
    rw = {x["R_w"]: x for x in B.get("Rw_series", [])}
    if 13.5 in rw:
        put("scf_tradeoff.tex", f"{rw[13.5]['wall']:.0f}~s at $R_w=13.5$~bohr against {B['wall_LRFT']:.0f}~s at 12.1~bohr and "
            f"{B['wall_KS']:.0f}~s for KS (a separate run on a loaded machine), i.e.\\ it is slower than dense KS")
    else:
        put("scf_tradeoff.tex", "--")
else:
    for f in ("scf_walls.tex", "scf_factor.tex", "scf_tradeoff.tex", "rigorous_rows.tex", "rigorous_fit.tex", "rigorous_cross.tex"):
        put(f, "--")

# every run of the repeated SCF timings (supplement)
raw = load("scaling_rigorous.json")
if raw:
    lines = []
    for N in sorted({r["N_e"] for r in raw}):
        def runs(sv):
            return ", ".join(f"{r['wall']:.1f}" for r in sorted((r for r in raw if r["N_e"] == N and r["solver"] == sv), key=lambda r: r["rep"]))
        its = sorted({r["iters"] for r in raw if r["N_e"] == N})
        lines.append(f"{N} & {runs('KS') or '--'} & {runs('LRes')} & {'/'.join(str(i) for i in its)} \\\\")
    put("sm_rigorous.tex", "\n".join(lines))
else:
    put("sm_rigorous.tex", "--")

# pole-rule table: add W_occ/gap
rows = []
for r in rc.get("R10_pole_rules", []):
    case = {"grid": lambda r: f"grid, $\\Delta x={r['dx']}$",
            "gap": lambda r: f"H$_{{{r['N_atoms']}}}$ chain",
            "deep_core": lambda r: f"core, $Z={r['Z_heavy']}$"}[r["series"]](r)
    rows.append(f"{case} & {r['spectral_width']:.0f} & {r['gap']:.3f} & {r['occ_width']:.2f} & "
                f"{r['occ_width'] / r['gap']:.1f} & {r['half_step_trapezoid']} & {r['gauss_legendre_circle']} & "
                f"{r['zolotarev']} \\\\")
write("rules.tex", rows)

# SM: R14, R15, R16, R18 tables
rows = []
for x in r14:
    rows.append(f"{x['case']} & {x['gap']:.3f} & {x['gauss_legendre']['min']:.1e} & {x['gauss_legendre']['max'] - 1:.1e} & "
                f"{x['zolotarev']['min']:.1e} & {x['zolotarev']['max'] - 1:.1e} \\\\")
write("sm_occrange.tex", rows)
rows = [f"{x['n_probes']} & {x['spacing_bohr']:.1f} & {x['rel_error']:.2f} \\\\" for x in r15]
write("sm_probing.tex", rows)
rows = [f"{x['threads']} & {x['lrft_warm']:.2f} & {x['ks']:.2f} & {x['load']:.1f} \\\\" for x in r18]
write("sm_threads.tex", rows)
rows = []
for x in (r16.get("scf", []) if r16 else []):
    rows.append(f"{x['N_e']} & {x['tol']:.0e} & {x['it']} & {sci(abs(x['dE_per_e']))} & {x['wall_lobpcg']:.1f} & "
                f"{x['wall_dense']:.1f} & {max(x['lobpcg_iters'])} \\\\")
write("sm_lobpcg.tex", rows)
