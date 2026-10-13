"""Compare the CLI (default --model gth, KS solver) with all-electron LDA (results/reference_dft.json,
PySCF aug-cc-pVQZ) and with experiment.  Writes results/literature.json.
Usage: python3 benchmarks/literature.py [dx] [model]

Experimental values (r_e in angstrom, dipole in debye) are standard tabulated numbers entered from memory
(NIST CCCBDB); they are not re-checked against a source here.
"""
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from lrft import cli  # noqa: E402

DX = sys.argv[1] if len(sys.argv) > 1 else "0.2"
MODEL = sys.argv[2] if len(sys.argv) > 2 else "gth"
D_PER_AU = 1 / 0.393430307
EXP_RE = dict(H2=0.741, LiH=1.595, HF=0.917, N2=1.098, CO=1.128)
EXP_DIP = dict(LiH=5.88, HF=1.82, CO=0.11, H2O=1.85, NH3=1.47, CH4=0.0, H2=0.0, N2=0.0)
TMP = os.path.join(ROOT, "results", "_tmp.json")


def run(args):
    cli.main(["run", *args, "--dx", DX, "--model", MODEL, "--json", TMP])
    return json.load(open(TMP))["results"]["ks"]


def dimer(path, s1, s2, d):
    open(path, "w").write(f"2\n\n{s1} 0 0 0\n{s2} 0 0 {d}\n")


def bond_length(s1, s2, d0, path):
    ds = d0 + np.array([-0.06, -0.03, 0, 0.03, 0.06])
    E = []
    for d in ds:
        dimer(path, s1, s2, d)
        E.append(run([path])["energy_Ha"])
    c = np.polyfit(ds, E, 3)
    xs = np.linspace(ds[0], ds[-1], 4001)
    re = float(xs[np.argmin(np.polyval(c, xs))])
    return re


ref = json.load(open(os.path.join(ROOT, "results", "reference_dft.json")))
res = {"dx": float(DX), "model": MODEL}
xyz = os.path.join(ROOT, "results", "_dimer.xyz")
for name, (s1, s2) in dict(H2=("H", "H"), LiH=("Li", "H"), HF=("F", "H"), N2=("N", "N"), CO=("C", "O")).items():
    re = bond_length(s1, s2, ref[name]["re"], xyz)
    dimer(xyz, s1, s2, re)
    r = run([xyz])
    res[name] = dict(re=re, dipole_D=float(np.linalg.norm(r["dipole_au"]) * D_PER_AU), homo_eV=r["homo_Ha"] * 27.211386)
for name in ("H2O", "NH3", "CH4", "He", "Ne"):
    r = run([name])
    res[name] = dict(dipole_D=float(np.linalg.norm(r["dipole_au"]) * D_PER_AU), homo_eV=r["homo_Ha"] * 27.211386)
res["rows"] = []
for name in ("H2", "LiH", "HF", "N2", "CO", "H2O", "NH3", "CH4", "He", "Ne"):
    key = name if name in ref and "re" in ref[name] else name + "_fixed_geom"
    a, t = res[name], ref[key]
    row = dict(name=name, homo_eV=a["homo_eV"], homo_AE=t["homo_eV"], dip_D=a["dipole_D"], dip_AE=t["dipole_D"],
               dip_exp=EXP_DIP.get(name))
    if "re" in a:
        row.update(re=a["re"], re_AE=t["re"], re_exp=EXP_RE[name])
    res["rows"].append(row)
    print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()}, flush=True)
for f in (xyz, TMP):
    os.remove(f)
json.dump(res, open(os.path.join(ROOT, "results", "literature.json"), "w"), indent=1)
