"""Command-line tool: Kohn-Sham and LRes-DFT calculations for atoms and molecules.

    python -m lrft run H2O                 # built-in molecule
    python -m lrft run mol.xyz --solver both --dx 0.25 --json out.json
    python -m lrft list

Isolated systems are placed in a cubic box (vacuum padding) with the periodic
solver of lrft.three_d; closed-shell, spin-unpolarised LDA.  See lrft/molecules.py
for the (model) element parametrisation.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

HA_EV = 27.211386245988


def build_parser():
    p = argparse.ArgumentParser(prog="lrft", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list", help="list built-in molecules and supported elements")
    r = sub.add_parser("run", help="self-consistent single-point calculation")
    r.add_argument("system", help="built-in name (see 'list') or path to an .xyz file (angstrom)")
    r.add_argument("--solver", choices=["ks", "lres", "both"], default="ks",
                   help="ks: dense Kohn-Sham; lres: local-resolvent windows; both: run both and compare (default ks)")
    r.add_argument("--charge", type=int, default=None, help="net charge (overrides the file/built-in value)")
    r.add_argument("--model", choices=["gth", "toy"], default="gth",
                   help="gth: GTH/HGH-PADE pseudopotentials + GTH-TZV2P basis (H-Ne, quantitative LDA); "
                        "toy: softened local pseudopotentials of the manuscript (default gth)")
    r.add_argument("--dx", type=float, default=None, help="grid spacing, bohr (default 0.2 gth, 0.3 toy)")
    r.add_argument("--padding", type=float, default=6.0, help="vacuum between molecule and box edge, bohr (default 6)")
    r.add_argument("--rcut", type=float, default=None, help="basis-function cutoff radius, bohr (default 7 gth, 4.5 toy)")
    r.add_argument("--rw", type=float, default=10.0, help="LRes window radius around each atom, bohr (default 10)")
    r.add_argument("-M", "--poles", type=int, default=64, help="number of complex energies M for LRes (default 64)")
    r.add_argument("--tol", type=float, default=1e-6, help="SCF tolerance on |dn|/N (default 1e-6)")
    r.add_argument("--maxiter", type=int, default=80)
    r.add_argument("--mix", type=float, default=0.4, help="density mixing parameter (default 0.4)")
    r.add_argument("--backend", choices=["auto", "python", "cpp"], default="auto",
                   help="C++ kernels if the extension is built (./build_cpp.sh), else Python (default auto)")
    r.add_argument("--threads", type=int, default=1)
    r.add_argument("--json", metavar="FILE", help="write results to a JSON file")
    r.add_argument("--cube", metavar="FILE", help="write the final density as a Gaussian cube file")
    r.add_argument("-v", "--verbose", action="store_true", help="print every SCF iteration")
    return p


def _set_threads(n):
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[var] = str(n)


def cmd_list():
    from .molecules import ELEMENTS, MOLECULES
    print("built-in molecules:", " ".join(MOLECULES))
    print("elements (model valence parametrisation):")
    for s, (zv, rc, se, pe) in ELEMENTS.items():
        print(f"  {s:2s}  Z_val={zv}  rc={rc:.2f} bohr")
    print("\nclosed-shell systems only; model local pseudopotentials (qualitative for Z > 2).")


def build_system(args):
    if args.dx is None:
        args.dx = 0.2 if args.model == "gth" else 0.3
    if args.rcut is None:
        args.rcut = 7.0 if args.model == "gth" else 4.5
    import numpy as np
    from .molecules import ANG, ELEMENTS, check_elements, load_molecule
    from .three_d import Crystal
    charge, atoms = load_molecule(args.system)
    if args.charge is not None:
        charge = args.charge
    check_elements(atoms)
    pseudo = None
    if args.model == "gth":
        from .gth import load_gth
        pseudo = [load_gth(s) for s, *_ in atoms]
        Z = np.array([p["zion"] for p in pseudo], float)
    else:
        Z = np.array([ELEMENTS[s][0] for s, *_ in atoms], float)
    nel = int(Z.sum()) - charge
    if nel < 0:
        raise ValueError("charge removes more electrons than the system has")
    if nel % 2:
        raise ValueError(f"{nel} electrons: only closed-shell (even electron count) systems are supported")
    pos = np.array([[x, y, z] for _, x, y, z in atoms]) * ANG
    ext = (pos.max(0) - pos.min(0)).max()
    n = int(np.ceil((ext + 2 * args.padding) / args.dx))
    n += n % 2
    L = n * args.dx
    pos = pos - 0.5 * (pos.max(0) + pos.min(0)) + 0.5 * L
    if pseudo is not None:
        rcs = np.array([2 ** 0.5 * p["rloc"] for p in pseudo])
    else:
        rcs = np.array([ELEMENTS[s][1] for s, *_ in atoms])
    cr = Crystal(pos, Z, np.full(3, L), (n, n, n), rc=1.0, rcs=rcs, charge=charge, pseudo=pseudo)
    atom_exps = [(ELEMENTS[s][2], ELEMENTS[s][3]) for s, *_ in atoms]
    return cr, atoms, atom_exps, charge


def ion_ion_energy(cr):
    """Electrostatic energy of the Gaussian ion charges (matching -Z erf(r/rc)/r),
    periodic, G=0 removed, with the Gaussian self-energies Z^2/(sqrt(2 pi) rc) subtracted."""
    import numpy as np
    from .three_d import gvectors, hartree
    import scipy.fft
    kx, ky, kz = gvectors(cr)
    G2 = kx**2 + ky**2 + kz**2
    vol = float(np.prod(cr.cell))
    rho = np.zeros(cr.shape, complex)
    for (x, y, z), Z, rc in zip(cr.pos, cr.Z, cr.atom_rc):
        rho += Z * np.exp(-1j * (kx * x + ky * y + kz * z) - G2 * rc**2 / 4)
    with np.errstate(divide="ignore", invalid="ignore"):
        e = 0.5 * np.sum(np.where(G2 > 0, 4 * np.pi * np.abs(rho) ** 2 / G2, 0.0)) / vol
    self_e = np.sum(cr.Z**2 / (np.sqrt(2 * np.pi) * cr.atom_rc))
    return float(e - self_e)


def dipole(cr, dens):
    """Dipole (e*bohr) about the box centre; valid for charge-neutral systems."""
    import numpy as np
    x = (np.arange(cr.shape[0]) * cr.dx)
    X, Y, Z_ = np.meshgrid(x, x, x, indexing="ij")
    el = np.array([np.sum(dens * c) for c in (X, Y, Z_)]) * cr.dV
    ion = (cr.Z[:, None] * cr.pos).sum(0)
    return ion - el


def write_cube(path, cr, atoms, dens):
    import numpy as np
    dx = cr.dx
    with open(path, "w") as f:
        f.write("lrft density\nelectron density (e/bohr^3)\n")
        f.write(f"{len(atoms):5d} {0:12.6f} {0:12.6f} {0:12.6f}\n")
        for k in range(3):
            row = [0.0] * 3
            row[k] = dx
            f.write(f"{cr.shape[k]:5d} {row[0]:12.6f} {row[1]:12.6f} {row[2]:12.6f}\n")
        for (s, *_), Zv, R in zip(atoms, cr.Z, cr.pos):
            f.write(f"{int(Zv):5d} {Zv:12.6f} {R[0]:12.6f} {R[1]:12.6f} {R[2]:12.6f}\n")
        flat = dens.reshape(-1, cr.shape[2])
        for row in flat:
            for i in range(0, len(row), 6):
                f.write(" ".join(f"{v:13.5e}" for v in row[i:i + 6]) + "\n")


def make_solvers(args, cr, basis, which):
    from . import lcao3d
    import numpy as np
    use_cpp = args.backend != "python"
    if use_cpp:
        try:
            from . import _lrft_cpp  # noqa: F401
        except ImportError:
            if args.backend == "cpp":
                raise SystemExit("C++ extension not built: run ./build_cpp.sh or use --backend python")
            use_cpp = False
    out = {}
    if which in ("ks", "both"):
        out["ks"] = lcao3d.KSLCAOCpp(basis, args.threads) if use_cpp else lcao3d.KSLCAO(basis)
    if which in ("lres", "both"):
        cells = cr.pos.copy()
        if use_cpp:
            out["lres"] = lcao3d.LRFTLCAOCpp(basis, cells, R_w=args.rw, M=args.poles, nthreads=args.threads)
        else:
            out["lres"] = lcao3d.LRFTLCAO(basis, cells, R_w=args.rw, M=args.poles)
    return out, ("C++" if use_cpp else "Python")


def cmd_run(args):
    _set_threads(args.threads)
    import numpy as np
    from . import three_d
    from .lcao3d import LCAOBasis, scf_lcao
    three_d.FFT_WORKERS = args.threads
    cr, atoms, atom_exps, charge = build_system(args)
    print(f"system {args.system}: {len(atoms)} atoms, {cr.n_elec} electrons, charge {charge:+d}")
    print(f"box {cr.cell[0]:.2f} bohr, grid {cr.shape[0]}^3 (dx={cr.dx:.3f}), rcut {args.rcut}")
    t0 = time.perf_counter()
    if args.model == "gth":
        from .gth import PseudoBasis
        basis = PseudoBasis(cr, rcut=args.rcut, analytic=False)
    else:
        basis = LCAOBasis(cr, rcut=args.rcut, atom_exps=atom_exps)
    solvers, backend = make_solvers(args, cr, basis, args.solver)
    print(f"basis: {basis.nb} functions; backend {backend}; setup {time.perf_counter() - t0:.1f}s")
    e_ii = ion_ion_energy(cr)
    results = {}
    for name, sol in solvers.items():
        r = scf_lcao(cr, sol, tol=args.tol, maxiter=args.maxiter, mix=args.mix, verbose=args.verbose)
        E = r["energy"] + e_ii
        last = r["last"]
        res = dict(solver=name, energy_Ha=E, energy_electronic_Ha=r["energy"], E_ion_ion_Ha=e_ii,
                   E_hartree_Ha=r["E_H"], E_xc_Ha=r["E_xc"], iterations=r["iterations"],
                   converged=bool(r["converged"]), time_s=r["time"], n_basis=basis.nb)
        if name == "ks":
            w = np.asarray(last[2])
            nocc = cr.n_elec // 2
            res["homo_Ha"] = float(w[nocc - 1])
            if len(w) > nocc:
                res["lumo_Ha"] = float(w[nocc])
                res["gap_Ha"] = float(w[nocc] - w[nocc - 1])
        else:
            res["chemical_potential_Ha"] = float(last[2])
        if charge == 0:
            res["dipole_au"] = dipole(cr, r["density"]).tolist()
        results[name] = (res, r["density"])
        flag = "" if res["converged"] else "  (NOT CONVERGED)"
        print(f"\n[{name}] E = {E:.8f} Ha ({E * HA_EV:.4f} eV)  iterations {res['iterations']}  {res['time_s']:.1f}s{flag}")
        if "homo_Ha" in res:
            print(f"  HOMO {res['homo_Ha'] * HA_EV:.3f} eV" + (f"  LUMO {res['lumo_Ha'] * HA_EV:.3f} eV  gap {res['gap_Ha'] * HA_EV:.3f} eV" if "gap_Ha" in res else ""))
        if "chemical_potential_Ha" in res:
            print(f"  chemical potential {res['chemical_potential_Ha'] * HA_EV:.3f} eV")
        if "dipole_au" in res:
            d = np.array(res["dipole_au"])
            print(f"  dipole |mu| = {np.linalg.norm(d) / 0.393430307:.4f} Debye")
    summary = {k: v[0] for k, v in results.items()}
    if len(results) == 2:
        dn = float(np.abs(results["lres"][1] - results["ks"][1]).sum() * cr.dV / cr.n_elec)
        dE = summary["lres"]["energy_Ha"] - summary["ks"]["energy_Ha"]
        summary["comparison"] = dict(dE_Ha=dE, dE_per_electron_Ha=dE / cr.n_elec, density_diff_per_electron=dn)
        print(f"\nLRes - KS: dE = {dE:.3e} Ha ({dE / cr.n_elec:.2e} per electron), density diff {dn:.2e} per electron")
    print("\nnote: " + ("GTH-PADE pseudopotentials + GTH-TZV2P basis" if args.model == "gth" else "model local pseudopotentials") + ", closed-shell LDA; energies depend on box/grid, compare only at identical settings.")
    if args.json:
        json.dump(dict(system=args.system, charge=charge, atoms=[list(a) for a in atoms], grid=cr.shape[0], dx=cr.dx,
                       results=summary), open(args.json, "w"), indent=2)
        print(f"wrote {args.json}")
    if args.cube:
        name = "ks" if "ks" in results else "lres"
        write_cube(args.cube, cr, atoms, results[name][1])
        print(f"wrote {args.cube}")
    return 0 if all(v[0]["converged"] for v in results.values()) else 2


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        if args.cmd == "list":
            return cmd_list() or 0
        return cmd_run(args)
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
