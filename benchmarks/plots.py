"""Figures for the report (static PNGs, light surface)."""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]   # categorical slots 1-4, fixed order
INK, MUTED, GRID = "#1f1f1e", "#6b6a64", "#e4e3dc"

plt.rcParams.update({"font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
                     "grid.linewidth": 0.6, "lines.linewidth": 2, "lines.markersize": 6})


def scaling():
    d = json.load(open(os.path.join(RES, "scaling.json")))
    rows = d["rows"]
    fig, ax = plt.subplots(figsize=(7, 4.6))
    series = [("t_lrft", "LRFT (exact O(N) recursion)", "o"),
              ("t_lrft_near", "LRFT (nearsighted blocks)", "s"),
              ("t_ks_tridiag", "KS, 1D-optimal tridiagonal LAPACK", "^"),
              ("t_ks_dense", "KS, dense diagonalisation", "D")]
    for (key, label, mk), col in zip(series, SERIES):
        pts = [(r["N_e"], r[key]) for r in rows if key in r]
        x, y = zip(*pts)
        p = d["exponents"].get(key)
        ax.loglog(x, y, marker=mk, color=col, label=f"{label}  (~N^{p:.2f})" if p else label)
        ax.annotate(f"N^{p:.1f}" if p else "", (x[-1], y[-1]), xytext=(6, 0),
                    textcoords="offset points", color=INK, va="center", fontsize=9)
    ax.set_xlabel("number of electrons N")
    ax.set_ylabel("time per density build n[v] (s, 1 thread)")
    ax.set_title("Cost of one self-consistent step", color=INK, loc="left")
    ax.legend(frameon=False, fontsize=8.5, loc="lower right")
    fig.tight_layout()
    fig.savefig(os.path.join(RES, "scaling.png"), dpi=150)


def convergence():
    d = json.load(open(os.path.join(RES, "accuracy.json")))["convergence"]
    rows = d["rows"]
    M = [r["M"] for r in rows]
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.semilogy(M, [r["err_density_L1"] for r in rows], marker="o", color=SERIES[0],
                label="measured density error  ||n_M - n_KS||_1")
    ax.semilogy(M, [r["bound"] for r in rows], marker="s", color=SERIES[1], linestyle="--",
                label="a-priori bound  N rho^M / (1+rho^M)  (Theorem 3)")
    ax.axhline(1e-13, color=MUTED, linewidth=1, linestyle=":")
    ax.text(M[0], 2e-13, "double-precision floor", color=MUTED, fontsize=8)
    ax.set_xlabel("number of complex energies M")
    ax.set_ylabel("error")
    ax.set_title("Exponential convergence in M ((H2)_16 chain)", color=INK, loc="left")
    ax.legend(frameon=False, fontsize=8.5)
    fig.tight_layout()
    fig.savefig(os.path.join(RES, "convergence.png"), dpi=150)


def scaling3d():
    d = json.load(open(os.path.join(RES, "three_d.json")))
    rows = d["cost"]
    fig, ax = plt.subplots(figsize=(7, 4.6))
    series = [("t_lrft", "LRFT 3D (nearsighted windows)", "o"),
              ("t_ks", "KS 3D (LCAO, dense generalised eigensolver)", "D")]
    for (key, label, mk), col in zip(series, [SERIES[0], SERIES[3]]):
        pts = [(r["N_e"], r[key]) for r in rows if key in r]
        x, y = zip(*pts)
        p = d["exponents"].get(key)
        ax.loglog(x, y, marker=mk, color=col, label=f"{label}  (~N^{p:.2f})")
        ax.annotate(f"N^{p:.1f}", (x[-1], y[-1]), xytext=(6, 0), textcoords="offset points",
                    color=INK, va="center", fontsize=9)
    ax.set_xlabel("number of electrons N  (H2 molecular crystal, n x n x n cells)")
    ax.set_ylabel("time per density build n[v] (s, 1 thread)")
    ax.set_title("3D: cost of one self-consistent step", color=INK, loc="left")
    ax.legend(frameon=False, fontsize=8.5, loc="lower right")
    fig.tight_layout()
    fig.savefig(os.path.join(RES, "scaling3d.png"), dpi=150)


if __name__ == "__main__":
    scaling()
    convergence()
    if os.path.exists(os.path.join(RES, "three_d.json")):
        scaling3d()
