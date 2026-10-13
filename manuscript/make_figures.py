"""Build all manuscript figures (vector PDF) from the stored results and a few
fast 1D calculations.  Run from the repository root:
    python3 manuscript/make_figures.py
"""
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyBboxPatch, Rectangle
from scipy import linalg

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from lrft.core import (Grid, Hartree, System, exchange, external_potential, ks_density,
                       lrft_fixed_N, nodes, scf, tridiag)
from benchmarks.accuracy import dimer_chain

RES = os.path.join(ROOT, "results")
FIG = os.path.join(ROOT, "manuscript", "figures")
os.makedirs(FIG, exist_ok=True)

# palette (categorical slots in fixed order) and ink
BLUE, ORANGE, AQUA, YELLOW, MAGENTA = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"
OCC = "#c2347a"      # occupied levels (distinct from the series colors)
INK, MUTED, GRID = "#1f1f1e", "#6b6a64", "#e4e3dc"
plt.rcParams.update({
    "font.family": "serif", "font.serif": ["DejaVu Serif"], "mathtext.fontset": "dejavuserif",
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8, "legend.fontsize": 6.8,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
    "lines.linewidth": 1.4, "lines.markersize": 4, "axes.linewidth": 0.6,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
})
COL1, COL2 = 3.4, 7.0   # PRB single / double column widths (inches)


def label(ax, s, x=-0.16, y=1.04):
    ax.text(x, y, s, transform=ax.transAxes, fontsize=9, fontweight="bold", color=INK, va="bottom")


def small_chain():
    """(H2)_3 chain used for the concept figure."""
    s = dimer_chain(3)
    g = Grid.around(s, dx=0.05, pad=6)
    r = scf(s, g, "ks", tol=1e-10)
    veff = (external_potential(s, g) + Hartree(g).potential(r.density) + exchange(r.density)[1])
    d, e = tridiag(veff, g.dx)
    eps, U = linalg.eigh_tridiagonal(d, e, select="i", select_range=(0, 60))
    U = U / np.sqrt(g.dx)
    return s, g, r, veff, eps, U


# ----------------------------------------------------------------------------
# Figure 1: concept
# ----------------------------------------------------------------------------

def _head(ax, title, sub):
    """Panel title with a one-line muted subtitle above the axes (nothing is drawn inside the data area)."""
    ax.set_title(title, loc="left", color=INK, pad=14)
    ax.text(0, 1.035, sub, transform=ax.transAxes, fontsize=6.3, color=MUTED, va="bottom")


def fig_concept():
    s, g, r, veff, eps, U = small_chain()
    nocc = s.n_elec // 2
    mu = 0.5 * (eps[nocc - 1] + eps[nocc])
    x = g.x
    e_lo = eps[0] - 0.3
    c, R = 0.5 * (mu + e_lo), 0.5 * (mu - e_lo)
    occ = eps[eps < mu]
    unocc = eps[(eps > mu) & (eps < mu + 0.3)]
    fig = plt.figure(figsize=(COL2, 5.0))
    gs = fig.add_gridspec(2, 3, hspace=0.72, wspace=0.55)

    # (a) density only (orbital-free DFT)
    ax = fig.add_subplot(gs[0, 0])
    ax.fill_between(x, r.density, color=BLUE, alpha=0.18, lw=0)
    ax.plot(x, r.density, color=BLUE)
    for X in s.X:
        ax.plot([X], [0], marker="o", color=INK, ms=3.5, clip_on=False, zorder=5)
    ax.set_xlabel("position $x$ (bohr)")
    ax.set_ylabel("density $n(x)$")
    ax.set_ylim(0, r.density.max() * 1.1)
    _head(ax, "Orbital-free DFT", "only $n(x)$; $T[n]$ modeled")
    label(ax, "(a)", x=-0.21, y=1.075)

    # (b) Kohn-Sham orbitals
    ax = fig.add_subplot(gs[0, 1])
    for i in range(nocc):
        off = 1.1 * i
        phi = U[:, i] / np.max(np.abs(U[:, i])) * 0.45
        ax.axhline(off, color=GRID, lw=0.6)
        ax.plot(x, off + phi, color=[BLUE, ORANGE, AQUA][i])
        ax.text(x[0] - 0.6, off, f"$\\phi_{i + 1}$", va="center", ha="right", fontsize=7, color=INK)
    for X in s.X:
        ax.plot([X], [-0.62], marker="o", color=INK, ms=3.5, clip_on=False)
    ax.set_yticks([])
    ax.set_xlabel("position $x$ (bohr)")
    ax.set_ylim(-0.7, 1.1 * (nocc - 1) + 0.6)
    ax.spines["left"].set_visible(False)
    _head(ax, "Kohn--Sham", "$N/2$ global orbitals")
    label(ax, "(b)", x=-0.21, y=1.075)

    # (c) energy-resolved density (the LRes variable)
    ax = fig.add_subplot(gs[0, 2])
    E = np.linspace(eps[0] - 0.15, mu + 0.45, 400)
    sig = 0.012
    w = np.exp(-((E[:, None] - eps[None, :]) / sig) ** 2 / 2) / (np.sqrt(2 * np.pi) * sig)
    ldos = w @ (U**2).T
    ax.imshow(np.log10(ldos + 1e-3), origin="lower", aspect="auto", cmap="Blues",
              extent=[x[0], x[-1], E[0], E[-1]], vmin=-1, vmax=np.log10(ldos.max()))
    ax.axhline(mu, color=ORANGE, lw=1.0, ls="--")
    ax.text(x[-1] - 0.5, mu + 0.015, "$\\mu$", color=ORANGE, fontsize=8, va="bottom", ha="right")
    ax.set_xlabel("position $x$ (bohr)")
    ax.set_ylabel("energy $\\varepsilon$ (Ha)")
    _head(ax, "Local density of states", "$n=\\int\\rho\\,d\\varepsilon$ up to $\\mu$; $T_s$ exact")
    label(ax, "(c)", x=-0.21, y=1.075)

    # (d) complex energy plane
    ax = fig.add_subplot(gs[1, 0])
    M = 16
    th = np.linspace(0, 2 * np.pi, 400)
    ax.plot(c + R * np.cos(th), R * np.sin(th), color=MUTED, lw=0.8, ls=":")
    zs = c + R * np.exp(1j * np.pi * (2 * np.arange(M) + 1) / M)
    ax.plot(zs.real, zs.imag, "o", color=ORANGE, ms=3.6, mec="white", mew=0.4, zorder=4)
    ax.axhline(0, color=GRID, lw=0.6, zorder=0)
    ax.plot(occ, 0 * occ, "|", color=OCC, ms=9, mew=1.6, zorder=3)
    ax.plot(unocc, 0 * unocc, "|", color=MUTED, ms=9, mew=1.0, zorder=3)
    for xv, txt, col, dx_ in ((e_lo, "$E_\\mathrm{lo}$", MUTED, -0.05), (mu, "$\\mu$", ORANGE, 0.05)):
        ax.plot([xv], [0], "o", ms=3.2, color=col, zorder=5)
        ax.annotate(txt, (xv, 0), xytext=(xv + dx_, -0.75 * R), textcoords="data", ha="center", va="top", fontsize=7,
                    color=col, arrowprops=dict(arrowstyle="-", color=col, lw=0.5, shrinkA=1, shrinkB=2))
    ax.text(occ.mean(), 0.2 * R, "occupied", color=OCC, fontsize=6.3, ha="center", va="bottom")
    ax.text(unocc.mean(), 0.2 * R, "empty", color=MUTED, fontsize=6.3, ha="center", va="bottom")
    ax.text(c, 1.28 * R, "$z_j$, $M=16$", color=ORANGE, fontsize=6.5, ha="center", va="bottom")
    ax.set_xlabel("Re $z$ (Ha)")
    ax.set_ylabel("Im $z$ (Ha)")
    ax.set_xlim(e_lo - 0.1, mu + 0.32)
    ax.set_ylim(-1.2 * R, 1.55 * R)
    ax.set_aspect("equal", adjustable="datalim")
    _head(ax, "Complex energies", "$g(x,z_j)$ on a circle")
    label(ax, "(d)", x=-0.21, y=1.075)

    # (e) smooth occupation
    ax = fig.add_subplot(gs[1, 1])
    ev = np.linspace(e_lo - 0.05, mu + 0.6, 800)
    u = (ev - c) / R
    ax.axvline(mu, color=ORANGE, lw=0.7, ls=":")
    ax.axvline(e_lo, color=MUTED, lw=0.7, ls=":")
    ax.plot(ev, (ev < mu) * (ev > e_lo), color=INK, lw=0.9, ls="--", label="exact step")
    for Mv, col in [(8, YELLOW), (16, AQUA), (64, BLUE)]:
        ax.plot(ev, 1 / (1 + u**Mv), color=col, label=f"$M={Mv}$")
    ax.plot(occ, 0 * occ - 0.07, "|", color=OCC, ms=6, mew=1.4, clip_on=False)
    ax.plot(unocc, 0 * unocc - 0.07, "|", color=MUTED, ms=6, mew=1.0, clip_on=False)
    ax.set_ylim(-0.1, 1.12)
    ax.set_xlabel("energy $\\varepsilon$ (Ha)")
    ax.set_ylabel("occupation")
    from matplotlib.lines import Line2D
    hs, ls = ax.get_legend_handles_labels()
    hs += [Line2D([], [], marker="|", ls="", color=OCC, ms=7, mew=1.4), Line2D([], [], marker="|", ls="", color=MUTED, ms=7, mew=1.0)]
    ls += ["occupied levels", "empty levels"]
    ax.legend(hs, ls, frameon=False, loc="upper left", bbox_to_anchor=(-0.02, -0.3), fontsize=6.3, handlelength=1.6,
              labelspacing=0.25, ncol=2, columnspacing=1.0, borderaxespad=0)
    _head(ax, "Occupation", "$r_M=1/(1+u^M)$")
    label(ax, "(e)", x=-0.21, y=1.075)

    # (f) nearsightedness: resolvent decay
    ax = fig.add_subplot(gs[1, 2])
    s2 = dimer_chain(24)
    g2 = Grid.around(s2, dx=0.1, pad=8)
    r2 = scf(s2, g2, "ks", tol=1e-8)
    v2 = external_potential(s2, g2) + Hartree(g2).potential(r2.density) + exchange(r2.density)[1]
    d2, e2 = tridiag(v2, g2.dx)
    res, mu2, elo2 = lrft_fixed_N(v2, g2.dx, s2.n_elec, 32)
    zs2, ws2 = nodes(32, mu2, elo2)
    H = np.diag(d2) + np.diag(e2, 1) + np.diag(e2, -1)
    i0 = np.argmin(np.abs(g2.x - s2.X[len(s2.X) // 2]))
    dist = np.abs(g2.x - g2.x[i0])
    P = np.zeros(g2.n)
    Rw = 12.0
    ax.axvspan(Rw, 40, color=GRID, alpha=0.55, lw=0, zorder=0)
    for k, (z, wgt) in enumerate(zip(zs2, ws2)):
        rhs = np.zeros(g2.n, complex)
        rhs[i0] = 1
        col = np.linalg.solve(z * np.eye(g2.n) - H, rhs)
        P += 4 * np.real(wgt * col) / 2
        if k in (0, len(zs2) - 1):
            lab = "$z_j$ near $\\mu$" if k == 0 else "$z_j$ far from $\\mu$"
            ax.semilogy(dist[i0:], np.abs(col[i0:]) + 1e-18, color=[ORANGE, AQUA][k > 0],
                        lw=1.0, label=f"$|G(x_0,x;z_j)|$, {lab}")
    ax.semilogy(dist[i0:], np.abs(P[i0:]) + 1e-18, color=BLUE, lw=1.2, label="density-matrix row")
    ax.set_xlim(0, 40)
    ax.set_ylim(1e-12, 1e4)
    ax.text(Rw + 1.2, 3e3, "outside the window", fontsize=6.3, color=MUTED, va="top")
    ax.annotate("", xy=(Rw, 2e2), xytext=(0, 2e2), arrowprops=dict(arrowstyle="<->", color=INK, lw=0.7))
    ax.text(Rw / 2, 6e2, "$R_w$", fontsize=7, color=INK, ha="center", va="bottom")
    ax.set_xlabel("distance $|x-x_0|$ (bohr)")
    ax.set_ylabel("magnitude")
    ax.legend(frameon=False, fontsize=6.3, loc="upper left", bbox_to_anchor=(-0.02, -0.3), handlelength=1.6,
              labelspacing=0.25, borderaxespad=0)
    _head(ax, "Nearsightedness", "decay allows a local window")
    label(ax, "(f)", x=-0.21, y=1.075)
    fig.savefig(os.path.join(FIG, "fig1_concept.pdf"))
    plt.close(fig)


# ----------------------------------------------------------------------------
# Figure 3: convergence (energy samples M in 1D, window radius in 3D)
# ----------------------------------------------------------------------------

def fig_convergence():
    rc = json.load(open(os.path.join(RES, "referee_checks.json")))
    r13 = rc["R13_theorem1_bound"]
    fig, axs = plt.subplots(1, 2, figsize=(COL2, 2.6))
    ax = axs[0]
    M = [r["M"] for r in r13["rows"]]
    N = r13["N"]
    ax.semilogy(M, [r["err_per_e"] for r in r13["rows"]], "o-", color=BLUE, label="measured")
    ax.semilogy(M, [r["bound_total"] / N for r in r13["rows"]], "s--", color=ORANGE,
                label="bound of Theorem 1")
    ax.axhspan(1e-18, 1e-15, color=GRID, alpha=0.6, lw=0)
    ax.set_ylim(1e-18, 10)
    ax.set_xlabel("number of poles $M$")
    ax.set_ylabel("density error per electron")
    ax.set_title("1D (H$_2$)$_{16}$ chain", loc="left", color=INK)
    ax.legend(frameon=False, loc="upper right")
    label(ax, "(a)", x=-0.13)
    ax = axs[1]
    series = []
    r1 = rc["R1_dc_comparison"]
    series.append(("crystal $a=6.0$, $\\Delta$=%.2f" % r1["gap"], [(r["R_w"], r["LRFT_M64_dn"]) for r in r1["rows"]], BLUE, "o"))
    am = rc["R3_amorphous"]
    series.append(("amorphous, $\\Delta$=%.2f" % am["gap"], [(r["R_w"], r["dn_per_e"]) for r in am["rows"]], MAGENTA, "D"))
    cols = {5.5: AQUA, 5.0: YELLOW, 4.5: ORANGE}
    for a in (5.5, 5.0, 4.5):
        rr = [r for r in rc["R2_gap_dependence"] if r["a"] == a and r["M"] == 64]
        series.append(("crystal $a=%.1f$, $\\Delta$=%.2f" % (a, rr[0]["gap"]), [(r["R_w"], r["dn_per_e"]) for r in rr], cols[a], "s"))
    for lab, pts, col, mk in series:
        x, y = zip(*pts)
        ax.semilogy(x, y, marker=mk, color=col, label=lab)
    ax.set_xlabel("window radius $R_w$ (bohr)")
    ax.set_ylabel("density error per electron")
    ax.set_title("3D H$_2$ systems", loc="left", color=INK)
    ax.legend(frameon=False, loc="lower left", fontsize=6)
    label(ax, "(b)", x=-0.13)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig3_convergence.pdf"))
    plt.close(fig)


def fig_rules():
    rc = json.load(open(os.path.join(RES, "referee_checks.json")))["R10_pole_rules"]
    fig, axs = plt.subplots(1, 3, figsize=(COL2, 2.4))
    keys = [("half_step_trapezoid", "half-step trapezoid", "o", BLUE),
            ("gauss_legendre_circle", "Gauss-Legendre (circle)", "s", AQUA),
            ("zolotarev", "Zolotarev", "^", ORANGE)]
    for r in rc:
        r["occ_over_gap"] = r["occ_width"] / r["gap"]
    panels = [("grid", "spectral_width", "spectral width $W$ (Ha)", True, "(a) grid refinement"),
              ("gap", "gap", "gap $\\Delta$ (Ha)", True, "(b) closing gap"),
              ("both", "occ_over_gap", "$W_\\mathrm{occ}/\\Delta$", True, "(c) all gapped series")]
    for ax, (ser, xk, xl, logx, title) in zip(axs, panels):
        if ser == "both":
            for sub, mkfill in (("gap", None), ("deep_core", "white")):
                rows = sorted([r for r in rc if r["series"] == sub], key=lambda r: r[xk])
                for k, lab, mk, col in keys:
                    ax.plot([r[xk] for r in rows], [r[k] for r in rows], marker=mk, color=col,
                            mfc=mkfill or col, ls="-" if sub == "gap" else ":")
            ax.text(0.03, 0.97, "solid: H chains\ndotted, open: deep core", transform=ax.transAxes,
                    va="top", fontsize=6, color=MUTED)
        else:
            rows = [r for r in rc if r["series"] == ser]
            for k, lab, mk, col in keys:
                ax.plot([r[xk] for r in rows], [r[k] for r in rows], marker=mk, color=col, label=lab)
        from matplotlib.ticker import ScalarFormatter, NullFormatter
        if logx:
            ax.set_xscale("log")
            ax.xaxis.set_major_formatter(ScalarFormatter())
            ax.xaxis.set_minor_formatter(NullFormatter())
        if ser == "grid":
            ax.set_ylim(0, 25)
        else:
            ax.set_yscale("log")
            ax.yaxis.set_major_formatter(ScalarFormatter())
            ax.yaxis.set_minor_formatter(NullFormatter())
            ax.set_yticks([10, 20, 50, 100, 200, 400])
        if ser == "gap":
            ax.set_xticks([0.015, 0.03, 0.06, 0.1])
            ax.invert_xaxis()
        ax.set_xlabel(xl)
        ax.set_title(title, loc="left", color=INK)
    axs[0].set_ylabel("complex solves for $10^{-6}$")
    axs[0].legend(frameon=False, fontsize=6, loc="lower right")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig4_rules.pdf"))
    plt.close(fig)


# ----------------------------------------------------------------------------
# Figure 4: accuracy (binding curve vs exact; density vs KS)
# ----------------------------------------------------------------------------

def fig_accuracy():
    ex = json.load(open(os.path.join(RES, "accuracy.json")))["exact"]
    h2 = [r for r in ex if r["system"].startswith("H2")]
    R = [float(r["system"].split("=")[1]) for r in h2]
    fig = plt.figure(figsize=(COL2, 3.0))
    gs = fig.add_gridspec(2, 2, width_ratios=[1, 1.05], height_ratios=[1, 1.25], hspace=0.12, wspace=0.34)
    ax = fig.add_subplot(gs[:, 0])
    ax.plot(R, [r["E_exact"] for r in h2], "o-", color=INK, label="exact many-body")
    ax.plot(R, [r["E_KS"] for r in h2], "s-", color=ORANGE, label="KS (exchange-only LDA)", ms=5, mfc="none")
    ax.plot(R, [r["E_LRFT"] for r in h2], "x--", color=BLUE, label="LRes-DFT (exchange-only LDA)", ms=5)
    ax.set_xlabel("bond length $R$ (bohr)")
    ax.set_ylabel("total energy (Ha)")
    ax.set_ylim(-1.5, -1.0)
    ax.set_title("1D H$_2$ binding curve", loc="left", color=INK, pad=8)
    ax.legend(frameon=False, loc="upper left", fontsize=6.4)
    label(ax, "(a)", x=-0.17)

    s = dimer_chain(8)
    g = Grid.around(s, dx=0.1, pad=8)
    ks = scf(s, g, "ks", tol=1e-11)
    l16 = scf(s, g, "lrft", M=16, tol=1e-11)
    l64 = scf(s, g, "lrft", M=64, tol=1e-11)
    top = fig.add_subplot(gs[0, 1])
    top.plot(g.x, ks.density, color=INK, lw=1.0)
    top.fill_between(g.x, ks.density, color=INK, alpha=0.08, lw=0)
    top.set_ylabel("$n_\\mathrm{KS}$ (bohr$^{-1}$)")
    top.set_ylim(0, ks.density.max() * 1.15)
    top.tick_params(labelbottom=False)
    top.set_title("1D (H$_2$)$_8$ chain", loc="left", color=INK, pad=8)
    label(top, "(b)", x=-0.17)
    bot = fig.add_subplot(gs[1, 1], sharex=top)
    bot.semilogy(g.x, np.abs(l16.density - ks.density) + 1e-17, color=YELLOW, lw=1.0, label="$M=16$")
    bot.semilogy(g.x, np.abs(l64.density - ks.density) + 1e-17, color=BLUE, lw=1.0, label="$M=64$")
    bot.set_ylim(1e-16, 1e-1)
    bot.set_yticks([1e-15, 1e-10, 1e-5])
    bot.set_xlabel("position $x$ (bohr)")
    bot.set_ylabel("$|n_M-n_\\mathrm{KS}|$")
    bot.legend(frameon=False, loc="lower center", ncol=2, fontsize=6.4, borderaxespad=0.1)
    fig.savefig(os.path.join(FIG, "fig5_accuracy.pdf"))
    plt.close(fig)


# ----------------------------------------------------------------------------
# Figure 5: cost (1D and 3D)
# ----------------------------------------------------------------------------

def fig_cost():
    s1 = json.load(open(os.path.join(RES, "scaling.json")))
    s3 = json.load(open(os.path.join(RES, "three_d.json")))
    fig, axs = plt.subplots(1, 2, figsize=(COL2, 2.75))
    ax = axs[0]
    ser = [("t_lrft", "LRes-DFT, recursion", "o", BLUE), ("t_lrft_near", "LRes-DFT, windows", "s", AQUA),
           ("t_ks_tridiag", "KS, tridiagonal (stebz)", "^", ORANGE), ("t_ks_dense", "KS, dense", "D", YELLOW)]
    for k, lab, mk, col in ser:
        pts = [(r["N_e"], r[k]) for r in s1["rows"] if k in r]
        x, y = zip(*pts)
        ax.loglog(x, y, marker=mk, color=col, label=lab)
    ax.set_xlabel("number of electrons $N$")
    ax.set_ylabel("time per density build (s)")
    ax.set_title("1D chains", loc="left", color=INK)
    ax.legend(frameon=False, loc="lower right", fontsize=6)
    label(ax, "(a)", x=-0.15)
    ax = axs[1]
    rows = s3["cost"]
    for k, lab, mk, col, ls in [("t_lrft", "LRes-DFT, warm step", "o", BLUE, "-"),
                                ("t_lrft_cold", "LRes-DFT, cold step", "o", AQUA, "--"),
                                ("t_ks", "KS, dense (dsygvx)", "D", YELLOW, "-")]:
        pts = [(r["N_e"], r[k]) for r in rows if k in r]
        x, y = zip(*pts)
        ax.loglog(x, y, marker=mk, color=col, ls=ls, label=lab, mfc="white" if ls == "--" else col)
    ax.set_xlabel("number of electrons $N$")
    ax.set_ylabel("time per density build (s)")
    ax.set_title("3D H$_2$ crystals", loc="left", color=INK)
    ax.legend(frameon=False, loc="upper left", fontsize=6)
    label(ax, "(b)", x=-0.15)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig6_cost.pdf"))
    plt.close(fig)


def fig_sm():
    """Supplemental figures: SCF histories, margin sensitivity, window spectra."""
    rc = json.load(open(os.path.join(RES, "referee_checks.json")))
    fig, axs = plt.subplots(1, 3, figsize=(COL2, 2.4))
    ax = axs[0]
    h = rc["R6_scf_histories"]
    for k, col, mk in [("1D_KS", ORANGE, "^"), ("1D_LRFT", BLUE, "o"), ("3D_KS", YELLOW, "D"), ("3D_LRFT", AQUA, "s")]:
        ax.semilogy(range(1, len(h[k]) + 1), h[k], marker=mk, ms=3, color=col, label=k.replace("LRFT", "LRes-DFT").replace("_", " "))
    ax.set_xlabel("SCF iteration")
    ax.set_ylabel("density change per electron")
    ax.legend(frameon=False, fontsize=6)
    ax.set_title("(a) SCF convergence", loc="left", color=INK)
    ax = axs[1]
    r5 = rc["R5_margin"]["rows"]
    for M, col in [(32, AQUA), (48, BLUE)]:
        rr = [r for r in r5 if r["M"] == M]
        ax.loglog([r["delta"] for r in rr], [r["err_per_e"] for r in rr], "o-", color=col, label=f"$M={M}$")
    ax.axvline(rc["R5_margin"]["gap"] / 2, color=MUTED, ls=":", lw=1)
    ax.text(rc["R5_margin"]["gap"] / 2 * 1.08, 3e-3, "$\\Delta/2$", color=MUTED, fontsize=7)
    ax.set_xlabel("margin $\\delta$ (Ha)")
    ax.set_ylabel("density error per electron")
    ax.legend(frameon=False, fontsize=6)
    ax.set_title("(b) margin $\\delta$", loc="left", color=INK)
    ax = axs[2]
    wd = rc["R12_window_diagnostics"]
    for name, col in [("crystal", BLUE), ("amorphous", MAGENTA)]:
        d = wd[name]["12.1"]
        e = np.array(d["hist_edges"])
        ax.stairs(d["hist_counts"], e, color=col, label=name)
        ax.axvline(d["homo"], color=col, ls=":", lw=0.8)
        ax.axvline(d["lumo"], color=col, ls="--", lw=0.8)
    ax.set_yscale("log")
    ax.set_xlabel("window eigenvalue (Ha)")
    ax.set_ylabel("count")
    ax.legend(frameon=False, fontsize=6)
    ax.set_title("(c) window spectra, $R_w=12.1$", loc="left", color=INK)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "figS1_supplement.pdf"))
    plt.close(fig)


def fig_grid():
    d = json.load(open(os.path.join(RES, "extra_checks.json")))["grid_refinement"]
    fig, ax = plt.subplots(figsize=(COL1, 2.4))
    dx = [r["dx"] for r in d]
    ax.loglog(dx, [r["cheb_degree_needed"] for r in d], "s-", color=ORANGE,
              label="Chebyshev expansion: polynomial degree")
    ax.loglog(dx, [r["M_needed"] for r in d], "o-", color=BLUE, label="LRFT: number of complex energies $M$")
    for r in d:
        ax.annotate(f"$W$={r['spectral_width']:.0f}", (r["dx"], r["cheb_degree_needed"]), xytext=(0, 6),
                    textcoords="offset points", ha="center", fontsize=6, color=MUTED)
    ax.invert_xaxis()
    ax.set_xticks(dx)
    ax.set_xticklabels([str(v) for v in dx])
    ax.minorticks_off()
    ax.set_xlim(0.25, 0.02)
    ax.set_xlabel("grid spacing $\\Delta x$ (bohr)  (finer $\\rightarrow$)")
    ax.set_ylabel("count for $10^{-6}$ accuracy")
    ax.set_ylim(10, 1e5)
    ax.legend(frameon=False, loc=(0.03, 0.33), fontsize=6.2)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig6_grid.pdf"))
    plt.close(fig)


if __name__ == "__main__":
    which = sys.argv[1:] or ["concept", "convergence", "rules", "accuracy", "cost", "sm"]
    for w in which:
        globals()[f"fig_{w}"]()
        print("done", w, flush=True)
