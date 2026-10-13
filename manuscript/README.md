# Manuscript: Local resolvent field theory (target: Physical Review B)

Build (from the repository root):

    python3 benchmarks/extra_checks.py   # grid-refinement vs Chebyshev, force consistency
    python3 manuscript/make_figures.py   # vector figures from results/ (+ fast 1D runs)
    python3 manuscript/make_tables.py    # LaTeX tables generated directly from results/*.json
    cd manuscript && latexmk -pdf main.tex

Contents: `main.tex` (REVTeX 4.2, PRB two-column), `refs.bib`, `figures/` (Figs. 1, 3-6;
Fig. 2, the algorithm flowchart, is drawn in TikZ inside main.tex), `tables/` (auto-generated).

To fill in before submission: author surname, affiliation, acknowledgments, and the
repository URL in the data-availability sentence. Every number in the tables comes from
`results/*.json`; the figure numbers come from the same files.
