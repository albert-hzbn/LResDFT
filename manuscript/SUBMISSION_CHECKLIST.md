# PRB submission checklist

Items the author must complete or verify before submission:

1. **Repository URL**: replace repository URL (https://github.com/albert-hzbn/LResDFT) is filled in `main.tex` and `cover_letter.tex`; consider a Zenodo DOI. Add
   acknowledgments/funding and suggested referees if wanted (both were removed as placeholders).
2. **References.** Entries were entered from memory, except the author's own papers, which were checked online.
   Verify every entry in `refs.bib` (authors, volume, page, year). The least certain are:
   - `Moussa2016`: J. Chem. Phys. volume and page (paper exists, 2016; volume/article not confirmed);
   - `Zeller2008`: article number (not found);
   - (verified Oct 2026: `ELSI2018`, `ZellerDeutz1982`, `Goedecker1993`, `SakuraiSugiura2003`, `AustinTrefethen2015`, `Linda2026Acta`);
   - (also verified via Crossref: `Moussa2016` 145, 164108; `Zeller2008` 20, 294215; `Kumar2026JCIM` 66, 7113);
   - `ZellerDeutz1982`, `Goedecker1993`, `SakuraiSugiura2003`, `AustinTrefethen2015`: added in revision, from memory;
3. **Literature positioning (searched Oct 2026).** The half-step trapezoid filter 1/(1+u^M) is the Butterworth filter of
   Güttel et al. (arXiv:1407.8078, Sec. 3.2); the manuscript now says so and claims only its use as a KS occupation function.
   No prior DFT use found, but not exhaustively ruled out. The closed-form 1D soft-Coulomb exchange is described as a
   derivation (Fourier-space K0 integral noted). Original item: The "what is and is not new" paragraph (Introduction) and Sec. VII rely on the cited
   Green's-function (LSMS/LSGF/KKR), pole-expansion (PEXSI, Moussa), FEAST and divide-and-conquer literature. A final
   search for (i) the half-step trapezoid filter used as a DFT occupation function and (ii) the closed-form 1D
   soft-Coulomb exchange energy is recommended before submission.
4. **Supplemental Material URL** (`SM` entry in `refs.bib`) is filled in by the journal.
5. **Data and code**: deposit the repository (e.g., Zenodo DOI). This includes `lrft/`, `src/`, `tests/`,
   `benchmarks/`, `results/`, `cas_verification.py` and `manuscript/make_*.py`.
6. **Known limitations stated in the paper** (do not remove them):
   - small gaps / metals not treated;
   - deep core states need an optimal pole rule;
   - window error larger for disorder and diffuse bases;
   - local pseudopotentials and LDA only;
   - 3D forces (with Pulay terms) not computed;
   - prefactor;
   - no comparison with ELPA/ELSI or other O(N) codes;
   - parallel efficiency not measured.
7. **Optional strengthening, if a referee asks**:
   - implement an optimal (Zolotarev or minimax) pole rule in the 1D code (comparison already in the paper);
   - a genuine 3D resolvent path (sparse LU or selected inversion per pole) to compare with the DC windows;
   - add finite electronic temperature and a metallic test;
   - compute 3D forces including Pulay terms;
   - compare with ELPA or a sparse Fermi-operator code;
   - test a covalent or ionic solid with nonlocal pseudopotentials.

**Build:**
```
./build_cpp.sh
python3 cas_verification.py
python3 -m pytest -q tests
python3 manuscript/make_figures.py
python3 manuscript/make_tables.py
cd manuscript && latexmk -pdf main.tex supplement.tex cover_letter.tex
```

**Data provenance:** every table body is generated from `results/*.json` by `make_tables.py`, and every figure by
`make_figures.py`.
