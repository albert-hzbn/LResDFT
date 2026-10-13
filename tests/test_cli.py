import json
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lrft import cli, molecules


def test_parse_xyz_charge():
    q, at = molecules.parse_xyz("2\ncharge=1 test\nHe 0 0 0\nH 0 0 0.774\n")
    assert q == 1 and [a[0] for a in at] == ["He", "H"]


def test_unknown_molecule_and_element(tmp_path):
    with pytest.raises(ValueError):
        molecules.load_molecule("nope")
    with pytest.raises(ValueError):
        molecules.check_elements([("Xx", 0, 0, 0)])


def test_open_shell_rejected(capsys):
    assert cli.main(["run", "H2", "--charge", "1", "--backend", "python"]) == 1
    assert "closed-shell" in capsys.readouterr().err


def test_h2_ks_equals_lres_and_json(tmp_path):
    out = tmp_path / "r.json"
    rc = cli.main(["run", "H2", "--solver", "both", "--backend", "python", "--json", str(out)])
    assert rc == 0
    r = json.load(open(out))["results"]
    assert r["ks"]["converged"] and r["lres"]["converged"]
    assert abs(r["comparison"]["dE_Ha"]) < 1e-8          # window covers the molecule: exact
    assert np.linalg.norm(r["ks"]["dipole_au"]) < 1e-6


def test_charged_system_and_ion_ion():
    from lrft.three_d import Crystal
    cr = Crystal(np.array([[5., 5, 5]]), np.array([2.]), np.full(3, 10.), (20, 20, 20), rcs=np.array([0.5]), charge=1)
    assert cr.n_elec == 1
    assert np.isfinite(cli.ion_ion_energy(cr))
