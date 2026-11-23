"""Quick checks against the released data (about 1 min):  python -m pytest tests -q"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rcca import datasets as ds, models as M  # noqa: E402
from rcca.composition import composition_key, parse  # noqa: E402
from rcca.phase import phase_metrics  # noqa: E402
from rcca.screener import grid  # noqa: E402


def test_parse():
    f = parse("Zr333Ti333(NbV)166")
    assert f["Nb"] == pytest.approx(166 / 998) and f["Zr"] == pytest.approx(333 / 998)
    assert parse("HfNbTaTiZr") == pytest.approx({e: 0.2 for e in ["Hf", "Nb", "Ta", "Ti", "Zr"]})
    assert composition_key("Ta20Nb20V20Mo20W20") == composition_key("MoNbTaVW")


def test_grid():
    assert len(grid()) == 43425


def test_training_sets():
    y = ds.load("yield")
    tr = ds.training(y, "yield")
    assert len(tr) == 453 and int(tr.highT_augmented.sum()) == 70 and tr.group.nunique() == 148
    assert len(ds.holdout(y)) == 43 and len(ds.external(y)) == 20
    d = ds.load("ductility")
    assert len(ds.training(d, "ductility")) == 121 and len(ds.holdout(d)) == 14


def test_ductility_model():
    d = ds.load("ductility")
    e = M.train(ds.training(d, "ductility"), "ductility")
    ho = ds.holdout(d)
    s = M.scores(ho.y, e.predict(ho)[0])
    assert s["R2"] == pytest.approx(0.768, abs=0.002) and s["MAE"] == pytest.approx(4.48, abs=0.01)
    assert e.calibration[0.68] == pytest.approx(4.57, abs=0.01)


def test_yield_model():
    y = ds.load("yield")
    e = M.train(ds.training(y, "yield"), "yield")
    ho = ds.holdout(y)
    s = M.scores(ho.y, e.predict(ho)[0])
    # the paper computed the descriptors from the exact reported compositions; the released names are rounded to
    # 0.1 at.%, which changes the hold-out MAE by about 0.01 MPa
    assert s["R2"] == pytest.approx(0.896, abs=0.002) and s["MAE"] == pytest.approx(107.84, abs=0.1)
    assert e.calibration[0.68] == pytest.approx(3.90, abs=0.02)


def test_phase_metrics():
    rel = pd.read_csv(ds.DATASETS / "phase_stability_9element_database.csv")
    sample = rel.groupby("n_elements", group_keys=False).apply(lambda g: g.sample(min(len(g), 30), random_state=0))
    p = phase_metrics(sample.Composition, 1000)
    for c in ("Gmix_bcc_meV_per_atom", "Hmix_bcc_meV_per_atom", "dG_bcc_to_hcp_meV_per_atom"):
        assert np.abs(p[c].round(2).to_numpy() - sample[c].to_numpy()).max() <= 0.0101
