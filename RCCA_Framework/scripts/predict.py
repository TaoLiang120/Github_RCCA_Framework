"""Predict yield strength, compressive fracture strain and phase metrics for your own compositions.

    python scripts/predict.py input.csv output.csv [--temperatures 295 1475] [--models results]

input.csv needs a column Composition (e.g. Ta20Nb20V20Mo20W20 or HfNbTaTiZr); an optional column temperature (K)
gives one yield-strength temperature per row, otherwise --temperatures is used for every row.
The models are read from --models (models_yield.pkl, models_ductility.pkl written by reproduce_paper.py); if they
are missing they are trained first (a few seconds).
Output columns: yield strength and fracture strain (mean of the 100 models) with the half-widths of the calibrated
68% (= 1 calibrated s.d.) and 95% intervals, the distance to the nearest curated composition of each dataset, and,
for compositions of three or more of Ti, V, Cr, Zr, Nb, Mo, Hf, Ta and W, the phase metrics at 300 and 1000 K.
Predictions far from the curated compositions (distance above about 0.2) are extrapolations.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from rcca import datasets as ds, models as M  # noqa: E402
from rcca.composition import parse  # noqa: E402
from rcca.features import add_features  # noqa: E402
from rcca.phase import phase_metrics  # noqa: E402
from rcca.screener import ELEMENTS, nearest  # noqa: E402


def get_models(path):
    ens = {}
    for prop in ("yield", "ductility"):
        f = Path(path) / f"models_{prop}.pkl"
        if f.exists():
            ens[prop] = M.Ensemble.load(f)
        else:
            df = ds.load(prop)
            ens[prop] = M.train(ds.training(df, prop), prop)
            Path(path).mkdir(parents=True, exist_ok=True)
            ens[prop].save(f)
    return ens


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input")
    ap.add_argument("output")
    ap.add_argument("--temperatures", type=float, nargs="+", default=[295.0, 1475.0])
    ap.add_argument("--models", default=str(ROOT / "results"))
    a = ap.parse_args()
    inp = pd.read_csv(a.input)
    if "temperature" not in inp:
        inp = pd.concat([inp.assign(temperature=T) for T in a.temperatures], ignore_index=True)
    ens = get_models(a.models)
    f = add_features(inp)
    out = inp.copy()
    for prop, name, unit in (("yield", "yield_strength", "MPa"), ("ductility", "fracture_strain", "pct")):
        mean, h68 = ens[prop].predict_calibrated(f, 0.68)
        _, h95 = ens[prop].predict_calibrated(f, 0.95)
        out[f"{name}_pred_{unit}"] = mean
        out[f"{name}_68pct_halfwidth_{unit}"] = h68
        out[f"{name}_95pct_halfwidth_{unit}"] = h95
    out["T_over_Tm"] = f.temp_ratio
    for prop, col in (("yield", "Yield_dataset.csv"), ("ductility", "Compression.csv")):
        dist, near = nearest(out.Composition, pd.read_csv(ds.DATASETS / col).Composition)
        out[f"distance_{prop}_dataset"], out[f"nearest_{prop}_composition"] = dist, near
    ok = [set(parse(c)) <= set(ELEMENTS) and len(parse(c)) >= 3 for c in out.Composition]
    for T in (300, 1000):
        ph = phase_metrics(out.Composition[ok], T).drop(columns=["Composition", "temperature_K"])
        for c in ph.columns:
            out[f"{c.replace('_meV_per_atom', '')}_{T}K_meV_per_atom"] = np.nan
            out.loc[ok, f"{c.replace('_meV_per_atom', '')}_{T}K_meV_per_atom"] = ph[c].to_numpy()
    out.to_csv(a.output, index=False, float_format="%.4g")
    print(out.to_string(max_cols=8))


if __name__ == "__main__":
    main()
