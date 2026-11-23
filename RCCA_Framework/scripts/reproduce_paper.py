"""Reproduce the models, the nine-element database and the candidates of the paper from ../Datasets.

    python scripts/reproduce_paper.py [--cvrsa] [--out results] [--n_jobs -1]

Steps
  1  datasets: descriptors, development/hold-out/external rows, Maresca-Curtin augmentation (yield strength)
  2  (--cvrsa) rerun CVRSA (nested choice of q*, then the final run; about 10 min on 10 cores);
     otherwise the released Survivor_CVRSA flags are used
  3  10 x 10 composition-grouped models (yield strength, compressive fracture strain), calibration factors,
     hold-out and external-test scores
  4  predictions for the 43,425 compositions at 295, 1475 and 1800 K; phase metrics at 1000 K (component expansion)
  5  candidate screen (criteria of the paper)
  6  comparison with the released files in ../Datasets
Outputs in --out: models_<prop>.pkl, scores.csv, predictions_9element_database.csv,
phase_stability_9element_database.csv, candidates.csv, candidates_top12.csv, funnel.csv, comparison.txt
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from rcca import cvrsa, datasets as ds, models as M  # noqa: E402
from rcca.composition import composition_key  # noqa: E402
from rcca.features import add_features  # noqa: E402
from rcca.phase import phase_metrics  # noqa: E402
from rcca.screener import grid, screen  # noqa: E402

TEMPS = (295.0, 1475.0, 1800.0)


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cvrsa", action="store_true", help="rerun CVRSA instead of using the released flags")
    ap.add_argument("--out", default=str(ROOT / "results"))
    ap.add_argument("--n_jobs", type=int, default=-1)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    report = []

    ens, rows = {}, []
    for prop in ("yield", "ductility"):
        df = ds.load(prop)
        dev = ds.development(df)
        surv = None
        if a.cvrsa:
            res = cvrsa.run(dev, prop, M.SETTINGS[prop]["features"], n_jobs=a.n_jobs)
            surv = res["survivors"]
            released = dev.Survivor_CVRSA.astype(str).str.lower().eq("true").to_numpy()
            report.append(f"CVRSA {prop}: q* = {res['q_star']:.2f}, excluded {int((~surv).sum())} of {len(dev)}; "
                          f"same observations as released: {bool((surv == released).all())}")
            res["curve"].to_csv(out / f"cvrsa_curve_{prop}.csv", index=False)
        tr = ds.training(df, prop, surv)
        log(f"{prop}: {len(tr)} training rows ({tr.group.nunique()} compositions)")
        e = M.train(tr, prop, n_jobs=a.n_jobs)
        e.save(out / f"models_{prop}.pkl")
        ens[prop] = e
        for name, part in (("hold-out", ds.holdout(df)), ("external test", ds.external(df))):
            mean, sd = e.predict(part)
            s = M.scores(part.y, mean)
            pm = M.per_model_scores(e, part)
            cov = float(np.mean(np.abs(part.y - mean) <= e.calibration[0.68] * sd))
            rows.append(dict(property=prop, set=name, n=len(part), R2_ensemble=s["R2"], MAE_ensemble=s["MAE"],
                             R2_per_model_mean=pm.R2.mean(), R2_per_model_sd=pm.R2.std(),
                             MAE_per_model_mean=pm.MAE.mean(), MAE_per_model_sd=pm.MAE.std(),
                             k68=e.calibration[0.68], k95=e.calibration[0.95], within_1_calibrated_sd=cov))
    scores = pd.DataFrame(rows)
    scores.to_csv(out / "scores.csv", index=False, float_format="%.4g")
    report.append(scores.round(3).to_string(index=False))
    log("models trained")

    # database
    comps = grid()
    db = pd.DataFrame({"Composition": comps})
    feats = add_features(db)
    parts = []
    for T in TEMPS:
        f = feats.assign(temperature=T, temp_ratio=np.minimum(T / feats.Tm_ROM, 1.0))
        ys, ys68 = ens["yield"].predict_calibrated(f, 0.68)
        _, ys95 = ens["yield"].predict_calibrated(f, 0.95)
        parts.append(pd.DataFrame({"Composition": comps, "temperature_K": int(T), "yield_strength_pred_MPa": ys,
                                   "yield_strength_68pct_halfwidth_MPa": ys68,
                                   "yield_strength_95pct_halfwidth_MPa": ys95}))
    du, du68 = ens["ductility"].predict_calibrated(feats, 0.68)
    _, du95 = ens["ductility"].predict_calibrated(feats, 0.95)
    pred = pd.concat(parts, ignore_index=True)
    dmap = pd.DataFrame({"Composition": comps, "fracture_strain_pred_pct": du,
                         "fracture_strain_68pct_halfwidth_pct": du68, "fracture_strain_95pct_halfwidth_pct": du95})
    pred = pred.merge(dmap, on="Composition")
    pred.to_csv(out / "predictions_9element_database.csv", index=False, float_format="%.4f")
    log("database predicted; phase metrics at 1000 K ...")
    ph = phase_metrics(comps, 1000)
    ph.to_csv(out / "phase_stability_9element_database.csv", index=False, float_format="%.4f")

    # screen
    p1475 = pred[pred.temperature_K == 1475].set_index("Composition")
    t = pd.DataFrame({"Composition": comps}).assign(
        YS=p1475.yield_strength_pred_MPa.reindex(comps).to_numpy(),
        YS_sd=p1475.yield_strength_68pct_halfwidth_MPa.reindex(comps).to_numpy(),
        eps=du, eps_sd=du68, Gmix_bcc=ph.Gmix_bcc_meV_per_atom.to_numpy(),
        dG_b2h=ph.dG_bcc_to_hcp_meV_per_atom.to_numpy())
    y_comp = pd.read_csv(ds.DATASETS / "Yield_dataset.csv").Composition
    d_comp = pd.read_csv(ds.DATASETS / "Compression.csv").Composition
    cand, top, funnel = screen(t, y_comp, d_comp)
    cand.to_csv(out / "candidates.csv", index=False, float_format="%.4g")
    top.to_csv(out / "candidates_top12.csv", index=False, float_format="%.4g")
    funnel.to_csv(out / "funnel.csv", index=False)
    report.append(funnel.to_string(index=False))

    # comparison with the released files
    key = lambda s: s.map(composition_key)  # noqa: E731
    rp = pd.read_csv(ds.DATASETS / "predictions_9element_database.csv")
    m = pred.assign(k=key(pred.Composition)).merge(rp.assign(k=key(rp.Composition)), on=["k", "temperature_K"])
    for c in ("yield_strength_pred_MPa", "fracture_strain_pred_pct"):
        d = (m[c + "_x"] - m[c + "_y"]).abs()
        report.append(f"{c}: {len(m)} rows, |difference| to the released file median {d.median():.3g}, "
                      f"95th percentile {d.quantile(0.95):.3g}, max {d.max():.3g}")
    rph = pd.read_csv(ds.DATASETS / "phase_stability_9element_database.csv")
    m = ph.assign(k=key(ph.Composition)).merge(rph.assign(k=key(rph.Composition)), on="k")
    d = max((m[c + "_x"].round(2) - m[c + "_y"]).abs().max() for c in
            ("Gmix_bcc_meV_per_atom", "Hmix_bcc_meV_per_atom", "dG_bcc_to_hcp_meV_per_atom"))
    report.append(f"phase metrics: {len(m)} compositions, max |difference| to the released file {d:.3g} meV/atom")
    rc = pd.read_csv(ds.DATASETS / "Candidates.csv")
    mine, theirs = set(key(cand.Composition)), set(key(rc.Composition))
    report.append(f"candidates: {len(mine)} (released {len(theirs)}), shared {len(mine & theirs)}; "
                  f"top 12 shared with Table S8: {len(set(key(top.Composition)) & set(key(rc[rc.in_Table_S8].Composition)))}")
    txt = "\n".join(report)
    (out / "comparison.txt").write_text(txt + "\n")
    print(txt)


if __name__ == "__main__":
    main()
