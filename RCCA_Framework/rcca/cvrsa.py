"""Cross-validated residual stability analysis (CVRSA), Supplementary Information section "CVRSA and dataset analysis".

An observation is a candidate for exclusion if its out-of-fold residual is persistently large: in repeated
composition-grouped k-fold cross-validation, |y - y_oof| exceeds z_thres times a fixed robust scale
s0 = 1.4826 MAD (of the residuals of the first iteration) in at least majority_frac of the repeats.
At most `batch` such observations (most often flagged first) are excluded per iteration, and the cross-validation
is repeated, until none is eligible or the fraction q* of the observations has been excluded.

q* is chosen by nested cross-validation: the compositions are split into n_outer folds (for each outer seed);
CVRSA runs inside the training part, and a model trained after excluding the first k = round(q n) observations of
the exclusion order is scored on the untouched validation part, for q = 0, q_step, ..., q_max. q* is the smallest q
whose mean paired MAE difference to q = 0 lies within se_factor standard errors of the minimum.
"""
import time

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import RepeatedStratifiedKFold

SEED = 20261109
SETTINGS = dict(n_splits=5, n_repeats=50, n_repeats_nested=20, batch=5, q_max=0.25, q_step=0.02, n_outer=5,
                outer_seeds=(0, 1), se_factor=0.5, seed=SEED)
# per-property thresholds of the paper (the ductility model has a weaker signal, so the defaults flag nothing)
THRESHOLDS = {"yield": dict(z_thres=3.0, majority_frac=0.3), "ductility": dict(z_thres=2.5, majority_frac=0.2)}


def gbr_params(n_features):
    return dict(n_estimators=500, learning_rate=0.01, max_depth=n_features, min_samples_leaf=3, min_samples_split=3,
                loss="squared_error", tol=0.001, warm_start=True)


def _mae(y, p):
    return float(np.mean(np.abs(np.asarray(y) - np.asarray(p))))


def _r2(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    return float(1.0 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2))


def oof_predictions(X, y, groups, n_splits, n_repeats, seed):
    """(n_repeats, n) out-of-fold predictions of repeated k-fold over the unique groups (one model seed)."""
    codes = pd.factorize(pd.Series(groups))[0]
    uniq = np.unique(codes)
    rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=seed)
    params = dict(gbr_params(X.shape[1]), random_state=seed)
    oof = np.full((n_repeats, len(y)), np.nan)
    for j, (gtr, gte) in enumerate(rskf.split(uniq, np.ones(len(uniq)))):
        tr, te = np.flatnonzero(np.isin(codes, uniq[gtr])), np.flatnonzero(np.isin(codes, uniq[gte]))
        oof[j // n_splits, te] = GradientBoostingRegressor(**params).fit(X[tr], y[tr]).predict(X[te])
    return oof


def sequential(df, features, target="y", group_col="group", n_splits=5, n_repeats=50, seed=SEED, q_max=0.25,
               z_thres=3.0, majority_frac=0.5, batch=5, verbose=False):
    """One CVRSA run. Returns (exclusions in order: DataFrame with row = position in df, iteration log, s0)."""
    df = df.reset_index(drop=True)
    rows = np.arange(len(df))
    n0 = len(df)
    max_remove = int(np.floor(q_max * n0 + 1e-9))
    removed, log, s0, it = [], [], None, 0
    while True:
        X, y = df.loc[rows, features].to_numpy(float), df.loc[rows, target].to_numpy(float)
        oof = oof_predictions(X, y, df.loc[rows, group_col].to_numpy(), n_splits, n_repeats, seed)
        resid = y[None, :] - oof
        if s0 is None:
            r = resid.ravel()
            s0 = 1.4826 * np.median(np.abs(r - np.median(r)))
        z = np.abs(resid) / s0
        flag, mean_z = (z > z_thres).mean(axis=0), z.mean(axis=0)
        r2 = np.array([_r2(y, p) for p in oof])
        mae = np.array([_mae(y, p) for p in oof])
        cand = np.flatnonzero(flag >= majority_frac)
        log.append(dict(iteration=it, n_retained=len(rows), n_removed=n0 - len(rows), R2_pooled=r2.mean(),
                        MAE_pooled=mae.mean(), n_eligible=len(cand)))
        if verbose:
            print(f"  it {it}: n={len(rows)} R2={r2.mean():.3f} MAE={mae.mean():.4g} eligible={len(cand)}", flush=True)
        if len(cand) == 0 or n0 - len(rows) >= max_remove:
            break
        order = cand[np.lexsort((-mean_z[cand], -flag[cand]))]
        drop = order[:min(batch, max_remove - (n0 - len(rows)))]
        for i in drop:
            removed.append(dict(rank=len(removed) + 1, iteration=it, row=int(rows[i]), y=y[i],
                                oof_mean=oof[:, i].mean(), flag_frac=flag[i], mean_absz=mean_z[i]))
        rows = np.delete(rows, drop)
        it += 1
    return pd.DataFrame(removed), pd.DataFrame(log), float(s0)


def outer_splits(groups, n_outer, seed):
    rng = np.random.default_rng(seed)
    g = np.asarray(groups).astype(str)
    uniq = rng.permutation(np.unique(g))
    for g_val in np.array_split(uniq, n_outer):
        val = np.isin(g, g_val)
        yield np.flatnonzero(~val), np.flatnonzero(val)


def _nested_one(df, features, target, group_col, split_id, tr, va, q_grid, s, thr):
    d_tr, d_va = df.iloc[tr].reset_index(drop=True), df.iloc[va].reset_index(drop=True)
    seed = s["seed"] + 1000 * split_id
    rem, _, _ = sequential(d_tr, features, target, group_col, s["n_splits"], s["n_repeats_nested"], seed,
                           max(q_grid), thr["z_thres"], thr["majority_frac"], s["batch"])
    order = rem["row"].tolist() if len(rem) else []
    params = dict(gbr_params(len(features)), random_state=seed)
    X_va, y_va = d_va[features].to_numpy(float), d_va[target].to_numpy(float)
    out = []
    for q in q_grid:
        k = min(int(round(q * len(d_tr))), len(order))
        keep = ~np.isin(np.arange(len(d_tr)), order[:k])
        m = GradientBoostingRegressor(**params).fit(d_tr.loc[keep, features].to_numpy(float),
                                                    d_tr.loc[keep, target].to_numpy(float))
        p = m.predict(X_va)
        out.append(dict(split=split_id, q=q, k_removed=k, natural_stop_k=len(order), MAE_val=_mae(y_va, p),
                        R2_val=_r2(y_va, p)))
    return out


def select_q_star(detail, se_factor=0.5):
    detail = detail.copy()
    base = detail[detail.q == 0].set_index("split").MAE_val
    detail["dMAE_val"] = detail.MAE_val - detail.split.map(base)
    curve = detail.groupby("q").agg(MAE_val_mean=("MAE_val", "mean"), dMAE_val_mean=("dMAE_val", "mean"),
                                    dMAE_val_std=("dMAE_val", "std"), n=("dMAE_val", "size"),
                                    k_removed_mean=("k_removed", "mean")).reset_index()
    curve["dMAE_val_SE"] = curve.dMAE_val_std.fillna(0.0) / np.sqrt(curve.n)
    i = int(curve.dMAE_val_mean.idxmin())
    thr = curve.loc[i, "dMAE_val_mean"] + se_factor * curve.loc[i, "dMAE_val_SE"]
    q_star = float(curve.loc[curve.dMAE_val_mean <= thr + 1e-12, "q"].min())
    return q_star, curve


def run(df, prop, features, target="y", group_col="group", outer_key="Composition", n_jobs=-1, verbose=True,
        **overrides):
    """Full CVRSA (nested choice of q*, then the final run). Returns dict(q_star, survivors (bool array over the rows
    of df), exclusions, curve, detail, log, s0).
    The outer splits shuffle the sorted unique values of `outer_key` (the composition names as written in the
    released files, as in the paper); the inner folds use `group_col` in order of appearance."""
    s = dict(SETTINGS, **{k: v for k, v in overrides.items() if k in SETTINGS})
    thr = dict(THRESHOLDS[prop], **{k: v for k, v in overrides.items() if k in ("z_thres", "majority_frac")})
    df = df.reset_index(drop=True)
    t0 = time.time()
    q_grid = np.round(np.arange(0.0, s["q_max"] + 1e-9, s["q_step"]), 4)
    jobs, sid = [], 0
    for o in s["outer_seeds"]:
        for tr, va in outer_splits(df[outer_key], s["n_outer"], s["seed"] + o):
            jobs.append((sid, tr, va))
            sid += 1
    if verbose:
        print(f"CVRSA {prop}: n={len(df)}, {df[group_col].nunique()} compositions; stage 1: {len(jobs)} outer "
              f"splits x {len(q_grid)} q values", flush=True)
    res = Parallel(n_jobs=n_jobs)(delayed(_nested_one)(df, features, target, group_col, i, tr, va, q_grid, s, thr)
                                  for i, tr, va in jobs)
    detail = pd.DataFrame([r for rows in res for r in rows])
    q_star, curve = select_q_star(detail, s["se_factor"])
    if verbose:
        print(f"  q* = {q_star:.2f} ({(time.time() - t0) / 60:.1f} min); stage 2: final run", flush=True)
    exc, log, s0 = sequential(df, features, target, group_col, s["n_splits"], s["n_repeats"], s["seed"], q_star,
                              thr["z_thres"], thr["majority_frac"], s["batch"], verbose=verbose)
    surv = np.ones(len(df), bool)
    if len(exc):
        surv[exc.row.to_numpy()] = False
    if verbose:
        print(f"  excluded {int((~surv).sum())} of {len(df)} ({(time.time() - t0) / 60:.1f} min)", flush=True)
    return dict(q_star=q_star, survivors=surv, exclusions=exc, curve=curve, detail=detail, log=log, s0=s0)
