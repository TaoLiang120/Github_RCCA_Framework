"""Gradient-boosting models trained by 10 x 10 repeated composition-grouped k-fold cross-validation.

Every repeat splits the compositions (not the rows) into 10 folds, so all reports of an alloy are either in the
training or in the test part. Each of the 100 fits is one model; a prediction is the mean of the 100 models and its
spread is their standard deviation. The spread is calibrated with the out-of-fold errors of the measured development
rows (calibration_factor).
"""
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import RepeatedKFold

from .features import DUCTILITY_FEATURES, YIELD_FEATURES

SEED = 20261109
BASE_PARAMS = dict(n_estimators=500, learning_rate=0.01, min_samples_split=3, loss="squared_error", tol=0.001,
                   warm_start=True)
# settings of the final models of the paper
SETTINGS = {
    "yield": dict(features=YIELD_FEATURES, params=dict(BASE_PARAMS, max_depth=7, min_samples_leaf=6)),
    "ductility": dict(features=DUCTILITY_FEATURES, params=dict(BASE_PARAMS, max_depth=4, min_samples_leaf=3)),
}


def grouped_splits(groups, n_splits=10, n_repeats=10, seed=SEED):
    """(repeat, fold, train_rows, test_rows) of repeated k-fold over the unique groups."""
    codes = pd.factorize(pd.Series(groups))[0]
    uniq = np.unique(codes)
    rkf = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=seed)
    for j, (tr, te) in enumerate(rkf.split(uniq)):
        yield (j // n_splits, j % n_splits, np.flatnonzero(np.isin(codes, uniq[tr])),
               np.flatnonzero(np.isin(codes, uniq[te])))


def _fit(params, X, y, tr, seed):
    m = GradientBoostingRegressor(**dict(params, random_state=seed))
    m.fit(X[tr], y[tr])
    return m


class Ensemble:
    """The 100 models of one property."""

    def __init__(self, models, features, oof=None, calibration=None, info=None):
        self.models, self.features = models, list(features)
        self.oof = oof                    # (n_repeats, n_rows) out-of-fold predictions of the training rows
        self.calibration = calibration or {}
        self.info = info or {}

    def predict(self, df, chunk=50000):
        """Mean and s.d. (over the models) of the predictions for the rows of df."""
        X = df[self.features].to_numpy(float)
        s1, s2 = np.zeros(len(X)), np.zeros(len(X))
        for m in self.models:
            for i in range(0, len(X), chunk):
                p = m.predict(X[i:i + chunk])
                s1[i:i + chunk] += p
                s2[i:i + chunk] += p * p
        mean = s1 / len(self.models)
        return mean, np.sqrt(np.clip(s2 / len(self.models) - mean ** 2, 0, None))

    def predict_calibrated(self, df, level=0.68):
        """Mean and half-width of the calibrated `level` interval (level 0.68: the calibrated s.d.)."""
        mean, sd = self.predict(df)
        return mean, self.calibration[level] * sd

    def save(self, path):
        with open(path, "wb") as fh:
            pickle.dump(self, fh)

    @staticmethod
    def load(path):
        with open(path, "rb") as fh:
            return pickle.load(fh)


def train(df, prop=None, features=None, params=None, target="y", group_col="group", n_splits=10, n_repeats=10,
          seed=SEED, n_jobs=-1, levels=(0.68, 0.95)):
    """Train the 100 models on the rows of df; model (repeat r, fold f) uses random_state seed + 1000 r + f.
    Rows with highT_augmented == 1 are used for training but not for the calibration."""
    if prop is not None:
        features = features or SETTINGS[prop]["features"]
        params = params or SETTINGS[prop]["params"]
    df = df.reset_index(drop=True)
    X, y = df[features].to_numpy(float), df[target].to_numpy(float)
    jobs = list(grouped_splits(df[group_col], n_splits, n_repeats, seed))
    models = Parallel(n_jobs=n_jobs)(delayed(_fit)(params, X, y, tr, seed + 1000 * r + f) for r, f, tr, _ in jobs)
    oof = np.full((n_repeats, len(df)), np.nan)
    for (r, f, tr, te), m in zip(jobs, models):
        oof[r, te] = m.predict(X[te])
    ens = Ensemble(models, features, oof, info=dict(prop=prop, params=params, seed=seed, n_rows=len(df),
                                                     n_groups=int(df[group_col].nunique())))
    measured = (df["highT_augmented"] == 0).to_numpy() if "highT_augmented" in df else np.ones(len(df), bool)
    z = np.abs(y - oof.mean(axis=0)) / oof.std(axis=0, ddof=1)
    ens.calibration = {lv: float(np.quantile(z[measured], lv)) for lv in levels}
    return ens


def scores(y, pred):
    y, pred = np.asarray(y, float), np.asarray(pred, float)
    r = y - pred
    return dict(R2=1.0 - np.sum(r ** 2) / np.sum((y - y.mean()) ** 2), MAE=np.mean(np.abs(r)),
                RMSE=np.sqrt(np.mean(r ** 2)), MAPE=100.0 * np.mean(np.abs(r) / np.abs(y)))


def per_model_scores(ens, df, target="y"):
    """Scores of each of the 100 models on the rows of df (e.g. the hold-out set): DataFrame, one row per model."""
    X = df[ens.features].to_numpy(float)
    return pd.DataFrame([scores(df[target], m.predict(X)) for m in ens.models])
