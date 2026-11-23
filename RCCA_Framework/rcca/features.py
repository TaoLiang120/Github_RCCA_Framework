"""Descriptors of the two property models (Table S3 of the Supplementary Information).

All descriptors are computed from the elemental table rcca/data/elements.csv (Table S2) and the composition:
  <p>_ROM            rule of mixtures, sum_i c_i p_i                                         Eq. (S10)
  <p>_DISTORT        distortion, sum_i (9/8) c_i sqrt(sum_j c_j ((p_i - p_j)/(p_i + p_j))^2)  Eq. (S12)
  <p>_DISTORT_Max    largest term i of the distortion sum
  temp_ratio         T / Tm_ROM (capped at 1)
  sigma_y0, delta_Eb0  zero-temperature flow stress and energy barrier of the Maresca-Curtin model (Eqs. S13-S14)
  ShearBurger_Surf   Rice-Thomson parameter 0.1 G b / gamma_s (G and gamma_s by the rule of mixtures)
The alloy atomic volume V = 4/3 pi r^3 uses the rule-of-mixtures radius of the elements (radius_TC, from the CALPHAD
lattice parameters). volume="tc_binary" instead averages the CALPHAD radii of the binary subsystems; it was used for
the alloys of the compression dataset that contain elements other than Ti, V, Cr, Zr, Nb, Mo, Hf, Ta and W (for the
nine elements both choices are identical), see rcca.datasets.
"""
from functools import lru_cache
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import interpolate

from .composition import ordered_elements, parse

DATA = Path(__file__).parent / "data"
GPA_TO_EV_PER_A3 = 0.00624150648
YIELD_FEATURES = ["temp_ratio", "delta_Eb0", "sigma_y0", "E_negativity", "volume_DISTORT", "Exp_Shear_DISTORT",
                  "Exp_Youngs_ROM"]
DUCTILITY_FEATURES = ["ShearBurger_Surf", "Exp_Shear_DISTORT", "volume_DISTORT", "volume_DISTORT_Max"]
ROM_KEYS = ["Tm", "E_negativity", "Exp_Shear", "Exp_Youngs", "Exp_Poisson", "Exp_Surf", "radius_TC"]


@lru_cache(maxsize=None)
def elements_table():
    return pd.read_csv(DATA / "elements.csv", index_col="element", float_precision="round_trip")


@lru_cache(maxsize=None)
def _binary_radius_models():
    t = pd.read_csv(DATA / "radius_TC_binary_300K.csv", float_precision="round_trip")
    return {s: interpolate.interp1d(g.x.to_numpy(), g.value.to_numpy()) for s, g in t.groupby("system")}


def _tc_binary_radius(frac, E):
    """Concentration-weighted average of the CALPHAD radii of the binary subsystems (rule of mixtures of the pair if
    no CALPHAD binary is available)."""
    els = ordered_elements(frac, E)
    n = len(els)
    if n == 1:
        return E.at[els[0], "radius_TC"]
    models = _binary_radius_models()
    r = 0.0
    for a, b in combinations(els, 2):
        ca, cb = frac[a], frac[b]
        s = ca + cb
        if a + b in models:           # table x = fraction of the second element of the system name
            v = float(models[a + b](cb / s))
        elif b + a in models:
            v = float(models[b + a](ca / s))
        else:
            v = (ca * E.at[a, "radius_TC"] + cb * E.at[b, "radius_TC"]) / s
        r += v * s / (n - 1)
    return r


def _distortion(frac, values):
    tot = mx = 0.0
    for i, ci in frac.items():
        vi = values[i]
        t = np.sqrt(sum(cj * ((vi - values[j]) / (vi + values[j])) ** 2 for j, cj in frac.items())) * ci * 9.0 / 8.0
        tot += t
        mx = max(mx, t)
    return tot, mx


def composition_features(formula, volume="rom"):
    """Temperature-independent descriptors of one composition (dict)."""
    E = elements_table()
    frac = parse(formula)
    missing = set(frac) - set(E.index)
    if missing:
        raise ValueError(f"{formula}: no elemental data for {sorted(missing)}")
    out = {k + "_ROM": sum(c * E.at[e, k] for e, c in frac.items()) for k in ROM_KEYS}
    out["E_negativity"] = out.pop("E_negativity_ROM")
    if volume == "rom":
        r = out["radius_TC_ROM"]
    elif volume == "tc_binary":
        r = _tc_binary_radius(frac, E)
    else:
        raise ValueError("volume must be 'rom' or 'tc_binary'")
    V = 4.0 * np.pi * r ** 3 / 3.0
    out["volume"] = V
    out["volume_DISTORT"], out["volume_DISTORT_Max"] = _distortion(frac, E["volume"])
    out["Exp_Shear_DISTORT"], out["Exp_Shear_DISTORT_Max"] = _distortion(frac, E["Exp_Shear"])

    # Maresca-Curtin parameters (A_sigma = 0.04, alpha = 1/12, A_E = 2; volume misfit with the elemental volumes)
    G, nu = out["Exp_Shear_ROM"], out["Exp_Poisson_ROM"]
    b = np.sqrt(3.0) / 2.0 * (2.0 * V) ** (1.0 / 3.0)
    misfit = sum(c * (E.at[e, "volume"] - V) ** 2 for e, c in frac.items()) / b ** 6
    p = (1.0 + nu) / (1.0 - nu)
    a = 1.0 / 12.0
    out["sigma_y0"] = 3.067 * 0.04 * a ** (-1.0 / 3.0) * G * p ** (4.0 / 3.0) * 1000.0 * misfit ** (2.0 / 3.0)
    out["delta_Eb0"] = 2.0 * a ** (1.0 / 3.0) * G * b ** 3 * p ** (2.0 / 3.0) * GPA_TO_EV_PER_A3 * misfit ** (1.0 / 3.0)
    out["ShearBurger_Surf"] = 0.1 * G * b / out["Exp_Surf_ROM"]
    return out


def add_features(df, temperature_col="temperature", volume="rom"):
    """Copy of df with all descriptors added (one evaluation per distinct composition).
    volume: "rom", "tc_binary", or a column name of df holding "rom"/"tc_binary" per row."""
    df = df.copy()
    if volume in ("rom", "tc_binary"):
        modes = pd.Series(volume, index=df.index)
    else:
        modes = df[volume]
    cache = {}
    rows = []
    for f, m in zip(df["Composition"], modes):
        if (f, m) not in cache:
            cache[(f, m)] = composition_features(f, m)
        rows.append(cache[(f, m)])
    feats = pd.DataFrame(rows, index=df.index)
    for c in feats.columns:
        df[c] = feats[c]
    if temperature_col in df:
        df["temp_ratio"] = np.minimum(df[temperature_col] / df["Tm_ROM"], 1.0)
    return df
