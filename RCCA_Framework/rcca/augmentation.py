"""Maresca-Curtin (MC) extrapolation of measured yield strengths to higher temperatures (SI, Eq. S17).

For each composition with measurements at two or more temperatures above 800 K (spanning at least 100 K and
softening by at least 0.1 MPa/K), sigma_y0 and Delta_Eb0 of the MC model are fitted to the lowest and highest of
those measurements; the model then gives values at 1100, 1400, 1700 and 2000 K above the highest measured
temperature and below the melting point (grid points above 0.9 Tm and values below 30 MPa are dropped).
Measurements of the same composition within 10 K are averaged first.
"""
import numpy as np
import pandas as pd

KB = 8.6173324e-05          # eV/K
STRAIN_RATE_RATIO = 1.0e7   # epsilon_0 / epsilon
ALPHA = 0.55
GRID = np.linspace(800.0, 2000.0, 5)
T_SCREEN = 800.0
YIELD_MIN = 30.0


def mc_yield(T, sigma_y0, delta_Eb0):
    """MC yield strength (MPa) at temperatures T (K)."""
    T = np.asarray(T, dtype=float)
    L = np.log(STRAIN_RATE_RATIO)
    y_low = sigma_y0 * (1.0 - (KB * L * T / delta_Eb0) ** (2.0 / 3.0))
    y_high = sigma_y0 * np.exp(-KB * L * T / ALPHA / delta_Eb0)
    return np.where(y_low / sigma_y0 < 0.5, y_high, y_low)


def fit_two_points(T1, T2, Y1, Y2):
    """sigma_y0 and Delta_Eb0 of the high-temperature MC branch through (T1, Y1) and (T2, Y2)."""
    L = np.log(STRAIN_RATE_RATIO)
    delta_Eb0 = KB * L / ALPHA / np.log(Y1 / Y2) * (T2 - T1)
    sigma_y0 = Y1 / np.exp(-KB * T1 * L / ALPHA / delta_Eb0)
    return sigma_y0, delta_Eb0


def augment(df, group_col="group", y_col="yield", T_col="temperature", tm_col="Tm_ROM", dT=10.0):
    """Augmented rows for the measured rows of df (one composition per group).
    Each new row copies the last measured row of its composition (all descriptors) and gets the new temperature,
    the MC yield strength and highT_augmented = 1; temp_ratio is recomputed."""
    new = []
    for _, g in df.sort_values([group_col, T_col], kind="mergesort").groupby(group_col, sort=True):
        Ts, Ys, t_last, n = [], [], None, 0
        for t, y in zip(g[T_col], g[y_col]):
            if t_last is not None and abs(t - t_last) < dT:
                Ys[-1] = (n * Ys[-1] + y) / (n + 1)
                n += 1
            else:
                Ts.append(t)
                Ys.append(y)
                t_last, n = t, 1
        Ts, Ys = np.array(Ts), np.array(Ys)
        hi = Ts > T_SCREEN
        TC, YC = Ts[hi], Ys[hi]
        if len(TC) < 2 or TC[-1] - TC[0] < 100.0 or (YC[0] - YC[-1]) / (TC[-1] - TC[0]) < 0.1:
            continue
        last = g.iloc[-1]
        tm = last[tm_col]
        i0 = int(np.searchsorted(GRID, TC[-1], side="right"))
        i1 = max(min(int(round((tm - GRID[0]) / (GRID[1] - GRID[0]))), len(GRID)), i0)
        temps = GRID[i0:i1]
        if len(temps) == 0:
            continue
        s0, e0 = fit_two_points(TC[0], TC[-1], YC[0], YC[-1])
        ys = np.where(temps < 0.9 * tm, mc_yield(temps, s0, e0), 1.0)
        for t, y in zip(temps, ys):
            if y >= YIELD_MIN:
                row = last.copy()
                row[T_col], row[y_col], row["highT_augmented"] = t, y, 1
                new.append(row)
    out = pd.DataFrame(new).reset_index(drop=True)
    if len(out):
        out["temp_ratio"] = np.minimum(out[T_col] / out[tm_col], 1.0)
    return out
