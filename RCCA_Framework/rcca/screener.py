"""Composition Screener: the design space and the candidate search of the paper (SI, "Composition Screener").

Criteria (all six must hold):
  (i)   Gmix of the BCC solid solution at 1000 K below 1.79 meV/atom (the MAE of component expansion for the
        ternary compositions at 1000 K)
  (ii)  dG_b2h = G(HCP) - G(BCC) at 1000 K above 0
  (iii) predicted yield strength at 1475 K >= 735 MPa (MoNbTaVW measured at 1473 K)
  (iv)  predicted room-temperature compressive fracture strain >= 20%
  (v)   at most 0.20 from the nearest composition of EACH curated dataset (distance = half the L1 distance of the
        atomic fractions = fraction of atoms to exchange)
  (vi)  not itself a curated composition
Ranking: P = Phi((YS - 735)/s_Y) Phi((eps - 20)/s_D) with the calibrated s.d. s_Y, s_D, i.e. both predictions are
treated as normal and their errors as independent. The best composition of each element set is listed first.
"""
from itertools import combinations_with_replacement

import numpy as np
import pandas as pd
from scipy.stats import norm

from .composition import fraction_matrix, ordered_elements, parse
from .features import elements_table

ELEMENTS = ["Ti", "V", "Cr", "Zr", "Nb", "Mo", "Hf", "Ta", "W"]
CRITERIA = dict(gmix_max=1.79, db2h_min=0.0, ys_min=735.0, duct_min=20.0, distance_max=0.20)


def grid(elements=ELEMENTS, step=10, min_elements=3):
    """All compositions of `elements` on a grid of `step` at.% with at least `min_elements` elements
    (43,425 for the nine elements, 10 at.% and three or more elements). Names list the elements by
    electronegativity with their at.%, e.g. Zr20Ta10Ti10Nb20V20W20."""
    n = 100 // step
    E = elements_table()
    names = []
    for combo in combinations_with_replacement(range(len(elements)), n):
        counts = np.bincount(combo, minlength=len(elements))
        if (counts > 0).sum() < min_elements:
            continue
        frac = {elements[i]: c * step for i, c in enumerate(counts) if c}
        names.append("".join(f"{e}{frac[e]}" for e in ordered_elements(frac, E)))
    return sorted(set(names))


def nearest(formulas, reference):
    """Distance to and name of the nearest reference composition for each formula."""
    ref = pd.Series(reference).drop_duplicates().to_numpy()
    els = sorted({e for f in list(formulas) + list(ref) for e in parse(f)})
    X, R = fraction_matrix(formulas, els), fraction_matrix(ref, els)
    dist, near = np.empty(len(X)), np.empty(len(X), dtype=object)
    for i in range(0, len(X), 4000):
        d = 0.5 * np.abs(X[i:i + 4000, None, :] - R[None]).sum(axis=2)
        j = d.argmin(axis=1)
        dist[i:i + 4000], near[i:i + 4000] = d[np.arange(len(j)), j], ref[j]
    return dist, near


def screen(table, yield_compositions, ductility_compositions, criteria=None, n_top=12):
    """table: one row per composition with Composition, YS (MPa, at the screening temperature), YS_sd (calibrated),
    eps (%), eps_sd, Gmix_bcc and dG_b2h (meV/atom). Returns (candidates sorted by P, best per element set, funnel)."""
    c = dict(CRITERIA, **(criteria or {}))
    t = table.copy()
    t["dist_Y"], t["nearest_Y"] = nearest(t.Composition, yield_compositions)
    t["dist_D"], t["nearest_D"] = nearest(t.Composition, ductility_compositions)
    t["P_both"] = norm.cdf((t.YS - c["ys_min"]) / t.YS_sd) * norm.cdf((t.eps - c["duct_min"]) / t.eps_sd)
    t["elements"] = ["-".join(e for e in ELEMENTS if e in parse(f)) for f in t.Composition]
    steps = [("all compositions", np.ones(len(t), bool)),
             (f"Gmix_bcc(1000 K) < {c['gmix_max']} meV/atom", t.Gmix_bcc < c["gmix_max"]),
             ("dG_b2h(1000 K) > 0", t.dG_b2h > c["db2h_min"]),
             (f"YS >= {c['ys_min']:.0f} MPa", t.YS >= c["ys_min"]),
             (f"fracture strain >= {c['duct_min']:.0f}%", t.eps >= c["duct_min"]),
             (f"within {c['distance_max']} of a yield-strength composition", t.dist_Y <= c["distance_max"]),
             (f"within {c['distance_max']} of a ductility composition", t.dist_D <= c["distance_max"]),
             ("not a curated composition", (t.dist_Y > 1e-6) & (t.dist_D > 1e-6))]
    mask = np.ones(len(t), bool)
    funnel = []
    for name, m in steps:
        mask &= np.asarray(m)
        funnel.append(dict(criterion=name, n_left=int(mask.sum())))
    cand = t[mask].sort_values("P_both", ascending=False)
    top = cand.groupby("elements", sort=False).head(1).head(n_top)
    return cand, top, pd.DataFrame(funnel)
