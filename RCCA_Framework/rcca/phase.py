"""Phase-stability metrics of the BCC solid solution by component expansion (CE), main text "Thermodynamic properties".

Inputs are CALPHAD (Thermo-Calc, TCHEA database) values of the binary and ternary systems at temperature T,
tabulated in rcca/data/phase_binary_<T>K.csv and phase_ternary_<T>K.csv (T = 300 and 1000 K):
  Gmix_TC          Gibbs free energy of mixing of the BCC solid solution (eV/atom)
  Hmix_TC          enthalpy of mixing of the BCC solid solution (eV/atom)
  bcc2hcp_Gmix_TC  dG_b2h = G(HCP) - G(BCC) (eV/atom)
Binary values are interpolated linearly; ternary values by a Clough-Tocher interpolation (Gmix, Hmix; with the
binary edges) or linearly (dG_b2h). Ternary compositions where Thermo-Calc gives no BCC or HCP solution were
filled beforehand by component expansion from their binaries (already contained in the tables).

For four to nine components (Muggianu model, Luo et al., J. Mater. Sci. Technol. 35 (2019) 2115):
  Gmix, Hmix   excess part of every ternary and binary subsystem, evaluated at the Muggianu projected composition and
               weighted by prod(c_i / x_i); the binary weights are reduced by the part already contained in the
               ternaries. The pure-element part (rule of mixtures of the element values, extrapolated from the binary
               systems; zero for Nb and V, the BCC reference elements) and, for Gmix, the ideal mixing term are added
               for the whole composition.
  dG_b2h       concentration-weighted mean of the ternary subsystems at their renormalized compositions.
A ternary composition takes the ternary value directly.
"""
from functools import lru_cache
from itertools import combinations
from math import comb
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import interpolate

from .composition import ordered_elements, parse
from .features import elements_table

DATA = Path(__file__).parent / "data"
KB = 8.6173324e-05
THERMO_KEYS = ("Gmix_TC", "Hmix_TC")       # extrapolated by Muggianu (excess part)
IDEAL_KEYS = ("Gmix_TC",)                  # contain the ideal mixing entropy
PURE_ZERO = ("Nb", "V")
KEYS = ("Gmix_TC", "Hmix_TC", "bcc2hcp_Gmix_TC")
EXTERNAL_ELEMENTS = ["Hf", "Zr", "Ta", "Ti", "Mn", "Nb", "Al", "V", "Cr", "Fe", "Co", "Re", "Ni", "Mo", "W", "Sc",
                     "Cu", "Zn", "Y", "Sn", "Pd"]   # order in which binary partners are scanned (median is order-free)


class PhaseTables:
    def __init__(self, T):
        self.T = float(T)
        b = pd.read_csv(DATA / f"phase_binary_{int(T)}K.csv", float_precision="round_trip")
        t = pd.read_csv(DATA / f"phase_ternary_{int(T)}K.csv", float_precision="round_trip")
        self.binary = {(s, k): interpolate.interp1d(g.x.to_numpy(), g.value.to_numpy())
                       for (s, k), g in b.groupby(["system", "key"])}
        self.ternary = {}
        for (s, k), g in t.groupby(["system", "key"]):
            pts, val = g[["x2", "x3"]].to_numpy(), g.value.to_numpy()
            f = interpolate.CloughTocher2DInterpolator(pts, val) if k in THERMO_KEYS else \
                interpolate.LinearNDInterpolator(pts, val)
            self.ternary[(s, k)] = f
        self.E = elements_table()
        self._pure = {}

    def system(self, els):
        return "".join(ordered_elements({e: 1 for e in els}, self.E))

    def model(self, els, x, key):
        """Value of the binary/ternary table for the elements els (in table order) at fractions x (same order)."""
        s = "".join(els)
        if len(els) == 2:
            f = self.binary.get((s, key))
            if f is not None:
                return float(f(x[1]))
            f = self.binary.get((els[1] + els[0], key))     # same electronegativity: other order in the table
            return None if f is None else float(f(x[0]))
        f = self.ternary.get((s, key))
        return None if f is None else float(f(np.array([x[1:]]))[0])

    def ideal(self, x):
        x = np.asarray(x, float)
        return self.T * KB * float(np.sum(x * np.log(x + 1e-20)))

    def pure(self, el, key):
        if el in PURE_ZERO:
            return 0.0
        if (el, key) in self._pure:
            return self._pure[(el, key)]
        vals = []
        for other in EXTERNAL_ELEMENTS:
            if other == el or other not in self.E.index:
                continue
            pair = ordered_elements({el: 1, other: 1}, self.E)
            i = pair.index(el)
            outs = []
            for c in (0.9, 0.8, 0.7):
                x = np.zeros(2)
                x[i], x[1 - i] = c, 1.0 - c
                o = self.model(pair, x, key)
                if o is None:
                    break
                if key in IDEAL_KEYS:
                    o -= self.ideal(x)
                outs.append(o)
            if len(outs) == 3:
                vals.append(3.0 * outs[0] - 3.0 * outs[1] + outs[2])
        self._pure[(el, key)] = float(np.median(vals)) if vals else 0.0
        return self._pure[(el, key)]

    def excess(self, out, els, x, key):
        if key in IDEAL_KEYS:
            out -= self.ideal(x)
        return out - sum(c * self.pure(e, key) for e, c in zip(els, x))


def _muggianu(c):
    c = np.asarray(c, float)
    xs = c + (1.0 - c.sum()) / len(c)
    return float(np.prod(c / xs)), xs


def expand(formula, tables, keys=KEYS):
    """Phase-stability metrics (eV/atom) of one composition of three to nine components."""
    frac = parse(formula)
    els = ordered_elements(frac, tables.E)
    c = np.array([frac[e] for e in els])
    n = len(els)
    if n < 3:
        raise ValueError("compositions with at least three components")
    out = {}
    for key in keys:
        if n == 3:
            out[key] = tables.model(els, c, key)
            continue
        terns = list(combinations(range(n), 3))
        if key in THERMO_KEYS:
            v = 0.0
            for t in terns:
                w, xs = _muggianu(c[list(t)])
                sub = [els[i] for i in t]
                v += w * tables.excess(tables.model(sub, xs, key), sub, xs, key)
            for pr in combinations(range(n), 2):
                w, xs = _muggianu(c[list(pr)])
                shared = 0.0
                for t in terns:
                    if pr[0] in t and pr[1] in t:
                        _, xt = _muggianu(c[list(t)])
                        k = [i for i in t if i not in pr][0]
                        shared += c[k] / xt[t.index(k)]
                w *= 1.0 - shared
                if abs(w) < 1e-12:
                    continue
                sub = [els[i] for i in pr]
                v += w * tables.excess(tables.model(sub, xs, key), sub, xs, key)
            v += sum(ci * tables.pure(e, key) for e, ci in zip(els, c))
            if key in IDEAL_KEYS:
                v += tables.ideal(c)
            out[key] = v
        else:
            v = 0.0
            for t in terns:
                ct = c[list(t)]
                v += tables.model([els[i] for i in t], ct / ct.sum(), key) * ct.sum() / comb(n - 1, 2)
            out[key] = v
    return out


def phase_metrics(formulas, T=1000):
    """DataFrame with Gmix_bcc, Hmix_bcc and dG_bcc_to_hcp (meV/atom) for the compositions (three to nine of Ti, V,
    Cr, Zr, Nb, Mo, Hf, Ta, W)."""
    tables = _tables(int(T))
    rows = [expand(f, tables) for f in formulas]
    df = pd.DataFrame(rows)
    return pd.DataFrame({"Composition": list(formulas), "temperature_K": int(T),
                         "Gmix_bcc_meV_per_atom": 1000.0 * df.Gmix_TC, "Hmix_bcc_meV_per_atom": 1000.0 * df.Hmix_TC,
                         "dG_bcc_to_hcp_meV_per_atom": 1000.0 * df.bcc2hcp_Gmix_TC})


@lru_cache(maxsize=None)
def _tables(T):
    return PhaseTables(T)
