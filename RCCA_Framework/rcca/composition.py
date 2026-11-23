"""Composition strings.

Formulas follow the usual chemical convention: "Ta20Nb20V20Mo20W20", "HfNbTaTiZr" or "Zr333Ti333(NbV)166".
A number after a group in parentheses multiplies every element of the group, so (NbV)166 = Nb166 V166.
"""
import re

import numpy as np

_TOKEN = re.compile(r"([A-Z][a-z]?)|(\()|(\))|(\d+\.?\d*|\.\d+)")


def parse(formula):
    """Formula -> {element: atomic fraction}, in the order of first appearance."""
    tokens = [m for m in _TOKEN.finditer(formula.replace(" ", ""))]
    if "".join(m.group(0) for m in tokens) != formula.replace(" ", ""):
        raise ValueError(f"cannot parse formula {formula!r}")
    pos = 0

    def number():
        nonlocal pos
        if pos < len(tokens) and tokens[pos].group(4):
            pos += 1
            return float(tokens[pos - 1].group(4))
        return 1.0

    def group():
        nonlocal pos
        out = {}
        while pos < len(tokens):
            el, opn, cls = tokens[pos].group(1), tokens[pos].group(2), tokens[pos].group(3)
            if el:
                pos += 1
                n = number()
                out[el] = out.get(el, 0.0) + n
            elif opn:
                pos += 1
                inner = group()
                if pos >= len(tokens) or not tokens[pos].group(3):
                    raise ValueError(f"unbalanced parentheses in {formula!r}")
                pos += 1
                n = number()
                for k, v in inner.items():
                    out[k] = out.get(k, 0.0) + v * n
            elif cls:
                return out
            else:
                raise ValueError(f"unexpected number in {formula!r}")
        return out

    amounts = group()
    if pos != len(tokens):
        raise ValueError(f"unbalanced parentheses in {formula!r}")
    total = sum(amounts.values())
    if total <= 0:
        raise ValueError(f"empty formula {formula!r}")
    return {k: v / total for k, v in amounts.items()}


def composition_key(formula, decimals=6):
    """Key that is equal for formulas of the same composition (used to keep all reports of an alloy in one fold)."""
    frac = parse(formula)
    return "".join(f"{el}{round(frac[el], decimals):.{decimals}f}" for el in sorted(frac))


def ordered_elements(frac, elements_table):
    """Elements sorted by electronegativity (then symbol), the order of the phase tables."""
    return sorted(frac, key=lambda e: (elements_table.at[e, "E_negativity"], e))


def fraction_matrix(formulas, elements):
    """Rows = formulas, columns = atomic fractions of `elements`."""
    rows = [parse(f) for f in formulas]
    unknown = {e for r in rows for e in r} - set(elements)
    if unknown:
        raise ValueError(f"elements {sorted(unknown)} not in the column list")
    return np.array([[r.get(e, 0.0) for e in elements] for r in rows])
