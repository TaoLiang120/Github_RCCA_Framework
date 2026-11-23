"""Released datasets (../Datasets) -> model-ready tables.

Yield_dataset.csv / Compression.csv hold every literature report as one row, with the flags
  external_test   the two external test alloys (MoNbTaVW, NbTaTiV), never used for model development
  hold_out        hold-out observations (10% of the compositions)
  Survivor_CVRSA  True = kept by CVRSA, False = excluded (development rows only)
Development rows = neither external nor hold-out (the CVRSA input); training rows = development rows kept by CVRSA,
plus, for the yield strength, the Maresca-Curtin augmented rows.
"""
from pathlib import Path

import numpy as np
import pandas as pd

from .augmentation import augment
from .composition import composition_key, parse
from .features import add_features

REPO = Path(__file__).resolve().parents[2]
DATASETS = REPO / "Datasets"
NINE = {"Ti", "V", "Cr", "Zr", "Nb", "Mo", "Hf", "Ta", "W"}
TARGET = {"yield": "yield", "ductility": "fracture strain"}
FILE = {"yield": "Yield_dataset.csv", "ductility": "Compression.csv"}


def _flag(s):
    return s.astype(str).str.lower().isin(["true", "1", "1.0"])


def load(prop, data_dir=DATASETS):
    """All rows of one dataset with descriptors, the column `group` (composition key) and `y` (target)."""
    df = pd.read_csv(Path(data_dir) / FILE[prop])
    df["external_test"] = _flag(df["external_test"])
    df["hold_out"] = _flag(df["hold_out"])
    df["group"] = [composition_key(f) for f in df["Composition"]]
    if prop == "ductility":
        # alloys with elements beyond the nine refractory elements: CALPHAD binary radii (as in the paper)
        df["_volume"] = ["rom" if set(parse(f)) <= NINE else "tc_binary" for f in df["Composition"]]
        df = add_features(df, volume="_volume").drop(columns="_volume")
    else:
        df = add_features(df, volume="rom")
    df["y"] = df[TARGET[prop]].astype(float)
    df["highT_augmented"] = 0
    return df


def development(df):
    """CVRSA input: rows that are neither external test nor hold-out."""
    return df[~df.external_test & ~df.hold_out].reset_index(drop=True)


def holdout(df):
    return df[df.hold_out].reset_index(drop=True)


def external(df):
    return df[df.external_test].reset_index(drop=True)


def training(df, prop, survivors=None):
    """Training rows: development rows kept by CVRSA (survivors: boolean array over development(df); default the
    released Survivor_CVRSA column), plus the MC-augmented rows for the yield strength."""
    dev = development(df)
    keep = _flag(dev["Survivor_CVRSA"]).to_numpy() if survivors is None else np.asarray(survivors, bool)
    tr = dev[keep].reset_index(drop=True)
    if prop == "yield":
        aug = augment(tr, group_col="group", y_col="y")
        tr = pd.concat([tr, aug], ignore_index=True)
    return tr
