# RCCA_Framework

Code for the paper *Learning high-temperature strength and room-temperature compressive ductility of refractory complex alloys from inconsistent data*.

The package rebuilds everything in `../Datasets` from the curated datasets:

- the two property models (yield strength and room-temperature compressive fracture strain), with calibrated uncertainty;
- the predictions and phase-stability metrics for the 43,425 compositions of the nine-element design space;
- the 35 candidate compositions.

It can also predict new compositions. It needs only NumPy, pandas, SciPy, scikit-learn and joblib.

## Quick start

```bash
pip install -r requirements.txt
python -m pytest tests -q                    # about 15 s
python scripts/reproduce_paper.py            # about 3 min; results in results/
python scripts/predict.py examples/compositions.csv my_predictions.csv --temperatures 295 1475
```

`reproduce_paper.py --cvrsa` also reruns CVRSA (about 10 min on 10 cores) instead of using the released `Survivor_CVRSA` flags.

## What each module does

| Module | Content | Paper |
|---|---|---|
| `rcca/composition.py` | Formula parsing (`Zr333Ti333(NbV)166`, `HfNbTaTiZr`), composition keys, fraction matrices | |
| `rcca/features.py` | Descriptors from the elemental table `data/elements.csv`: rule of mixtures, distortion, T/Tm, Maresca–Curtin σy0 and ΔEb0, Rice–Thomson parameter | Tables S2, S3 |
| `rcca/augmentation.py` | Maresca–Curtin extrapolation of measured yield strengths to 1100–2000 K | SI, Eq. (S17) |
| `rcca/cvrsa.py` | Cross-validated residual stability analysis: nested choice of the excluded fraction q*, then sequential exclusion | SI, CVRSA |
| `rcca/models.py` | 100 gradient-boosting models from 10 × 10 composition-grouped cross-validation; ensemble mean, s.d. and calibration | Methods |
| `rcca/phase.py` | Component expansion (Muggianu) of the CALPHAD binary and ternary values to 4–9 components at 300 and 1000 K | Thermodynamic properties |
| `rcca/screener.py` | Design-space grid, distance to the curated data, the six screening criteria and the probability P | Composition Screener |
| `rcca/datasets.py` | Reads `../Datasets/*.csv` and returns the development, training, hold-out and external-test rows | |

## Reproduction check

These are the results of `reproduce_paper.py`, compared with the paper and the released files:

| Quantity | This code | Paper |
|---|---|---|
| CVRSA (`--cvrsa`) | yield strength q* = 0.02, compression q* = 0.06; the same 7 + 7 excluded observations | the same |
| Yield strength, hold-out (43 obs.), ensemble mean | R² 0.896, MAE 107.9 MPa | R² 0.90, MAE 108 MPa |
| Yield strength, calibration factors (68% / 95%) | 3.91 / 10.4 | 3.9 / 10.1 |
| Fracture strain, hold-out (14 obs.), ensemble mean | R² 0.768, MAE 4.48% | R² 0.77, MAE 4.5% |
| Fracture strain, calibration factors | 4.57 / 8.88 | 4.6 / 8.9 |
| External test alloys within ± 1 calibrated s.d. | 18 of 20 | 18 of 20 |
| Phase metrics, 43,425 compositions | identical (≤ 0.01 meV/atom) | |
| Predictions vs `predictions_9element_database.csv` | yield strength: median difference 0.3 MPa; fracture strain: identical (≤ 0.005%) | |
| Candidates | the same 35; the same 12 in Table S8 | 35 |

The fracture-strain model is reproduced exactly. For the yield strength, the paper computed the descriptors from the exact reported compositions, while the released composition names are rounded to 0.1 at.%. This changes the descriptors by about 1e-7 (relative), which moves a few tree splits.

## Notes

- **Alloy volume.** The alloy atomic volume uses the rule-of-mixtures CALPHAD radius of the elements. For the compression-dataset alloys that contain elements other than the nine design elements (for example Al and Sc), the paper averaged the CALPHAD radii of the binary subsystems (`data/radius_TC_binary_300K.csv`). `datasets.py` does the same, and for the nine elements both give the same value.
- **Phase tables.** `data/phase_*` hold the Thermo-Calc (TCHEA) binary and ternary values used for component expansion. Ternaries without a BCC or HCP solution in Thermo-Calc were already filled from their binaries. Read these files with `float_precision="round_trip"`, as the package does: the triangulation of the regular ternary grid is sensitive to the last digit.
- **Version sensitivity.** The models are trained with fixed seeds (`random_state = 20261109 + 1000 r + f` for repeat r and fold f). Other scikit-learn versions may give slightly different models.
- **Calibration.** The calibrated intervals describe the errors for alloys similar to the curated ones. Treat predictions for compositions far from the data (distance above about 0.2, given by `predict.py`) as extrapolations.
- **Limitations.**
  - The fracture strain is measured in compression; it is a preliminary screen, not a tensile ductility.
  - The phase metrics compare only the BCC and HCP solid solutions; Laves and other intermetallic phases are not included.
