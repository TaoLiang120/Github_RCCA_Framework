# Data for "Learning high-temperature strength and room-temperature ductility of refractory complex alloys from inconsistent data"

## Files

| File | Content |
|---|---|
| `Yield_dataset.csv` | Curated yield-strength dataset (453 observations of 172 compositions). Every literature report is kept as its own row. |
| `Compression.csv` | Curated room-temperature compressive dataset (145 observations of 122 compositions). |
| `predictions_9element_database.csv` | Predicted yield strength and compressive fracture strain for all 43,425 compositions of Ti, V, Cr, Zr, Nb, Mo, Hf, Ta and W on a 10 at.% grid, at 295, 1475 and 1800 K (130,275 rows). |
| `phase_stability_9element_database.csv` | Phase-stability metrics of the BCC solid solution at 1000 K for the same 43,425 compositions. |
| `Candidates.csv` | The 37 candidate compositions of the paper; the 12 listed in Table S8 are flagged. |

All files use the same composition names (`Composition`, subscripts in at.%), so they can be joined on this column.

## Curated datasets (`Yield_dataset.csv`, `Compression.csv`)

| Column | Meaning |
|---|---|
| `Composition` | Alloy composition as reported |
| `phase` | Reported phase(s) |
| `processing method` | Processing route (for example as-cast, arc melting) |
| `temperature` | Test temperature (K) |
| `yield` | Compressive yield strength (MPa) |
| `fracture strain` | Compressive fracture strain (%) |
| `fracture strain quantifier` | `>` if the sample did not fracture (the value is a lower bound) |
| `doi`, `year` | Source publication |
| `external_test` | 1 = external test alloy (MoNbTaVW, NbTaTiV), withheld from all model development |
| `hold_out` | 1 = hold-out observation, not used for training |
| `Survivor_CVRSA` | True = kept for training after CVRSA; False = excluded by CVRSA (not assumed to be wrong) |

## Property predictions (`predictions_9element_database.csv`)

Each prediction is the mean of the 100 models of the paper (10 × 10 repeated composition-grouped cross-validation). The uncertainty is the s.d. of the 100 predictions scaled to the out-of-fold errors of the measured development data (calibrated s.d.).

| Column | Meaning |
|---|---|
| `Composition` | Composition (subscripts in at.%) |
| `temperature_K` | Temperature (K): 295, 1475 or 1800 |
| `yield_strength_pred_MPa` | Predicted yield strength (MPa) |
| `yield_strength_68pct_halfwidth_MPa` | Half-width of the 68% interval (= 1 calibrated s.d.) |
| `yield_strength_95pct_halfwidth_MPa` | Half-width of the 95% interval |
| `fracture_strain_pred_pct` | Predicted room-temperature compressive fracture strain (%); the same for all three temperatures |
| `fracture_strain_68pct_halfwidth_pct`, `fracture_strain_95pct_halfwidth_pct` | Half-widths of the 68% and 95% intervals |

## Phase-stability metrics (`phase_stability_9element_database.csv`)

Values at 1000 K, the temperature of the phase screens of the paper. Binary and ternary values come from Thermo-Calc (High Entropy Alloy database); ternary compositions for which Thermo-Calc gives no BCC or HCP solution are filled by component expansion from their binary subsystems, and compositions with four to nine elements are obtained by component expansion (Muggianu model) from their ternary subsystems.

| Column | Meaning |
|---|---|
| `Composition` | Composition (subscripts in at.%) |
| `n_elements` | Number of elements |
| `temperature_K` | 1000 |
| `Gmix_bcc_meV_per_atom` | Gibbs free energy of mixing of the BCC solid solution (meV/atom) |
| `Hmix_bcc_meV_per_atom` | Enthalpy of mixing of the BCC solid solution (meV/atom) |
| `dG_bcc_to_hcp_meV_per_atom` | ΔG^b2h = G(HCP) − G(BCC) (meV/atom); positive values mean the BCC solid solution is more stable than the HCP solid solution |

## Candidates (`Candidates.csv`)

The 37 compositions that meet all six criteria of the paper: ΔG^bcc at 1000 K below 1.79 meV/atom; ΔG^b2h at 1000 K above 0; predicted yield strength at 1475 K of at least 735 MPa; predicted room-temperature compressive fracture strain of at least 20%; at most 20 at.% from the nearest composition of each curated dataset; not itself a curated composition. `P_both` is the probability of exceeding both property benchmarks, treating each prediction as normal with the calibrated s.d. and the two as independent. `distance_*` is the fraction of atoms that must be exchanged to reach the nearest curated composition. `in_Table_S8` marks the composition with the highest `P_both` of each of the 12 highest-ranked element combinations.

## Notes

- The fracture strain is measured in compression. It is a preliminary screen for room-temperature ductility, not a tensile ductility.
- The phase metrics compare only the BCC and HCP solid solutions. Laves, B2 and other intermetallic phases are not included, so a composition that passes the phase screens may not form a single BCC phase.
- Predictions far from the curated compositions are less reliable. At 295 K, about 1% of the compositions (very high melting points) have T/Tm below the range of the training data; the model then returns the value at the edge of that range.
