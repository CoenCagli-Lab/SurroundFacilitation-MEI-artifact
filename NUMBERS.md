# NUMBERS

Every number printed across the two manuscripts, emitted by
`scripts/emit_numbers.py` into `results/numbers.json` and indexed here. Every
key carries the conventions its value was measured under: which mask (`binary`
or `soft`) and which MEI (`ideal` or `numerical`).

Every number assumes the 93-pixel image spans 2.67 deg, the value stated in their Methods.

Every value here is an output of the code in this repository.

## Geometry

| key | value | conventions |
|---|---|---|
| `geom.energy_inside_mask` | 97.3302 | model A, mask binary, mei ideal |
| `geom.energy_inside_mask_soft` | 96.4257 | model A, mask soft, mei numerical |
| `geom.energy_outside_mask` | 2.66983 | model A, mask binary, mei ideal |
| `geom.env_2sigma_diameter_px` | 28 | model A, mei ideal |
| `geom.image_area_px` | 8649 | — |
| `geom.mask_area_px_closedform` | 557 | model A, mask binary, mei ideal |
| `geom.mask_diameter_px` | 27 | model A, mask binary, mei ideal |
| `geom.mask_radius_sigma.A` | 1.93188 | model A, mask soft, mei numerical |
| `geom.mask_radius_sigma.B` | 1.74266 | model B_d0_2, mask soft, mei numerical |
| `geom.mask_radius_sigma.C` | 1.74078 | model C, mask soft, mei numerical |
| `geom.norm_inside_mask` | 98.6561 | model A, mask binary, mei ideal |
| `geom.norm_outside_mask` | 16.3396 | model A, mask binary, mei ideal |
| `geom.surround_area_px` | 8092 | model A, mask binary, mei ideal |

## MEI fidelity

| key | value | conventions |
|---|---|---|
| `mei.corr_with_filter.A` | 0.998932 | model A, mei numerical |
| `mei.corr_with_filter.B` | 0.991913 | model B_d0_2, mei numerical |
| `mei.corr_with_filter.C` | 0.992567 | model C, mei numerical |

## Model A core result and headline

| key | value | conventions |
|---|---|---|
| `artifact.drive_ratio` | 27.312 | model A, mask soft, mei numerical, measure drive, objective maximizing |
| `artifact.drive_ratio_closedform` | 32.04 | model A, mask binary, mei ideal, measure drive |
| `artifact.letter_headline` | rule min(|facilitation|, |suppression|) > 25; facilitation 26.6355; suppression -26.6355; margin 1.63547; holds True | model A, mask soft, mei numerical |
| `artifact.response_facilitation` | 26.6355 | model A, mask soft, mei numerical, measure response, objective maximizing |
| `artifact.response_facilitation_closedform` | 31.2596 | model A, mask binary, mei ideal, measure response |
| `artifact.response_suppression` | -26.6355 | model A, mask soft, mei numerical, measure response, objective minimizing |

## Dilation (Figure 2)

| key | value | conventions |
|---|---|---|
| `dilation.dilation_px` | array[6] — 0, 4, 9, 14, … | — |
| `dilation.drive_pct.model_a` | array[6] — 27.31, 7.178, 1.136, 0.1128, … | model A, measure drive, objective maximizing |
| `dilation.drive_pct.model_b_d0_0` | array[6] — 14.89, 0.08465, -0.1047, -0.004335, … | model B_d0_0, measure drive, objective maximizing |
| `dilation.drive_pct.model_b_d0_2` | array[6] — 22.4, 1.121, -0.2724, -0.01086, … | model B_d0_2, measure drive, objective maximizing |
| `dilation.drive_pct.model_c` | array[6] — 16.94, 0.9255, -0.09074, 0.02049, … | model C, measure drive, objective maximizing |
| `dilation.expansion_px_at_3sigma` | 9 | model A |
| `dilation.facilitation_at_3sigma.A` | 1.10817 | model A, measure response, objective maximizing |
| `dilation.facilitation_at_3sigma.B` | 1.04282 | model B_d0_2, measure response, objective maximizing |
| `dilation.facilitation_at_3sigma.C` | 30.7094 | model C, measure response, objective maximizing |
| `dilation.response_pct.model_a` | array[6] — 26.64, 7.005, 1.108, 0.1101, … | model A, measure response, objective maximizing |
| `dilation.response_pct.model_b_d0_0` | array[6] — 5.392, -1.507, -2.893, -3.246, … | model B_d0_0, measure response, objective maximizing |
| `dilation.response_pct.model_b_d0_2` | array[6] — 12.74, 1.963, 1.043, 0.302, … | model B_d0_2, measure response, objective maximizing |
| `dilation.response_pct.model_c` | array[6] — 49.05, 37.12, 30.71, 20.03, … | model C, measure response, objective maximizing |
| `dilation.suppression_at_25px.A` | -0.000257043 | model A, measure response, objective minimizing |
| `dilation.suppression_at_25px.B` | -45.0651 | model B_d0_2, measure response, objective minimizing |
| `dilation.suppression_at_25px.C` | -41.5321 | model C, measure response, objective minimizing |
| `dilation.suppression_undilated.A` | -26.6355 | model A, measure response, objective minimizing |
| `dilation.suppression_undilated.B` | -63.7571 | model B_d0_2, measure response, objective minimizing |
| `dilation.suppression_undilated.C` | -58.1491 | model C, measure response, objective minimizing |

## Annulus (Figure 3)

| key | value | conventions |
|---|---|---|
| `annulus.drive_pct_of_full.model_a` | array[10] — 100, 78.44, 54.33, 34.53, … | model A, measure drive, stimulus annulus |
| `annulus.drive_pct_of_full.model_b_d0_0` | array[10] — 100, 78.15, 51.53, 28.51, … | model B_d0_0, measure drive, stimulus annulus |
| `annulus.drive_pct_of_full.model_b_d0_2` | array[10] — 100, 79.46, 58.01, 34.19, … | model B_d0_2, measure drive, stimulus annulus |
| `annulus.drive_pct_of_full.model_c` | array[10] — 100, 78.53, 56.1, 31.16, … | model C, measure drive, stimulus annulus |
| `annulus.radii_px` | array[10] — 0, 3.365, 6.729, 8.972, … | model A |
| `annulus.response_rel_blank.model_a` | array[10] — 51.13, 40.32, 28.23, 18.31, … | model A, measure response, stimulus annulus |
| `annulus.response_rel_blank.model_b_d0_0` | array[10] — 32.46, 25.51, 17.94, 10.65, … | model B_d0_0, measure response, stimulus annulus |
| `annulus.response_rel_blank.model_b_d0_2` | array[10] — 38.14, 30.77, 23.65, 15.02, … | model B_d0_2, measure response, stimulus annulus |
| `annulus.response_rel_blank.model_c` | array[10] — 50.55, 40.19, 29.6, 17.25, … | model C, measure response, stimulus annulus |

---

To rebuild this index from the payloads already in `results/`:
`python3 scripts/emit_numbers.py`, about a minute. To recompute the values
themselves first: `python3 scripts/run_all.py`, a little over an hour.
