# Project Scripts
# Commented scripts for the VAEP vs GAT thesis

Every statement in every script has a comment saying what it does. The scripts run the same logic as the originals and give the same results. Most are also faster and lighter on memory.

## How to run

1. Install the packages from `requirements.txt` (numpy, pandas, scipy, scikit-learn, xgboost, socceraction, torch, torch-geometric, statsbombpy, matplotlib, tqdm).
2. Keep all files from this folder together. The scripts import `common.py`, `common_spadl.py`, `common_gat.py`, `common_stats.py` and `common_error.py` from the same folder.
3. By default the scripts use your original folder, `C:\Users\Shiva Goud\Desktop\Football data explorer`. To use another folder, set `FOOTBALL_PROJECT_ROOT` first.
   - PowerShell: `$env:FOOTBALL_PROJECT_ROOT = "D:\my\project"`
4. The old check that the interpreter path contains `.vaep-venv` is now optional. Set `REQUIRE_ENV_NAME=.vaep-venv` to turn it back on.
5. Run the scripts in file-name order, for example `python 06_train_vaep_baseline.py`.

## Which file is which step

The numbers printed by each script (its STEP label) are not the same as the file-name prefix. This table links them.

| File | Printed step | What it does |
|---|---|---|
| 00_check_vaep_environment.py | STEP 01 | Checks that the packages and socceraction functions are available |
| 01_create_final_common_split.py | "06" | Freezes the 24 / 5 / 5 train, validation and test split of the 34 prototype matches |
| 02_rebuild_literal_next10_target.py | 07C | Rebuilds the next-10-actions target from SPADL actions |
| 03_independent_target_verification.py | 08A | Rebuilds the target a second way and compares (expected to show 340 extra rows) |
| 04_diagnose_340_extra_observations.py | 08B | Explains the 340 extra rows |
| 05_independent_target_verification_with_eligibility.py | 08C | Applies the eligibility rule and confirms 0 mismatches |
| 06_train_vaep_baseline.py | 09 | Trains the VAEP XGBoost baseline (prototype) |
| 07_train_gat_baseline.py | 10 | Trains the GAT (prototype) |
| 08_delong_test_gat_vs_vaep.py | 11 | DeLong test, GAT vs VAEP (prototype) |
| 09_build_full_scale_target.py | 12 | Builds the full-scale target (426 matches) |
| 10_diagnose_verification_failures_.py | 12B | Diagnoses the goal-verification failures |
| 11_train_vaep_full_scale.py | 13 | VAEP baseline at full scale |
| 12_train_gat_full_scale.py | 14 | Builds the 360 graphs and trains the GAT at full scale |
| 13_gat_prototype_visual_overview.py | - | Figures for the GAT prototype |
| 14_vaep_prototype_visual_overview.py | - | Figures for the VAEP prototype |
| 15_delong_prototype_visual_overview.py | - | Figures for the prototype DeLong test |
| 16_parity_xgboost_vaep_plus_360.py | 27 | Parity run: XGBoost on VAEP features plus a 360 summary |
| 17_parity_gat_plus_vaep_features.py | 28 | Parity run: GAT that also receives VAEP's features |
| 18_parity_decomposition.py | 29 | Splits the VAEP-GAT gap into input and model effects |
| 19_check_which_scripts_are_load_bearing.py | - | Read-only check before archiving prototype scripts |
| 21_A_gat_feature_ablation.py | 20 | GAT node-feature ablation |
| 21_B_gat_feature_count_ablation.py | 22 | Feature-count ladder, 8 down to 1 |
| 21_C_gat_k_ablation.py | 19 | k-nearest-neighbours ablation |
| 21_D__gat_k8_standalone.py | 19B | k=8 only, safe to resume |
| 21_E_gat_features_at_horizon5.py | 23 | Feature-count ladder at label horizon N=5 |
| 22_label_horizon_ablation.py | 21 | Label horizon ablation, N = 5 to 13 and 15 |
| 23_delong_test_full_scale.py | 18 | DeLong test at full scale (N=10, 7-feature GAT) |
| 23_A_delong_best_features_both_horizons.py | 24 | DeLong test for the best feature set at N=10 and N=5 |
| 24_error_analysis_vaep_vs_gat.py | 25 | Error analysis at N=10 |
| 25_error_analysis_vaep_vs_gat_n5.py | 26 | Error analysis at N=5 |
| Plot__label_horizon.py | chart for 21 | VAEP AUC by label horizon |
| Plot_feature_count_n10.py | chart for 22 | GAT feature-count ladder at N=10 |

## Shared modules

| File | Holds |
|---|---|
| common.py | Paths, header printing, file checks, error wrapper |
| common_spadl.py | StatsBomb to SPADL, the window labels, VAEP features |
| common_gat.py | GAT model, graph helpers, training loop, evaluation |
| common_stats.py | The paired DeLong test |
| common_error.py | The error analysis shared by scripts 24 and 25 |

## What changed and what did not

- Same: every seed, every model setting, the order of operations that fixes the random numbers, and the files written (names, columns, rows).
- Faster: window labels use cumulative sums instead of nested loops, 360 frames are looked up by event id instead of re-scanned for each action, edge lists are built with numpy, metadata search stops as soon as every match is found, and evaluations are computed once and reused.
- Lighter: matches are joined to the target one at a time, big tables are freed as soon as they are used, and feature matrices are written in pieces.
- Not faster: the GAT training loops. Making them faster would change the order in which random numbers are drawn, and that would change the results. They were left alone on purpose.

## Things to know

- Script 03 compares against the saved target without the eligibility rule, so its row-count check is expected to fail with 340 extra rows. Scripts 04 and 05 explain and resolve that.
- Script 02 uses socceraction's default StatsBomb getter, as the original did. That default needs `statsbombpy` and network access. Set `STATSBOMB_GETTER=local` to read your local files instead.
- Some header texts mention an old VAEP AUC (0.7731 or 0.7746). They are kept as written in the originals.
- Script 06 no longer passes `use_label_encoder=False` to XGBoost. That option was removed in xgboost 2.0, and it never changed the model.
- Script 03 imports `pandera` to print its version, as the original did. It is installed together with socceraction.
- `tqdm` is in requirements.txt but none of these scripts use it.

## How the results were checked

Each rewritten script was run next to its original on the same input data, and the output files were compared byte for byte (CSV files, PNG figures, saved model weights and the printed log). All of them matched. The only difference in the printed log is the line `[PASS] Python environment`, which now appears only if you set `REQUIRE_ENV_NAME`.

The checks ran on a subset of your data (8 to 12 matches, plus synthetic predictions for the DeLong scripts), because the full 426-match data was not available here. Please run script 18 or 23 once on your machine and confirm that the headline numbers match your thesis tables.

Measured time on that subset (original, then rewrite):

| Script | Original | Rewrite |
|---|---|---|
| 02 rebuild target | 261 s | 12 s |
| 12 GAT full scale | 6 min 01 s | 4 min 15 s |
| 13 GAT overview figures | 108 s | 22 s |
| 18 parity decomposition | 142 s | 46 s |
| 07 GAT prototype | 369 s | 280 s (training dominates) |

The other scripts take about as long as before on this small data. The gains grow with the 426-match data, where the old nested loops and per-action table scans cost the most.
