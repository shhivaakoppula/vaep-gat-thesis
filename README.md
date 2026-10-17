# Dynamic Positional Value Modelling Using Graph Neural Networks

**A comparison of a Graph Attention Network with the VAEP baseline on StatsBomb 360 data**

> **M.Sc. thesis, Data Science (Digital Health)**<br>
> SRH University, Campus Fürth.<br>
> **Author:** Shiva Goud Koppula<br>
> **Supervisor:** Prof. Dr. Clifford Sell<br>
> **Second examiner:** Prof. Dr. Clemens Werkmeister

---

## Contents

1. [Summary](#1-summary)
2. [Research question and hypotheses](#2-research-question-and-hypotheses)
3. [Background](#3-background)
4. [Data](#4-data)
5. [Method](#5-method)
6. [Experimental design: separating input and model](#6-experimental-design-separating-input-and-model)
7. [Results](#7-results)
8. [Robustness checks and error analysis](#8-robustness-checks-and-error-analysis)
9. [Limitations](#9-limitations)
10. [Repository structure](#10-repository-structure)
11. [Scripts](#11-scripts)
12. [Installation and how to reproduce](#12-installation-and-how-to-reproduce)
13. [Computing environment and run times](#13-computing-environment-and-run-times)
14. [Future work](#14-future-work)
15. [Data licence and citation](#15-data-licence-and-citation)
16. [References](#16-references)
17. [Contact](#17-contact)

---

## 1. Summary

Football produces well over a thousand recorded actions per match, but a match report counts only a few of them. Action-value models such as VAEP (Valuing Actions by Estimating Probabilities) give every action a value by estimating how it changes the probability that the team scores or concedes soon afterwards.

VAEP describes an action with a table of features (action type, result, location, and the previous actions). It does not use the positions of the other players on the pitch. StatsBomb 360 data adds a freeze frame to many actions: the position of every visible player at the moment of the action. This thesis asks whether a model that can use those positions predicts an upcoming goal better than VAEP.

I represent each action as a graph. Each visible player is a node, edges connect each player to the nearest neighbours, and a graph attention network (GAT) reads the graph at the node of the acting player. I compare this GAT with the XGBoost model of VAEP on the same validation actions, using AUC-ROC, a paired DeLong test and a match-level bootstrap.

**Main finding:** under the same evaluation, VAEP (AUC-ROC 0.7731) scores higher than the GAT (0.7178). A four-cell experiment shows that most of the gap comes from the action context that VAEP has and the GAT does not, and from the model type, not from the positional information itself.

## 2. Research question and hypotheses

**RQ1.** Does a graph attention network that can see the position of every visible player predict an upcoming goal better than VAEP?

- H0: the AUC-ROC of VAEP and the GAT on the same validation actions is equal.
- H1: the two AUC-ROC values differ.

The test is two-sided and uses the paired DeLong test, because both models score the same actions.

Market-value modelling and the search for undervalued players were part of the early project plan. They are not part of the final thesis and appear only under future work.

## 3. Background

- **VAEP** (Decroos et al., 2019) values an action by the change in the probability of scoring and the probability of conceding within the next N actions. N = 10 is the main horizon in this thesis, and N = 5 is the second horizon.
- **SPADL** (Soccer Player Action Description Language) is the standard table format for football actions that VAEP uses.
- **GAT** (Veličković et al., 2018) is a graph neural network in which each node learns how much attention to give to each neighbour.
- **StatsBomb 360** provides, for many events, a freeze frame with the position of each visible player, a flag for teammate or opponent, a flag for the goalkeeper and a flag for the acting player.

Tree-based models are often hard to beat on tabular data (Grinsztajn et al., 2022; McElfresh et al., 2023). The closest football study I found reported that a GAT lost to a tree-based baseline on a goal task. This is why the thesis tests the comparison carefully instead of assuming the graph model wins.

## 4. Data

Source: StatsBomb open data with 360 frames (https://github.com/statsbomb/open-data).

| Measure | Value |
|---|---|
| Matches | 426 |
| Events | 1,585,353 |
| Columns per event | 146 |
| Training matches | 292 (six international tournaments) |
| Validation matches | 134 (five club teams plus one AFCON match) |

Training competitions: UEFA Euro 2020 (51 matches), UEFA Euro 2024 (51), UEFA Women's Euro 2022 (31), UEFA Women's Euro 2025 (31), and the FIFA World Cup tournaments listed in Table 4 of the thesis.

Validation teams: Bayer 04 Leverkusen (34 matches), FC Barcelona (35), Paris Saint-Germain (58), Inter Miami (6) and one AFCON match.

The split is by competition, not by random action, so no match appears in both sets. There is no separate test set. The validation set was also used for the robustness checks (see Limitations).

**The data is not included in this repository.** See `data/README.md` for the download steps.

## 5. Method

### 5.1 Label

For every action, the label is 1 if the acting team scores a goal within the next N actions and 0 otherwise. Main horizon N = 10. The second horizon N = 5 and a sweep over ten horizons are reported as robustness checks.

### 5.2 VAEP baseline (XGBoost)

- Input set A: type and result of the current action, start and end location, and the same information for the previous actions, in SPADL format.
- Model: XGBoost (a sum of K regression trees with a logistic output).
- Reproduction check: my reproduction of VAEP reaches an AUC-ROC of 0.7693 in the prototype, in line with the published model.

### 5.3 Graph attention network

- Input set B: for each visible player, x and y position, teammate flag, goalkeeper flag, actor flag, and distance and angle to goal (seven node features in the main model).
- Graph: one graph per action, each visible player is a node, edges connect each player to its k = 5 nearest neighbours.
- Layers: PyTorch Geometric `GATConv` with 4 attention heads. The edge flag (teammate or opponent) enters the attention score.
- Readout: the vector of the actor node.
- Optimiser: Adam. Single seed (42).

### 5.4 Evaluation

- Primary metric: AUC-ROC on the validation actions.
- Secondary metrics: PR-AUC and Brier score.
- Test: paired DeLong test (z and p value) for the difference between two AUC values on the same actions.
- Interval: match-level bootstrap with 2,000 resamples.
- Error analysis: AUC-ROC by decile of the actions of each match, capture rate of goal-leading actions, calibration by decile, and disagreement on goal-leading actions.

## 6. Experimental design: separating input and model

VAEP and the GAT differ in two things at once: the input and the model. A direct comparison cannot say which one causes a gap. I therefore ran a four-cell design that crosses model and input.

| Cell | Model | Input | Meaning |
|---|---|---|---|
| 1 | XGBoost | A | VAEP as published |
| 2 | GAT | B | The GAT of RQ1 |
| 3 | XGBoost | A + B | Trees with action context and a summary of the freeze frame |
| 4 | GAT | A + B | Graph model that also receives the action context (late fusion) |

Cell 4 overfits (late fusion). Its epoch-5 AUC-ROC was 0.7706, and the reported final value is 0.7484.

## 7. Results

Validation AUC-ROC at N = 10, single seed 42:

| Cell | Model | Input | AUC-ROC |
|---|---|---|---|
| 1 | XGBoost | A | 0.7731 |
| 2 | GAT | B | 0.7178 (best configuration: 0.7222) |
| 3 | XGBoost | A + B | 0.7760 |
| 4 | GAT | A + B | 0.7484 |

Decomposition of the total gap (0.0553 between cell 1 and cell 2):

| Effect | Size |
|---|---|
| Model effect | 0.0276 |
| Action-context effect | 0.0306 |
| Positional effect | 0.0029 |
| Total gap | 0.0553 |

Reading: the positional information adds very little to the trees (0.0029). Most of the gap comes from the model type and from the action context that the GAT does not receive in its main form.

The full tables (AUC-ROC, PR-AUC, Brier score, DeLong z and p values, bootstrap intervals) are in `results/` and in Chapter 4 and Appendix D of the thesis.

## 8. Robustness checks and error analysis

- Number of neighbours k = 3, 5, 8 (k = 5 re-run as a self-check).
- Five node-feature groupings.
- A ladder of node-feature counts from eight features down to one, at N = 10 and N = 5.
- Ten label horizons for VAEP, and the comparison of VAEP with the best GAT at both horizons.
- Per-match comparison for matches with at least three positives (117 matches; 10 matches had fewer).
- Capture rate: the share of all goal-leading actions found in each model's highest-scored actions.
- Calibration by decile of each model's own score, and disagreement on goal-leading actions.

## 9. Limitations

- Single random seed (42). I did not estimate the variation between seeds.
- No separate test set. The validation set was used for the robustness checks, so the numbers are validation numbers.
- The GAT in cell 4 overfits. The comparison in cell 4 is therefore a lower bound for a fused graph model.
- The validation set contains mostly club matches and the training set only international matches, so the split also changes the type of football.
- I did not break the error analysis down by action type or pitch zone.
- I did not test the stability or validity of player ratings built from the action values.

## 10. Repository structure

```
vaep-gat-thesis/
├── README.md
├── requirements.txt
├── .gitignore
├── data/
│   └── README.md           how to download the StatsBomb data (no data files)
├── scripts/                all scripts, numbered 00 to 25 as in the VS Code folder
├── results/                small result tables (CSV)
├── figures/                figures used in the thesis (PNG)
└── thesis/
    └── Thesis_Final_Submission.pdf
```

## 11. Scripts

Scripts are numbered in the order I wrote and ran them. Run them in numeric order. File names in the repository are the same as in Appendix E of the thesis.

| Numbers | Purpose |
|---|---|
| 00 to 08 | Prototype on 34 matches: data extraction, matches, events, frames, first VAEP and GAT runs |
| 09, 10 | Full-scale label and its check |
| 11 | Full-scale VAEP (XGBoost) |
| 12 | Full-scale GAT |
| 13 to 15 | Overview figures for the prototype |
| 16 to 18 | Parity experiment: the four cells with the same validation actions |
| 19, 20 | Utility scripts |
| 21 A to E | Ablations: number of neighbours, node-feature groupings, feature-count ladder |
| 22 | Label-horizon sweep |
| 23, 23 A | DeLong tests |
| 24, 25 | Error analysis |
| plot scripts (2) | Final figures for the thesis |

Output files keep the earlier run-log prefixes (06 to 29) in their names, so a file prefix and a script number can differ. Table E2 in Appendix E of the thesis maps each script to its output.

## 12. Installation and how to reproduce

**Requirements:** Python [version], about 16 GB of memory. No GPU is needed.

```bash
git clone https://github.com/shhivaakoppula/vaep-gat-thesis.git
cd vaep-gat-thesis
python -m venv .venv
source .venv/bin/activate        # on Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Main packages: pandas, numpy, scipy, scikit-learn, xgboost, torch, torch-geometric, socceraction, matplotlib.

**Data:**

1. Download https://github.com/statsbomb/open-data (or clone it).
2. Place it in `data/` (see `data/README.md` for the exact folder names the scripts expect).

**Run:**

```bash
python scripts/00_...py
...
python scripts/25_...py
```

Replace the file names with the names in `scripts/`. Steps 11, 12 and 21 are slow (hours). To check the pipeline first, run the prototype scripts 00 to 08 on the 34 prototype matches.

Random seed: 42 in all scripts.

## 13. Computing environment and run times

All runs ran on a personal laptop with 16 GB of memory and no graphics processor.

| Step | Approximate time |
|---|---|
| Building the graphs, training set | 90 minutes |
| Building the graphs, validation set | 42 minutes |
| One full GAT training run | 150 minutes |
| One ablation run | 130 to 650 minutes |

The longest ablation (653.5 minutes) includes one epoch of 28,816 seconds, almost certainly because the laptop went to sleep. It is not real computing time.

## 14. Future work

- More random seeds, and a separate held-out test set.
- A GAT that receives the action context early (early fusion) instead of late fusion.
- Error analysis by action type and pitch zone.
- Player ratings built from the action values, and their stability.
- Market-value modelling and the search for undervalued players (the original RQ2 and RQ3), which this thesis does not address.

## 15. Data licence and citation

The StatsBomb open data is covered by StatsBomb's own licence, which requires attribution. If you use it, credit StatsBomb and follow the terms at https://github.com/statsbomb/open-data.

Code licence: [add your licence here, for example MIT].

To cite this work:

```
Koppula, S. G. (2026). Dynamic Positional Value Modelling Using Graph Neural Networks:
A Comparison of a Graph Attention Network with the VAEP Baseline. M.Sc. thesis,
SRH University, Campus Fürth.
```

## 16. References

- Decroos, T., Bransen, L., Van Haaren, J., & Davis, J. (2019). Actions speak louder than goals: Valuing player actions in soccer. KDD 2019.
- Decroos, T., Dzyuba, V., Van Haaren, J., & Davis, J. (2020). Interpretable prediction of goals in soccer / look-ahead horizon in VAEP. IJCAI-20, pp. 4696–4700.
- Veličković, P., et al. (2018). Graph attention networks. ICLR 2018.
- Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. KDD 2016.
- Fey, M., & Lenssen, J. E. (2019). Fast graph representation learning with PyTorch Geometric.
- DeLong, E. R., DeLong, D. M., & Clarke-Pearson, D. L. (1988). Comparing the areas under two or more correlated receiver operating characteristic curves. Biometrics, 44(3).
- Grinsztajn, L., Oyallon, E., & Varoquaux, G. (2022). Why do tree-based models still outperform deep learning on typical tabular data? NeurIPS 2022.
- McElfresh, D., et al. (2023). When do neural nets outperform boosted trees on tabular data? NeurIPS 2023.
- StatsBomb (2024). StatsBomb open data. https://github.com/statsbomb/open-data

The full reference list (49 entries) is in the thesis.

## 17. Contact

Shiva Goud Koppula, SRH University, Campus Fürth.
GitHub: https://github.com/shhivaakoppula
