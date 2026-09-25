# Business Entity Resolution Pipeline

## Overview
This repository contains the complete, production-grade, self-contained machine learning pipeline for the **Business Entity Resolution Challenge**. Given business records across 3 independent, noisy data sources, it resolves each Source 1 reference entity to all matching entities in Source 2 and Source 3, optimizing the precision-weighted macro $F_{0.5}$ metric.

---

## Directory Structure
```
code/business_entity_resolution/
├── README.md              # Reproduction instructions & architecture documentation
├── requirements.txt       # Pinned dependencies
└── src/
    ├── __init__.py
    ├── config.py          # Paths, domain stopwords, hyper-parameters
    ├── preprocess.py      # Transliteration, URL stripping, slug extraction, address parsing
    ├── blocking.py        # Country-scoped multi-key inverted index candidate generation
    ├── features.py        # 19-dimensional C-accelerated string similarity feature extraction
    ├── model.py           # LightGBM GBDT training, macro F0.5 scoring, greedy conflict resolution
    ├── model_benchmark.py # Multi-model bake-off (LightGBM, XGBoost, Random Forest, Logistic Regression) & 5-fold CV
    ├── comprehensive_eda.py # Statistical outlier analysis (IQR), noise quantification, Seaborn heatmap & violins
    ├── eda_plots.py       # Visualization suite for EDA, thresholds, and feature importances
    └── pipeline.py        # Master pipeline orchestrator (train -> validate -> predict -> export)
```

---

## System Requirements
- **OS**: macOS / Linux / Windows
- **Python**: 3.10+ (Recommended Python 3.11)
- **RAM**: $\ge 16$ GB recommended
- **CPU**: Multi-core processor (8+ threads recommended)

---

## Installation & Setup

1. **Create and activate a virtual environment**:
```bash
python3 -m venv .venv
source .venv/bin/activate
```

2. **Install dependencies**:
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

## End-to-End Reproduction

Run the full end-to-end pipeline (data loading $\rightarrow$ blocking $\rightarrow$ training $\rightarrow$ validation $\rightarrow$ test candidate generation $\rightarrow$ scoring $\rightarrow$ output generation):

```bash
python -m src.pipeline
```

This single command will:
1. Train the LightGBM classifier on the training set cohort with early stopping.
2. Optimize the decision threshold for macro $F_{0.5}$ on held-out validation data.
3. Stream test candidate pairs and model predictions across France, US, and India into:
   - `output/candidate_pairs.tsv`
   - `output/matching_results.tsv`

---

## Output Validation

Run the official challenge validator to verify format compliance:

```bash
python3 ../../student_resource/utils/validate_submission.py \
    --matching ../../output/matching_results.tsv \
    --candidate ../../output/candidate_pairs.tsv \
    --test-dir ../../student_resource/dataset/test
```

Expected output:
```
PASS
```

---

## Exploratory Data Analysis & Visualization Suites

To regenerate all 15 publication-grade visualization figures and statistical summaries:

1. **Statistical Outlier Detection & Seaborn Suite**:
```bash
python -m src.comprehensive_eda
```
Generates text length IQR fences, cross-source noise prevalence metrics, Seaborn feature correlation heatmap, violin class separability distributions, and probability density separation.

2. **Multi-Model Benchmark & 5-Fold Cross-Validation**:
```bash
python -m src.model_benchmark
```
Runs a 4-model comparison (LightGBM, XGBoost, Random Forest, Logistic Regression) evaluating ROC-AUC, PR-AUC, optimal $F_{0.5}$, latency, ROC/PR curves, confusion matrix, and 5-fold Stratified CV stability.

3. **Core Domain & Match Multiplicity Suite**:
```bash
python -m src.eda_plots
```
Generates country distributions, match multiplicity histograms, $F_{0.5}$ decision threshold curve, and LightGBM feature importances.

