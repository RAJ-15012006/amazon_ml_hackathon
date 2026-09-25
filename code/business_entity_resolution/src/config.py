"""
Configuration module for Business Entity Resolution Challenge.
"""
import os
from pathlib import Path

# Paths
SRC_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SRC_DIR.parent
CODE_DIR = PROJECT_ROOT.parent
WORKSPACE_ROOT = CODE_DIR.parent

# Detect dataset directory dynamically
potential_dataset_roots = [
    WORKSPACE_ROOT / "student_resource" / "dataset",
    WORKSPACE_ROOT / "dataset",
    PROJECT_ROOT / "dataset",
    Path.cwd() / "student_resource" / "dataset",
    Path.cwd() / "dataset"
]

DATASET_ROOT = None
for p in potential_dataset_roots:
    if p.exists() and (p / "train").exists():
        DATASET_ROOT = p
        break

if DATASET_ROOT is None:
    DATASET_ROOT = WORKSPACE_ROOT / "student_resource" / "dataset"

TRAIN_DATA_DIR = DATASET_ROOT / "train"
TEST_DATA_DIR = DATASET_ROOT / "test"
OUTPUT_DIR = WORKSPACE_ROOT / "output"
PLOTS_DIR = WORKSPACE_ROOT / "plots"

# File Paths
TRAIN_S1 = TRAIN_DATA_DIR / "train_source1.tsv"
TRAIN_S2 = TRAIN_DATA_DIR / "train_source2.tsv"
TRAIN_S3 = TRAIN_DATA_DIR / "train_source3.tsv"
TRAIN_GT = TRAIN_DATA_DIR / "train_ground_truth.tsv"

TEST_S1 = TEST_DATA_DIR / "test_source1.tsv"
TEST_S2 = TEST_DATA_DIR / "test_source2.tsv"
TEST_S3 = TEST_DATA_DIR / "test_source3.tsv"

OUTPUT_MATCHING = OUTPUT_DIR / "matching_results.tsv"
OUTPUT_CANDIDATES = OUTPUT_DIR / "candidate_pairs.tsv"

# Normalization & Blocking Parameters
LEGAL_SUFFIXES = {
    "inc", "incorporated", "corp", "corporation", "ltd", "limited", "pvt", "private",
    "llc", "llp", "co", "company", "sarl", "sas", "sasu", "eurl", "sci", "sa", "gmbh",
    "associates", "enterprise", "enterprises", "services", "industries"
}

COMMON_ADDR_WORDS = {
    "street", "st", "saint", "road", "rd", "avenue", "ave", "lane", "ln", "drive", "dr",
    "boulevard", "blvd", "rue", "chemin", "allée", "impasse", "near", "opp", "opposite",
    "behind", "floor", "fl", "shop", "no", "block", "sector", "suite", "ste", "apt",
    "apartment", "pmb", "po", "box", "north", "south", "east", "west", "new", "city",
    "center", "centre", "india", "delhi", "mumbai", "york", "paris", "california", "texas"
}

MAX_DOC_FREQ = 2500
TOP_K_TOKENS = 5
OPTIMAL_THRESHOLD = 0.80

# LightGBM Hyperparameters
LGBM_PARAMS = {
    "objective": "binary",
    "metric": "binary_logloss",
    "boosting_type": "gbdt",
    "learning_rate": 0.08,
    "num_leaves": 31,
    "max_depth": 7,
    "feature_fraction": 0.85,
    "bagging_fraction": 0.85,
    "bagging_freq": 1,
    "min_child_samples": 20,
    "verbose": -1,
    "n_jobs": -1,
    "random_state": 42
}
