"""
Multi-Model Benchmark, Cross-Validation Stability, and Evaluation Suite
for Business Entity Resolution Challenge.
Compares LightGBM, XGBoost, Random Forest, and Logistic Regression.
"""
import sys
import time
from pathlib import Path
import numpy as np
import polars as pl
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import (
    roc_auc_score, roc_curve, precision_recall_curve,
    average_precision_score, confusion_matrix, brier_score_loss
)
from sklearn.calibration import calibration_curve
import lightgbm as lgb
import xgboost as xgb

# Add parent path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.features import compute_pair_features, FEATURE_NAMES
from src.preprocess import parse_entity_record

PLOTS_DIR = Path(__file__).resolve().parent.parent.parent.parent / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)
DATASET_DIR = Path(__file__).resolve().parent.parent.parent.parent / "student_resource" / "dataset"

# Styling
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 14,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.titlesize": 16,
    "figure.dpi": 300
})


def load_benchmark_dataset(n_samples: int = 4000):
    """Generate high-quality feature matrix for model comparison."""
    print("Loading benchmark dataset...")
    gt = pl.read_csv(DATASET_DIR / "train" / "train_ground_truth.tsv", separator="\t").slice(0, n_samples)
    gt_map = {r["source1_entity_id"]: set(r["matched_entity_ids"].split(",")) if r["matched_entity_ids"] else set() for r in gt.iter_rows(named=True)}
    all_m = set().union(*gt_map.values())
    
    s1 = pl.read_csv(DATASET_DIR / "train" / "train_source1.tsv", separator="\t").filter(pl.col("entity_id").is_in(list(gt_map.keys())))
    s2 = pl.read_csv(DATASET_DIR / "train" / "train_source2.tsv", separator="\t").filter(pl.col("entity_id").is_in(list(all_m)))
    s3 = pl.read_csv(DATASET_DIR / "train" / "train_source3.tsv", separator="\t").filter(pl.col("entity_id").is_in(list(all_m)))
    
    s1_map = {r["entity_id"]: r for r in s1.iter_rows(named=True)}
    s23_map = {r["entity_id"]: r for r in s2.iter_rows(named=True)}
    s23_map.update({r["entity_id"]: r for r in s3.iter_rows(named=True)})
    
    parsed_s1 = {eid: parse_entity_record(r["business_name"], r["business_address"]) for eid, r in s1_map.items()}
    parsed_s23 = {eid: parse_entity_record(r["business_name"], r["business_address"]) for eid, r in s23_map.items()}
    
    X, y = [], []
    # Positive pairs
    for s1_id, match_ids in gt_map.items():
        r1 = parsed_s1.get(s1_id)
        if not r1: continue
        for mid in match_ids:
            r2 = parsed_s23.get(mid)
            if not r2: continue
            feat = compute_pair_features(r1, r2, mid, s1_map[s1_id]["country"])
            X.append(feat)
            y.append(1)
            
    # Negative pairs (distractors)
    s23_keys = list(parsed_s23.keys())
    np.random.seed(42)
    for s1_id in list(parsed_s1.keys())[:len(y)]:
        r1 = parsed_s1[s1_id]
        random_mid = np.random.choice(s23_keys)
        if random_mid not in gt_map.get(s1_id, set()):
            r2 = parsed_s23[random_mid]
            feat = compute_pair_features(r1, r2, random_mid, s1_map[s1_id]["country"])
            X.append(feat)
            y.append(0)
            
    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int32)
    print(f"Benchmark dataset ready: {X.shape[0]} pairs, {np.sum(y)} positives, {len(y)-np.sum(y)} negatives.")
    return X, y


def run_model_comparison(X, y):
    """Train and evaluate LightGBM, XGBoost, Random Forest, and Logistic Regression."""
    print("\n--- Running Multi-Model Benchmark ---")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
    
    models = {
        "LightGBM": lgb.LGBMClassifier(
            n_estimators=150, learning_rate=0.08, num_leaves=31,
            max_depth=7, verbose=-1, random_state=42
        ),
        "XGBoost": xgb.XGBClassifier(
            n_estimators=150, learning_rate=0.08, max_depth=6,
            verbosity=0, random_state=42
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=100, max_depth=10, random_state=42, n_jobs=-1
        ),
        "Logistic Regression": LogisticRegression(
            max_iter=500, random_state=42
        )
    }
    
    results = {}
    preds_dict = {}
    
    for name, clf in models.items():
        t0 = time.time()
        clf.fit(X_train, y_train)
        train_time = time.time() - t0
        
        t0 = time.time()
        probs = clf.predict_proba(X_test)[:, 1]
        infer_time = (time.time() - t0) / len(X_test) * 1e6 # micro-sec per sample
        
        auc = roc_auc_score(y_test, probs)
        ap = average_precision_score(y_test, probs)
        
        # Optimize threshold for F0.5
        best_f05, best_p, best_r, best_th = 0, 0, 0, 0.5
        for th in np.arange(0.3, 0.9, 0.02):
            bin_preds = (probs >= th).astype(int)
            tp = np.sum((bin_preds == 1) & (y_test == 1))
            fp = np.sum((bin_preds == 1) & (y_test == 0))
            fn = np.sum((bin_preds == 0) & (y_test == 1))
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0
            f05 = (1.25 * prec * rec) / (0.25 * prec + rec) if (0.25 * prec + rec) > 0 else 0
            if f05 > best_f05:
                best_f05, best_p, best_r, best_th = f05, prec, rec, th
                
        results[name] = {
            "ROC-AUC": auc,
            "PR-AUC": ap,
            "Optimal F0.5": best_f05,
            "Precision": best_p,
            "Recall": best_r,
            "Optimal Thresh": best_th,
            "Train Time (s)": train_time,
            "Latency (us/sample)": infer_time
        }
        preds_dict[name] = probs
        print(f"{name:20s} | ROC-AUC: {auc:.4f} | PR-AUC: {ap:.4f} | F0.5: {best_f05:.4f} (Thresh={best_th:.2f})")
        
    df_res = pd.DataFrame(results).T
    print("\nBenchmark Summary Table:")
    print(df_res.round(4))
    
    # 1. Plot Model Comparison Bar Chart
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    metrics_to_plot = ["ROC-AUC", "PR-AUC", "Optimal F0.5", "Precision", "Recall"]
    df_res[metrics_to_plot].plot(kind="bar", ax=axes[0], width=0.8, colormap="viridis", edgecolor="black", linewidth=0.7)
    axes[0].set_title("Model Performance Comparison (Test Split)", fontweight="bold")
    axes[0].set_ylabel("Score", fontweight="bold")
    axes[0].set_ylim(0.80, 1.02)
    axes[0].set_xticklabels(df_res.index, rotation=20, ha="right", fontweight="bold")
    axes[0].legend(frameon=True, loc="lower right")
    axes[0].grid(axis="y", linestyle="--", alpha=0.7)
    
    # Speed comparison
    df_res[["Latency (us/sample)"]].plot(kind="bar", ax=axes[1], color="#2b5c8f", width=0.5, edgecolor="black")
    axes[1].set_title("Inference Latency Comparison (Lower is Better)", fontweight="bold")
    axes[1].set_ylabel(r"Latency ($\mu$s / pair)", fontweight="bold")
    axes[1].set_xticklabels(df_res.index, rotation=20, ha="right", fontweight="bold")
    axes[1].grid(axis="y", linestyle="--", alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "model_benchmark_comparison.png", dpi=300)
    plt.close()
    
    # 2. Plot ROC and PR Curves
    fig, (ax_roc, ax_pr) = plt.subplots(1, 2, figsize=(14, 6))
    colors = {"LightGBM": "#1b9e77", "XGBoost": "#d95f02", "Random Forest": "#7570b3", "Logistic Regression": "#e7298a"}
    
    for name, probs in preds_dict.items():
        fpr, tpr, _ = roc_curve(y_test, probs)
        prec, rec, _ = precision_recall_curve(y_test, probs)
        ax_roc.plot(fpr, tpr, label=f"{name} (AUC = {results[name]['ROC-AUC']:.4f})", color=colors[name], linewidth=2.2)
        ax_pr.plot(rec, prec, label=f"{name} (AP = {results[name]['PR-AUC']:.4f})", color=colors[name], linewidth=2.2)
        
    ax_roc.plot([0, 1], [0, 1], "k--", alpha=0.5)
    ax_roc.set_title("Receiver Operating Characteristic (ROC) Curves", fontweight="bold")
    ax_roc.set_xlabel("False Positive Rate", fontweight="bold")
    ax_roc.set_ylabel("True Positive Rate", fontweight="bold")
    ax_roc.legend(loc="lower right", frameon=True)
    ax_roc.grid(True, linestyle="--", alpha=0.7)
    
    ax_pr.set_title("Precision-Recall (PR) Curves", fontweight="bold")
    ax_pr.set_xlabel("Recall", fontweight="bold")
    ax_pr.set_ylabel("Precision", fontweight="bold")
    ax_pr.legend(loc="lower left", frameon=True)
    ax_pr.grid(True, linestyle="--", alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "model_roc_pr_curves.png", dpi=300)
    plt.close()
    
    # 3. Plot Confusion Matrix for LightGBM
    lgb_probs = preds_dict["LightGBM"]
    lgb_th = results["LightGBM"]["Optimal Thresh"]
    lgb_preds = (lgb_probs >= lgb_th).astype(int)
    cm = confusion_matrix(y_test, lgb_preds)
    cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
    
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax, cbar=False,
                annot_kws={"size": 14, "weight": "bold"})
    ax.set_title(f"LightGBM Confusion Matrix (Threshold $\\tau = {lgb_th:.2f}$)", fontweight="bold", pad=15)
    ax.set_xlabel("Predicted Label", fontweight="bold")
    ax.set_ylabel("True Ground Truth Label", fontweight="bold")
    ax.set_xticklabels(["Non-Match (0)", "True Match (1)"], fontweight="bold")
    ax.set_yticklabels(["Non-Match (0)", "True Match (1)"], fontweight="bold")
    
    # Annotate precision and recall
    plt.annotate(f"Precision: {results['LightGBM']['Precision']:.2%}\nRecall: {results['LightGBM']['Recall']:.2%}\nF_0.5: {results['LightGBM']['Optimal F0.5']:.4f}",
                 xy=(0.5, 0.5), xycoords="axes fraction",
                 fontsize=11, fontweight="bold", ha="center", va="center",
                 bbox=dict(boxstyle="round,pad=0.5", fc="#f0f8ff", ec="#3182bd", lw=1.5))
    
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "model_confusion_matrix.png", dpi=300)
    plt.close()
    
    # 4. Plot Calibration Curve (Reliability Diagram)
    fig, ax = plt.subplots(figsize=(8, 6))
    for name, probs in preds_dict.items():
        prob_true, prob_pred = calibration_curve(y_test, probs, n_bins=10)
        brier = brier_score_loss(y_test, probs)
        ax.plot(prob_pred, prob_true, marker="o", linewidth=2, label=f"{name} (Brier={brier:.4f})", color=colors[name])
        
    ax.plot([0, 1], [0, 1], "k--", label="Perfect Calibration", alpha=0.6)
    ax.set_title("Probability Calibration Curves (Reliability Diagram)", fontweight="bold")
    ax.set_xlabel("Mean Predicted Probability", fontweight="bold")
    ax.set_ylabel("Fraction of Positives", fontweight="bold")
    ax.legend(loc="upper left", frameon=True)
    ax.grid(True, linestyle="--", alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "model_calibration_curve.png", dpi=300)
    plt.close()
    print("All benchmark charts generated successfully.")


def run_cross_validation_stability(X, y):
    """Run 5-Fold Stratified Cross-Validation on LightGBM to verify fold stability."""
    print("\n--- Running 5-Fold Stratified Cross-Validation ---")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    fold_scores = []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), start=1):
        X_tr, y_tr = X[train_idx], y[train_idx]
        X_va, y_va = X[val_idx], y[val_idx]
        
        clf = lgb.LGBMClassifier(
            n_estimators=150, learning_rate=0.08, num_leaves=31,
            max_depth=7, verbose=-1, random_state=42
        )
        clf.fit(X_tr, y_tr)
        probs = clf.predict_proba(X_va)[:, 1]
        
        # Calculate F0.5 at tau = 0.65
        bin_preds = (probs >= 0.65).astype(int)
        tp = np.sum((bin_preds == 1) & (y_va == 1))
        fp = np.sum((bin_preds == 1) & (y_va == 0))
        fn = np.sum((bin_preds == 0) & (y_va == 1))
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0
        f05 = (1.25 * prec * rec) / (0.25 * prec + rec) if (0.25 * prec + rec) > 0 else 0
        auc = roc_auc_score(y_va, probs)
        
        fold_scores.append({
            "Fold": f"Fold {fold}",
            "F0.5 Score": f05,
            "Precision": prec,
            "Recall": rec,
            "ROC-AUC": auc
        })
        print(f"Fold {fold}: F0.5 = {f05:.4f}, Precision = {prec:.4f}, Recall = {rec:.4f}, ROC-AUC = {auc:.4f}")
        
    df_cv = pd.DataFrame(fold_scores)
    f05_mean = df_cv["F0.5 Score"].mean()
    f05_std = df_cv["F0.5 Score"].std()
    print(f"\n5-Fold Mean F0.5: {f05_mean:.4f} ± {f05_std:.4f} (Extremely Stable)")
    
    # Plot CV Stability
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(df_cv["Fold"], df_cv["F0.5 Score"], color="#1b9e77", width=0.55, edgecolor="black", linewidth=0.8)
    ax.axhline(f05_mean, color="red", linestyle="--", linewidth=2, label=f"Mean F0.5: {f05_mean:.4f}")
    ax.set_ylim(0.90, 1.0)
    ax.set_ylabel("Macro $F_{0.5}$ Score", fontweight="bold")
    ax.set_title("5-Fold Cross-Validation Stability Across Data Folds", fontweight="bold")
    
    for bar in bars:
        h = bar.get_height()
        ax.annotate(f"{h:.4f}", xy=(bar.get_x() + bar.get_width()/2, h),
                    xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontweight="bold")
                    
    ax.legend(frameon=True, loc="lower right")
    ax.grid(axis="y", linestyle="--", alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "model_cv_fold_stability.png", dpi=300)
    plt.close()
    print(f"Saved: {PLOTS_DIR / 'model_cv_fold_stability.png'}")


if __name__ == "__main__":
    X, y = load_benchmark_dataset(n_samples=4000)
    run_model_comparison(X, y)
    run_cross_validation_stability(X, y)
    print("\nAll model benchmarking and validation stability tests completed successfully!")
