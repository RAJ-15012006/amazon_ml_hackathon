"""
LightGBM training, threshold optimization, and macro F0.5 scoring module.
"""
from typing import Dict, List, Set, Tuple
import numpy as np
import lightgbm as lgb
try:
    from .config import LGBM_PARAMS, OPTIMAL_THRESHOLD
except (ImportError, ValueError):
    from src.config import LGBM_PARAMS, OPTIMAL_THRESHOLD


def compute_entity_f_beta(pred_set: Set[str], true_set: Set[str], beta: float = 0.5) -> float:
    """
    Compute entity-level F_beta score:
    - Both empty (singleton correctly identified): 1.0
    - One empty and one non-empty: 0.0
    - Otherwise standard F_beta formula.
    """
    if not pred_set and not true_set:
        return 1.0
    if not pred_set or not true_set:
        return 0.0
    tp = len(pred_set & true_set)
    if tp == 0:
        return 0.0
    prec = tp / len(pred_set)
    rec = tp / len(true_set)
    b2 = beta * beta  # 0.25 for beta=0.5
    denom = b2 * prec + rec
    if denom == 0:
        return 0.0
    return (1.0 + b2) * prec * rec / denom


def compute_macro_f05(
    predictions_by_s1: Dict[str, Set[str]],
    ground_truth_by_s1: Dict[str, Set[str]],
    all_s1_ids: List[str]
) -> float:
    """Compute macro-average F0.5 over all evaluation S1 entities."""
    scores = []
    for s1_id in all_s1_ids:
        preds = predictions_by_s1.get(s1_id, set())
        trues = ground_truth_by_s1.get(s1_id, set())
        scores.append(compute_entity_f_beta(preds, trues, beta=0.5))
    return float(np.mean(scores))


def train_matching_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray
) -> lgb.Booster:
    """Train LightGBM binary classifier with early stopping."""
    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
    
    model = lgb.train(
        LGBM_PARAMS,
        train_data,
        num_boost_round=400,
        valid_sets=[train_data, val_data],
        callbacks=[lgb.early_stopping(30, verbose=False)]
    )
    return model


def resolve_conflicts_and_filter(
    candidate_scores_by_s1: Dict[str, List[Tuple[str, float]]],
    threshold: float = OPTIMAL_THRESHOLD
) -> Dict[str, Set[str]]:
    """
    Filter predictions by probability threshold and enforce the domain constraint:
    Each S2 / S3 entity belongs to AT MOST ONE S1 entity (greedy assignment by highest probability).
    """
    # Collect all (score, s1_id, cid) tuples
    all_pairs = []
    for s1_id, cands in candidate_scores_by_s1.items():
        for cid, score in cands:
            if score >= threshold:
                all_pairs.append((score, s1_id, cid))
                
    # Sort descending by model confidence
    all_pairs.sort(key=lambda x: x[0], reverse=True)
    
    assigned_targets: Set[str] = set()
    final_matches: Dict[str, Set[str]] = {s1_id: set() for s1_id in candidate_scores_by_s1}
    
    for score, s1_id, cid in all_pairs:
        if cid not in assigned_targets:
            final_matches[s1_id].add(cid)
            assigned_targets.add(cid)
            
    return final_matches


def optimize_threshold(
    candidate_scores_by_s1: Dict[str, List[Tuple[str, float]]],
    ground_truth_by_s1: Dict[str, Set[str]],
    val_s1_ids: List[str]
) -> Tuple[float, float]:
    """Search for optimal probability threshold on validation set."""
    best_thresh = 0.5
    best_score = -1.0
    
    for thresh in np.arange(0.50, 0.92, 0.05):
        filtered_preds = resolve_conflicts_and_filter(candidate_scores_by_s1, threshold=thresh)
        score = compute_macro_f05(filtered_preds, ground_truth_by_s1, val_s1_ids)
        if score > best_score:
            best_score = score
            best_thresh = float(thresh)
            
    return best_thresh, best_score
