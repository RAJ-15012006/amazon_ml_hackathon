"""
End-to-end execution pipeline for the Business Entity Resolution Challenge.
Handles model training, validation, threshold optimization, and test inference across countries.
"""
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Set, Tuple
import numpy as np
import polars as pl
import lightgbm as lgb
from tqdm import tqdm

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from src.config import (
        TRAIN_S1, TRAIN_S2, TRAIN_S3, TRAIN_GT,
        TEST_S1, TEST_S2, TEST_S3,
        OUTPUT_MATCHING, OUTPUT_CANDIDATES,
        OUTPUT_DIR, PLOTS_DIR, OPTIMAL_THRESHOLD, LGBM_PARAMS
    )
    from src.blocking import CountryBlocker
    from src.features import compute_pair_features
    from src.model import (
        compute_macro_f05,
        optimize_threshold,
        resolve_conflicts_and_filter,
        train_matching_model
    )
else:
    from .config import (
        TRAIN_S1, TRAIN_S2, TRAIN_S3, TRAIN_GT,
        TEST_S1, TEST_S2, TEST_S3,
        OUTPUT_MATCHING, OUTPUT_CANDIDATES,
        OUTPUT_DIR, PLOTS_DIR, OPTIMAL_THRESHOLD, LGBM_PARAMS
    )
    from .blocking import CountryBlocker
    from .features import compute_pair_features
    from .model import (
        compute_macro_f05,
        optimize_threshold,
        resolve_conflicts_and_filter,
        train_matching_model
    )


def load_training_data_cohort(
    n_sample_per_country: int = 30000
) -> Tuple[Dict[str, Set[str]], pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """Load a balanced training cohort of US and India S1 records with their targets."""
    print(f"Loading ground truth and sampling {n_sample_per_country} per country...")
    gt = pl.read_csv(TRAIN_GT, separator="\t")
    s1 = pl.read_csv(TRAIN_S1, separator="\t")
    
    # Split S1 by country
    s1_us = s1.filter(pl.col("country") == "US").slice(0, n_sample_per_country)
    s1_in = s1.filter(pl.col("country") == "India").slice(0, n_sample_per_country)
    s1_sample = pl.concat([s1_us, s1_in])
    sample_ids = set(s1_sample["entity_id"])
    
    # Filter ground truth for sample
    gt_sample = gt.filter(pl.col("source1_entity_id").is_in(sample_ids))
    gt_map: Dict[str, Set[str]] = {}
    needed_m_ids: Set[str] = set()
    
    for r in gt_sample.iter_rows(named=True):
        m_str = r["matched_entity_ids"]
        if m_str:
            ms = set(m_str.split(","))
            gt_map[r["source1_entity_id"]] = ms
            needed_m_ids.update(ms)
        else:
            gt_map[r["source1_entity_id"]] = set()
            
    print(f"Loaded {len(s1_sample)} S1 entities with {len(needed_m_ids)} matched targets.")
    
    # Load targets + distractors from S2 and S3
    print("Loading S2 and S3 pools...")
    s2 = pl.read_csv(TRAIN_S2, separator="\t")
    s3 = pl.read_csv(TRAIN_S3, separator="\t")
    
    s2_targets = s2.filter(pl.col("entity_id").is_in(needed_m_ids))
    s2_distract = s2.slice(0, 100000)
    s2_pool = pl.concat([s2_targets, s2_distract]).unique(subset=["entity_id"])
    
    s3_targets = s3.filter(pl.col("entity_id").is_in(needed_m_ids))
    s3_distract = s3.slice(0, 100000)
    s3_pool = pl.concat([s3_targets, s3_distract]).unique(subset=["entity_id"])
    
    return gt_map, s1_sample, s2_pool, s3_pool


def build_training_dataset(
    gt_map: Dict[str, Set[str]],
    s1_df: pl.DataFrame,
    s2_df: pl.DataFrame,
    s3_df: pl.DataFrame
) -> Tuple[np.ndarray, np.ndarray, Dict[str, List[Tuple[str, float, int]]], List[str]]:
    """Generate candidate pairs and extract features for training."""
    countries = ["US", "India"]
    all_X = []
    all_y = []
    all_s1_candidates = {}
    all_s1_ids = []
    
    for country in countries:
        print(f"\n--- Generating training candidates for {country} ---")
        blocker = CountryBlocker(country)
        
        c_s2 = s2_df.filter(pl.col("country") == country)
        c_s3 = s3_df.filter(pl.col("country") == country)
        
        blocker.add_target_pool(
            c_s2["entity_id"].to_list(),
            c_s2["business_name"].to_list(),
            c_s2["business_address"].to_list()
        )
        blocker.add_target_pool(
            c_s3["entity_id"].to_list(),
            c_s3["business_name"].to_list(),
            c_s3["business_address"].to_list()
        )
        print(f"Building {country} inverted index over {len(blocker.target_records)} targets...")
        blocker.build_index()
        
        c_s1 = s1_df.filter(pl.col("country") == country)
        s1_ids = c_s1["entity_id"].to_list()
        s1_names = c_s1["business_name"].to_list()
        s1_addrs = c_s1["business_address"].to_list()
        all_s1_ids.extend(s1_ids)
        
        for s1_id, name, addr in zip(s1_ids, s1_names, s1_addrs):
            s1_rec, cands = blocker.get_candidates_for_s1(name, addr)
            true_set = gt_map.get(s1_id, set())
            cand_info = []
            
            for cid in cands:
                rec2 = blocker.target_records[cid]
                feat = compute_pair_features(s1_rec, rec2, cid, country)
                label = 1 if cid in true_set else 0
                all_X.append(feat)
                all_y.append(label)
                cand_info.append((cid, label))
                
            all_s1_candidates[s1_id] = cand_info
            
    X = np.array(all_X, dtype=np.float32)
    y = np.array(all_y, dtype=np.int32)
    print(f"\nTraining dataset ready: {X.shape} features, {np.sum(y)} positive pairs ({np.sum(y)/len(y)*100:.1f}%)")
    return X, y, all_s1_candidates, all_s1_ids


def train_and_validate(
    n_sample_per_country: int = 30000
) -> Tuple[lgb.Booster, float]:
    """Train LightGBM model and optimize decision threshold for macro F0.5."""
    gt_map, s1_sample, s2_pool, s3_pool = load_training_data_cohort(n_sample_per_country)
    X, y, all_s1_candidates, all_s1_ids = build_training_dataset(gt_map, s1_sample, s2_pool, s3_pool)
    
    # Split train and validation by S1 entities to avoid any entity leakage
    np.random.seed(42)
    shuffled_s1 = np.random.permutation(all_s1_ids)
    n_val = int(len(shuffled_s1) * 0.25)
    val_s1_set = set(shuffled_s1[:n_val])
    val_s1_list = list(shuffled_s1[:n_val])
    train_s1_set = set(shuffled_s1[n_val:])
    
    # Map index in X, y to S1
    idx_train, idx_val = [], []
    current_idx = 0
    val_candidate_indices = {}
    
    for s1_id in all_s1_ids:
        cands = all_s1_candidates.get(s1_id, [])
        cand_idx_list = []
        for cid, label in cands:
            if s1_id in val_s1_set:
                idx_val.append(current_idx)
                cand_idx_list.append((cid, current_idx))
            else:
                idx_train.append(current_idx)
            current_idx += 1
        if s1_id in val_s1_set:
            val_candidate_indices[s1_id] = cand_idx_list
            
    X_train, y_train = X[idx_train], y[idx_train]
    X_val, y_val = X[idx_val], y[idx_val]
    
    print(f"Training split: {X_train.shape[0]} pairs, Validation split: {X_val.shape[0]} pairs.")
    model = train_matching_model(X_train, y_train, X_val, y_val)
    
    # Predict on validation pairs
    print("Scoring validation set and optimizing threshold for macro F0.5...")
    val_preds = model.predict(X_val)
    
    # Group predictions by S1
    val_scores_by_s1 = {}
    for s1_id, cand_pairs in val_candidate_indices.items():
        val_scores_by_s1[s1_id] = []
        for cid, feat_idx in cand_pairs:
            val_idx_pos = idx_val.index(feat_idx)
            val_scores_by_s1[s1_id].append((cid, float(val_preds[val_idx_pos])))
            
    best_th, best_f05 = optimize_threshold(val_scores_by_s1, gt_map, val_s1_list)
    print(f"\n==========================================")
    print(f"Optimal Threshold: {best_th:.2f}")
    print(f"Validation Macro F0.5 Score: {best_f05:.4f}")
    print(f"==========================================\n")
    
    # Save model artifact
    model_path = Path(__file__).resolve().parent / "lgbm_model.txt"
    model.save_model(str(model_path))
    print(f"Saved trained model to {model_path}")
    
    return model, best_th


def run_test_inference(
    model: lgb.Booster,
    threshold: float = OPTIMAL_THRESHOLD,
    chunk_size: int = 25000
):
    """
    Execute streaming inference over test sets for France, US, and India.
    Generates output/candidate_pairs.tsv and output/matching_results.tsv.
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    matching_f = open(OUTPUT_MATCHING, "w", encoding="utf-8")
    candidate_f = open(OUTPUT_CANDIDATES, "w", encoding="utf-8")
    
    matching_f.write("source1_entity_id\tmatched_entity_ids\n")
    candidate_f.write("source1_entity_id\tcandidate_entity_ids\n")
    
    countries = ["France", "US", "India"]
    total_s1_processed = 0
    total_candidates_written = 0
    total_matches_written = 0
    
    for country in countries:
        print(f"\n=======================================================")
        print(f"Processing Test Country: {country}")
        print(f"=======================================================")
        
        t0 = time.time()
        print(f"Loading {country} test targets from S2 and S3...")
        s2_c = pl.read_csv(TEST_S2, separator="\t").filter(pl.col("country") == country)
        s3_c = pl.read_csv(TEST_S3, separator="\t").filter(pl.col("country") == country)
        
        blocker = CountryBlocker(country)
        blocker.add_target_pool(
            s2_c["entity_id"].to_list(),
            s2_c["business_name"].to_list(),
            s2_c["business_address"].to_list()
        )
        blocker.add_target_pool(
            s3_c["entity_id"].to_list(),
            s3_c["business_name"].to_list(),
            s3_c["business_address"].to_list()
        )
        print(f"Building {country} inverted index for {len(blocker.target_records)} targets ({time.time()-t0:.2f}s)...")
        blocker.build_index()
        
        # Load S1 records for this country
        s1_c = pl.read_csv(TEST_S1, separator="\t").filter(pl.col("country") == country)
        n_s1_country = len(s1_c)
        print(f"Running candidate generation and ML scoring on {n_s1_country} S1 records...")
        
        s1_ids = s1_c["entity_id"].to_list()
        s1_names = s1_c["business_name"].to_list()
        s1_addrs = s1_c["business_address"].to_list()
        
        # Process in chunks to maintain low memory profile and stream outputs
        for start_i in tqdm(range(0, n_s1_country, chunk_size), desc=f"Scoring {country}"):
            end_i = min(start_i + chunk_size, n_s1_country)
            chunk_ids = s1_ids[start_i:end_i]
            chunk_names = s1_names[start_i:end_i]
            chunk_addrs = s1_addrs[start_i:end_i]
            
            chunk_s1_records = []
            chunk_cands_by_s1 = {}
            chunk_pairs_X = []
            pair_pointers = [] # (s1_id, cid)
            
            # Step 1: Generate candidates
            for s1_id, name, addr in zip(chunk_ids, chunk_names, chunk_addrs):
                s1_rec, cands = blocker.get_candidates_for_s1(name, addr)
                chunk_cands_by_s1[s1_id] = cands
                
                # Write candidate pairs immediately
                cand_str = ",".join(cands)
                candidate_f.write(f"{s1_id}\t{cand_str}\n")
                total_candidates_written += len(cands)
                
                for cid in cands:
                    rec2 = blocker.target_records[cid]
                    feat = compute_pair_features(s1_rec, rec2, cid, country)
                    chunk_pairs_X.append(feat)
                    pair_pointers.append((s1_id, cid))
                    
            # Step 2: Score candidates with LightGBM
            if chunk_pairs_X:
                X_batch = np.array(chunk_pairs_X, dtype=np.float32)
                batch_probs = model.predict(X_batch)
                
                # Group scores by S1
                scored_by_s1: Dict[str, List[Tuple[str, float]]] = {s1_id: [] for s1_id in chunk_ids}
                for (s1_id, cid), prob in zip(pair_pointers, batch_probs):
                    scored_by_s1[s1_id].append((cid, float(prob)))
                    
                # Resolve conflicts and apply optimal threshold
                chunk_matches = resolve_conflicts_and_filter(scored_by_s1, threshold=threshold)
            else:
                chunk_matches = {s1_id: set() for s1_id in chunk_ids}
                
            # Step 3: Write matching results
            for s1_id in chunk_ids:
                matches = chunk_matches.get(s1_id, set())
                match_str = ",".join(sorted(matches))
                matching_f.write(f"{s1_id}\t{match_str}\n")
                total_matches_written += len(matches)
                total_s1_processed += 1
                
        # Clean up country structures to release memory
        del blocker
        del s2_c
        del s3_c
        del s1_c
        
    matching_f.close()
    candidate_f.close()
    
    print("\n=======================================================")
    print(f"Test Inference Completed Successfully!")
    print(f"Total S1 entities processed: {total_s1_processed}")
    print(f"Total candidates produced : {total_candidates_written}")
    print(f"Total matches predicted   : {total_matches_written}")
    print(f"Matching file: {OUTPUT_MATCHING}")
    print(f"Candidate file: {OUTPUT_CANDIDATES}")
    print(f"=======================================================")


if __name__ == "__main__":
    t_start = time.time()
    print("Starting Business Entity Resolution Pipeline...")
    model, best_thresh = train_and_validate(n_sample_per_country=30000)
    run_test_inference(model, threshold=best_thresh, chunk_size=25000)
    print(f"All operations completed in {(time.time() - t_start)/60:.2f} minutes.")
