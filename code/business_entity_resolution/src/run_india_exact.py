"""
Exact India-only inference runner matching pipeline.py 100%.
Appends India predictions to existing France + US results.
"""
import os
import sys
import time
import shutil
from pathlib import Path
from typing import Dict, List, Set, Tuple
import numpy as np
import polars as pl
import lightgbm as lgb
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import (
    TEST_S1, TEST_S2, TEST_S3,
    OUTPUT_MATCHING, OUTPUT_CANDIDATES,
    OUTPUT_DIR
)
from src.blocking import CountryBlocker
from src.features import compute_pair_features
from src.model import resolve_conflicts_and_filter

def main():
    t_start = time.time()
    country = "India"
    chunk_size = 25000
    threshold = 0.65  # Optimal threshold evaluated on validation set (F0.5 = 0.9555)
    model_path = Path(__file__).resolve().parent / "lgbm_model.txt"

    print("=" * 60)
    print("Exact India Inference - Business Entity Resolution")
    print("=" * 60)
    print(f"Loading trained LightGBM model from {model_path}...")
    model = lgb.Booster(model_file=str(model_path))

    # Reset output files to France + US baseline
    fr_us_matching = OUTPUT_DIR / "matching_results_fr_us.tsv"
    fr_us_candidate = OUTPUT_DIR / "candidate_pairs_fr_us.tsv"

    print("Initializing final output files from France + US baseline...")
    shutil.copy2(str(fr_us_matching), str(OUTPUT_MATCHING))
    shutil.copy2(str(fr_us_candidate), str(OUTPUT_CANDIDATES))

    # Count initial lines
    with open(OUTPUT_MATCHING, "r", encoding="utf-8") as f:
        init_m_lines = sum(1 for _ in f)
    with open(OUTPUT_CANDIDATES, "r", encoding="utf-8") as f:
        init_c_lines = sum(1 for _ in f)
    print(f"Base files ready: {init_m_lines} matching rows, {init_c_lines} candidate rows (France + US complete).")

    # Load targets from S2 and S3 for India
    t0 = time.time()
    print("\nLoading India test targets from S2 and S3...")
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
    print(f"Building India inverted index for {len(blocker.target_records)} targets ({time.time()-t0:.2f}s)...")
    blocker.build_index()

    # Free raw polars target dataframes to conserve memory
    del s2_c, s3_c

    # Load S1 records for India
    s1_c = pl.read_csv(TEST_S1, separator="\t").filter(pl.col("country") == country)
    n_s1_country = len(s1_c)
    print(f"Running candidate generation and ML scoring on {n_s1_country} India S1 records...")

    s1_ids = s1_c["entity_id"].to_list()
    s1_names = s1_c["business_name"].to_list()
    s1_addrs = s1_c["business_address"].to_list()
    del s1_c

    total_candidates_written = 0
    total_matches_written = 0
    total_s1_processed = 0

    with open(OUTPUT_MATCHING, "a", encoding="utf-8") as matching_f, \
         open(OUTPUT_CANDIDATES, "a", encoding="utf-8") as candidate_f:

        for start_i in tqdm(range(0, n_s1_country, chunk_size), desc="Scoring India"):
            end_i = min(start_i + chunk_size, n_s1_country)
            chunk_ids = s1_ids[start_i:end_i]
            chunk_names = s1_names[start_i:end_i]
            chunk_addrs = s1_addrs[start_i:end_i]

            chunk_cands_by_s1 = {}
            chunk_pairs_X = []
            pair_pointers = []

            # Step 1: Candidate Generation (Blocking)
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

            # Step 2: ML Scoring
            if chunk_pairs_X:
                X_batch = np.array(chunk_pairs_X, dtype=np.float32)
                batch_probs = model.predict(X_batch)

                scored_by_s1: Dict[str, List[Tuple[str, float]]] = {s1_id: [] for s1_id in chunk_ids}
                for (s1_id, cid), prob in zip(pair_pointers, batch_probs):
                    scored_by_s1[s1_id].append((cid, float(prob)))

                chunk_matches = resolve_conflicts_and_filter(scored_by_s1, threshold=threshold)
            else:
                chunk_matches = {s1_id: set() for s1_id in chunk_ids}

            # Step 3: Write Matching Results
            for s1_id in chunk_ids:
                matches = chunk_matches.get(s1_id, set())
                match_str = ",".join(sorted(matches))
                matching_f.write(f"{s1_id}\t{match_str}\n")
                total_matches_written += len(matches)
                total_s1_processed += 1

            matching_f.flush()
            candidate_f.flush()

    total_time = (time.time() - t_start) / 60
    print("\n" + "=" * 60)
    print(f"India Inference Complete in {total_time:.2f} minutes!")
    print(f"India S1 processed : {total_s1_processed}")
    print(f"India candidates   : {total_candidates_written}")
    print(f"India matches      : {total_matches_written}")

    # Verify total line counts
    with open(OUTPUT_MATCHING, "r", encoding="utf-8") as f:
        final_m = sum(1 for _ in f)
    with open(OUTPUT_CANDIDATES, "r", encoding="utf-8") as f:
        final_c = sum(1 for _ in f)

    print(f"Final matching_results.tsv : {final_m} lines (expected 1732545)")
    print(f"Final candidate_pairs.tsv  : {final_c} lines (expected 1732545)")
    print("=" * 60)

if __name__ == "__main__":
    main()
