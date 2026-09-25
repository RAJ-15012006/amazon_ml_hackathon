"""
Resume India inference from chunk 8 (record index 200,000).
Appends remaining 609,986 India records to output/matching_results.tsv
and output/candidate_pairs.tsv.
"""
import os
import sys
import time
import subprocess
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
    OUTPUT_DIR, WORKSPACE_ROOT
)
from src.blocking import CountryBlocker
from src.features import compute_pair_features
from src.model import resolve_conflicts_and_filter

def main():
    t_start = time.time()
    country = "India"
    chunk_size = 25000
    threshold = 0.65
    resume_from_index = 200000  # Exactly 8 chunks of 25,000 already completed
    model_path = Path(__file__).resolve().parent / "lgbm_model.txt"

    print("=" * 60)
    print("Resuming India Inference from Entity Index 200,000")
    print("=" * 60)
    print(f"Loading trained LightGBM model from {model_path}...")
    model = lgb.Booster(model_file=str(model_path))

    # Verify existing file line counts
    with open(OUTPUT_MATCHING, "r", encoding="utf-8") as f:
        m_lines = sum(1 for _ in f)
    with open(OUTPUT_CANDIDATES, "r", encoding="utf-8") as f:
        c_lines = sum(1 for _ in f)

    print(f"Existing files check: {m_lines} matching rows, {c_lines} candidate rows.")
    assert m_lines == 1122559, f"Expected 1122559 matching rows, found {m_lines}"
    assert c_lines == 1122559, f"Expected 1122559 candidate rows, found {c_lines}"
    print("Both output files match perfectly at chunk 8 boundary (1,122,559 lines).")

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

    del s2_c, s3_c

    # Load S1 records for India
    s1_c = pl.read_csv(TEST_S1, separator="\t").filter(pl.col("country") == country)
    n_s1_country = len(s1_c)
    remaining_count = n_s1_country - resume_from_index
    print(f"Total India S1 records: {n_s1_country}. Remaining to process: {remaining_count} ({remaining_count / chunk_size:.1f} chunks).")

    s1_ids = s1_c["entity_id"].to_list()
    s1_names = s1_c["business_name"].to_list()
    s1_addrs = s1_c["business_address"].to_list()
    del s1_c

    total_candidates_written = 0
    total_matches_written = 0
    total_s1_processed = 0

    with open(OUTPUT_MATCHING, "a", encoding="utf-8") as matching_f, \
         open(OUTPUT_CANDIDATES, "a", encoding="utf-8") as candidate_f:

        for start_i in tqdm(range(resume_from_index, n_s1_country, chunk_size), desc="Scoring India (Resumed)"):
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
    print(f"India Remaining Inference Complete in {total_time:.2f} minutes!")
    print(f"India S1 processed in this run : {total_s1_processed}")

    # Verify total line counts
    with open(OUTPUT_MATCHING, "r", encoding="utf-8") as f:
        final_m = sum(1 for _ in f)
    with open(OUTPUT_CANDIDATES, "r", encoding="utf-8") as f:
        final_c = sum(1 for _ in f)

    print(f"Final matching_results.tsv : {final_m} lines (expected 1732545)")
    print(f"Final candidate_pairs.tsv  : {final_c} lines (expected 1732545)")
    print("=" * 60)

    if final_m == 1732545 and final_c == 1732545:
        print("\nAll lines verified! Running official submission validator...")
        val_cmd = [
            sys.executable,
            str(WORKSPACE_ROOT / "student_resource" / "utils" / "validate_submission.py"),
            "--matching", str(OUTPUT_MATCHING),
            "--candidate", str(OUTPUT_CANDIDATES),
            "--test-dir", str(WORKSPACE_ROOT / "student_resource" / "dataset" / "test")
        ]
        res = subprocess.run(val_cmd, capture_output=True, text=True)
        print("Validator stdout:\n", res.stdout)
        if res.stderr:
            print("Validator stderr:\n", res.stderr)
        if res.returncode == 0:
            print("\n>>> VALIDATOR RESULT: PASS (Exit code 0)! <<<")
            print("Running package_submission.py...")
            pkg_res = subprocess.run([sys.executable, str(WORKSPACE_ROOT / "package_submission.py")], capture_output=True, text=True)
            print(pkg_res.stdout)
        else:
            print(f"\n>>> VALIDATOR FAILED WITH CODE {res.returncode} <<<")
    else:
        print(f"ERROR: Expected 1732545 lines, got {final_m} and {final_c}")

if __name__ == "__main__":
    main()
