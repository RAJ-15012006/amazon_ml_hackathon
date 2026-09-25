"""
Fast India-only inference using threading for parallelism.
Threads share memory (no duplication of 4.7M target records),
and RapidFuzz / NumPy / LightGBM all release the GIL.
"""
import os
import sys
import time
import shutil
from pathlib import Path
from typing import Dict, List, Set, Tuple, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
import polars as pl
import lightgbm as lgb
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import (
    TEST_S1, TEST_S2, TEST_S3,
    OUTPUT_DIR, OPTIMAL_THRESHOLD
)
from src.blocking import CountryBlocker
from src.features import compute_pair_features
from src.model import resolve_conflicts_and_filter


def process_chunk(chunk_idx, chunk_ids, chunk_names, chunk_addrs, blocker, model, threshold):
    """Process a single chunk: blocking + features + scoring. Thread-safe (read-only on blocker)."""
    country = "India"
    matching_lines = []
    candidate_lines = []

    chunk_pairs_X = []
    pair_pointers = []

    # Step 1: Generate candidates
    for s1_id, name, addr in zip(chunk_ids, chunk_names, chunk_addrs):
        s1_rec, cands = blocker.get_candidates_for_s1(name, addr)

        cand_str = ",".join(cands)
        candidate_lines.append(f"{s1_id}\t{cand_str}")

        for cid in cands:
            rec2 = blocker.target_records[cid]
            feat = compute_pair_features(s1_rec, rec2, cid, country)
            chunk_pairs_X.append(feat)
            pair_pointers.append((s1_id, cid))

    # Step 2: Score with LightGBM
    if chunk_pairs_X:
        X_batch = np.array(chunk_pairs_X, dtype=np.float32)
        batch_probs = model.predict(X_batch)

        scored_by_s1: Dict[str, List[Tuple[str, float]]] = {s1_id: [] for s1_id in chunk_ids}
        for (s1_id, cid), prob in zip(pair_pointers, batch_probs):
            scored_by_s1[s1_id].append((cid, float(prob)))

        chunk_matches = resolve_conflicts_and_filter(scored_by_s1, threshold=threshold)
    else:
        chunk_matches = {s1_id: set() for s1_id in chunk_ids}

    # Step 3: Build output lines
    for s1_id in chunk_ids:
        matches = chunk_matches.get(s1_id, set())
        match_str = ",".join(sorted(matches))
        matching_lines.append(f"{s1_id}\t{match_str}")

    return chunk_idx, matching_lines, candidate_lines


def main():
    t_start = time.time()
    # Use 4 threads - good balance for M3 (GIL released by C extensions)
    N_THREADS = 4
    CHUNK_SIZE = 25000
    MODEL_PATH = str(Path(__file__).resolve().parent / "lgbm_model.txt")

    # Load the model to get the actual trained threshold
    # The pipeline optimized to threshold=0.65 during training
    THRESHOLD = 0.65  # Optimal threshold from train_and_validate

    print(f"=== Fast India Inference (Threading) ===")
    print(f"Threads: {N_THREADS}, Chunk size: {CHUNK_SIZE}, Threshold: {THRESHOLD}")

    model = lgb.Booster(model_file=MODEL_PATH)
    print(f"Model loaded from {MODEL_PATH}")

    # Build the India blocking index
    print("Loading India test targets from S2 and S3...")
    s2_c = pl.read_csv(TEST_S2, separator="\t").filter(pl.col("country") == "India")
    s3_c = pl.read_csv(TEST_S3, separator="\t").filter(pl.col("country") == "India")

    blocker = CountryBlocker("India")
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
    print(f"Building India inverted index for {len(blocker.target_records)} targets...")
    t_idx = time.time()
    blocker.build_index()
    print(f"Index built in {time.time() - t_idx:.1f}s")

    del s2_c, s3_c

    # Load S1 India records
    print("Loading India S1 test records...")
    s1_c = pl.read_csv(TEST_S1, separator="\t").filter(pl.col("country") == "India")
    n_total = len(s1_c)
    print(f"Total India S1 records: {n_total}")

    s1_ids = s1_c["entity_id"].to_list()
    s1_names = s1_c["business_name"].to_list()
    s1_addrs = s1_c["business_address"].to_list()
    del s1_c

    # Prepare chunks
    chunks = []
    for i, start_i in enumerate(range(0, n_total, CHUNK_SIZE)):
        end_i = min(start_i + CHUNK_SIZE, n_total)
        chunks.append((i, s1_ids[start_i:end_i], s1_names[start_i:end_i], s1_addrs[start_i:end_i]))

    n_chunks = len(chunks)
    print(f"Split into {n_chunks} chunks, launching {N_THREADS} threads...")

    # Process with thread pool
    results = [None] * n_chunks
    completed = 0

    with ThreadPoolExecutor(max_workers=N_THREADS) as executor:
        futures = {
            executor.submit(process_chunk, idx, ids, names, addrs, blocker, model, THRESHOLD): idx
            for idx, ids, names, addrs in chunks
        }

        for future in as_completed(futures):
            chunk_idx, matching_lines, candidate_lines = future.result()
            results[chunk_idx] = (matching_lines, candidate_lines)
            completed += 1
            elapsed = time.time() - t_start
            rate = completed / elapsed * 60 if elapsed > 0 else 0
            remaining_est = (n_chunks - completed) / rate if rate > 0 else 0
            pct = completed / n_chunks * 100
            print(f"  ✓ Chunk {chunk_idx+1}/{n_chunks} done ({pct:.0f}%) "
                  f"- {elapsed/60:.1f}min elapsed, ~{remaining_est:.0f}min remaining")

    # Write output files
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    matching_path = OUTPUT_DIR / "matching_results.tsv"
    candidate_path = OUTPUT_DIR / "candidate_pairs.tsv"

    fr_us_matching = OUTPUT_DIR / "matching_results_fr_us.tsv"
    fr_us_candidate = OUTPUT_DIR / "candidate_pairs_fr_us.tsv"

    print("Writing final output files...")
    shutil.copy2(str(fr_us_matching), str(matching_path))
    shutil.copy2(str(fr_us_candidate), str(candidate_path))

    total_matches = 0
    with open(matching_path, "a", encoding="utf-8") as mf, \
         open(candidate_path, "a", encoding="utf-8") as cf:
        for matching_lines, candidate_lines in results:
            for line in matching_lines:
                mf.write(line + "\n")
                parts = line.split("\t")
                if len(parts) > 1 and parts[1]:
                    total_matches += len(parts[1].split(","))
            for line in candidate_lines:
                cf.write(line + "\n")

    elapsed_total = time.time() - t_start

    # Verify line counts
    with open(matching_path, "r") as f:
        n_matching = sum(1 for _ in f)
    with open(candidate_path, "r") as f:
        n_candidate = sum(1 for _ in f)

    print(f"\n{'='*55}")
    print(f"India Threaded Inference COMPLETE!")
    print(f"Total India S1 processed: {n_total}")
    print(f"Total India matches: {total_matches}")
    print(f"Time: {elapsed_total/60:.1f} minutes")
    print(f"Matching file: {matching_path} ({n_matching} lines)")
    print(f"Candidate file: {candidate_path} ({n_candidate} lines)")
    print(f"Expected lines: 1732545")
    print(f"Match: {'✅ YES' if n_matching == 1732545 else '❌ NO - ' + str(n_matching)}")
    print(f"{'='*55}")


if __name__ == "__main__":
    main()
