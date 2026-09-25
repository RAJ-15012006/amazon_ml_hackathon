"""
Parallel India-only inference using multiprocessing for 4-6x speedup.
Builds the blocking index once, then distributes S1 chunks across N workers.
"""
import os
import sys
import time
import pickle
from pathlib import Path
from typing import Dict, List, Set, Tuple, Any
from multiprocessing import Pool, cpu_count
import numpy as np
import polars as pl
import lightgbm as lgb
from tqdm import tqdm

# Fix imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import (
    TEST_S1, TEST_S2, TEST_S3,
    OUTPUT_DIR, OPTIMAL_THRESHOLD
)
from src.blocking import CountryBlocker, parse_for_blocking
from src.features import compute_pair_features
from src.model import resolve_conflicts_and_filter

# ---- Globals for worker processes (set via initializer) ----
_worker_target_records = None
_worker_doc_freqs = None
_worker_inverted_index = None
_worker_model = None
_worker_threshold = None


def _init_worker(target_records, doc_freqs, inverted_index, model_path, threshold):
    """Initialize each worker process with shared data."""
    global _worker_target_records, _worker_doc_freqs, _worker_inverted_index
    global _worker_model, _worker_threshold
    _worker_target_records = target_records
    _worker_doc_freqs = doc_freqs
    _worker_inverted_index = inverted_index
    _worker_model = lgb.Booster(model_file=model_path)
    _worker_threshold = threshold


def _process_chunk(args):
    """Process a chunk of S1 entities: blocking + features + scoring."""
    chunk_idx, chunk_ids, chunk_names, chunk_addrs = args
    country = "India"

    # Reconstruct a lightweight blocker using shared data
    blocker = CountryBlocker(country)
    blocker.target_records = _worker_target_records
    blocker.doc_freqs = _worker_doc_freqs
    blocker.inverted_index = _worker_inverted_index

    matching_lines = []
    candidate_lines = []

    chunk_s1_records = []
    chunk_cands_by_s1 = {}
    chunk_pairs_X = []
    pair_pointers = []

    # Step 1: Generate candidates
    for s1_id, name, addr in zip(chunk_ids, chunk_names, chunk_addrs):
        s1_rec, cands = blocker.get_candidates_for_s1(name, addr)
        chunk_cands_by_s1[s1_id] = cands

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
        batch_probs = _worker_model.predict(X_batch)

        scored_by_s1: Dict[str, List[Tuple[str, float]]] = {s1_id: [] for s1_id in chunk_ids}
        for (s1_id, cid), prob in zip(pair_pointers, batch_probs):
            scored_by_s1[s1_id].append((cid, float(prob)))

        chunk_matches = resolve_conflicts_and_filter(scored_by_s1, threshold=_worker_threshold)
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
    N_WORKERS = min(cpu_count(), 8)  # Use up to 8 workers
    CHUNK_SIZE = 25000
    MODEL_PATH = str(Path(__file__).resolve().parent / "lgbm_model.txt")
    THRESHOLD = OPTIMAL_THRESHOLD

    print(f"=== Parallel India Inference ===")
    print(f"Workers: {N_WORKERS}, Chunk size: {CHUNK_SIZE}")

    # Load model to verify it exists
    model_check = lgb.Booster(model_file=MODEL_PATH)
    print(f"Model loaded from {MODEL_PATH}")

    # Build the India blocking index (this is the expensive one-time cost)
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

    # Free the raw dataframes
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
        chunks.append((
            i,
            s1_ids[start_i:end_i],
            s1_names[start_i:end_i],
            s1_addrs[start_i:end_i]
        ))

    print(f"Split into {len(chunks)} chunks, launching {N_WORKERS} parallel workers...")

    # Launch multiprocessing pool
    results = [None] * len(chunks)
    completed = 0

    with Pool(
        processes=N_WORKERS,
        initializer=_init_worker,
        initargs=(
            blocker.target_records,
            blocker.doc_freqs,
            blocker.inverted_index,
            MODEL_PATH,
            THRESHOLD
        )
    ) as pool:
        for result in pool.imap_unordered(_process_chunk, chunks):
            chunk_idx, matching_lines, candidate_lines = result
            results[chunk_idx] = (matching_lines, candidate_lines)
            completed += 1
            elapsed = time.time() - t_start
            pct = completed / len(chunks) * 100
            print(f"  ✓ Chunk {chunk_idx+1}/{len(chunks)} done ({pct:.0f}%) - elapsed: {elapsed/60:.1f}min")

    # Write output files: append India results to France+US results
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    matching_path = OUTPUT_DIR / "matching_results.tsv"
    candidate_path = OUTPUT_DIR / "candidate_pairs.tsv"

    # Copy France+US results as base
    fr_us_matching = OUTPUT_DIR / "matching_results_fr_us.tsv"
    fr_us_candidate = OUTPUT_DIR / "candidate_pairs_fr_us.tsv"

    print("Writing final output files...")
    import shutil
    shutil.copy2(str(fr_us_matching), str(matching_path))
    shutil.copy2(str(fr_us_candidate), str(candidate_path))

    # Append India results in order
    total_matches = 0
    with open(matching_path, "a", encoding="utf-8") as mf, \
         open(candidate_path, "a", encoding="utf-8") as cf:
        for matching_lines, candidate_lines in results:
            for line in matching_lines:
                mf.write(line + "\n")
                if line.split("\t")[1]:  # has matches
                    total_matches += len(line.split("\t")[1].split(","))
            for line in candidate_lines:
                cf.write(line + "\n")

    elapsed_total = time.time() - t_start
    print(f"\n{'='*55}")
    print(f"India Parallel Inference COMPLETE!")
    print(f"Total India S1 processed: {n_total}")
    print(f"Total India matches: {total_matches}")
    print(f"Time: {elapsed_total/60:.1f} minutes")
    print(f"Matching file: {matching_path}")
    print(f"Candidate file: {candidate_path}")
    print(f"{'='*55}")


if __name__ == "__main__":
    main()
