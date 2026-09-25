"""
Precision Refinement Script for Business Entity Resolution.
Transforms raw matching results into high-precision, competition-winning output:
1. Filters out co-located false positives (businesses at same address but different names).
2. Uses distinctive name matching (removes address/city tokens).
3. Enforces global 1-to-N greedy conflict resolution across the full 1.73M entities.
4. Caps matches per entity to at most 3 from S2 and 3 from S3 (aligns with GT distribution).
5. Protects singletons (entities with no confident match get empty list -> full 1.0 score).
"""
import sys
import time
import shutil
import re
from pathlib import Path
from collections import defaultdict, Counter
from typing import Dict, List, Set, Tuple
import numpy as np
import polars as pl
from rapidfuzz import fuzz
from unidecode import unidecode
from tqdm import tqdm

WORKSPACE_ROOT = Path(__file__).resolve().parent
TEST_DIR = WORKSPACE_ROOT / "student_resource" / "dataset" / "test"
OUTPUT_DIR = WORKSPACE_ROOT / "output"
MATCHING_FILE = OUTPUT_DIR / "matching_results.tsv"
BACKUP_FILE = OUTPUT_DIR / "matching_results_raw_067.tsv"
PRECISION_FILE = OUTPUT_DIR / "matching_results_precision.tsv"

URL_PAT = re.compile(r'(https?://|www\.|\.com|\.in|\.co|\.fr|@)', re.I)
NON_ALPHANUM = re.compile(r'[^a-z0-9\s]')

LEGAL_SUFFIXES = {
    'inc', 'incorporated', 'corp', 'corporation', 'ltd', 'limited', 'pvt', 'private',
    'llc', 'llp', 'co', 'company', 'sarl', 'sa', 'sasu', 'eurl', 'sci', 'gmbh',
    'associates', 'enterprise', 'enterprises', 'services', 'industries', 'cie'
}

def clean_name(name: str) -> str:
    if not name:
        return ""
    t = URL_PAT.sub(" ", unidecode(name).lower())
    words = [w for w in NON_ALPHANUM.sub(" ", t).split() if w not in LEGAL_SUFFIXES]
    return " ".join(words)

def clean_addr(addr: str) -> str:
    if not addr:
        return ""
    t = URL_PAT.sub(" ", unidecode(addr).lower())
    return " ".join(NON_ALPHANUM.sub(" ", t).split())

def main():
    t0 = time.time()
    print("=" * 65)
    print("Precision Refinement Pipeline - Amazon ML Challenge 2026")
    print("=" * 65)

    # Step 1: Backup current file if not already backed up
    if not BACKUP_FILE.exists():
        print(f"Creating backup of raw 0.676 score file at {BACKUP_FILE.name}...")
        shutil.copy2(str(MATCHING_FILE), str(BACKUP_FILE))
    else:
        print(f"Backup already exists at {BACKUP_FILE.name}.")

    # Step 2: Read candidate pairs from the raw matching results
    print("\nReading candidate pairs from matching_results.tsv...")
    s1_ordered_ids = []
    pairs = []
    unique_target_ids = set()

    with open(MATCHING_FILE, "r", encoding="utf-8") as f:
        header = next(f)
        for line in f:
            p = line.strip().split("\t")
            s1_id = p[0]
            s1_ordered_ids.append(s1_id)
            if len(p) > 1 and p[1]:
                for cid in p[1].split(","):
                    pairs.append((s1_id, cid))
                    unique_target_ids.add(cid)

    print(f"Total S1 entities: {len(s1_ordered_ids):,}")
    print(f"Total pairs to evaluate: {len(pairs):,}")
    print(f"Unique target records: {len(unique_target_ids):,}")

    # Step 3: Load test source files
    print("\nLoading test_source1.tsv...")
    s1_df = pl.read_csv(TEST_DIR / "test_source1.tsv", separator="\t")
    s1_names = {}
    s1_addrs = {}
    for r in s1_df.iter_rows(named=True):
        s1_names[r["entity_id"]] = clean_name(r["business_name"])
        s1_addrs[r["entity_id"]] = clean_addr(r["business_address"])
    del s1_df

    print("Loading test_source2.tsv and test_source3.tsv for targets...")
    s2_df = pl.read_csv(TEST_DIR / "test_source2.tsv", separator="\t").filter(pl.col("entity_id").is_in(unique_target_ids))
    s3_df = pl.read_csv(TEST_DIR / "test_source3.tsv", separator="\t").filter(pl.col("entity_id").is_in(unique_target_ids))

    target_names = {}
    target_addrs = {}
    for r in s2_df.iter_rows(named=True):
        target_names[r["entity_id"]] = clean_name(r["business_name"])
        target_addrs[r["entity_id"]] = clean_addr(r["business_address"])
    del s2_df

    for r in s3_df.iter_rows(named=True):
        target_names[r["entity_id"]] = clean_name(r["business_name"])
        target_addrs[r["entity_id"]] = clean_addr(r["business_address"])
    del s3_df

    print(f"Loaded all text records in {time.time() - t0:.1f}s.")

    # Step 4: High-precision pair scoring
    print("\nScoring pairs with precision metrics...")
    scored_pairs = []
    
    # Pre-extract sets of address words for S1 to detect co-located businesses
    s1_addr_word_sets = {eid: set(addr.split()) for eid, addr in s1_addrs.items()}

    t_score = time.time()
    for s1_id, cid in tqdm(pairs, desc="Precision Scoring", mininterval=2.0):
        n1 = s1_names.get(s1_id, "")
        n2 = target_names.get(cid, "")
        if not n1 or not n2:
            continue

        # 1. Token sort and token set ratios
        sim_sort = fuzz.token_sort_ratio(n1, n2)
        sim_set = fuzz.token_set_ratio(n1, n2)
        sim_ratio = fuzz.ratio(n1, n2)

        # 2. Check if name is just co-located city/address tokens
        a1_words = s1_addr_word_sets.get(s1_id, set())
        n1_words = set(n1.split())
        n2_words = set(n2.split())
        
        # Distinctive tokens (not appearing in the address)
        dist1 = n1_words - a1_words
        dist2 = n2_words - a1_words

        if dist1 and dist2:
            dist_jaccard = len(dist1 & dist2) / len(dist1 | dist2)
        else:
            dist_jaccard = 1.0

        # Address similarity
        a1 = s1_addrs.get(s1_id, "")
        a2 = target_addrs.get(cid, "")
        addr_sim = fuzz.token_set_ratio(a1, a2) if (a1 and a2) else 50.0

        # High-precision filtering rules:
        # A pair is genuine if:
        # - High token sort ratio (>= 72) AND distinctive tokens don't conflict (dist_jaccard >= 0.40 or high ratio)
        # - OR exact token set match (>= 90) with high Levenshtein (>= 70)
        # - OR exact raw ratio (>= 85)
        # AND address cannot be completely contradictory (>= 40)
        is_valid = False
        if addr_sim >= 35:
            if sim_sort >= 72 and (dist_jaccard >= 0.30 or sim_ratio >= 75):
                is_valid = True
            elif sim_set >= 90 and sim_ratio >= 68:
                is_valid = True
            elif sim_ratio >= 82:
                is_valid = True

        if is_valid:
            # Composite quality score for ranking
            score = 0.50 * sim_sort + 0.25 * sim_set + 0.15 * sim_ratio + 0.10 * addr_sim
            scored_pairs.append((score, s1_id, cid))

    print(f"Scoring completed in {time.time() - t_score:.1f}s.")
    print(f"Pairs retained after precision filter: {len(scored_pairs):,} / {len(pairs):,} ({len(scored_pairs)/len(pairs)*100:.1f}%)")

    # Step 5: Global Greedy Conflict Resolution with Source Caps
    print("\nApplying Global Greedy Conflict Resolution (1-to-N matching & source caps)...")
    scored_pairs.sort(key=lambda x: x[0], reverse=True)

    assigned_targets = set()
    s2_counts = Counter()
    s3_counts = Counter()
    final_matches = defaultdict(list)

    # Maximum allowed matches per source per S1 entity
    MAX_S2 = 3
    MAX_S3 = 3

    for score, s1_id, cid in scored_pairs:
        # Enforce global uniqueness: target can only belong to ONE S1 entity
        if cid in assigned_targets:
            continue

        # Enforce source caps (matches Ground Truth distribution)
        if cid.startswith("S2-"):
            if s2_counts[s1_id] >= MAX_S2:
                continue
            s2_counts[s1_id] += 1
        else:
            if s3_counts[s1_id] >= MAX_S3:
                continue
            s3_counts[s1_id] += 1

        assigned_targets.add(cid)
        final_matches[s1_id].append(cid)

    # Step 6: Write new high-precision matching_results.tsv
    print(f"\nWriting precision-refined output to {MATCHING_FILE}...")
    with open(MATCHING_FILE, "w", encoding="utf-8") as f:
        f.write("source1_entity_id\tmatched_entity_ids\n")
        for s1_id in s1_ordered_ids:
            ms = final_matches.get(s1_id, [])
            f.write(f"{s1_id}\t{','.join(sorted(ms))}\n")

    # Summary Statistics
    match_lengths = [len(final_matches[s1_id]) for s1_id in s1_ordered_ids]
    arr = np.array(match_lengths)

    print("\n" + "=" * 65)
    print("REFINEMENT COMPLETE - NEW MATCH DISTRIBUTION:")
    print("=" * 65)
    print(f"Total S1 entities          : {len(arr):,}")
    print(f"Singletons (0 matches)     : {np.sum(arr == 0):,} ({np.mean(arr == 0)*100:.2f}%)")
    print(f"1 match                    : {np.sum(arr == 1):,} ({np.mean(arr == 1)*100:.2f}%)")
    print(f"2 matches                  : {np.sum(arr == 2):,} ({np.mean(arr == 2)*100:.2f}%)")
    print(f"3 matches                  : {np.sum(arr == 3):,} ({np.mean(arr == 3)*100:.2f}%)")
    print(f"4-6 matches                : {np.sum((arr >= 4) & (arr <= 6)):,} ({np.mean((arr >= 4) & (arr <= 6))*100:.2f}%)")
    print(f"> 6 matches                : {np.sum(arr > 6):,} ({np.mean(arr > 6)*100:.2f}%)")
    print(f"Mean matches per S1 entity : {np.mean(arr):.2f} (GT baseline: 3.46)")
    print(f"Total pairs in submission  : {np.sum(arr):,}")
    print(f"Total runtime              : {(time.time() - t0)/60:.2f} minutes")
    print("=" * 65)

if __name__ == "__main__":
    main()
