"""
Feature extraction module for Entity Resolution candidate pairs.
Computes string similarity, token overlap, numeric matching, and domain indicators.
"""
from typing import Dict, Any, List
import numpy as np
from rapidfuzz import fuzz

FEATURE_NAMES = [
    "name_levenshtein_ratio",
    "name_partial_ratio",
    "name_token_sort_ratio",
    "name_token_set_ratio",
    "name_slug_ratio",
    "name_slug_containment",
    "name_token_jaccard",
    "name_token_overlap_count",
    "name_len_diff",
    "addr_token_set_ratio",
    "addr_token_jaccard",
    "addr_token_overlap_count",
    "num_jaccard",
    "num_overlap_count",
    "addr_missing",
    "is_source_2",
    "country_us",
    "country_india",
    "country_france"
]


def compute_pair_features(
    r1: Dict[str, Any],
    r2: Dict[str, Any],
    cid: str,
    country: str
) -> List[float]:
    """Compute 19-dimensional feature vector for an (S1, S2/S3) pair."""
    # Name features
    n1, n2 = r1["raw_name"], r2["raw_name"]
    s1, s2 = r1["name_slug"], r2["name_slug"]
    
    f_name_lev = fuzz.ratio(n1, n2) / 100.0
    f_name_part = fuzz.partial_ratio(n1, n2) / 100.0
    f_token_sort = fuzz.token_sort_ratio(n1, n2) / 100.0
    f_token_set = fuzz.token_set_ratio(n1, n2) / 100.0
    
    f_slug_ratio = fuzz.ratio(s1, s2) / 100.0 if (s1 and s2) else 0.0
    f_slug_in = 1.0 if (s1 and s2 and (s1 in s2 or s2 in s1)) else 0.0
    
    n_inter = len(r1["name_tokens"] & r2["name_tokens"])
    n_union = len(r1["name_tokens"] | r2["name_tokens"]) or 1
    f_name_jaccard = n_inter / n_union
    f_len_diff = abs(len(n1) - len(n2))
    
    # Address features
    a1, a2 = r1["raw_address"], r2["raw_address"]
    if a1 and a2:
        f_addr_set = fuzz.token_set_ratio(a1, a2) / 100.0
        a_inter = len(r1["address_tokens"] & r2["address_tokens"])
        a_union = len(r1["address_tokens"] | r2["address_tokens"]) or 1
        f_addr_jaccard = a_inter / a_union
    else:
        f_addr_set = 0.0
        a_inter = 0
        f_addr_jaccard = 0.0
        
    # Numeric features
    nums1, nums2 = r1["nums"], r2["nums"]
    num_inter = len(nums1 & nums2)
    num_union = len(nums1 | nums2)
    if num_union > 0:
        f_num_jaccard = num_inter / num_union
    else:
        f_num_jaccard = 0.5 if (not nums1 and not nums2) else 0.0
        
    # Domain & meta features
    f_addr_missing = 1.0 if not r2["has_address"] else 0.0
    f_is_s2 = 1.0 if cid.startswith("S2-") else 0.0
    f_c_us = 1.0 if country == "US" else 0.0
    f_c_in = 1.0 if country == "India" else 0.0
    f_c_fr = 1.0 if country == "France" else 0.0
    
    return [
        f_name_lev,
        f_name_part,
        f_token_sort,
        f_token_set,
        f_slug_ratio,
        f_slug_in,
        f_name_jaccard,
        float(n_inter),
        float(f_len_diff),
        f_addr_set,
        f_addr_jaccard,
        float(a_inter),
        f_num_jaccard,
        float(num_inter),
        f_addr_missing,
        f_is_s2,
        f_c_us,
        f_c_in,
        f_c_fr
    ]
