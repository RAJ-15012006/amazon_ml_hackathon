"""
Candidate generation (blocking) module for Business Entity Resolution.
Implements country-partitioned multi-key inverted indexing on distinctive tokens,
exact slugs, and address components with high recall and tight candidate density.
"""
from typing import Dict, List, Set, Tuple, Any, Iterator
import re
from collections import Counter, defaultdict
from unidecode import unidecode
from rapidfuzz import fuzz
try:
    from .config import MAX_DOC_FREQ, TOP_K_TOKENS, LEGAL_SUFFIXES, COMMON_ADDR_WORDS
except (ImportError, ValueError):
    from src.config import MAX_DOC_FREQ, TOP_K_TOKENS, LEGAL_SUFFIXES, COMMON_ADDR_WORDS

URL_PATTERN = re.compile(
    r"(https?://|www\.|\.com|\.org|\.net|\.c0m|\.co\.in|\.co|\.in|\.fr|\.io|@)",
    re.IGNORECASE
)
NON_ALPHANUM = re.compile(r"[^a-z0-9\s]")
TWO_PLUS_DIGITS = re.compile(r"\b\d{2,}\b")

# Combined domain stopwords for blocking keys
BLOCKING_STOPWORDS = LEGAL_SUFFIXES | COMMON_ADDR_WORDS | {
    "association", "societe", "centre", "center", "club", "ecole", "institut",
    "comite", "groupe", "group", "service", "services", "france", "paris",
    "limited", "private", "enterprise", "enterprises", "solutions"
}


def parse_for_blocking(name: str, address: str) -> Tuple[str, str, Set[str], Set[str], str, Set[str], Set[str]]:
    """
    Parse name and address into representations optimized for inverted indexing and filtering.
    Returns: (raw_name, raw_addr, name_tokens, addr_tokens, name_slug, nums, combined_tokens)
    """
    n_str = name if name else ""
    a_str = address if address else ""
    
    clean_n = URL_PATTERN.sub(" ", unidecode(n_str).lower())
    clean_a = URL_PATTERN.sub(" ", unidecode(a_str).lower())
    
    n_tokens = {
        w for w in NON_ALPHANUM.sub(" ", clean_n).split()
        if len(w) >= 3 and w not in BLOCKING_STOPWORDS
    }
    a_tokens = {
        w for w in NON_ALPHANUM.sub(" ", clean_a).split()
        if len(w) >= 3 and w not in BLOCKING_STOPWORDS
    }
    
    n_slug = re.sub(r"[^a-z0-9]", "", clean_n)
    nums = set(TWO_PLUS_DIGITS.findall(clean_a))
    combined_tokens = n_tokens | a_tokens
    
    return clean_n, clean_a, n_tokens, a_tokens, n_slug, nums, combined_tokens


class CountryBlocker:
    """
    Inverted-index blocker scoped to a single country to guarantee zero cross-country noise
    and achieve maximum throughput and minimal memory overhead.
    """
    def __init__(self, country: str):
        self.country = country
        self.target_records: Dict[str, Dict[str, Any]] = {}
        self.doc_freqs: Counter = Counter()
        self.inverted_index: Dict[Any, List[str]] = defaultdict(list)
        
    def add_target_pool(self, entity_ids: List[str], names: List[str], addresses: List[str]):
        """Add Source 2 or Source 3 records to the candidate target pool."""
        for eid, name, addr in zip(entity_ids, names, addresses):
            rn, ra, nt, at, ns, nums, all_t = parse_for_blocking(name, addr)
            self.doc_freqs.update(all_t)
            self.target_records[eid] = {
                "raw_name": rn,
                "raw_address": ra,
                "name_tokens": nt,
                "address_tokens": at,
                "name_slug": ns,
                "nums": nums,
                "combined_tokens": all_t,
                "has_address": bool(ra)
            }
            
    def build_index(self):
        """Index target records by their rarest distinctive tokens and slugs."""
        for eid, rec in self.target_records.items():
            all_t = rec["combined_tokens"]
            # Pick top rarest distinctive tokens
            sorted_tokens = sorted(all_t, key=lambda t: self.doc_freqs[t])
            for t in sorted_tokens[:TOP_K_TOKENS]:
                if self.doc_freqs[t] <= MAX_DOC_FREQ:
                    self.inverted_index[t].append(eid)
                    
            # Index name slug if long enough
            slug = rec["name_slug"]
            if len(slug) >= 5:
                self.inverted_index[("slug", slug)].append(eid)
                
    def get_candidates_for_s1(
        self,
        s1_name: str,
        s1_address: str
    ) -> Tuple[Dict[str, Any], List[str]]:
        """
        Query inverted index for an S1 entity and return:
        (s1_parsed_record, list_of_filtered_candidate_ids)
        """
        rn1, ra1, nt1, at1, ns1, nums1, all_t1 = parse_for_blocking(s1_name, s1_address)
        s1_rec = {
            "raw_name": rn1,
            "raw_address": ra1,
            "name_tokens": nt1,
            "address_tokens": at1,
            "name_slug": ns1,
            "nums": nums1,
            "combined_tokens": all_t1,
            "has_address": bool(ra1)
        }
        
        raw_candidates: Set[str] = set()
        sorted_tokens = sorted(all_t1, key=lambda t: self.doc_freqs.get(t, 0))
        for t in sorted_tokens[:TOP_K_TOKENS]:
            if t in self.inverted_index:
                raw_candidates.update(self.inverted_index[t])
                
        if len(ns1) >= 5:
            slug_key = ("slug", ns1)
            if slug_key in self.inverted_index:
                raw_candidates.update(self.inverted_index[slug_key])
                
        # Candidate refinement filter
        filtered_candidates: List[str] = []
        for cid in raw_candidates:
            rec2 = self.target_records[cid]
            nt2 = rec2["name_tokens"]
            at2 = rec2["address_tokens"]
            ns2 = rec2["name_slug"]
            nums2 = rec2["nums"]
            rn2 = rec2["raw_name"]
            
            has_name_overlap = bool(nt1 & nt2)
            has_addr_overlap = bool(at1 & at2)
            has_num_overlap = bool(nums1 and nums2 and (nums1 & nums2))
            
            # 1. Exact or sub-slug match
            if ns1 and ns2 and (ns1 in ns2 or ns2 in ns1):
                filtered_candidates.append(cid)
            # 2. Name token overlap AND address token overlap
            elif has_name_overlap and has_addr_overlap:
                filtered_candidates.append(cid)
            # 3. Name token overlap AND address numeric match
            elif has_name_overlap and has_num_overlap:
                filtered_candidates.append(cid)
            # 4. Target record has no address: name token match with high Levenshtein ratio
            elif not rec2["has_address"] and has_name_overlap and fuzz.ratio(rn1, rn2) >= 65:
                filtered_candidates.append(cid)
            # 5. High slug similarity
            elif ns1 and ns2 and fuzz.ratio(ns1, ns2) >= 75:
                filtered_candidates.append(cid)
            # 6. Address match with moderate name similarity
            elif has_addr_overlap and has_num_overlap and ns1 and ns2 and fuzz.ratio(ns1, ns2) >= 50:
                filtered_candidates.append(cid)
                
        return s1_rec, filtered_candidates
