import re
from typing import List, Dict, Tuple, Optional
from rapidfuzz import fuzz
from .normalize import clean_name, canonical_name_key, normalize_address

FEATURE_NAMES = [
    "name_ratio",
    "name_token_sort_ratio",
    "name_token_set_ratio",
    "name_exact_clean",
    "name_canonical_exact",
    "name_len_ratio",
    "addr_ratio",
    "addr_token_sort_ratio",
    "addr_token_set_ratio",
    "addr_missing",
    "addr_shared_numbers",
    "source_is_s2",
    "country_is_in",
    "country_is_us",
    "country_is_fr"
]

NUMBER_FINDER = re.compile(r"\b\d+\b")

def extract_pairwise_features(
    s1_name: str,
    s1_addr: str,
    s1_country: str,
    tgt_id: str,
    tgt_name: str,
    tgt_addr: str,
    tgt_country: str
) -> List[float]:
    """Extract fast numeric similarity features for a candidate pair."""
    # Normalized strings
    s1_n_clean = clean_name(s1_name)
    tgt_n_clean = clean_name(tgt_name)
    s1_n_canon = canonical_name_key(s1_name)
    tgt_n_canon = canonical_name_key(tgt_name)
    
    s1_a_norm = normalize_address(s1_addr)
    tgt_a_norm = normalize_address(tgt_addr)
    
    # 1. Name features
    n_ratio = fuzz.ratio(s1_n_clean, tgt_n_clean) / 100.0
    n_tsort = fuzz.token_sort_ratio(s1_n_clean, tgt_n_clean) / 100.0
    n_tset = fuzz.token_set_ratio(s1_n_clean, tgt_n_clean) / 100.0
    n_exact = 1.0 if s1_n_clean and s1_n_clean == tgt_n_clean else 0.0
    n_canon = 1.0 if s1_n_canon and s1_n_canon == tgt_n_canon else 0.0
    
    max_len = max(len(s1_n_clean), len(tgt_n_clean), 1)
    min_len = min(len(s1_n_clean), len(tgt_n_clean))
    n_len_ratio = min_len / max_len
    
    # 2. Address features
    addr_missing = 1.0 if not tgt_a_norm else 0.0
    if not addr_missing:
        a_ratio = fuzz.ratio(s1_a_norm, tgt_a_norm) / 100.0
        a_tsort = fuzz.token_sort_ratio(s1_a_norm, tgt_a_norm) / 100.0
        a_tset = fuzz.token_set_ratio(s1_a_norm, tgt_a_norm) / 100.0
        
        s1_nums = set(NUMBER_FINDER.findall(s1_a_norm))
        tgt_nums = set(NUMBER_FINDER.findall(tgt_a_norm))
        shared_nums = 1.0 if (s1_nums and tgt_nums and (s1_nums & tgt_nums)) else 0.0
    else:
        a_ratio = 0.0
        a_tsort = 0.0
        a_tset = 0.0
        shared_nums = 0.0
        
    # 3. Source & country metadata
    is_s2 = 1.0 if tgt_id.startswith("S2-") else 0.0
    c_in = 1.0 if s1_country == "India" else 0.0
    c_us = 1.0 if s1_country == "US" else 0.0
    c_fr = 1.0 if s1_country == "France" else 0.0
    
    return [
        n_ratio,
        n_tsort,
        n_tset,
        n_exact,
        n_canon,
        n_len_ratio,
        a_ratio,
        a_tsort,
        a_tset,
        addr_missing,
        shared_nums,
        is_s2,
        c_in,
        c_us,
        c_fr
    ]
