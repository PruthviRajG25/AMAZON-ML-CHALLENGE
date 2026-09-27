import sys
from collections import defaultdict
from typing import Dict, List, Set, Tuple, Optional
from .normalize import clean_name, canonical_name_key, token_signature_key, extract_address_keys

class CountryInvertedIndex:
    """In-memory multi-key inverted index for a single country to bound memory usage."""
    def __init__(self, max_key_freq: int = 500):
        self.max_key_freq = max_key_freq
        self.index: Dict[str, List[str]] = defaultdict(list)
        self.target_data: Dict[str, Tuple[str, str]] = {} # id -> (clean_name, norm_addr)

    def add_target_record(self, entity_id: str, name: str, address: str):
        c_name = canonical_name_key(name)
        t_sig = token_signature_key(name)
        addr_keys = extract_address_keys(address)
        
        self.target_data[entity_id] = (clean_name(name), address)
        
        # Add keys to inverted index
        if c_name:
            self.index[f"cn_{c_name}"].append(entity_id)
        if t_sig and t_sig != c_name:
            self.index[f"ts_{t_sig}"].append(entity_id)
        for ak in addr_keys:
            self.index[f"ak_{ak}"].append(entity_id)

    def prune_high_frequency_keys(self):
        """Remove overly broad keys that match too many records to prevent candidate blowup."""
        keys_to_delete = [k for k, v in self.index.items() if len(v) > self.max_key_freq]
        for k in keys_to_delete:
            del self.index[k]

    def query(self, name: str, address: str, max_candidates: int = 25) -> List[str]:
        """Query index using S1 name and address keys."""
        c_name = canonical_name_key(name)
        t_sig = token_signature_key(name)
        addr_keys = extract_address_keys(address)
        
        candidates: Set[str] = set()
        
        if c_name and f"cn_{c_name}" in self.index:
            candidates.update(self.index[f"cn_{c_name}"])
            
        if len(candidates) < max_candidates and t_sig and f"ts_{t_sig}" in self.index:
            candidates.update(self.index[f"ts_{t_sig}"])
            
        if len(candidates) < max_candidates:
            for ak in addr_keys:
                if f"ak_{ak}" in self.index:
                    candidates.update(self.index[f"ak_{ak}"])
                    if len(candidates) >= max_candidates * 2:
                        break
                        
        if len(candidates) > max_candidates:
            return list(candidates)[:max_candidates]
        return list(candidates)
