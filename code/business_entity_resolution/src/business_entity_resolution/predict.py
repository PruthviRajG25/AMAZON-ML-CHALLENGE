import os
import sys
from collections import defaultdict
from typing import Dict, List, Set, Tuple, Optional
import numpy as np
from .config import PathConfig, BlockingConfig, ModelConfig
from .normalize import clean_name, canonical_name_key, token_signature_key, extract_address_keys
from .features import extract_pairwise_features
from .model import MatchClassifier

def run_test_inference(
    artifact_path: Optional[str] = None,
    output_dir: Optional[str] = None,
    threshold: float = 0.50,
    max_test_s1: Optional[int] = None
):
    """Run ultra-fast, memory-efficient single-pass streaming test inference."""
    paths = PathConfig()
    b_cfg = BlockingConfig()
    
    if artifact_path is None:
        artifact_path = os.path.join(paths.artifact_dir, "match_model.joblib")
    if output_dir is None:
        output_dir = paths.output_dir
        
    os.makedirs(output_dir, exist_ok=True)
    matching_file = os.path.join(output_dir, "matching_results.tsv")
    candidate_file = os.path.join(output_dir, "candidate_pairs.tsv")
    
    print(f"=== Starting Memory-Efficient Test Inference (Output: {output_dir}) ===")
    
    # 1. Load trained classifier
    clf = MatchClassifier()
    model_loaded = False
    if os.path.isfile(artifact_path):
        clf.load(artifact_path)
        model_loaded = True
        print(f"Loaded classifier from {artifact_path}")
    else:
        print(f"WARNING: Model artifact not found at {artifact_path}. Falling back to rule-based scoring.")
        
    # 2. Build Inverted Index from S1 Reference Entities
    print("Loading test_source1.tsv and indexing reference keys...")
    s1_all_order: List[str] = []
    s1_data: Dict[str, Tuple[str, str, str]] = {}
    # country -> key -> list of s1_ids
    s1_key_index: Dict[str, Dict[str, List[str]]] = defaultdict(lambda: defaultdict(list))
    
    with open(paths.test_source1, "r", encoding="utf-8", errors="replace") as f:
        f.readline()
        for line in f:
            p = line.rstrip("\r\n").split("\t")
            s1_id = p[0]
            name = p[1] if len(p) > 1 else ""
            addr = p[2] if len(p) > 2 else ""
            country = p[3] if len(p) > 3 else ""
            
            s1_all_order.append(s1_id)
            s1_data[s1_id] = (name, addr, country)
            
            c_name = canonical_name_key(name)
            t_sig = token_signature_key(name)
            addr_keys = extract_address_keys(addr)
            
            c_idx = s1_key_index[country]
            if c_name:
                c_idx[f"cn_{c_name}"].append(s1_id)
            if t_sig and t_sig != c_name:
                c_idx[f"ts_{t_sig}"].append(s1_id)
            for ak in addr_keys:
                c_idx[f"ak_{ak}"].append(s1_id)
                
            if max_test_s1 and len(s1_all_order) >= max_test_s1:
                break
                
    print(f"Loaded and indexed {len(s1_all_order)} S1 entities across {len(s1_key_index)} countries.")
    
    # Track candidate IDs and matched IDs per S1
    candidates_map: Dict[str, List[str]] = defaultdict(list)
    candidates_set: Dict[str, Set[str]] = defaultdict(set)
    matches_map: Dict[str, List[str]] = defaultdict(list)
    
    # 3. Stream through test_source2.tsv and test_source3.tsv
    for src_path in [paths.test_source2, paths.test_source3]:
        src_name = os.path.basename(src_path)
        print(f"Streaming {src_name} and matching candidates on-the-fly...")
        line_count = 0
        matches_found = 0
        
        with open(src_path, "r", encoding="utf-8", errors="replace") as f:
            f.readline()
            for line in f:
                line_count += 1
                p = line.rstrip("\r\n").split("\t")
                if len(p) < 4:
                    continue
                tid, tname, taddr, tcountry = p[0], p[1], p[2], p[3]
                
                # Check if country exists in S1
                if tcountry not in s1_key_index:
                    continue
                    
                c_idx = s1_key_index[tcountry]
                c_name = canonical_name_key(tname)
                t_sig = token_signature_key(tname)
                addr_keys = extract_address_keys(taddr)
                
                # Retrieve matching S1 IDs
                matched_s1_ids: Set[str] = set()
                if c_name and f"cn_{c_name}" in c_idx:
                    matched_s1_ids.update(c_idx[f"cn_{c_name}"])
                if t_sig and f"ts_{t_sig}" in c_idx:
                    matched_s1_ids.update(c_idx[f"ts_{t_sig}"])
                for ak in addr_keys:
                    if f"ak_{ak}" in c_idx:
                        matched_s1_ids.update(c_idx[f"ak_{ak}"])
                        
                if not matched_s1_ids:
                    continue
                    
                # For each matched S1 entity, record candidate and score
                for s1_id in matched_s1_ids:
                    c_set = candidates_set[s1_id]
                    if tid in c_set or len(c_set) >= b_cfg.max_candidates_per_entity:
                        continue
                    c_set.add(tid)
                    candidates_map[s1_id].append(tid)
                    
                    # Score candidate
                    s1_name, s1_addr, s1_country = s1_data[s1_id]
                    feat = extract_pairwise_features(s1_name, s1_addr, s1_country, tid, tname, taddr, tcountry)
                    
                    if model_loaded:
                        prob = clf.predict_proba(np.array([feat]))[0]
                        if prob >= threshold:
                            matches_map[s1_id].append(tid)
                            matches_found += 1
                    else:
                        # Fallback
                        if s1_name.lower() == tname.lower() and s1_name:
                            matches_map[s1_id].append(tid)
                            matches_found += 1
                            
        print(f"  Finished {src_name} ({line_count} rows scanned). Matches linked so far: {sum(len(v) for v in matches_map.values())}")
        
    # 4. Write output files in exact order of test_source1.tsv
    print("\nWriting final submission TSV files...")
    with open(matching_file, "w", encoding="utf-8") as fm, open(candidate_file, "w", encoding="utf-8") as fc:
        fm.write("source1_entity_id\tmatched_entity_ids\n")
        fc.write("source1_entity_id\tcandidate_entity_ids\n")
        
        for s1_id in s1_all_order:
            m_list = matches_map.get(s1_id, [])
            c_list = candidates_map.get(s1_id, [])
            
            fm.write(f"{s1_id}\t{','.join(m_list)}\n")
            fc.write(f"{s1_id}\t{','.join(c_list)}\n")
            
    print(f"Output files successfully written:\n  1. {matching_file}\n  2. {candidate_file}")

if __name__ == "__main__":
    run_test_inference()
