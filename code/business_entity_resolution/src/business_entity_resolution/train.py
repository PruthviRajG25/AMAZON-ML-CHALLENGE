import os
import sys
import numpy as np
from collections import defaultdict
from typing import Dict, Set, Tuple, List, Optional
from .config import PathConfig, ModelConfig, BlockingConfig
from .blocking import CountryInvertedIndex
from .features import extract_pairwise_features
from .model import MatchClassifier
from .evaluate import evaluate_macro_f05

def train_pipeline(
    sample_size: int = 25000,
    val_size: int = 5000,
    artifact_path: Optional[str] = None
) -> float:
    """Train LightGBM matching model and optimize decision threshold for Macro F0.5."""
    paths = PathConfig()
    m_cfg = ModelConfig()
    b_cfg = BlockingConfig()
    
    if artifact_path is None:
        artifact_path = os.path.join(paths.artifact_dir, "match_model.joblib")
        
    print(f"=== Starting Training Pipeline (Sample: {sample_size}, Val: {val_size}) ===")
    
    # 1. Load S1 training records
    s1_train = {} # id -> (name, addr, country)
    with open(paths.train_source1, "r", encoding="utf-8", errors="replace") as f:
        f.readline()
        for line in f:
            p = line.rstrip("\r\n").split("\t")
            s1_train[p[0]] = (p[1], p[2], p[3])
            if len(s1_train) >= sample_size + val_size:
                break
                
    s1_all_ids = list(s1_train.keys())
    train_ids = set(s1_all_ids[:sample_size])
    val_ids = set(s1_all_ids[sample_size:sample_size + val_size])
    
    # 2. Load ground truth for these entities
    gt_all: Dict[str, Set[str]] = defaultdict(set)
    targets_needed: Set[str] = set()
    with open(paths.train_ground_truth, "r", encoding="utf-8", errors="replace") as f:
        f.readline()
        for line in f:
            p = line.rstrip("\r\n").split("\t")
            if p[0] in s1_train:
                if len(p) > 1 and p[1]:
                    mids = set(p[1].split(","))
                    gt_all[p[0]] = mids
                    targets_needed.update(mids)
                else:
                    gt_all[p[0]] = set()
                    
    print(f"Loaded {len(s1_train)} S1 entities. Targets needed: {len(targets_needed)}")
    
    # 3. Read target records needed (S2 + S3)
    target_data = {} # id -> (name, addr, country)
    for sf in [paths.train_source2, paths.train_source3]:
        with open(sf, "r", encoding="utf-8", errors="replace") as f:
            f.readline()
            for line in f:
                p = line.rstrip("\r\n").split("\t")
                if p[0] in targets_needed:
                    target_data[p[0]] = (p[1] if len(p) > 1 else "", p[2] if len(p) > 2 else "", p[3] if len(p) > 3 else "")
                    
    # 4. Build country indices
    indices: Dict[str, CountryInvertedIndex] = defaultdict(CountryInvertedIndex)
    for tid, (tname, taddr, tc) in target_data.items():
        indices[tc].add_target_record(tid, tname, taddr)
    for idx in indices.values():
        idx.prune_high_frequency_keys()
        
    print("Generating candidate pairs and training features...")
    X_train_list = []
    y_train_list = []
    
    for s1_id in train_ids:
        s1_name, s1_addr, s1_c = s1_train[s1_id]
        true_mids = gt_all[s1_id]
        
        # Blocking candidates
        cands = indices[s1_c].query(s1_name, s1_addr, max_candidates=b_cfg.max_candidates_per_entity)
        
        # Include true matches to ensure positive representation
        all_candidates = set(cands) | (true_mids & set(target_data.keys()))
        
        for cid in all_candidates:
            if cid not in target_data:
                continue
            tname, taddr, tc = target_data[cid]
            feat = extract_pairwise_features(s1_name, s1_addr, s1_c, cid, tname, taddr, tc)
            label = 1 if cid in true_mids else 0
            X_train_list.append(feat)
            y_train_list.append(label)
            
    X_train = np.array(X_train_list)
    y_train = np.array(y_train_list)
    print(f"Constructed {len(X_train)} training pairs (Positives: {np.sum(y_train)}, Negatives: {len(y_train) - np.sum(y_train)})")
    
    # 5. Fit model
    clf = MatchClassifier(m_cfg)
    clf.fit(X_train, y_train)
    clf.save(artifact_path)
    print(f"Model saved to {artifact_path}")
    
    # 6. Evaluate on Validation Set & Tune Threshold
    print("Tuning threshold on validation set...")
    val_gt = {sid: gt_all[sid] for sid in val_ids}
    
    val_pairs = []
    val_meta = []
    for s1_id in val_ids:
        s1_name, s1_addr, s1_c = s1_train[s1_id]
        cands = indices[s1_c].query(s1_name, s1_addr, max_candidates=b_cfg.max_candidates_per_entity)
        for cid in cands:
            if cid in target_data:
                tname, taddr, tc = target_data[cid]
                feat = extract_pairwise_features(s1_name, s1_addr, s1_c, cid, tname, taddr, tc)
                val_pairs.append(feat)
                val_meta.append((s1_id, cid))
                
    if val_pairs:
        X_val = np.array(val_pairs)
        scores = clf.predict_proba(X_val)
    else:
        scores = np.array([])
        
    best_thresh = 0.70
    best_f05 = 0.0
    
    for thresh in np.arange(0.50, 0.90, 0.05):
        val_preds: Dict[str, Set[str]] = defaultdict(set)
        for i, prob in enumerate(scores):
            if prob >= thresh:
                s1_id, cid = val_meta[i]
                val_preds[s1_id].add(cid)
        metrics = evaluate_macro_f05(val_gt, val_preds)
        print(f"  Threshold {thresh:.2f} -> Macro F0.5: {metrics['macro_f05']:.4f} (P: {metrics['macro_precision']:.4f}, R: {metrics['macro_recall']:.4f}, Singletons: {metrics['singleton_accuracy']:.4f})")
        if metrics['macro_f05'] > best_f05:
            best_f05 = metrics['macro_f05']
            best_thresh = thresh
            
    print(f"Selected Optimal Threshold: {best_thresh:.2f} (Macro F0.5: {best_f05:.4f})")
    return best_f05

if __name__ == "__main__":
    train_pipeline()
