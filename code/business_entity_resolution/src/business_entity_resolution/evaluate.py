from typing import Dict, Set, List, Tuple

def compute_entity_f05(gt_ids: Set[str], pred_ids: Set[str]) -> Tuple[float, float, float]:
    """Compute (f05, precision, recall) for a single Source 1 entity."""
    # Singleton cases
    if not gt_ids and not pred_ids:
        return 1.0, 1.0, 1.0
    if not gt_ids and pred_ids:
        return 0.0, 0.0, 1.0
    if gt_ids and not pred_ids:
        return 0.0, 1.0, 0.0
        
    true_positives = len(gt_ids & pred_ids)
    if true_positives == 0:
        return 0.0, 0.0, 0.0
        
    precision = true_positives / len(pred_ids)
    recall = true_positives / len(gt_ids)
    
    denom = 0.25 * precision + recall
    if denom == 0:
        f05 = 0.0
    else:
        f05 = (1.25 * precision * recall) / denom
        
    return f05, precision, recall

def evaluate_macro_f05(
    ground_truth: Dict[str, Set[str]],
    predictions: Dict[str, Set[str]]
) -> Dict[str, float]:
    """Compute official competition Macro F0.5 across all Source 1 entities."""
    total_entities = len(ground_truth)
    if total_entities == 0:
        return {"macro_f05": 0.0, "macro_precision": 0.0, "macro_recall": 0.0, "singleton_accuracy": 0.0}
        
    total_f05 = 0.0
    total_p = 0.0
    total_r = 0.0
    
    singleton_count = 0
    singleton_correct = 0
    
    for s1_id, gt_set in ground_truth.items():
        pred_set = predictions.get(s1_id, set())
        f05, p, r = compute_entity_f05(gt_set, pred_set)
        
        total_f05 += f05
        total_p += p
        total_r += r
        
        if not gt_set:
            singleton_count += 1
            if not pred_set:
                singleton_correct += 1
                
    singleton_acc = (singleton_correct / singleton_count) if singleton_count > 0 else 1.0
    
    return {
        "macro_f05": total_f05 / total_entities,
        "macro_precision": total_p / total_entities,
        "macro_recall": total_r / total_entities,
        "singleton_accuracy": singleton_acc,
        "total_evaluated": total_entities
    }
