import pytest
from business_entity_resolution.evaluate import compute_entity_f05, evaluate_macro_f05

def test_singleton_cases():
    # Both empty -> 1.0
    f05, p, r = compute_entity_f05(set(), set())
    assert f05 == 1.0
    
    # False positive on singleton -> 0.0
    f05, p, r = compute_entity_f05(set(), {"S2-123"})
    assert f05 == 0.0
    
    # Missed match -> 0.0
    f05, p, r = compute_entity_f05({"S2-123"}, set())
    assert f05 == 0.0

def test_f05_calculation_example_from_spec():
    # Example in challenge README:
    # Ground truth: S2-00047, S3-00812
    # Predicted: S2-00047, S2-00193, S3-00812
    # Precision = 2/3, Recall = 1.0 -> F0.5 = 0.714
    gt = {"S2-00047", "S3-00812"}
    pred = {"S2-00047", "S2-00193", "S3-00812"}
    f05, p, r = compute_entity_f05(gt, pred)
    assert abs(p - (2.0 / 3.0)) < 1e-3
    assert abs(r - 1.0) < 1e-3
    assert abs(f05 - 0.714) < 1e-3

def test_macro_evaluation():
    gt = {
        "S1-1": {"S2-1", "S3-1"},
        "S1-2": set(), # singleton
    }
    pred = {
        "S1-1": {"S2-1", "S3-1"}, # score 1.0
        "S1-2": set(), # score 1.0
    }
    res = evaluate_macro_f05(gt, pred)
    assert res["macro_f05"] == 1.0
    assert res["singleton_accuracy"] == 1.0
