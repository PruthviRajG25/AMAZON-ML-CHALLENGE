import os
from dataclasses import dataclass

@dataclass
class PathConfig:
    # Default dataset directories
    train_dir: str = os.path.join("student_resource", "dataset", "train")
    test_dir: str = os.path.join("student_resource", "dataset", "test")
    output_dir: str = "output"
    artifact_dir: str = os.path.join("code", "business_entity_resolution", "artifacts")
    
    # Train files
    train_source1: str = os.path.join(train_dir, "train_source1.tsv")
    train_source2: str = os.path.join(train_dir, "train_source2.tsv")
    train_source3: str = os.path.join(train_dir, "train_source3.tsv")
    train_ground_truth: str = os.path.join(train_dir, "train_ground_truth.tsv")
    
    # Test files
    test_source1: str = os.path.join(test_dir, "test_source1.tsv")
    test_source2: str = os.path.join(test_dir, "test_source2.tsv")
    test_source3: str = os.path.join(test_dir, "test_source3.tsv")
    
    # Outputs
    matching_output: str = os.path.join(output_dir, "matching_results.tsv")
    candidate_output: str = os.path.join(output_dir, "candidate_pairs.tsv")

@dataclass
class BlockingConfig:
    max_candidates_per_entity: int = 25
    min_token_len: int = 3
    # Candidate generation keys
    use_canonical_name: bool = True
    use_token_signature: bool = True
    use_address_key: bool = True

@dataclass
class ModelConfig:
    threshold: float = 0.70
    random_state: int = 42
    n_estimators: int = 150
    learning_rate: float = 0.05
    num_leaves: int = 31
    class_weight: str = "balanced"
