import os
import joblib
from typing import Optional
import numpy as np
from lightgbm import LGBMClassifier
from .config import ModelConfig
from .features import FEATURE_NAMES

class MatchClassifier:
    """LightGBM pairwise binary classification model."""
    def __init__(self, config: Optional[ModelConfig] = None):
        self.config = config or ModelConfig()
        self.model = LGBMClassifier(
            n_estimators=self.config.n_estimators,
            learning_rate=self.config.learning_rate,
            num_leaves=self.config.num_leaves,
            class_weight=self.config.class_weight,
            random_state=self.config.random_state,
            n_jobs=-1,
            verbose=-1
        )
        self.feature_names = FEATURE_NAMES

    def fit(self, X: np.ndarray, y: np.ndarray):
        """Fit model on feature matrix and binary labels."""
        self.model.fit(X, y)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return probability of match (label=1)."""
        if len(X) == 0:
            return np.array([])
        return self.model.predict_proba(X)[:, 1]

    def save(self, model_path: str):
        """Persist model to disk."""
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        joblib.dump(self.model, model_path)

    def load(self, model_path: str):
        """Load persisted model from disk."""
        self.model = joblib.load(model_path)
