"""Initial probability-producing candidate classifier definitions."""

from __future__ import annotations

from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def candidate_models(seed: int = 42) -> dict[str, Pipeline]:
    return {
        "logistic_regression": Pipeline([
            ("imputer", SimpleImputer(strategy="median")), ("scale", StandardScaler()),
            ("model", LogisticRegression(max_iter=400, class_weight="balanced", random_state=seed)),
        ]),
        "random_forest": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", RandomForestClassifier(n_estimators=120, max_depth=20, min_samples_leaf=3, class_weight="balanced_subsample", n_jobs=-1, random_state=seed)),
        ]),
        "extra_trees": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", ExtraTreesClassifier(n_estimators=120, max_depth=24, min_samples_leaf=2, class_weight="balanced", n_jobs=-1, random_state=seed)),
        ]),
        "hist_gradient_boosting": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", HistGradientBoostingClassifier(max_iter=160, learning_rate=0.08, max_leaf_nodes=31, l2_regularization=1.0, random_state=seed)),
        ]),
        "mlp": Pipeline([
            ("imputer", SimpleImputer(strategy="median")), ("scale", StandardScaler()),
            ("model", MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=80, early_stopping=True, validation_fraction=0.15, random_state=seed)),
        ]),
    }
