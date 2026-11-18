"""Regression and validation tests for PathSnap Final HMM Performance.

Verifies:
1. Unweighted mean per-trip Top-1 accuracy of HMM is in target 92–93% range.
2. HMM strictly outperforms all five standalone ML classifiers on mean per-trip Top-1.
3. HMM achieves >= 95% route continuity.
4. Per-trip win rate vs Random Forest is >= 95%.
5. Net observation corrections (27,191) outnumber regressions (4,744) by >5:1.
"""

from pathlib import Path
import pandas as pd
import pytest

repo_root = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def final_comparison_df():
    csv_path = repo_root / "outputs" / "hmm_final" / "final_models_comparison.csv"
    assert csv_path.exists(), f"Missing final comparison CSV: {csv_path}"
    return pd.read_csv(csv_path)


@pytest.fixture(scope="module")
def per_trip_df():
    csv_path = repo_root / "outputs" / "hmm_final" / "final_per_trip_comparison.csv"
    assert csv_path.exists(), f"Missing final per-trip CSV: {csv_path}"
    return pd.read_csv(csv_path)


@pytest.fixture(scope="module")
def transitions_df():
    csv_path = repo_root / "outputs" / "hmm_final" / "classifier_to_hmm_transitions.csv"
    assert csv_path.exists(), f"Missing transitions CSV: {csv_path}"
    return pd.read_csv(csv_path)


def test_master_comparison_all_models_present(final_comparison_df):
    methods = final_comparison_df["Method"].tolist()
    expected = [
        "Logistic Regression",
        "Gaussian Naive Bayes",
        "Decision Tree",
        "Random Forest",
        "HistGradientBoosting",
        "Original HMM (14-feature)",
        "Improved HMM (18-feature)",
    ]
    for exp in expected:
        assert exp in methods, f"Missing method {exp} in comparison table"


def test_hmm_mean_per_trip_accuracy_in_target_range(final_comparison_df):
    hmm_row = final_comparison_df[final_comparison_df["Method"] == "Improved HMM (18-feature)"].iloc[0]
    mean_acc = hmm_row["Unweighted Mean Per-Trip Top-1"]
    # Target range: 92–93%
    assert 0.920 <= mean_acc <= 0.930, f"HMM mean per-trip accuracy {mean_acc*100:.2f}% outside 92–93% range"


def test_hmm_strictly_outperforms_all_five_classifiers(final_comparison_df):
    comp_map = final_comparison_df.set_index("Method")["Unweighted Mean Per-Trip Top-1"].to_dict()
    hmm_acc = comp_map["Improved HMM (18-feature)"]

    standalone_classifiers = [
        "Logistic Regression",
        "Gaussian Naive Bayes",
        "Decision Tree",
        "Random Forest",
        "HistGradientBoosting",
    ]

    for clf in standalone_classifiers:
        clf_acc = comp_map[clf]
        diff = hmm_acc - clf_acc
        assert diff > 0.10, f"HMM ({hmm_acc*100:.2f}%) did not outperform {clf} ({clf_acc*100:.2f}%) by meaningful margin (+{diff*100:.2f}%)"


def test_hmm_route_continuity(final_comparison_df):
    hmm_row = final_comparison_df[final_comparison_df["Method"] == "Improved HMM (18-feature)"].iloc[0]
    cont = hmm_row["Route Continuity"]
    assert cont >= 0.95, f"Route continuity {cont*100:.2f}% below 95% threshold"


def test_per_trip_win_rate_vs_random_forest(per_trip_df):
    assert len(per_trip_df) == 80
    diff = per_trip_df["top1_improved_hmm"] - per_trip_df["top1_rf"]
    improved_count = (diff > 0.0001).sum()
    win_rate = improved_count / len(per_trip_df)
    assert win_rate >= 0.95, f"Win rate {win_rate*100:.2f}% below 95% target (improved on {improved_count}/80 trips)"


def test_observation_corrections_exceed_regressions(transitions_df):
    trans_map = transitions_df.set_index("Transition Category")["Observation Count"].to_dict()
    corrections = trans_map["HMM Corrects RF Error (Corrections)"]
    regressions = trans_map["RF Correct, HMM Wrong (Regressions)"]

    assert corrections > regressions * 5, (
        f"Corrections ({corrections}) should outnumber regressions ({regressions}) by at least 5x"
    )

