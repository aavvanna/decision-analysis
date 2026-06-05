"""Tests for analysis.experiments."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import pytest
from experiments import (
    run_trial,
    run_balanced_market_experiment,
    run_imbalanced_experiment,
    correlation_score,
)


def _make_synthetic_population(n_a: int = 60, n_e: int = 60, dim: int = 10, seed: int = 1):
    rng = np.random.default_rng(seed)
    A = (rng.random((n_a, dim)) > 0.5).astype(np.float32)
    E = (rng.random((n_e, dim)) > 0.5).astype(np.float32)
    return A, E


def test_run_trial_returns_required_metrics():
    A, E = _make_synthetic_population()
    metrics = run_trial(A, E, sample_size_a=10, sample_size_e=10, seed=42)
    required = {
        "avg_applicant_rank_a_opt",
        "avg_employer_rank_a_opt",
        "avg_applicant_rank_e_opt",
        "avg_employer_rank_e_opt",
        "pairs_changed",
        "blocking_a_opt",
        "blocking_e_opt",
        "correlation_score",
    }
    assert required.issubset(metrics.keys())


def test_run_trial_always_returns_stable_matchings():
    A, E = _make_synthetic_population()
    for seed in range(5):
        metrics = run_trial(A, E, sample_size_a=12, sample_size_e=12, seed=seed)
        assert metrics["blocking_a_opt"] == 0
        assert metrics["blocking_e_opt"] == 0


def test_run_trial_demonstrates_proposer_advantage():
    A, E = _make_synthetic_population(n_a=80, n_e=80, dim=12, seed=7)
    advantages = []
    for seed in range(20):
        m = run_trial(A, E, sample_size_a=15, sample_size_e=15, seed=seed)
        advantages.append(m["avg_applicant_rank_e_opt"] - m["avg_applicant_rank_a_opt"])
    # On average across many trials, applicants do better when they propose.
    assert np.mean(advantages) >= 0


def test_run_balanced_market_experiment_aggregates_trials():
    A, E = _make_synthetic_population(n_a=50, n_e=50)
    df = run_balanced_market_experiment(A, E, sizes=[8, 12], n_trials=5, base_seed=0)
    assert isinstance(df, pd.DataFrame)
    # 2 sizes * 5 trials
    assert len(df) == 10
    assert "size_a" in df.columns
    assert "size_e" in df.columns
    assert "trial" in df.columns
    assert "avg_applicant_rank_a_opt" in df.columns


def test_correlation_score_is_nonnegative():
    rng = np.random.default_rng(0)
    S = rng.random((20, 20))
    assert correlation_score(S) >= 0


def test_run_imbalanced_experiment_aggregates_trials():
    A, E = _make_synthetic_population(n_a=50, n_e=50)
    df = run_imbalanced_experiment(
        A, E, fixed_size_e=10, sizes_a=[6, 10, 14], n_trials=3, base_seed=0
    )
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 9  # 3 sizes_a * 3 trials
    assert "trial" in df.columns
    assert "size_a" in df.columns
    assert "size_e" in df.columns
    assert (df["size_e"] == 10).all()
    assert set(df["size_a"].unique()) == {6, 10, 14}
