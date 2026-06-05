"""Experiment runners for the proposer-advantage study."""
from __future__ import annotations
from typing import List
import numpy as np
import pandas as pd

from data_processing import cosine_similarity_matrix, build_preference_lists
from gale_shapley import applicant_optimal, employer_optimal


def correlation_score(sim_matrix: np.ndarray) -> float:
    """Std of column means of the similarity matrix.

    High value => a few employers attract everyone (concentrated preferences).
    Low value => preferences are uniform.
    """
    return float(np.std(sim_matrix.mean(axis=0)))


def _sample(rows: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    idx = rng.choice(rows.shape[0], size=k, replace=False)
    return rows[idx]


def run_trial(
    applicants_pop: np.ndarray,
    employers_pop: np.ndarray,
    sample_size_a: int,
    sample_size_e: int,
    seed: int,
) -> dict:
    """One trial: sample sub-market, build prefs, run both variants, collect metrics."""
    rng = np.random.default_rng(seed)
    A = _sample(applicants_pop, sample_size_a, rng)
    E = _sample(employers_pop, sample_size_e, rng)

    S = cosine_similarity_matrix(A, E)
    a_prefs, e_prefs = build_preference_lists(S, seed=seed)

    a_opt = applicant_optimal(a_prefs, e_prefs)
    e_opt = employer_optimal(a_prefs, e_prefs)

    applicants_seen = a_opt["matching"].keys() | e_opt["matching"].keys()
    pairs_changed = sum(
        1 for a in applicants_seen
        if a_opt["matching"].get(a) != e_opt["matching"].get(a)
    )

    return {
        "seed": seed,
        "size_a": sample_size_a,
        "size_e": sample_size_e,
        "avg_applicant_rank_a_opt": a_opt["avg_applicant_rank"],
        "avg_employer_rank_a_opt": a_opt["avg_employer_rank"],
        "avg_applicant_rank_e_opt": e_opt["avg_applicant_rank"],
        "avg_employer_rank_e_opt": e_opt["avg_employer_rank"],
        "pairs_changed": pairs_changed,
        "blocking_a_opt": len(a_opt["blocking_pairs"]),
        "blocking_e_opt": len(e_opt["blocking_pairs"]),
        "correlation_score": correlation_score(S),
    }


def run_balanced_market_experiment(
    applicants_pop: np.ndarray,
    employers_pop: np.ndarray,
    sizes: List[int],
    n_trials: int,
    base_seed: int = 0,
) -> pd.DataFrame:
    """For each size in `sizes`, run n_trials balanced markets (n_a = n_e = size)."""
    rows = []
    for size in sizes:
        for trial in range(n_trials):
            metrics = run_trial(
                applicants_pop, employers_pop, size, size, seed=base_seed + size * 1000 + trial
            )
            metrics["trial"] = trial
            rows.append(metrics)
    return pd.DataFrame(rows)


def run_imbalanced_experiment(
    applicants_pop: np.ndarray,
    employers_pop: np.ndarray,
    fixed_size_e: int,
    sizes_a: List[int],
    n_trials: int,
    base_seed: int = 0,
) -> pd.DataFrame:
    """Vary applicant-side size while keeping employer side fixed."""
    rows = []
    for size_a in sizes_a:
        for trial in range(n_trials):
            metrics = run_trial(
                applicants_pop, employers_pop, size_a, fixed_size_e,
                seed=base_seed + size_a * 1000 + trial,
            )
            metrics["trial"] = trial
            rows.append(metrics)
    return pd.DataFrame(rows)
