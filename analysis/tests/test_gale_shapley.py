"""Tests for analysis.gale_shapley."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest
from gale_shapley import (
    deferred_acceptance,
    applicant_optimal,
    employer_optimal,
    find_blocking_pairs,
    avg_rank,
)


# Classic 4x4 example from Gale & Shapley (1962), used in many textbooks.
# Applicants A,B,C,D; Employers W,X,Y,Z.
# Applicant prefs (index into employer list 0=W,1=X,2=Y,3=Z):
APP_PREFS = [
    [0, 1, 2, 3],  # A: W>X>Y>Z
    [1, 0, 2, 3],  # B: X>W>Y>Z
    [0, 1, 2, 3],  # C: W>X>Y>Z
    [3, 2, 1, 0],  # D: Z>Y>X>W
]
EMP_PREFS = [
    [2, 0, 1, 3],  # W: C>A>B>D
    [0, 1, 2, 3],  # X: A>B>C>D
    [3, 2, 1, 0],  # Y: D>C>B>A
    [0, 1, 2, 3],  # Z: A>B>C>D
]


def test_deferred_acceptance_returns_complete_matching():
    matching = deferred_acceptance(APP_PREFS, EMP_PREFS)
    assert len(matching) == 4
    assert set(matching.keys()) == {0, 1, 2, 3}
    assert set(matching.values()) == {0, 1, 2, 3}


def test_applicant_optimal_is_stable():
    matching = deferred_acceptance(APP_PREFS, EMP_PREFS)
    blocking = find_blocking_pairs(matching, APP_PREFS, EMP_PREFS)
    assert blocking == []


def test_applicant_optimal_gives_applicants_best_stable_partner():
    matching = deferred_acceptance(APP_PREFS, EMP_PREFS)
    # Sanity: average proposer rank should be low (proposers do well).
    assert avg_rank(matching, APP_PREFS) <= 2.0


def test_find_blocking_pairs_detects_unstable_matching():
    # Forced unstable: A->X, B->W, C->Y, D->Z.
    bad_matching = {0: 1, 1: 0, 2: 2, 3: 3}
    blocking = find_blocking_pairs(bad_matching, APP_PREFS, EMP_PREFS)
    # A prefers W over X, and W prefers A over B -> (A=0, W=0) is blocking
    assert (0, 0) in blocking


def test_avg_rank_is_one_indexed():
    # Each applicant gets their first choice -> avg rank = 1.0
    # Per-applicant prefs put their assigned partner at index 0 (top choice).
    matching = {0: 0, 1: 1, 2: 2, 3: 3}
    prefs = [
        [0, 1, 2, 3],  # applicant 0's top choice is 0
        [1, 0, 2, 3],  # applicant 1's top choice is 1
        [2, 0, 1, 3],  # applicant 2's top choice is 2
        [3, 0, 1, 2],  # applicant 3's top choice is 3
    ]
    assert avg_rank(matching, prefs) == 1.0


def test_applicant_optimal_helper():
    result = applicant_optimal(APP_PREFS, EMP_PREFS)
    assert result["matching"] is not None
    assert result["is_stable"] is True
    assert result["avg_applicant_rank"] >= 1.0
    assert result["avg_employer_rank"] >= 1.0


def test_employer_optimal_helper():
    result = employer_optimal(APP_PREFS, EMP_PREFS)
    assert result["matching"] is not None
    assert result["is_stable"] is True


def test_proposer_advantage_holds():
    # The proposing side should weakly prefer its outcome -> lower (better) avg rank.
    a_result = applicant_optimal(APP_PREFS, EMP_PREFS)
    e_result = employer_optimal(APP_PREFS, EMP_PREFS)
    assert a_result["avg_applicant_rank"] <= e_result["avg_applicant_rank"]
    assert e_result["avg_employer_rank"] <= a_result["avg_employer_rank"]


def test_unbalanced_market_one_side_short():
    # 3 applicants, 2 employers -> one applicant left unmatched.
    app_prefs = [[0, 1], [0, 1], [1, 0]]
    emp_prefs = [[0, 1, 2], [0, 1, 2]]
    matching = deferred_acceptance(app_prefs, emp_prefs)
    assert len(matching) == 2  # only 2 matched
