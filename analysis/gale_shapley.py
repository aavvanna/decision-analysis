"""Gale-Shapley deferred acceptance algorithm and stability utilities.

Mirrors the JS reference implementation in js/gale-shapley.js. All preference
lists are zero-indexed (proposer index -> ordered list of receiver indices,
most preferred first).
"""
from __future__ import annotations
from typing import Dict, List, Tuple


def deferred_acceptance(
    proposer_prefs: List[List[int]],
    receiver_prefs: List[List[int]],
) -> Dict[int, int]:
    """Run deferred acceptance with proposers proposing.

    Returns a dict mapping proposer index -> receiver index. Proposers whose
    preference lists are exhausted without a match are omitted from the result.
    """
    n_proposers = len(proposer_prefs)
    n_receivers = len(receiver_prefs)

    rank_lookup: List[Dict[int, int]] = [
        {p: i for i, p in enumerate(receiver_prefs[r])} for r in range(n_receivers)
    ]
    next_proposal = [0] * n_proposers
    held_by = [-1] * n_receivers
    matched_to = [-1] * n_proposers
    free = list(range(n_proposers))

    while free:
        next_free: List[int] = []
        for p in free:
            if next_proposal[p] >= len(proposer_prefs[p]):
                continue
            r = proposer_prefs[p][next_proposal[p]]
            next_proposal[p] += 1
            current = held_by[r]
            if current == -1:
                held_by[r] = p
                matched_to[p] = r
            else:
                rank_current = rank_lookup[r].get(current, float("inf"))
                rank_new = rank_lookup[r].get(p, float("inf"))
                if rank_new < rank_current:
                    held_by[r] = p
                    matched_to[p] = r
                    matched_to[current] = -1
                    next_free.append(current)
                else:
                    next_free.append(p)
        free = next_free

    return {p: r for p, r in enumerate(matched_to) if r != -1}


def avg_rank(matching: Dict[int, int], prefs: List[List[int]]) -> float:
    """Average 1-indexed rank that proposers got in `matching` per `prefs`."""
    if not matching:
        return 0.0
    total = 0
    for p, r in matching.items():
        total += prefs[p].index(r) + 1
    return total / len(matching)


def find_blocking_pairs(
    matching: Dict[int, int],
    applicant_prefs: List[List[int]],
    employer_prefs: List[List[int]],
) -> List[Tuple[int, int]]:
    """Return all blocking pairs (applicant, employer).

    `matching` is applicant -> employer. A pair (a, e) blocks if both a and e
    strictly prefer each other over their current partners (or being unmatched).
    """
    n_apps = len(applicant_prefs)
    n_emps = len(employer_prefs)
    receiver_match: Dict[int, int] = {e: a for a, e in matching.items()}

    blocking: List[Tuple[int, int]] = []
    for a in range(n_apps):
        for e in range(n_emps):
            current_e = matching.get(a, -1)
            if current_e == e:
                continue
            current_a = receiver_match.get(e, -1)
            rank_a_current = applicant_prefs[a].index(current_e) if current_e != -1 else len(applicant_prefs[a])
            rank_a_new = applicant_prefs[a].index(e) if e in applicant_prefs[a] else len(applicant_prefs[a])
            rank_e_current = employer_prefs[e].index(current_a) if current_a != -1 else len(employer_prefs[e])
            rank_e_new = employer_prefs[e].index(a) if a in employer_prefs[e] else len(employer_prefs[e])
            if rank_a_new < rank_a_current and rank_e_new < rank_e_current:
                blocking.append((a, e))
    return blocking


def applicant_optimal(
    applicant_prefs: List[List[int]],
    employer_prefs: List[List[int]],
) -> dict:
    """Run A-proposing deferred acceptance. Matching is applicant -> employer."""
    matching = deferred_acceptance(applicant_prefs, employer_prefs)
    blocking = find_blocking_pairs(matching, applicant_prefs, employer_prefs)
    emp_ranks_sum = 0
    for a, e in matching.items():
        emp_ranks_sum += employer_prefs[e].index(a) + 1
    avg_emp = emp_ranks_sum / len(matching) if matching else 0.0
    return {
        "matching": matching,
        "avg_applicant_rank": avg_rank(matching, applicant_prefs),
        "avg_employer_rank": avg_emp,
        "blocking_pairs": blocking,
        "is_stable": len(blocking) == 0,
        "type": "applicant-optimal",
    }


def employer_optimal(
    applicant_prefs: List[List[int]],
    employer_prefs: List[List[int]],
) -> dict:
    """Run E-proposing deferred acceptance. Matching is applicant -> employer."""
    raw = deferred_acceptance(employer_prefs, applicant_prefs)  # employer -> applicant
    matching = {a: e for e, a in raw.items()}
    blocking = find_blocking_pairs(matching, applicant_prefs, employer_prefs)
    emp_ranks_sum = 0
    for a, e in matching.items():
        emp_ranks_sum += employer_prefs[e].index(a) + 1
    avg_emp = emp_ranks_sum / len(matching) if matching else 0.0
    return {
        "matching": matching,
        "avg_applicant_rank": avg_rank(matching, applicant_prefs),
        "avg_employer_rank": avg_emp,
        "blocking_pairs": blocking,
        "is_stable": len(blocking) == 0,
        "type": "employer-optimal",
    }
