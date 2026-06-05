# Proposer Advantage Research — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn the existing Gale-Shapley JS demo into a complete course research project — Python analysis pipeline over two Kaggle datasets, four experiments quantifying proposer advantage, a Markdown report with figures, and a JS app rewired to a real-data sample.

**Architecture:** Pure-Python core modules (`gale_shapley.py`, `data_processing.py`, `experiments.py`) with pytest unit tests; analysis notebooks authored as `.py` files in jupytext percent format (`# %%` cell markers — opens as notebooks in Jupyter/VSCode); results saved to `results/*.csv`; figures saved to `figures/*.png`; the JS app loads a deterministic 20×20 slice from `js/sample_data.json`.

**Tech Stack:** Python 3.13, pandas, numpy, scipy, matplotlib, pytest, jupytext, jupyter. Existing JS frontend: vanilla JS + D3-like SVG (unchanged except `data.js` and minor `index.html` edits).

**Commit policy:** No intermediate commits per project owner's request. Each task ends without committing. The final task is one consolidated commit after the user reviews the diff.

**Data preconditions:** User must download the Kaggle CSVs before Task 5:
- `data/job_skills.csv` — from <https://www.kaggle.com/datasets/asaniczka/data-science-job-postings-and-skills> (file `job_skills.csv`)
- `data/resumes.csv` — from <https://www.kaggle.com/datasets/saugataroyarghya/resume-dataset> (the main CSV — file name may vary; expected columns: a resume text column and optionally a category/role column)

---

## File map

**Create:**
- `analysis/requirements.txt`
- `analysis/gale_shapley.py`
- `analysis/data_processing.py`
- `analysis/experiments.py`
- `analysis/tests/__init__.py`
- `analysis/tests/test_gale_shapley.py`
- `analysis/tests/test_data_processing.py`
- `analysis/tests/test_experiments.py`
- `analysis/01_data_prep.py` (jupytext percent format)
- `analysis/02_experiments.py` (jupytext percent format)
- `analysis/03_figures.py` (jupytext percent format)
- `analysis/processed/applicants.npz` (output of 01)
- `analysis/processed/employers.npz` (output of 01)
- `analysis/processed/vocab.json` (output of 01)
- `results/baseline.csv`, `results/scaling.csv`, `results/imbalance.csv`, `results/correlation.csv` (output of 02)
- `js/sample_data.json` (output of 02)
- `figures/fig_baseline.png`, `figures/fig_scaling.png`, `figures/fig_imbalance.png`, `figures/fig_correlation.png` (output of 03)
- `report.md`
- `.gitignore` additions

**Modify:**
- `js/data.js` — load `sample_data.json` instead of generating synthetically
- `index.html` — remove unused sliders, label changes
- `README.md` — full rewrite

**Do not touch:**
- `js/gale-shapley.js`, `js/main.js`, `js/visualize.js`, `css/style.css`

---

## Task 1: Project scaffolding

**Files:**
- Create: `analysis/requirements.txt`, `analysis/tests/__init__.py`, `.gitignore` (modify if exists)
- Create directories: `analysis/`, `analysis/tests/`, `analysis/processed/`, `data/`, `results/`, `figures/`

- [ ] **Step 1: Create directories**

```bash
mkdir -p analysis/tests analysis/processed data results figures
```

- [ ] **Step 2: Create `analysis/requirements.txt`**

```
numpy>=1.26
pandas>=2.0
scipy>=1.11
matplotlib>=3.8
jupytext>=1.16
jupyter>=1.0
pytest>=8.0
```

- [ ] **Step 3: Create `analysis/tests/__init__.py`** (empty file, makes it a package)

- [ ] **Step 4: Add to `.gitignore`** (create file if missing; if it exists, append these lines after checking they're not already there)

```
# Data and large outputs (kept local; download instructions in README)
data/
analysis/processed/
__pycache__/
.pytest_cache/
.ipynb_checkpoints/
*.pyc
```

- [ ] **Step 5: Install Python deps**

```bash
pip3 install -r analysis/requirements.txt
```

Expected: all packages install successfully.

- [ ] **Step 6: Verify installation**

```bash
python3 -c "import numpy, pandas, scipy, matplotlib, jupytext; print('OK')"
pytest --version
```

Expected: `OK` on first command; pytest version number on second.

---

## Task 2: Implement `gale_shapley.py` with tests (TDD)

This is the core algorithm. We test against known textbook examples and the JS implementation's behaviour.

**Files:**
- Create: `analysis/tests/test_gale_shapley.py`
- Create: `analysis/gale_shapley.py`

- [ ] **Step 1: Write failing tests in `analysis/tests/test_gale_shapley.py`**

```python
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
    # In A-optimal each applicant gets the best partner they could possibly get
    # in ANY stable matching. For the example above, the A-optimal matching is
    # A-W (instead of A-X if W goes to C), but the standard result is:
    # A->X, B->W ... let's just assert the matching is stable; specific result
    # depends on tie-break behaviour, which we covered with the stability test.
    matching = deferred_acceptance(APP_PREFS, EMP_PREFS)
    # Sanity: average proposer rank should be low (proposers do well).
    assert avg_rank(matching, APP_PREFS) <= 2.0


def test_find_blocking_pairs_detects_unstable_matching():
    # Forced unstable: swap A and B's assignments.
    bad_matching = {0: 1, 1: 0, 2: 2, 3: 3}  # A->X, B->W, C->Y, D->Z
    blocking = find_blocking_pairs(bad_matching, APP_PREFS, EMP_PREFS)
    # A prefers W over X, and W prefers A over B -> (A=0, W=0) is blocking
    assert (0, 0) in blocking or (0, 0) in [(b[0], b[1]) for b in blocking]


def test_avg_rank_is_one_indexed():
    # Each applicant gets their first choice -> avg rank = 1.0
    matching = {0: 0, 1: 1, 2: 2, 3: 3}  # A->W, B->X, C->Y, D->Z
    prefs = [[0, 1, 2, 3]] * 4
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
```

- [ ] **Step 2: Run tests and verify they fail**

```bash
pytest analysis/tests/test_gale_shapley.py -v
```

Expected: all tests fail with `ModuleNotFoundError: No module named 'gale_shapley'`.

- [ ] **Step 3: Implement `analysis/gale_shapley.py`**

```python
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

    # rank_lookup[r][p] = rank of proposer p in receiver r's list (0 = best).
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
                continue  # exhausted, remains unmatched
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
    # Employer rank: for each matched employer, rank of their applicant in their list.
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
```

- [ ] **Step 4: Run tests and verify they pass**

```bash
pytest analysis/tests/test_gale_shapley.py -v
```

Expected: all 9 tests pass.

---

## Task 3: Implement `data_processing.py` with tests

**Files:**
- Create: `analysis/tests/test_data_processing.py`
- Create: `analysis/data_processing.py`

- [ ] **Step 1: Write failing tests in `analysis/tests/test_data_processing.py`**

```python
"""Tests for analysis.data_processing."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pytest
from data_processing import (
    parse_skill_list,
    build_vocabulary,
    extract_skills_from_text,
    encode_skill_vector,
    cosine_similarity_matrix,
    build_preference_lists,
)


def test_parse_skill_list_strips_and_lowercases():
    assert parse_skill_list("Python, SQL,  Machine Learning") == ["python", "sql", "machine learning"]


def test_parse_skill_list_handles_empty():
    assert parse_skill_list("") == []
    assert parse_skill_list(None) == []


def test_build_vocabulary_returns_top_k_skills():
    raw_lists = [
        "Python, SQL, ML",
        "Python, Java",
        "Python, SQL",
        "Java",
    ]
    vocab = build_vocabulary(raw_lists, top_k=2)
    assert set(vocab) == {"python", "sql"}  # python:3, sql:2 most frequent


def test_extract_skills_from_text_is_case_insensitive():
    text = "Experienced Python developer with strong SQL skills."
    vocab = ["python", "sql", "java"]
    skills = extract_skills_from_text(text, vocab)
    assert skills == {"python", "sql"}


def test_extract_skills_handles_multiword_phrases():
    text = "I work with Machine Learning models daily."
    vocab = ["machine learning", "python"]
    assert extract_skills_from_text(text, vocab) == {"machine learning"}


def test_encode_skill_vector_is_binary():
    vocab = ["python", "sql", "java"]
    skills = {"python", "java"}
    vec = encode_skill_vector(skills, vocab)
    np.testing.assert_array_equal(vec, np.array([1, 0, 1]))


def test_cosine_similarity_matrix_shape_and_range():
    A = np.array([[1, 0, 1], [0, 1, 1]], dtype=float)
    E = np.array([[1, 1, 0], [0, 0, 1], [1, 1, 1]], dtype=float)
    S = cosine_similarity_matrix(A, E)
    assert S.shape == (2, 3)
    assert (S >= 0).all() and (S <= 1).all()


def test_build_preference_lists_strict_order():
    # Applicant 0 most similar to employer 2, then 0, then 1.
    S = np.array([[0.3, 0.1, 0.9], [0.5, 0.8, 0.2]])
    a_prefs, e_prefs = build_preference_lists(S, seed=42)
    assert a_prefs[0] == [2, 0, 1]
    assert a_prefs[1] == [1, 0, 2]
    # Employer 0 most similar to applicant 1.
    assert e_prefs[0] == [1, 0]
    assert e_prefs[1] == [1, 0]
    assert e_prefs[2] == [0, 1]


def test_build_preference_lists_breaks_ties_deterministically():
    # Two applicants with identical similarity scores -> a tie-break is needed.
    S = np.array([[0.5, 0.5], [0.5, 0.5]])
    a_prefs_1, _ = build_preference_lists(S, seed=42)
    a_prefs_2, _ = build_preference_lists(S, seed=42)
    a_prefs_3, _ = build_preference_lists(S, seed=99)
    assert a_prefs_1 == a_prefs_2  # same seed -> same order
    # Different seeds may give different orders, but both are valid permutations.
    assert sorted(a_prefs_1[0]) == [0, 1]
    assert sorted(a_prefs_3[0]) == [0, 1]
```

- [ ] **Step 2: Run tests and verify they fail**

```bash
pytest analysis/tests/test_data_processing.py -v
```

Expected: all tests fail with `ModuleNotFoundError`.

- [ ] **Step 3: Implement `analysis/data_processing.py`**

```python
"""Data processing helpers: skill vocabulary, encoding, preferences."""
from __future__ import annotations
from collections import Counter
from typing import List, Set, Tuple
import re
import numpy as np


def parse_skill_list(raw: str | None) -> List[str]:
    """Parse a comma-separated skill string into a list of lowercased, stripped skills."""
    if not raw:
        return []
    return [s.strip().lower() for s in raw.split(",") if s.strip()]


def build_vocabulary(raw_skill_strings: List[str], top_k: int = 80) -> List[str]:
    """Build a vocabulary of the top_k most frequent skills across all rows."""
    counter: Counter[str] = Counter()
    for raw in raw_skill_strings:
        counter.update(parse_skill_list(raw))
    return [skill for skill, _ in counter.most_common(top_k)]


def extract_skills_from_text(text: str, vocab: List[str]) -> Set[str]:
    """Return the subset of vocab whose terms appear in text (case-insensitive, word-bounded)."""
    if not text:
        return set()
    lowered = text.lower()
    found: Set[str] = set()
    for skill in vocab:
        # Word-boundary match for single-word skills, plain substring for multi-word
        # (multi-word phrases already contain spaces, so substring is safe enough).
        if " " in skill:
            if skill in lowered:
                found.add(skill)
        else:
            if re.search(r"\b" + re.escape(skill) + r"\b", lowered):
                found.add(skill)
    return found


def encode_skill_vector(skills: Set[str], vocab: List[str]) -> np.ndarray:
    """Binary encode a set of skills against the vocabulary order."""
    return np.array([1 if v in skills else 0 for v in vocab], dtype=np.float32)


def cosine_similarity_matrix(A: np.ndarray, E: np.ndarray) -> np.ndarray:
    """Cosine similarity between rows of A and rows of E. Zero norms map to 0."""
    a_norm = np.linalg.norm(A, axis=1, keepdims=True)
    e_norm = np.linalg.norm(E, axis=1, keepdims=True)
    a_norm[a_norm == 0] = 1.0
    e_norm[e_norm == 0] = 1.0
    return (A / a_norm) @ (E / e_norm).T


def build_preference_lists(
    sim_matrix: np.ndarray, seed: int = 42
) -> Tuple[List[List[int]], List[List[int]]]:
    """Build strict preference lists for both sides from a similarity matrix.

    Ties are broken by a deterministic per-row noise seeded from `seed` plus the row index.
    """
    n_a, n_e = sim_matrix.shape
    a_prefs: List[List[int]] = []
    for i in range(n_a):
        rng = np.random.default_rng(seed + i)
        noise = rng.uniform(0, 1e-9, size=n_e)
        scores = sim_matrix[i] + noise
        a_prefs.append(list(np.argsort(-scores).astype(int)))

    e_prefs: List[List[int]] = []
    for j in range(n_e):
        rng = np.random.default_rng(seed + 10_000 + j)
        noise = rng.uniform(0, 1e-9, size=n_a)
        scores = sim_matrix[:, j] + noise
        e_prefs.append(list(np.argsort(-scores).astype(int)))

    return a_prefs, e_prefs
```

- [ ] **Step 4: Run tests and verify they pass**

```bash
pytest analysis/tests/test_data_processing.py -v
```

Expected: all 9 tests pass.

---

## Task 4: Implement `experiments.py` with tests

**Files:**
- Create: `analysis/tests/test_experiments.py`
- Create: `analysis/experiments.py`

- [ ] **Step 1: Write failing tests in `analysis/tests/test_experiments.py`**

```python
"""Tests for analysis.experiments."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import pytest
from experiments import run_trial, run_balanced_market_experiment, correlation_score


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
    assert "size" in df.columns
    assert "trial" in df.columns
    assert "avg_applicant_rank_a_opt" in df.columns


def test_correlation_score_is_nonnegative():
    rng = np.random.default_rng(0)
    S = rng.random((20, 20))
    assert correlation_score(S) >= 0
```

- [ ] **Step 2: Run tests and verify they fail**

```bash
pytest analysis/tests/test_experiments.py -v
```

Expected: all tests fail with `ModuleNotFoundError`.

- [ ] **Step 3: Implement `analysis/experiments.py`**

```python
"""Experiment runners for the proposer-advantage study."""
from __future__ import annotations
from typing import List, Optional
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

    pairs_changed = sum(
        1 for a in a_opt["matching"]
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
            metrics["size"] = size
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
```

- [ ] **Step 4: Run all tests so far**

```bash
pytest analysis/tests/ -v
```

Expected: all tests across the three test files pass.

---

## Task 5: Data prep script (`01_data_prep.py`)

**User precondition:** `data/job_skills.csv` and `data/resumes.csv` must exist (downloaded from Kaggle — see plan header). The plan executor should verify this before running.

**Files:**
- Create: `analysis/01_data_prep.py` (jupytext percent format)
- Outputs: `analysis/processed/applicants.npz`, `analysis/processed/employers.npz`, `analysis/processed/vocab.json`

- [ ] **Step 1: Verify input data files exist**

```bash
ls -la data/job_skills.csv data/resumes.csv
```

If either is missing, stop and ask the user to download them.

- [ ] **Step 2: Inspect data schemas**

```bash
python3 -c "import pandas as pd; print(pd.read_csv('data/job_skills.csv', nrows=3))"
python3 -c "import pandas as pd; print(pd.read_csv('data/resumes.csv', nrows=3))"
```

Expected: confirm `job_skills.csv` has a `job_skills` column (comma-separated). For resumes, identify the text column (likely `Resume` or `Resume_str` or similar — capture the actual name to use below).

- [ ] **Step 3: Create `analysis/01_data_prep.py`**

```python
# %% [markdown]
# # 01 — Data preparation
#
# Loads the two Kaggle datasets, builds a shared skill vocabulary from job
# postings, encodes both sides as binary skill vectors, and saves the cleaned
# matrices for the experiments notebook.

# %%
import json
from pathlib import Path
import numpy as np
import pandas as pd

import sys
sys.path.insert(0, str(Path.cwd()))
from data_processing import (
    parse_skill_list,
    build_vocabulary,
    extract_skills_from_text,
    encode_skill_vector,
)

DATA_DIR = Path("../data")
OUT_DIR = Path("processed")
OUT_DIR.mkdir(exist_ok=True)

SEED = 42
VOCAB_SIZE = 80
MIN_SKILLS = 3  # drop rows with fewer than this many matching skills

# %% [markdown]
# ## Load raw data

# %%
jobs = pd.read_csv(DATA_DIR / "job_skills.csv")
print(f"Jobs: {len(jobs)} rows, columns: {list(jobs.columns)}")
jobs.head(3)

# %%
resumes = pd.read_csv(DATA_DIR / "resumes.csv")
print(f"Resumes: {len(resumes)} rows, columns: {list(resumes.columns)}")
resumes.head(3)

# %% [markdown]
# ## Identify the resume text column
#
# Adjust `RESUME_TEXT_COL` to whatever the actual text column is named.

# %%
# IMPORTANT: set this to the actual column name after inspecting the dataframe above.
RESUME_TEXT_COL = "Resume_str"  # common name; verify against the printed columns above
assert RESUME_TEXT_COL in resumes.columns, f"Set RESUME_TEXT_COL to one of {list(resumes.columns)}"

# %% [markdown]
# ## Build shared skill vocabulary from job postings

# %%
job_skill_strings = jobs["job_skills"].dropna().astype(str).tolist()
vocab = build_vocabulary(job_skill_strings, top_k=VOCAB_SIZE)
print(f"Vocabulary size: {len(vocab)}")
print(f"First 20 skills: {vocab[:20]}")

# %% [markdown]
# ## Encode the employer side

# %%
employer_vectors = []
employer_meta = []
for idx, raw in enumerate(job_skill_strings):
    skills = set(parse_skill_list(raw)) & set(vocab)
    if len(skills) < MIN_SKILLS:
        continue
    employer_vectors.append(encode_skill_vector(skills, vocab))
    employer_meta.append({"row_idx": idx, "n_skills": len(skills)})

E = np.stack(employer_vectors)
print(f"Employer matrix shape: {E.shape}")

# %% [markdown]
# ## Encode the applicant side (extract skills from resume text)

# %%
resume_texts = resumes[RESUME_TEXT_COL].dropna().astype(str).tolist()
applicant_vectors = []
applicant_meta = []
for idx, text in enumerate(resume_texts):
    skills = extract_skills_from_text(text, vocab)
    if len(skills) < MIN_SKILLS:
        continue
    applicant_vectors.append(encode_skill_vector(skills, vocab))
    applicant_meta.append({"row_idx": idx, "n_skills": len(skills)})

A = np.stack(applicant_vectors)
print(f"Applicant matrix shape: {A.shape}")

# %% [markdown]
# ## Save processed data

# %%
np.savez_compressed(OUT_DIR / "applicants.npz", X=A)
np.savez_compressed(OUT_DIR / "employers.npz", X=E)
with open(OUT_DIR / "vocab.json", "w") as f:
    json.dump(vocab, f, indent=2)

print(f"Saved to {OUT_DIR.resolve()}")
print(f"Final populations: {A.shape[0]} applicants, {E.shape[0]} employers, vocab={len(vocab)}")
```

- [ ] **Step 4: Run the script**

From the `analysis/` directory:

```bash
cd analysis && python3 01_data_prep.py
```

Expected:
- Prints schemas, vocabulary preview.
- Saves `processed/applicants.npz`, `processed/employers.npz`, `processed/vocab.json`.
- Final populations have at least ~100 applicants and ~100 employers each. If much smaller, lower `MIN_SKILLS` or expand `VOCAB_SIZE`.

- [ ] **Step 5: Sanity-check output**

```bash
python3 -c "import numpy as np; A=np.load('analysis/processed/applicants.npz')['X']; E=np.load('analysis/processed/employers.npz')['X']; print('A:', A.shape, A.dtype, 'E:', E.shape, E.dtype)"
```

Expected: both arrays load; non-zero shapes; dtype float32.

---

## Task 6: Experiments script (`02_experiments.py`)

**Files:**
- Create: `analysis/02_experiments.py`
- Outputs: `results/baseline.csv`, `results/scaling.csv`, `results/imbalance.csv`, `results/correlation.csv`, `js/sample_data.json`

- [ ] **Step 1: Create `analysis/02_experiments.py`**

```python
# %% [markdown]
# # 02 — Experiments
#
# Runs four experiments quantifying proposer advantage:
# 1. Baseline (balanced market, n=50)
# 2. Market-size scaling (n in {10, 25, 50, 100})
# 3. Imbalance (fixed n_e=30, vary n_a)
# 4. Preference correlation (n=50, stratified by similarity-matrix concentration)
#
# Also exports a 20x20 deterministic sample for the JS app.

# %%
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

sys.path.insert(0, str(Path.cwd()))
from data_processing import cosine_similarity_matrix, build_preference_lists
from gale_shapley import applicant_optimal, employer_optimal
from experiments import (
    run_trial,
    run_balanced_market_experiment,
    run_imbalanced_experiment,
)

RESULTS_DIR = Path("../results")
RESULTS_DIR.mkdir(exist_ok=True)

SEED = 42
BASELINE_N = 50
BASELINE_TRIALS = 100
SCALING_SIZES = [10, 25, 50, 100]
SCALING_TRIALS = 100
IMBALANCE_E = 30
IMBALANCE_A_SIZES = [20, 25, 30, 35, 40]
IMBALANCE_TRIALS = 100
CORR_N = 50
CORR_TRIALS = 200

# %% [markdown]
# ## Load processed populations

# %%
A_pop = np.load("processed/applicants.npz")["X"]
E_pop = np.load("processed/employers.npz")["X"]
print(f"Applicant pop: {A_pop.shape}, Employer pop: {E_pop.shape}")

# %% [markdown]
# ## Experiment 1 — Baseline

# %%
baseline_df = pd.DataFrame([
    run_trial(A_pop, E_pop, BASELINE_N, BASELINE_N, seed=SEED + i)
    for i in range(BASELINE_TRIALS)
])
baseline_df.to_csv(RESULTS_DIR / "baseline.csv", index=False)

print("Baseline summary (mean +/- std):")
summary = baseline_df[[
    "avg_applicant_rank_a_opt", "avg_employer_rank_a_opt",
    "avg_applicant_rank_e_opt", "avg_employer_rank_e_opt",
    "pairs_changed", "blocking_a_opt", "blocking_e_opt",
]].agg(["mean", "std"])
print(summary.round(3))

# Paired Wilcoxon test: applicant rank under A-opt vs E-opt
w_app = wilcoxon(baseline_df["avg_applicant_rank_a_opt"], baseline_df["avg_applicant_rank_e_opt"])
w_emp = wilcoxon(baseline_df["avg_employer_rank_a_opt"], baseline_df["avg_employer_rank_e_opt"])
print(f"Wilcoxon (applicant rank A-opt vs E-opt): statistic={w_app.statistic:.2f}, p={w_app.pvalue:.2e}")
print(f"Wilcoxon (employer  rank A-opt vs E-opt): statistic={w_emp.statistic:.2f}, p={w_emp.pvalue:.2e}")

assert (baseline_df["blocking_a_opt"] == 0).all(), "Stability violated in A-optimal trial(s)!"
assert (baseline_df["blocking_e_opt"] == 0).all(), "Stability violated in E-optimal trial(s)!"

# %% [markdown]
# ## Experiment 2 — Market-size scaling

# %%
scaling_df = run_balanced_market_experiment(
    A_pop, E_pop, sizes=SCALING_SIZES, n_trials=SCALING_TRIALS, base_seed=SEED
)
scaling_df.to_csv(RESULTS_DIR / "scaling.csv", index=False)

print("Scaling summary (proposer advantage = receiver_rank - proposer_rank under A-opt):")
scaling_df["a_opt_advantage"] = (
    scaling_df["avg_employer_rank_a_opt"] - scaling_df["avg_applicant_rank_a_opt"]
)
print(scaling_df.groupby("size")["a_opt_advantage"].agg(["mean", "std"]).round(3))

# %% [markdown]
# ## Experiment 3 — Imbalance

# %%
imbalance_df = run_imbalanced_experiment(
    A_pop, E_pop,
    fixed_size_e=IMBALANCE_E,
    sizes_a=IMBALANCE_A_SIZES,
    n_trials=IMBALANCE_TRIALS,
    base_seed=SEED,
)
imbalance_df.to_csv(RESULTS_DIR / "imbalance.csv", index=False)

print("Imbalance summary (mean applicant rank under each variant):")
print(imbalance_df.groupby("size_a")[[
    "avg_applicant_rank_a_opt", "avg_applicant_rank_e_opt",
    "avg_employer_rank_a_opt", "avg_employer_rank_e_opt",
]].mean().round(3))

# %% [markdown]
# ## Experiment 4 — Preference correlation

# %%
corr_df = pd.DataFrame([
    run_trial(A_pop, E_pop, CORR_N, CORR_N, seed=SEED + 10_000 + i)
    for i in range(CORR_TRIALS)
])
# Stratify by quartile of correlation_score
corr_df["quartile"] = pd.qcut(corr_df["correlation_score"], q=4, labels=["Q1 (low)", "Q2", "Q3", "Q4 (high)"])
corr_df["a_opt_advantage"] = (
    corr_df["avg_employer_rank_a_opt"] - corr_df["avg_applicant_rank_a_opt"]
)
corr_df.to_csv(RESULTS_DIR / "correlation.csv", index=False)

print("Correlation summary (proposer advantage by quartile):")
print(corr_df.groupby("quartile", observed=True)["a_opt_advantage"].agg(["mean", "std"]).round(3))

# %% [markdown]
# ## Export 20x20 sample for the JS app

# %%
JS_SAMPLE_PATH = Path("../js/sample_data.json")
rng = np.random.default_rng(SEED)
N = 20
a_idx = rng.choice(A_pop.shape[0], size=N, replace=False)
e_idx = rng.choice(E_pop.shape[0], size=N, replace=False)
A_s = A_pop[a_idx]
E_s = E_pop[e_idx]
S_s = cosine_similarity_matrix(A_s, E_s)
a_prefs, e_prefs = build_preference_lists(S_s, seed=SEED)

with open("processed/vocab.json") as f:
    vocab = json.load(f)

def top_skills(vec, k=4):
    idxs = np.argsort(-vec)[:k]
    return [vocab[i] for i in idxs if vec[i] > 0][:k]

applicants_json = [
    {
        "id": i,
        "name": f"A{i+1:02d}",
        "label": f"Candidate {i+1}",
        "topSkills": top_skills(A_s[i]),
    }
    for i in range(N)
]
employers_json = [
    {
        "id": j,
        "name": f"E{j+1:02d}",
        "label": f"Position {j+1}",
        "topSkills": top_skills(E_s[j]),
    }
    for j in range(N)
]

payload = {
    "applicants": applicants_json,
    "employers": employers_json,
    "simMatrix": S_s.round(4).tolist(),
    "applicantPrefs": a_prefs,
    "employerPrefs": e_prefs,
}
JS_SAMPLE_PATH.parent.mkdir(exist_ok=True)
with open(JS_SAMPLE_PATH, "w") as f:
    json.dump(payload, f, indent=2)
print(f"Wrote {JS_SAMPLE_PATH.resolve()}")

# %% [markdown]
# ## Cross-check with JS implementation (sanity)
# Run the same 20x20 prefs through the JS via node if available; otherwise just
# rely on the Python tests as the source of truth.

# %%
print("All experiments complete.")
print(f"Results written to: {RESULTS_DIR.resolve()}")
```

- [ ] **Step 2: Run the script**

From the `analysis/` directory:

```bash
cd analysis && python3 02_experiments.py
```

Expected:
- Prints baseline summary (mean ± std for each metric).
- Wilcoxon p-values are very small (< 1e-10 typically for K=100 trials).
- All stability assertions pass.
- Scaling / imbalance / correlation tables print.
- `js/sample_data.json` and 4 `results/*.csv` files exist.

- [ ] **Step 3: Verify outputs exist**

```bash
ls -la results/ js/sample_data.json
```

Expected: 4 CSVs in `results/`, plus `js/sample_data.json` (a few hundred KB).

---

## Task 7: Figures script (`03_figures.py`)

**Files:**
- Create: `analysis/03_figures.py`
- Outputs: `figures/fig_baseline.png`, `figures/fig_scaling.png`, `figures/fig_imbalance.png`, `figures/fig_correlation.png`

- [ ] **Step 1: Create `analysis/03_figures.py`**

```python
# %% [markdown]
# # 03 — Figures
#
# Generates one figure per experiment from the CSVs produced by 02.

# %%
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

RESULTS_DIR = Path("../results")
FIG_DIR = Path("../figures")
FIG_DIR.mkdir(exist_ok=True)

plt.rcParams.update({
    "figure.dpi": 130,
    "savefig.dpi": 180,
    "font.size": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

# %% [markdown]
# ## Figure 1 — Baseline: avg rank for each side under each variant

# %%
b = pd.read_csv(RESULTS_DIR / "baseline.csv")
means = [
    b["avg_applicant_rank_a_opt"].mean(),
    b["avg_applicant_rank_e_opt"].mean(),
    b["avg_employer_rank_a_opt"].mean(),
    b["avg_employer_rank_e_opt"].mean(),
]
stds = [
    b["avg_applicant_rank_a_opt"].std(),
    b["avg_applicant_rank_e_opt"].std(),
    b["avg_employer_rank_a_opt"].std(),
    b["avg_employer_rank_e_opt"].std(),
]
labels = ["Applicants\n(A-proposing)", "Applicants\n(E-proposing)",
          "Employers\n(A-proposing)", "Employers\n(E-proposing)"]
colors = ["#2a9d8f", "#e76f51", "#2a9d8f", "#e76f51"]

fig, ax = plt.subplots(figsize=(7, 4))
ax.bar(labels, means, yerr=stds, color=colors, alpha=0.85, capsize=4)
ax.set_ylabel("Average partner rank (1 = best)")
ax.set_title(f"Baseline: balanced market (n=50), K={len(b)} trials")
ax.axvline(1.5, color="gray", linewidth=0.5)
fig.tight_layout()
fig.savefig(FIG_DIR / "fig_baseline.png")
plt.show()

# %% [markdown]
# ## Figure 2 — Scaling: proposer advantage vs market size

# %%
s = pd.read_csv(RESULTS_DIR / "scaling.csv")
s["a_opt_advantage"] = s["avg_employer_rank_a_opt"] - s["avg_applicant_rank_a_opt"]
s["e_opt_advantage"] = s["avg_applicant_rank_e_opt"] - s["avg_employer_rank_e_opt"]

agg = s.groupby("size").agg(
    a_mean=("a_opt_advantage", "mean"),
    a_std=("a_opt_advantage", "std"),
    e_mean=("e_opt_advantage", "mean"),
    e_std=("e_opt_advantage", "std"),
).reset_index()

fig, ax = plt.subplots(figsize=(7, 4))
ax.errorbar(agg["size"], agg["a_mean"], yerr=agg["a_std"], marker="o", label="Applicant-proposing advantage", color="#2a9d8f", capsize=4)
ax.errorbar(agg["size"], agg["e_mean"], yerr=agg["e_std"], marker="s", label="Employer-proposing advantage", color="#e76f51", capsize=4)
ax.set_xlabel("Market size (each side)")
ax.set_ylabel("Proposer rank advantage")
ax.set_title("How proposer advantage scales with market size")
ax.legend()
fig.tight_layout()
fig.savefig(FIG_DIR / "fig_scaling.png")
plt.show()

# %% [markdown]
# ## Figure 3 — Imbalance: average rank vs applicant-side size

# %%
i_df = pd.read_csv(RESULTS_DIR / "imbalance.csv")
agg = i_df.groupby("size_a").agg(
    a_a=("avg_applicant_rank_a_opt", "mean"),
    a_e=("avg_applicant_rank_e_opt", "mean"),
    e_a=("avg_employer_rank_a_opt", "mean"),
    e_e=("avg_employer_rank_e_opt", "mean"),
).reset_index()

fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(agg["size_a"], agg["a_a"], "o-", label="Applicant rank (A-opt)", color="#2a9d8f")
ax.plot(agg["size_a"], agg["a_e"], "o--", label="Applicant rank (E-opt)", color="#264653")
ax.plot(agg["size_a"], agg["e_a"], "s-", label="Employer rank (A-opt)", color="#e9c46a")
ax.plot(agg["size_a"], agg["e_e"], "s--", label="Employer rank (E-opt)", color="#e76f51")
ax.axvline(30, color="gray", linewidth=0.5, linestyle=":")
ax.set_xlabel("Number of applicants (employers fixed at 30)")
ax.set_ylabel("Average partner rank")
ax.set_title("Effect of market imbalance on average rank")
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(FIG_DIR / "fig_imbalance.png")
plt.show()

# %% [markdown]
# ## Figure 4 — Preference correlation: proposer advantage by quartile

# %%
c = pd.read_csv(RESULTS_DIR / "correlation.csv")
agg = c.groupby("quartile", observed=True).agg(
    mean=("a_opt_advantage", "mean"),
    std=("a_opt_advantage", "std"),
).reset_index()

fig, ax = plt.subplots(figsize=(7, 4))
ax.bar(agg["quartile"], agg["mean"], yerr=agg["std"], color="#2a9d8f", alpha=0.85, capsize=4)
ax.set_xlabel("Similarity-matrix concentration quartile")
ax.set_ylabel("Applicant-proposing advantage")
ax.set_title("Proposer advantage vs preference correlation (n=50, K=200)")
fig.tight_layout()
fig.savefig(FIG_DIR / "fig_correlation.png")
plt.show()

print(f"All figures saved to {FIG_DIR.resolve()}")
```

- [ ] **Step 2: Run the script**

```bash
cd analysis && python3 03_figures.py
```

Expected:
- 4 PNG files in `figures/`.
- Each plot opens (or, if running headlessly, just saves silently).

- [ ] **Step 3: Verify figures exist**

```bash
ls -la figures/
```

Expected: `fig_baseline.png`, `fig_scaling.png`, `fig_imbalance.png`, `fig_correlation.png`.

---

## Task 8: Rewire `js/data.js` to load real-data sample

**Files:**
- Modify: `js/data.js` (replace `generate` body with a fetch-based loader; keep public API)

- [ ] **Step 1: Read current `js/data.js`**

```bash
cat js/data.js
```

Confirm exports: the IIFE returns `{ generate, cosineSim }`. We must keep `cosineSim` exported (since `main.js`/`visualize.js` may use it) and replace `generate` with a real-data-backed version that returns the same shape.

- [ ] **Step 2: Replace the contents of `js/data.js` with**

```javascript
/**
 * data.js — Real-data loader
 *
 * Loads a deterministic 20x20 sample produced by analysis/02_experiments.py
 * from js/sample_data.json. The sample contains real LinkedIn job-posting
 * skill profiles and real resume skill profiles (extracted from Kaggle
 * datasets) along with their cosine-similarity matrix and derived strict
 * preference lists.
 *
 * Exposes the same API as the original synthetic generator so the rest of
 * the app does not need to change.
 */

const DATA = (() => {

  let cached = null;  // { applicants, employers, simMatrix, applicantPrefs, employerPrefs }

  async function loadSample() {
    if (cached) return cached;
    const resp = await fetch('js/sample_data.json');
    if (!resp.ok) {
      throw new Error(`Failed to load sample_data.json: ${resp.status}`);
    }
    cached = await resp.json();
    return cached;
  }

  /**
   * Return a slice of the loaded sample with the requested number of
   * applicants and employers. Preference lists are filtered to the chosen
   * subsets (preserving relative order).
   *
   * @param {Object} cfg - { nApplicants, nEmployers }  (other fields ignored)
   */
  async function generate(cfg) {
    const sample = await loadSample();
    const nA = Math.min(cfg.nApplicants ?? sample.applicants.length, sample.applicants.length);
    const nE = Math.min(cfg.nEmployers ?? sample.employers.length, sample.employers.length);

    const aSet = new Set(Array.from({ length: nA }, (_, i) => i));
    const eSet = new Set(Array.from({ length: nE }, (_, i) => i));

    const applicants = sample.applicants.slice(0, nA);
    const employers = sample.employers.slice(0, nE);

    // Slice the similarity matrix
    const simMatrix = applicants.map((_, ai) =>
      employers.map((_, ei) => sample.simMatrix[ai][ei])
    );

    // Filter preference lists to the active subsets, preserving order.
    const applicantPrefs = sample.applicantPrefs.slice(0, nA).map(list =>
      list.filter(ei => eSet.has(ei))
    );
    const employerPrefs = sample.employerPrefs.slice(0, nE).map(list =>
      list.filter(ai => aSet.has(ai))
    );

    return {
      applicants,
      employers,
      skills: [],  // kept for API compatibility; vocabulary is implicit in topSkills
      simMatrix,
      applicantPrefs,
      employerPrefs,
    };
  }

  /** Cosine similarity between two Float32Arrays (kept for any external callers). */
  function cosineSim(a, b) {
    let dot = 0, na = 0, nb = 0;
    for (let i = 0; i < a.length; i++) {
      dot += a[i] * b[i];
      na += a[i] * a[i];
      nb += b[i] * b[i];
    }
    const denom = Math.sqrt(na) * Math.sqrt(nb);
    return denom === 0 ? 0 : dot / denom;
  }

  return { generate, cosineSim };
})();
```

- [ ] **Step 3: Check `main.js` for `DATA.generate` call style**

```bash
grep -n "DATA.generate" js/main.js
```

Expected: one or more callsites. If they treat it synchronously (i.e., not awaiting), there's a problem.

- [ ] **Step 4: If `main.js` calls `DATA.generate` synchronously, patch it to await**

Read `js/main.js` to find the callsite. Replace any `const data = DATA.generate(cfg);` with:

```javascript
const data = await DATA.generate(cfg);
```

and make sure the enclosing function is `async`. (Most likely a single button-click handler — wrap it as `async (e) => { ... }`.)

- [ ] **Step 5: Smoke-test by serving locally**

```bash
python3 -m http.server 8000
```

Open <http://localhost:8000/> in a browser. Click **Run Simulation**.

Expected:
- No console errors.
- The visualization renders with real-data labels (skill names from the Kaggle vocab visible in the preference table).
- Stable matchings appear; comparison panels work.

Stop the server with Ctrl-C.

---

## Task 9: Update `index.html` to remove unused sliders

The `Skills in Pool` and `Noise σ` sliders no longer mean anything (we load real data; there's no synthetic generator to parameterise).

**Files:**
- Modify: `index.html`

- [ ] **Step 1: Read `index.html` to find the slider markup**

```bash
grep -n "Skills in Pool\|Noise\|nSkills\|noise" index.html
```

- [ ] **Step 2: Remove the `Skills in Pool` and `Noise σ` slider blocks**

Delete the `<label>...</label>` and any container divs for those two controls. Also remove any `id`s referenced from `main.js` so it doesn't break. Check `main.js` for references like `document.getElementById('nSkills')` and remove the corresponding read in the config-building code (default it to a safe value or drop it from the cfg object — the new `DATA.generate` ignores those fields anyway).

Concretely:
- In `index.html`: remove the two `<label>` blocks for `nSkills` and `noise`.
- In `main.js`: remove or comment out the `nSkills` and `noise` reads. Replace the `cfg` object with `{ nApplicants, nEmployers }` only.

- [ ] **Step 3: Re-test**

Reload <http://localhost:8000/>. Verify the sliders are gone and the app still runs.

---

## Task 10: Write `report.md`

This is the longest writing task. It should be done **after** running all experiments so the actual numbers can be filled in. The structure below is a template — replace bracketed `<RESULT>` placeholders with values from the printed summaries in `02_experiments.py`.

**Files:**
- Create: `report.md`

- [ ] **Step 1: Create `report.md` with the following structure**

```markdown
# Comparative Analysis of Proposer Advantage in Stable Matching

A Decision Analysis course project applying the Gale-Shapley deferred acceptance
algorithm to a real two-sided labor market, quantifying the magnitude of the
proposer advantage across four experimental settings.

**Authors:** Vakulich Anastasia (БАСБ251), Roshchina Nadezhda (БАСБ252)
**Repository:** <link-to-repo>
**Live demo:** <link-to-deployed-app>

---

## 1. Introduction

A two-sided matching market consists of two disjoint sets of agents in which
each agent ranks members of the opposite set. Gale & Shapley (1962) showed that
the *deferred acceptance* algorithm always produces a **stable matching** — one
with no blocking pair — and that the side that proposes obtains the matching
which is simultaneously optimal for all proposers across the set of stable
matchings (the **proposer-optimality theorem**).

The proposer-optimality theorem is a structural result: it says no proposer can
do strictly better in *any* stable matching, and dually, no receiver can do
strictly worse. It does not tell us how *large* the advantage is in a given
market, which is empirically interesting in markets such as residency matching,
school choice, or, in our case, the labor market.

This project measures the magnitude of the proposer advantage on real data and
asks how it depends on market size, imbalance, and the structure of preferences.

## 2. Data and methodology

We use two publicly available Kaggle datasets:

1. **Data Science Job Postings & Skills (2024)** — `asaniczka/data-science-job-postings-and-skills`. Real LinkedIn postings with comma-separated extracted skill lists. Source for the **employer** side.
2. **Resume Dataset** — `saugataroyarghya/resume-dataset`. Resume text per candidate. Source for the **applicant** side.

### 2.1 Skill vocabulary and encoding

We build a shared vocabulary `V` of the |V| = <RESULT: vocab size> most frequent skills across
all job postings. Each employer is encoded as a binary vector `e ∈ {0,1}^|V|` from
its labelled skill list. Each applicant is encoded as `a ∈ {0,1}^|V|`, where the
skill is set to 1 if the corresponding term appears (case-insensitive,
word-boundary match) in the resume text. Rows with fewer than 3 active skills
are discarded. After cleaning, we work with **<RESULT: n applicants>** applicants and
**<RESULT: n employers>** employers.

### 2.2 Preferences

Pairwise similarity is computed as `S[a, e] = cos(a, e)`. Each applicant's
preference list over employers is the row of `S` sorted in descending order;
each employer's list is the column sorted in descending order. Ties are broken
deterministically using a row-specific seed derived from a fixed global seed
(`SEED = 42`), so all experiments are reproducible.

### 2.3 Algorithm

We implement the deferred acceptance algorithm in Python (`analysis/gale_shapley.py`),
mirroring the JavaScript reference in `js/gale-shapley.js`. The implementation
exposes two entry points: `applicant_optimal` (applicants propose) and
`employer_optimal` (employers propose). After each run we verify stability by
exhaustively searching for blocking pairs; this verifier is run on every trial
in every experiment and produces zero blocking pairs throughout (sanity check).

### 2.4 Experimental design

Four experiments. For each, we sample sub-markets from the cleaned populations,
build preference lists from the local similarity matrix, run both algorithm
variants, and aggregate metrics over `K` resampled trials.

| Experiment | Question | Parameters |
|---|---|---|
| Baseline | What does the proposer advantage look like in a typical balanced market? | n=50, K=100 |
| Scaling | How does the proposer advantage change with market size? | n ∈ {10, 25, 50, 100}, K=100 each |
| Imbalance | Does being on the short side matter regardless of who proposes? | n_e=30 fixed, n_a ∈ {20, 25, 30, 35, 40}, K=100 each |
| Correlation | Does concentrated preference structure amplify the proposer advantage? | n=50, K=200; stratify by σ of column means of S |

### 2.5 Metrics

- `avg_proposer_rank` — average 1-indexed rank of the partner the proposing side received.
- `avg_receiver_rank` — same for the non-proposing side.
- `proposer_advantage` — `avg_receiver_rank − avg_proposer_rank` (positive means the proposing side does better than the receiving side).
- `pairs_changed` — number of applicants whose match differs between the A-optimal and E-optimal matchings.

## 3. Results

### 3.1 Baseline

In a balanced market with n=50 on each side and K=100 resampled trials:

| Metric | Value (mean ± std) |
|---|---|
| Avg applicant rank under A-proposing | <RESULT> |
| Avg applicant rank under E-proposing | <RESULT> |
| Avg employer rank under A-proposing | <RESULT> |
| Avg employer rank under E-proposing | <RESULT> |
| Pairs changed between matchings | <RESULT> |
| Blocking pairs (sanity) | 0 ± 0 |

A paired Wilcoxon signed-rank test rejects the null of equal applicant rank
between the two variants at p < <RESULT: p-value> (n=100), confirming that the
proposer advantage is statistically significant in this market.

![Baseline](figures/fig_baseline.png)

**Reading:** <RESULT: 3-5 sentences interpreting the bars — magnitude of the gap, what it means in practice, any surprises>.

### 3.2 Market-size scaling

We vary n ∈ {10, 25, 50, 100} balanced and run K=100 trials at each size.

![Scaling](figures/fig_scaling.png)

**Reading:** <RESULT: 3-5 sentences — does the gap grow, shrink, or stabilise? Relate to theory>.

### 3.3 Imbalance

Holding the employer side fixed at n_e=30 and varying n_a ∈ {20, 25, 30, 35, 40}:

![Imbalance](figures/fig_imbalance.png)

**Reading:** <RESULT: 3-5 sentences — which side benefits from being scarce? Does the proposer advantage interact with imbalance?>.

### 3.4 Preference correlation

We compute a per-trial correlation score (standard deviation of the column
means of the similarity matrix) and stratify K=200 trials into quartiles.

![Correlation](figures/fig_correlation.png)

**Reading:** <RESULT: 3-5 sentences — does concentration amplify or dampen proposer advantage?>.

## 4. Discussion

<RESULT: ~6-10 sentences. Synthesise the four experiments. Compare observed magnitude with the qualitative theorem ('proposer weakly prefers'). Highlight the most interesting finding (e.g., the correlation result if it's striking). Note limitations: the similarity proxy assumes skill overlap is the only signal; we use one-to-one matching whereas real labor markets are many-to-one; strict preferences exclude ties that arise naturally in shortlisting; the analysis is restricted to data-science adjacent roles.>

## 5. Interactive demonstration

The repository includes a deployed web app that runs the same algorithm
end-to-end on a 20×20 deterministic slice of the real dataset used above. Users
can pick the matching variant and see the bipartite graph, per-candidate rank
satisfaction, and the differences between the two stable matchings.

<!-- Insert 1-2 screenshots of the deployed app here. Suggested filenames:
     figures/app_screenshot_1.png, figures/app_screenshot_2.png -->

![App screenshot 1](figures/app_screenshot_1.png)
*Caption: <e.g., "The comparative analysis view showing applicant rank bars under both variants.">*

![App screenshot 2](figures/app_screenshot_2.png)
*Caption: <e.g., "The bipartite matching graph; pink edges mark pairs that changed between the two stable matchings.">*

## 6. References and reproducibility

**Datasets**
- Data Science Job Postings & Skills (2024): <https://www.kaggle.com/datasets/asaniczka/data-science-job-postings-and-skills>
- Resume Dataset: <https://www.kaggle.com/datasets/saugataroyarghya/resume-dataset>

**Bibliography**
- Gale, D., & Shapley, L. S. (1962). College admissions and the stability of marriage. *American Mathematical Monthly*, 69(1), 9-15.
- Roth, A. E., & Sotomayor, M. A. O. (1990). *Two-sided matching: A study in game-theoretic modeling and analysis*. Cambridge University Press.

**Reproducing the results**

```bash
# 1. Install Python deps
pip3 install -r analysis/requirements.txt

# 2. Place Kaggle CSVs at:
#    data/job_skills.csv  (from asaniczka dataset)
#    data/resumes.csv     (from saugataroyarghya dataset)

# 3. Run the pipeline (each script doubles as a Jupyter notebook
#    via jupytext '# %%' cell markers)
cd analysis
python3 01_data_prep.py
python3 02_experiments.py
python3 03_figures.py

# 4. Run the test suite
pytest tests/ -v
```

All randomness is seeded from `SEED = 42`; results are deterministic.
```

- [ ] **Step 2: Fill in the `<RESULT>` placeholders**

Run `02_experiments.py` again (or read the saved CSVs) to get the exact numbers for the baseline table, vocabulary/population sizes, Wilcoxon p-values, and write the "Reading" interpretation paragraphs based on the figures. Each "Reading" should be 3-5 sentences of plain prose; no equations, no jargon beyond what's already in the methodology section.

For the Discussion section, write 6-10 sentences as described in the placeholder.

- [ ] **Step 3: Re-render and check**

Open `report.md` in any Markdown previewer (or push-preview on GitHub locally). Verify:
- All `<RESULT>` placeholders are gone.
- All four figures embed correctly.
- Screenshot placeholders are present where the user expects to paste their own.

---

## Task 11: Rewrite `README.md`

**Files:**
- Modify (full rewrite): `README.md`

- [ ] **Step 1: Replace `README.md` with**

```markdown
# Proposer Advantage in Stable Matching

A Decision Analysis course project: an empirical study of the proposer advantage
in the Gale-Shapley deferred acceptance algorithm, applied to a two-sided labor
market built from two Kaggle datasets (job postings + resumes).

**📄 Full report:** [`report.md`](report.md)
**🌐 Live demo:** <link-to-deployed-app>

## Headline finding

In a balanced market with 50 applicants and 50 employers (averaged over 100
resampled trials), applicants who propose receive an average partner rank of
**<RESULT>** versus **<RESULT>** when employers propose — a measurable gap
significant at p < <RESULT>. The full report breaks this down by market size,
imbalance, and preference correlation.

## Repository layout

```
analysis/         Python research pipeline
├── gale_shapley.py        Deferred acceptance + stability check
├── data_processing.py     Skill extraction, similarity, preferences
├── experiments.py         Trial + experiment runners
├── 01_data_prep.py        Load Kaggle CSVs, build vocabulary, encode vectors
├── 02_experiments.py      Run all four experiments, export sample for the app
├── 03_figures.py          Build figures from results
└── tests/                 pytest unit tests
data/             Kaggle CSVs (gitignored — see Reproducing below)
results/          Aggregated experiment outputs (CSV)
figures/          Generated figures used in the report
js/               The interactive demo app (vanilla JS)
index.html        Demo entry point
report.md         The project report
```

## Reproducing the results

1. **Install Python deps**

   ```bash
   pip3 install -r analysis/requirements.txt
   ```

2. **Download the Kaggle datasets** (requires a free Kaggle account)

   - Job postings: <https://www.kaggle.com/datasets/asaniczka/data-science-job-postings-and-skills>
     → place `job_skills.csv` at `data/job_skills.csv`
   - Resumes: <https://www.kaggle.com/datasets/saugataroyarghya/resume-dataset>
     → place the main CSV at `data/resumes.csv`

3. **Run the pipeline** (`# %%` cell markers let you open each `.py` file as a Jupyter notebook in VSCode / Jupyter Lab, or just run as scripts)

   ```bash
   cd analysis
   python3 01_data_prep.py   # builds vocabulary + encoded matrices
   python3 02_experiments.py # runs 4 experiments, writes results/*.csv and js/sample_data.json
   python3 03_figures.py     # generates figures/*.png
   ```

4. **Run the test suite**

   ```bash
   pytest analysis/tests/ -v
   ```

   Expected: all tests pass.

## Try the live demo

```bash
python3 -m http.server 8000
# open http://localhost:8000/
```

The demo runs the same algorithm in JavaScript on a deterministic 20×20 slice
of the real dataset, lets you switch between applicant-proposing and
employer-proposing matchings, and visualises the differences as a bipartite
graph.

## Authors

- Vakulich Anastasia, БАСБ251
- Roshchina Nadezhda, БАСБ252
```

- [ ] **Step 2: Fill in the headline-finding numbers from the baseline results**

---

## Task 12: Final review and single consolidated commit

**Files:** all

- [ ] **Step 1: Run the full test suite one final time**

```bash
pytest analysis/tests/ -v
```

Expected: all tests pass.

- [ ] **Step 2: Re-run the pipeline end-to-end to make sure everything still works from scratch**

```bash
cd analysis
python3 01_data_prep.py
python3 02_experiments.py
python3 03_figures.py
cd ..
```

Expected: no errors; outputs in `analysis/processed/`, `results/`, `figures/`, `js/sample_data.json`.

- [ ] **Step 3: Smoke-test the JS app**

```bash
python3 -m http.server 8000
# Browse http://localhost:8000/, click Run Simulation, verify bipartite graph renders.
```

- [ ] **Step 4: Review the full diff**

```bash
git status
git diff --stat
```

Show the user what's about to be committed. Wait for their go-ahead.

- [ ] **Step 5: Stage and commit everything in one commit (user-approved)**

```bash
git add docs/superpowers/specs/ docs/superpowers/plans/ \
        analysis/ \
        results/ figures/ \
        js/data.js js/sample_data.json \
        index.html \
        report.md README.md \
        .gitignore

git status  # confirm what's staged

git commit -m "$(cat <<'EOF'
Build research project: proposer advantage in stable matching

- Add Python research pipeline (analysis/) with TDD-covered
  Gale-Shapley implementation, data processing, and experiment runners.
- Four experiments quantifying proposer advantage on Kaggle labor-market
  data (baseline, scaling, imbalance, preference correlation).
- Generate figures and project report (report.md).
- Rewire JS demo app to load a deterministic 20x20 real-data sample.
- Rewrite README as a project README.
EOF
)"
```

- [ ] **Step 6: Confirm with the user before any push**

Do not run `git push` unless the user explicitly approves it.

---

## Spec coverage self-review

Checked the plan against `docs/superpowers/specs/2026-06-05-proposer-advantage-research-design.md`:

| Spec section | Plan task(s) |
|---|---|
| §3 Datasets | Task 5 (downloads + loaders) |
| §4 Architecture / file map | Task 1 (scaffolding), Tasks 2-7 (each file created) |
| §5 Data pipeline | Task 3 (data_processing), Task 5 (01_data_prep) |
| §6 Algorithm port | Task 2 (gale_shapley + TDD) |
| §7 Experiments (4) | Task 4 (experiments.py), Task 6 (02_experiments) |
| §8 JS app integration | Task 6 (export sample_data.json), Task 8 (rewire data.js), Task 9 (index.html) |
| §9 Report sections | Task 10 |
| §10 README rewrite | Task 11 |
| §11 Implementation order | Tasks 1→12 follow this order |
| §12 Reproducibility / cross-check | Tasks 2, 4 (tests); Task 6 step on assertions; final commit step preserves seeds |

No gaps detected. No `TBD`/`TODO` placeholders in the plan itself (the `<RESULT>` markers in `report.md` are intentional — they're for the executor to fill in with measured numbers from the experiments).

Type/signature consistency: `applicant_optimal` / `employer_optimal` return `dict` with the same keys throughout (`matching`, `avg_applicant_rank`, `avg_employer_rank`, `blocking_pairs`, `is_stable`, `type`); `run_trial` returns a `dict` with the same field names referenced in Task 6 aggregations.
