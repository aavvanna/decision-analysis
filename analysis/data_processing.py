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
    """Return the subset of vocab whose terms appear in text (case-insensitive).

    Single-token alphanumeric skills (e.g. 'python', 'sql') are matched with a
    word-boundary regex to avoid spurious sub-matches. Skills containing any
    non-alphanumeric character (e.g. 'c++', 'c#', 'node.js', 'machine learning')
    are matched as plain substrings — `\\b` cannot reliably delimit non-word
    characters, and false positives are extremely unlikely at the noun-phrase
    scale of resume/job text.
    """
    if not text:
        return set()
    lowered = text.lower()
    found: Set[str] = set()
    for skill in vocab:
        if re.fullmatch(r"[a-z0-9]+", skill):
            if re.search(r"\b" + re.escape(skill) + r"\b", lowered):
                found.add(skill)
        else:
            if skill in lowered:
                found.add(skill)
    return found


def encode_skill_vector(skills: Set[str], vocab: List[str]) -> np.ndarray:
    """Binary encode a set of skills against the vocabulary order."""
    return np.array([1 if v in skills else 0 for v in vocab], dtype=np.float32)


def cosine_similarity_matrix(A: np.ndarray, E: np.ndarray) -> np.ndarray:
    """Cosine similarity between rows of A and rows of E. Zero-norm rows map to 0."""
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
