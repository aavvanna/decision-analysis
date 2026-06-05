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


def test_extract_skills_handles_skills_with_special_chars():
    """Skills with non-word characters (C++, C#, Node.js) must not be silently dropped."""
    text = "Experienced C++ developer, also writes C# and Node.js services."
    vocab = ["c++", "c#", "node.js", "python"]
    assert extract_skills_from_text(text, vocab) == {"c++", "c#", "node.js"}


def test_extract_skills_does_not_substring_match_inside_words():
    """Single-token alphanumeric skills must still respect word boundaries."""
    text = "javascript is not java"
    vocab = ["java"]
    # 'java' appears as a standalone word, so it should match.
    assert extract_skills_from_text(text, vocab) == {"java"}

    text2 = "I use javascript daily"
    # 'java' should NOT match inside 'javascript'.
    assert extract_skills_from_text(text2, ["java"]) == set()


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
    assert sorted(a_prefs_1[0]) == [0, 1]
    assert sorted(a_prefs_3[0]) == [0, 1]
