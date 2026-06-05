# %% [markdown]
# # 01 — Data preparation
#
# Loads the two Kaggle datasets, builds a shared skill vocabulary from the job
# postings, encodes both sides as binary skill vectors, and saves the cleaned
# matrices for the experiments notebook.
#
# Note: the resume dataset stores skills as stringified Python lists in its
# `skills` column, so we parse those directly with `ast.literal_eval` instead
# of extracting from free text.

# %%
import ast
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path.cwd()))
from data_processing import (
    parse_skill_list,
    build_vocabulary,
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
resumes = pd.read_csv(DATA_DIR / "resume_data.csv")
print(f"Resumes: {len(resumes)} rows")
print(f"Has skills column: {'skills' in resumes.columns}")

# %% [markdown]
# ## Build shared skill vocabulary from job postings
#
# Skills in job postings are comma-separated strings; we parse with our existing
# `parse_skill_list` helper and take the top-K most frequent.

# %%
job_skill_strings = jobs["job_skills"].dropna().astype(str).tolist()
vocab = build_vocabulary(job_skill_strings, top_k=VOCAB_SIZE)
print(f"Vocabulary size: {len(vocab)}")
print(f"First 20 skills: {vocab[:20]}")

# %% [markdown]
# ## Encode the employer side
#
# Each row's skill list, intersected with the vocab.

# %%
employer_vectors = []
n_dropped_employers = 0
for raw in job_skill_strings:
    skills = set(parse_skill_list(raw)) & set(vocab)
    if len(skills) < MIN_SKILLS:
        n_dropped_employers += 1
        continue
    employer_vectors.append(encode_skill_vector(skills, vocab))

E = np.stack(employer_vectors)
print(f"Employer matrix: {E.shape}  (dropped {n_dropped_employers} with <{MIN_SKILLS} matching skills)")

# %% [markdown]
# ## Parse and encode the applicant side
#
# `resumes['skills']` is a stringified Python list. We safely parse with
# `ast.literal_eval`, normalise to lowercase + stripped strings, intersect
# with the vocab.

# %%
def parse_python_list_skills(raw):
    """Parse a stringified Python list of skill names into a set of lowercased strings."""
    if not isinstance(raw, str) or not raw.strip():
        return set()
    try:
        parsed = ast.literal_eval(raw)
    except (ValueError, SyntaxError):
        return set()
    if not isinstance(parsed, list):
        return set()
    return {s.strip().lower() for s in parsed if isinstance(s, str) and s.strip()}


# %%
vocab_set = set(vocab)
applicant_vectors = []
n_dropped_applicants = 0
for raw in resumes["skills"].fillna(""):
    skills = parse_python_list_skills(raw) & vocab_set
    if len(skills) < MIN_SKILLS:
        n_dropped_applicants += 1
        continue
    applicant_vectors.append(encode_skill_vector(skills, vocab))

A = np.stack(applicant_vectors)
print(f"Applicant matrix: {A.shape}  (dropped {n_dropped_applicants} with <{MIN_SKILLS} matching skills)")

# %% [markdown]
# ## Save processed data

# %%
np.savez_compressed(OUT_DIR / "applicants.npz", X=A)
np.savez_compressed(OUT_DIR / "employers.npz", X=E)
with open(OUT_DIR / "vocab.json", "w") as f:
    json.dump(vocab, f, indent=2)

print(f"\nSaved to {OUT_DIR.resolve()}")
print(f"Final populations: {A.shape[0]} applicants, {E.shape[0]} employers, vocab={len(vocab)}")
