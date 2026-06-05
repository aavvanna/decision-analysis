# Proposer Advantage in Stable Matching — Research Project Design

**Date:** 2026-06-05
**Authors:** Vakulich Anastasia, Roshchina Nadezhda
**Course:** Decision Analysis

## 1. Purpose

Turn the existing repository — currently a JS demo of the Gale-Shapley algorithm running on synthetic data — into a complete course research project: an empirical study of the **proposer advantage** in two-sided matching on real labor-market data, plus a written report, with the JS app preserved as an interactive demo fed by a slice of the real data.

## 2. Scope

**In scope**
- Python-based research pipeline using two Kaggle datasets.
- Four experiments quantifying proposer advantage (baseline, market size, imbalance, preference correlation).
- A Markdown project report with embedded generated figures and placeholders for app screenshots.
- Rewiring the JS app to load a real-data sample (no algorithm or UI rewrite).
- A rewritten README describing the project (replacing the current usage guide).

**Out of scope**
- Many-to-one matching (capacity constraints).
- Ties in preference lists.
- Re-implementing the JS algorithm or visualisation.
- Skill extraction via ML/NER — we use vocabulary matching only.

## 3. Datasets

Both from Kaggle, English-language:

1. **Data Science Job Postings & Skills (2024)** — `asaniczka/data-science-job-postings-and-skills`. Real LinkedIn postings with pre-extracted skill lists (`job_skills.csv`). Used as the **employer** side.
2. **Resume Dataset** — `saugataroyarghya/resume-dataset`. Resume text per candidate; skills are extracted by matching against the shared skill vocabulary built from the first dataset. Used as the **applicant** side.

Treating the two sides as independent samples from a real labor market is the same setting as the theoretical Gale-Shapley setup, and is more realistic than a curated paired dataset.

## 4. Architecture

```
decision-analysis-project/
├── data/                          # NEW — Kaggle CSVs (gitignored, download instructions in README)
│   ├── job_skills.csv
│   └── resumes.csv
├── analysis/                      # NEW — Python research code
│   ├── 01_data_prep.ipynb         # Load, clean, build shared skill vocabulary, encode vectors
│   ├── 02_experiments.ipynb       # Run all 4 experiments, save results/*.csv
│   ├── 03_figures.ipynb           # Build figures from results/, save to figures/
│   ├── gale_shapley.py            # Python implementation + stability check
│   ├── data_processing.py         # Skill extraction, cosine sim, preference list construction
│   ├── experiments.py             # Experiment runner with seeded trials
│   └── requirements.txt           # Python deps (pandas, numpy, matplotlib, scipy, jupyter)
├── results/                       # NEW — aggregated CSVs from experiments
├── figures/                       # NEW — PNG figures referenced by report.md
├── js/
│   ├── data.js                    # MODIFIED — loads sample_data.json instead of generating
│   └── sample_data.json           # NEW — 20×20 real-data slice exported from notebook
├── index.html                     # MODIFIED — remove noise/skill-pool sliders, keep size selector
├── docs/superpowers/specs/        # NEW — this design doc
├── report.md                      # NEW — project report
├── README.md                      # REWRITTEN
└── (existing: css/, js/gale-shapley.js, js/main.js, js/visualize.js — untouched)
```

## 5. Data pipeline

1. Load `job_skills.csv` (one row per posting, comma-separated skill string).
2. Build **shared skill vocabulary** `V` = top-K most frequent skills across all postings (target K ≈ 50–100; choose based on coverage of resume text).
3. For each employer: binary vector `e ∈ {0,1}^|V|` from its skill list.
4. For each resume: scan text (case-insensitive substring match) for each skill in `V`; binary vector `a ∈ {0,1}^|V|`.
5. Drop applicants/employers with fewer than 3 skills in `V` (avoids degenerate zero-vectors and ties).
6. Similarity matrix `S[a, e] = cos(a, e)`.
7. Preference lists: rows of `S` sorted descending; ties broken by a deterministic per-row salt seeded from a global RNG seed (recorded in the report for reproducibility).

## 6. Algorithm

Direct port of `js/gale-shapley.js` to `analysis/gale_shapley.py`:
- `deferred_acceptance(proposer_prefs, receiver_prefs) -> matching: dict`
- `applicant_optimal(prefs) -> result`
- `employer_optimal(prefs) -> result`
- `find_blocking_pairs(matching, applicant_prefs, employer_prefs) -> list` — stability verifier.
- `avg_rank(matching, prefs) -> float`.

All experiments call these functions; the JS implementation remains the source of truth for the live demo and serves as a cross-check (run one configuration in both, assert identical matchings).

## 7. Experiments

For each experiment: K=100 trials with distinct RNG seeds; each trial **resamples** applicants and employers from the full populations.

### 7.1 Baseline
One balanced market (`n_a = n_e = 50`). Report mean ± std of:
- avg applicant rank under A-proposing
- avg employer rank under A-proposing
- avg applicant rank under E-proposing
- avg employer rank under E-proposing
- pairs changed between the two matchings
- proposer advantage = receiver-side rank − proposer-side rank
- stability check (must be 0 blocking pairs in all trials)

Paired Wilcoxon test: per-trial proposer-side rank vs receiver-side rank under each variant.

### 7.2 Market-size scaling
`n ∈ {10, 25, 50, 100}` balanced. For each n, K=100 trials. Plot mean proposer advantage vs n with error bars.

### 7.3 Imbalance
Fix `n_e = 30`, vary `n_a ∈ {20, 25, 30, 35, 40}`. K=100 trials each. Question: does the short side benefit regardless of who proposes?

### 7.4 Preference correlation
Each trial gets a **correlation score** = std of column means of `S` (high std ⇒ a few "star" employers dominate, low std ⇒ flat preferences). At fixed `n = 50`, K=200 trials; stratify into quartiles by correlation score and report proposer advantage per quartile. Hypothesis: higher correlation amplifies proposer advantage.

### Metrics summary
- `avg_proposer_rank`, `avg_receiver_rank` (1-indexed; smaller = better)
- `pairs_changed` between A-optimal and E-optimal
- `proposer_advantage = avg_receiver_rank − avg_proposer_rank`
- `n_blocking_pairs` (sanity)
- statistical test results where applicable

## 8. JS app integration

- Export from `02_experiments.ipynb`: a deterministic 20×20 slice → `js/sample_data.json` with `applicants`, `employers`, `simMatrix`, `applicantPrefs`, `employerPrefs` (same shape as the current `DATA.generate(cfg)` return value).
- `js/data.js`: replace `randomSkillVector` / `generate` path with `fetch('js/sample_data.json')`. Keep the same async-compatible API so `main.js`, `gale-shapley.js`, `visualize.js` are untouched.
- `index.html`: remove **Skills in Pool** and **Noise σ** sliders (no longer meaningful); keep **Applicants** and **Employers** sliders but cap them at the loaded sample size (slice the prefs at runtime).

## 9. Report (`report.md`)

Tight prose, ~6 rendered pages. Figures are mandatory in every results subsection; placeholders for screenshots in the demo section.

1. **Introduction** (~½ page) — Two-sided markets, deferred acceptance, statement of proposer-optimality theorem, why labor-market relevance.
2. **Data & methodology** (~1 page) — Dataset citations, shared skill vocabulary, vector encoding, cosine similarity, strict preference generation, RNG seed.
3. **Algorithm** (~½ page) — Pseudocode, stability definition, A-optimal vs E-optimal, link to JS source as reference.
4. **Experiments & results** (~2½ pages) — One subsection per experiment, each with **one figure** (saved in `figures/`) and 3–5 sentences of interpretation.
5. **Discussion** (~½ page) — Magnitude of proposer advantage on real data, role of preference correlation, limitations (similarity proxy, no ties, no capacities, single domain).
6. **Interactive demonstration** (~¼ page) — One paragraph describing the deployed app and what it shows. **Placeholders for 1–2 screenshots** (`figures/app_screenshot_1.png`, `figures/app_screenshot_2.png`) that the user will paste in themselves.
7. **References & reproducibility** (~¼ page) — Dataset URLs, reproduction commands, repo link, authors.

## 10. README rewrite

Replaces the current user-guide-style README. Sections:
- One-paragraph project description.
- Headline finding (one or two numbers from the baseline experiment).
- Repo layout.
- Reproducibility: install Python deps, where to put Kaggle CSVs, command to run notebooks.
- Live demo link + one sentence about what it shows.
- Link to `report.md` for full write-up.
- Authors.

## 11. Implementation order

1. Add `analysis/` skeleton, `requirements.txt`, `.gitignore` for `data/` and `results/`.
2. `data_processing.py` and `gale_shapley.py`.
3. `01_data_prep.ipynb` (produces clean `applicants.parquet`, `employers.parquet`, `vocab.json`).
4. `experiments.py` + `02_experiments.ipynb` (produces `results/*.csv` and `js/sample_data.json`).
5. `03_figures.ipynb` (produces `figures/*.png`).
6. Rewire `js/data.js` and `index.html`.
7. Write `report.md` referencing saved figures.
8. Rewrite `README.md`.

## 12. Reproducibility & quality bar

- Single global RNG seed declared at top of each notebook.
- Cross-check: run one configuration in both Python and JS; assert identical matchings (logged in `02_experiments.ipynb`).
- Stability check asserted in every trial; report aborts if any blocking pair is found.
- All figures regenerable from saved `results/*.csv` without re-running experiments.
