# Proposer Advantage in Stable Matching

A Decision Analysis course project: an empirical study of the **proposer advantage** in the Gale-Shapley deferred acceptance algorithm, applied to a two-sided labor market built from two real Kaggle datasets (job postings + resumes).

**📄 Full report:** [`report.md`](report.md)
**🌐 Live demo:** [here](https://aavvanna.github.io/decision-analysis/) (or serve the repo locally — see below)

---

## Headline findings

On a balanced market of 50 applicants × 50 employers (averaged over 100 resampled trials):

- Applicants get a better average partner rank when they propose (**8.35 vs 8.67**, paired Wilcoxon **p ≈ 5 × 10⁻¹²**).
- Employers get a better average rank when they propose (**4.88 vs 5.41**, p ≈ 5 × 10⁻¹²).
- About **2 of 50 pairs** change between the two stable matchings.
- **Stability holds in 100 % of 1 600+ trials** across all experiments — zero blocking pairs ever observed.

Three structural findings from the four experiments:

1. **Proposer advantage grows with market size** (0.09 → 0.77 rank units as n goes 10 → 100).
2. **Imbalance dominates proposer advantage** — when one side is strictly shorter, the gap between A-proposing and E-proposing collapses to within noise; the short side wins regardless of mechanism.
3. **Preference concentration amplifies proposer advantage** by ~25–35 % across quartiles, but never overturns the size or imbalance effects.

See [`report.md`](report.md) for the figures and the full discussion.

## Repository layout

```
analysis/                 Python research pipeline
├── gale_shapley.py       Deferred acceptance + stability verifier
├── data_processing.py    Skill vocabulary, encoding, similarity, preferences
├── experiments.py        Per-trial runner and experiment aggregators
├── 01_data_prep.py       Loads Kaggle CSVs, builds vocab, saves encoded matrices
├── 02_experiments.py     Runs all 4 experiments; exports js/sample_data.json
├── 03_figures.py         Generates figures from results/*.csv
├── tests/                pytest unit tests (26 tests)
└── requirements.txt
data/                     Kaggle CSVs (gitignored — see Reproducing below)
results/                  Aggregated experiment outputs (CSV)
figures/                  Generated figures used in the report
js/                       Interactive demo app (vanilla JS)
├── data.js               Loads sample_data.json (real 20×20 slice)
├── sample_data.json      Deterministic real-data sample for the demo
├── gale-shapley.js       JS Gale-Shapley implementation (cross-checked with Python)
├── main.js               UI orchestration
└── visualize.js          Bipartite graph + rank bar rendering
index.html                Demo entry point
css/style.css
report.md                 The project report
docs/superpowers/         Spec and implementation plan
```

## Reproducing the results

1. **Install Python deps**

   ```bash
   python3 -m venv .venv
   .venv/bin/pip install -r analysis/requirements.txt
   ```

2. **Download the Kaggle datasets** (requires a free Kaggle account)

   - Job postings: <https://www.kaggle.com/datasets/asaniczka/data-science-job-postings-and-skills>
     → place `job_skills.csv` at `data/job_skills.csv`
   - Resumes: <https://www.kaggle.com/datasets/saugataroyarghya/resume-dataset>
     → place the main CSV at `data/resume_data.csv`

3. **Run the pipeline** (`# %%` cell markers let you open each `.py` file as a Jupyter notebook in VSCode / Jupyter Lab, or just run as scripts)

   ```bash
   cd analysis
   ../.venv/bin/python3 01_data_prep.py      # builds vocabulary + encoded matrices
   ../.venv/bin/python3 02_experiments.py    # runs 4 experiments, writes results/*.csv + js/sample_data.json
   ../.venv/bin/python3 03_figures.py        # generates figures/*.png
   ```

4. **Run the test suite**

   ```bash
   .venv/bin/pytest analysis/tests/ -v
   ```

   Expected: all 26 tests pass.

All randomness is seeded from `SEED = 42`; results are deterministic.

## Try the live demo

The demo runs the same algorithm in JavaScript on a deterministic 20 × 20 slice of the real dataset, lets you switch between applicant-proposing and employer-proposing matchings, and visualises the differences as a bipartite graph.

```bash
python3 -m http.server 8000
# then open http://localhost:8000/
```

## Authors

- Vakulich Anastasia, БАСБ251
- Roshchina Nadezhda, БАСБ252
