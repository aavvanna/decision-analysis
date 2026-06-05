# %% [markdown]
# # 02 — Experiments
#
# Runs four experiments quantifying proposer advantage on the real labor-market
# data prepared in 01_data_prep.py.
#
# 1. **Baseline** — balanced market (n=50), K=100 trials
# 2. **Market-size scaling** — n in {10, 25, 50, 100}, K=100 trials each
# 3. **Imbalance** — fixed n_e=30, vary n_a in {20,25,30,35,40}, K=100 trials each
# 4. **Preference correlation** — n=50, K=200 trials; stratify by similarity-matrix concentration
#
# Also exports a deterministic 20×20 sample for the JS demo app.

# %%
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

sys.path.insert(0, str(Path.cwd()))
from data_processing import cosine_similarity_matrix, build_preference_lists
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
# ## Experiment 1 — Baseline (balanced market, n=50)

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

w_app = wilcoxon(baseline_df["avg_applicant_rank_a_opt"], baseline_df["avg_applicant_rank_e_opt"])
w_emp = wilcoxon(baseline_df["avg_employer_rank_a_opt"], baseline_df["avg_employer_rank_e_opt"])
print(f"\nWilcoxon (applicant rank A-opt vs E-opt): statistic={w_app.statistic:.2f}, p={w_app.pvalue:.2e}")
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

# NOTE: balanced markets => size_a == size_e; group by either.
scaling_df["a_opt_advantage"] = (
    scaling_df["avg_employer_rank_a_opt"] - scaling_df["avg_applicant_rank_a_opt"]
)
print("Scaling summary (A-proposing advantage = avg_employer_rank - avg_applicant_rank):")
print(scaling_df.groupby("size_a")["a_opt_advantage"].agg(["mean", "std"]).round(3))

# Stability sanity for all scaling trials
assert (scaling_df["blocking_a_opt"] == 0).all()
assert (scaling_df["blocking_e_opt"] == 0).all()

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

print("Imbalance summary (mean of avg rank under each variant, by applicant-side size):")
print(imbalance_df.groupby("size_a")[[
    "avg_applicant_rank_a_opt", "avg_applicant_rank_e_opt",
    "avg_employer_rank_a_opt", "avg_employer_rank_e_opt",
]].mean().round(3))

assert (imbalance_df["blocking_a_opt"] == 0).all()
assert (imbalance_df["blocking_e_opt"] == 0).all()

# %% [markdown]
# ## Experiment 4 — Preference correlation
#
# Each trial has a correlation_score (std of column means of S). Stratify trials
# into quartiles by this score.

# %%
corr_df = pd.DataFrame([
    run_trial(A_pop, E_pop, CORR_N, CORR_N, seed=SEED + 10_000 + i)
    for i in range(CORR_TRIALS)
])
corr_df["quartile"] = pd.qcut(
    corr_df["correlation_score"], q=4, labels=["Q1 (low)", "Q2", "Q3", "Q4 (high)"]
)
corr_df["a_opt_advantage"] = (
    corr_df["avg_employer_rank_a_opt"] - corr_df["avg_applicant_rank_a_opt"]
)
corr_df.to_csv(RESULTS_DIR / "correlation.csv", index=False)

print("Correlation summary (A-proposing advantage by quartile):")
print(corr_df.groupby("quartile", observed=True)["a_opt_advantage"].agg(["mean", "std"]).round(3))

# %% [markdown]
# ## Export 20×20 sample for the JS app
#
# Deterministic slice; produces js/sample_data.json with the same shape the JS
# DATA.generate(cfg) used to return (applicants, employers, simMatrix,
# applicantPrefs, employerPrefs).

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
    "applicantPrefs": [[int(x) for x in row] for row in a_prefs],
    "employerPrefs": [[int(x) for x in row] for row in e_prefs],
}
JS_SAMPLE_PATH.parent.mkdir(exist_ok=True)
with open(JS_SAMPLE_PATH, "w") as f:
    json.dump(payload, f, indent=2)
print(f"Wrote {JS_SAMPLE_PATH.resolve()}")

# %% [markdown]
# ## Cross-check: compare Python and JS implementations on the same prefs
#
# We don't run JS here (no Node dependency), but we sanity-check by re-running
# the same 20×20 preferences through Python `applicant_optimal` and verifying
# stability + matching size.

# %%
from gale_shapley import applicant_optimal, employer_optimal

xcheck_a = applicant_optimal(a_prefs, e_prefs)
xcheck_e = employer_optimal(a_prefs, e_prefs)
print(f"Cross-check A-opt: matching size={len(xcheck_a['matching'])}, stable={xcheck_a['is_stable']}")
print(f"Cross-check E-opt: matching size={len(xcheck_e['matching'])}, stable={xcheck_e['is_stable']}")
assert xcheck_a["is_stable"] and xcheck_e["is_stable"]
assert len(xcheck_a["matching"]) == N
assert len(xcheck_e["matching"]) == N

# %%
print("\nAll experiments complete.")
print(f"Results written to: {RESULTS_DIR.resolve()}")
print(f"JS sample written to: {JS_SAMPLE_PATH.resolve()}")
