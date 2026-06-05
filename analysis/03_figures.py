# %% [markdown]
# # 03 — Figures
#
# Generates one figure per experiment from the CSVs produced by 02_experiments.
#
# Key metric definitions used throughout:
#   applicant_proposer_gain = avg_applicant_rank_e_opt - avg_applicant_rank_a_opt
#       (positive => A-opt gives applicants a better rank than E-opt)
#   employer_proposer_gain  = avg_employer_rank_a_opt  - avg_employer_rank_e_opt
#       (positive => E-opt gives employers a better rank than A-opt)
# Both are expected to be positive by the Gale-Shapley theorem.

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

C_APP = "#2a9d8f"   # applicant-side colour (teal)
C_EMP = "#e76f51"   # employer-side colour (coral)


def add_gain_columns(df):
    df = df.copy()
    df["applicant_proposer_gain"] = df["avg_applicant_rank_e_opt"] - df["avg_applicant_rank_a_opt"]
    df["employer_proposer_gain"] = df["avg_employer_rank_a_opt"] - df["avg_employer_rank_e_opt"]
    return df


# %% [markdown]
# ## Figure 1 — Baseline: avg rank for each side under each variant

# %%
b = pd.read_csv(RESULTS_DIR / "baseline.csv")
b = add_gain_columns(b)

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
colors = [C_APP, C_APP, C_EMP, C_EMP]
hatches = ["", "//", "", "//"]

fig, ax = plt.subplots(figsize=(7, 4))
bars = ax.bar(labels, means, yerr=stds, color=colors, alpha=0.85, capsize=4,
              edgecolor="black", linewidth=0.4)
for bar, h in zip(bars, hatches):
    bar.set_hatch(h)

ax.set_ylabel("Average partner rank (1 = best)")
ax.set_title(f"Baseline: balanced market (n=50), K={len(b)} trials")
# annotate proposer gains
app_gain = b["applicant_proposer_gain"].mean()
emp_gain = b["employer_proposer_gain"].mean()
ax.text(0.5, max(means) * 1.05, f"applicant proposer gain = {app_gain:+.2f}", ha="center", fontsize=9, color=C_APP)
ax.text(2.5, max(means) * 1.05, f"employer proposer gain = {emp_gain:+.2f}", ha="center", fontsize=9, color=C_EMP)

fig.tight_layout()
fig.savefig(FIG_DIR / "fig_baseline.png")
plt.show()

print(f"Baseline: applicant proposer gain = {app_gain:+.3f} (std {b['applicant_proposer_gain'].std():.3f})")
print(f"Baseline: employer proposer gain  = {emp_gain:+.3f} (std {b['employer_proposer_gain'].std():.3f})")

# %% [markdown]
# ## Figure 2 — Scaling: proposer gain vs market size

# %%
s = pd.read_csv(RESULTS_DIR / "scaling.csv")
s = add_gain_columns(s)

agg = s.groupby("size_a").agg(
    a_mean=("applicant_proposer_gain", "mean"),
    a_std=("applicant_proposer_gain", "std"),
    e_mean=("employer_proposer_gain", "mean"),
    e_std=("employer_proposer_gain", "std"),
).reset_index()

fig, ax = plt.subplots(figsize=(7, 4))
ax.errorbar(agg["size_a"], agg["a_mean"], yerr=agg["a_std"], marker="o",
            label="Applicant proposer gain (A-opt vs E-opt)", color=C_APP, capsize=4)
ax.errorbar(agg["size_a"], agg["e_mean"], yerr=agg["e_std"], marker="s",
            label="Employer proposer gain (E-opt vs A-opt)", color=C_EMP, capsize=4)
ax.axhline(0, color="gray", linewidth=0.5)
ax.set_xlabel("Market size (each side)")
ax.set_ylabel("Proposer rank gain (rank units)")
ax.set_title("How proposer advantage scales with market size")
ax.legend(fontsize=8, loc="upper left")
ax.set_xticks(agg["size_a"])
fig.tight_layout()
fig.savefig(FIG_DIR / "fig_scaling.png")
plt.show()

print("Scaling: proposer gains by market size")
print(agg.round(3).to_string(index=False))

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
ax.plot(agg["size_a"], agg["a_a"], "o-",  label="Applicant rank (A-opt)", color=C_APP)
ax.plot(agg["size_a"], agg["a_e"], "o--", label="Applicant rank (E-opt)", color=C_APP, alpha=0.6)
ax.plot(agg["size_a"], agg["e_a"], "s-",  label="Employer rank (A-opt)", color=C_EMP)
ax.plot(agg["size_a"], agg["e_e"], "s--", label="Employer rank (E-opt)", color=C_EMP, alpha=0.6)
ax.axvline(30, color="gray", linewidth=0.5, linestyle=":")
ax.text(30.3, ax.get_ylim()[1] * 0.95, "balanced", fontsize=8, color="gray")
ax.set_xlabel("Number of applicants (employers fixed at 30)")
ax.set_ylabel("Average partner rank")
ax.set_title("Effect of market imbalance on average rank")
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(FIG_DIR / "fig_imbalance.png")
plt.show()

print("Imbalance: mean ranks by applicant-side size")
print(agg.round(3).to_string(index=False))

# %% [markdown]
# ## Figure 4 — Preference correlation: proposer gain by quartile

# %%
c = pd.read_csv(RESULTS_DIR / "correlation.csv")
c = add_gain_columns(c)

# 'quartile' was saved as a string; preserve order.
order = ["Q1 (low)", "Q2", "Q3", "Q4 (high)"]
c["quartile"] = pd.Categorical(c["quartile"], categories=order, ordered=True)

agg = c.groupby("quartile", observed=True).agg(
    app_mean=("applicant_proposer_gain", "mean"),
    app_std=("applicant_proposer_gain", "std"),
    emp_mean=("employer_proposer_gain", "mean"),
    emp_std=("employer_proposer_gain", "std"),
).reset_index()

x = np.arange(len(agg))
width = 0.38

fig, ax = plt.subplots(figsize=(7, 4))
ax.bar(x - width/2, agg["app_mean"], width, yerr=agg["app_std"],
       label="Applicant proposer gain", color=C_APP, alpha=0.85, capsize=4)
ax.bar(x + width/2, agg["emp_mean"], width, yerr=agg["emp_std"],
       label="Employer proposer gain", color=C_EMP, alpha=0.85, capsize=4)
ax.set_xticks(x)
ax.set_xticklabels(agg["quartile"].astype(str))
ax.set_xlabel("Similarity-matrix concentration quartile")
ax.set_ylabel("Proposer rank gain")
ax.set_title(f"Proposer advantage vs preference correlation (n=50, K={len(c)})")
ax.axhline(0, color="gray", linewidth=0.5)
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(FIG_DIR / "fig_correlation.png")
plt.show()

print("Correlation: proposer gains by concentration quartile")
print(agg.round(3).to_string(index=False))

# %%
print(f"\nAll figures saved to {FIG_DIR.resolve()}")
