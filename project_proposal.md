# Theme

Comparative Analysis of Proposer Advantage in Stable Matching: A Job-Applicant Market Application

# Goals, data, methods, models

**Goals:** The primary goal of this project is to analyze the structural differences, trade-offs, and fairness
outcomes between different matching mechanisms in a two-sided decision market. We aim to quantify the
mathematical proposer advantage by directly comparing the outcomes of an Applicant-proposing system
versus an Employer-proposing system. As an extension we may also explore how these dynamics shift when
introducing capacity constraints (many-to-one matching) or ties in the preference lists.

**Data:** We will use a publicly available tabular dataset (sourced from Kaggle or GitHub) containing Job
Postings (required skills, industry) and Applicant Resumes (candidate skills, education). To ensure a
computationally manageable bipartite graph, we will work with a controlled subset of this data.

**Methods & Models:**
- Preference Generation: We will use basic similarity metrics (e.g., cosine similarity of text
features/skills) to generate strict, ranked preference lists for both sides of the market (Applicants
ranking Jobs, and Jobs ranking Applicants).
- Models: The problem will be modeled as a Bipartite Graph, utilizing the Gale-Shapley algorithm to
find stable matchings.
- Comparative Analysis: We will execute both the Applicant-optimal and Employer-optimal variants
of the algorithm. We will analyze the resulting graphs to verify stability (the absence of blocking
pairs) and statistically compare the average rank satisfaction across both sides to measure the
magnitude of the proposer advantage.

# Group members
- Вакулич Анастасия / Vakulich Anastasia, БАСБ251
- Рощина Надежда / Roshchina Nadezhda, БАСБ252