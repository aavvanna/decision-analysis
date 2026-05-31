# Stable Matching — Proposer Advantage Analysis

Interactive web app demonstrating the Gale-Shapley algorithm for stable matching in a job-applicant market, with comparative analysis of proposer advantage.

**Live demo:** https://[your-username].github.io/stable-matching/

## Project Description

This project analyzes structural differences between:
- **Applicant-Optimal** matching (applicants propose → best stable matching for applicants)
- **Employer-Optimal** matching (employers propose → best stable matching for employers)

### Features
- Synthetic job posting & resume data generation (skill vectors, cosine similarity)
- Gale-Shapley deferred acceptance algorithm (both variants)
- Bipartite graph visualization of matchings
- Statistical comparison: average rank satisfaction on both sides
- Stability verification (blocking pair detection)
- Preference list tables and similarity matrix

### Methods
- **Preference generation:** cosine similarity of skill vectors (binary/continuous, with Gaussian noise)
- **Algorithm:** Gale-Shapley O(n²)
- **Metrics:** Average rank (1=best), number of blocking pairs, pairs changed between matchings

## Deploy to GitHub Pages

1. Fork or clone this repo
2. Go to **Settings → Pages → Source → main branch / root**
3. Visit `https://[your-username].github.io/[repo-name]/`

No build step required — pure HTML/CSS/JS.

## Authors
- Vakulich Anastasia (БАСБ251)  
- Roshchina Nadezhda (БАСБ252)

НИУ ВШЭ · Decision Analysis Course · 2025
