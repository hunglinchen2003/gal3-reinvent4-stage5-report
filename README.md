# Galectin-3 AI Drug Design — Stage 1–4 Diagnosis & Stage 5 Commercial Phenylamine / 2F→NO₂ SAR

Interactive standalone HTML report and reproducible local GPU pipeline for **Galectin-3 CRD (PDB `5E89` chain A)** non-carbohydrate inhibitor optimization using **REINVENT 4.9.34**, **DynamicBind**, **smina (Vina / Vinardo)**, **GNINA v1.3.3 GPU CNN**, **PLIP**, and **ADMET-AI**.

## Live Interactive Report (GitHub Pages)
- **Standalone HTML**: `docs/index.html` (self-contained with embedded `3Dmol-min.js`, 2D RDKit renderings, interactive multi-generation charts, Pareto scatter plots, and 3D binding poses for 53 structures — works completely offline or on GitHub Pages).

## Repository Contents
- `docs/index.html` — Published standalone interactive HTML report.
- `data/` — Extracted PDB `5E89` chain A receptor, reference SDFs (`TD-139`, `GB1107`, `c310_15`, `c318_12`), and Stage 1–4 93-molecule dataset (`explorer_93_molecules.csv`).
- `stage5_aniline_sar/stage5_aniline_sar_results.csv` — Complete 40-molecule local benchmark (3-seed smina Vina & Vinardo at pH 7.4 + GNINA v1.3.3 GPU CNNscore/CNNaffinity + ADMET-AI + RDKit QED/SA/Brenk + CRD 7-residue contacts).
- `scripts/benchmark_aniline_sar.py` — Reproducible parallel docking + GNINA GPU CNN + ADMET-AI benchmark script.
- `scripts/build_standalone_report.py` — Standalone HTML report generator.
- `stage5_reinvent4/` — Ready-to-run REINVENT 4 Stage 5 multi-objective RL configuration (`stage5_staged_learning.toml`), custom ExternalProcess scorer (`stage5_scorer.py`), and seed SMILES (`stage5_seeds.smi`).

> **Scientific Note**: All docking and CNN affinity scores are computational proxies, not experimental $K_d$ values. Experimental validation (FP / SPR / ITC + Ames / microsomal stability) is required before advancing lead candidates.
