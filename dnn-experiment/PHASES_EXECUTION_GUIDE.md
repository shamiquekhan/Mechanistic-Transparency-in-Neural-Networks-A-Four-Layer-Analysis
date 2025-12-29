# Phases Execution Guide

This guide provides step-by-step instructions for running and validating each phase of the DNN interpretability experiment.

## 1. Environment Setup
- Install Python 3.11+
- Install dependencies:
  ```sh
  pip install torch torchvision numpy scipy scikit-learn matplotlib seaborn umap-learn networkx
  ```
- Ensure the following folders exist: `data/`, `models/`, `visualizations/`, `analysis/`

## 2. Phase 1: Geometric Analysis
- Run: `python phase1_geometric_analysis.py`
- Outputs: PCA, UMAP, separability plots, summary table (see `visualizations/`)
- JSON: `analysis/geometric_properties.json`

## 3. Phase 2: Feature Interpretability
- Run: `python phase2_feature_interpretability.py`
- Outputs: Neuron heatmaps, selectivity, polysemanticity (see `visualizations/`)
- JSON: `analysis/phase2_feature_interpretability.json`

## 4. Phase 3: Circuit Analysis
- Run: `python phase3_circuit_analysis.py`
- Outputs: Circuit diagrams, ablation, saliency (see `visualizations/`)
- JSON: `analysis/phase3_circuit_analysis.json` (if generated)

## 5. Phase 4: Causal Interventions
- Run: `python phase4_causal_interventions.py`
- Outputs: Before/after accuracy, update matrix, dashboard (see `visualizations/`)
- JSON: `analysis/phase4_causal_interventions.json` (if generated)

## 6. Validation Checklist
- After each phase, verify all expected PNG and JSON files are present (see `PHASE_OUTPUTS_INDEX.md`).
- Review figures for expected patterns and results.

## 7. Documentation
- See `README.md` for project overview.
- See `EXPERIMENT_SUMMARY.md` for executive summary.
- See `QUICK_REFERENCE_TABLES.md` for metrics and phase comparison.

## 8. Troubleshooting
- If any script fails, check Python and package versions.
- Ensure NumPy is <2.0 for compatibility with some dependencies.
- Re-run scripts after fixing any errors.
