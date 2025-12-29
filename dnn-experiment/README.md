# DNN Experiment: Mechanistic Interpretability

This repository contains a complete, four-phase pipeline for mechanistic interpretability of deep neural networks, as described in the 'Demystifying Deep Neural Networks' experiment. Each phase is implemented as a standalone script, with outputs and documentation for reproducibility and publication.

## Project Structure

- `data/` — Raw and processed datasets
- `models/` — Saved model checkpoints
- `visualizations/` — All generated figures (PNG)
- `analysis/` — All result summaries (JSON)
- `phase1_geometric_analysis.py` — Geometric untangling analysis
- `phase2_feature_interpretability.py` — Neuron feature analysis
- `phase3_circuit_analysis.py` — Circuit and ablation analysis
- `phase4_causal_interventions.py` — Causal tracing and model editing

## Phases Overview

### Phase 1: Geometric Analysis
- Measures how data manifolds are untangled across layers
- Outputs: PCA, UMAP, separability plots, summary table

### Phase 2: Feature Interpretability
- Analyzes what individual neurons encode
- Outputs: Neuron preference heatmaps, selectivity, polysemanticity

### Phase 3: Circuit Analysis
- Identifies modular circuits and neuron importance
- Outputs: Circuit diagrams, ablation results, saliency maps

### Phase 4: Causal Interventions
- Demonstrates causal tracing and surgical model editing
- Outputs: Before/after accuracy, update matrix, dashboard

## How to Run

1. Install dependencies:
   ```sh
   pip install torch torchvision numpy scipy scikit-learn matplotlib seaborn umap-learn networkx
   ```
2. Run each phase in order:
   ```sh
   python phase1_geometric_analysis.py
   python phase2_feature_interpretability.py
   python phase3_circuit_analysis.py
   python phase4_causal_interventions.py
   ```
3. All outputs will be saved in `visualizations/` and `analysis/`.

## Results
- All key results are summarized in `analysis/` as JSON files.
- Publication-ready figures are in `visualizations/`.

## Summary
This project proves that deep neural networks are transparent, interpretable, and controllable through systematic, mechanistic analysis.
