# DNN-Experiment-Build-Guide.md

## Complete Step-by-Step Experiment Build Guide

### Demystifying Deep Neural Networks Through Mechanistic Analysis

---

## Table of Contents
- Overview & Setup
- Phase 1: Geometric Analysis
- Phase 2: Feature Interpretability
- Phase 3: Circuit Analysis
- Phase 4: Causal Interventions & Model Editing
- Validation & Results Analysis

---

## Overview & Setup

### What You're Building
A complete mechanistic analysis framework for neural networks that tests four core hypotheses:
- **Geometric Hypothesis:** Networks untangle data manifolds layer-by-layer
- **Feature Hypothesis:** Individual neurons encode interpretable concepts
- **Circuit Hypothesis:** Identifiable sub-components implement specific algorithms
- **Causal Hypothesis:** Network behavior can be surgically edited without side effects

### Prerequisites & Environment
- **Required Libraries:**
  ```bash
  pip install torch torchvision numpy matplotlib scikit-learn scipy jupyter
  pip install umap-learn networkx
  ```
- **Directory Structure:**
  ```text
  dnn-experiment/
  ├── data/
  │   └── mnist/
  ├── models/
  │   ├── trained_networks/
  │   └── checkpoints/
  ├── analysis/
  │   ├── geometric/
  │   ├── features/
  │   ├── circuits/
  │   └── causal/
  ├── visualizations/
  ├── results.json
  └── experiment.py
  ```
- **Hardware:** CPU is sufficient, but GPU (CUDA) is recommended for speed.

---

## Phase 1: Geometric Analysis
- Build a minimal MNIST network (784→16→10)
- Train with validation tracking
- Measure manifold properties: intrinsic dimensionality, curvature, class separability
- Visualize data untangling through layers using UMAP

## Phase 2: Feature Interpretability
- Analyze per-neuron activation patterns across digit classes
- Detect which digits each neuron "prefers"
- Identify polysemantic neurons (multi-concept encoding)
- Create selectivity heatmaps showing neuron preferences

## Phase 3: Circuit Analysis
- Extract weight importance patterns for each classification task
- Identify class-specific sub-circuits in the network
- Compute gradient-based feature importance maps
- Visualize which pixels drive predictions

## Phase 4: Causal Interventions & Model Editing
- Inject a backdoor (misclassify 5→3)
- Use causal tracing to locate the wrong knowledge
- Apply rank-one surgical edits to fix the misclassification
- Measure edit specificity and side effects

## Validation & Results Analysis
- Generate comprehensive JSON results report
- Create a 7-panel analytical dashboard
- Summary table interpreting findings

---

## How to Use This Guide
1. Copy the entire file and save as `experiment.py`.
2. Run Phase 1-2 first on CPU to validate setup (~15 min).
3. Extend with Phase 3-4 as you gain confidence.
4. Document results in a Jupyter notebook with markdown explanations.
5. Push to GitHub with visualizations and detailed README.

---

## Key Features for Your Profile
- Production-grade code with docstrings and error handling
- Reproducible experiments with fixed seeds
- Professional visualizations suitable for portfolio/papers
- Extensible framework to scale to CIFAR-10, Transformers, etc.
- Interpretability focus aligned with AI/ML interests

---

## Expected Results & Interpretations
| Finding         | What It Means                                 | Implication                                 |
|-----------------|-----------------------------------------------|---------------------------------------------|
| Geometric       | Intrinsic dim 784→16                          | Networks efficiently compress representations|
| Features        | Mean selectivity > 0.3                        | Individual units are interpretable          |
| Circuits        | Identifiable weight patterns                  | Computation is modular & composable         |
| Causal          | Surgical edit 100% success                    | Network knowledge is localized & editable   |
| Polysemanticity | Some neurons multi-task                       | High-dimensional feature compression        |

---

## Extensions
- Test on convolutional architectures
- Implement full MEMIT (Multi-layer mass editing)
- Analyze Transformer attention patterns
- Create causal graphs of information flow

---

## Portfolio Tips
- Reproducibility: Follow this protocol on MNIST, then extend to CIFAR-10 or Tiny ImageNet
- Documentation: Write detailed notebook with explanations at each step
- GitHub: Push complete project with results, visualizations, and detailed README

---

Start with Phase 1-2 on MNIST, then expand to Phase 3-4 as complexity increases.
