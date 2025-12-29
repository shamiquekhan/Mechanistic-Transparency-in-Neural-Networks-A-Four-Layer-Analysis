# DNN-Experiment-Build-Guide.md

## Complete Step-by-Step Experiment Build Guide

### Demystifying Deep Neural Networks Through Mechanistic Analysis

---

## Table of Contents
- Overview & Setup
- Phase 1: Geometric Analysis

---
## Overview & Setup

### What You're Building
A complete mechanistic analysis framework for neural networks that tests four core hypotheses:
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
  │   └── checkpoints/
  │   ├── geometric/
  │   ├── features/
  └── experiment.py
  ```
## Phase 1: Geometric Analysis
- Build a minimal MNIST network (784→16→10)
## Phase 2: Feature Interpretability
- Analyze per-neuron activation patterns across digit classes
## Phase 3: Circuit Analysis
- Extract weight importance patterns for each classification task

- Inject a backdoor (misclassify 5→3)
- Use causal tracing to locate the wrong knowledge
- Apply rank-one surgical edits to fix the misclassification
- Measure edit specificity and side effects

## Validation & Results Analysis
- Generate comprehensive JSON results report
- Create a 7-panel analytical dashboard

---
4. Document results in a Jupyter notebook with markdown explanations.

---

## Key Features for Your Profile
- Production-grade code with docstrings and error handling
- Reproducible experiments with fixed seeds
- Professional visualizations suitable for portfolio/papers
- Extensible framework to scale to CIFAR-10, Transformers, etc.
- Interpretability focus aligned with AI/ML interests


## Expected Results & Interpretations
| Circuits        | Identifiable weight patterns                  | Computation is modular & composable         |
| Polysemanticity | Some neurons multi-task                       | High-dimensional feature compression        |

- Test on convolutional architectures
- Analyze Transformer attention patterns
- Create causal graphs of information flow

---

## Portfolio Tips
- Reproducibility: Follow this protocol on MNIST, then extend to CIFAR-10 or Tiny ImageNet
- Documentation: Write detailed notebook with explanations at each step
- GitHub: Push complete project with results, visualizations, and detailed README

---

Start with Phase 1-2 on MNIST, then expand to Phase 3-4 as complexity increases.
