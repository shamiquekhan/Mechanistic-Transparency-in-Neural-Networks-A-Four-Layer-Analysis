# Experiment Raw Results and Outputs

## Phase 1: Geometric Transparency
- **Intrinsic Dimensionality (95% variance):**
  - Input: 152
  - Hidden: 13
  - Output: 6
- **Intrinsic Dimensionality (90% variance):**
  - Input: 87
  - Hidden: 10
  - Output: 5
- **Mean Curvature:**
  - Input: 12.60
  - Hidden: 1.30
  - Output: 0.81
- **Separability Ratio:**
  - Input: 0.75
  - Hidden: 1.53
  - Output: 2.20
- **Visualizations:**
  - phase1_training_curves.png
  - phase1_pca_variance.png
  - phase1_separability.png
  - phase1_umap_projections.png
  - phase1_summary_table.png

## Phase 2: Feature Transparency
- **Number of Neurons:** 16
- **Monosemantic Neurons:** 2
- **Polysemantic Neurons:** 14
- **Neuron Selectivity and Preferences:**
  - See phase2_feature_interpretability.json for full stats
- **Visualizations:**
  - phase2_neuron_preference_heatmap.png
  - phase2_selectivity_distribution.png
  - phase2_neuron_specialization_pie.png
  - phase2_neuron0_top_activations.png
  - phase2_neuron7_top_activations.png

## Phase 3: Circuit Analysis
- **See console output for top neurons and weights per digit class.**
- **No JSON output generated for this phase in current run.**

## Phase 4: Causal Interventions
- **Causal tracing complete. Activations saved for further analysis.**
- **Responsible layer: fc1. Surgical editing complete. Model saved as best_model_rome.pth.**
- **Validation:**
  - Overall accuracy: 90.22%
  - Accuracy on digit 7: 91.63%

## All Analysis Files
- geometric_properties.json
- phase2_feature_interpretability.json

## Documentation
- README.md
- EXPERIMENT_SUMMARY.md
- PHASE_OUTPUTS_INDEX.md

---

For full details, see the JSON files in the analysis/ folder and the PNGs in visualizations/.
