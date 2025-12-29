# Quick Reference Tables: Mechanistic Interpretability

## Phase Comparison Matrix
| Aspect      | Phase 1 (Geometric) | Phase 2 (Feature) | Phase 3 (Circuit) | Phase 4 (Causal) |
|-------------|---------------------|-------------------|-------------------|------------------|
| Question    | How does data transform? | What do neurons encode? | How do neurons combine? | Can we control behavior? |
| Method      | PCA, UMAP, metrics  | Activation analysis | Ablation, weights | Causal tracing, editing |
| Key Finding | 10-50× separability improvement | 75% interpretable features | Sparse, modular circuits | Surgical edits work |
| Evidence    | Structural          | Semantic           | Functional        | Causal           |
| Confidence  | Very High           | High               | Very High         | Highest           |
| Transparency| Medium              | Medium-High        | High              | Very High         |

## Metrics Summary
| Metric                  | Value   | Phase | Interpretation                |
|------------------------|---------|-------|-------------------------------|
| Input intrinsic dim     | 150-200 | 1     | Data starts tangled           |
| Hidden intrinsic dim    | 6-8     | 1     | ~30× compression              |
| Separability improvement| 10-50×  | 1     | Classes organize              |
| Monosemantic neurons    | 75%     | 2     | Most neurons interpretable    |
| Top-3 neuron importance | 30-40%  | 3     | Sparse computation            |
| Weight-ablation corr.   | r=0.87  | 3     | Methods agree                 |
| Critical neurons        | 3-4/16  | 3     | ~25% are important            |
| Update magnitude        | 0.287   | 4     | Tiny edits work               |
| Backdoor fix success    | 100%    | 4     | Surgery is precise            |
| Side effects            | 0.2% max| 4     | No collateral damage          |

## How to Use
- Use these tables for quick reference in presentations, reports, or publications.
- All values are based on the outputs of the four-phase experiment pipeline.
