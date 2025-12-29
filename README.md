# Demystifying Neural Networks: A Four-Phase Mechanistic Interpretability Study

A complete 4-phase mechanistic interpretability pipeline proving neural networks are transparent, interpretable, and controllable.

## Project Structure
- **dnn-experiment/**: All code, outputs, and documentation
- **DNN-Experiment-Build-Guide.md**: Step-by-step build and usage guide

## Phases
1. **Geometric Transparency**: Data untangling, dimensionality reduction, separability
2. **Feature Transparency**: Neuron interpretability, selectivity, polysemanticity
3. **Circuit Analysis**: Circuit modularity, sparsity, connection usage
4. **Causal Interventions**: Backdoor injection, causal tracing, surgical model editing (ROME)

## How to Run
1. Clone this repository
2. Install dependencies (see requirements in dnn-experiment/README.md)
3. Run each phase script in order:
   ```bash
   python dnn-experiment/phase1_geometric_analysis.py
   python dnn-experiment/phase2_feature_interpretability.py
   python dnn-experiment/phase3_circuit_analysis.py
   python dnn-experiment/phase4_causal_interventions.py
   # For advanced causal tracing and editing:
   python dnn-experiment/phase4_causal_tracing.py
   python dnn-experiment/phase4_surgical_editing.py
   python dnn-experiment/phase4_validation.py
   ```
4. Review outputs in `dnn-experiment/visualizations/` and `dnn-experiment/analysis/`

## Key Results
- Data untangles 3×, compresses to 6D
- 93.17% MNIST accuracy (clean)
- Most neurons are polysemantic, some monosemantic
- Circuits are sparse and modular
- Causal bugs can be injected and surgically fixed
- Backdoor bug (7→1) injected and surgically fixed (ROME)
- After edit: Digit 7 accuracy 91.63%, overall 90.22%, <0.3% of network changed, minimal side effects

## Documentation
- All results, outputs, and summaries are in the repo
- See `dnn-experiment/result.md` for raw outputs and per-phase details
- See `dnn-experiment/EXPERIMENT_SUMMARY.md` for executive summary and latest metrics

---

**For full details, see the documentation and code in the dnn-experiment folder.**
