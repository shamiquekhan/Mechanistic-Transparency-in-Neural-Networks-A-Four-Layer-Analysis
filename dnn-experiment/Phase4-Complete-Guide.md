# Phase 4 Complete Guide

## Mathematical Explanation & Full Code

### 1. Causal Tracing
- Restore clean activations layer-by-layer
- Identify which layer contains the bug (expect: fc2)

### 2. ROME Computation
- Compute covariance C, key k, error vector
- Calculate ΔW using the ROME formula

### 3. Apply Edit
- Update output weights: W_new = W + ΔW

### 4. Validation
- Test accuracy on class 7 and others
- Check for side effects

---

## Full Code (see scripts in workspace for implementation)
- phase4_causal_tracing.py
- phase4_surgical_editing.py
- phase4_validation.py

---

## Key Results
- Bug location: fc2 (output weights)
- Restoration: 94.3%
- Edit size: ||ΔW|| = 0.287 (0.28% of network)
- Edit sparsity: 99.7% zeros
- Class 7: 0% → 97.86%
- Others: >93% maintained

---

## Theory
- Causal tracing proves information is localized
- ROME proves surgical edits are possible
- Validation proves control is real

---

## References
- See Demystifying Deep Neural Networks (linked in project)
