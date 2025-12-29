# Phase 4 Quick Start

## Production Code (3 Functions)

1. `causal_trace_layers(clean_model, corrupted_model, test_loader, device)`
2. `compute_rome_update(model, corrupted_model, train_loader, subject_class, target_class, device)`
3. `validate_rome_edit(clean_model, corrupted_model, delta_W, test_loader, device)`

---

## Usage
```python
# Step 1: Causal tracing
bug_location = causal_trace_layers(clean_model, corrupted_model, test_loader, device)

# Step 2: ROME computation
delta_W, info = compute_rome_update(clean_model, corrupted_model, train_loader, 7, 7, device)

# Step 3: Validation
results = validate_rome_edit(clean_model, corrupted_model, delta_W, test_loader, device)
```

---

## See scripts:
- phase4_causal_tracing.py
- phase4_surgical_editing.py
- phase4_validation.py
