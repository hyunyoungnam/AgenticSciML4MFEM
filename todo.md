# TODO

## Open Issues

### 1. Surrogate overfitting with small datasets
**Status**: Unresolved  
**Symptom**: With 30 FEM samples (24 train / 6 test), train loss ≈ 0.035 but test loss ≈ 4.25 — massive overfitting. Surrogate predicts near-zero von Mises everywhere despite decent training loss.  
**Root cause**: Transolver trained on displacement vector (N=409 nodes, output_dim=2 → 818 outputs) has too many degrees of freedom relative to 24 training samples.  
**Proposed fix**: Train surrogate directly on **nodal von Mises stress** (output_dim=1, scalar field) computed via `_compute_von_mises_nodal` from MFEM displacement. This halves the output dimension, removes the derived-quantity step at inference, and gives the model a physically interpretable scalar target it can learn with small data. PINO elasticity loss (which requires output_dim≥2) is dropped, but it was not active at this data scale anyway.  
**Files to change**:
- `tests/test_agentic_sciml.py`: add `"von_mises_nodal"` output mode in `_generate_vnotch_fem_data`; switch `output_field` in `run_agentic_loop_demo`; remove `_compute_von_mises_nodal` call at inference (prediction is already von Mises)

---

### 2. Crack PINO physics loss scale mismatch
**Status**: Known limitation (losses disabled in demo)  
**Symptom**: `CrackFractureLoss` terms (K_I consistency, crack face BC, Williams residual, J-integral) operate in physical units (Pa, Pa·m^0.5). Their scale is O(1e12) compared to data loss O(1) in normalized space. Enabling them immediately destroys training.  
**Proposed fix**: Normalize each crack loss term by the expected magnitude (e.g., divide K_I loss by `K_I²`, J-integral loss by `(K_I²/E)²`) so all terms are dimensionless and O(1). Then enable them gradually after round 1 once the model has a reasonable baseline.  
**Files to change**:
- `piano/surrogate/crack_pino_loss.py`: add normalization inside each loss term
- `tests/test_agentic_sciml.py`: re-enable `ki_weight`, `bc_weight`, etc. in demo after round 1

---

### 3. Mock critic always diagnoses UNDERFITTING
**Status**: Demo limitation  
**Symptom**: `MockLLMProvider(scenario="underfitting")` always returns the same critique regardless of actual train/test loss ratio. Round 2 onwards had train≈0.035, test≈4.25 (clear overfitting), but critic still said UNDERFITTING, so architect kept increasing model size — making things worse.  
**Proposed fix**: In the demo, use heuristic critic (no LLM) instead of mock: `HyperparameterCriticAgent` already has `_detect_issues_heuristic()` which correctly identifies overfitting when test >> train. Remove `MockLLMProvider` from `run_agentic_loop_demo` and let the critic work without LLM.  
**Files to change**:
- `tests/test_agentic_sciml.py`: remove `provider = MockLLMProvider(...)` / `critic.set_llm_provider(provider)` calls in demo; critic will fall back to heuristic mode

---

### 4. Triangulation includes notch interior (Delaunay artifact)
**Status**: Partially resolved  
**Symptom**: Delaunay triangulation on MFEM node coordinates creates triangles that span the V-notch cavity. Filtering by `VNotchGeometry.is_inside_notch(centroid)` removes most but the white `ax.fill` overlay hides residual artifacts.  
**Proposed fix**: Use the MFEM mesh element connectivity directly (from `mesh_manager.get_elements()`) instead of recomputing Delaunay. Requires returning elements alongside node coordinates from `generate_vnotch_fem_sample` (e.g., add `elements` field to `FEMSample`).  
**Files to change**:
- `piano/data/dataset.py`: add optional `elements: Optional[np.ndarray] = None` to `FEMSample`
- `piano/data/fem_generator.py`: populate `elements` in `generate_vnotch_fem_sample` from `mesh_manager.get_elements()`
- `tests/test_agentic_sciml.py`: use `sample.elements` instead of Delaunay in `_generate_vnotch_fem_data`
