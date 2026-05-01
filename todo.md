# TODO

## Resolved Issues

### ✅ 1. Surrogate overfitting with small datasets
**Status**: Resolved  
**Fix applied**:
- Switched training target from displacement (output_dim=2) to nodal von Mises stress (output_dim=1) to halve the output space
- Test loss improved ~4× once DeepONet was also introduced
- **Superseded by issue #2 resolution**: output is now displacement (N, 2) again to enable PINO, with tip-weighted MSE handling the singularity instead of log-transform

### ✅ 2. Crack PINO physics loss / displacement incompatibility
**Status**: Resolved  
**Root cause**: `CrackFractureLoss` and `PINOElasticityLoss` both require displacement `(N, 2)`, but surrogate was predicting von Mises scalar `(N, 1)`. Additionally, `use_pino` gate in `trainer.py` required `coord_dim == 2`, which failed silently when using 6-feature enriched trunk coordinates.  
**Fix applied**:
- Switched training target back to displacement `(N, 2)` in `_generate_vnotch_fem_data` (`output_field="displacement"`)
- Removed `coord_dim == 2` constraint from `use_pino` gate in `trainer.py`
- Trainer now slices `coords_t[0, :, :2]` before passing to both `PINOElasticityLoss` and `CrackFractureLoss`, so 6-feature enriched coordinates work correctly
- `pino_eq_weight=0.1` (default) activates the label-free equilibrium residual from round 1
- Von Mises derived from predicted displacement at evaluation time via `_compute_von_mises_nodal`

### ✅ 3. Mock critic always diagnoses UNDERFITTING
**Status**: Resolved (approach changed)  
**Original fix**: Added `_analyze_heuristic()` fallback with ratio-based overfitting check  
**Current state**: Heuristic fallback has been **removed entirely**. `HyperparameterCriticAgent.analyze_training()` now raises `RuntimeError` if no LLM provider is set. The mock LLM provider is used for tests and demo; the real `AnthropicProvider` is used in production via `--use-real-llm`.  
**Remaining**: Mock LLM for critic still returns UNDERFITTING every round regardless of actual training curves — acceptable for tests but means the demo agent loop does not switch strategy when overfitting begins. Fixed only when running with `--use-real-llm`.

### ✅ 4. Triangulation includes notch interior (Delaunay artifact)
**Status**: Resolved  
**Fix applied**:
- Added `elements: Optional[np.ndarray] = None` to `FEMSample`
- `generate_vnotch_fem_sample` populates `elements` from `mesh_gen.generate()` — already filters notch interior
- Demo uses `sample.elements` instead of `Delaunay(coords).simplices`
- `FEMDataset.save/load` handles `elements.npy` per sample

### ✅ 5. AnthropicProvider uses outdated Claude 3 models
**Status**: Resolved  
**Fix applied**:
- Updated to `claude-opus-4-7`, `claude-sonnet-4-6`, `claude-haiku-4-5-20251001`
- Default changed to `claude-haiku-4-5-20251001`
- `--use-real-llm` CLI flag wires `AnthropicProvider` to all three agents

### ✅ 6. DeepONet has no uncertainty quantification in ensemble mode
**Status**: Resolved  
**Fix applied**:
- `EnsembleModel.build()` calls `torch.manual_seed(42 + i)` before each member so weights are uncorrelated

### ✅ 7. Surrogate predicts flat/uniform stress field (misses singularity)
**Status**: Resolved  
**Root cause**: MSE loss minimised by predicting the spatial mean when the singularity (σ ∝ 1/√r) dominates the variance.  
**Fix applied**:
- Polar trunk features `(r, log_r, sin_θ, cos_θ)` relative to notch tip give the trunk explicit geometric signal
- Tip-weighted MSE (`tip_weight=2.0`) upweights near-tip nodes so the loss cannot ignore the singularity
- (Log-transform applied to von Mises targets was part of a previous approach; current displacement-based training uses tip weighting instead)

### ✅ 8. Surrogate shows swirling artifacts in far-field
**Status**: Resolved  
**Root cause**: (a) Raw `θ = atan2(dy, dx)` has a ±π branch-cut discontinuity → trunk maps it to a spatial jump in basis functions. (b) Trunk overfits spatially with no dropout.  
**Fix applied**:
- Replaced `arctan2` with `(sin_theta, cos_theta)` in `coords_enriched`
- Added `trunk_dropout: float = 0.1` to `DeepONetConfig` — trunk MLP uses it; branch uses `dropout` (0.0)

### ✅ 9. Heuristic fallback in HyperparameterCriticAgent
**Status**: Resolved  
**Fix applied**:
- Removed `_analyze_heuristic()` from `HyperparameterCriticAgent`
- `analyze_training()` now raises `RuntimeError` if `_llm_provider is None`
- All three agents (Critic, Architect, Physicist) now require `set_llm_provider()` before use
- `detect_issues_heuristic()` kept for lightweight gating in `should_trigger_hpo()` — not an LLM substitute

### ✅ 10. `trunk_dropout` not tunable by Architect agent
**Status**: Resolved  
**Root cause**: Three-layer gap — `trunk_dropout` was in `DeepONetConfig` but (a) absent from the LLM system prompt, (b) not parsed by `_parse_changes()`, and (c) not forwarded by `apply_changes()`.  
**Fix applied**:
- `ARCHITECT_SYSTEM`: added `trunk_dropout` entry with explanation of branch vs trunk regularisation roles
- `_parse_changes()`: added `'trunk_dropout'` to `float` patterns
- `apply_changes()` DeepONet branch: added `trunk_dropout=changes.get("trunk_dropout", base.get("trunk_dropout", 0.1))`
- Output format template updated so the LLM knows to emit the field

### ✅ 11. `pytest-asyncio` not installed
**Status**: Resolved  
**Fix applied**:
- Installed `pytest-asyncio` — was in `requirements.txt` but missing from environment
- All 23 async agent tests now pass

---

## Open Issues

### A. Mock LLM critic cannot detect regime shift (overfitting after round 2)
**Status**: Known limitation  
**Symptom**: From round 3 onwards the training loss is ~1e-4 while test loss spikes to 4–14×. The mock LLM for the critic returns UNDERFITTING every round regardless, so the agent keeps increasing model capacity instead of adding regularisation.  
**Impact**: Demo loop does not converge to the best model when using MockLLMProvider.  
**Fix**: Use `--use-real-llm` with `ANTHROPIC_API_KEY` set. The real critic reads the train/test gap and correctly diagnoses OVERFITTING, allowing the Architect to respond with dropout/weight-decay increases.

### B. CrackFractureLoss weights still at 0.0
**Status**: Known limitation  
**Context**: `ki_weight`, `bc_weight`, `williams_weight`, `j_weight` are all 0.0 in the demo. The surrogate now predicts displacement `(N, 2)`, so the losses are technically compatible — the unit scale mismatch (O(1) normalised prediction vs O(1e12 Pa²) loss) is the remaining blocker.  
**Proposed fix**: Normalise the crack loss terms by E² so they are dimensionless and comparable to the data loss, then expose the weights to the Physicist agent for tuning.  
**Files to change**:
- `piano/surrogate/crack_pino_loss.py`: divide each term by `E²` to make it dimensionless
- `tests/test_agentic_sciml.py`: set non-zero initial crack weights once scale is fixed
- `piano/agents/roles/physicist.py`: add crack loss weights to the tunable parameter list
