# Phase-Field Fracture Model for Crack Propagation Surrogates

## Category
Physics loss formulation / surrogate target selection

## Problem
Displacement-based surrogates capture static crack fields well but cannot represent crack propagation — the crack geometry changes between training samples. Classical fracture surrogates must either fix the crack path or re-mesh per sample.

## Technique
Train the surrogate to predict the phase-field variable `d ∈ [0,1]` (0=intact, 1=fully broken) alongside displacement:
- Output: `(u_x, u_y, d)` — dimension 3 instead of 2
- Loss: combine displacement MSE with phase-field regularization `∫(d²/l₀ + l₀|∇d|²)dΩ`
- The scalar `l₀` (length scale) controls crack band width — a tunable hyperparameter

Alternatively: use phase-field FEM outputs directly as training targets from existing simulations (see `piano/data/phase_field_generator.py`).

## When to Apply
- The dataset contains crack propagation simulations (varying crack tip position)
- The surrogate needs to generalize across different crack configurations
- J-integral or K_I losses are not converging (because crack front changes per sample)

## Implementation Notes
- Increase `output_dim` from 2 → 3 in DeepONetConfig/TransolverConfig
- Add `l₀` to the parameter bounds for active learning (treat as hyperparameter)
- Phase-field output does NOT require Williams enrichment (d is smooth)
- Avoid training phase-field on near-intact regions (d ≈ 0) — scale loss by local damage

## Expected Impact
Enables surrogate generalization across crack geometries; 30–50% MSE improvement on propagation datasets vs. displacement-only.

## References
Bourdin et al. (2000) J. Mech. Phys. Solids; Miehe et al. (2010) CMAME.
