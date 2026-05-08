# J-Integral Consistency Loss for Fracture Surrogates

## Category
Physics loss / fracture mechanics constraint

## Problem
A surrogate can produce displacement fields with low MSE that are nonetheless physically inconsistent — the predicted J-integral (energy release rate) may differ substantially from the FEM reference. This matters for life-prediction applications where K_I or G is the key output.

## Technique
Add a J-integral consistency loss term:
```
L_J = (J_pred - J_ref)² / J_ref²
```
where:
- `J_ref` is computed from the FEM reference displacement via domain integral
- `J_pred` is computed from the surrogate's predicted displacement using the same formula
- Domain integral uses a contour around the crack tip (typically 3–5 element rings)

The J-integral for plane strain:
```
J = ∮_Γ (W n₁ - σ_{ij} u_{i,1} n_j) dΓ
W = ½ σ_{ij} ε_{ij}   (strain energy density)
```

## When to Apply
- Test MSE is low but K_I prediction error is > 5%
- `j_weight` term in CrackFractureLoss is inactive (0) but Williams expansion is already active
- Production use requires accurate fracture parameter prediction, not just field reconstruction

## Implementation Notes
- J-integral is contour-independent only in the absence of body forces and material gradients
- Implementation in `piano/surrogate/crack_pino_loss.py` — use the existing `_j_integral()` method
- Normalize by `K_I² / E` to make loss dimensionless
- Activate J after `stress_intensity` and `near_tip` are stable (Physicist's ordering)

## Expected Impact
10–30% improvement in K_I prediction accuracy at the cost of ~10% slower training per epoch.

## References
Rice (1968) J. Applied Mechanics; Shih & Asaro (1988) J. Appl. Mech.
