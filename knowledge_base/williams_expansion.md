# Williams Asymptotic Expansion for Crack-Tip Fields

## Category
Physics-informed feature engineering / near-tip basis enrichment

## Problem
Standard neural operators (DeepONet, Transolver) learn smooth mappings from data. The stress/displacement field near a crack tip has an r^{-1/2} singularity (Mode I opening), which standard basis functions cannot represent without explicit enrichment.

## Technique
Augment the trunk/coordinate input with Williams expansion terms:
```
σ_ij ≈ K_I / √(2πr) * f_ij(θ) + K_II / √(2πr) * g_ij(θ) + T * δ_ij + ...
```
Encode as additional input features: `(1/√r, cos(θ/2), sin(θ/2), cos(3θ/2), sin(3θ/2))`.
`r` = distance from tip, `θ` = angle w.r.t. crack plane.

## When to Apply
- Test loss plateaus while near-tip error remains high
- Ensemble std is large only in the near-tip zone (r < 0.1·a, where a = crack half-length)
- Physicist reports `stress_intensity` or `near_tip` terms have negligible effect

## Implementation Notes
- Compute `(r, θ)` relative to current crack tip position, not a fixed point
- Use `(sin θ/2, cos θ/2)` instead of `θ` directly to avoid branch-cut discontinuity
- Combine with tip-weighted MSE so the singularity is not averaged away during training
- `r_min` cutoff of ~1e-4 avoids division by zero

## Expected Impact
2–5× reduction in near-tip L2 error. Negligible cost (feature engineering only).

## References
Williams (1957), J. Applied Mechanics; Bažant & Planas (1998) Ch. 3.
