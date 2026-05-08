# XFEM-Style Discontinuity Enrichment for Neural Operators

## Category
Architecture enrichment / discontinuity handling

## Problem
Mesh-based FEM captures the crack face traction-free boundary condition (σ·n = 0) exactly because elements align with the crack. Neural operators operating on nodal data inherit this geometry from the mesh, but the loss function may not enforce it — resulting in predicted tractions that violate physics near the crack face.

## Technique
Heaviside enrichment: for each node, compute a sign function based on which side of the crack plane it lies on:
```
H(x) = +1  if above crack plane
H(x) = -1  if below crack plane
```
Append `H(x)` as an additional trunk feature so the network can represent a discontinuous field across the crack faces.

## When to Apply
- Traction-free BC loss (`bc_weight` term in CrackFractureLoss) is nonzero but not decreasing
- Predicted displacement field shows smooth rather than discontinuous opening near crack faces
- Williams expansion is active but `K_II` (mode II) indicator is unexpectedly large

## Implementation Notes
- Only sensible when the trunk has explicit `(x, y)` coordinate inputs
- Sign is `np.sign(y - y_crack)` for a horizontal crack; generalize to `np.sign((x-x_tip)·n_perp)` for inclined cracks
- For a V-notch, use the notch half-angle to define the two crack faces
- Pair with CrackFractureLoss `traction_free` term active

## Expected Impact
Reduces BC violation loss by 3–10× after 50 epochs when the BC term was previously stuck.

## References
Moës et al. (1999) IJNME; Sukumar & Prévost (2003) IJNME.
