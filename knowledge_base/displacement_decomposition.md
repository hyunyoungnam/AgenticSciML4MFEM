# Symmetric/Antisymmetric Displacement Decomposition

## Category
Architecture / output representation

## Problem
For Mode I (opening) dominated fracture in symmetric geometries, the displacement field has a natural decomposition: u_x is antisymmetric across the crack plane, u_y is symmetric. Training a single DeepONet/Transolver to predict both components jointly forces the model to discover this symmetry from data, which increases the effective learning difficulty.

## Technique
Decompose outputs explicitly:
- u_symmetric (u_y, Mode I): use symmetric trunk features
- u_antisymmetric (u_x, Mode II): use antisymmetric trunk features (multiply by H(y) Heaviside)

Or equivalently: train two separate branch/trunk networks for the symmetric and antisymmetric components. The final prediction is:
```
u_x = u_antisymmetric * H(y - y_crack)
u_y = u_symmetric
```

## When to Apply
- Geometry is symmetric about the crack plane (V-notch centered at y=0, uniform traction)
- Analyst reports far-field swirling artifacts despite sin/cos θ trunk features
- mode-II indicator (u_x asymmetry) is much smaller than mode-I (< 5% in the dataset)

## Implementation Notes
- Requires knowing the crack plane orientation at architecture build time
- Adds ~2× parameters compared to a single shared network — monitor for capacity explosion
- Incompatible with mixed-mode loading where Mode II is significant
- Alternative: use symmetry as a soft regularizer rather than a hard constraint

## Expected Impact
15–25% reduction in far-field reconstruction error when geometry is truly symmetric.

## References
Raju & Newman (1979) Eng. Fracture Mech.; Fan et al. (2023) Comput. Methods Appl. Mech. Eng.
