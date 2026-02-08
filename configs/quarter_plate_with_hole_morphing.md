# Quarter plate with hole – morphing configuration

Region-based morphing for changing hole radius while preserving mesh topology.
All node roles (moving, anchor, morphing) are assigned from **geometric rules** only; no set names from the .inp are used.

---

## Configuration (YAML)

```yaml
problem_type: quarter_plate_with_hole

design_variable:
  name: hole_radius_change
  symbol: delta_R
  description: "Radial displacement of the hole boundary (positive = enlarge hole)."

geometry:
  hole_center: [0.0, 0.0]
  # Initial hole radius (length units). Set to match your mesh or infer from mesh.
  initial_hole_radius: 2.5
  # Outer radius of the transition band; beyond this nodes are anchors.
  transition_outer_radius: 8.0
  # Tolerance for classifying hole boundary: nodes with d in [R0 - tol, R0 + tol].
  tolerance: 0.15

reassignment:
  # Anchors closer than this to the hole boundary (distance R0) are reassigned to morphing.
  min_anchor_distance_from_hole: 0.5
  # Optional: minimum transition width in length units (transition_outer_radius - R0 >= this).
  min_transition_width: 2.0

regions:
  hole_boundary:
    physical_meaning: "Nodes on the hole boundary; they define the new hole size."
    role: moving
    assignment:
      rule: distance_to_center
      bounds: [R0 - tolerance, R0 + tolerance]
    displacement:
      type: radial
      magnitude: delta_R
      origin: hole_center
    idw_p: 4

  transition:
    physical_meaning: "Band around the hole where displacement decays smoothly."
    role: morphing
    assignment:
      rule: distance_to_center
      bounds: (R0 + tolerance, R_transition]
    idw_p: 2

  far_field:
    physical_meaning: "Fixed far field; remain fixed during morphing."
    role: anchor
    assignment:
      rule: distance_to_center
      bounds: "> R_transition"
    idw_p: null
```

---

## Parameter reference

- **geometry.hole_center**: Center of the hole (quarter symmetry: origin).
- **geometry.initial_hole_radius (R0)**: Current hole radius; hole boundary nodes lie at distance ≈ R0.
- **geometry.transition_outer_radius (R_transition)**: Nodes with distance > R_transition are anchors.
- **geometry.tolerance**: Half-width of the band that counts as "hole boundary" (distance in [R0−tol, R0+tol]).
- **regions.*.role**: `moving` (prescribed displacement), `anchor` (fixed), `morphing` (IDW interpolated).
- **regions.*.idw_p**: IDW exponent p for morphing nodes in that region (higher p = more local; lower p = smoother decay).
