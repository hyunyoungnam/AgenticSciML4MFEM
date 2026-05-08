"""
Generate phase field fracture training dataset.

Produces N samples with varied crack geometry, material properties, and loading
using the AT-2 phase field model (DOLFINx). Saves as FEMDataset for surrogate training.

Usage:
    ~/miniforge3/envs/piano/bin/python scripts/generate_phase_field_data.py
    ~/miniforge3/envs/piano/bin/python scripts/generate_phase_field_data.py --n-samples 100 --resolution 35
"""

import argparse
import time
from pathlib import Path

import numpy as np
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from piano.data.phase_field_generator import (
    PhaseFieldFEMConfig,
    ParameterBounds,
    generate_phase_field_sample,
)
from piano.data.dataset import FEMDataset, FEMSample, DatasetConfig
from scipy.stats import qmc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-samples", type=int, default=50)
    parser.add_argument("--resolution", type=int, default=30,
                        help="Mesh resolution (nodes on longest edge)")
    parser.add_argument("--n-load-steps", type=int, default=20,
                        help="Number of load increments")
    parser.add_argument("--output-dir", type=str, default="phase_field_data",
                        help="Output directory relative to project root")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    project_root = Path(__file__).parent.parent
    output_dir = project_root / args.output_dir

    # Griffith critical traction for a 1m plate is ~10-37 MPa over this parameter range.
    # Using 5-25 MPa gives a mix of: elastic-only, damage onset, and partial fracture.
    # Griffith critical traction: sigma_c = K_Ic / (Y * sqrt(pi * a))
    # where K_Ic = sqrt(E * G_c / (1 - nu^2)), Y = 1.12 for edge crack
    # Set traction = sigma_c * load_ratio, load_ratio ~ U[0.6, 1.4]
    # → every sample is in the interesting near-critical regime
    E_range           = (150e9, 250e9)
    nu_range          = (0.25, 0.35)
    G_c_range         = (1000.0, 5000.0)
    crack_length_range = (0.2, 0.5)
    load_ratio_range   = (0.6, 1.4)   # sigma / sigma_c_griffith

    # Latin Hypercube Sampling for good parameter coverage (5 dims)
    sampler = qmc.LatinHypercube(d=5, seed=args.seed)
    lhs = sampler.random(n=args.n_samples)

    E_vals      = E_range[0]            + lhs[:, 0] * (E_range[1]            - E_range[0])
    nu_vals     = nu_range[0]           + lhs[:, 1] * (nu_range[1]           - nu_range[0])
    load_ratios = load_ratio_range[0]   + lhs[:, 2] * (load_ratio_range[1]   - load_ratio_range[0])
    G_c_vals    = G_c_range[0]          + lhs[:, 3] * (G_c_range[1]          - G_c_range[0])
    crack_vals  = crack_length_range[0] + lhs[:, 4] * (crack_length_range[1] - crack_length_range[0])

    # Compute per-sample critical traction and scale
    Y = 1.12  # edge crack geometry factor
    K_Ic_vals = np.sqrt(E_vals * G_c_vals / (1 - nu_vals**2))
    sigma_c_vals = K_Ic_vals / (Y * np.sqrt(np.pi * crack_vals))
    traction_vals = sigma_c_vals * load_ratios

    # ParameterBounds is only used for config metadata; traction_range is recorded as actual range
    bounds = ParameterBounds(
        E_range=tuple(E_range),
        nu_range=tuple(nu_range),
        traction_range=(float(traction_vals.min()), float(traction_vals.max())),
        G_c_range=tuple(G_c_range),
        crack_length_range=tuple(crack_length_range),
    )

    dataset_config = DatasetConfig(
        name="phase_field_fracture",
        parameter_names=["E", "nu", "traction", "G_c", "crack_length"],
        parameter_bounds={
            "E":            list(bounds.E_range),
            "nu":           list(bounds.nu_range),
            "traction":     list(bounds.traction_range),
            "G_c":          list(bounds.G_c_range),
            "crack_length": list(bounds.crack_length_range),
        },
        output_fields=["displacement", "damage", "von_mises"],
        storage_dir=output_dir,
    )
    dataset = FEMDataset(config=dataset_config)

    print(f"Generating {args.n_samples} phase field samples "
          f"(resolution={args.resolution}, load_steps={args.n_load_steps})")
    print(f"Output: {output_dir}")
    print()

    t_start = time.time()
    n_success = 0
    n_fail = 0

    for i in range(args.n_samples):
        cfg = PhaseFieldFEMConfig(
            geometry_type="edge_crack",
            crack_length=crack_vals[i],
            resolution=args.resolution,
            n_load_steps=args.n_load_steps,
        )
        t0 = time.time()
        sample = generate_phase_field_sample(
            E=E_vals[i],
            nu=nu_vals[i],
            traction=traction_vals[i],
            G_c=G_c_vals[i],
            config=cfg,
        )
        dt = time.time() - t0

        if sample is not None and sample.is_valid:
            dataset.add_sample(sample)
            n_success += 1
            elapsed = time.time() - t_start
            eta = elapsed / n_success * (args.n_samples - i - 1)
            max_dmg = sample.damage.max() if sample.damage is not None else 0.0
            print(f"  [{i+1:3d}/{args.n_samples}] OK  "
                  f"nodes={sample.coordinates.shape[0]:5d}  "
                  f"crack={crack_vals[i]:.3f}  "
                  f"trac={traction_vals[i]/1e6:.1f}MPa  "
                  f"ratio={load_ratios[i]:.2f}  "
                  f"dmg={max_dmg:.3f}  "
                  f"{dt:.1f}s  ETA {eta:.0f}s")
        else:
            n_fail += 1
            print(f"  [{i+1:3d}/{args.n_samples}] FAIL  crack={crack_vals[i]:.3f}")

    total = time.time() - t_start
    print(f"\nDone: {n_success} OK, {n_fail} failed  ({total:.1f}s total, "
          f"{total/max(1,n_success):.1f}s/sample)")

    # Save dataset
    save_path = dataset.save(output_dir)
    print(f"Saved to: {save_path}")

    # Quick sanity check
    from piano.data.dataset import FEMDataset as DS
    loaded = DS.load(save_path)
    s0 = loaded[0]
    print(f"\nVerification: {len(loaded)} samples loaded")
    print(f"  Sample 0: coords={s0.coordinates.shape}, "
          f"displacement={s0.displacement.shape if s0.displacement is not None else None}, "
          f"damage={s0.damage.shape if s0.damage is not None else None}")
    print(f"  Params: {s0.parameters}")


if __name__ == "__main__":
    main()
