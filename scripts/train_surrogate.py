"""
Train Transolver surrogate on the phase field fracture dataset.

Model: Transolver (d_model=64, n_layers=3, slice_num=16) — sized for 50 samples.
Inputs (per node): Williams-enriched coordinates [x, y, r, log_r, sinθ, cosθ, sin(θ/2), cos(θ/2)]
                   + 5 global parameters [E, nu, traction, G_c, crack_length] broadcast to each node
Outputs (per node): [u_x, u_y, von_mises, damage]  — 4 fields jointly

Usage:
    ~/miniforge3/envs/piano/bin/python scripts/train_surrogate.py
    ~/miniforge3/envs/piano/bin/python scripts/train_surrogate.py --epochs 500 --no-ensemble
    ~/miniforge3/envs/piano/bin/python scripts/train_surrogate.py --target displacement  # u_x, u_y only
"""

import argparse
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from piano.data.dataset import FEMDataset
from piano.surrogate.base import TransolverConfig
from piano.surrogate.trainer import SurrogateTrainer, TrainingConfig


# ── Williams coordinate enrichment ────────────────────────────────────────────

def enrich_coords(coords: np.ndarray, crack_tip_x: float, crack_y: float = 0.5) -> np.ndarray:
    """Enrich (N,2) mesh coordinates with Williams near-tip features.

    Returns (N, 8): [x, y, r, log_r, sinθ, cosθ, sin(θ/2), cos(θ/2)]
    where (r, θ) are polar coords centred at the crack tip.
    sin(θ/2) / cos(θ/2) encode the mode-I displacement discontinuity across θ=±π.
    """
    dx = coords[:, 0] - crack_tip_x
    dy = coords[:, 1] - crack_y
    r  = np.hypot(dx, dy).clip(1e-8)
    th = np.arctan2(dy, dx)
    r_col    = r[:, None]
    logr_col = np.log(r)[:, None]
    sin_th   = np.sin(th)[:, None]
    cos_th   = np.cos(th)[:, None]
    sin_half = np.sin(th / 2)[:, None]
    cos_half = np.cos(th / 2)[:, None]
    return np.concatenate([coords, r_col, logr_col, sin_th, cos_th, sin_half, cos_half], axis=1).astype(np.float32)


# ── Data loading ──────────────────────────────────────────────────────────────

def load_dataset(data_dir: str, target: str):
    """Load dataset and return (parameters, coordinates_list, outputs_list).

    target: 'displacement' | 'stress' | 'damage'

    Von Mises stress is log1p-transformed before training so the model works on
    a smooth log-scale rather than trying to fit a 4-order-of-magnitude spike at
    the crack tip.  The training script records this transform in metadata so
    predictions can be exp1m-inverted.
    """
    ds = FEMDataset.load(data_dir)
    samples = list(ds)
    print(f"Loaded {len(samples)} samples from {data_dir}")

    params_list = []
    coords_list = []
    outputs_list = []
    skipped = 0

    for s in samples:
        if s.displacement is None:
            skipped += 1
            continue

        if target == "stress" and s.von_mises is None:
            skipped += 1
            continue

        p = s.parameters
        param_vec = np.array([
            p["E"], p["nu"], p["traction"], p["G_c"], p["crack_length"]
        ], dtype=np.float32)

        crack_tip_x = p["crack_length"]
        enriched = enrich_coords(s.coordinates.astype(np.float32), crack_tip_x)

        if target == "displacement":
            out = s.displacement.astype(np.float32)                         # (N, 2)
        elif target == "stress":
            # log1p transform: collapses 4-decade spike to ~1-decade smooth field
            vm = np.log1p(s.von_mises[:, None].astype(np.float32))          # (N, 1)
            out = vm
        else:  # damage
            out = s.damage[:, None].astype(np.float32)                      # (N, 1)

        params_list.append(param_vec)
        coords_list.append(enriched)
        outputs_list.append(out)

    if skipped:
        print(f"  Skipped {skipped} incomplete samples")

    parameters = np.stack(params_list)
    output_dim = outputs_list[0].shape[1]
    print(f"  Parameters: {parameters.shape},  coord_dim: {coords_list[0].shape[1]},  output_dim: {output_dim}")
    return parameters, coords_list, outputs_list


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir",   default="phase_field_data")
    parser.add_argument("--save-dir",   default="surrogate_model")
    parser.add_argument("--target",     default="displacement",
                        choices=["displacement", "stress", "damage"])
    parser.add_argument("--epochs",     type=int,   default=300)
    parser.add_argument("--d-model",    type=int,   default=64)
    parser.add_argument("--n-layers",   type=int,   default=3)
    parser.add_argument("--slice-num",  type=int,   default=16)
    parser.add_argument("--dropout",    type=float, default=0.1)
    parser.add_argument("--lr",         type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int,   default=4)
    parser.add_argument("--no-ensemble", action="store_true")
    parser.add_argument("--n-ensemble", type=int,   default=3)
    parser.add_argument("--tip-weight", type=float, default=0.0,
                        help=">0 upweights nodes near crack tip by 1+w/r")
    parser.add_argument("--seed",       type=int,   default=42)
    args = parser.parse_args()

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # ── load data ─────────────────────────────────────────────────────────────
    parameters, coords_list, outputs_list = load_dataset(args.data_dir, args.target)
    n_samples = len(parameters)

    # Crack tip for tip-weighted MSE (use median crack length over dataset)
    median_crack = float(np.median(parameters[:, 4]))  # crack_length column
    tip_coords = np.array([median_crack, 0.5], dtype=np.float32)

    # ── model config ──────────────────────────────────────────────────────────
    cfg = TransolverConfig(
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=max(1, args.d_model // 16),  # head_dim=16
        slice_num=args.slice_num,
        mlp_ratio=2.0,
        dropout=args.dropout,
        learning_rate=args.lr,
        batch_size=args.batch_size,
        epochs=args.epochs,
        patience=args.epochs // 3,
        scheduler_type="cosine",
        optimizer_type="adamw",
        tip_weight=args.tip_weight,
    )

    n_params = sum(p.numel() for p in __import__('piano.surrogate.transolver',
                   fromlist=['TransolverModel']).TransolverModel(cfg).__class__(cfg).parameters()
                   if False) or None  # computed after build

    train_config = TrainingConfig(
        surrogate_config=cfg,
        use_ensemble=not args.no_ensemble,
        n_ensemble=args.n_ensemble,
        normalize_inputs=True,
        normalize_outputs=True,
        train_test_split=0.1,
        random_seed=args.seed,
        save_dir=save_dir,
        tip_coords=tip_coords if args.tip_weight > 0 else None,
    )

    print(f"\nTransolver config:")
    print(f"  d_model={args.d_model}, n_layers={args.n_layers}, "
          f"slice_num={args.slice_num}, n_heads={cfg.n_heads}")
    print(f"  dropout={args.dropout}, lr={args.lr}, epochs={args.epochs}")
    print(f"  ensemble={'off' if args.no_ensemble else f'{args.n_ensemble} members (bootstrap)'}")
    print(f"  target={args.target}  (output_dim={outputs_list[0].shape[1]})")
    print()

    # ── train ─────────────────────────────────────────────────────────────────
    trainer = SurrogateTrainer(train_config)
    t0 = time.time()
    result = trainer.train(parameters, coords_list, outputs_list)
    elapsed = time.time() - t0

    if not result.success:
        print(f"Training failed: {result.error_message}")
        sys.exit(1)

    print(f"\nTraining complete in {elapsed:.1f}s")
    print(f"  Train loss (final): {result.train_loss:.4e}")
    print(f"  Test  loss (best):  {result.test_loss:.4e}")
    print(f"  Metrics: {result.metrics}")
    if result.model_path:
        print(f"  Model saved: {result.model_path}")

    # ── learning curves ────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle(f"Transolver training — target={args.target}, d_model={args.d_model}, "
                 f"n_layers={args.n_layers}", fontsize=11)

    h = result.history
    epochs_run = len(h["train_loss"])

    ax = axes[0]
    ax.semilogy(h["train_loss"], label="train")
    ax.semilogy(h["test_loss"],  label="test", linestyle="--")
    ax.set_xlabel("Epoch"); ax.set_ylabel("MSE loss (normalised)")
    ax.set_title("Learning curves"); ax.legend(); ax.grid(True, alpha=0.3)
    ax.text(0.98, 0.95, f"best test={result.test_loss:.3e}",
            transform=ax.transAxes, ha="right", va="top", fontsize=9)

    ax = axes[1]
    # Relative error per field on test set
    metrics = result.metrics
    names = [k for k in metrics if "rel" in k.lower() or "mae" in k.lower() or "r2" in k.lower()]
    if names:
        vals = [metrics[k] for k in names]
        ax.bar(range(len(names)), vals, color="steelblue", alpha=0.8)
        ax.set_xticks(range(len(names))); ax.set_xticklabels(names, rotation=30, ha="right", fontsize=8)
        ax.set_title("Test metrics"); ax.grid(True, alpha=0.3, axis="y")
    else:
        ax.text(0.5, 0.5, "No per-field metrics", transform=ax.transAxes, ha="center")

    plt.tight_layout()
    curve_path = save_dir / "training_curves.png"
    plt.savefig(curve_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Curves saved: {curve_path}")

    # ── qualitative predictions on 2 test samples ──────────────────────────────
    _plot_predictions(trainer, parameters, coords_list, outputs_list,
                      result, args, save_dir)


def _plot_predictions(trainer, parameters, coords_list, outputs_list, result, args, save_dir):
    """Plot surrogate vs FEA for 2 held-out test samples."""
    import matplotlib.tri as mtri
    from piano.data.dataset import FEMDataset

    # Reload raw dataset to get element connectivity for triangulation
    ds = FEMDataset.load(args.data_dir)
    samples = [s for s in ds if s.displacement is not None and
               (args.target not in ("all", "stress") or s.von_mises is not None)]

    if len(samples) < 2:
        return

    rng = np.random.default_rng(args.seed)
    test_idx = rng.choice(len(samples), size=min(2, len(samples)), replace=False)

    target = args.target
    field_labels = {
        "displacement":["u_x (μm)", "u_y (μm)"],
        "stress":      ["σ_vm (MPa)"],
        "damage":      ["d"],
    }[target]
    # stress predictions come out of the model in log1p(Pa); invert before plotting
    stress_scale = lambda v: (np.expm1(v) / 1e6) if target == "stress" else v
    scales = {
        "displacement": [1e6, 1e6],
        "stress":       [1.0],   # handled by stress_scale above
        "damage":       [1.0],
    }[target]
    cmaps = {
        "displacement": ["RdBu_r", "RdBu_r"],
        "stress":       ["plasma"],
        "damage":       ["hot_r"],
    }[target]

    n_fields = len(field_labels)
    fig, axes = plt.subplots(len(test_idx) * 2, n_fields,
                             figsize=(4 * n_fields, 3.5 * len(test_idx) * 2))
    if axes.ndim == 1:
        axes = axes[np.newaxis, :]
    fig.suptitle(f"Surrogate vs FEA — {target} fields", fontsize=12)

    for row_pair, si in enumerate(test_idx):
        s = samples[si]
        p = s.parameters
        param_vec = np.array([p["E"], p["nu"], p["traction"], p["G_c"], p["crack_length"]],
                              dtype=np.float32)[np.newaxis, :]
        enriched = enrich_coords(s.coordinates.astype(np.float32),
                                 p["crack_length"], crack_y=0.5)

        # Ground truth (in the same transform space the model was trained on)
        if target == "displacement":
            gt = s.displacement
        elif target == "stress":
            gt = np.log1p(s.von_mises[:, None])
        else:  # damage
            gt = s.damage[:, None]

        pred = trainer.predict(param_vec, enriched)  # (N, output_dim)

        tri = mtri.Triangulation(s.coordinates[:, 0], s.coordinates[:, 1],
                                  triangles=s.elements)
        cl = p["crack_length"]

        for fi in range(n_fields):
            sc = scales[fi]
            cmap = cmaps[fi]
            gt_f   = stress_scale(gt[:, fi]) * sc
            pred_f = stress_scale(pred[:, fi]) * sc
            vmin, vmax = gt_f.min(), gt_f.max()

            for sub_row, (vals, row_label) in enumerate([(gt_f, "FEA"), (pred_f, "Surrogate")]):
                ax = axes[row_pair * 2 + sub_row, fi]
                tc = ax.tripcolor(tri, vals, shading="gouraud", cmap=cmap,
                                  vmin=vmin, vmax=vmax)
                ax.plot([0, cl], [0.5, 0.5], "w-", lw=1)
                ax.plot(cl, 0.5, "w^", ms=3)
                ax.set_aspect("equal"); ax.set_xlim(0,1); ax.set_ylim(0,1)
                ax.set_xticks([]); ax.set_yticks([])
                title = f"{row_label}: {field_labels[fi]}"
                if sub_row == 1:
                    rel_err = np.abs(gt_f - pred_f).mean() / (np.abs(gt_f).mean() + 1e-12)
                    title += f"  (rel={rel_err:.2f})"
                ax.set_title(title, fontsize=7)
                plt.colorbar(tc, ax=ax, shrink=0.8, pad=0.01)

    plt.tight_layout()
    pred_path = save_dir / "predictions.png"
    plt.savefig(pred_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Predictions saved: {pred_path}")


if __name__ == "__main__":
    main()
