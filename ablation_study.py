"""
Optuna-based Hyperparameter Sensitivity Analysis for Transolver.

This script performs an ablation study to identify which hyperparameters
have the largest influence on neural operator performance.

Features:
- TPE sampler (Bayesian optimization)
- MedianPruner (kills bad trials early)
- Hyperparameter importance analysis
- SQLite persistence for resumable studies
- Visualization of results

Usage:
    # Quick test (20 trials)
    python ablation_study.py --n-trials 20 --train-dir train01

    # Full study (100 trials)
    python ablation_study.py --n-trials 100 --train-dir train01 train02

    # Resume interrupted study
    python ablation_study.py --n-trials 100 --study-name my_study
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent))

from piano.surrogate.base import TransolverConfig
from piano.surrogate.transolver import TransolverModel


# =============================================================================
# Data Loading (reuse from train_transolver)
# =============================================================================

def load_training_data(
    train_dirs: List[str],
    max_samples: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load training data by running FEM on each mesh file.

    Returns:
        parameters:    (N, 3) - [E/GPa, nu, load]
        coordinates:   (N, max_elems, 2) - element centers, zero-padded
        stress_fields: (N, max_elems) - von Mises stress, zero-padded
    """
    from train_transolver import load_training_data as _load
    params, coords, stress, _ = _load(train_dirs, max_samples)
    return params, coords, stress


# =============================================================================
# Objective Function
# =============================================================================

def create_objective(
    parameters: np.ndarray,
    coordinates: np.ndarray,
    stress_fields: np.ndarray,
    max_epochs: int = 100,
    patience: int = 30,
    batch_size: int = 16,
):
    """Create Optuna objective function with bound data."""

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Precompute train/val split (90/10)
    n_samples = len(parameters)
    n_train = int(n_samples * 0.9)

    np.random.seed(42)
    indices = np.random.permutation(n_samples)
    train_idx = indices[:n_train]
    val_idx = indices[n_train:]

    # Normalize
    param_mean = parameters.mean(axis=0)
    param_std = parameters.std(axis=0) + 1e-8
    stress_mean = stress_fields.mean()
    stress_std = stress_fields.std() + 1e-8

    params_norm = (parameters - param_mean) / param_std
    stress_norm = (stress_fields - stress_mean) / stress_std

    # Convert to tensors
    train_params = torch.tensor(params_norm[train_idx], dtype=torch.float32, device=device)
    train_coords = torch.tensor(coordinates[train_idx], dtype=torch.float32, device=device)
    train_stress = torch.tensor(stress_norm[train_idx], dtype=torch.float32, device=device).unsqueeze(-1)

    val_params = torch.tensor(params_norm[val_idx], dtype=torch.float32, device=device)
    val_coords = torch.tensor(coordinates[val_idx], dtype=torch.float32, device=device)
    val_stress = torch.tensor(stress_norm[val_idx], dtype=torch.float32, device=device).unsqueeze(-1)

    def objective(trial: optuna.Trial) -> float:
        """Single trial: sample hyperparameters, train model, return val loss."""

        # Sample hyperparameters
        optimizer_type = trial.suggest_categorical("optimizer", ["adamw", "adam", "sgd"])
        lr = trial.suggest_float("lr", 1e-4, 5e-3, log=True)
        scheduler_type = trial.suggest_categorical("scheduler", ["plateau", "cosine", "none"])
        activation = trial.suggest_categorical("activation", ["gelu", "relu", "silu"])
        pino_weight = trial.suggest_float("pino_weight", 0.0, 0.5)
        pino_eq_weight = trial.suggest_float("pino_eq_weight", 0.0, 0.2)
        d_model = trial.suggest_categorical("d_model", [64, 128, 256])
        n_layers = trial.suggest_int("n_layers", 2, 6)
        slice_num = trial.suggest_categorical("slice_num", [16, 32, 64])
        dropout = trial.suggest_float("dropout", 0.0, 0.2)

        # Build config
        config = TransolverConfig(
            slice_num=slice_num,
            n_heads=8,
            d_model=d_model,
            n_layers=n_layers,
            mlp_ratio=4.0,
            dropout=dropout,
            learning_rate=lr,
            batch_size=batch_size,
            epochs=max_epochs,
            patience=patience,
            output_dim=1,
            pino_weight=pino_weight,
            pino_eq_weight=pino_eq_weight,
            optimizer_type=optimizer_type,
            scheduler_type=scheduler_type,
            activation=activation,
        )

        # Build model
        model = TransolverModel(config)
        model.build(
            input_dim=parameters.shape[1],
            coord_dim=2,
            num_points=coordinates.shape[1],
        )
        model.to(device)

        # Create optimizer
        if optimizer_type == "adamw":
            optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
        elif optimizer_type == "adam":
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        else:  # sgd
            optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)

        # Create scheduler
        if scheduler_type == "plateau":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', factor=0.5, patience=10
            )
        elif scheduler_type == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=max_epochs, eta_min=1e-6
            )
        else:
            scheduler = None

        criterion = nn.MSELoss()

        # Training loop
        best_val_loss = float('inf')
        patience_counter = 0

        for epoch in range(max_epochs):
            model.train()
            perm = torch.randperm(len(train_params))
            epoch_loss = 0.0

            for i in range(0, len(train_params), batch_size):
                idx = perm[i:i + batch_size]
                optimizer.zero_grad()
                pred = model.forward(train_params[idx], train_coords[idx])
                loss = criterion(pred, train_stress[idx])
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                epoch_loss += loss.item() * len(idx)

            train_loss = epoch_loss / len(train_params)

            # Validation
            model.eval()
            with torch.no_grad():
                val_loss = criterion(model.forward(val_params, val_coords), val_stress).item()

            # Update scheduler
            if scheduler is not None:
                if scheduler_type == "plateau":
                    scheduler.step(val_loss)
                else:
                    scheduler.step()

            # Early stopping check
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1

            # Report intermediate value for pruning
            trial.report(val_loss, epoch)

            # Check if trial should be pruned
            if trial.should_prune():
                raise optuna.TrialPruned()

            if patience_counter >= patience:
                break

        return best_val_loss

    return objective


# =============================================================================
# Visualization
# =============================================================================

def plot_results(study: optuna.Study, output_dir: Path):
    """Generate visualization plots for the study."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Parameter importance
    try:
        importances = optuna.importance.get_param_importances(study)

        fig, ax = plt.subplots(figsize=(10, 6))
        params = list(importances.keys())
        values = list(importances.values())

        y_pos = np.arange(len(params))
        ax.barh(y_pos, values, align='center')
        ax.set_yticks(y_pos)
        ax.set_yticklabels(params)
        ax.invert_yaxis()
        ax.set_xlabel('Importance')
        ax.set_title('Hyperparameter Importance')
        ax.grid(True, axis='x', alpha=0.3)

        fig.tight_layout()
        fig.savefig(output_dir / 'param_importance.png', dpi=150)
        plt.close(fig)
        print(f"Saved: {output_dir / 'param_importance.png'}")

        # Save importance as JSON
        with open(output_dir / 'param_importance.json', 'w') as f:
            json.dump(importances, f, indent=2)

    except Exception as e:
        print(f"Could not compute parameter importance: {e}")

    # 2. Optimization history
    fig, ax = plt.subplots(figsize=(10, 5))
    trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    values = [t.value for t in trials]
    best_values = np.minimum.accumulate(values)

    ax.plot(range(1, len(values) + 1), values, 'o-', alpha=0.5, label='Trial value')
    ax.plot(range(1, len(values) + 1), best_values, 'r-', linewidth=2, label='Best so far')
    ax.set_xlabel('Trial')
    ax.set_ylabel('Validation Loss')
    ax.set_title('Optimization History')
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_dir / 'optimization_history.png', dpi=150)
    plt.close(fig)
    print(f"Saved: {output_dir / 'optimization_history.png'}")

    # 3. Parallel coordinate plot (top 20 trials)
    try:
        top_trials = sorted(
            [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE],
            key=lambda t: t.value
        )[:20]

        if len(top_trials) >= 5:
            fig = optuna.visualization.matplotlib.plot_parallel_coordinate(
                study,
                params=['lr', 'd_model', 'n_layers', 'slice_num', 'dropout', 'pino_weight']
            )
            fig.savefig(output_dir / 'parallel_coordinate.png', dpi=150, bbox_inches='tight')
            plt.close(fig)
            print(f"Saved: {output_dir / 'parallel_coordinate.png'}")
    except Exception as e:
        print(f"Could not create parallel coordinate plot: {e}")

    # 4. Slice plot for key parameters
    try:
        for param in ['lr', 'd_model', 'n_layers', 'pino_weight']:
            fig = optuna.visualization.matplotlib.plot_slice(study, params=[param])
            fig.savefig(output_dir / f'slice_{param}.png', dpi=150, bbox_inches='tight')
            plt.close(fig)
            print(f"Saved: {output_dir / f'slice_{param}.png'}")
    except Exception as e:
        print(f"Could not create slice plots: {e}")


def save_best_config(study: optuna.Study, output_dir: Path):
    """Save best hyperparameters to JSON."""
    best_trial = study.best_trial

    config = {
        'best_value': best_trial.value,
        'best_params': best_trial.params,
        'n_trials': len(study.trials),
        'n_completed': len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]),
        'n_pruned': len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]),
    }

    output_path = output_dir / 'best_config.json'
    with open(output_path, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"Saved: {output_path}")

    return config


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Optuna hyperparameter sensitivity analysis")
    parser.add_argument("--train-dir", type=str, nargs="+", default=["train01"],
                        help="Training data directories")
    parser.add_argument("--n-trials", type=int, default=50, help="Number of Optuna trials")
    parser.add_argument("--max-epochs", type=int, default=100, help="Max epochs per trial")
    parser.add_argument("--patience", type=int, default=30, help="Early stopping patience")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--max-samples", type=int, default=None, help="Max training samples")
    parser.add_argument("--study-name", type=str, default="ablation_study",
                        help="Optuna study name (for resuming)")
    parser.add_argument("--output-dir", type=str, default="outputs/ablation",
                        help="Output directory for results")
    parser.add_argument("--n-jobs", type=int, default=1, help="Number of parallel jobs")
    args = parser.parse_args()

    project_root = Path(__file__).parent
    train_dirs = [str(project_root / d) for d in args.train_dir]
    output_dir = project_root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    db_path = output_dir / f"{args.study_name}.db"

    print("=" * 60)
    print("Optuna Hyperparameter Sensitivity Analysis")
    print("=" * 60)
    print(f"Train dirs:    {args.train_dir}")
    print(f"N trials:      {args.n_trials}")
    print(f"Max epochs:    {args.max_epochs}")
    print(f"Patience:      {args.patience}")
    print(f"Batch size:    {args.batch_size}")
    print(f"Study name:    {args.study_name}")
    print(f"Database:      {db_path}")
    print(f"Output dir:    {output_dir}")
    print("=" * 60)

    # Load training data
    print("\nLoading training data...")
    parameters, coordinates, stress_fields = load_training_data(
        train_dirs, max_samples=args.max_samples
    )
    print(f"Loaded {len(parameters)} samples")
    print(f"  Parameters shape:  {parameters.shape}")
    print(f"  Coordinates shape: {coordinates.shape}")
    print(f"  Stress shape:      {stress_fields.shape}")

    # Create objective function
    objective = create_objective(
        parameters,
        coordinates,
        stress_fields,
        max_epochs=args.max_epochs,
        patience=args.patience,
        batch_size=args.batch_size,
    )

    # Create or load study
    storage = f"sqlite:///{db_path}"
    study = optuna.create_study(
        study_name=args.study_name,
        storage=storage,
        load_if_exists=True,
        direction="minimize",
        sampler=TPESampler(seed=42),
        pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=10),
    )

    # Check if resuming
    n_existing = len(study.trials)
    if n_existing > 0:
        print(f"\nResuming study with {n_existing} existing trials")
        print(f"Current best value: {study.best_value:.6f}")

    # Run optimization
    n_remaining = max(0, args.n_trials - n_existing)
    if n_remaining > 0:
        print(f"\nRunning {n_remaining} new trials...")
        study.optimize(
            objective,
            n_trials=n_remaining,
            n_jobs=args.n_jobs,
            show_progress_bar=True,
        )

    # Print results
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)

    n_completed = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
    n_pruned = len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])

    print(f"Total trials:    {len(study.trials)}")
    print(f"Completed:       {n_completed}")
    print(f"Pruned:          {n_pruned}")
    print(f"Best val loss:   {study.best_value:.6f}")
    print("\nBest hyperparameters:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")

    # Save results and visualizations
    print("\n" + "=" * 60)
    print("Saving results and visualizations...")
    print("=" * 60)

    save_best_config(study, output_dir)
    plot_results(study, output_dir)

    print("\nDone!")
    print(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
