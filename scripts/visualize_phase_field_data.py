"""
Visualize phase field fracture training dataset.

Produces:
  phase_field_data/overview.png         — 5×2 grid: damage field for 10 samples
  phase_field_data/detail_sample.png    — 4-panel deep dive (mesh, disp_y, disp_mag, damage)
  phase_field_data/fields_sample.png    — 5-panel: u_x, u_y, |u|, von Mises, damage
  phase_field_data/fields_overview.png  — 3×5 grid comparing fields across 5 samples
  phase_field_data/param_coverage.png   — scatter matrix of 5 input parameters

Usage:
    ~/miniforge3/envs/piano/bin/python scripts/visualize_phase_field_data.py
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from piano.data.dataset import FEMDataset


def load_tri(s):
    xy = s.coordinates
    return mtri.Triangulation(xy[:, 0], xy[:, 1], triangles=s.elements)


def annotate_crack(ax, crack_length, crack_y=0.5, color="w", lw=1.5, crack_color=None):
    c = crack_color if crack_color is not None else color
    ax.plot([0, crack_length], [crack_y, crack_y], color=c, lw=lw)
    ax.plot(crack_length, crack_y, color=c, marker="^", ms=4)
    ax.set_aspect("equal")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xticks([]); ax.set_yticks([])


def tripcolor_panel(ax, triang, vals, title, cmap, label, crack_length,
                    vmin=None, vmax=None, crack_color="w"):
    tc = ax.tripcolor(triang, vals, shading="gouraud", cmap=cmap, vmin=vmin, vmax=vmax)
    annotate_crack(ax, crack_length, color=crack_color)
    plt.colorbar(tc, ax=ax, label=label, shrink=0.85, pad=0.01)
    ax.set_title(title, fontsize=8, pad=3)
    return tc


# ── load ──────────────────────────────────────────────────────────────────────
ds = FEMDataset.load("phase_field_data")
samples = list(ds)
n = len(samples)
print(f"Loaded {n} samples")

dmg_maxs  = np.array([s.damage.max() for s in samples])
disp_maxs = np.array([np.linalg.norm(s.displacement, axis=1).max() for s in samples])
vm_maxs   = np.array([s.von_mises.max() if s.von_mises is not None else 0.0 for s in samples])


# ── 1. Overview: damage field for 10 representative samples ──────────────────
sort_idx = np.argsort([s.parameters["crack_length"] for s in samples])
picks = [sort_idx[i] for i in np.linspace(0, n - 1, 10, dtype=int)]

fig, axes = plt.subplots(2, 5, figsize=(15, 6.5))
fig.suptitle("Phase Field FEA Dataset — Damage Field Overview (50 samples)", fontsize=12, y=0.98)

for ax, idx in zip(axes.flat, picks):
    s = samples[idx]
    p = s.parameters
    triang = load_tri(s)
    tc = ax.tripcolor(triang, s.damage, shading="gouraud", cmap="hot_r", vmin=0, vmax=1)
    annotate_crack(ax, p["crack_length"], crack_color="k")
    ax.set_title(f"a={p['crack_length']:.2f}  σ={p['traction']/1e6:.1f}MPa\n"
                 f"G_c={p['G_c']:.0f}  d_max={s.damage.max():.3f}", fontsize=7, pad=2)

plt.colorbar(tc, ax=axes, label="Damage d  (0=intact, 1=failed)", shrink=0.6, pad=0.02)
plt.savefig("phase_field_data/overview.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved phase_field_data/overview.png")


# ── 2. Detailed 4-panel for the highest-traction sample ──────────────────────
# Use the sample with highest von Mises (most loaded) for the most informative detail view
idx_best = int(np.argmax(vm_maxs))
s = samples[idx_best]
p = s.parameters
triang = load_tri(s)
cl, cy = p["crack_length"], 0.5
disp_y   = s.displacement[:, 1]
disp_mag = np.linalg.norm(s.displacement, axis=1)

fig, axes = plt.subplots(1, 4, figsize=(16, 4))
fig.suptitle(
    f"Highest von Mises sample — crack={cl:.3f}  σ={p['traction']/1e6:.1f}MPa  "
    f"G_c={p['G_c']:.0f}  E={p['E']/1e9:.0f}GPa  ν={p['nu']:.3f}",
    fontsize=10
)

# Panel 1: mesh
ax = axes[0]
ax.triplot(triang, lw=0.2, color="gray")
ax.plot([0, cl], [cy, cy], "r-", lw=1.5)
ax.plot(cl, cy, "r^", ms=5)
ax.set_aspect("equal"); ax.set_xlim(0,1); ax.set_ylim(0,1)
ax.set_xticks([]); ax.set_yticks([])
ax.set_title(f"Mesh  ({s.coordinates.shape[0]} nodes, {s.elements.shape[0]} elems)", fontsize=8)

tripcolor_panel(axes[1], triang, disp_y * 1e6,   "Vertical displacement u_y",   "RdBu_r", "u_y (μm)", cl)
tripcolor_panel(axes[2], triang, disp_mag * 1e6,  "Displacement magnitude |u|",  "viridis", "|u| (μm)", cl)
tripcolor_panel(axes[3], triang, s.damage,        "Damage field d",               "hot_r",  "d", cl, vmin=0, vmax=1)

# Zoom inset around crack tip in damage panel
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
ax_ins = inset_axes(axes[3], width="40%", height="40%", loc="upper right")
r = 0.12
ax_ins.tripcolor(triang, s.damage, shading="gouraud", cmap="hot_r", vmin=0, vmax=1)
ax_ins.set_xlim(cl - r, cl + r); ax_ins.set_ylim(cy - r, cy + r)
ax_ins.plot(cl, cy, "c^", ms=4)
ax_ins.set_xticks([]); ax_ins.set_yticks([])
mark_inset(axes[3], ax_ins, loc1=2, loc2=4, fc="none", ec="cyan", lw=0.8)

plt.tight_layout()
plt.savefig("phase_field_data/detail_sample.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved phase_field_data/detail_sample.png")


# ── 3. Full fields figure: all 5 training targets for one sample ──────────────
fig, axes = plt.subplots(1, 5, figsize=(20, 4))
fig.suptitle(
    f"All Training Fields — crack={cl:.3f}  σ={p['traction']/1e6:.1f}MPa  "
    f"G_c={p['G_c']:.0f}  E={p['E']/1e9:.0f}GPa  ν={p['nu']:.3f}",
    fontsize=10
)

disp_x = s.displacement[:, 0]

tripcolor_panel(axes[0], triang, disp_x * 1e6,
                "Horizontal displacement u_x", "RdBu_r", "u_x (μm)", cl)
tripcolor_panel(axes[1], triang, disp_y * 1e6,
                "Vertical displacement u_y",   "RdBu_r", "u_y (μm)", cl)
tripcolor_panel(axes[2], triang, disp_mag * 1e6,
                "Displacement magnitude |u|",  "viridis", "|u| (μm)", cl)
tripcolor_panel(axes[3], triang, s.von_mises / 1e6,
                "Von Mises stress σ_vm",        "plasma",  "σ_vm (MPa)", cl,
                vmax=np.percentile(s.von_mises, 98) / 1e6)
tripcolor_panel(axes[4], triang, s.damage,
                "Damage field d",               "hot_r",   "d", cl, vmin=0, vmax=1)

plt.tight_layout()
plt.savefig("phase_field_data/fields_sample.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved phase_field_data/fields_sample.png")


# ── 4. Fields overview: 5 samples × 3 fields (u_y, von Mises, damage) ────────
# Pick 5 samples spanning traction range
trac_vals = np.array([s.parameters["traction"] for s in samples])
trac_idx  = np.argsort(trac_vals)
sel = [trac_idx[i] for i in np.linspace(0, n - 1, 5, dtype=int)]

fig, axes = plt.subplots(3, 5, figsize=(18, 11))
fig.suptitle("Displacement, Von Mises & Damage — 5 samples spanning traction range", fontsize=12)

row_labels = ["u_y (μm)", "σ_vm (MPa)", "Damage d"]
cmaps = ["RdBu_r", "plasma", "hot_r"]

# Shared colour limits per row
uy_all  = np.concatenate([samples[i].displacement[:, 1] * 1e6 for i in sel])
vm_all  = np.concatenate([samples[i].von_mises / 1e6 for i in sel if samples[i].von_mises is not None])
uy_lim  = (uy_all.min(), uy_all.max())
vm_lim  = (0, np.percentile(vm_all, 98))

for col, idx in enumerate(sel):
    s2 = samples[idx]
    p2 = s2.parameters
    tri2 = load_tri(s2)
    cl2, cy2 = p2["crack_length"], 0.5
    uy2  = s2.displacement[:, 1] * 1e6
    vm2  = s2.von_mises / 1e6 if s2.von_mises is not None else np.zeros(len(uy2))
    dmg2 = s2.damage

    col_title = (f"a={cl2:.2f}  σ={p2['traction']/1e6:.1f}MPa\n"
                 f"G_c={p2['G_c']:.0f}")
    axes[0, col].set_title(col_title, fontsize=8, pad=3)

    for row, (field, cmap, lbl, vlim) in enumerate(zip(
        [uy2, vm2, dmg2],
        cmaps,
        row_labels,
        [uy_lim, vm_lim, (0, 1)],
    )):
        ax = axes[row, col]
        tc = ax.tripcolor(tri2, field, shading="gouraud", cmap=cmap,
                          vmin=vlim[0], vmax=vlim[1])
        annotate_crack(ax, cl2, color="k" if cmap == "hot_r" else "w")
        if col == 4:
            plt.colorbar(tc, ax=ax, label=lbl, shrink=0.9)

    if col == 0:
        for row, lbl in enumerate(row_labels):
            axes[row, 0].set_ylabel(lbl, fontsize=8)

plt.tight_layout()
plt.savefig("phase_field_data/fields_overview.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved phase_field_data/fields_overview.png")


# ── 5. Parameter coverage scatter matrix ─────────────────────────────────────
param_names  = ["crack_length", "traction", "G_c", "E", "nu"]
param_labels = ["Crack length (m)", "Traction (MPa)", "G_c (J/m²)", "E (GPa)", "ν"]
param_scale  = [1.0, 1e-6, 1.0, 1e-9, 1.0]

data = np.array([[s.parameters[k] * sc for k, sc in zip(param_names, param_scale)]
                 for s in samples])

fig, axes = plt.subplots(5, 5, figsize=(12, 12))
fig.suptitle("Parameter Space Coverage (50 LHS samples)", fontsize=12, y=0.98)

for i in range(5):
    for j in range(5):
        ax = axes[i, j]
        if i == j:
            ax.hist(data[:, i], bins=10, color="steelblue", edgecolor="white", lw=0.4)
            ax.set_xlabel(param_labels[i], fontsize=7)
        else:
            sc = ax.scatter(data[:, j], data[:, i], c=vm_maxs / 1e6, cmap="plasma",
                            s=20, alpha=0.85, vmin=0, vmax=vm_maxs.max() / 1e6)
            if i == 4:
                ax.set_xlabel(param_labels[j], fontsize=7)
            if j == 0:
                ax.set_ylabel(param_labels[i], fontsize=7)
        ax.tick_params(labelsize=6)

plt.colorbar(sc, ax=axes, label="Max von Mises (MPa)", shrink=0.5, pad=0.02)
plt.savefig("phase_field_data/param_coverage.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved phase_field_data/param_coverage.png")


print("\nSummary:")
print(f"  Samples:               {n}")
print(f"  Displacement range:    [{disp_maxs.min():.2e}, {disp_maxs.max():.2e}] m")
print(f"  Von Mises range:       [{vm_maxs.min()/1e6:.1f}, {vm_maxs.max()/1e6:.1f}] MPa")
print(f"  Highest-stress sample: idx={idx_best}, σ_vm_max={vm_maxs[idx_best]/1e6:.1f} MPa")
