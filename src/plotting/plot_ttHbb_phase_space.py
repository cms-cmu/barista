#!/usr/bin/env python3
"""
Plot 2D Phase Space (lead vs subl dijet mass) for ttHbb and QCD Multijet Background
with various proposed Signal Region (SR) and Sideband (SB) shape overlays:
- SB: Outer Box [50, 200] x [50, 200] GeV \ SR
- SR Option 1: Diamond (Rotated Square) with vertices (125, 170), (170, 125), (125, 80), (80, 125)
- SR Option 2: Circle with R_H < 35 GeV
- SR Option 3: 4-Pointed Star (Astroid / concave polygon)
- SR Option 4: Square Box [100, 150] x [100, 150] GeV
"""

import os
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.path import Path
import numpy as np
from coffea.util import load


def get_2d_values(h_obj, process, tag, region="inclusive"):
    try:
        sub = h_obj[{"process": process, "tag": tag, "region": region}]
    except Exception:
        try:
            sub = h_obj[{"process": process, "tag": tag, "region": sum}]
        except Exception:
            try:
                sub = h_obj[{"process": process, "tag": tag}]
            except Exception:
                sub = h_obj[{"process": process}]
    
    while len(sub.axes) > 2:
        sub = sub[{sub.axes[0].name: sum}]
    
    vals = sub.values()
    x_edges = sub.axes[0].edges
    y_edges = sub.axes[1].edges
    return vals, x_edges, y_edges


def create_star_path(cx=125, cy=125, r_outer=50, r_inner=22, n_points=4):
    """Create a 4-pointed star path."""
    angles = np.linspace(0, 2 * np.pi, 2 * n_points, endpoint=False)
    # Rotate by 45 deg so points are aligned with axes
    angles += np.pi / 2
    radii = np.where(np.arange(2 * n_points) % 2 == 0, r_outer, r_inner)
    verts = [(cx + r * np.cos(a), cy + r * np.sin(a)) for a, r in zip(angles, radii)]
    verts.append(verts[0])
    codes = [Path.MOVETO] + [Path.LINETO] * (len(verts) - 2) + [Path.CLOSEPOLY]
    return Path(verts, codes)


def plot_phase_space(coffea_file: str, output_path: str = "output/ttHbb/plots_ttHbb/"):
    os.makedirs(output_path, exist_ok=True)
    print(f"Loading {coffea_file}...")
    hists = load(coffea_file)

    hist_collection = hists.get("hists", hists)

    h2d_name = None
    for k in hist_collection.keys():
        if "lead_vs_subl_m" in k and "selected" in k:
            h2d_name = k
            break
        elif "lead_vs_subl_m" in k:
            h2d_name = k

    if not h2d_name:
        h2d_name = "quadJet_selected.lead_vs_subl_m" if "quadJet_selected.lead_vs_subl_m" in hist_collection else list(hist_collection.keys())[0]

    print(f"Using histogram: {h2d_name}")
    h = hist_collection[h2d_name]

    # Extract 2D arrays
    vals_qcd, x_edges, y_edges = get_2d_values(h, "data", "threeTag", region="inclusive")
    vals_tth, _, _ = get_2d_values(h, "ttHbb", "fourTag", region="inclusive")

    # =========================================================================
    # 1. RAW 2D PLOT: Completely uncut, full phase space without ANY overlays
    # =========================================================================
    fig_raw, axes_raw = plt.subplots(1, 2, figsize=(16, 7))

    im0 = axes_raw[0].imshow(
        vals_qcd.T,
        extent=[x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]],
        origin='lower',
        aspect='auto',
        cmap='turbo',
    )
    axes_raw[0].set_title("QCD Multijet Background (3-tag Data) - Full Phase Space", fontsize=15, fontweight='bold')
    axes_raw[0].set_xlabel("Selected Quad Jet Lead Boson Candidate Mass [GeV]", fontsize=13)
    axes_raw[0].set_ylabel("Selected Quad Jet Subl Boson Candidate Mass [GeV]", fontsize=13)
    axes_raw[0].set_xlim(0, 250)
    axes_raw[0].set_ylim(0, 250)
    fig_raw.colorbar(im0, ax=axes_raw[0], fraction=0.046, pad=0.04)

    im1 = axes_raw[1].imshow(
        vals_tth.T,
        extent=[x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]],
        origin='lower',
        aspect='auto',
        cmap='turbo',
    )
    axes_raw[1].set_title("ttH(bb) Signal (4-tag MC) - Full Phase Space", fontsize=15, fontweight='bold')
    axes_raw[1].set_xlabel("Selected Quad Jet Lead Boson Candidate Mass [GeV]", fontsize=13)
    axes_raw[1].set_ylabel("Selected Quad Jet Subl Boson Candidate Mass [GeV]", fontsize=13)
    axes_raw[1].set_xlim(0, 250)
    axes_raw[1].set_ylim(0, 250)
    fig_raw.colorbar(im1, ax=axes_raw[1], fraction=0.046, pad=0.04)

    raw_file = os.path.join(output_path, "ttHbb_phase_space_raw.png")
    fig_raw.tight_layout()
    fig_raw.savefig(raw_file, dpi=150)
    print(f"Saved raw uncut 2D plot to: {raw_file}")
    plt.close(fig_raw)

    # =========================================================================
    # 2. COMBINED STUDY PLOT: All SR options overlaid on Multijet & ttHbb
    # =========================================================================
    fig_study, axes_study = plt.subplots(1, 2, figsize=(16, 7))

    im0 = axes_study[0].imshow(
        vals_qcd.T,
        extent=[x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]],
        origin='lower',
        aspect='auto',
        cmap='turbo',
    )
    axes_study[0].set_title("QCD Multijet (3-tag Data) + Proposed SR/SB", fontsize=15, fontweight='bold')
    axes_study[0].set_xlabel("Selected Quad Jet Lead Boson Candidate Mass [GeV]", fontsize=13)
    axes_study[0].set_ylabel("Selected Quad Jet Subl Boson Candidate Mass [GeV]", fontsize=13)
    fig_study.colorbar(im0, ax=axes_study[0], fraction=0.046, pad=0.04)

    im1 = axes_study[1].imshow(
        vals_tth.T,
        extent=[x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]],
        origin='lower',
        aspect='auto',
        cmap='turbo',
    )
    axes_study[1].set_title("ttH(bb) Signal (4-tag MC) + Proposed SR/SB", fontsize=15, fontweight='bold')
    axes_study[1].set_xlabel("Selected Quad Jet Lead Boson Candidate Mass [GeV]", fontsize=13)
    axes_study[1].set_ylabel("Selected Quad Jet Subl Boson Candidate Mass [GeV]", fontsize=13)
    fig_study.colorbar(im1, ax=axes_study[1], fraction=0.046, pad=0.04)

    for ax in axes_study:
        # Common Sideband Outer Boundary: Box [50, 220] x [50, 220] GeV
        sb_box = patches.Rectangle((50, 50), 170, 170, linewidth=2.5, edgecolor='magenta', facecolor='none', linestyle='--', label='SB Outer Box: $[50, 220]$')
        ax.add_patch(sb_box)

        # SR Option 1: Diamond / Rotated Rectangle
        diamond_pts = np.array([[125, 200], [200, 125], [125, 80], [80, 125]])
        diamond = patches.Polygon(diamond_pts, closed=True, linewidth=2.5, edgecolor='white', facecolor='none', linestyle='-', label='SR: Rotated Square')
        ax.add_patch(diamond)

        # SR Option 2: Circle R_H < 35
        sr_circle = patches.Circle((125, 125), 35, linewidth=2.0, edgecolor='cyan', facecolor='none', linestyle=':', label='SR: Circle $R_H < 35$')
        ax.add_patch(sr_circle)

        # SR Option 3: 4-Pointed Star
        star_path = create_star_path(cx=125, cy=125, r_outer=45, r_inner=20, n_points=4)
        star_patch = patches.PathPatch(star_path, linewidth=2.0, edgecolor='yellow', facecolor='none', linestyle='-.', label='SR: 4-Pointed Star')
        ax.add_patch(star_patch)

        ax.set_xlim(0, 250)
        ax.set_ylim(0, 250)
        ax.legend(loc='upper right', framealpha=0.85, fontsize=10)

    study_file = os.path.join(output_path, "ttHbb_phase_space_study.png")
    fig_study.tight_layout()
    fig_study.savefig(study_file, dpi=150)
    print(f"Saved study 2D plot to: {study_file}")
    plt.close(fig_study)

    # =========================================================================
    # 3. MULTI-PANEL COMPARISON: Individual subplots for each SR definition
    # =========================================================================
    fig_comp, axes_comp = plt.subplots(2, 3, figsize=(20, 13))

    sr_definitions = [
        ("Option 1: Rotated Square (Selected)", [
            patches.Polygon(np.array([[125, 200], [200, 125], [125, 80], [80, 125]]), closed=True, linewidth=2.5, edgecolor='white', facecolor='none', linestyle='-', label='SR: Rotated Square')
        ]),
        ("Option 2: Circle (R < 35 GeV)", [
            patches.Circle((125, 125), 35, linewidth=2.5, edgecolor='cyan', facecolor='none', linestyle='-', label='SR: Circle $R < 35$')
        ]),
        ("Option 3: 4-Pointed Star", [
            patches.PathPatch(create_star_path(cx=125, cy=125, r_outer=45, r_inner=20, n_points=4), linewidth=2.5, edgecolor='yellow', facecolor='none', linestyle='-', label='SR: Star')
        ]),
    ]

    for col, (title, sr_patches_list) in enumerate(sr_definitions):
        # Top Row: QCD Multijet
        ax_qcd = axes_comp[0, col]
        ax_qcd.imshow(
            vals_qcd.T,
            extent=[x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]],
            origin='lower',
            aspect='auto',
            cmap='turbo',
        )
        ax_qcd.add_patch(patches.Rectangle((50, 50), 170, 170, linewidth=2.0, edgecolor='magenta', facecolor='none', linestyle='--', label='SB Box [50,220]'))
        for p in sr_patches_list:
            import copy
            ax_qcd.add_patch(copy.copy(p))
        ax_qcd.set_title(f"QCD Multijet: {title}", fontsize=13, fontweight='bold')
        ax_qcd.set_xlabel("Lead Boson Candidate Mass [GeV]", fontsize=11)
        ax_qcd.set_ylabel("Subl Boson Candidate Mass [GeV]", fontsize=11)
        ax_qcd.set_xlim(0, 250)
        ax_qcd.set_ylim(0, 250)
        ax_qcd.legend(loc='upper right', framealpha=0.8, fontsize=9)

        # Bottom Row: ttHbb Signal
        ax_tth = axes_comp[1, col]
        ax_tth.imshow(
            vals_tth.T,
            extent=[x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]],
            origin='lower',
            aspect='auto',
            cmap='turbo',
        )
        ax_tth.add_patch(patches.Rectangle((50, 50), 170, 170, linewidth=2.0, edgecolor='magenta', facecolor='none', linestyle='--', label='SB Box [50,220]'))
        for p in sr_patches_list:
            import copy
            ax_tth.add_patch(copy.copy(p))
        ax_tth.set_title(f"ttH(bb) Signal: {title}", fontsize=13, fontweight='bold')
        ax_tth.set_xlabel("Lead Boson Candidate Mass [GeV]", fontsize=11)
        ax_tth.set_ylabel("Subl Boson Candidate Mass [GeV]", fontsize=11)
        ax_tth.set_xlim(0, 250)
        ax_tth.set_ylim(0, 250)
        ax_tth.legend(loc='upper right', framealpha=0.8, fontsize=9)

    comp_file = os.path.join(output_path, "ttHbb_sr_options_comparison.png")
    fig_comp.tight_layout()
    fig_comp.savefig(comp_file, dpi=150)
    print(f"Saved options comparison plot to: {comp_file}")
    plt.close(fig_comp)


if __name__ == "__main__":
    coffea_file = sys.argv[1] if len(sys.argv) > 1 else "output/ttHbb/histAll_NoJCM.coffea"
    out_dir = sys.argv[2] if len(sys.argv) > 2 else "output/ttHbb/plots_ttHbb/"
    plot_phase_space(coffea_file, out_dir)
