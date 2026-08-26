#!/usr/bin/env python3
"""
Plot 2D Phase Space up to 1 TeV for ttH(bb) Signal with current SR/SB and alternative proposals.
"""

import os
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.path import Path
from matplotlib.colors import LogNorm
import numpy as np
from coffea.util import load

def get_2d_hist(h_obj, process="ttHbb", tag="fourTag", region="inclusive"):
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

def plot_1tev(coffea_file: str, output_dir: str = "tmp/plots_signal_study/"):
    os.makedirs(output_dir, exist_ok=True)
    print(f"Loading {coffea_file}...")
    data = load(coffea_file)
    hist_collection = data.get("hists", data)

    h2d_name = None
    for k in hist_collection.keys():
        if "lead_vs_subl_m" in k and "selected" in k:
            h2d_name = k
            break
        elif "lead_vs_subl_m" in k:
            h2d_name = k
    
    if not h2d_name:
        h2d_name = list(hist_collection.keys())[0]

    print(f"Using histogram: {h2d_name}")
    h = hist_collection[h2d_name]

    vals_tth, x_edges, y_edges = get_2d_hist(h, process="ttHbb", tag="fourTag", region="inclusive")

    # Center coordinates of each 2D bin
    x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])
    y_centers = 0.5 * (y_edges[:-1] + y_edges[1:])
    X, Y = np.meshgrid(x_centers, y_centers, indexing='ij')

    total_yield = np.sum(vals_tth)

    # Current Diamond SR:
    # dx = (m_lead - 125)/45 if m_lead >= 125 else (125 - m_lead)/25
    # dy = (m_subl - 125)/35 if m_subl >= 125 else (125 - m_subl)/45
    dx = np.where(X >= 125.0, (X - 125.0)/45.0, (125.0 - X)/25.0)
    dy = np.where(Y >= 125.0, (Y - 125.0)/35.0, (125.0 - Y)/45.0)
    mask_sr = (dx + dy <= 1.0) & (dx >= 0) & (dy >= 0)

    # Current SB Box: [70, 200] x [50, 180] \ SR
    mask_sb1 = (X >= 70.0) & (X <= 200.0) & (Y >= 50.0) & (Y <= 180.0) & (~mask_sr)

    # SB Box [50, 220] x [50, 220] \ SR
    mask_sb2 = (X >= 50.0) & (X <= 220.0) & (Y >= 50.0) & (Y <= 220.0) & (~mask_sr)

    # Mass threshold regions
    mask_200 = (X <= 200.0) & (Y <= 200.0)
    mask_250 = (X <= 250.0) & (Y <= 250.0)
    mask_500 = (X <= 500.0) & (Y <= 500.0)
    mask_out250 = ~mask_250
    mask_out500 = ~mask_500

    y_sr = np.sum(vals_tth[mask_sr])
    y_sb1 = np.sum(vals_tth[mask_sb1])
    y_sb2 = np.sum(vals_tth[mask_sb2])
    y_200 = np.sum(vals_tth[mask_200])
    y_250 = np.sum(vals_tth[mask_250])
    y_500 = np.sum(vals_tth[mask_500])
    y_out250 = np.sum(vals_tth[mask_out250])
    y_out500 = np.sum(vals_tth[mask_out500])

    print("\n=======================================================")
    print(f"Total ttHbb (4-tag, inclusive) yield in 2D space: {total_yield:.2f}")
    print("=======================================================")
    print(f"Inside Current Diamond SR:         {y_sr:10.2f} ({y_sr/total_yield*100:5.1f}%)")
    print(f"Inside SB Box [70,200]x[50,180]:   {y_sb1:10.2f} ({y_sb1/total_yield*100:5.1f}%) -> SB / SR ratio: {y_sb1/y_sr:.2f}")
    print(f"Inside SB Box [50,220]x[50,220]:   {y_sb2:10.2f} ({y_sb2/total_yield*100:5.1f}%) -> SB / SR ratio: {y_sb2/y_sr:.2f}")
    print(f"Inside [0, 200] x [0, 200] GeV:    {y_200:10.2f} ({y_200/total_yield*100:5.1f}%)")
    print(f"Inside [0, 250] x [0, 250] GeV:    {y_250:10.2f} ({y_250/total_yield*100:5.1f}%)")
    print(f"Outside 250 GeV:                   {y_out250:10.2f} ({y_out250/total_yield*100:5.1f}%)")
    print(f"Inside [0, 500] x [0, 500] GeV:    {y_500:10.2f} ({y_500/total_yield*100:5.1f}%)")
    print(f"Outside 500 GeV:                   {y_out500:10.2f} ({y_out500/total_yield*100:5.1f}%)")

    # 1. 2D Plot Extended to 1 TeV
    fig, ax = plt.subplots(figsize=(10, 8.5))
    v_pos = vals_tth[vals_tth > 0]
    vmin = np.percentile(v_pos, 5) if len(v_pos) > 0 else 1e-4
    vmax = np.max(vals_tth)

    im = ax.imshow(
        vals_tth.T,
        extent=[x_edges[0], x_edges[-1], y_edges[0], y_edges[-1]],
        origin='lower',
        aspect='auto',
        cmap='turbo',
        norm=LogNorm(vmin=max(1e-3, vmin), vmax=vmax),
    )
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(r"$\mathrm{t\bar{t}H(b\bar{b})}$ Signal Yield (Log Scale)", fontsize=12)

    diamond_x = [125, 170, 125, 100, 125]
    diamond_y = [160, 125, 80, 125, 160]
    ax.plot(diamond_x, diamond_y, color='red', lw=2.5, label='Current SR Diamond')

    rect_sb = patches.Rectangle((50, 50), 170, 170, linewidth=2, edgecolor='lime', facecolor='none', linestyle='--', label='Current SB Box [50, 220] GeV')
    ax.add_patch(rect_sb)

    ax.axvline(250, color='white', linestyle=':', lw=1.5, label='Previous 250 GeV Cutoff')
    ax.axhline(250, color='white', linestyle=':', lw=1.5)

    ax.set_title(r"$\mathrm{t\bar{t}H(b\bar{b})}$ Signal 2D Phase Space up to 1 TeV", fontsize=14, fontweight='bold')
    ax.set_xlabel(r"Lead Dijet Candidate Mass $m_{\mathrm{lead}}$ [GeV]", fontsize=12)
    ax.set_ylabel(r"Subl Dijet Candidate Mass $m_{\mathrm{subl}}$ [GeV]", fontsize=12)
    ax.set_xlim(0, 1000)
    ax.set_ylim(0, 1000)
    ax.legend(loc='upper right', fontsize=11)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "ttHbb_phase_space_1TeV.png"), dpi=300)
    plt.close(fig)

    # 2. 2D Plot Zoomed to 500 GeV
    fig, ax = plt.subplots(figsize=(10, 8.5))
    mask_x500 = x_edges <= 500
    mask_y500 = y_edges <= 500
    sub_vals = vals_tth[:sum(mask_x500)-1, :sum(mask_y500)-1]

    im = ax.imshow(
        sub_vals.T,
        extent=[0, 500, 0, 500],
        origin='lower',
        aspect='auto',
        cmap='turbo',
        norm=LogNorm(vmin=max(1e-2, vmin), vmax=vmax),
    )
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(r"$\mathrm{t\bar{t}H(b\bar{b})}$ Signal Yield (Log Scale)", fontsize=12)

    ax.plot(diamond_x, diamond_y, color='red', lw=2.5, label='Current SR Diamond')
    rect_sb = patches.Rectangle((50, 50), 170, 170, linewidth=2, edgecolor='lime', facecolor='none', linestyle='--', label='Current SB Box [50, 220] GeV')
    ax.add_patch(rect_sb)
    ax.axvline(250, color='white', linestyle=':', lw=1.5, label='250 GeV Boundary')
    ax.axhline(250, color='white', linestyle=':', lw=1.5)

    ax.set_title(r"$\mathrm{t\bar{t}H(b\bar{b})}$ Signal 2D Phase Space up to 500 GeV", fontsize=14, fontweight='bold')
    ax.set_xlabel(r"Lead Dijet Candidate Mass $m_{\mathrm{lead}}$ [GeV]", fontsize=12)
    ax.set_ylabel(r"Subl Dijet Candidate Mass $m_{\mathrm{subl}}$ [GeV]", fontsize=12)
    ax.set_xlim(0, 500)
    ax.set_ylim(0, 500)
    ax.legend(loc='upper right', fontsize=11)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "ttHbb_phase_space_500GeV.png"), dpi=300)
    plt.close(fig)

    # 3. 1D Projections
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    lead_1d = np.sum(vals_tth, axis=1)
    subl_1d = np.sum(vals_tth, axis=0)

    axes[0].step(x_centers, lead_1d, where='mid', color='royalblue', lw=2, label=r'$m_{\mathrm{lead}}$')
    axes[0].fill_between(x_centers, lead_1d, step='mid', color='royalblue', alpha=0.3)
    axes[0].axvline(125, color='red', linestyle='--', label=r'$m_H = 125$ GeV')
    axes[0].axvline(250, color='gray', linestyle=':', label='250 GeV cutoff')
    axes[0].set_title(r"Lead Dijet Mass $m_{\mathrm{lead}}$ Projection", fontsize=13, fontweight='bold')
    axes[0].set_xlabel("Mass [GeV]", fontsize=11)
    axes[0].set_ylabel("Yield / 10 GeV", fontsize=11)
    axes[0].set_yscale('log')
    axes[0].set_xlim(0, 1000)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].step(y_centers, subl_1d, where='mid', color='coral', lw=2, label=r'$m_{\mathrm{subl}}$')
    axes[1].fill_between(y_centers, subl_1d, step='mid', color='coral', alpha=0.3)
    axes[1].axvline(125, color='red', linestyle='--', label=r'$m_H = 125$ GeV')
    axes[1].axvline(250, color='gray', linestyle=':', label='250 GeV cutoff')
    axes[1].set_title(r"Subl Dijet Mass $m_{\mathrm{subl}}$ Projection", fontsize=13, fontweight='bold')
    axes[1].set_xlabel("Mass [GeV]", fontsize=11)
    axes[1].set_ylabel("Yield / 10 GeV", fontsize=11)
    axes[1].set_yscale('log')
    axes[1].set_xlim(0, 1000)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "ttHbb_1D_projections_1TeV.png"), dpi=300)
    plt.close(fig)

    print("Plots saved in", output_dir)

if __name__ == "__main__":
    if len(sys.argv) > 1:
        cfile = sys.argv[1]
    else:
        cfile = "tmp/histAll_ttHbb_1TeV.coffea"
    plot_1tev(cfile)
