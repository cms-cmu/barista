#!/usr/bin/env python3
"""
Script to plot Higgs self-coupling modifier kappa_lambda 68% CL expected constraints
projected to higher integrated luminosities (300, 500, 1000 fb^-1) using ROOT and cmsstyle.
"""

import os
import sys
import numpy as np
from array import array

# Ensure ROOT runs in batch mode
import ROOT as rt
rt.gROOT.SetBatch(True)

# Add the directory containing cmsstyle.py to the beginning of sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../stat_analysis/plots")))
import cmsstyle as CMS


def get_projected_bounds(interval_138, lumi_target, lumi_ref=138.0, sm_val=1.0):
    """
    Scale the 68% CL expected interval relative to sm_val (1.0) by sqrt(lumi_ref / lumi_target).
    """
    delta_low = interval_138[0] - sm_val
    delta_high = interval_138[1] - sm_val
    
    factor = np.sqrt(lumi_ref / lumi_target)
    
    low_bound = sm_val + delta_low * factor
    high_bound = sm_val + delta_high * factor
    return low_bound, high_bound


def create_band_graph(lumi_grid, intervals, color, alpha=0.15):
    """
    Create a TGraph representing the shaded band for a given case.
    """
    n_points = len(lumi_grid)
    x_points = array('d', [0.] * (2 * n_points))
    y_points = array('d', [0.] * (2 * n_points))
    
    for i, l in enumerate(lumi_grid):
        low, high = get_projected_bounds(intervals, l)
        # Upper boundary goes from left to right
        x_points[i] = l
        y_points[i] = high
        # Lower boundary goes from right to left
        x_points[2 * n_points - 1 - i] = l
        y_points[2 * n_points - 1 - i] = low
        
    graph = rt.TGraph(2 * n_points, x_points, y_points)
    graph.SetFillColorAlpha(color, alpha)
    graph.SetLineColor(rt.kWhite)
    graph.SetLineWidth(0)
    return graph


def create_boundary_graphs(lumi_grid, intervals, color, line_style=1, line_width=2):
    """
    Create two TGraphs representing the upper and lower boundary lines.
    """
    n_points = len(lumi_grid)
    x_points = array('d', list(lumi_grid))
    y_low = array('d', [0.] * n_points)
    y_high = array('d', [0.] * n_points)
    
    for i, l in enumerate(lumi_grid):
        low, high = get_projected_bounds(intervals, l)
        y_low[i] = low
        y_high[i] = high
        
    g_low = rt.TGraph(n_points, x_points, y_low)
    g_high = rt.TGraph(n_points, x_points, y_high)
    
    for g in [g_low, g_high]:
        g.SetLineColor(color)
        g.SetLineStyle(line_style)
        g.SetLineWidth(line_width)
        g.SetFillStyle(0)
        
    return g_low, g_high


def create_marker_graphs(lumi_points, intervals, color, marker_style, marker_size=1.2):
    """
    Create TGraphs containing only the discrete marker points.
    """
    n_points = len(lumi_points)
    x_points = array('d', list(lumi_points))
    y_low = array('d', [0.] * n_points)
    y_high = array('d', [0.] * n_points)
    
    for i, l in enumerate(lumi_points):
        low, high = get_projected_bounds(intervals, l)
        y_low[i] = low
        y_high[i] = high
        
    g_low = rt.TGraph(n_points, x_points, y_low)
    g_high = rt.TGraph(n_points, x_points, y_high)
    
    for g in [g_low, g_high]:
        g.SetMarkerColor(color)
        g.SetMarkerStyle(marker_style)
        g.SetMarkerSize(marker_size)
        g.SetLineColor(rt.kWhite)
        g.SetLineWidth(0)
        
    return g_low, g_high


def main():
    import argparse
    import json
    
    parser = argparse.ArgumentParser(description="Plot Higgs self-coupling modifier kappa_lambda projections.")
    parser.add_argument("-i", "--input-json", dest="input_json",
                        default="~/workingArea/HH4b/combination/inference-devel/data/store/PlotMultipleUpperLimits/hh_model_run23__model_boosted_run23/multidatacards_730c226afb/m125.0/poi_r/dev/ranges__poi_r__params_r_gghh1.0_r_qqhh1.0_r_vhh1.0_kt1.0_CV1.0_C2V1.0__scan_kl_-15.0_20.0_n36__fzp_allConstrainedNuisances.json",
                        help="Path to input JSON file containing expected limits")
    parser.add_argument("-o", "--output-dir", dest="output_dir",
                        default="output/self_coupling_projection",
                        help="Output directory for plots")
    args = parser.parse_args()
    
    input_path = os.path.expanduser(args.input_json)
    if os.path.exists(input_path):
        print(f"Loading expected limits from {input_path}")
        with open(input_path, 'r') as f:
            data = json.load(f)
        
        intervals = {
            "Nominal": tuple(data["kl__r__expected__Run2Nominal"][0]),
            "Lowpt": tuple(data["kl__r__expected__Run2Lowpt"][0]),
            "Combination": tuple(data["kl__r__expected__Combination"][0])
        }
    else:
        print(f"Warning: Input JSON {input_path} not found. Falling back to default intervals.")
        intervals = {
            "Nominal": (-3.29, 11.02),
            "Lowpt": (-5.33, 12.18),
            "Combination": (-2.74, 10.11)
        }
        
    print("Expected 68% CL intervals at 138/fb:")
    for case, bounds in intervals.items():
        print(f"  {case}: [{bounds[0]:.4f}, {bounds[1]:.4f}]")
    
    # Define colors (using Petroff colors defined or custom)
    colors = {
        "Combination": rt.TColor.GetColor("#e42536"), # Petroff Red
        "Nominal": rt.TColor.GetColor("#5790fc"),     # Petroff Blue
        "Lowpt": rt.TColor.GetColor("#ffa90e")        # Petroff Orange/Yellow
    }
    
    # Marker styles
    marker_styles = {
        "Combination": 20, # Full circle
        "Nominal": 21,     # Full square
        "Lowpt": 22        # Full triangle up
    }
    
    # Discrete projection points
    lumi_points = [138.0, 300.0, 500.0, 1000.0]
    
    # Grid for smooth curves
    lumi_grid = np.linspace(138.0, 1000.0, 200)
    
    # Setup CMS style
    CMS.setCMSStyle()
    CMS.cms_lumi = "13 TeV"
    CMS.cms_energy = ""
    CMS.writeExtraText = False
    CMS.SetCmsText("CMS #font[52]{#scale[0.76]{ Preliminary}}")
    CMS.SetCmsTextFont(62) # Set precision to 2 (62 instead of 61) to allow LaTeX parsing of font and scale
    CMS.ResetAdditionalInfo() # Clear additional info from cmsstyle to prevent empty gaps
    
    # Create canvas
    # X-axis: 0 to 1000 fb^-1
    # Y-axis: -7 to 20 (wider range to avoid text/legend overlap)
    canv = CMS.cmsCanvas(
        "c_self_coupling_projection",
        0.0, 1000.0,
        -7.0, 20.0,
        "Integrated Luminosity [fb^{#minus1}]",
        "68% CL interval for #kappa_{#lambda}",
        square=True,
        extraSpace=0.02
    )
    
    # Adjust X-axis ticks to start at 0 and step by 500 (2 primary divisions)
    # 502 means 2 primary divisions (0, 500, 1000) and 5 secondary divisions within each (steps of 100)
    CMS.GetcmsCanvasHist(canv).GetXaxis().SetNdivisions(-502)
    
    # Draw additional info inside the frame using TLatex
    latex = rt.TLatex()
    latex.SetNDC()
    latex.SetTextFont(42)
    latex.SetTextSize(0.035)
    latex.DrawLatex(0.17, 0.82, "HH #rightarrow 4b")
    latex.DrawLatex(0.17, 0.77, "Projected 68% CL")
    
    # Create graphs
    graphs = {}
    for case in ["Lowpt", "Nominal", "Combination"]:
        color = colors[case]
        m_style = marker_styles[case]
        
        # Band
        band = create_band_graph(lumi_grid, intervals[case], color, alpha=0.15)
        # Set line and marker properties on the band graph for the legend
        band.SetLineColor(color)
        band.SetLineStyle(1)
        band.SetLineWidth(2)
        band.SetMarkerColor(color)
        band.SetMarkerStyle(m_style)
        band.SetMarkerSize(1.3)
        
        # Boundaries
        g_low, g_high = create_boundary_graphs(lumi_grid, intervals[case], color, line_style=1, line_width=2)
        # Markers
        m_low, m_high = create_marker_graphs(lumi_points, intervals[case], color, m_style, marker_size=1.3)
        
        graphs[case] = {
            "band": band,
            "low_line": g_low,
            "high_line": g_high,
            "low_marker": m_low,
            "high_marker": m_high
        }
        
    # Draw order:
    # 1. SM horizontal line
    sm_line = rt.TLine(0.0, 1.0, 1000.0, 1.0)
    sm_line.SetLineColor(rt.kGray+1)
    sm_line.SetLineStyle(2)
    sm_line.SetLineWidth(2)
    sm_line.Draw("same")
    
    # 2. Bands (largest first to smallest last: Lowpt -> Nominal -> Combination)
    for case in ["Lowpt", "Nominal", "Combination"]:
        graphs[case]["band"].Draw("f same")
        
    # 3. Lines
    for case in ["Lowpt", "Nominal", "Combination"]:
        graphs[case]["low_line"].Draw("l same")
        graphs[case]["high_line"].Draw("l same")
        
    # 4. Markers
    for case in ["Lowpt", "Nominal", "Combination"]:
        graphs[case]["low_marker"].Draw("p same")
        graphs[case]["high_marker"].Draw("p same")
        
    # Create Legend
    # Legend coordinates in NDC (shifted right to avoid curves)
    leg = CMS.cmsLeg(0.58, 0.62, 0.90, 0.85, textSize=0.035)
    leg.SetFillStyle(1001)
    leg.SetFillColor(rt.kWhite)
    
    # Add entries (lpf option to draw line, point, and fill)
    leg.AddEntry(graphs["Combination"]["band"], "Combination", "lpf")
    leg.AddEntry(graphs["Nominal"]["band"], "Nominal", "lpf")
    leg.AddEntry(graphs["Lowpt"]["band"], "Low p_{T}", "lpf")
    leg.AddEntry(sm_line, "SM (#kappa_{#lambda} = 1)", "l")
    
    # Redraw axis and frame to make sure tick marks are on top
    canv.RedrawAxis()
    canv.GetFrame().Draw()
    
    # Save the output
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    
    CMS.SaveCanvas(canv, f"{output_dir}/self_coupling_projection.png", close=False)
    CMS.SaveCanvas(canv, f"{output_dir}/self_coupling_projection.pdf", close=True)
    
    print(f"Plots successfully saved to {output_dir}/")


if __name__ == '__main__':
    main()
