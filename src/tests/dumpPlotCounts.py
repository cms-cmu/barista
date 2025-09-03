import argparse
from coffea.util import load
import sys
import os
import numpy as np
sys.path.insert(0, os.getcwd())
from src.plotting.plots import makePlot, load_config, load_hists, read_axes_and_cuts
import src.plotting.helpers as plot_helpers
import src.plotting.iPlot_config as cfg
import matplotlib.pyplot as plt

def print_counts_yaml(var, cut, region, counts):

    outputFile.write(f"{'_'.join([var,cut,region])}:\n")
    outputFile.write(f"    var:\n")
    outputFile.write(f"        {var}\n")
    outputFile.write(f"    cut:\n")
    outputFile.write(f"        {cut}\n")
    outputFile.write(f"    region:\n")
    outputFile.write(f"           {region}\n")
    outputFile.write(f"    counts:\n")
    outputFile.write(f"           {counts.tolist()}\n")
    outputFile.write("\n\n")


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='uproot_plots')
    parser.add_argument('-i','--inputFile', default='hists.pkl', help='Input File. Default: hists.pkl')
    parser.add_argument('-o','--outputFile', default='knownCounts.yml', help='Input File. Default: hists.pkl')
    args = parser.parse_args()

    outputFile = open(f'{args.outputFile}', 'w')

    metadata = "coffea4bees/plots/metadata/plotsAll.yml"
    cfg.plotConfig = load_config(metadata)
    cfg.hists = load_hists([args.inputFile])
    cfg.axisLabels, cfg.cutList = read_axes_and_cuts(cfg.hists, cfg.plotConfig)

    default_args = {"doRatio":0, "rebin":4, "norm":0, "process":"Multijet"}

    test_vectors = [("SvB_MA.ps", "passPreSel", "SR"),
                    ("SvB_MA.ps", "passPreSel", "SB"),
                    ("SvB_MA.ps", "passSvB",    "SR"),
                    ("SvB_MA.ps", "passSvB",    "SB"),
                    ("SvB_MA.ps", "failSvB",    "SR"),
                    ("SvB_MA.ps", "failSvB",    "SB"),

                    ("SvB_MA.ps_hh", "passPreSel", "SR"),
                    ("SvB_MA.ps_hh", "passPreSel", "SB"),
                    ("SvB_MA.ps_hh", "passSvB",    "SR"),
                    ("SvB_MA.ps_hh", "passSvB",    "SB"),
                    ("SvB_MA.ps_hh", "failSvB",    "SR"),
                    ("SvB_MA.ps_hh", "failSvB",    "SB"),

                    ("SvB_MA.ps_zh", "passPreSel", "SR"),
                    ("SvB_MA.ps_zh", "passPreSel", "SB"),
                    ("SvB_MA.ps_zh", "passSvB",    "SR"),
                    ("SvB_MA.ps_zh", "passSvB",    "SB"),
                    ("SvB_MA.ps_zh", "failSvB",    "SR"),
                    ("SvB_MA.ps_zh", "failSvB",    "SB"),

                    ("SvB_MA.ps_zz", "passPreSel", "SR"),
                    ("SvB_MA.ps_zz", "passPreSel", "SB"),
                    ("SvB_MA.ps_zz", "passSvB",    "SR"),
                    ("SvB_MA.ps_zz", "passSvB",    "SB"),
                    ("SvB_MA.ps_zz", "failSvB",    "SR"),
                    ("SvB_MA.ps_zz", "failSvB",    "SB"),


                    ]

    for tv in test_vectors:

        var    = tv[0]
        cut    = tv[1]
        region = tv[2]
        print(f"testing {var}, {cut}, {region}")
        fig, axes = makePlot(cfg, var=var, cut=cut, region=region,
                             outputFolder=cfg.outputFolder, **default_args)

        ax = axes[0]
        for i in range(len(ax.lines)):

            if hasattr(ax.lines[i], "get_label") and ax.lines[i].get_label() == '_nolegend_':
                counts = ax.lines[i].get_ydata()
                break

        print_counts_yaml(var, cut, region, counts)
        plt.close()
