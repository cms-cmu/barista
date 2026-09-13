#!/usr/bin/env python3
import os
import json
import array
import argparse
import ROOT

def main():
    parser = argparse.ArgumentParser(description="Create signal ROOT histograms from stitched JSON")
    parser.add_argument("-i", "--input", default="output/ttHbb_stitched/histAll_ttHbb_stitched.json",
                        help="Input JSON file containing ttHbb")
    parser.add_argument("-o", "--output", default="output/ttHbb_mixeddata_stitched_closure/root_inputs/hist_signal_ttHbb.root",
                        help="Output ROOT file for closure test")
    parser.add_argument("--var", default="SvB_MA.ps", help="Variable name in JSON")
    args = parser.parse_args()

    with open(args.input) as f:
        d = json.load(f)

    if args.var not in d:
        raise KeyError(f"Variable {args.var} not found in {args.input}. Keys: {list(d.keys())}")

    svb = d[args.var]
    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    var_prefix = args.var.replace(".", "_")
    f_out = ROOT.TFile(args.output, "RECREATE")
    years = ["UL16_preVFP", "UL16_postVFP", "UL17", "UL18"]

    tot_integral = 0.0
    for y in years:
        if y not in svb["ttHbb"]:
            continue
        dat = svb["ttHbb"][y]["fourTag"]["SR"]
        edges = dat["edges"]
        vals = dat["values"]
        vars_ = dat["variances"]

        h = ROOT.TH1F(f"{var_prefix}_ttHbb_{y}_fourTag_SR", f"{var_prefix}_ttHbb_{y}_fourTag_SR",
                      len(edges) - 1, array.array("d", edges))
        for b in range(1, len(edges)):
            h.SetBinContent(b, vals[b - 1])
            h.SetBinError(b, vars_[b - 1] ** 0.5)
        h.Write()
        tot_integral += h.Integral()
        print(f"  {var_prefix}_ttHbb_{y}_fourTag_SR: {h.Integral():.2f}")

    f_out.Close()
    print(f"Successfully created {args.output} with total signal integral = {tot_integral:.2f}")

if __name__ == "__main__":
    main()
