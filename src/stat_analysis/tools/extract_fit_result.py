#!/usr/bin/env python3
# coding: utf-8

"""
Script to extract fitted parameters, uncertainties, and covariance/correlation matrices
from a RooFitResult or RooWorkspace snapshot into a clean JSON file.

Example usage:
    python3 src/stat_analysis/tools/extract_fit_result.py fitDiagnostics.root fit_s -o fit_result.json
    python3 src/stat_analysis/tools/extract_fit_result.py fitDiagnostics.root fit_s -o fit_result.json --keep "CMS_*"
"""

import os
import sys
import json
import argparse
import fnmatch
from collections import OrderedDict


def multi_match(val, patterns):
    if not patterns:
        return True
    if isinstance(patterns, str):
        patterns = [patterns]
    for p in patterns:
        if fnmatch.fnmatch(val, p):
            return True
    return False


def matrix_to_list(matrix):
    import numpy as np
    cls = matrix.ClassName()
    dtype = np.float64 if "double" in cls.lower() else np.float32
    arr = np.frombuffer(matrix.GetMatrixArray(), dtype=dtype, count=matrix.GetNoElements())
    return arr.reshape((matrix.GetNrows(), matrix.GetNcols())).tolist()


def extract_fit_result(input_file, obj_name, output_file=None, keep_patterns=None, skip_patterns=None):
    import ROOT
    ROOT.gROOT.SetBatch(True)

    tfile = ROOT.TFile.Open(os.path.abspath(input_file), "READ")
    if not tfile or tfile.IsZombie():
        raise FileNotFoundError(f"Cannot open ROOT file: {input_file}")

    snapshot_name = None
    if ":" in obj_name:
        obj_name, snapshot_name = obj_name.split(":", 1)

    tobj = tfile.Get(obj_name)
    if not tobj:
        tfile.Close()
        raise ValueError(f"Object '{obj_name}' not found in {input_file}")

    is_workspace = isinstance(tobj, ROOT.RooWorkspace)
    is_fitresult = isinstance(tobj, ROOT.RooFitResult)

    if not is_workspace and not is_fitresult:
        tfile.Close()
        raise TypeError(f"Object '{obj_name}' is neither RooWorkspace nor RooFitResult")

    if is_workspace and snapshot_name:
        tobj.loadSnapshot(snapshot_name)

    if is_workspace:
        var_set = tobj.allVars()
        it = var_set.createIterator()
    else:
        var_list = tobj.floatParsFinal()
        it = var_list.createIterator()

    params_data = OrderedDict()
    param = it.Next()
    while param:
        name = param.GetName()
        keep = multi_match(name, keep_patterns)
        if keep and skip_patterns:
            keep = not multi_match(name, skip_patterns)

        if keep:
            p_dict = {
                "value": param.getValV(),
                "min": param.getMin(),
                "max": param.getMax(),
                "error": getattr(param, "getError", lambda: 0.0)(),
            }
            if hasattr(param, "getAsymErrorLo") and hasattr(param, "getAsymErrorHi"):
                p_dict["error_lo"] = param.getAsymErrorLo()
                p_dict["error_hi"] = param.getAsymErrorHi()
            params_data[name] = p_dict

        param = it.Next()

    result = {
        "file": os.path.basename(input_file),
        "object": obj_name,
        "parameters": params_data,
    }

    if is_fitresult:
        result["status"] = int(tobj.status())
        result["cov_qual"] = int(tobj.covQual())
        pars = tobj.floatParsFinal()
        labels = [pars[i].GetName() for i in range(len(pars))]
        result["matrix_labels"] = labels
        result["covariance"] = matrix_to_list(tobj.covarianceMatrix())
        result["correlation"] = matrix_to_list(tobj.correlationMatrix())

    tfile.Close()

    if output_file:
        out_path = os.path.abspath(output_file)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2)
        print(f"Extracted {len(params_data)} parameters to: {out_path}")
    else:
        print(json.dumps(result, indent=2))

    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input", help="Path to ROOT file")
    parser.add_argument("object", help="Name of RooFitResult (e.g. 'fit_s', 'fit_b') or RooWorkspace (e.g. 'w:MultiDimFit')")
    parser.add_argument("-o", "--output", help="Output JSON path")
    parser.add_argument("--keep", nargs="+", help="Wildcard patterns of parameters to keep")
    parser.add_argument("--skip", nargs="+", help="Wildcard patterns of parameters to skip")

    args = parser.parse_args()
    extract_fit_result(
        input_file=args.input,
        obj_name=args.object,
        output_file=args.output,
        keep_patterns=args.keep,
        skip_patterns=args.skip,
    )


if __name__ == "__main__":
    main()
