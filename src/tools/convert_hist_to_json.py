import os, sys
import argparse
import logging
import json
import tempfile
import numpy as np
from coffea.util import load

def extract_hist_data(h, base_selection, custom_axes, axis_indices, edges, centers):
    """
    Extracts 1D histogram data from a multi-dimensional hist.Hist object using fast numpy slicing.
    """
    values_all = h.values(flow=True)
    variances_all = h.variances(flow=True)
    if variances_all is None:
        variances_all = np.zeros_like(values_all)
        
    num_axes = len(h.axes)
    
    # Construct base slicing list
    base_slicing = [slice(None)] * num_axes
    for name, val in base_selection.items():
        if name in axis_indices:
            base_slicing[axis_indices[name]] = h.axes[name].index(val)
            
    # Helper to slice, sum over remaining dimensions, and build the 1D dict
    def get_sliced_1d(slicing_dict):
        slicing = list(base_slicing)
        for name, val in slicing_dict.items():
            if name in axis_indices:
                slicing[axis_indices[name]] = h.axes[name].index(val)
                
        v_slice = values_all[tuple(slicing)]
        var_slice = variances_all[tuple(slicing)]
        
        # Sum over any remaining dimensions except the last one (variable/coordinate axis)
        if v_slice.ndim > 1:
            axes_to_sum = tuple(range(v_slice.ndim - 1))
            v_slice = np.sum(v_slice, axis=axes_to_sum)
            var_slice = np.sum(var_slice, axis=axes_to_sum)
            
        underflow_val = float(v_slice[0])
        underflow_var = float(var_slice[0])
        overflow_val = float(v_slice[-1])
        overflow_var = float(var_slice[-1])
        
        values_list = v_slice[1:-1].tolist()
        variances_list = var_slice[1:-1].tolist()
        
        return {
            'edges': edges,
            'centers': centers,
            'values': values_list,
            'variances': variances_list,
            'underflow_value': underflow_val,
            'underflow_variance': underflow_var,
            'overflow_value': overflow_val,
            'overflow_variance': overflow_var,
        }

    results = {}
    
    # 1. Base case: sum/OR over all custom axes
    results[''] = get_sliced_1d({})
    
    # 2. Slice each custom axis to pass (True) or fail (False)
    for ax in custom_axes:
        base_name = ax
        if ax.startswith("pass_"):
            base_name = ax[5:]
        elif ax.startswith("fail_"):
            base_name = ax[5:]
        results[f"_pass_{base_name}"] = get_sliced_1d({ax: True})
        results[f"_fail_{base_name}"] = get_sliced_1d({ax: False})
        
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert coffea hist to JSON',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--histos', dest="histos", nargs="+", default=None,
                        help='List of histograms to convert. If not provided, automatically detects all histograms.')
    parser.add_argument('-o', '--output', dest="output",
                        default="./histos/histAll.json", help='Output file and directory.')
    parser.add_argument('-i', '--input_file', dest='input_file',
                        default="../analysis/hists/histAll.coffea", help="File with coffea hists")
    parser.add_argument('-s', '--syst_file', dest='systematics_file', action='store_true',
                        default=False, help="File contains systematic variations")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    logging.info(f"\nRunning with these parameters: {args}")

    coffea_data = load(args.input_file)
    coffea_hists = coffea_data["hists"] if "hists" in coffea_data else coffea_data
    json_dict = {}

    # If no histograms specified, auto-detect all histograms in the file
    if not args.histos:
        args.histos = [k for k, v in coffea_hists.items() if hasattr(v, 'axes')]
        logging.info(f"Automatically detected histograms to convert: {args.histos}")

    for ih in args.histos:
        if ih not in coffea_hists:
            logging.warning(f"Histogram {ih} not found in input file, skipping.")
            continue
            
        h = coffea_hists[ih]
        json_dict[ih] = {}
        
        # Precompute axis mappings
        axis_indices = {ax.name: idx for idx, ax in enumerate(h.axes)}
        standard_axes = {'process', 'year', 'tag', 'region', 'variation'}
        custom_axes = [ax.name for ax in h.axes[:-1] if ax.name not in standard_axes]
        
        # Precompute edges and centers for the variable axis (always the last axis)
        var_axis = h.axes[-1]
        edges = var_axis.edges.tolist()
        centers = var_axis.centers.tolist()
        
        # Identify ranges for selection axes (fallback to ["nominal"] if missing)
        processes = list(h.axes['process']) if 'process' in axis_indices else ["nominal"]
        years = list(h.axes['year']) if 'year' in axis_indices else ["nominal"]
        tags = list(h.axes['tag']) if 'tag' in axis_indices else ["nominal"]
        regions = list(h.axes['region']) if 'region' in axis_indices else ["nominal"]
        variations = list(h.axes['variation']) if 'variation' in axis_indices else ["nominal"]

        if not args.systematics_file:
            for iprocess in processes:
                json_dict[ih][iprocess] = {}
                for iy in years:
                    json_dict[ih][iprocess][iy] = {}
                    for itag in tags:
                        json_dict[ih][iprocess][iy][itag] = {}
                        
                        # Populate regions
                        for iregion in regions:
                            selection = {}
                            if 'process' in axis_indices: selection['process'] = iprocess
                            if 'year' in axis_indices: selection['year'] = iy
                            if 'tag' in axis_indices: selection['tag'] = itag
                            if 'region' in axis_indices: selection['region'] = iregion
                            
                            region_hists = extract_hist_data(h, selection, custom_axes, axis_indices, edges, centers)
                            for suffix, data in region_hists.items():
                                json_dict[ih][iprocess][iy][itag][f"{iregion}{suffix}"] = data
        else:
            for iprocess in processes:
                json_dict[ih][iprocess] = {}
                for iy in years:
                    json_dict[ih][iprocess][iy] = {}
                    for ivar in variations:
                        json_dict[ih][iprocess][iy][ivar] = {}
                        for itag in tags:
                            json_dict[ih][iprocess][iy][ivar][itag] = {}
                            
                            # Populate regions
                            for iregion in regions:
                                selection = {}
                                if 'process' in axis_indices: selection['process'] = iprocess
                                if 'year' in axis_indices: selection['year'] = iy
                                if 'variation' in axis_indices: selection['variation'] = ivar
                                if 'tag' in axis_indices: selection['tag'] = itag
                                if 'region' in axis_indices: selection['region'] = iregion
                                
                                region_hists = extract_hist_data(h, selection, custom_axes, axis_indices, edges, centers)
                                for suffix, data in region_hists.items():
                                    json_dict[ih][iprocess][iy][ivar][itag][f"{iregion}{suffix}"] = data

    logging.info(f"Saving histos in json format in {args.output}")
    output_dir = '/'.join(args.output.split('/')[:-1])
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    tmp_fd, tmp_path = tempfile.mkstemp(dir=output_dir or '.', suffix='.json.tmp')
    try:
        with os.fdopen(tmp_fd, 'w') as f:
            json.dump(json_dict, f)
        os.replace(tmp_path, args.output)
    except:
        os.unlink(tmp_path)
        raise
