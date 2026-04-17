"""
Convert coffea histogram files (.coffea) to a JSON format suitable for
downstream combine inputs (make_combine_inputs.py) or general inspection.

Each histogram is written as a nested dict mirroring its axes, with the
innermost leaf being the dict produced by hist_to_json():

    {
        "edges":              [...],
        "centers":            [...],
        "values":             [...],
        "variances":          [...],
        "underflow_value":    float,
        "underflow_variance": float,
        "overflow_value":     float,
        "overflow_variance":  float,
    }

Axis categories are written as their native string (StrCategory) or integer
(IntCategory) values — no mapping is needed since barista Collection axes
already use human-readable string labels.

Axis handling:
  --select AXIS=VALUE   Fix an axis to one value (not iterated, not summed).
  --sum AXIS            Sum over all values of an axis (collapses it).
  Boolean axes          Always summed unless listed in --select.
  All other axes        Iterated — each value becomes a nested dict key.

Usage:
    # Convert all histograms, Boolean axes summed automatically
    python src/tools/convert_coffea_to_json.py \\
        -i analysis/hists/histAll.coffea \\
        -o histos/histAll.json

    # Select specific histograms
    python src/tools/convert_coffea_to_json.py \\
        -i analysis/hists/histAll.coffea \\
        -o histos/histAll.json \\
        --histos SvB.phh SvB.ptt

    # Fix Boolean axis to True (coffea4bees style)
    python src/tools/convert_coffea_to_json.py \\
        -i analysis/hists/histAll.coffea \\
        -o histos/histAll.json \\
        --select passPreSel=True

    # Select only SR, sum SR+CR together, or iterate both separately
    python src/tools/convert_coffea_to_json.py ... --select region=SR   # SR only
    python src/tools/convert_coffea_to_json.py ... --sum region          # SR+CR merged
    python src/tools/convert_coffea_to_json.py ... (no flag)             # SR and CR as separate keys

    # IntCategory axes with integer bin values — provide a mapping file
    python src/tools/convert_coffea_to_json.py \\
        -i analysis/hists/histAll.coffea \\
        -o histos/histAll.json \\
        --mapping-config my_mapping.json
"""

import os
import argparse
import logging
import json
import numpy as np
from coffea.util import load
import hist as hist_module


def hist_to_json(coffea_hist):
    """Convert a 1-D coffea histogram to a JSON-serialisable dict.

    Parameters
    ----------
    coffea_hist : hist.Hist
        A histogram with exactly one remaining (variable/regular) axis after
        all category axes have been sliced away.

    Returns
    -------
    dict
    """
    main_axis = coffea_hist.axes[0]
    return {
        'edges':              main_axis.edges.tolist(),
        'centers':            main_axis.centers.tolist(),
        'values':             coffea_hist.values().tolist(),
        'variances':          coffea_hist.variances().tolist(),
        'underflow_value':    float(coffea_hist[hist_module.loc(-np.inf)].value),
        'underflow_variance': float(coffea_hist[hist_module.loc(-np.inf)].variance),
        'overflow_value':     float(coffea_hist[hist_module.loc(+np.inf)].value),
        'overflow_variance':  float(coffea_hist[hist_module.loc(+np.inf)].variance),
    }


def _parse_select_value(raw):
    """Convert a CLI --select value string to the appropriate Python type."""
    if raw == 'True':  return True
    if raw == 'False': return False
    try:
        return int(raw)
    except ValueError:
        pass
    return raw


def convert_histogram(hist_obj, fixed_axes, sum_axes=None, category_maps=None):
    """Recursively convert a coffea histogram to a nested dict.

    Axes are handled as follows (in priority order):
      1. In fixed_axes  → sliced to the given value, not a nesting level.
      2. In sum_axes    → summed over all values, not a nesting level.
      3. Boolean        → summed over automatically (same as sum_axes).
      4. Everything else → iterated; each value becomes a nested dict key.
    The final Regular/Variable axis is left for hist_to_json() to serialise.

    Parameters
    ----------
    hist_obj : hist.Hist
    fixed_axes : dict
        Axis name → value to fix (e.g. {'passPreSel': True, 'region': 'SR'}).
    sum_axes : list of str or None
        Axis names to sum over (e.g. ['region'] to merge SR+CR).
    category_maps : dict or None
        Optional mapping of IntCategory int values to string keys.
        Format: {axis_name: {int_value: str_key, ...}}
        Example: {'tag': {0: 'threeTag', 1: 'fourTag', 2: 'other'},
                  'region': {0: 'SR', 1: 'SB', 2: 'other'}}
        StrCategory axes are unaffected (already strings).

    Returns
    -------
    dict
    """
    if sum_axes is None:
        sum_axes = []
    if category_maps is None:
        category_maps = {}

    def _is_variable_axis(axis):
        """True for Regular/Variable axes (the one to histogram over)."""
        return type(axis).__name__ in ('Regular', 'Variable')

    def _is_boolean_axis(axis):
        """True for Boolean axes (created with ... in Collection)."""
        return type(axis).__name__ == 'Boolean'

    # Classify each axis
    iter_axes  = []  # (axis_name, [(raw_val, json_key), ...]) — become nested keys
    slice_axes = {}  # axis_name → fixed value
    axes_to_sum = [] # axis names to collapse with sum

    for axis in hist_obj.axes:
        name = axis.name
        if _is_variable_axis(axis):
            # Variable axis — leave for hist_to_json
            continue
        if name in fixed_axes:
            slice_axes[name] = fixed_axes[name]
        elif name in sum_axes or _is_boolean_axis(axis):
            axes_to_sum.append(name)
        else:
            mapping = category_maps.get(name, {})
            pairs = []
            for val in axis:
                json_key = val if isinstance(val, str) else mapping.get(val, str(val))
                pairs.append((val, json_key))
            iter_axes.append((name, pairs))

    def _recurse(path_dict, remaining_iter, current_selection):
        if not remaining_iter:
            selection = dict(current_selection)
            for name, val in slice_axes.items():
                selection[name] = val
            for name in axes_to_sum:
                selection[name] = sum
            try:
                sliced = hist_obj[selection]
                path_dict.update(hist_to_json(sliced))
            except Exception as e:
                logging.warning(f"Could not extract slice {selection}: {e}")
            return

        axis_name, pairs = remaining_iter[0]
        rest = remaining_iter[1:]
        for raw_val, json_key in pairs:
            path_dict[json_key] = {}
            _recurse(path_dict[json_key], rest, {**current_selection, axis_name: raw_val})

    result = {}
    _recurse(result, iter_axes, {})
    return result


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='Convert coffea histograms to JSON',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-i', '--input_file', dest='input_file', required=True,
                        help='Input .coffea file')
    parser.add_argument('--collection', dest='collection', default='hists',
                        help='Top-level key in the coffea file to read histograms from '
                             '(default: hists). Use e.g. hists_4j2b for the 4j2b collection.')
    parser.add_argument('-o', '--output', dest='output',
                        default='./histos/histAll.json',
                        help='Output JSON file path')
    parser.add_argument('--histos', dest='histos', nargs='+', default=None,
                        help='Histogram names to convert (default: all)')
    parser.add_argument('--select', dest='select', nargs='+', default=[],
                        metavar='AXIS=VALUE',
                        help='Fix an axis to one value instead of iterating. '
                             'e.g. --select region=SR passPreSel=True. '
                             'Boolean values: True/False.')
    parser.add_argument('--sum', dest='sum_axes', nargs='+', default=[],
                        metavar='AXIS',
                        help='Sum over all values of an axis, collapsing it. '
                             'e.g. --sum region  merges SR+CR into one histogram. '
                             'Boolean axes are always summed unless in --select.')
    parser.add_argument('--mapping-config', dest='mapping_config', default=None,
                        metavar='FILE',
                        help='JSON file mapping IntCategory int values to string keys. '
                             'Format: {"axis_name": {int_val: "string_key", ...}}. '
                             'StrCategory axes are unaffected.')
    parser.add_argument('-v', '--verbose', action='store_true', default=False,
                        help='Debug-level logging')
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format='%(levelname)s:%(message)s')
    logging.info(f"Running with parameters: {args}")

    # Parse --select AXIS=VALUE pairs
    fixed_axes = {}
    for item in args.select:
        if '=' not in item:
            parser.error(f"--select values must be AXIS=VALUE, got: {item!r}")
        name, raw_val = item.split('=', 1)
        fixed_axes[name] = _parse_select_value(raw_val)
    if fixed_axes:
        logging.info(f"Fixed axes: {fixed_axes}")

    if args.sum_axes:
        overlap = set(args.sum_axes) & set(fixed_axes)
        if overlap:
            parser.error(f"Axes cannot appear in both --select and --sum: {overlap}")
        logging.info(f"Summed axes: {args.sum_axes}")

    # Load optional integer-to-string category mappings
    category_maps = {}
    if args.mapping_config:
        with open(args.mapping_config, 'r') as f:
            raw = json.load(f)
        for axis_name, mapping in raw.items():
            category_maps[axis_name] = {}
            for k, v in mapping.items():
                try:
                    category_maps[axis_name][int(k)] = v
                except ValueError:
                    category_maps[axis_name][k] = v
        logging.info(f"Category mappings loaded for axes: {list(category_maps.keys())}")

    logging.info(f"Loading {args.input_file}")
    coffea_data = load(args.input_file)
    if args.collection in coffea_data:
        coffea_hists = coffea_data[args.collection]
    else:
        available = [k for k in coffea_data.keys() if 'hists' in k]
        logging.warning(f"Collection '{args.collection}' not found. "
                        f"Available hist collections: {available}. Falling back to raw file.")
        coffea_hists = coffea_data

    hist_list = args.histos if args.histos else list(coffea_hists.keys())
    logging.info(f"Converting: {hist_list}")

    json_dict = {}
    for hist_name in hist_list:
        if hist_name not in coffea_hists:
            logging.warning(f"'{hist_name}' not found, skipping")
            continue
        hist_obj = coffea_hists[hist_name]
        axes_names = [a.name for a in hist_obj.axes]
        logging.info(f"  {hist_name}: axes = {axes_names}")
        try:
            json_dict[hist_name] = convert_histogram(
                hist_obj, fixed_axes,
                sum_axes=args.sum_axes,
                category_maps=category_maps,
            )
            logging.info(f"  ✓ {hist_name}")
        except Exception as e:
            logging.error(f"  ✗ {hist_name}: {e}")
            if args.verbose:
                import traceback; traceback.print_exc()

    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with open(args.output, 'w') as f:
        json.dump(json_dict, f)

    logging.info(f"Saved {len(json_dict)} histograms to {args.output}")
