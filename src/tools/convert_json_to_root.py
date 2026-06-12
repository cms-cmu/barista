"""
Convert a JSON histogram file (produced by convert_coffea_to_json.py or similar)
into a ROOT file containing TH1F histograms.

The JSON format expected at the leaf level is the dict produced by hist_to_json():
    {
        "edges": [...],
        "centers": [...],
        "values": [...],
        "variances": [...],
        "underflow_value": float,   # optional
        "underflow_variance": float,
        "overflow_value": float,
        "overflow_variance": float,
    }

Arbitrarily deep nesting above the leaf is supported. Each level of nesting
can optionally be written as a subdirectory in the ROOT file (--dirs).

The importable function json_to_TH1() is the core converter and is used
directly by make_combine_inputs.py in analysis packages.

Usage (standalone):
    python src/tools/convert_json_to_root.py \\
        -f histos/histAll.json \\
        -o output/histAll.root \\
        --rebin 5

    # Variable binning (space-separated bin edges):
    python src/tools/convert_json_to_root.py \\
        -f histos/histAll.json -o output/histAll.root \\
        --rebin-edges 0.0 0.2 0.4 0.6 0.8 1.0

    # Write nesting levels as ROOT directories:
    python src/tools/convert_json_to_root.py \\
        -f histos/histAll.json -o output/histAll.root --dirs

    # Select only specific top-level histogram keys:
    python src/tools/convert_json_to_root.py \\
        -f histos/histAll.json -o output/histAll.root \\
        --histos SvB.phh SvB.ptt
"""

import os
import array
import argparse
import logging
import json

import ROOT
ROOT.gROOT.SetBatch(True)


def json_to_TH1(hist_dict, name, rebin=1):
    """Convert a leaf histogram dict to a ROOT TH1F.

    Parameters
    ----------
    hist_dict : dict
        Leaf dict with keys: edges, centers, values, variances,
        and optionally underflow_value/variance, overflow_value/variance.
    name : str
        Name (and title) for the TH1F.
    rebin : int or list of float
        Uniform rebin factor (int) or list of bin edges for variable rebinning.

    Returns
    -------
    ROOT.TH1F
    """
    edges     = hist_dict['edges']
    centers   = hist_dict['centers']
    values    = hist_dict['values']
    variances = hist_dict['variances']

    rHist = ROOT.TH1F(name, name, len(centers), edges[0], edges[-1])
    rHist.Sumw2()

    # underflow
    uval = hist_dict.get('underflow_value', 0.0)
    uvar = hist_dict.get('underflow_variance', 0.0)
    rHist.SetBinContent(0, uval)
    rHist.SetBinError(0, ROOT.TMath.Sqrt(abs(uvar)))

    for ibin in range(1, len(centers) + 1):
        rHist.SetBinContent(ibin, values[ibin - 1])
        rHist.SetBinError(ibin, ROOT.TMath.Sqrt(abs(variances[ibin - 1])))

    # overflow
    oval = hist_dict.get('overflow_value', 0.0)
    ovar = hist_dict.get('overflow_variance', 0.0)
    rHist.SetBinContent(len(centers) + 1, oval)
    rHist.SetBinError(len(centers) + 1, ROOT.TMath.Sqrt(abs(ovar)))

    if isinstance(rebin, list):
        rHist = rHist.Rebin(len(rebin) - 1, f"{name}_rebinned",
                            array.array('d', rebin))
    elif rebin != 1:
        rHist.Rebin(rebin)

    return rHist


def _is_leaf(node):
    """Return True if node is a histogram leaf dict (has 'values' key)."""
    return isinstance(node, dict) and 'values' in node


def _write_node(node, root_file, name_parts, rebin, use_dirs):
    """Recursively walk the JSON tree and write TH1Fs into root_file.

    Parameters
    ----------
    node : dict or leaf dict
        Current node in the JSON tree.
    root_file : ROOT.TFile
        Open ROOT file to write into.
    name_parts : list of str
        Path components accumulated so far (used to build histogram names
        and, when use_dirs=True, subdirectory paths).
    rebin : int or list
        Passed through to json_to_TH1.
    use_dirs : bool
        If True, each nesting level becomes a ROOT subdirectory.
        If False, the full path is flattened into the histogram name.
    """
    if _is_leaf(node):
        flat_name = '_'.join(name_parts)
        hist = json_to_TH1(node, flat_name, rebin)

        if use_dirs and len(name_parts) > 1:
            dir_path = '/'.join(name_parts[:-1])
            tdir = root_file.Get(dir_path)
            if not tdir:
                # mkdir -p equivalent
                parts = name_parts[:-1]
                current = root_file
                for part in parts:
                    sub = current.Get(part)
                    if not sub:
                        current.mkdir(part)
                        sub = current.Get(part)
                    current = sub
                tdir = root_file.Get(dir_path)
            tdir.cd()
            hist.SetName(name_parts[-1])
            hist.SetTitle(name_parts[-1])
        else:
            root_file.cd()

        hist.Write()
        logging.debug(f"  wrote {'/'.join(name_parts)}")
        return

    if not isinstance(node, dict):
        logging.warning(f"Unexpected non-dict node at {'/'.join(name_parts)}, skipping")
        return

    for key, child in node.items():
        _write_node(child, root_file, name_parts + [str(key)], rebin, use_dirs)


def convert_json_to_root(input_file, output_file, histos=None,
                         rebin=1, use_dirs=False):
    """Convert a JSON histogram file to ROOT.

    Parameters
    ----------
    input_file : str
        Path to input JSON file.
    output_file : str
        Path to output ROOT file (will be overwritten if it exists).
    histos : list of str or None
        Top-level histogram keys to convert. None means convert all.
    rebin : int or list of float
        Uniform rebin factor or list of bin edges.
    use_dirs : bool
        Write nesting levels as ROOT subdirectories.
    """
    logging.info(f"Reading {input_file}")
    with open(input_file, 'r') as f:
        json_hists = json.load(f)

    if histos:
        missing = [h for h in histos if h not in json_hists]
        if missing:
            logging.warning(f"Histograms not found in JSON: {missing}")
        json_hists = {k: v for k, v in json_hists.items() if k in histos}

    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if os.path.exists(output_file):
        os.remove(output_file)
        logging.info(f"Removed existing file: {output_file}")

    root_file = ROOT.TFile(output_file, 'recreate')

    for hist_name, hist_data in json_hists.items():
        logging.info(f"Converting {hist_name}")
        _write_node(hist_data, root_file, [hist_name], rebin, use_dirs)

    root_file.Close()
    logging.info(f"Written: {output_file}")


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='Convert JSON histogram file to ROOT TH1Fs',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-f', '--file', dest='file_to_convert', required=True,
                        help='Input JSON file')
    parser.add_argument('-o', '--output', dest='output_file',
                        default='./output/histAll.root',
                        help='Output ROOT file path')
    parser.add_argument('--histos', dest='histos', nargs='+', default=None,
                        help='Top-level histogram keys to convert (default: all)')
    parser.add_argument('-r', '--rebin', dest='rebin', type=int, default=1,
                        help='Uniform rebin factor')
    parser.add_argument('--rebin-edges', dest='rebin_edges', nargs='+',
                        type=float, default=None,
                        help='Variable bin edges (overrides --rebin)')
    parser.add_argument('--dirs', dest='use_dirs', action='store_true',
                        default=False,
                        help='Write nesting levels as ROOT subdirectories')
    parser.add_argument('-v', '--verbose', action='store_true', default=False,
                        help='Verbose output')
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format='%(levelname)s:%(message)s')
    logging.info(f"Running with parameters: {args}")

    rebin = args.rebin_edges if args.rebin_edges is not None else args.rebin

    convert_json_to_root(
        args.file_to_convert,
        args.output_file,
        histos=args.histos,
        rebin=rebin,
        use_dirs=args.use_dirs,
    )
