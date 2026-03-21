import os
import sys
import hist
import yaml
import copy
import argparse
from coffea.util import load
import src.plotting.iPlot_config as cfg
import src.plotting.helpers as plot_helpers
import src.plotting.helpers_make_plot_dict as plot_helpers_make_plot_dict
import src.plotting.helpers_make_plot as plot_helpers_make_plot
from src.plotting.plot_types import RenderOptions
from dataclasses import fields as _dataclass_fields

def init_arg_parser():

    parser = argparse.ArgumentParser(description='plots', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument(dest="inputFile",
                        default='hists.pkl', nargs='+',
                        help='Input File. Default: hists.pkl')

    parser.add_argument('-l', '--labelNames', dest="fileLabels",
                        default=["fileA", "fileB"], nargs='+',
                        help='label Names when more than one input file')

    parser.add_argument('-o', '--outputFolder', default=None,
                        help='Folder for output folder. Default: plots/')

    parser.add_argument('-m', '--metadata', dest="metadata",
                        default="coffea4bees/plots/metadata/plotsAll.yml",
                        help='Metadata file.')

    parser.add_argument('--modifiers', dest="modifiers",
                        default="coffea4bees/plots/metadata/plotModifiers.yml",
                        help='Metadata file.')

    parser.add_argument('--only', dest="list_of_hists",
                        default=[], nargs='+',
                        help='If given only plot these hists')

    parser.add_argument('-s', '--skip', dest="skip_hists",
                        default=[], nargs='+',
                        help='Name of hists to skip')


    parser.add_argument('--doTest', action="store_true", help='Metadata file.')
    parser.add_argument('--debug', action="store_true", help='')
    parser.add_argument('--signal', action="store_true", help='')
    parser.add_argument('--year',   help='')
    parser.add_argument('--combine_input_files', action="store_true", help='')

    return parser


def parse_args():

    parser = init_arg_parser()

    args = parser.parse_args()
    return args


#def init_config(args):
#    cfg.plotConfig = load_config_bbWW(args.metadata)
#    cfg.outputFolder = args.outputFolder
#    cfg.combine_input_files = args.combine_input_files
#    cfg.plotModifiers = yaml.safe_load(open(args.modifiers, 'r'))
#
#    if cfg.outputFolder:
#        if not os.path.exists(cfg.outputFolder):
#            os.makedirs(cfg.outputFolder)
#
#    cfg.hists = load_hists(args.inputFile)
#    cfg.fileLabels = args.fileLabels
#    cfg.axisLabels, cfg.cutList = read_axes_and_cuts(cfg.hists, cfg.plotConfig)
#
#    return cfg


# ---------------------------------------------------------------------------
# Auto-normalization map: built once from RenderOptions field names.
# Maps lowercased+underscore-stripped form → canonical field name.
# Handles any camelCase / snake_case / mixed-case variant for free.
# ---------------------------------------------------------------------------
_RENDER_FIELDS: frozenset = frozenset(f.name for f in _dataclass_fields(RenderOptions))
_AUTO_NORMALIZE_MAP: dict = {
    name.lower().replace('_', ''): name
    for name in _RENDER_FIELDS
}

# ---------------------------------------------------------------------------
# Explicit alias table — only for cases auto-normalization cannot handle:
#   • Abbreviations (normalize → norm, flow → add_flow, uniform → uniform_bins)
#   • Genuinely different names (outdir → outputFolder)
#   • Short-prefix fields where the long form doesn't collapse to the short
#     (ratio_lim → rlim, ratio_label → rlabel, do_legend → legend)
# ---------------------------------------------------------------------------
_KWARG_ALIASES: dict = {
    # norm
    'normalize':        'norm',
    'normalise':        'norm',
    'normalized':       'norm',
    'normalised':       'norm',
    # add_flow
    'flow':             'add_flow',
    # uniform_bins
    'uniform':          'uniform_bins',
    # outputFolder
    'outdir':           'outputFolder',
    'output_dir':       'outputFolder',
    # doRatio  ('Ratio' → 'ratio' ≠ 'doratio')
    'Ratio':            'doRatio',
    # rlim  (ratiolim ≠ rlim after stripping)
    'ratio_lim':        'rlim',
    'ratio_limits':     'rlim',
    # rlabel
    'ratio_label':      'rlabel',
    # legend  (dolegend ≠ legend after stripping)
    'doLegend':         'legend',
    'do_legend':        'legend',
    # legend_loc
    'legend_location':  'legend_loc',
    # write_yaml  ('save' ≠ 'write')
    'saveYaml':         'write_yaml',
    'save_yaml':        'write_yaml',
    # fmt
    'format':           'fmt',
    'output_format':    'fmt',
}


def _normalize_kwargs(kwargs: dict) -> dict:
    """Fold kwarg aliases into canonical RenderOptions names (in-place).

    Two passes:
      1. Explicit alias table — abbreviations and genuinely different names.
      2. Auto-normalization — resolves any camelCase / snake_case / mixed-case
         variant by stripping underscores and lowercasing, then matching against
         the known RenderOptions field set.  Covers e.g. ``addFlow``, ``doRatio``,
         ``uniformBins``, ``writeYaml``, ``cmsText``, ``yScale``, etc. without
         needing an explicit entry per variant.
    """
    # Pass 1: explicit aliases
    for alias, canonical in _KWARG_ALIASES.items():
        if alias in kwargs:
            kwargs.setdefault(canonical, kwargs.pop(alias))

    # Pass 2: auto camelCase / snake_case normalization
    for key in list(kwargs):
        if key not in _RENDER_FIELDS:
            canonical = _AUTO_NORMALIZE_MAP.get(key.lower().replace('_', ''))
            if canonical is not None:
                kwargs.setdefault(canonical, kwargs.pop(key))

    return kwargs


def _is_axis_opts_list(axis_opts):
    return any(isinstance(v, list) for v in axis_opts.values())


def makePlot(cfg, var='selJets.pt',
             cut=None, axis_opts={"region":"SR"}, **kwargs):
    r"""
    Takes Options:

       debug    : False,
       var      : 'selJets.pt',
       cut      : None,
       axis_opts : dict ({"region":"SR"})

       plotting opts
        'doRatio'  : bool (True)
        'rebin'    : int (1),
    """

    _normalize_kwargs(kwargs)
    process = kwargs.get("process", None)
    year    = kwargs.get("year", "RunII")
    debug   = kwargs.get("debug", False)
    if debug: print(f"In makePlot kwargs={kwargs}")

    if (isinstance(cut, list)) or _is_axis_opts_list(axis_opts) or (len(cfg.hists) > 1 and not cfg.combine_input_files) or (isinstance(var, list)) or (isinstance(process, list)) or (isinstance(year, list)):
        try:
            if debug: print(f"makePlot: getting plot data from list")
            plot_data =  plot_helpers_make_plot_dict.get_plot_dict_from_list(cfg=cfg, var=var, cut=cut, axis_opts=axis_opts, **kwargs)
            if debug: print(f"makePlot got plot data")
            return plot_helpers_make_plot.make_plot_from_dict(plot_data)
        except ValueError as e:
            raise ValueError(e)

    elif not cut:
        plot_data = plot_helpers_make_plot_dict.get_plot_dict_from_config(cfg=cfg, var=var, cut=None, axis_opts=axis_opts, **kwargs)
        return plot_helpers_make_plot.make_plot_from_dict(plot_data)

    if debug: print(f"makePlot: getting plot data from config")
    plot_data = plot_helpers_make_plot_dict.get_plot_dict_from_config(cfg=cfg, var=var, cut=cut, axis_opts=axis_opts,  **kwargs)
    return plot_helpers_make_plot.make_plot_from_dict(plot_data)



def make2DPlot(cfg, process, var='selJets.pt',
               cut=None, axis_opts={"region":"SR"}, **kwargs):
    r"""
    Takes Options:

       process  : str
       debug    : False,
       var      : 'selJets.pt',
       year     : "2017",
       cut      : None,
       axis_opts : dict ({"region":"SR"})

       plotting opts
        'rebin'    : int (1),
    """

    _normalize_kwargs(kwargs)
    year    = kwargs.get("year", "RunII")
    debug   = kwargs.get("debug", False)
    if debug: print(f"In make2DPlot kwargs={kwargs}")

    if (isinstance(cut, list)) or _is_axis_opts_list(axis_opts) or (len(cfg.hists) > 1 and not cfg.combine_input_files) or (isinstance(var, list)) or (isinstance(process, list)) or (isinstance(year, list)):
        try:
            plot_data =  plot_helpers_make_plot_dict.get_plot_dict_from_list(cfg=cfg, var=var, cut=cut, axis_opts=axis_opts, process=process, do2d=True, **kwargs)
            return plot_helpers_make_plot.make_plot_from_dict(plot_data, do2d=True)
        except ValueError as e:
            raise ValueError(e)

    plot_data = plot_helpers_make_plot_dict.get_plot_dict_from_config(cfg=cfg, var=var, cut=cut, axis_opts=axis_opts, process=process, do2d=True, **kwargs)

    #
    # Make the plot
    #
    return plot_helpers_make_plot.make_plot_from_dict(plot_data, do2d=True)


def load_hists(input_hists):
    hists = []
    for _inFile in input_hists:
        with open(_inFile, 'rb') as hfile:
            hists.append(load(hfile))

    return hists


def read_axes_and_cuts(hists, plotConfig, hist_keys=['hists']):

    axisLabelsDict = {}
    cutListDict = {}

    for hk in hist_keys:

        axisLabelsDict[hk] = {}
        cutListDict[hk] = []

        axisLabelsDict[hk]["var"] = hists[0][hk].keys()
        var1 = list(hists[0][hk].keys())[0]

        for a in hists[0][hk][var1].axes:
            axisName = a.name

            if axisName == var1:
                continue

            if isinstance(a, hist.axis.Boolean):
                cutListDict[hk].append(axisName)
                continue

            if a.extent > 20:
                continue   # HACK to skip the variable bins FIX

            axisLabelsDict[hk][axisName] = []

            for iBin in range(a.extent):

                value = a.value(iBin)

                axisLabelsDict[hk][axisName].append(value)

    return axisLabelsDict, cutListDict


def print_cfg(cfg):


    for hk in cfg.axisLabelsDict.keys():

        print(f"Hist key... {hk}")

        print("\tCuts...")
        for c in cfg.cutListDict[hk]:
            print(f"\t\t{c}")


        for axis, values in cfg.axisLabelsDict[hk].items():
            if axis in ["var","process"]:
                continue
            print(f"\t{axis}:")
            for v in values:
                print(f"\t\t{v}")

        print("Processes...")
        for key, values in cfg.plotConfig.items():
            if key in ["hists", "stack"]:
                for _key, _ in values.items():
                    print(f"\t\t{_key}")
