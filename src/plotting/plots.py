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


def load_config_4b(metadata):
    """  Load meta data
    """
    plotConfig = yaml.safe_load(open(metadata, 'r'))

    # for backwards compatibility
    if "codes" not in plotConfig:
        plotConfig['codes'] = {
            'region' : {
                'SR': 2,
                'SB': 1,
                'other': 0,
                2: 'SR',
                1: 'SB',
                0: 'other',
            },
            'tag' : {
                'threeTag': 3,
                'fourTag': 4,
                'other': 0,
                3: 'threeTag',
                4: 'fourTag',
                0: 'other',
            },
        }


    #
    # Expand
    #
    proc_templates = []
    for _hist_proc, _hist_proc_config in plotConfig["hists"].items():
        if not _hist_proc.find("XXX") == -1 and "nSamples" in _hist_proc_config:
            proc_templates.append(_hist_proc)

    for template in proc_templates:
        _hist_proc_config = plotConfig["hists"][template]

        for nS in range(_hist_proc_config["nSamples"]):
            proc_name = template.replace("XXX",str(nS))
            plotConfig["hists"][proc_name] = copy.deepcopy(_hist_proc_config)
            plotConfig["hists"][proc_name]["process"]  = proc_name
            plotConfig["hists"][proc_name]["label"]  = plotConfig["hists"][proc_name]["label"].replace("XXX", str(nS))
            plotConfig["hists"][proc_name]["fillcolor"]  = plot_helpers.COLORS[nS]
            plotConfig["hists"][proc_name]["edgecolor"]  = plot_helpers.COLORS[nS]

        plotConfig["hists"].pop(template)


    return plotConfig


def load_config_bbWW(metadata):
    """  Load meta data
    """
    plotConfig = yaml.safe_load(open(metadata, 'r'))

    return plotConfig



def init_config(args):
    cfg.plotConfig = load_config_bbWW(args.metadata)
    cfg.outputFolder = args.outputFolder
    cfg.combine_input_files = args.combine_input_files
    cfg.plotModifiers = yaml.safe_load(open(args.modifiers, 'r'))

    if cfg.outputFolder:
        if not os.path.exists(cfg.outputFolder):
            os.makedirs(cfg.outputFolder)

    cfg.hists = load_hists(args.inputFile)
    cfg.fileLabels = args.fileLabels
    cfg.axisLabels, cfg.cutList = read_axes_and_cuts(cfg.hists, cfg.plotConfig)

    return cfg


def makePlot(cfg, var='selJets.pt',
             cut="passPreSel", axis_opts={"region":"SR"}, **kwargs):
    r"""
    Takes Options:

       debug    : False,
       var      : 'selJets.pt',
       cut      : "passPreSel",
       axis_opts : dict ({"region":"SR"})

       plotting opts
        'doRatio'  : bool (False)
        'rebin'    : int (1),
    """

    process = kwargs.get("process", None)
    year    = kwargs.get("year", "RunII")
    debug   = kwargs.get("debug", False)
    if debug: print(f"In makePlot kwargs={kwargs}")

    axis_opts_list = False
    for _, v in axis_opts.items():
        if type(v) is list:
            axis_opts_list = True
            break

    if (type(cut) is list) or axis_opts_list or (len(cfg.hists) > 1 and not cfg.combine_input_files) or (type(var) is list) or (type(process) is list) or (type(year) is list):
        try:
            plot_data =  plot_helpers_make_plot_dict.get_plot_dict_from_list(cfg=cfg, var=var, cut=cut, axis_opts=axis_opts, **kwargs)
            return plot_helpers_make_plot.make_plot_from_dict(plot_data)
        except ValueError as e:
            raise ValueError(e)

    elif not cut:
        plot_data = plot_helpers_make_plot_dict.get_plot_dict_from_config(cfg=cfg, var=var, cut=None, axis_opts=axis_opts, **kwargs)
        return plot_helpers_make_plot.make_plot_from_dict(plot_data)

    plot_data = plot_helpers_make_plot_dict.get_plot_dict_from_config(cfg=cfg, var=var, cut=cut, axis_opts=axis_opts,  **kwargs)
    return plot_helpers_make_plot.make_plot_from_dict(plot_data)



def make2DPlot(cfg, process, var='selJets.pt',
               cut="passPreSel", axis_opts={"region":"SR"}, **kwargs):
    r"""
    Takes Options:

       process  : str
       debug    : False,
       var      : 'selJets.pt',
       year     : "2017",
       cut      : "passPreSel",
       axis_opts : dict ({"region":"SR"})

       plotting opts
        'rebin'    : int (1),
    """

    year    = kwargs.get("year", "RunII")
    debug   = kwargs.get("debug", False)
    if debug: print(f"In make2DPlot kwargs={kwargs}")

    axis_opts_list = False
    for _, v in axis_opts.items():
        if type(v) is list:
            axis_opts_list = True
            break


    if (type(cut) is list) or axis_opts_list or (len(cfg.hists) > 1 and not cfg.combine_input_files) or (type(var) is list) or (type(process) is list) or (type(year) is list):
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


def read_axes_and_cuts(hists, plotConfig):

    axisLabels = {}
    cutList = []

    axisLabels["var"] = hists[0]['hists'].keys()
    var1 = list(hists[0]['hists'].keys())[0]

    for a in hists[0]['hists'][var1].axes:
        axisName = a.name

        if axisName == var1:
            continue

        if isinstance(a, hist.axis.Boolean):
            cutList.append(axisName)
            continue

        if a.extent > 20:
            continue   # HACK to skip the variable bins FIX

        axisLabels[axisName] = []

        for iBin in range(a.extent):

            value = a.value(iBin)

            axisLabels[axisName].append(value)

    return axisLabels, cutList


def print_cfg(cfg):

    print("Cuts...")
    for c in cfg.cutList:
        print(f"\t{c}")


    for axis, values in cfg.axisLabels.items():
        if axis in ["var","process"]:
            continue
        print(f"{axis}:")
        for v in values:
            print(f"\t{v}")

    print("Processes...")
    for key, values in cfg.plotConfig.items():
        if key in ["hists", "stack"]:
            for _key, _ in values.items():
                print(f"\t{_key}")
