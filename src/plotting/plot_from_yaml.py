import yaml
import sys
import os
sys.path.insert(0, os.getcwd())
from src.plotting.helpers_make_plot import make_plot_from_dict
import argparse

parser = argparse.ArgumentParser(description='plots', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--input_yaml_files', nargs='+', help='Input yaml file')

parser.add_argument('-o', '--outputFolder', default="./",
                    help='Folder for output folder. Default: ./')

args = parser.parse_args()


for input_yaml_file in args.input_yaml_files:
    with open(input_yaml_file, "r") as yfile:
        loaded_data = yaml.safe_load(yfile)

    loaded_data["kwargs"]["outputFolder"] = args.outputFolder

    # No need to rewrite the yaml
    loaded_data["kwargs"]["write_yaml"] = False

    if "is_2d_hist" in loaded_data:
        make_plot_from_dict(loaded_data, do2d=True)
    else:
        make_plot_from_dict(loaded_data)
