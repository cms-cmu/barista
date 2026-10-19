import argparse
import logging
import sys
from pathlib import Path

# Ensure 'src' parent directory is on the path so cloudpickle can resolve
# modules pickled with 'src.*' references
_src_parent = str(Path(__file__).resolve().parent.parent.parent)
if _src_parent not in sys.path:
    sys.path.insert(0, _src_parent)

from coffea.util import load, save
import hist

def merge_coffea_files( files_to_merge, output_file ):
    """docstring for merge_coffea_files"""

    output = {}

    output = load(files_to_merge[0])
    for ifile in files_to_merge[1:]:
        logging.info(f'Merging {ifile}')
        iout = load(ifile)
        for ikey in iout.keys():
            if ikey not in output.keys():
                output[ikey] = iout[ikey]
            elif "hists" in ikey:
                for ihist in iout[ikey].keys():
                    logging.info(f'   Merging histogram {ihist}')
                    if ihist not in output[ikey]:
                        output[ikey][ihist] = iout[ikey][ihist]
                    elif isinstance(output[ikey][ihist], dict) and isinstance(iout[ikey][ihist], dict):
                        output[ikey][ihist] = output[ikey][ihist] | iout[ikey][ihist]
                    else:
                        try:
                            output[ikey][ihist] += iout[ikey][ihist]
                        except Exception as e:
                            try:
                                h1 = output[ikey][ihist]
                                h2 = iout[ikey][ihist]
                                process_axes = [ax.name for ax in h1.axes if ax.name == "process"]
                                if process_axes:
                                    cats1 = list(h1.axes["process"])
                                    cats2 = [c for c in h2.axes["process"] if c not in cats1]
                                    new_cats = cats1 + cats2
                                    new_ax = hist.axis.StrCategory(new_cats, name="process", label=h1.axes["process"].label)
                                    other_axes = [ax for ax in h1.axes if ax.name != "process"]
                                    h_new = hist.Hist(new_ax, *other_axes, storage=h1.storage_type())
                                    for p in h1.axes["process"]:
                                        h_new[{"process": p}] += h1[{"process": p}]
                                    for p in h2.axes["process"]:
                                        h_new[{"process": p}] += h2[{"process": p}]
                                    output[ikey][ihist] = h_new
                                    logging.info(f'   Successfully merged histogram {ihist} with combined process axes.')
                                else:
                                    raise e
                            except Exception as ex:
                                logging.warning(f'   Could not merge histogram {ihist}: {ex}. Skipping.')
            else:
                output[ikey] = output[ikey] | iout[ikey]

    hfile = f'{output_file}'
    logging.info(f'\nSaving file {hfile}')
    save(output, hfile)


if __name__ == '__main__':

    #
    # input parameters
    #
    parser = argparse.ArgumentParser(
        description='Merge several coffea files', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-o', '--output', dest="output_file",
                        default="hists.coffea", help='Output file.')
    parser.add_argument('-f', '--files', nargs='+', dest='files_to_merge', default=[], help="List of files to merge")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    logging.info(f"\nRunning with these parameters: {args}")

    merge_coffea_files( args.files_to_merge, args.output_file )
