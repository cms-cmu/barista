import argparse
import logging
from coffea.util import load, save

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
            elif ikey == "hists":
                for ihist in iout[ikey].keys():
                    try:
                        output[ikey][ihist] += iout[ikey][ihist]
                    except KeyError:
                        output[ikey][ihist] = iout[ikey][ihist]
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
