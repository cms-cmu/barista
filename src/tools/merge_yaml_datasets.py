import yaml
import argparse


if __name__ == '__main__':

    #
    # input parameters
    #
    parser = argparse.ArgumentParser(description='Run coffea processor', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-m', '--main_file', dest="main_file", default="datasets_HH4b.yml", help='Main datasets file.')
    parser.add_argument('-o', '--output_file', dest="output_file", default="datasets_HH4b_merged.yml", help='Output file.')
    parser.add_argument('-f', '--files_to_add', nargs='+', dest="files_to_add", default=["datasets_HH4b.yml"], help='Files to add.')
    parser.add_argument('-a', '--add_to_dataset', dest="add_to_dataset", default="", help='String to add to dataset.')
    args = parser.parse_args()

    main_file = yaml.safe_load(open(args.main_file, 'r'))

    for ifile in args.files_to_add:

        tmp_file = yaml.full_load(open(ifile, 'r'))

        for ikey in tmp_file.keys():
            tmp_split = ('_UL' if 'UL' in ikey else '_20')
            dataset = ikey.split( tmp_split )[0]
            year = tmp_split.split('_')[1] + '_'.join(ikey.split(tmp_split)[1:])
            if dataset in main_file['datasets']:
                for iname in ['source', 'kFactor',
                              'lumi', 'xs',
                              'cutFlowFourTag', 'cutFlowFourTagUnitWeight',
                              'cutFlowThreeTag', 'cutFlowThreeTagUnitWeight',
                              'reproducible', 'lumis_processed',
                              ]:
                    if iname in tmp_file[ikey].keys(): tmp_file[ikey].pop(iname)

                if args.add_to_dataset:
                    dataset = f"{dataset}_{args.add_to_dataset}"
                    if dataset not in main_file['datasets']:
                        main_file['datasets'][dataset] = { year: {} }
                    else:
                        main_file['datasets'][dataset][year] = {}

                if 'data' in dataset:
                    era = year[-1]
                    year = ''.join(year[:-1])

                    if 'picoAOD' not in main_file['datasets'][dataset][year]:
                        main_file['datasets'][dataset][year]['picoAOD'] = {}

                    main_file['datasets'][dataset][year]['picoAOD'][era] = tmp_file[ikey]
                else:
                    main_file['datasets'][dataset][year]['picoAOD'] = tmp_file[ikey]

    yaml.dump(main_file, open(args.output_file, 'w'), default_flow_style=False)
