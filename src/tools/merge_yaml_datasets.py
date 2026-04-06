import yaml
import argparse
import re


def _parse_dataset_and_year(key):
    """
    Split a skimmer output key into (dataset_name, year).

    Handles conventions:
      DatasetName_UL18           -> (DatasetName, UL18)
      DatasetName_2022_preEE     -> (DatasetName, 2022_preEE)
      DatasetName_2023_BPix      -> (DatasetName, 2023_BPix)
      DatasetName_2024           -> (DatasetName, 2024)
    """
    m = re.search(r'_(UL\d+|20\d{2}(?:_\w+)?)$', key)
    if m:
        return key[:m.start()], m.group(1)
    # Fallback to old split logic
    tmp_split = '_UL' if 'UL' in key else '_20'
    parts = key.split(tmp_split, 1)
    dataset = parts[0]
    year = (tmp_split.lstrip('_') + parts[1]) if len(parts) > 1 else ''
    return dataset, year


def _parse_data_era(key, year):
    """
    For data datasets, extract the era label from the skimmer key.
    e.g. key='data__EGamma0_2022_preEEC', year='2022_preEE' -> era='C'
         key='data__EGamma0_2023_preBPixC01', year='2023_preBPix' -> era='C01'
    """
    return key[key.index(year) + len(year):]


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Run coffea processor', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-m', '--main_file', dest="main_file", default="datasets_HH4b.yml", help='Main datasets file.')
    parser.add_argument('-o', '--output_file', dest="output_file", default="datasets_HH4b_merged.yml", help='Output file.')
    parser.add_argument('-f', '--files_to_add', nargs='+', dest="files_to_add", default=["datasets_HH4b.yml"], help='Files to add.')
    parser.add_argument('-a', '--add_to_dataset', dest="add_to_dataset", default="", help='String to add to dataset.')
    args = parser.parse_args()

    main_file = yaml.safe_load(open(args.main_file, 'r'))

    # Check if main_file has 'datasets' key (old format) or not (new format)
    if 'datasets' in main_file:
        datasets = main_file['datasets']
    else:
        datasets = main_file

    _STRIP_FIELDS = {
        'source', 'kFactor', 'lumi', 'xs',
        'cutFlowFourTag', 'cutFlowFourTagUnitWeight',
        'cutFlowThreeTag', 'cutFlowThreeTagUnitWeight',
        'reproducible', 'lumis_processed', 'missing', 'cutflow',
        'sumw_raw', 'sumw2_raw',
    }

    for ifile in args.files_to_add:

        tmp_file = yaml.full_load(open(ifile, 'r'))

        for ikey in tmp_file.keys():
            missing = tmp_file[ikey].get('missing', {})
            if missing:
                raise RuntimeError(
                    f"Missing files detected in {ifile} under key '{ikey}':\n"
                    + '\n'.join(f'  {k}: {v}' for k, v in missing.items())
                )

            dataset, year = _parse_dataset_and_year(ikey)


            if dataset not in datasets:
                continue

            for field in _STRIP_FIELDS:
                tmp_file[ikey].pop(field, None)

            if args.add_to_dataset:
                dataset = f"{dataset}_{args.add_to_dataset}"
                if dataset not in datasets:
                    datasets[dataset] = {year: {}}
                else:
                    datasets[dataset][year] = {}

            if 'data' in dataset:
                era = _parse_data_era(ikey, year)

                if 'picoAOD' not in datasets[dataset][year]:
                    datasets[dataset][year]['picoAOD'] = {}

                datasets[dataset][year]['picoAOD'][era] = tmp_file[ikey]
            else:
                datasets[dataset][year]['picoAOD'] = tmp_file[ikey]

    # Write back in the same format as input
    if 'datasets' in main_file:
        main_file['datasets'] = datasets
        yaml.dump(main_file, open(args.output_file, 'w'), default_flow_style=False)
    else:
        yaml.dump(datasets, open(args.output_file, 'w'), default_flow_style=False)
