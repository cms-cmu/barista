import yaml
import argparse
import os


def _match_key(key, datasets, data_tier='nanoAOD'):
    """
    Find the (dataset, year, era) a skimmer output key belongs to, using the metadata itself.

    MC keys are {dataset}_{year}; data keys are {dataset}_{year}{era} with era a key of the
    year's data_tier block (e.g. data_2025B11, data_2023_BPixD). era is None for MC.
    Returns None when nothing in the metadata matches.
    """
    best = None
    for dataset, years in datasets.items():
        if not isinstance(years, dict) or not key.startswith(f"{dataset}_"):
            continue
        for year, tiers in years.items():
            prefix = f"{dataset}_{year}"
            if not key.startswith(prefix):
                continue
            rest = key[len(prefix):]
            eras = tiers.get(data_tier) if isinstance(tiers, dict) else None
            if rest == '':
                match = (dataset, year, None)
            elif isinstance(eras, dict) and rest in eras:
                match = (dataset, year, rest)
            else:
                continue
            if best is None or len(prefix) > len(f"{best[0]}_{best[1]}"):
                best = match
    return best


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Run coffea processor', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-m', '--main_file', dest="main_file", default="datasets_HH4b.yml", help='Main datasets file.')
    parser.add_argument('-o', '--output_file', dest="output_file", default="datasets_HH4b_merged.yml", help='Output file.')
    parser.add_argument('-f', '--files_to_add', nargs='+', dest="files_to_add", default=["datasets_HH4b.yml"], help='Files to add.')
    parser.add_argument('-a', '--add_to_dataset', dest="add_to_dataset", default="", help='String to add to dataset.')
    parser.add_argument('-n', '--new_output_file', dest="new_output_file", default="",
                        help='Also write only the entries added by this merge to this file.')
    args = parser.parse_args()

    if os.path.isdir(args.main_file):
        main_file = {'datasets': {}}
        for f in sorted(os.listdir(args.main_file)):
            if f.endswith(('.yml', '.yaml')):
                content = yaml.safe_load(open(os.path.join(args.main_file, f), 'r'))
                if content:
                    if 'datasets' in content:
                        main_file['datasets'].update(content['datasets'])
                    else:
                        main_file['datasets'].update(content)
    else:
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
        'cutFlowTwoTag', 'cutFlowTwoTagUnitWeight',
        'reproducible', 'lumis_processed', 'missing', 'cutflow',
        'sumw_raw', 'sumw2_raw',
    }

    # Expand any directories in files_to_add to their contained YAML files
    expanded_files = []
    for entry in args.files_to_add:
        if os.path.isdir(entry):
            expanded_files.extend(
                sorted(
                    os.path.join(entry, f)
                    for f in os.listdir(entry)
                    if f.endswith(('.yml', '.yaml'))
                )
            )
        else:
            expanded_files.append(entry)

    new_entries = {}
    unmatched = []
    for ifile in expanded_files:

        tmp_file = yaml.full_load(open(ifile, 'r')) or {}

        for ikey in tmp_file.keys():
            if not isinstance(tmp_file[ikey], dict):
                continue
            missing = tmp_file[ikey].get('missing', {})
            if missing:
                raise RuntimeError(
                    f"Missing files detected in {ifile} under key '{ikey}':\n"
                    + '\n'.join(f'  {k}: {v}' for k, v in missing.items())
                )

            match = _match_key(ikey, datasets)
            if match is None:
                if 'total_events' in tmp_file[ikey]:
                    unmatched.append(f"{ifile}: {ikey}")
                else:
                    print(f"Skipping '{ikey}' in {ifile}: not a skimmer output")
                continue
            dataset, year, era = match

            for field in _STRIP_FIELDS:
                tmp_file[ikey].pop(field, None)

            if args.add_to_dataset:
                dataset = f"{dataset}_{args.add_to_dataset}"
                if dataset not in datasets:
                    datasets[dataset] = {year: {}}
                else:
                    datasets[dataset][year] = {}

            new_year = new_entries.setdefault(dataset, {}).setdefault(year, {})
            if era is not None:
                datasets[dataset][year].setdefault('picoAOD', {})[era] = tmp_file[ikey]
                new_year.setdefault('picoAOD', {})[era] = tmp_file[ikey]
            else:
                datasets[dataset][year]['picoAOD'] = tmp_file[ikey]
                new_year['picoAOD'] = tmp_file[ikey]

    if unmatched:
        # Skipping skimmer outputs silently once dropped every data skim from the merged file
        raise RuntimeError("No dataset/year (and era) in the metadata matches these skimmer outputs:\n  "
                           + '\n  '.join(unmatched))

    if args.new_output_file:
        with open(args.new_output_file, 'w') as f:
            yaml.dump({'datasets': new_entries}, f, default_flow_style=False)

    # Write back in the same format as input
    if 'datasets' in main_file:
        main_file['datasets'] = datasets
        yaml.dump(main_file, open(args.output_file, 'w'), default_flow_style=False)
    else:
        yaml.dump(datasets, open(args.output_file, 'w'), default_flow_style=False)
