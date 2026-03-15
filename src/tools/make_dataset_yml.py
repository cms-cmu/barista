"""
Convert a picoaod_datasets output file (produced by runner.py -o) into a
datasets YAML suitable for use with -m in runner.py.

The picoaod_datasets format is:
    data_2022_EEE:
      files: [...]
      bad_files: [...]    # ignored

The output format mirrors coffea4bees/metadata/datasets_HH4b_Run3/:
    <dataset_name>:
      2022_EE:
        picoAOD:
          E:
            files: [...]

Works for any picoaod_datasets output: mixeddata_all, JetDeClustered, 4b skims, etc.

Usage:
    python coffea4bees/analysis/tools/make_dataset_yml.py \\
        -i output/mixeddata_make_dataset_Run3_all/picoaod_datasets_mixeddata_Run3_noTT_pz.yml \\
        -o coffea4bees/metadata/datasets_HH4b_Run3/mixeddata_all.yml \\
        -n mixeddata_all_noTT
"""

import argparse
import yaml


# Order matters: longer/more-specific prefixes must come first.
YEAR_PREFIXES = ["2023_preBPix", "2023_BPix", "2022_preEE", "2022_EE",
                 "UL18", "UL17", "UL16_preVFP", "UL16_postVFP"]


def parse_dataset_key(key: str):
    """
    Split a picoaod dataset key like 'data_2022_EEE' into (year, era).
    Strips any leading word and underscore (e.g. 'data_', 'JetDeClustered_')
    before matching year prefixes. Returns (None, None) if no match.
    """
    # Try matching directly first, then after stripping the first '_'-delimited word
    candidates = [key]
    parts = key.split("_", 1)
    if len(parts) == 2:
        candidates.append(parts[1])

    for name in candidates:
        for year in YEAR_PREFIXES:
            if name.startswith(year):
                era = name[len(year):]
                return year, era

    return None, None


def convert(input_file: str, output_file: str, dataset_name: str):
    with open(input_file) as f:
        picoaod = yaml.full_load(f)

    result = {dataset_name: {}}

    skipped = []
    for key, data in picoaod.items():
        year, era = parse_dataset_key(key)
        if year is None:
            skipped.append(key)
            continue

        files = data.get("files")
        if not files:
            print(f"  [warn] {key}: no 'files' field, skipping")
            continue

        result[dataset_name].setdefault(year, {"picoAOD": {}})
        if era:
            result[dataset_name][year]["picoAOD"][era] = {"files": sorted(files)}
        else:
            # No era suffix (e.g. MC datasets): store files directly
            result[dataset_name][year]["picoAOD"] = {"files": sorted(files)}

    if skipped:
        print(f"[warn] Skipped unrecognised dataset keys: {skipped}")

    with open(output_file, "w") as f:
        yaml.dump(result, f, default_flow_style=False)

    print(f"Written: {output_file}")
    for year in sorted(result[dataset_name]):
        picoAOD = result[dataset_name][year]["picoAOD"]
        eras = sorted(picoAOD) if isinstance(picoAOD, dict) and "files" not in picoAOD else ["(no era)"]
        print(f"  {year}: {eras}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert picoaod_datasets yml to analysis datasets yml",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("-i", "--input", required=True, help="Input picoaod_datasets yml")
    parser.add_argument("-o", "--output", required=True, help="Output datasets yml")
    parser.add_argument(
        "-n", "--name", default="mixeddata_all_noTT", help="Top-level dataset name"
    )
    args = parser.parse_args()

    convert(args.input, args.output, args.name)
