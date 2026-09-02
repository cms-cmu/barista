from __future__ import annotations
import os
import sys
import logging
from copy import copy
from rich.pretty import pretty_repr

import uproot
import psutil
import re
import fnmatch
import subprocess
from urllib.parse import urlparse
from coffea.dataset_tools import rucio_utils

def _natural_sort_key(s: str):
    """Sort strings containing numbers in human/natural order."""
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]

def expand_directory_files(directory_path: str, pattern: str = "*.root") -> list[str]:
    """Recursively list all matching files in a local or remote XRootD directory."""
    files = []
    if directory_path.startswith("root://"):
        parsed = urlparse(directory_path)
        host = parsed.netloc
        path = parsed.path
        try:
            cmd = ["xrdfs", host, "ls", "-R", path]
            output = subprocess.check_output(cmd, text=True, stderr=subprocess.DEVNULL)
            for line in output.splitlines():
                line = line.strip()
                if not line:
                    continue
                fname = os.path.basename(line)
                if fname.endswith(".root") and fnmatch.fnmatch(fname, pattern):
                    clean_line = line.lstrip('/')
                    if directory_path.startswith(f"root://{host}//"):
                        files.append(f"root://{host}//{clean_line}")
                    else:
                        files.append(f"root://{host}/{clean_line}")
        except Exception as e:
            logging.warning(f"Failed to list XRootD directory '{directory_path}' with xrdfs: {e}")
    else:
        clean_path = directory_path[7:] if directory_path.startswith("file://") else directory_path
        clean_path = os.path.expanduser(clean_path)
        for root_dir, _, filenames in os.walk(clean_path):
            for fname in filenames:
                if fname.endswith(".root") and fnmatch.fnmatch(fname, pattern):
                    files.append(os.path.join(root_dir, fname))

    files.sort(key=_natural_sort_key)
    return files

def checking_input_files(outfiles):
    '''Check if the input files are corrupted'''
    logging.info(f"Checking {len(outfiles)} input files for corruption...")

    good_files = []
    corrupted_count = 0

    for outfile in outfiles:
        try:
            # Attempt to open the file with uproot to check for corruption
            uproot.open(outfile + ":Events")
            good_files.append(outfile)
        except Exception as e:
            corrupted_count += 1
            logging.error(f"Error opening file {outfile}: {e}")
            logging.error(f"Skipping corrupted file {outfile}")

    logging.info(f"File check complete: {len(good_files)} good files, {corrupted_count} corrupted files")
    return good_files

def list_of_files(
    ifile,
    allowlist_sites: list = ['T3_US_FNALLPC'],
    blocklist_sites: list = [''],
    rucio_regex_sites: str = 'T[23]',
    test: bool = False,
    test_files: int = 5,
    check_input_files: bool = False
    ):
    '''Check if ifile is root file or dataset to check in rucio'''

    if isinstance(ifile, list):
        ifile = checking_input_files(ifile) if check_input_files else ifile
        return ifile[:(test_files if test else None)]
    elif isinstance(ifile, dict):
        dir_path = ifile.get('path', '')
        pattern = ifile.get('pattern', '*.root')
        file_list = expand_directory_files(dir_path, pattern=pattern)
        file_list = checking_input_files(file_list) if check_input_files else file_list
        return file_list[:(test_files if test else None)]
    elif isinstance(ifile, str) and (ifile.endswith('/') or ifile.startswith('root://') or (ifile.startswith(('file://', '/')) and not ifile.endswith(('.root', '.txt')) and os.path.isdir(ifile))):
        file_list = expand_directory_files(ifile)
        file_list = checking_input_files(file_list) if check_input_files else file_list
        return file_list[:(test_files if test else None)]
    elif ifile.endswith('.txt'):
        file_list = [
            jfile.rstrip() if jfile.startswith(('root','file')) else f'root://cmseos.fnal.gov/{jfile.rstrip()}' for jfile in open(ifile).readlines()]
        file_list = checking_input_files(file_list) if check_input_files else file_list
        return file_list[:(test_files if test else None)]
    else:
        rucio_client = rucio_utils.get_rucio_client()
        outfiles, outsite, sites_counts = rucio_utils.get_dataset_files_replicas(
            ifile, client=rucio_client, regex_sites=fr"{rucio_regex_sites}", mode="first", allowlist_sites=allowlist_sites, blocklist_sites=blocklist_sites)
        good_files = checking_input_files(outfiles) if check_input_files else outfiles
        return good_files[:(test_files if test else None)]

def _friend_merge_name(path1: str, path0: str, name: str, **_):
    return f'{path1}/{path0.replace("picoAOD", name)}'

def process_memory():
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    return mem_info.rss

def profile(func):
    def wrapper(*args, **kwargs):
        mem_before = process_memory()
        result = func(*args, **kwargs)
        mem_after = process_memory()
        logging.info("{}:consumed memory (before, after, diff): {:,}".format(
            func.__name__,
            mem_before, mem_after, mem_after - mem_before))
        return result
    return wrapper

def apply_storage_remap(obj, remaps):
    """
    Recursively walk a nested dict/list structure and replace string prefixes
    according to the remaps list, where each entry is {'from': str, 'to': str}.
    Only strings that start with a 'from' prefix are modified.
    """
    if isinstance(obj, str):
        for remap in remaps:
            if obj.startswith(remap['from']):
                return remap['to'] + obj[len(remap['from']):]
        return obj
    elif isinstance(obj, dict):
        return {k: apply_storage_remap(v, remaps) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [apply_storage_remap(item, remaps) for item in obj]
    return obj

def get_dataset_type(dataset_name):
    """Determine the type of dataset based on its name."""
    if dataset_name == 'mixeddata':
        return 'mixed_data'
    if dataset_name == 'mixeddata_4b':
        return 'mixeddata_4b'
    elif dataset_name in ['mixeddata_4b_noTT']:
        return 'mixeddata_4b_noTT'
    elif dataset_name.startswith('mixeddata_all') or dataset_name.startswith('mixeddata_Run2') or dataset_name.startswith('mixeddata_Run3') or dataset_name.startswith('mixeddata_4b_v'):
        return 'mixeddata_all'
    elif dataset_name in ['mixeddata_4b_pz']:
        return 'mixeddata_4b_pz'
    elif dataset_name == 'datamixed':
        return 'data_mixed'
    elif dataset_name.startswith('synthetic_data_noTT'):
        return 'synthetic_data_noTT'
    elif dataset_name.startswith('synthetic_data'):
        return 'synthetic_data'
    elif dataset_name == 'data_3b_for_mixed':
        return 'data_for_mix'
    elif dataset_name in ['TTToHadronic_for_mixed', 'TTToSemiLeptonic_for_mixed', 'TTTo2L2Nu_for_mixed']:
        return 'tt_for_mixed'
    elif dataset_name == 'data' or dataset_name.startswith('data__'):
        return 'data'
    else:
        return 'mc'

def create_fileset_entry(dataset_key, files, metadata_entry, args, config_runner):
    """Create a standardized fileset entry."""
    return {
        'files': list_of_files(
            files,
            test=args.test,
            test_files=config_runner['test_files'],
            allowlist_sites=config_runner['allowlist_sites'],
            blocklist_sites=config_runner['blocklist_sites'],
            rucio_regex_sites=config_runner['rucio_regex_sites']
        ),
        'metadata': metadata_entry
    }

def process_mc_dataset(dataset, year, metadata, metadata_dataset, fileset, args, config_runner):
    """Process MC dataset configuration."""
    logging.info("Config MC")
    if config_runner['data_tier'].startswith('pico'):
        if 'data' not in dataset:
            metadata_dataset[dataset]['genEventSumw'] = metadata['datasets'][dataset][year][config_runner['data_tier']]['sumw']
        meta_files = metadata['datasets'][dataset][year][config_runner['data_tier']]['files']
    else:
        metadata_dataset[dataset]['genEventSumw'] = 1
        meta_files = metadata['datasets'][dataset][year][config_runner['data_tier']]

    dataset_key = f"{dataset}_{year}"
    fileset[dataset_key] = create_fileset_entry(dataset_key, meta_files, metadata_dataset[dataset], args, config_runner)
    logging.debug(f'Dataset {dataset_key} with {len(fileset[dataset_key]["files"])} files')

def process_sample_based_dataset(dataset_type, name_prefix, dataset, year, metadata, metadata_dataset, fileset, args, config_runner, extra_metadata_fn=None):
    """Process datasets that create multiple samples (mixed, synthetic, etc.)."""
    type_names = {
        'mixed_data': 'Mixed Data',
        'mixeddata_4b': 'New Mixed Data',
        'data_mixed': 'Data Mixed',
        'synthetic_data': 'Synthetic Data'
    }
    logging.info(f"Config {type_names.get(dataset_type, dataset_type.title())}")

    nSamples = metadata['datasets'][dataset].get("nSamples", 15)
    sample_config = metadata['datasets'][dataset][year][config_runner['data_tier']]
    logging.info(f"Number of samples is {nSamples}")

    if getattr(args, 'samples', None):
        sample_indices = []
        for s in args.samples:
            s_str = str(s).lstrip("v")
            sample_indices.append(int(s_str))
    else:
        sample_indices = list(range(nSamples))

    for v in sample_indices:
        sample_name = f"{name_prefix}_v{v}"
        idataset = f'{sample_name}_{year}'

        metadata_dataset[idataset] = copy(metadata_dataset[dataset])
        metadata_dataset[idataset]['processName'] = sample_name

        if extra_metadata_fn:
            extra_metadata_fn(metadata_dataset[idataset], sample_config, v)

        if 'files_template' in sample_config:
            sample_files = [f.replace("XXX", str(v)) for f in sample_config['files_template']]
        elif f"{dataset}_v{v}" in metadata['datasets'] and year in metadata['datasets'][f"{dataset}_v{v}"]:
            sample_files = metadata['datasets'][f"{dataset}_v{v}"][year][config_runner['data_tier']]['files']
        else:
            sample_files = [f for f in sample_config.get('files', []) if f"_v{v}/" in f or f"_v{v}." in f or f"_v{v}_" in f]
        logging.debug(f"samples_files is {sample_files}")
        logging.debug(f"files_template is {sample_config['files_template']}")
        fileset[idataset] = create_fileset_entry(idataset, sample_files, metadata_dataset[idataset], args, config_runner)
        logging.debug(f'Dataset {idataset} with {len(fileset[idataset]["files"])} files')
        logging.debug(f'metadata_dataset is')
        logging.debug(f'idataset is {idataset}')
        logging.debug(pretty_repr(metadata_dataset))

def process_data_for_mix(dataset, year, metadata, metadata_dataset, fileset, args, config_runner):
    """Process data for mixed dataset configuration."""
    logging.info("Config Data for Mixed")

    nMixedSamples = metadata['datasets'][dataset]["nSamples"]
    use_kfold = config_runner.get("use_kfold", False)
    use_ZZinSB = config_runner.get("use_ZZinSB", False)
    use_ZZandZHinSB = config_runner.get("use_ZZandZHinSB", False)
    data_3b_mix_config = metadata['datasets'][dataset][year][config_runner['data_tier']]

    logging.info(f"Number of mixed samples is {nMixedSamples}")
    logging.info(f"Using kfolding? {use_kfold}")
    logging.info(f"Using ZZinSB? {use_ZZinSB}")
    logging.info(f"Using ZZandZHinSB? {use_ZZandZHinSB}")

    idataset = f'{dataset}_{year}'
    metadata_dataset[idataset] = copy(metadata_dataset[dataset])
    metadata_dataset[idataset]['JCM_loads'] = [
        data_3b_mix_config['JCM_load_template'].replace("XXX", str(v))
        for v in range(nMixedSamples)
    ]

    template_mapping = {
        'use_kfold': ('FvT_file_kfold_template', 'FvT_name_kfold_template'),
        'use_ZZinSB': ('FvT_file_ZZinSB_template', 'FvT_name_ZZinSB_template'),
        'use_ZZandZHinSB': ('FvT_file_ZZandZHinSB_template', 'FvT_name_ZZandZHinSB_template')
    }

    file_template, name_template = None, None
    for flag, (f_tmpl, n_tmpl) in template_mapping.items():
        if config_runner.get(flag, False):
            file_template, name_template = f_tmpl, n_tmpl
            break
    else:
        file_template, name_template = 'FvT_file_template', 'FvT_name_template'

    metadata_dataset[idataset]['FvT_files'] = [
        data_3b_mix_config[file_template].replace("XXX", str(v))
        for v in range(nMixedSamples)
    ]
    metadata_dataset[idataset]['FvT_names'] = [
        data_3b_mix_config[name_template].replace("XXX", str(v))
        for v in range(nMixedSamples)
    ]

    fileset[idataset] = create_fileset_entry(idataset, data_3b_mix_config['files'], metadata_dataset[idataset], args, config_runner)
    logging.debug(f'Dataset {idataset} with {len(fileset[idataset]["files"])} files')

def process_tt_for_mixed(dataset, year, metadata, metadata_dataset, fileset, args, config_runner):
    """Process TT for mixed dataset configuration."""
    logging.info("Config TT for Mixed")

    nMixedSamples = metadata['datasets'][dataset]["nSamples"]
    TT_3b_mix_config = metadata['datasets'][dataset][year][config_runner['data_tier']]
    logging.info(f"Number of mixed samples is {nMixedSamples}")

    idataset = f'{dataset}_{year}'
    metadata_dataset[idataset] = copy(metadata_dataset[dataset])
    metadata_dataset[idataset]['FvT_files'] = [
        TT_3b_mix_config['FvT_file_template'].replace("XXX", str(v))
        for v in range(nMixedSamples)
    ]
    metadata_dataset[idataset]['FvT_names'] = [
        TT_3b_mix_config['FvT_name_template'].replace("XXX", str(v))
        for v in range(nMixedSamples)
    ]
    metadata_dataset[idataset]['genEventSumw'] = TT_3b_mix_config['sumw']

    fileset[idataset] = create_fileset_entry(idataset, TT_3b_mix_config['files'], metadata_dataset[idataset], args, config_runner)
    logging.debug(f'Dataset {idataset} with {len(fileset[idataset]["files"])} files')

def process_data_dataset(dataset, year, metadata, metadata_dataset, fileset, args, config_runner):
    """Process regular data dataset configuration."""
    for iera, ifile in metadata['datasets'][dataset][year][config_runner['data_tier']].items():
        if args.era and iera not in args.era:
            continue
        idataset = f'{dataset}_{year}{iera}'
        meta = copy(metadata_dataset[dataset])
        meta['era'] = iera
        files = ifile['files'] if config_runner['data_tier'].startswith('pico') else ifile
        fileset[idataset] = create_fileset_entry(idataset, files, meta, args, config_runner)
        metadata_dataset[idataset] = meta
        logging.debug(f'Dataset {idataset} with {len(fileset[idataset]["files"])} files')

def add_fvt_metadata(meta, config, v):
    """Helper function to add FvT metadata for mixed data."""
    meta['FvT_name'] = config['FvT_name_template'].replace("XXX", str(v))
    meta['FvT_file'] = config['FvT_file_template'].replace("XXX", str(v))

def find_matching_dataset(dataset, metadata):
    """Find matching dataset in metadata, supporting substring matching."""
    if dataset in metadata['datasets']:
        return dataset

    matching_keys = [key for key in metadata['datasets'].keys() if dataset in key]
    if len(matching_keys) == 1:
        matched_dataset = matching_keys[0]
        logging.info(f"Found matching dataset: '{matched_dataset}' for input '{dataset}'")
        return matched_dataset
    elif len(matching_keys) > 1:
        logging.error(f"Multiple matches found for '{dataset}': {matching_keys}. Please be more specific.")
        return None
    else:
        logging.error(f"{dataset} name not found in metadatafile")
        return None

def get_run_from_year(year):
    if year in ['2016', 'UL16_postVFP', 'UL16_preVFP', '2017', 'UL17', '2018', 'UL18']:
        return 'Run2'
    return 'Run3'

def calculate_cross_section(matched_dataset, dataset_type, metadata, year=None):
    """Calculate cross-section for a given dataset."""
    if (dataset_type == 'data' or
        matched_dataset in ['mixeddata', 'datamixed', 'data_3b_for_mixed', 'synthetic_data', 'synthetic_data_noTT'] or
        'xs' not in metadata['datasets'][matched_dataset]):
        return 1.0

    xs = metadata['datasets'][matched_dataset]['xs']
    if hasattr(xs, 'keys') or isinstance(xs, dict):
        if year in xs:
            xs = xs[year]
        else:
            run = get_run_from_year(year)
            if run in xs:
                xs = xs[run]
            else:
                raise KeyError(f"Cross-section for dataset {matched_dataset} not found for year {year} or run {run} in {xs}")

    if isinstance(xs, str):
        return eval(xs)
    return float(xs)

def apply_datasets_filter(datasets, filter_cfg):
    """Exclude specific dataset paths from the datasets metadata dictionary."""
    if not filter_cfg or 'exclude' not in filter_cfg:
        return datasets

    exclude_paths = filter_cfg['exclude']
    for path in exclude_paths:
        parts = path.split('.')
        # Navigate and delete
        d = datasets.get('datasets', datasets)
        parent = None
        last_key = None
        found = True
        for part in parts:
            if isinstance(d, dict) and part in d:
                parent = d
                last_key = part
                d = d[part]
            else:
                found = False
                break
        if found and parent is not None and last_key is not None:
            logging.info(f"  Filtering datasets: excluded path '{path}'")
            del parent[last_key]
    return datasets
