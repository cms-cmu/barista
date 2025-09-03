from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml
import importlib
import json
import logging
import os
import time
import warnings
from memory_profiler import profile
from copy import copy
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING

import dask
import uproot
import fsspec
import psutil
import yaml
from omegaconf import OmegaConf
from src.addhash import get_git_diff, get_git_revision_hash
from coffea import processor
from coffea.dataset_tools import rucio_utils
from coffea.nanoevents import NanoAODSchema, PFNanoAODSchema
from coffea.util import save
from dask.distributed import performance_report
from distributed.diagnostics.plugin import WorkerPlugin
from rich.logging import RichHandler
from rich.pretty import pretty_repr
from src.skimmer.picoaod import fetch_metadata, integrity_check, resize

if TYPE_CHECKING:
    from src.data_formats.root import Friend

dask.config.set({'logging.distributed': 'error'})

NanoAODSchema.warn_missing_crossrefs = False
warnings.filterwarnings("ignore")

@dataclass
class WorkerInitializer(WorkerPlugin):
    uproot_xrootd_retry_delays: list[float] = None

    def setup(self, worker=None):
        if delays := self.uproot_xrootd_retry_delays:
            from src.data_formats.root.patch import uproot_XRootD_retry
            uproot_XRootD_retry(len(delays) + 1, delays)

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

# inner psutil function
def process_memory():
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    return mem_info.rss

# decorator function
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


# Dataset processing helper functions
def get_dataset_type(dataset_name):
    """Determine the type of dataset based on its name."""
    
    if dataset_name == 'mixeddata':
        return 'mixed_data'
    elif dataset_name == 'datamixed':
        return 'data_mixed'
    elif dataset_name == 'synthetic_data':
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
    logging.info(f'Dataset {dataset_key} with {len(fileset[dataset_key]["files"])} files')


def process_sample_based_dataset(dataset_type, name_prefix, dataset, year, metadata, metadata_dataset, fileset, args, config_runner, extra_metadata_fn=None):
    """Process datasets that create multiple samples (mixed, synthetic, etc.)."""
    type_names = {
        'mixed_data': 'Mixed Data',
        'data_mixed': 'Data Mixed', 
        'synthetic_data': 'Synthetic Data'
    }
    logging.info(f"Config {type_names.get(dataset_type, dataset_type.title())}")
    
    nSamples = metadata['datasets'][dataset]["nSamples"]
    sample_config = metadata['datasets'][dataset][year][config_runner['data_tier']]
    logging.info(f"Number of samples is {nSamples}")
    
    for v in range(nSamples):
        sample_name = f"{name_prefix}_v{v}"
        idataset = f'{sample_name}_{year}'
        
        metadata_dataset[idataset] = copy(metadata_dataset[dataset])
        metadata_dataset[idataset]['processName'] = sample_name
        
        # Apply extra metadata if provided
        if extra_metadata_fn:
            extra_metadata_fn(metadata_dataset[idataset], sample_config, v)
        
        sample_files = [f.replace("XXX", str(v)) for f in sample_config['files_template']]
        fileset[idataset] = create_fileset_entry(idataset, sample_files, metadata_dataset[idataset], args, config_runner)
        logging.info(f'Dataset {idataset} with {len(fileset[idataset]["files"])} files')


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
    
    # Select appropriate template based on configuration
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
    logging.info(f'Dataset {idataset} with {len(fileset[idataset]["files"])} files')


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
    logging.info(f'Dataset {idataset} with {len(fileset[idataset]["files"])} files')


def process_data_dataset(dataset, year, metadata, metadata_dataset, fileset, args, config_runner):
    """Process regular data dataset configuration.
    
    Supports datasets named like 'data', 'data_Muon', 'data_Electron', etc.
    Structure: datasets[dataset][year][data_tier] -> eras
    """
    for iera, ifile in metadata['datasets'][dataset][year][config_runner['data_tier']].items():
        if iera not in args.era:
            continue
        idataset = f'{dataset}_{year}{iera}'
        meta = copy(metadata_dataset[dataset])
        meta['era'] = iera
        files = ifile['files'] if config_runner['data_tier'].startswith('pico') else ifile
        fileset[idataset] = create_fileset_entry(idataset, files, meta, args, config_runner)
        metadata_dataset[idataset] = meta
        logging.info(f'Dataset {idataset} with {len(fileset[idataset]["files"])} files')


def add_fvt_metadata(meta, config, v):
    """Helper function to add FvT metadata for mixed data."""
    meta['FvT_name'] = config['FvT_name_template'].replace("XXX", str(v))
    meta['FvT_file'] = config['FvT_file_template'].replace("XXX", str(v))


def setup_condor_cluster(config_runner):
    """Setup Condor cluster configuration."""
    from distributed import Client
    from lpcjobqueue import LPCCondorCluster
    
    logging.info("Initializing HTCondor cluster configuration...")
    
    cluster_args = {
        'transfer_input_files': config_runner['condor_transfer_input_files'],
        'shared_temp_directory': '/tmp',
        'cores': config_runner['condor_cores'],
        'memory': config_runner['condor_memory'],
        'ship_env': False,
        'scheduler_options': {'dashboard_address': config_runner['dashboard_address']},
        'worker_extra_args': [
            f"--worker-port 10000:10100",
            f"--nanny-port 10100:10200",
        ]
    }
    logging.info("Cluster arguments: ")
    logging.info(pretty_repr(cluster_args))

    logging.info("Creating HTCondor cluster...")
    cluster = LPCCondorCluster(**cluster_args)
    
    logging.info(f"Setting up adaptive scaling (min: {config_runner['min_workers']}, max: {config_runner['max_workers']})")
    cluster.adapt(minimum=config_runner['min_workers'], maximum=config_runner['max_workers'])
    
    logging.info("Creating Dask client...")
    client = Client(cluster)

    logging.info('Waiting for at least one worker...')
    client.wait_for_workers(1)
    logging.info('HTCondor cluster setup complete!')
    return client


def setup_local_cluster(config_runner, args):
    """Setup local Dask cluster configuration."""
    from dask.distributed import Client, LocalCluster
    
    n_workers = 4 if args.skimming else 6
    cluster_args = {
        'n_workers': n_workers,
        'memory_limit': config_runner['condor_memory'],
        'threads_per_worker': 1,
        'dashboard_address': config_runner['dashboard_address'],
    }
    cluster = LocalCluster(**cluster_args)
    return Client(cluster)


def setup_pico_base_name(configs):
    """Determine the pico base name based on configuration."""
    config_config = configs.get("config", {})
    config_runner = configs.get("runner", {})
    
    # Check for explicit pico_base_name first
    if (pico_base_name := config_config.get("pico_base_name")) is not None:
        return pico_base_name
    
    # Check for special configurations
    if "declustering_rand_seed" in config_config:
        return f'picoAOD_seed{config_config["declustering_rand_seed"]}'
    
    class_name = config_runner.get("class_name")
    if class_name == "SubSampler":
        return 'picoAOD_PSData'
    elif class_name == "Skimmer" and config_config.get("skim4b", False):
        return 'picoAOD_fourTag'
    
    return None


def create_reproducible_info(args):
    """Create reproducible information dictionary."""
    return {
        'date': datetime.today().strftime('%Y-%m-%d %H:%M:%S'),
        'hash': args.githash if args.githash else get_git_revision_hash(),
        'args': str(args),
        'diff': args.gitdiff if args.gitdiff else str(get_git_diff()),
    }


def compute_with_client(client, func, *args, **kwargs):
    """Helper to compute with or without dask client."""
    if client is not None:
        return client.compute(func(*args, dask=True, **kwargs), sync=True)
    else:
        return func(*args, dask=False, **kwargs)


def find_matching_dataset(dataset, metadata):
    """Find matching dataset in metadata, supporting substring matching."""
    if dataset in metadata['datasets']:
        return dataset
    
    # Look for a key that contains the dataset name
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


def calculate_cross_section(matched_dataset, dataset_type, metadata):
    """Calculate cross-section for a given dataset."""
    # Data datasets should have xs=1
    if (dataset_type == 'data' or 
        matched_dataset in ['mixeddata', 'datamixed', 'data_3b_for_mixed', 'synthetic_data'] or 
        'xs' not in metadata['datasets'][matched_dataset]):
        return 1.0
    
    xs = metadata['datasets'][matched_dataset]['xs']
    return xs if isinstance(xs, float) else eval(xs)


def setup_schema(config_runner):
    """Convert string schema names to actual schema classes."""
    if isinstance(config_runner['schema'], str):
        schema_mapping = {
            "NanoAODSchema": NanoAODSchema,
            "PFNanoAODSchema": PFNanoAODSchema
        }
        if config_runner['schema'] not in schema_mapping:
            raise ValueError(f"Unknown schema: {config_runner['schema']}")
        config_runner['schema'] = schema_mapping[config_runner['schema']]


def setup_config_defaults(config_runner, args):
    """Set up all configuration defaults in one place."""
    defaults = {
        'data_tier': 'picoAOD',
        'chunksize': 1_000 if args.test else 100_000,
        'maxchunks': 1 if args.test else None,
        'schema': NanoAODSchema,
        'test_files': 5,
        'allowlist_sites': ['T3_US_FNALLPC'],
        'blocklist_sites': [''],
        'rucio_regex_sites': "T[23]",
        'class_name': 'analysis',
        'condor_cores': 2,
        'condor_memory': '4GB',
        'condor_transfer_input_files': ['coffea4bees/', 'src/'],
        'min_workers': 1,
        'max_workers': 100,
        'workers': 2,
        'skipbadfiles': False,
        'dashboard_address': 10200,
        'friend_base': None,
        'friend_base_argname': "make_classifier_input",
        'friend_metafile': 'friends',
        'friend_merge_step': 100_000,
        'write_coffea_output': True,
        'uproot_xrootd_retry_delays': [5, 15, 45]
    }
    
    for key, default_value in defaults.items():
        config_runner.setdefault(key, default_value)


def setup_executor(config_runner, args, client, pool):
    """Setup processor executor based on configuration."""
    executor_args = {
        'schema': config_runner['schema'],
        'savemetrics': True,
        'skipbadfiles': config_runner['skipbadfiles'],
        'xrootdtimeout': 600
    }
    
    if args.debug:
        logging.info("Running iterative executor in debug mode")
        return processor.iterative_executor, executor_args
    elif args.condor or args.run_dask:
        executor_args.update({
            "client": client,
            "align_clusters": False,
            "status": False  # disable progressbar for Dask
        })
        return processor.dask_executor, executor_args
    else:
        logging.info("Running futures executor")
        executor_args.update({
            "pool": pool,
            "workers": config_runner['workers']
        })
        return processor.futures_executor, executor_args


def process_skimming_output(output, fileset, configs, config_runner, args, client):
    """Process output for skimming jobs."""
    # Check integrity of the output
    output, complete = integrity_check(fileset, output)
    if not complete and (config_runner["maxchunks"] is None) and not args.test:
        logging.error("The jobs above failed. Merging is skipped.")
        return output
    
    # Prepare resize arguments
    kwargs = {
        'base_path': configs["config"]["base_path"],
        'output': output,
        'step': config_runner.get("basketsize", configs["config"]["step"]),
        'chunk_size': config_runner.get("picosize", config_runner["chunksize"]),
    }
    
    # Add pico_base_name if needed
    if (pico_base_name := setup_pico_base_name(configs)) is not None:
        kwargs["pico_base_name"] = pico_base_name
    
    # Resize output
    output = compute_with_client(client, resize, **kwargs)
    
    # Keep only file names for each chunk
    for dataset, chunks in output.items():
        chunks['files'] = [str(f.path) for f in chunks['files']]
    
    return output


def process_metadata_output(output, fileset, config_runner, args, client):
    """Process and save metadata for skimming jobs."""
    metadata = compute_with_client(client, fetch_metadata, fileset)
    metadata = processor.accumulate(metadata)

    for ikey in metadata:
        if ikey in output:
            metadata[ikey].update(output[ikey])
            metadata[ikey]['reproducible'] = create_reproducible_info(args)

            if (config_runner["data_tier"] in ['picoAOD'] and 
                "genEventSumw" in fileset[ikey]["metadata"]):
                metadata[ikey]["sumw"] = fileset[ikey]["metadata"]["genEventSumw"]

    # Save metadata file
    output_file = ('picoaod_datasets.yml' if args.output_file.endswith('coffea') 
                   else args.output_file)
    dfile = f'{args.output_path}/{output_file}'
    yaml.dump(metadata, open(dfile, 'w'), default_flow_style=False)
    logging.info(f'Saving metadata file {dfile}')


def process_analysis_output(output, args):
    """Process output for analysis jobs."""
    output['reproducible'] = {
        args.output_file: create_reproducible_info(args)
    }
    
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)


def process_friend_trees(output, config_runner, configs, args, client):
    """Process friend tree metadata if it exists."""
    friend_base = (config_runner["friend_base"] or 
                   configs.get("config", {}).get(config_runner["friend_base_argname"], None))
    friends = output.get("friends", None)
    
    if friend_base is not None and friends is not None:
        from src.data_formats.awkward.zip import NanoAOD

        merge_kw = {
            'step': config_runner["friend_merge_step"],
            'base_path': friend_base,
            'naming': _friend_merge_name,
            'transform': NanoAOD(regular=False, jagged=True),
        }
        
        if args.run_dask:
            merged_friends = client.compute(
                {k: friends[k].merge(**merge_kw, clean=False, dask=True) 
                 for k in friends},
                sync=True,
            )
            for v in friends.values():
                v.reset(confirm=False)
            friends = merged_friends
        else:
            for k, v in friends.items():
                friends[k] = v.merge(**merge_kw)

        from src.storage.eos import EOS
        from src.utils.json import DefaultEncoder

        metafile = (EOS(args.output_path) / str(args.output_file)).with_suffix(".json")
        with fsspec.open(metafile, "wt") as f:
            json.dump(friends, f, cls=DefaultEncoder)
        
        logging.info("The following friends trees are created:")
        logging.info(pretty_repr([*friends.keys()]))
        logging.info(f"Saved friend tree metadata to {metafile}")


def save_coffea_output(output, config_runner, args):
    """Save the final coffea output file."""
    if config_runner['write_coffea_output']:
        hfile = f'{args.output_path}/{args.output_file}'
        logging.info(f'Saving file {hfile}')
        save(output, hfile)


@profile
def run_job(fileset, configs, config_runner, executor, executor_args, args, client, tstart):
    """Run the main processing job."""
    # Get the processor instance
    processor_name = args.processor.split('.')[0].replace("/", '.')
    analysis_class = getattr(importlib.import_module(processor_name), config_runner['class_name'])
    
    output, metrics = processor.run_uproot_job(
        fileset,
        treename='Events',
        processor_instance=analysis_class(**configs.get('config', {})),
        executor=executor,
        executor_args=executor_args,
        chunksize=config_runner['chunksize'],
        maxchunks=config_runner['maxchunks'],
    )
    elapsed = time.time() - tstart
    nEvent = metrics['entries']
    logging.info(f'Metrics:')
    logging.info(pretty_repr(metrics))
    logging.info(f'{nEvent/elapsed:,.0f} events/s total ({nEvent}/{elapsed})')

    # Process output based on job type
    if args.skimming:
        output = process_skimming_output(output, fileset, configs, config_runner, args, client)
        
        # Log performance again after processing
        elapsed = time.time() - tstart
        nEvent = metrics['entries']
        logging.info(f'{nEvent/elapsed:,.0f} events/s total ({nEvent}/{elapsed})')
        
        process_metadata_output(output, fileset, config_runner, args, client)
    else:
        process_analysis_output(output, args)
        process_friend_trees(output, config_runner, configs, args, client)
        save_coffea_output(output, config_runner, args)


if __name__ == '__main__':

    # Configure argument parser
    parser = argparse.ArgumentParser(
        description='Run coffea processor for high-energy physics analysis',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Input/Output files and paths
    io_group = parser.add_argument_group('Input/Output Configuration')
    io_group.add_argument(
        '-p', '--processor',
        dest="processor",
        default="coffea4bees/analysis/processors/processor_HH4b.py",
        help='Path to the processor Python file'
    )
    io_group.add_argument(
        '-c', '--configs',
        dest="configs",
        default="coffea4bees/analysis/metadata/HH4b.yml",
        help='Path to the main configuration YAML file'
    )
    io_group.add_argument(
        '-m', '--metadata',
        dest="metadata",
        default="coffea4bees/metadata/datasets_HH4b.yml",
        help='Path to the datasets metadata YAML file'
    )
    io_group.add_argument(
        '--triggers',
        dest="triggers",
        default="coffea4bees/metadata/triggers_HH4b.yml",
        help='Path to the triggers metadata YAML file'
    )
    io_group.add_argument(
        '-l', '--luminosities',
        dest="luminosities",
        default="coffea4bees/metadata/luminosities_HH4b.yml",
        help='Path to the luminosities metadata YAML file'
    )
    io_group.add_argument(
        '-o', '--output',
        dest="output_file",
        default="hists.coffea",
        help='Name of the output file'
    )
    io_group.add_argument(
        '-op', '--output-path',
        dest="output_path",
        default="hists/",
        help='Directory path where output files will be saved'
    )

    # Load corrections metadata and extract year keys
    with open('src/physics/corrections.yml', 'r') as f:
        corrections_metadata = yaml.safe_load(f)
    year_choices = list(corrections_metadata.keys())

    # Data selection options
    data_group = parser.add_argument_group('Data Selection')
    data_group.add_argument(
        '-y', '--years',
        nargs='+',
        dest='years',
        default=['UL18'],
        choices=year_choices,
        help=f"Year(s) of data to process (as in src/physics/corrections.yml). Choices: {', '.join(year_choices)}. Examples: --years UL17 UL18"
    )
    data_group.add_argument(
        '-d', '--datasets',
        nargs='+',
        dest='datasets',
        default=['HH4b', 'ZZ4b', 'ZH4b'],
        help='Dataset name(s) to process. Examples: --datasets HH4b ZZ4b'
    )
    data_group.add_argument(
        '-e', '--eras',
        nargs='+',
        dest='era',
        default=['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H'],
        help='Data era(s) to process (data only). Examples: --eras A B C'
    )

    # Processing mode options
    mode_group = parser.add_argument_group('Processing Mode')
    mode_group.add_argument(
        '-s', '--skimming',
        dest="skimming",
        action="store_true",
        default=False,
        help='Run in skimming mode instead of analysis mode'
    )
    mode_group.add_argument(
        '-t', '--test',
        dest="test",
        action="store_true",
        default=False,
        help='Run in test mode with limited number of files'
    )
    mode_group.add_argument(
        '--systematics',
        nargs='+',
        dest="systematics",
        default=None,
        help='List of systematics to apply (e.g., "others jes all")'
    )

    # Execution environment options
    exec_group = parser.add_argument_group('Execution Environment')
    exec_group.add_argument(
        '--dask',
        dest="run_dask",
        action="store_true",
        default=False,
        help='Use Dask for distributed processing'
    )
    exec_group.add_argument(
        '--condor',
        dest="condor",
        action="store_true",
        default=False,
        help='Submit jobs to HTCondor cluster'
    )

    # Debugging and quality control
    debug_group = parser.add_argument_group('Debugging and Quality Control')
    debug_group.add_argument(
        '--debug',
        dest="debug",
        action="store_true",
        default=False,
        help='Enable debug mode with verbose logging'
    )
    debug_group.add_argument(
        '--check-input-files',
        dest="check_input_files",
        action="store_true",
        default=False,
        help='Check input files for corruption before processing'
    )

    # Reproducibility options
    repro_group = parser.add_argument_group('Reproducibility')
    repro_group.add_argument(
        '--githash',
        dest="githash",
        default="",
        help='Override git hash for reproducibility tracking'
    )
    repro_group.add_argument(
        '--gitdiff',
        dest="gitdiff",
        default="",
        help='Override git diff for reproducibility tracking'
    )

    # Parse command line arguments
    args = parser.parse_args()

    # Configure logging
    logging_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(
        level=logging_level,
        handlers=[RichHandler(level=logging_level, markup=True)],
        format="%(message)s",
    )
    # Disable verbose logging from third-party libraries
    logging.getLogger('numba').setLevel(logging.WARNING)
    logging.getLogger("lpcjobqueue").setLevel(logging.WARNING)

    logging.info(f"Running with these parameters: {args}")

    # Load configuration and metadata files
    logging.info("Loading configuration and metadata files...")
    logging.info(f"Loading configs from: {args.configs}")
    configs = yaml.safe_load(open(args.configs, 'r'))
    
    if not 'config' in configs: configs['config'] = {}
    # Add corrections_metadata to configs
    logging.info("Loading corrections metadata from: src/physics/corrections.yml")
    configs['config']['corrections_metadata'] = corrections_metadata

    if args.systematics:
        logging.info(f"Systematics to run: {args.systematics}")
        configs['config']['run_systematics'] = args.systematics

    logging.info(f"Loading datasets metadata from: {args.metadata}")
    # load all .yml files in given metadata directory
    if os.path.isdir(args.metadata):
        files= [OmegaConf.load(os.path.join(args.metadata, f)) for f in os.listdir(args.metadata) if f.endswith(('.yaml', '.yml'))]
        datasets = OmegaConf.to_container(OmegaConf.create({'datasets': OmegaConf.merge(*files)}), resolve=True)
    else:   
        #backward compatibility if .yml file is directly provided
        datasets = yaml.safe_load(open(args.metadata, 'r'))

    
    logging.info(f"Loading triggers metadata from: {args.triggers}")
    triggers = yaml.safe_load(open(args.triggers, 'r'))
    
    logging.info(f"Loading luminosities metadata from: {args.luminosities}")
    luminosities = yaml.safe_load(open(args.luminosities, 'r'))
    
    metadata = {**datasets, **triggers, **luminosities}
    logging.info("Successfully loaded all metadata files")

    # Setup configuration
    logging.info("Setting up configuration defaults...")
    config_runner = configs['runner'] if 'runner' in configs.keys() else {}
    setup_config_defaults(config_runner, args)
    setup_schema(config_runner)
    logging.info(f"Configuration setup complete. Data tier: {config_runner['data_tier']}, Schema: {config_runner['schema'].__name__}")

    # Process datasets
    logging.info(f"Starting dataset processing for {len(args.years)} year(s) and {len(args.datasets)} dataset(s)")
    metadata_dataset = {}
    fileset = {}
    
    for year in args.years:
        logging.info(f"Processing year: {year}")
        for dataset in args.datasets:
            logging.info(f"Processing dataset: {dataset}")
            
            # Find matching dataset
            matched_dataset = find_matching_dataset(dataset, metadata)
            if matched_dataset is None:
                logging.warning(f"Skipping dataset {dataset} - no match found")
                continue

            if year not in metadata['datasets'][matched_dataset]:
                logging.warning(f"Skipping {dataset} for {year} - year not available in metadata")
                continue

            # Determine dataset type and cross-section
            dataset_type = get_dataset_type(matched_dataset)
            xsec = calculate_cross_section(matched_dataset, dataset_type, metadata)
            logging.info(f"Dataset type: {dataset_type}, Cross-section: {xsec}")


            metadata_dataset[matched_dataset] = {
                'year': year,
                'processName': matched_dataset,
                'xs': xsec,
                'lumi': float(metadata['luminosities'][year]),
                'trigger': metadata['triggers'][year],
            }
            # Main dataset processing logic            
            if dataset_type == 'mc':
                process_mc_dataset(matched_dataset, year, metadata, metadata_dataset, fileset, args, config_runner)
            elif dataset_type == 'mixed_data':
                process_sample_based_dataset('mixed_data', 'mix', matched_dataset, year, metadata, metadata_dataset, fileset, args, config_runner, add_fvt_metadata)
            elif dataset_type == 'data_mixed':
                process_sample_based_dataset('data_mixed', 'mix', matched_dataset, year, metadata, metadata_dataset, fileset, args, config_runner)
            elif dataset_type == 'synthetic_data':
                process_sample_based_dataset('synthetic_data', 'syn', matched_dataset, year, metadata, metadata_dataset, fileset, args, config_runner)
            elif dataset_type == 'data_for_mix':
                process_data_for_mix(matched_dataset, year, metadata, metadata_dataset, fileset, args, config_runner)
            elif dataset_type == 'tt_for_mixed':
                process_tt_for_mixed(matched_dataset, year, metadata, metadata_dataset, fileset, args, config_runner)
            elif dataset_type == 'data':
                process_data_dataset(matched_dataset, year, metadata, metadata_dataset, fileset, args, config_runner)

    # Summary of processed datasets
    logging.info(f"Dataset processing complete. Total datasets in fileset: {len(fileset)}")
    if fileset:
        total_files = sum(len(dataset_info['files']) for dataset_info in fileset.values())
        logging.info(f"Total files across all datasets: {total_files}")

    # Setup compute environment
    logging.info("Setting up compute environment...")
    client = None
    pool = None
    
    if args.condor:
        logging.info("Configuring HTCondor cluster execution...")
        args.run_dask = True
        client = setup_condor_cluster(config_runner)
    elif args.run_dask:
        logging.info("Configuring local Dask cluster execution...")
        client = setup_local_cluster(config_runner, args)
    else:
        logging.info("Configuring local process pool execution...")
        # Setup process pool for futures executor
        worker_initializer = WorkerInitializer(uproot_xrootd_retry_delays=config_runner['uproot_xrootd_retry_delays'])
        pool = ProcessPoolExecutor(max_workers=config_runner['workers'], initializer=worker_initializer.setup)
        logging.info(f"Process pool created with {config_runner['workers']} workers")

    # Register worker plugin if using Dask
    if client is not None:
        logging.info("Registering worker plugin for Dask client...")
        worker_initializer = WorkerInitializer(uproot_xrootd_retry_delays=config_runner['uproot_xrootd_retry_delays'])
        client.register_plugin(worker_initializer)

    # Setup executor
    logging.info("Setting up processor executor...")
    executor, executor_args = setup_executor(config_runner, args, client, pool)

    logging.info(f"Executor arguments:")
    logging.info(pretty_repr(executor_args))
    
    # Setup processor
    logging.info("Loading processor class...")
    processor_name = args.processor.split('.')[0].replace("/", '.')
    analysis_class = getattr(importlib.import_module(processor_name), config_runner['class_name'])
    logging.info(f"Successfully loaded processor: {processor_name}.{config_runner['class_name']}")

    # Log fileset information
    logging.info(f"Final fileset contains {len(fileset)} datasets:")
    for dataset_key in sorted(fileset.keys()):
        logging.info(f"  - {dataset_key}: {len(fileset[dataset_key]['files'])} files")

    logging.debug(f"Detailed fileset:")
    logging.debug(pretty_repr(fileset))

    # Start job execution
    logging.info("=" * 60)
    logging.info("STARTING JOB EXECUTION")
    logging.info("=" * 60)
    tstart = time.time()

    #
    # Run dask performance only in dask jobs
    #
    if args.run_dask:
        dask_report_file = f'/tmp/-dask-report-{datetime.today().strftime("%Y-%m-%d_%H-%M-%S")}.html'
        logging.info(f"Starting Dask job with performance reporting to: {dask_report_file}")
        with performance_report(filename=dask_report_file):
            run_job(fileset, configs, config_runner, executor, executor_args, args, client, tstart)
        
        # Cleanup cluster and client
        logging.info("Cleaning up Dask resources...")
        for obj_name, obj in [("cluster", cluster), ("client", client)]:
            try:
                obj.close()
                logging.info(f"Successfully closed {obj_name}")
            except (RuntimeError, NameError) as e:
                logging.warning(f"Error closing {obj_name}: {e}")
        
        logging.info(f'Dask performance report saved in {dask_report_file}')
    else:
        logging.info("Starting local job execution...")
        run_job(fileset, configs, config_runner, executor, executor_args, args, client, tstart)

    # Final cleanup
    if pool is not None:
        logging.info("Shutting down process pool...")
        pool.shutdown(wait=True)
        logging.info("Process pool shutdown complete")
    
    logging.info("=" * 60)
    logging.info("JOB EXECUTION COMPLETED SUCCESSFULLY")
    logging.info("=" * 60)
