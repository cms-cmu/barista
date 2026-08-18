from __future__ import annotations
import os
import sys
import time
import importlib
import logging
import json
import yaml
from copy import copy
from datetime import datetime
from rich.pretty import pretty_repr
from coffea import processor
from coffea.util import save
import fsspec

from coffea.nanoevents import NanoAODSchema, PFNanoAODSchema
if hasattr(NanoAODSchema, 'error_missing_event_ids'):
    NanoAODSchema.error_missing_event_ids = False

from src.compat import COFFEA_2025
from src.utils.addhash import get_git_diff, get_git_revision_hash, find_git_root

def profile(func):
    # Re-use profile decorator
    def wrapper(*args, **kwargs):
        import psutil
        process = psutil.Process(os.getpid())
        mem_before = process.memory_info().rss
        result = func(*args, **kwargs)
        mem_after = process.memory_info().rss
        logging.info("{}:consumed memory (before, after, diff): {:,}".format(
            func.__name__,
            mem_before, mem_after, mem_after - mem_before))
        return result
    return wrapper

def create_reproducible_info(args):
    info = {
        'date': datetime.today().strftime('%Y-%m-%d %H:%M:%S'),
        'args': str(args),
    }

    barista_root = os.getcwd()
    if getattr(args, 'githash', None) or getattr(args, 'gitdiff', None):
        info['barista'] = {
            'hash': args.githash if args.githash else get_git_revision_hash(barista_root),
            'diff': args.gitdiff if args.gitdiff else str(get_git_diff(barista_root)),
        }
    else:
        info['barista'] = {
            'hash': get_git_revision_hash(barista_root),
            'diff': str(get_git_diff(barista_root)),
        }

    processor_path = args.processor
    processor_repo_name = processor_path.split('/')[0] if '/' in processor_path else 'processor'
    processor_repo_root = find_git_root(processor_path)

    if processor_repo_root and processor_repo_root != barista_root:
        info[processor_repo_name] = {
            'hash': get_git_revision_hash(processor_repo_root),
            'diff': str(get_git_diff(processor_repo_root)),
        }
    else:
        info[processor_repo_name] = {
            'hash': 'Same repository as barista' if processor_repo_root == barista_root else 'Not a git repository',
            'diff': '',
        }
    return info

def compute_with_client(client, func, *args, **kwargs):
    """Helper to compute with or without dask client."""
    if client is not None:
        return client.compute(func(*args, dask=True, **kwargs), sync=True, retries=3)
    else:
        return func(*args, dask=False, **kwargs)

def find_free_port(preferred: int) -> int:
    """Return preferred port if free, otherwise let the OS pick one."""
    import socket
    with socket.socket() as s:
        try:
            s.bind(('', preferred))
        except OSError:
            s.bind(('', 0))
        return s.getsockname()[1]

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
        'worker_memory': '4GB',
        'condor_transfer_input_files': ['coffea4bees/', 'src/'],
        'min_workers': 1,
        'max_workers': 1000 if getattr(args, 'shared_dask', False) else 400,
        'workers': 2,
        'skipbadfiles': False,
        'dashboard_address': 10200,
        'friend_base': None,
        'friend_base_argname': "make_classifier_input",
        'friend_merge_step': 100_000,
        'write_coffea_output': True,
        'uproot_xrootd_retry_delays': [5, 15, 30, 60, 120],
        'dask_retries': 3,
        'slurm_cores': 4,
        'slurm_partition': 'work',
        'slurm_qos': 'cpu_light',
        'slurm_walltime': '08:00:00',
        'slurm_log_directory': 'slurm_logs',
        'slurm_job_extra': [],
    }

    for key, default_value in defaults.items():
        config_runner.setdefault(key, default_value)

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
    elif class_name == "HemiMixer":
        return 'picoAOD_mixed_all'
    elif class_name == "MixedDataSplitter":
        return f'picoAOD_mixed_v{config_config.get("mixed_subsample")}'
    elif class_name == "Skimmer" and config_config.get("skim4b", False):
        return 'picoAOD_fourTag'

    # Check for classifier based configurations (lowpt-run3)
    if config_runner.get("run_SvB", False) and config_runner.get("run_FeynNet", False):
        return "picoAOD_SvB_FeynNet"
    elif config_runner.get("run_SvB", False):
        return "picoAOD_SvB"
    elif config_runner.get("run_FeynNet", False):
        return "picoAOD_FeynNet"
    elif config_runner.get("run_dilep_ttbar_crosscheck", False):
        return "picoAOD_dilep"
    elif config_runner.get("skimming", False):
        return "picoAOD"

    return None

def setup_executor(config_runner, args, client, pool):
    """Setup processor executor based on configuration."""
    if COFFEA_2025:
        runner_args = {
            'schema': config_runner['schema'],
            'savemetrics': True,
            'skipbadfiles': config_runner['skipbadfiles'],
            'xrootdtimeout': 600,
            'chunksize': config_runner['chunksize'],
            'maxchunks': config_runner['maxchunks'],
        }
        if args.debug:
            logging.info("Running iterative executor in debug mode")
            return processor.IterativeExecutor(), runner_args
        elif args.condor or args.run_dask:
            return processor.DaskExecutor(
                client=client,
                status=args.run_dask and not args.condor,
                retries=config_runner['dask_retries'],
            ), runner_args
        else:
            logging.info("Running futures executor")
            return processor.FuturesExecutor(workers=config_runner['workers']), runner_args
    else:
        executor_args = {
            'schema': config_runner['schema'],
            'savemetrics': True,
            'skipbadfiles': config_runner['skipbadfiles'],
            'xrootdtimeout': 900,
        }
        if args.debug:
            logging.info("Running iterative executor in debug mode")
            return processor.iterative_executor, executor_args
        elif args.condor or args.run_dask:
            executor_args.update({
                "client": client,
                "align_clusters": False,
                "status": args.run_dask and not args.condor,
            })
            return processor.dask_executor, executor_args
        else:
            logging.info("Running futures executor")
            executor_args.update({
                "pool": pool,
                "workers": config_runner['workers'],
            })
            return processor.futures_executor, executor_args

def process_skimming_output(output, fileset, configs, config_runner, args, client):
    """Process output for skimming jobs."""
    from src.skimmer.picoaod import integrity_check, resize
    output, complete = integrity_check(fileset, output)
    if not complete and (config_runner["maxchunks"] is None) and not args.test:
        logging.error("The jobs above failed. Merging is skipped.")
        return output

    kwargs = {
        'base_path': configs["config"]["base_path"],
        'output': output,
        'step': config_runner.get("basketsize", configs["config"]["step"]),
        'chunk_size': config_runner.get("picosize", config_runner["chunksize"]),
    }

    if (pico_base_name := setup_pico_base_name(configs)) is not None:
        kwargs["pico_base_name"] = pico_base_name

    output = compute_with_client(client, resize, **kwargs)

    for dataset, chunks in output.items():
        chunks['files'] = [str(f.path) for f in chunks['files']]
        if output[dataset].get("missing", {}).get("file_missing"):
            logging.info(f'Merging completed successfully for "{dataset}" — ignore the missing file warnings above, some files had zero selected events or failed silently.')

    return output

def process_metadata_output(output, fileset, config_runner, args, client):
    """Process and save metadata for skimming jobs."""
    from src.skimmer.picoaod import fetch_metadata
    metadata = compute_with_client(client, fetch_metadata, fileset)
    metadata = processor.accumulate(metadata)

    for ikey in metadata:
        if ikey in output:
            metadata[ikey].update(output[ikey])
            metadata[ikey]['reproducible'] = create_reproducible_info(args)

            if (config_runner["data_tier"] in ['picoAOD'] and
                "genEventSumw" in fileset[ikey]["metadata"]):
                metadata[ikey]["sumw"] = fileset[ikey]["metadata"]["genEventSumw"]

    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)
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

def process_friend_trees(output, config_runner, configs, args, client, fileset=None):
    """Process friend tree metadata if it exists."""
    friend_base = (config_runner["friend_base"] or
                   configs.get("config", {}).get(config_runner["friend_base_argname"], None))
    friends = output.get("friends", None)

    if friend_base is not None and friends is not None:
        from src.data_formats.awkward.zip import NanoAOD

        path1_to_dataset = {}
        if fileset:
            for dataset_key, dataset_info in fileset.items():
                for f in dataset_info["files"]:
                    parent_dir = f.rstrip('/').split('/')[-2]
                    path1_to_dataset[parent_dir] = dataset_key

        def _merge_naming(path0, path1, name, **_):
            dir_name = path1_to_dataset.get(path1, path1)
            fname = path0.replace("picoAOD", name)
            # When several source era-dirs collapse into a single dataset dir
            # (dir_name != path1 — e.g. a synthetic sample folds
            # data_2022_preEEB/C/D into one syn_noTT_v0_2022_preEE), the
            # source picoAODs share a basename (picoAOD_seed0.root), so the
            # friend filenames (SvB_seed0.root) would collide and silently
            # overwrite each other — only the last chunk survives on disk while
            # the index still records every intended per-chunk UUID. Prefix the
            # source era to keep them unique. When the era dir is preserved
            # (dir_name == path1 — e.g. mixeddata), basenames are already unique
            # per era, so the historical name is kept byte-for-byte.
            if dir_name != path1:
                fname = f'{path1}_{fname}'
            return f'{dir_name}/{fname}'

        merge_kw = {
            'step': config_runner["friend_merge_step"],
            'base_path': friend_base,
            'naming': _merge_naming,
            'transform': NanoAOD(regular=False, jagged=True),
        }

        if args.run_dask:
            merged_friends = client.compute(
                {k: friends[k].merge(**merge_kw, clean=False, dask=True)
                 for k in friends},
                sync=True,
                retries=3,
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
    processor_name = args.processor.split('.')[0].replace("/", '.')
    analysis_class = getattr(importlib.import_module(processor_name), config_runner['class_name'])
    logging.debug(f'Running on fileset {pretty_repr(fileset)}')

    if COFFEA_2025:
        runner_kwargs = dict(
            executor=executor,
            schema=executor_args['schema'],
            savemetrics=executor_args['savemetrics'],
            skipbadfiles=executor_args['skipbadfiles'],
            xrootdtimeout=executor_args['xrootdtimeout'],
            chunksize=executor_args['chunksize'],
            maxchunks=executor_args['maxchunks'],
        )
        runner = processor.Runner(**runner_kwargs)
        result = runner(
            fileset,
            treename='Events',
            processor_instance=analysis_class(**configs.get('config', {})),
        )
        if isinstance(result, tuple):
            output, metrics = result
        else:
            output = result
            metrics = output.pop('metrics', {}) if isinstance(output, dict) else {}
    else:
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
    nEvent = metrics.get('entries', 0)
    logging.info(f'Metrics:')
    logging.info(pretty_repr(metrics))
    logging.info(f'{nEvent/elapsed:,.0f} events/s total ({nEvent}/{elapsed})')

    if args.skimming:
        output = process_skimming_output(output, fileset, configs, config_runner, args, client)
        elapsed = time.time() - tstart
        nEvent = metrics['entries']
        logging.info(f'{nEvent/elapsed:,.0f} events/s total ({nEvent}/{elapsed})')
        process_metadata_output(output, fileset, config_runner, args, client)
    else:
        process_analysis_output(output, args)
        process_friend_trees(output, config_runner, configs, args, client, fileset=fileset)
        save_coffea_output(output, config_runner, args)
