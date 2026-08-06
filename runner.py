from __future__ import annotations

import logging
import os
import sys
import time
import atexit
import yaml
import importlib
import inspect
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor
from rich.logging import RichHandler
from rich.pretty import pretty_repr
from omegaconf import OmegaConf
import copy

# Monkey-patch coffea's rucio_utils to prevent KeyError: 'rse' for incomplete SITECONF JSONs
try:
    from coffea.dataset_tools import rucio_utils
    def patched_get_xrootd_sites_map():
        import json, os, time
        from collections import defaultdict
        sites_xrootd_access = defaultdict(dict)
        cache_valid = False
        if os.path.exists(".sites_map.json"):
            file_time = os.path.getmtime(".sites_map.json")
            if file_time > time.time() - 600:
                cache_valid = True
        if not os.path.exists(".sites_map.json") or not cache_valid:
            siteconf_dir = "/cvmfs/cms.cern.ch/SITECONF/"
            if os.path.exists(siteconf_dir):
                sites = [
                    (s, os.path.join(siteconf_dir, s, "storage.json"))
                    for s in os.listdir(siteconf_dir)
                    if s.startswith("T")
                ]
                for site_name, conf in sites:
                    if not os.path.exists(conf):
                        continue
                    try:
                        data = json.load(open(conf))
                    except Exception:
                        continue
                    for site in data:
                        if site.get("type") != "DISK":
                            continue
                        if site.get("rse") is None:
                            continue
                        for proc in site.get("protocols", []):
                            if proc.get("protocol") == "XRootD":
                                if proc.get("access") not in ["global-ro", "global-rw"]:
                                    continue
                                if "prefix" not in proc:
                                    if "rules" in proc:
                                        for rule in proc["rules"]:
                                            sites_xrootd_access[site["rse"]][rule["lfn"]] = rule["pfn"]
                                else:
                                    sites_xrootd_access[site["rse"]] = proc["prefix"]
            json.dump(sites_xrootd_access, open(".sites_map.json", "w"))
        return json.load(open(".sites_map.json"))
    rucio_utils.get_xrootd_sites_map = patched_get_xrootd_sites_map
except Exception:
    pass

from coffea import processor
from dask.distributed import performance_report

# Import from our modular sub-package
from src.runner.cli import parse_args, make_parser
from src.runner.env import setup_environment, print_reproducibility_info, check_and_setup_proxy, sync_nfs_writes
from src.runner.cluster import setup_shared_dask_client, setup_condor_cluster, setup_slurm_cluster, setup_local_cluster, WorkerInitializer
from src.runner.dataset import (
    apply_storage_remap, find_matching_dataset, get_dataset_type, calculate_cross_section,
    process_mc_dataset, process_sample_based_dataset, process_data_for_mix, process_tt_for_mixed,
    process_data_dataset, add_fvt_metadata, apply_datasets_filter
)
from src.runner.orchestrator import (
    setup_schema, setup_executor, run_job, find_free_port, setup_config_defaults, setup_pico_base_name
)

# Global variable to track temp directory for cleanup (Condor)
_temp_condor_dir = None

def cleanup_temp_condor_dir():
    """Cleanup the temporary directory created for Condor code transfer."""
    global _temp_condor_dir
    if _temp_condor_dir and os.path.exists(_temp_condor_dir):
        logging.info(f"Cleaning up temporary Condor directory: {_temp_condor_dir}")
        import shutil
        try:
            shutil.rmtree(_temp_condor_dir)
        except OSError as e:
            logging.error(f"Error cleaning up Condor directory: {e}")
class CustomFormatter(logging.Formatter):
    COLORS = {
        'DEBUG': '\033[36m',    # Cyan
        'INFO': '\033[32m',     # Green
        'WARNING': '\033[33m',  # Yellow
        'ERROR': '\033[31m',    # Red
        'CRITICAL': '\033[41m', # Red background
    }
    RESET = '\033[0m'

    def format(self, record):
        # Format time to matches the exact user timestamp style
        asctime = self.formatTime(record, "%y/%m/%d %H:%M:%S")
        use_color = sys.stdout.isatty()
        color = self.COLORS.get(record.levelname, '') if use_color else ''
        reset = self.RESET if use_color else ''
        colored_level = f"{color}{record.levelname:<8}{reset}" if color else f"{record.levelname:<8}"
        
        source = f"{record.filename}:{record.lineno}"
        header = f"[{asctime}] {colored_level} {source}"
        message = record.getMessage()
        return f"{header}\n{message}"

if __name__ == '__main__':
    # 1. Parse arguments (supports both standard arguments and loading from YAML config)
    args = parse_args()

    # 2. Configure logging
    logging_level = logging.DEBUG if getattr(args, 'debug', False) else logging.INFO
    handler = RichHandler(level=logging_level, show_time=False, show_level=False, show_path=False, markup=False)
    handler.setFormatter(CustomFormatter())
    logging.basicConfig(
        level=logging_level,
        handlers=[handler],
        force=True,
    )
    # Disable verbose logging from third-party libraries
    logging.getLogger('numba').setLevel(logging.WARNING)
    logging.getLogger("lpcjobqueue").setLevel(logging.WARNING)
    logging.getLogger("dask_jobqueue").setLevel(logging.WARNING)

    # Re-execute under mprof if requested and not already running under it
    if getattr(args, 'run_performance', False) and not os.environ.get("RUNNER_MPROF_ACTIVE"):
        os.environ["RUNNER_MPROF_ACTIVE"] = "1"
        import shutil
        mprof_cmd = shutil.which("mprof")
        if not mprof_cmd:
            logging.error("mprof command not found! Cannot run memory profiling. Please ensure memory_profiler is installed.")
            sys.exit(1)
        
        output_dir = getattr(args, 'output_path', 'output/')
        output_file = getattr(args, 'output_file', 'test.coffea')
        os.makedirs(output_dir, exist_ok=True)
        
        username = os.environ.get("USER", "barista")
        dat_dir = f"/tmp/{username}"
        os.makedirs(dat_dir, exist_ok=True)
        base_name = os.path.splitext(output_file)[0]
        mprofile_dat = os.path.join(dat_dir, f"mprofile_{base_name}.dat")
        mprofile_png = os.path.join(output_dir, "performance", f"mprofile_{base_name}.png")
        os.makedirs(os.path.dirname(mprofile_png), exist_ok=True)
        
        clean_argv = [arg for arg in sys.argv if arg != "--run-performance"]
        run_cmd = [mprof_cmd, "run", "-C", "-o", mprofile_dat, sys.executable] + clean_argv
        
        logging.info(f"Running performance profiling: {' '.join(run_cmd)}")
        import subprocess
        res = subprocess.run(run_cmd)
        
        if res.returncode == 0 and os.path.exists(mprofile_dat):
            logging.info("Generating performance plot...")
            plot_cmd = [mprof_cmd, "plot", "-o", mprofile_png, mprofile_dat]
            subprocess.run(plot_cmd)
            logging.info(f"Performance plot created successfully: {mprofile_png}")
            
        sys.exit(res.returncode)

    # 3. Setup environment and check proxy (porting functionalities from run-analysis-processor.sh)
    setup_environment(args)
    
    # Check if we should setup/verify the grid proxy
    if not getattr(args, 'not_do_proxy', False):
        check_and_setup_proxy(args)
    else:
        logging.info("Proxy setup skipped (--not-do-proxy / not_do_proxy: true)")

    # Print reproducibility details
    print_reproducibility_info(args)

    # 4. Handle start_cluster_daemon mode (Daemon process setup)
    if getattr(args, 'start_cluster_daemon', False):
        logging.info("Running in daemon mode. Skipping metadata loading and dataset processing.")
        configs = yaml.safe_load(open(args.configs, 'r'))
        if not 'config' in configs:
            configs['config'] = {}
        with open("src/physics/corrections.yml", "r") as f:
            corrections_metadata = yaml.safe_load(f)
        configs['config']['corrections_metadata'] = corrections_metadata
        config_runner = configs['runner'] if 'runner' in configs.keys() else {}
        args.shared_dask = True
        setup_config_defaults(config_runner, args)
        if getattr(args, 'dashboard_address', None) is not None:
            config_runner['dashboard_address'] = args.dashboard_address
        if getattr(args, 'worker_memory', None) is not None:
            config_runner['worker_memory'] = args.worker_memory
        if getattr(args, 'slurm_qos', None) is not None:
            config_runner['slurm_qos'] = args.slurm_qos
        if config_runner['dashboard_address'] != 0:
            requested = config_runner['dashboard_address']
            config_runner['dashboard_address'] = find_free_port(requested)
        setup_schema(config_runner)

        args.run_dask = True
        client, cluster = setup_shared_dask_client(args, config_runner)
        sys.exit(0)

    # 5. Load configuration and metadata files
    logging.info(">>> Checking and creating output directory")
    logging.info(f"Output directory: {args.output_path}")
    if args.output_path and not os.path.exists(args.output_path):
        os.makedirs(args.output_path)

    logging.info("Loading configuration and metadata files...")
    logging.info(f"Loading configs from: {args.configs}")
    configs = yaml.safe_load(open(args.configs, 'r'))

    # Apply config overrides
    if getattr(args, 'config_overrides', None):
        logging.info(">>> Applying config overrides")
        if not 'config' in configs:
            configs['config'] = {}
        for key, val in args.config_overrides.items():
            orig_val = configs['config'].get(key, '<Not Set>')
            configs['config'][key] = val
            print(f"  Override: config.{key} = {val} (original: {orig_val})")

    logging.info(">>> Modifying config")
    print(yaml.dump(configs, default_flow_style=False))

    if not 'config' in configs:
        configs['config'] = {}
    
    # Load corrections_metadata
    logging.info("Loading corrections metadata from: src/physics/corrections.yml")
    with open("src/physics/corrections.yml", "r") as f:
        corrections_metadata = yaml.safe_load(f)
    configs['config']['corrections_metadata'] = corrections_metadata

    # Handle blinding (replaces sed configuration patches)
    if getattr(args, 'blind', False):
        logging.info("Blinding SR region: setting blind = True in configuration")
        configs['config']['blind'] = True

    if getattr(args, 'systematics', None):
        logging.info(f"Systematics to run: {args.systematics}")
        configs['config']['run_systematics'] = args.systematics

    # Load datasets metadata (supports multiple files merging)
    if getattr(args, 'datasets_metadata_files', None):
        logging.info(">>> Merging datasets metadata files")
        merged_datasets = {}
        for fpath in args.datasets_metadata_files:
            print(f"  Loading: {fpath}")
            with open(fpath, 'r') as f:
                f_data = yaml.safe_load(f)
                if isinstance(f_data, dict):
                    if 'datasets' in f_data:
                        merged_datasets.update(f_data['datasets'])
                    else:
                        merged_datasets.update(f_data)
        datasets = {'datasets': merged_datasets}
        print(f"Merged datasets metadata: loaded {len(merged_datasets)} top-level dataset keys.")
    else:
        logging.info(f"Loading datasets metadata from: {args.metadata}")
        if os.path.isdir(args.metadata):
            files = [OmegaConf.load(os.path.join(args.metadata, f)) for f in os.listdir(args.metadata) if f.endswith(('.yaml', '.yml'))]
            datasets = OmegaConf.to_container(OmegaConf.create({'datasets': OmegaConf.merge(*files)}), resolve=True)
        else:
            datasets = yaml.safe_load(open(args.metadata, 'r'))
            if isinstance(datasets, dict) and 'datasets' not in datasets:
                datasets = {'datasets': datasets}

    # Apply dataset exclusions/filters
    if getattr(args, 'datasets_filter', None):
        logging.info(">>> Applying dataset exclusions")
        datasets = apply_datasets_filter(datasets, args.datasets_filter)
        logging.info(">>> Modified Datasets Structure:")
        summary_lines = []
        for k, v in datasets.get('datasets', {}).items():
            if isinstance(v, dict):
                years = [y for y in v.keys() if y not in ['xs', 'nSamples', 'type', 'label', 'xsec', 'color']]
                summary_lines.append(f"  - {k}: years={years}")
            else:
                summary_lines.append(f"  - {k}: {type(v)}")
        print("\n".join(summary_lines))

    # Load triggers and luminosities metadata
    logging.info(f"Loading triggers metadata from: {args.triggers}")
    triggers = yaml.safe_load(open(args.triggers, 'r'))

    logging.info(f"Loading luminosities metadata from: {args.luminosities}")
    luminosities = yaml.safe_load(open(args.luminosities, 'r'))

    metadata = {**datasets, **triggers, **luminosities}

    # Apply storage remappings if provided
    if getattr(args, 'storage_remap', None):
        logging.info(f"Applying storage remaps from: {args.storage_remap}")
        with open(args.storage_remap, 'r') as f:
            remap_config = yaml.safe_load(f)
        remaps = remap_config.get('remaps', [])
        logging.info(f"  {len(remaps)} prefix remap(s) loaded")
        metadata = apply_storage_remap(metadata, remaps)
        logging.info("Storage remapping applied.")

    logging.info("Successfully loaded all metadata files")

    # 6. Setup runner configuration defaults
    logging.info("Setting up configuration defaults...")
    config_runner = configs['runner'] if 'runner' in configs.keys() else {}
    setup_config_defaults(config_runner, args)
    
    if getattr(args, 'dashboard_address', None) is not None:
        config_runner['dashboard_address'] = args.dashboard_address
    if getattr(args, 'worker_memory', None) is not None:
        config_runner['worker_memory'] = args.worker_memory
    if getattr(args, 'slurm_qos', None) is not None:
        config_runner['slurm_qos'] = args.slurm_qos

    if config_runner['dashboard_address'] != 0:
        requested = config_runner['dashboard_address']
        config_runner['dashboard_address'] = find_free_port(requested)
        if config_runner['dashboard_address'] != requested:
            logging.info(f"Dashboard port {requested} in use, using {config_runner['dashboard_address']} instead.")
    setup_schema(config_runner)
    logging.info(f"Configuration setup complete. Data tier: {config_runner['data_tier']}, Schema: {config_runner['schema'].__name__}")

    # 7. Process datasets to build fileset
    logging.info(f"Starting dataset processing for {len(args.years)} year(s) and {len(args.datasets)} dataset(s)")
    metadata_dataset = {}
    fileset = {}

    for year in args.years:
        logging.info(f"Processing year: {year}")
        for dataset in args.datasets:
            logging.info(f"Processing dataset: {dataset}")

            matched_dataset = find_matching_dataset(dataset, metadata)
            if matched_dataset is None:
                logging.warning(f"Skipping dataset {dataset} - no match found")
                continue

            if year not in metadata['datasets'][matched_dataset]:
                logging.warning(f"Skipping {dataset} for {year} - year not available in metadata")
                continue

            dataset_type = get_dataset_type(matched_dataset)
            xsec = calculate_cross_section(matched_dataset, dataset_type, metadata, year)
            logging.info(f"Dataset type: {dataset_type}, Cross-section: {xsec}")

            metadata_dataset[matched_dataset] = {
                'year': year,
                'processName': matched_dataset,
                'xs': xsec,
                'lumi': float(metadata['luminosities'][year]),
                'trigger': metadata['triggers'][year],
            }

            if dataset_type == 'mc':
                process_mc_dataset(matched_dataset, year, metadata, metadata_dataset, fileset, args, config_runner)
            elif dataset_type == 'mixed_data':
                process_sample_based_dataset('mixed_data', 'mix', matched_dataset, year, metadata, metadata_dataset, fileset, args, config_runner, add_fvt_metadata)
            elif dataset_type == 'mixeddata_all':
                process_data_dataset(matched_dataset, year, metadata, metadata_dataset, fileset, args, config_runner)
            elif dataset_type == 'mixeddata_4b':
                process_sample_based_dataset('mixeddata_4b', 'mix', matched_dataset, year, metadata, metadata_dataset, fileset, args, config_runner)
            elif dataset_type in ['mixeddata_4b_noTT']:
                process_sample_based_dataset('mixeddata_4b', 'mix_noTT', matched_dataset, year, metadata, metadata_dataset, fileset, args, config_runner)
            elif dataset_type in ['mixeddata_4b_pz']:
                process_sample_based_dataset('mixeddata_4b', 'mix_pz', matched_dataset, year, metadata, metadata_dataset, fileset, args, config_runner)
            elif dataset_type == 'data_mixed':
                process_sample_based_dataset('data_mixed', 'mix', matched_dataset, year, metadata, metadata_dataset, fileset, args, config_runner)
            elif dataset_type == 'synthetic_data':
                process_sample_based_dataset('synthetic_data', 'syn', matched_dataset, year, metadata, metadata_dataset, fileset, args, config_runner)
            elif dataset_type == 'synthetic_data_noTT':
                process_sample_based_dataset('synthetic_data', 'syn_noTT', matched_dataset, year, metadata, metadata_dataset, fileset, args, config_runner)
            elif dataset_type == 'data_for_mix':
                process_data_for_mix(matched_dataset, year, metadata, metadata_dataset, fileset, args, config_runner)
            elif dataset_type == 'tt_for_mixed':
                process_tt_for_mixed(matched_dataset, year, metadata, metadata_dataset, fileset, args, config_runner)
            elif dataset_type == 'data':
                process_data_dataset(matched_dataset, year, metadata, metadata_dataset, fileset, args, config_runner)

    logging.info(f"Dataset processing complete. Total datasets in fileset: {len(fileset)}")
    logging.debug(f"fileset is {pretty_repr(fileset)}")
    if fileset:
        total_files = sum(len(dataset_info['files']) for dataset_info in fileset.values())
        logging.info(f"Total files across all datasets: {total_files}")

    # 8. Setup compute environment (Standalone Dask/HTCondor/SLURM clusters or local thread pool)
    logging.info("Setting up compute environment...")
    client = None
    pool = None
    cluster = None

    atexit.register(cleanup_temp_condor_dir)

    if getattr(args, 'condor', False) or getattr(args, 'slurm', False) or getattr(args, 'scheduler_address', None) or getattr(args, 'run_dask', False):
        args.run_dask = True
        if getattr(args, 'shared_dask', False):
            logging.info("Configuring shared Dask cluster client...")
            client, cluster = setup_shared_dask_client(args, config_runner)
        else:
            if getattr(args, 'scheduler_address', None):
                logging.info(f"Connecting to explicit Dask scheduler at {args.scheduler_address}...")
                from dask.distributed import Client
                client = Client(args.scheduler_address)
                cluster = None
            elif getattr(args, 'condor', False):
                logging.info("Configuring standalone LPCCondorCluster...")
                from src.runner.cluster import create_code_tarball
                tarball_path, _temp_condor_dir = create_code_tarball(config_runner['condor_transfer_input_files'], tmpdir=args.tmpdir)
                client, cluster, log_dir = setup_condor_cluster(config_runner, tarball_path)
            elif getattr(args, 'slurm', False):
                logging.info("Configuring standalone SLURMCluster...")
                client, cluster = setup_slurm_cluster(config_runner)
            elif getattr(args, 'run_dask', False):
                logging.info("Configuring standalone LocalCluster...")
                client, cluster = setup_local_cluster(config_runner)
    else:
        logging.info("Configuring local process pool execution...")
        worker_initializer = WorkerInitializer(uproot_xrootd_retry_delays=config_runner['uproot_xrootd_retry_delays'])
        pool = ProcessPoolExecutor(max_workers=config_runner['workers'], initializer=worker_initializer.setup)
        logging.info(f"Process pool created with {config_runner['workers']} workers")

    if client is not None and (not getattr(args, 'shared_dask', False) or getattr(args, 'start_cluster_daemon', False) or getattr(args, 'scheduler_address', None)):
        logging.info("Registering worker plugin for Dask client...")
        worker_initializer = WorkerInitializer(uproot_xrootd_retry_delays=config_runner['uproot_xrootd_retry_delays'])
        client.register_plugin(worker_initializer)

    # 9. Setup executor and load processor class
    logging.info("Setting up processor executor...")
    executor, executor_args = setup_executor(config_runner, args, client, pool)
    logging.info(f"Executor arguments:")
    logging.info(pretty_repr(executor_args))

    logging.info("Loading processor class...")
    processor_name = args.processor.split('.')[0].replace("/", '.')
    analysis_class = getattr(importlib.import_module(processor_name), config_runner['class_name'])
    logging.info(f"Successfully loaded processor: {processor_name}.{config_runner['class_name']}")

    # Inject per-year friends
    year_friends = {}
    if getattr(args, 'friends', None) and 'friends' in inspect.signature(analysis_class.__init__).parameters:
        logging.info(f"Loading friends metadata from: {args.friends}")
        friends_by_year = yaml.safe_load(open(args.friends, 'r')).get('friends', {})
        for year in args.years:
            for k, v in friends_by_year.get(year, {}).items():
                if k in year_friends and year_friends[k] != v:
                    logging.warning(f"Friends key '{k}' has conflicting values across years {args.years}; using value for {year}")
                year_friends[k] = v
        if year_friends:
            existing_friends = configs.get('config', {}).get('friends') or {}
            configs.setdefault('config', {})['friends'] = {**year_friends, **existing_friends}
            logging.info(f"Injected per-year friends for {args.years}: {list(year_friends.keys())}")

    # Inject per-year weights if specified and accepted by the processor
    if getattr(args, 'weights', None) and 'weights' in inspect.signature(analysis_class.__init__).parameters:
        logging.info(f"Loading weights metadata from: {args.weights}")
        configs.setdefault('config', {})['weights'] = args.weights

    logging.info(f"Final fileset contains {len(fileset)} datasets:")
    print("\n".join(f"  - {dk}: {len(fileset[dk]['files'])} files" for dk in sorted(fileset.keys())))

    # 10. Execute the job
    logging.info("=" * 60)
    logging.info("STARTING JOB EXECUTION")
    logging.info("=" * 60)
    tstart = time.time()

    if getattr(args, 'run_dask', False):
        os.makedirs(args.output_path, exist_ok=True)
        dask_report_file = f'{args.output_path}/barista-dask-report-{datetime.today().strftime("%Y-%m-%d_%H-%M-%S")}.html'
        logging.info(f"Starting Dask job with performance reporting to: {dask_report_file}")
        _cm = performance_report(filename=dask_report_file)
        _cm.__enter__()
        _exc = None
        try:
            run_job(fileset, configs, config_runner, executor, executor_args, args, client, tstart)
        except BaseException as e:
            _exc = e
            raise
        finally:
            try:
                _cm.__exit__(*(type(_exc), _exc, _exc.__traceback__) if _exc else (None, None, None))
            except Exception as e:
                logging.warning(f"Dask performance report teardown failed (does not change job outcome): {type(e).__name__}: {e}")

        logging.info("Cleaning up Dask resources...")
        for obj_name, obj in [("cluster", cluster), ("client", client)]:
            if obj is not None:
                try:
                    obj.close()
                    logging.info(f"Successfully closed {obj_name}")
                except (RuntimeError, NameError, AttributeError) as e:
                    logging.warning(f"Error closing {obj_name}: {e}")

        logging.info(f'Dask performance report saved in {dask_report_file}')
    else:
        logging.info("Starting local job execution...")
        run_job(fileset, configs, config_runner, executor, executor_args, args, client, tstart)

    # 11. Final cleanups and NFS synchronization
    if pool is not None:
        logging.info("Shutting down process pool...")
        pool.shutdown(wait=True)
        logging.info("Process pool shutdown complete")

    logging.info("=" * 60)
    logging.info("JOB EXECUTION COMPLETED SUCCESSFULLY")
    logging.info("=" * 60)

    # Sync and sleep to flush NFS writes before exiting
    sync_nfs_writes()
    # Trigger CI pipeline rerun
    os._exit(0)
