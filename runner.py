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
import shutil

from omegaconf import OmegaConf
from src.utils.addhash import get_git_diff, get_git_revision_hash, find_git_root
from coffea import processor
from coffea.dataset_tools import rucio_utils
from coffea.nanoevents import NanoAODSchema, PFNanoAODSchema

# Patch for SITECONF storage.json entries that use 'site' instead of 'rse'
# (seen after a SITECONF format change; coffea's get_xrootd_sites_map assumes 'rse' always present)
_orig_get_xrootd_sites_map = rucio_utils.get_xrootd_sites_map
def _patched_get_xrootd_sites_map():
    import coffea.dataset_tools.rucio_utils as _ru
    _orig_json_load = _ru.json.load
    def _safe_json_load(f):
        data = _orig_json_load(f)
        for entry in data:
            if 'rse' not in entry and 'site' in entry:
                entry['rse'] = entry['site']
        return data
    _ru.json.load = _safe_json_load
    try:
        # Retry on corrupt/empty .sites_map.json: a concurrent job may be
        # mid-write of the cache file when we read it
        for attempt in range(5):
            try:
                return _orig_get_xrootd_sites_map()
            except json.JSONDecodeError:
                if attempt == 4:
                    raise
                time.sleep(2 + 2 * attempt)
    finally:
        _ru.json.load = _orig_json_load
rucio_utils.get_xrootd_sites_map = _patched_get_xrootd_sites_map

from coffea.util import save
from dask.distributed import performance_report
from distributed.diagnostics.plugin import WorkerPlugin, SchedulerPlugin
from rich.logging import RichHandler
from rich.pretty import pretty_repr
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
from dask.distributed import performance_report, WorkerPlugin
from dataclasses import dataclass

# Defined here (in __main__) so cloudpickle serializes it by value, not by
# module reference. Workers need to deserialize this before the code tarball
# is extracted, so it cannot live in src.runner.cluster.
@dataclass
class WorkerInitializer(WorkerPlugin):
    uproot_xrootd_retry_delays: list[float] = None

    def setup(self, worker=None):
        self.worker = worker
        import os, tarfile, sys, logging
        if os.path.exists("code_barista.tar.gz"):
            if not os.path.exists(".code_extracted"):
                logging.info("Extracting code_barista.tar.gz on worker...")
                with tarfile.open("code_barista.tar.gz", "r:gz") as tar:
                    tar.extractall()
                with open(".code_extracted", "w") as f:
                    f.write("extracted\n")
                logging.info("Code package extracted successfully")
        if os.getcwd() not in sys.path:
            sys.path.insert(0, os.getcwd())
        if delays := self.uproot_xrootd_retry_delays:
            from src.data_formats.root.patch import uproot_XRootD_retry
            uproot_XRootD_retry(len(delays) + 1, delays)

    def transition(self, key, start, finish, **kwargs):
        pass

# Import from our modular sub-package
from src.runner.cli import parse_args, make_parser
from src.runner.env import setup_environment, print_reproducibility_info, check_and_setup_proxy, sync_nfs_writes
from src.runner.cluster import setup_shared_dask_client, setup_condor_cluster, setup_slurm_cluster, setup_local_cluster
from src.runner.dataset import (
    apply_storage_remap, find_matching_dataset, get_dataset_type, calculate_cross_section,
    process_mc_dataset, process_sample_based_dataset, process_data_for_mix, process_tt_for_mixed,
    process_data_dataset, add_fvt_metadata, apply_datasets_filter,
    expand_directory_files, list_of_files
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
            s.bind(('', preferred))
        except OSError:
            s.bind(('', 0))
        return s.getsockname()[1]

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
        if output[dataset].get("missing", {}).get("file_missing"):
            logging.info(f'Merging completed successfully for "{dataset}" — ignore the missing file warnings above, some files had zero selected events or failed silently.')

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

        # Build reverse mapping: parent dir name (path1) -> dataset key
        # This allows the naming function to use the dataset key as the output
        # subdirectory even when input files live in era-named subdirs (e.g. mixeddata_all)
        path1_to_dataset = {}
        if fileset:
            for dataset_key, dataset_info in fileset.items():
                for f in dataset_info["files"]:
                    parent_dir = f.rstrip('/').split('/')[-2]
                    path1_to_dataset[parent_dir] = dataset_key

        def _merge_naming(path0, path1, name, **_):
            dir_name = path1_to_dataset.get(path1, path1)
            return f'{dir_name}/{path0.replace("picoAOD", name)}'

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

def run_daemon_monitoring_loop(client, cluster, scheduler_json_path, idle_timeout):
    """Monitor connected clients and active tasks, shut down when idle."""
    logging.info("Dask cluster daemon monitoring loop started.")
    idle_start = None
    seen_client = False
    startup_deadline = time.time() + max(idle_timeout * 3, 1800)

    while True:
        try:
            # Query scheduler info
            scheduler_info = client.scheduler_info()

            # Count connected clients, excluding this daemon client itself

            try:
                def get_active_clients(dask_scheduler):
                    return [c for c in dask_scheduler.clients.keys() if c != 'fire-and-forget']
                connected_clients = client.run_on_scheduler(get_active_clients)
                active_clients = max(0, len(connected_clients) - 1)
            except Exception as e:
                logging.error(f"Error querying clients on scheduler: {e}")
                connected_clients = scheduler_info.get('clients', {})
                active_clients = max(0, len(connected_clients) - 1)

            connected_clients = scheduler_info.get('clients', {})
            active_clients = max(0, len(connected_clients) - 1)

            # Query number of processing tasks
            processing_tasks = client.processing()
            n_tasks = sum(len(tasks) for tasks in processing_tasks.values()) if processing_tasks else 0

            logging.debug(f"Daemon status: {active_clients} active clients, {n_tasks} tasks processing.")

            if active_clients > 0 or n_tasks > 0:
                seen_client = True
                if idle_start is not None:
                    logging.info("Cluster is active again. Resetting idle timer.")
                idle_start = None
            elif not seen_client:
                # No job has connected yet; the scheduler is warming up, not idle.
                # Backstop: if no client ever connects, shut down so we don't hold
                # Condor workers forever (e.g. job crashed before connecting).
                if time.time() >= startup_deadline:
                    logging.info("No client connected before startup deadline. Shutting down.")
                    break
            else:
                if idle_start is None:
                    idle_start = time.time()
                    logging.info(f"Cluster is idle. Starting countdown to shutdown (timeout: {idle_timeout}s)...")
                else:
                    elapsed = time.time() - idle_start
                    if elapsed >= idle_timeout:
                        logging.info(f"Cluster idle timeout reached ({idle_timeout}s). Shutting down.")
                        break
        except Exception as e:
            logging.error(f"Error in daemon monitoring loop: {e}")
            break

        time.sleep(30)

    # Shutdown logic
    # Immediately delete the JSON file so new clients do not try to connect to a dying scheduler
    try:
        if os.path.exists(scheduler_json_path):
            os.remove(scheduler_json_path)
    except OSError:
        pass

    logging.info("Shutting down Dask cluster and workers...")
    try:
        client.close()
    except Exception:
        pass
    try:
        cluster.close()
    except Exception:
        pass
    logging.info("Daemon shutdown complete. Exiting.")


def setup_shared_dask_client(args, config_runner):
    """Check for/connect to an existing cluster daemon, or spawn one if needed."""
    import hashlib
    import getpass
    import subprocess
    from distributed import Client

    # 1. Determine namespaced file paths
    workspace_hash = os.environ.get("BARISTA_WORKSPACE_HASH")
    if not workspace_hash:
        workspace_path = os.path.abspath(os.path.dirname(__file__))
        workspace_hash = hashlib.md5(workspace_path.encode('utf-8')).hexdigest()[:8]
    username = getpass.getuser()
    daemon_dir = f"/tmp/barista_{username}"
    os.makedirs(daemon_dir, exist_ok=True)
    scheduler_json_path = f"{daemon_dir}/dask_scheduler_{workspace_hash}.json"
    daemon_log_path = f"{daemon_dir}/dask_daemon_{workspace_hash}.log"

    # 2. Explicit scheduler address bypass
    if args.scheduler_address:
        logging.info(f"Connecting to explicit Dask scheduler at {args.scheduler_address}...")
        client = Client(args.scheduler_address)
        return client, None

    def log_daemon_info(data):
        if "daemon_log" in data:
            logging.info(f"Dask daemon log: {data['daemon_log']}")
        if "worker_log_dir" in data:
            logging.info(f"Condor worker log directory: {data['worker_log_dir']}")

    # 3. Connection retry / reuse logic (client process)
    if not args.start_cluster_daemon:
        # Wait up to 15 seconds for the scheduler JSON to appear and be readable
        # (to handle shared filesystem sync/latency)
        start_wait = time.time()
        data = None
        while time.time() - start_wait < 15:
            if os.path.exists(scheduler_json_path):
                try:
                    with open(scheduler_json_path, "r") as f:
                        data = json.load(f)
                    if data and "address" in data:
                        break
                except Exception:
                    pass
            time.sleep(1)

        if not data:
            # Print daemon log to help diagnose why it failed to start
            if os.path.exists(daemon_log_path):
                try:
                    with open(daemon_log_path) as f:
                        tail = f.read()[-4000:]
                    logging.error(f"Dask daemon log ({daemon_log_path}):\n{tail}")
                except Exception:
                    pass
            raise RuntimeError(
                f"Dask scheduler connection file not found: {scheduler_json_path}. "
                f"Check the daemon log at {daemon_log_path} for details."
            )

        address = data["address"]
        logging.info(f"Connecting to shared Dask scheduler at {address}...")

        # Retry connecting to the Dask scheduler to handle high load / slow startup
        client = None
        for attempt in range(1, 6):
            try:
                client = Client(address, timeout="180s")
                logging.info(f"Successfully connected to Dask scheduler (attempt {attempt}/5)!")
                # This per-job process connects to a shared daemon's scheduler, so
                # unlike the cluster-creating path it never logged the dashboard.
                # Emit it here (plus the scheduler host) so monitors
                # (barista_console / runner_monitor) can scan this log, build a
                # reachable dashboard URL, and show live worker/task progress.
                logging.info(f"Dask dashboard: {client.dashboard_link}")
                logging.info(f"Dask scheduler host: {address.split('://')[1].split(':')[0]}")
                log_daemon_info(data)
                return client, None
            except Exception as e:
                if attempt == 5:
                    raise RuntimeError(f"Failed to connect to Dask scheduler at {address} after 5 attempts: {e}")
                logging.warning(f"Connection attempt {attempt}/5 failed: {e}. Retrying in 2 seconds...")
                time.sleep(2)

    # 5. Daemon setup logic (daemon process)
    else:
        logging.info("Initializing Dask cluster daemon...")
        global _temp_condor_dir
        log_dir = None
        if args.condor:
            logging.info("Configuring LPCCondorCluster daemon...")
            tarball_path, _temp_condor_dir = create_code_tarball(config_runner['condor_transfer_input_files'], tmpdir=args.tmpdir)
            client, cluster, log_dir = setup_condor_cluster(config_runner, tarball_path)
        elif args.slurm:
            logging.info("Configuring SLURMCluster daemon...")
            client, cluster = setup_slurm_cluster(config_runner)
        elif args.run_dask:
            logging.info("Configuring LocalCluster daemon...")
            client, cluster = setup_local_cluster(config_runner)
        else:
            raise ValueError("Daemon started without a valid cluster type flag (--condor, --slurm, or --dask)")

        # Write info to JSON
        info_data = {
            "address": client.scheduler.address,
            "pid": os.getpid(),
            "daemon_log": daemon_log_path,
        }
        if log_dir is not None:
            info_data["worker_log_dir"] = log_dir
        with open(scheduler_json_path, "w") as f:
            json.dump(info_data, f)

        # Register worker plugin
        logging.info("Registering worker plugin for Dask client in daemon...")
        worker_initializer = WorkerInitializer(uproot_xrootd_retry_delays=config_runner['uproot_xrootd_retry_delays'])
        client.register_plugin(worker_initializer)

        # Enter monitoring loop (exits process on timeout)
        run_daemon_monitoring_loop(client, cluster, scheduler_json_path, args.idle_timeout)
        sys.exit(0)


def make_parser():

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
        default="coffea4bees/metadata/datasets/",
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
        '--friends',
        dest="friends",
        default="coffea4bees/metadata/friends/friends_HH4b.yml",
        type=lambda x: None if x.lower() == 'none' else x,
        help='Path to the per-year friends metadata YAML file (None to disable)'
    )
    # Central weights configuration path
    io_group.add_argument(
        '--weights',
        dest="weights",
        default="coffea4bees/metadata/weights/weights_HH4b.yml",
        type=lambda x: None if x.lower() == 'none' else x,
        help='Path to the per-year weights/models metadata YAML file (None to disable)'
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
    io_group.add_argument(
        '--dashboard-address',
        dest="dashboard_address",
        default=None,
        type=int,
        metavar='PORT',
        help='Port for the Dask dashboard (default: 10200). Use 0 to let the OS pick a free port, e.g. when running many parallel jobs.'
    )
    io_group.add_argument(
        '--storage-remap',
        dest="storage_remap",
        default=None,
        metavar='FILE',
        help='Path to a YAML file defining storage prefix remappings (e.g. FNAL EOS -> CMU EOS). '
             'Applied to all file paths in the datasets metadata at load time.'
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
        default=['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'C01', 'C02', 'C03', 'C04', 'C11', 'C12', 'C13', 'C14', 'C3', 'C4', 'D1', 'D2', 'D01', 'D02', 'D11', 'D12', 'F1', 'F2', 'F3', 'G1', 'G2', 'I2', 'I3' ],
        help='Data era(s) to process (data only). Examples: --eras A B C.'
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
    exec_group.add_argument(
        '--slurm',
        dest="slurm",
        action="store_true",
        default=False,
        help='Submit Dask workers as SLURM jobs (for falcon compute cluster)'
    )
    exec_group.add_argument(
        '--worker-memory',
        dest="worker_memory",
        default=None,
        help='Override worker memory (e.g. 8GB). Overrides the value in the config file.'
    )
    exec_group.add_argument(
        '--slurm-qos',
        dest="slurm_qos",
        default=None,
        help='Override SLURM QoS for worker jobs (e.g. cpu_light, cpu_medium, cpu_heavy). Overrides slurm_qos in the config file.'
    )
    exec_group.add_argument(
        '--tmpdir',
        dest="tmpdir",
        default=None,
        help='Parent directory for the condor code-tarball temp dir (defaults to /uscmst1b_scratch/lpc1/3DayLifetime/$USER)'
    )
    exec_group.add_argument(
        '--start-cluster-daemon',
        dest="start_cluster_daemon",
        action="store_true",
        default=False,
        help=argparse.SUPPRESS
    )
    exec_group.add_argument(
        '--shared-dask',
        dest="shared_dask",
        action="store_true",
        default=False,
        help='Use a shared Dask cluster daemon'
    )
    exec_group.add_argument(
        '--idle-timeout',
        dest="idle_timeout",
        type=int,
        default=600,
        help='Time in seconds to wait before shutting down an idle cluster (default: 600s / 10m)'
    )
    exec_group.add_argument(
        '--scheduler-address',
        dest="scheduler_address",
        default=None,
        help='Address of an existing Dask scheduler to connect to (e.g. tcp://IP:PORT)'
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
    return parser

from src.runner.logging import CustomFormatter

if __name__ == '__main__':
    # 1. Parse arguments (supports both standard arguments and loading from YAML config)
    args = parse_args()

    # 2. Configure logging
    use_color = not getattr(args, 'no_color', False) and os.environ.get("NO_COLOR") not in ("1", "true", "True")
    logging_level = logging.DEBUG if getattr(args, 'debug', False) else logging.INFO
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging_level)
    handler.setFormatter(CustomFormatter(use_color=use_color))
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
        configs = copy.deepcopy(args.configs) if isinstance(args.configs, dict) else (yaml.safe_load(open(args.configs, 'r')) or {})
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
        client, cluster = setup_shared_dask_client(args, config_runner, WorkerInitializer)
        sys.exit(0)

    # 5. Load configuration and metadata files
    logging.info(">>> Checking and creating output directory")
    logging.info(f"Output directory: {args.output_path}")
    if args.output_path and not os.path.exists(args.output_path):
        os.makedirs(args.output_path)

    logging.info("Loading configuration and metadata files...")
    logging.info(f"Loading configs from: {args.configs}")
    configs = copy.deepcopy(args.configs) if isinstance(args.configs, dict) else (yaml.safe_load(open(args.configs, 'r')) or {})

    # If a full job YAML was supplied (Mode 1) and points to an external analysis_config,
    # merge any top-level 'runner' or 'config' block overrides from the job YAML.
    if getattr(args, 'job_yaml_path', None) and args.job_yaml_path != args.configs and isinstance(args.job_yaml_path, str) and os.path.exists(args.job_yaml_path):
        job_yaml = yaml.safe_load(open(args.job_yaml_path, 'r')) or {}
        if 'runner' in job_yaml and isinstance(job_yaml['runner'], dict):
            configs.setdefault('runner', {}).update(job_yaml['runner'])
        if 'config' in job_yaml and isinstance(job_yaml['config'], dict):
            configs.setdefault('config', {}).update(job_yaml['config'])

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
            client, cluster = setup_shared_dask_client(args, config_runner, WorkerInitializer)
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
