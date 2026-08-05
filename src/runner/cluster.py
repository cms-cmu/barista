from __future__ import annotations
import os
import sys
import time
import socket
import json
import logging
import getpass
import uuid
import hashlib
from dataclasses import dataclass
from rich.pretty import pretty_repr

import dask
from dask.distributed import WorkerPlugin, SchedulerPlugin
import dask.distributed
import distributed

@dataclass
class WorkerInitializer(WorkerPlugin):
    uproot_xrootd_retry_delays: list[float] = None

    def setup(self, worker=None):
        self.worker = worker
        import os
        import tarfile
        import sys

        # Unpack code package if present (for HTCondor jobs)
        if os.path.exists("code_barista.tar.gz"):
            if not os.path.exists(".code_extracted"):
                logging.info("Extracting code_barista.tar.gz on worker...")
                with tarfile.open("code_barista.tar.gz", "r:gz") as tar:
                    tar.extractall()
                with open(".code_extracted", "w") as f:
                    f.write("extracted\n")
                logging.info("Code package extracted successfully")
            else:
                logging.info("Code package already extracted, skipping")

        # Add current directory to Python path so imports work
        if os.getcwd() not in sys.path:
            sys.path.insert(0, os.getcwd())
            logging.info(f"Added {os.getcwd()} to sys.path")

        if delays := self.uproot_xrootd_retry_delays:
            from src.data_formats.root.patch import uproot_XRootD_retry
            uproot_XRootD_retry(len(delays) + 1, delays)

    def transition(self, key, start, finish, **kwargs):
        if finish == "executing":
            worker_name = self.worker.name if self.worker else "unknown"

def create_code_tarball(condor_transfer_input_files, tmpdir=None):
    """Create a tarball of code in a temporary directory.

    Each job gets a unique temporary directory to avoid conflicts
    between concurrent jobs on shared clusters.
    """
    import tarfile
    logging.info("Creating code tarball for HTCondor transfer...")
    
    # Use specified tmpdir or default to LPC scratch directory
    if tmpdir is None:
        user = getpass.getuser()
        tmpdir = f"/uscmst1b_scratch/lpc1/3DayLifetime/{user}/condor_tmp"
        
    temp_dir = os.path.join(tmpdir, f"barista_{uuid.uuid4().hex[:8]}")
    os.makedirs(temp_dir, exist_ok=True)
    tarball_path = os.path.join(temp_dir, "code_barista.tar.gz")

    with tarfile.open(tarball_path, "w:gz") as tar:
        for path in condor_transfer_input_files:
            if os.path.exists(path):
                logging.info(f"  Adding {path} to tarball...")
                tar.add(path)
            else:
                logging.warning(f"  Warning: path {path} not found, skipping...")
                
    logging.info(f"Code tarball created successfully at {tarball_path}")
    return tarball_path, temp_dir

def setup_condor_cluster(config_runner, tarball_path):
    from lpcjobqueue import LPCCondorCluster

    logging.info("Initializing HTCondor cluster configuration...")

    _log_base = f'/uscmst1b_scratch/lpc1/3DayLifetime/{getpass.getuser()}/condor_logs'
    _default_log_dir = f'{_log_base}_{uuid.uuid4().hex[:8]}'

    cluster_args = {
        'transfer_input_files': [tarball_path],
        'shared_temp_directory': '/tmp',
        'cores': config_runner['condor_cores'],
        'memory': config_runner['worker_memory'],
        'ship_env': False,
        'log_directory': config_runner.get('log_directory', _default_log_dir),
        'scheduler_options': {'dashboard_address': f":{config_runner['dashboard_address']}"},
        'worker_extra_args': [
            f"--worker-port 10000:10100",
            f"--nanny-port 10100:10200",
        ],
        'job_extra_directives': {
            'leave_in_queue': 'False',
            'periodic_remove': '(JobStatus == 5 && (CurrentTime - EnteredCurrentStatus) > 300)'
        },
        'env_extra': ['export PYTHONPATH=.:$PYTHONPATH'],
    }
    if config_runner.get('worker_log_directory'):
        cluster_args['log_directory'] = config_runner['worker_log_directory']

    if os.getenv("WORKER_IMAGE"):
        logging.info(f"Overriding worker image with: {os.getenv('WORKER_IMAGE')}")
        cluster_args['image'] = os.getenv("WORKER_IMAGE")

    logging.info("Cluster arguments: ")
    logging.info(pretty_repr(cluster_args))

    logging.info("Creating HTCondor cluster...")
    cluster = LPCCondorCluster(**cluster_args)

    logging.info("Creating Dask client...")
    client = dask.distributed.Client(cluster)

    logging.info(f"Setting up adaptive scaling (min: {config_runner['min_workers']}, max: {config_runner['max_workers']})")
    cluster.adapt(minimum=config_runner['min_workers'], maximum=config_runner['max_workers'])
    logging.info(f"Dask dashboard: {client.dashboard_link}")
    logging.info(f"Dask scheduler host: {socket.gethostname()}")

    log_dir = cluster_args['log_directory']
    logging.info(f"Condor worker log directory: {log_dir}")

    class WorkerLostLogger(SchedulerPlugin):
        def _find_log(self, worker_addr):
            import glob
            try:
                for path in glob.glob(f"{log_dir}/worker-*.err"):
                    with open(path) as f:
                        if worker_addr in f.read():
                            return path
            except OSError:
                pass
            return None

        def transition(self, key, start, finish, *args, worker=None, **kwargs):
            if finish != "erred":
                return
            exc = kwargs.get("exception")
            exc_text = None
            if exc is not None:
                try:
                    from distributed.protocol import deserialize
                    err = deserialize(exc.header, exc.frames) if hasattr(exc, "header") else exc
                    exc_text = repr(err)
                except Exception:
                    exc_text = repr(exc)
            if exc_text:
                logging.error(f"Task failed: {key}: {exc_text}")
            elif worker is not None:
                log_file = self._find_log(worker)
                if log_file:
                    logging.error(f"Task permanently failed: {key} -> {log_file}")
                else:
                    logging.error(f"Task permanently failed: {key} on {worker} (log not found, check {log_dir}/)")

    client.register_plugin(WorkerLostLogger())

    logging.info('HTCondor cluster setup complete!')
    return client, cluster, log_dir

def setup_slurm_cluster(config_runner):
    """Setup Dask SLURMCluster for falcon compute nodes."""
    from dask_jobqueue import SLURMCluster

    log_base = config_runner.get('slurm_log_directory', 'slurm_logs')
    log_dir = os.path.abspath(os.path.join(log_base, uuid.uuid4().hex[:8]))
    os.makedirs(log_dir, exist_ok=True)

    barista_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    bin_dir = os.path.join(barista_root, 'software', 'slurm')
    worker_python = os.path.join(bin_dir, 'dask-worker-python')

    if not os.access(worker_python, os.X_OK):
        raise FileNotFoundError(f"dask worker wrapper not found or not executable: {worker_python}")

    hostname = socket.gethostname()
    is_bridges2 = 'bridges2' in hostname or 'psc' in hostname

    if not is_bridges2:
        os.environ['PATH'] = bin_dir + os.pathsep + os.environ.get('PATH', '')

    default_partition = 'RM-shared' if is_bridges2 else 'work'
    partition = config_runner.get('slurm_partition', default_partition)
    if partition == 'work' and is_bridges2:
        partition = 'RM-shared'

    job_extra = list(config_runner.get('slurm_job_extra', []))

    account = os.environ.get('SLURM_ACCOUNT')
    if is_bridges2 and account:
        job_extra.append(f"-A {account}")

    cluster_args = {
        'cores': config_runner['slurm_cores'],
        'memory': config_runner['worker_memory'],
        'walltime': config_runner.get('slurm_walltime', '08:00:00'),
        'queue': partition,
        'job_extra_directives': job_extra,
        'log_directory': log_dir,
        'python': worker_python,
        'scheduler_options': {'dashboard_address': f":{config_runner['dashboard_address']}"},
    }
    if is_bridges2:
        cluster_args['processes'] = 1
    if config_runner.get('slurm_qos') and not is_bridges2:
        cluster_args['job_extra_directives'] = (
            list(cluster_args['job_extra_directives']) + [f"--qos={config_runner['slurm_qos']}"]
        )

    logging.info("Creating SLURMCluster with args:")
    logging.info(pretty_repr(cluster_args))

    cluster = SLURMCluster(**cluster_args)
    cluster.adapt(
        minimum=config_runner['min_workers'],
        maximum=config_runner['max_workers'],
    )

    client = dask.distributed.Client(cluster)
    logging.info(f"Dask dashboard: {client.dashboard_link}")
    logging.info(f"Dask scheduler host: {socket.gethostname()}")
    logging.info(f"SLURM worker log directory: {log_dir}")

    logging.info("SLURM cluster setup complete!")
    return client, cluster

def setup_local_cluster(config_runner):
    """Setup local Dask cluster configuration."""
    from dask.distributed import LocalCluster

    dashboard_addr = config_runner['dashboard_address']
    cluster_args = {
        'n_workers': config_runner['workers'],
        'memory_limit': config_runner['worker_memory'],
        'threads_per_worker': 1,
        'dashboard_address': f":{dashboard_addr}",
        'scheduler_port': 0 if dashboard_addr == 0 else 8786,
    }
    cluster = LocalCluster(**cluster_args)
    client = dask.distributed.Client(cluster)
    logging.info(f"Dask dashboard: {client.dashboard_link}")
    logging.info(f"Dask scheduler host: {socket.gethostname()}")
    if dashboard_addr != 0:
        logging.info(f"  SSH tunnel:   ssh -L {dashboard_addr}:<compute_node>:{dashboard_addr} <login_node>")
    return client, cluster

def run_daemon_monitoring_loop(client, cluster, scheduler_json_path, idle_timeout):
    """Monitor connected clients and active tasks, shut down when idle."""
    logging.info("Dask cluster daemon monitoring loop started.")
    idle_start = None

    while True:
        try:
            scheduler_info = client.scheduler_info()

            try:
                def get_active_clients(dask_scheduler):
                    return [c for c in dask_scheduler.clients.keys() if c != 'fire-and-forget']
                connected_clients = client.run_on_scheduler(get_active_clients)
                active_clients = max(0, len(connected_clients) - 1)
            except Exception as e:
                logging.error(f"Error querying clients on scheduler: {e}")
                connected_clients = scheduler_info.get('clients', {})
                active_clients = max(0, len(connected_clients) - 1)

            processing_tasks = client.processing()
            n_tasks = sum(len(tasks) for tasks in processing_tasks.values()) if processing_tasks else 0

            logging.debug(f"Daemon status: {active_clients} active clients, {n_tasks} tasks processing.")

            if active_clients > 0 or n_tasks > 0:
                if idle_start is not None:
                    logging.info("Cluster is active again. Resetting idle timer.")
                idle_start = None
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
    # We import internally to keep imports clean
    global _temp_condor_dir
    
    workspace_hash = os.environ.get("BARISTA_WORKSPACE_HASH")
    if not workspace_hash:
        workspace_path = os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
        workspace_hash = hashlib.md5(workspace_path.encode('utf-8')).hexdigest()[:8]
    username = getpass.getuser()
    daemon_dir = f"/tmp/barista_{username}"
    os.makedirs(daemon_dir, exist_ok=True)
    scheduler_json_path = f"{daemon_dir}/dask_scheduler_{workspace_hash}.json"
    daemon_log_path = f"{daemon_dir}/dask_daemon_{workspace_hash}.log"

    if args.scheduler_address:
        logging.info(f"Connecting to explicit Dask scheduler at {args.scheduler_address}...")
        client = distributed.Client(args.scheduler_address)
        return client, None

    def log_daemon_info(data):
        if "daemon_log" in data:
            logging.info(f"Dask daemon log: {data['daemon_log']}")
        if "worker_log_dir" in data:
            logging.info(f"Condor worker log directory: {data['worker_log_dir']}")

    if not args.start_cluster_daemon:
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

        client = None
        for attempt in range(1, 6):
            try:
                client = distributed.Client(address, timeout="5s")
                logging.info(f"Successfully connected to Dask scheduler (attempt {attempt}/5)!")
                logging.info(f"Dask dashboard: {client.dashboard_link}")
                logging.info(f"Dask scheduler host: {address.split('://')[1].split(':')[0]}")
                log_daemon_info(data)
                return client, None
            except Exception as e:
                if attempt == 5:
                    raise RuntimeError(f"Failed to connect to Dask scheduler at {address} after 5 attempts: {e}")
                logging.warning(f"Connection attempt {attempt}/5 failed: {e}. Retrying in 2 seconds...")
                time.sleep(2)

    else:
        logging.info("Initializing Dask cluster daemon...")
        log_dir = None
        if args.condor:
            logging.info("Configuring LPCCondorCluster daemon...")
            tarball_path, temp_dir = create_code_tarball(config_runner['condor_transfer_input_files'], tmpdir=args.tmpdir)
            client, cluster, log_dir = setup_condor_cluster(config_runner, tarball_path)
        elif args.slurm:
            logging.info("Configuring SLURMCluster daemon...")
            client, cluster = setup_slurm_cluster(config_runner)
        elif args.run_dask:
            logging.info("Configuring LocalCluster daemon...")
            client, cluster = setup_local_cluster(config_runner)
        else:
            raise ValueError("Daemon started without a valid cluster type flag (--condor, --slurm, or --dask)")

        info_data = {
            "address": client.scheduler.address,
            "pid": os.getpid(),
            "daemon_log": daemon_log_path,
        }
        if log_dir is not None:
            info_data["worker_log_dir"] = log_dir
        with open(scheduler_json_path, "w") as f:
            json.dump(info_data, f)

        logging.info("Registering worker plugin for Dask client in daemon...")
        worker_initializer = WorkerInitializer(uproot_xrootd_retry_delays=config_runner['uproot_xrootd_retry_delays'])
        client.register_plugin(worker_initializer)

        run_daemon_monitoring_loop(client, cluster, scheduler_json_path, args.idle_timeout)
        sys.exit(0)
