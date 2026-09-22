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
import shutil
from rich.pretty import pretty_repr

import dask
from dask.distributed import SchedulerPlugin
import dask.distributed
import distributed

def get_default_scratch():
    user = getpass.getuser()
    candidate = f"/uscmst1b_scratch/lpc1/3DayLifetime/{user}"
    if os.path.exists(candidate) and os.access(candidate, os.W_OK):
        return candidate
    fallback = f"/tmp/{user}/barista_scratch"
    os.makedirs(fallback, exist_ok=True)
    return fallback

# ---------------------------------------------------------------------------
# HTCondor site handling: 'lpc' (FNAL, lpcjobqueue) or 'lxplus' (CERN, dask_lxplus)
# ---------------------------------------------------------------------------

LXPLUS_DEFAULT_WORKER_IMAGE = "/cvmfs/unpacked.cern.ch/gitlab-registry.cern.ch/cms-cmu/barista:latest"
TARBALL_EXCLUDES = {".git", "__pycache__", ".pixi", ".pytest_cache", ".lxplus_site"}


def detect_condor_site(config_runner=None):
    """Return the HTCondor site used by --condor: 'lpc' or 'lxplus'.

    Precedence: the runner config key `condor_site` (also set by --condor-site), the BARISTA_SITE
    environment variable exported by run_container, then the hostname. Defaults to 'lpc' so the
    historical meaning of --condor is preserved.
    """
    site = (config_runner or {}).get('condor_site')
    if site:
        return str(site).lower()
    env_site = os.environ.get('BARISTA_SITE', '').lower()
    if env_site.startswith('lxplus'):
        return 'lxplus'
    if env_site.startswith('lpc'):
        return 'lpc'
    if 'lxplus' in socket.gethostname():
        return 'lxplus'
    return 'lpc'


def eos_xrootd_url(path):
    """Map an EOS FUSE path to its xrootd URL, or None if `path` is not on a known EOS instance."""
    if path.startswith('/eos/user/') or path.startswith('/eos/home-'):
        return 'root://eosuser.cern.ch/' + path
    if path.startswith('/eos/cms/'):
        return 'root://eoscms.cern.ch/' + path
    return None


def _tarball_filter(tarinfo):
    """Drop VCS, bytecode and local-environment directories from the code tarball."""
    if any(part in TARBALL_EXCLUDES for part in tarinfo.name.split('/')):
        return None
    return tarinfo

def create_code_tarball(condor_transfer_input_files, tmpdir=None):
    """Create a tarball of code in a temporary directory.

    Each job gets a unique temporary directory to avoid conflicts
    between concurrent jobs on shared clusters.
    """
    import tarfile
    logging.info("Creating code tarball for HTCondor transfer...")
    
    # Use specified tmpdir or default to scratch directory
    if tmpdir is None:
        scratch = get_default_scratch()
        tmpdir = f"{scratch}/condor_tmp"
        
    temp_dir = os.path.join(tmpdir, f"barista_{uuid.uuid4().hex[:8]}")
    os.makedirs(temp_dir, exist_ok=True)
    tarball_path = os.path.join(temp_dir, "code_barista.tar.gz")

    with tarfile.open(tarball_path, "w:gz") as tar:
        for path in condor_transfer_input_files:
            if os.path.exists(path):
                logging.info(f"  Adding {path} to tarball...")
                tar.add(path, filter=_tarball_filter)
            else:
                logging.warning(f"  Warning: path {path} not found, skipping...")
                
    logging.info(f"Code tarball created successfully at {tarball_path}")
    return tarball_path, temp_dir

def register_worker_lost_logger(client, log_dir):
    """Log permanently failed tasks together with the worker log that mentions the worker address."""

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


def setup_condor_cluster(config_runner, tarball_path, proxy_path=None, site=None):
    """Create the HTCondor-backed Dask cluster for the detected (or given) site.

    Returns (client, cluster, log_dir). 'lpc' uses lpcjobqueue (FNAL), 'lxplus' uses dask_lxplus (CERN).
    """
    site = site or detect_condor_site(config_runner)
    if site == 'lxplus':
        return setup_lxplus_condor_cluster(config_runner, tarball_path, proxy_path)
    if site == 'lpc':
        return setup_lpc_condor_cluster(config_runner, tarball_path)
    raise ValueError(f"Unknown condor_site '{site}' (expected 'lpc' or 'lxplus')")


def setup_lpc_condor_cluster(config_runner, tarball_path):
    """Setup Dask LPCCondorCluster (FNAL LPC, lpcjobqueue)."""
    from lpcjobqueue import LPCCondorCluster

    logging.info("Initializing LPC HTCondor cluster configuration...")

    scratch = get_default_scratch()
    _log_base = f'{scratch}/condor_logs'
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
            "--death-timeout 300",
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
    register_worker_lost_logger(client, log_dir)

    logging.info('LPC HTCondor cluster setup complete!')
    return client, cluster, log_dir


def _lxplus_worker_image(config_runner):
    image = (config_runner.get('lxplus_worker_image')
             or os.environ.get('WORKER_IMAGE')
             or os.environ.get('COFFEA_IMAGE_FULL')
             or LXPLUS_DEFAULT_WORKER_IMAGE)
    if not image.startswith('/cvmfs/'):
        logging.warning(f"Worker image '{image}' is not a /cvmfs/unpacked.cern.ch path; "
                        "CERN HTCondor expects unpacked images for MY.SingularityImage.")
    return image


def _lxplus_eos_scratch(config_runner):
    """Return the writable EOS scratch directory for tarball staging and worker logs, or None."""
    scratch = config_runner.get('lxplus_eos_scratch')
    if not scratch:
        return None
    scratch = scratch.format(user=getpass.getuser())
    try:
        os.makedirs(scratch, exist_ok=True)
        if not os.access(scratch, os.W_OK):
            raise OSError("directory is not writable")
    except OSError as e:
        logging.warning(f"EOS scratch {scratch} is not usable ({e}). Falling back to /tmp: the code tarball "
                        "is spooled with every worker job and worker stdout/stderr will not be retrievable.")
        return None
    return scratch


def setup_lxplus_condor_cluster(config_runner, tarball_path, proxy_path=None):
    """Setup Dask workers on CERN HTCondor from lxplus with dask_lxplus.CernCluster.

    dask_lxplus submits with `condor_submit -spool` (the CERN schedd is a remote machine), runs the
    workers inside the analysis image via MY.SingularityImage and gives every job the user's Kerberos
    credential (MY.SendCredential). Because of the spooling, the code tarball is staged once on EOS and
    fetched by URL, and worker stdout/stderr are delivered to EOS through `output_destination`.
    """
    try:
        from dask_lxplus import CernCluster
    except ImportError as e:
        raise ImportError(
            "dask_lxplus is not installed in this container. Use an analysis image built from the current "
            "software/dockerfiles/Dockerfile_analysis (it lists dask-lxplus) or, as an interim measure on "
            "lxplus, run `./run_container lxplus-setup` to install it into .lxplus_site/."
        ) from e
    from src.runner.orchestrator import find_free_port

    logging.info("Initializing lxplus HTCondor cluster configuration (dask_lxplus)...")
    image = _lxplus_worker_image(config_runner)
    eos_scratch = _lxplus_eos_scratch(config_runner)
    cleanup_paths = []

    # Code tarball: stage once on EOS and hand the workers an xrootd URL, so the starter fetches it
    # instead of -spool copying it onto the schedd for every worker job.
    tarball_ref = tarball_path
    if eos_scratch and not os.path.abspath(tarball_path).startswith(eos_scratch):
        stage_dir = os.path.join(eos_scratch, 'condor_tmp', f"barista_{uuid.uuid4().hex[:8]}")
        os.makedirs(stage_dir, exist_ok=True)
        staged = os.path.join(stage_dir, os.path.basename(tarball_path))
        shutil.copyfile(tarball_path, staged)
        cleanup_paths.append(stage_dir)
        tarball_ref = eos_xrootd_url(staged) or staged
        logging.info(f"Code tarball staged on EOS: {tarball_ref}")
    elif tarball_path.startswith('/eos/'):
        tarball_ref = eos_xrootd_url(tarball_path) or tarball_path

    # Worker logs: an EOS directory (delivered by the xrootd transfer plugin) unless overridden.
    if config_runner.get('worker_log_directory'):
        log_dir = config_runner['worker_log_directory']
    elif eos_scratch:
        log_dir = os.path.join(eos_scratch, 'condor_logs', f"dask_{uuid.uuid4().hex[:8]}")
    else:
        log_dir = f"{get_default_scratch()}/condor_logs_{uuid.uuid4().hex[:8]}"
    os.makedirs(log_dir, exist_ok=True)

    job_extra = {
        '+JobFlavour': f'"{config_runner["lxplus_job_flavour"]}"',
        'transfer_input_files': tarball_ref,
        'should_transfer_files': 'YES',
        'when_to_transfer_output': 'ON_EXIT',
        'transfer_executable': 'False',
        'leave_in_queue': 'False',
        'periodic_remove': '(JobStatus == 5 && (CurrentTime - EnteredCurrentStatus) > 300)',
    }
    if config_runner.get('lxplus_send_credential', True):
        job_extra['MY.SendCredential'] = 'True'
    log_url = eos_xrootd_url(log_dir) if log_dir.startswith('/eos/') else None
    if log_url:
        job_extra.update({
            'output_destination': log_url.rstrip('/') + '/',
            'Output': 'worker-$(ClusterId).$(ProcId).out',
            'Error': 'worker-$(ClusterId).$(ProcId).err',
            'MY.XRDCP_CREATE_DIR': 'True',
        })
    else:
        logging.warning(f"Worker log directory {log_dir} is not on EOS: with spooled submission the worker "
                        "stdout/stderr stay in the schedd spool (see condor_transfer_data).")
    if proxy_path and os.path.exists(proxy_path):
        job_extra['x509userproxy'] = os.path.abspath(proxy_path)
    else:
        logging.warning("No X509 proxy passed to the workers (remote xrootd reads will fail without one).")

    port = find_free_port(int(config_runner['lxplus_scheduler_port']))
    cluster_args = {
        'cores': int(config_runner['condor_cores']),
        'processes': 1,
        'memory': config_runner['worker_memory'],
        'disk': config_runner['lxplus_disk_per_worker'],
        'worker_image': image,
        'container_runtime': 'singularity',
        'batch_name': config_runner['lxplus_batch_name'],
        'death_timeout': config_runner['lxplus_death_timeout'],
        # Spill to the HTCondor scratch directory (RequestDisk) instead of the container's /tmp.
        'local_directory': '${_CONDOR_SCRATCH_DIR:-.}',
        'log_directory': log_dir,
        'scheduler_options': {
            'host': socket.gethostname(),
            'port': port,
            'dashboard_address': f":{config_runner['dashboard_address']}",
        },
        'job_extra_directives': job_extra,
        # Joined into the `/bin/sh -c` worker command: shell expansion works, condor $(macros) do not.
        'job_script_prologue': [
            'export PYTHONPATH=$PWD:$PYTHONPATH',
            'export XRD_RUNFORKHANDLER=1',
            'export MALLOC_TRIM_THRESHOLD_=0',
            'export X509_USER_PROXY=${X509_USER_PROXY:-$PWD/x509_proxy}',
        ],
        'worker_extra_args': [],  # CernCluster appends --worker-port 10000:10100 itself
    }

    logging.info("Cluster arguments: ")
    logging.info(pretty_repr(cluster_args))

    logging.info("Creating CernCluster (HTCondor @ CERN)...")
    cluster = CernCluster(**cluster_args)
    cluster.barista_cleanup_paths = cleanup_paths

    logging.info("Creating Dask client...")
    client = dask.distributed.Client(cluster)

    logging.info(f"Setting up adaptive scaling (min: {config_runner['min_workers']}, max: {config_runner['max_workers']})")
    cluster.adapt(minimum=config_runner['min_workers'], maximum=config_runner['max_workers'])
    logging.info(f"Dask dashboard: {client.dashboard_link}")
    logging.info(f"Dask scheduler: {socket.gethostname()}:{port}")
    logging.info(f"Condor worker log directory: {log_dir}")
    register_worker_lost_logger(client, log_dir)

    logging.info('lxplus HTCondor cluster setup complete!')
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

def cleanup_cluster_paths(cluster):
    """Remove staging directories (code tarball copies) recorded on the cluster object."""
    for path in getattr(cluster, 'barista_cleanup_paths', None) or []:
        try:
            shutil.rmtree(path, ignore_errors=True)
            logging.info(f"Removed staging directory: {path}")
        except Exception as e:
            logging.warning(f"Could not remove staging directory {path}: {e}")


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
            time.sleep(10)
            continue

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
    cleanup_cluster_paths(cluster)
    logging.info("Daemon shutdown complete. Exiting.")

def setup_shared_dask_client(args, config_runner, WorkerInitializer=None):
    """Check for/connect to an existing cluster daemon, or spawn one if needed."""
    # We import internally to keep imports clean
    global _temp_condor_dir
    
    workspace_path = os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
    workspace_hash = os.environ.get("BARISTA_WORKSPACE_HASH")
    if not workspace_hash:
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
        while time.time() - start_wait < 120:
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
        for attempt in range(1, 11):
            try:
                client = distributed.Client(address, timeout="30s")
                logging.info(f"Successfully connected to Dask scheduler (attempt {attempt}/10)!")
                logging.info(f"Dask dashboard: {client.dashboard_link}")
                logging.info(f"Dask scheduler host: {address.split('://')[1].split(':')[0]}")
                log_daemon_info(data)
                return client, None
            except Exception as e:
                if attempt == 10:
                    raise RuntimeError(f"Failed to connect to Dask scheduler at {address} after 10 attempts: {e}")
                logging.warning(f"Connection attempt {attempt}/10 failed: {e}. Retrying in 3 seconds...")
                time.sleep(3)

    else:
        logging.info("Initializing Dask cluster daemon...")
        log_dir = None
        if args.condor:
            site = detect_condor_site(config_runner)
            logging.info(f"Configuring HTCondor Dask cluster daemon (site: {site})...")
            tarball_path, temp_dir = create_code_tarball(config_runner['condor_transfer_input_files'], tmpdir=args.tmpdir)
            client, cluster, log_dir = setup_condor_cluster(
                config_runner, tarball_path, proxy_path=os.environ.get('X509_USER_PROXY'), site=site)
            cluster.barista_cleanup_paths = list(getattr(cluster, 'barista_cleanup_paths', [])) + [temp_dir]
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

        idle_timeout = getattr(args, 'idle_timeout', None) or config_runner.get('idle_timeout', 3600)
        run_daemon_monitoring_loop(client, cluster, scheduler_json_path, idle_timeout)
        sys.exit(0)
