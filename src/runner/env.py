from __future__ import annotations
import os
import sys
import time
import subprocess
import logging

def setup_environment(args) -> None:
    # Cap NumExpr threads to avoid oversubscribing shared nodes
    numexpr_threads = os.environ.get("SLURM_CPUS_PER_TASK", "4")
    os.environ["NUMEXPR_MAX_THREADS"] = numexpr_threads
    logging.info(f"NUMEXPR_MAX_THREADS set to {numexpr_threads}")

    # Prevent pycache issues
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"

def print_reproducibility_info(args) -> None:
    # Mimics run-analysis-processor.sh output header
    logging.info(">>> Configuration")
    logging.info(f"Processor:          {args.processor}")
    logging.info(f"Datasets Metadata:  {args.metadata}")
    logging.info(f"Config:             {args.configs}")
    logging.info(f"Triggers:           {args.triggers}")
    logging.info(f"Luminosities:       {args.luminosities}")
    logging.info(f"Friends:            {args.friends}")
    logging.info(f"Weights:            {args.weights}")
    logging.info(f"Datasets:           {' '.join(args.datasets) if isinstance(args.datasets, list) else args.datasets}")
    logging.info(f"Year:               {' '.join(args.years) if isinstance(args.years, list) else args.years}")
    logging.info(f"Output filename:    {os.path.join(args.output_path, args.output_file)}")
    logging.info(f"Test mode:          {'enabled' if getattr(args, 'test', False) else 'disabled'}")
    logging.info(f"Condor mode:        {'enabled' if getattr(args, 'condor', False) else 'disabled'}")
    logging.info(f"Blind mode:         {'enabled' if getattr(args, 'blind', False) or getattr(args, 'job_yaml_path', None) else 'disabled'}")
    logging.info(f"Log file:           (none)")
    logging.info(f"Dashboard address:  {args.dashboard_address if args.dashboard_address else '(default: 10200)'}")
    logging.info(f"Additional flags:   (none)")

def check_and_setup_proxy(args) -> None:
    # Match setup_proxy in src/scripts/common.sh
    logging.info(">>> Setting up proxy")
    logging.info(">>> Including proxy")
    
    proxy_path = os.path.join(os.getcwd(), "proxy/x509_proxy")
    if not os.path.exists(proxy_path):
        logging.error("Error: x509_proxy file not found!")
        logging.error("Run manually:")
        logging.error("mkdir -p proxy && voms-proxy-init -voms cms -valid 192:00 -out ./proxy/x509_proxy")
        logging.error("and try again.")
        sys.exit(1)
        
    os.environ["X509_USER_PROXY"] = proxy_path
    logging.info(">>> Checking proxy")
    
    # Run voms-proxy-info to display info and check if the proxy is valid and not expired
    try:
        subprocess.run(["voms-proxy-info"], check=True)
        subprocess.run(["voms-proxy-info", "-exists", "-valid", "0:10"], check=True)
    except subprocess.CalledProcessError as e:
        logging.error(f"Error checking proxy or proxy is expired/invalid: {e}")
        logging.error("Please renew your proxy by running manually:")
        logging.error("voms-proxy-init -rfc -voms cms --valid 168:00 -out ./proxy/x509_proxy")
        sys.exit(1)

def sync_nfs_writes() -> None:
    # Mimic flush NFS writes and sleep to sync files
    try:
        os.sync()
    except AttributeError:
        # os.sync is only available on Unix
        pass
    time.sleep(10)
