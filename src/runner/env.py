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
    print("############### Configuration")
    print(f"Processor:          {args.processor}")
    print(f"Datasets Metadata:  {args.metadata}")
    print(f"Config:             {args.configs}")
    print(f"Triggers:           {args.triggers}")
    print(f"Luminosities:       {args.luminosities}")
    print(f"Friends:            {args.friends}")
    print(f"Weights:            {args.weights}")
    print(f"Datasets:           {' '.join(args.datasets) if isinstance(args.datasets, list) else args.datasets}")
    print(f"Year:               {' '.join(args.years) if isinstance(args.years, list) else args.years}")
    print(f"Output filename:    {os.path.join(args.output_path, args.output_file)}")
    print(f"Test mode:          {'enabled' if getattr(args, 'test', False) else 'disabled'}")
    print(f"Condor mode:        {'enabled' if getattr(args, 'condor', False) else 'disabled'}")
    print(f"Blind mode:         {'enabled' if getattr(args, 'blind', False) or getattr(args, 'job_yaml_path', None) else 'disabled'}")
    print(f"Log file:           (none)")
    print(f"Dashboard address:  {args.dashboard_address if args.dashboard_address else '(default: 10200)'}")
    print(f"Additional flags:   (none)")
    print("")

def check_and_setup_proxy(args) -> None:
    # Match setup_proxy in src/scripts/common.sh
    print("############### Setting up proxy")
    print("############### Including proxy")
    
    proxy_path = os.path.join(os.getcwd(), "proxy/x509_proxy")
    if not os.path.exists(proxy_path):
        print("Error: x509_proxy file not found!")
        print("Run manually:")
        print("mkdir -p proxy && voms-proxy-init -voms cms -valid 192:00 -out ./proxy/x509_proxy")
        print("and try again.")
        sys.exit(1)
        
    os.environ["X509_USER_PROXY"] = proxy_path
    print("############### Checking proxy")
    
    # Run voms-proxy-info
    try:
        subprocess.run(["voms-proxy-info"], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error checking proxy: {e}")
        sys.exit(1)

def sync_nfs_writes() -> None:
    # Mimic flush NFS writes and sleep to sync files
    try:
        os.sync()
    except AttributeError:
        # os.sync is only available on Unix
        pass
    time.sleep(10)
