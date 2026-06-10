#!/usr/bin/env python3
import sys
import os
import glob
import subprocess

def check_log_status(jobid):
    log_files = glob.glob(f"condor_logs/job_*_{jobid}.log")
    if not log_files:
        return None
    
    log_path = log_files[0]
    if not os.path.exists(log_path):
        return None
        
    try:
        with open(log_path, "r") as f:
            content = f.read()
            
        events = content.split("...\n")
        status = "running"
        for event in events:
            event = event.strip()
            if not event:
                continue
            code = event[:3]
            if code == "005":
                status = "success"
            elif code == "009":
                status = "failed"
            elif code == "012":
                # Job is held. Since LPC held jobs usually need manual intervention,
                # we can check if it gets released later.
                status = "failed"
            elif code == "013":
                # Job was released
                status = "running"
        return status
    except Exception as e:
        print(f"Error reading log file {log_path}: {e}", file=sys.stderr)
        return None

def main():
    if len(sys.argv) < 2:
        print("Usage: status_wrapper.py <jobid>", file=sys.stderr)
        sys.exit(1)
        
    jobid = sys.argv[-1]
    
    if jobid.startswith("local_job_"):
        print("success")
        sys.exit(0)
    
    # Try parsing the log file first
    status = check_log_status(jobid)
    if status is not None:
        print(status)
        sys.exit(0)
        
    # Fallback to condor_q / condor_history using -global to query all schedds
    try:
        # Check active queue
        out = subprocess.check_output(f"condor_q -global {jobid} -format '%s\n' JobStatus 2>/dev/null", shell=True).decode().strip()
        if not out:
            # Check history (may fail if schedd is not local, but we try)
            out_hist = subprocess.check_output(f"condor_history {jobid} -limit 1 -format '%s\n' JobStatus 2>/dev/null", shell=True).decode().strip()
            if out_hist == "4":
                print("success")
            elif out_hist in ["3", "6"]:
                print("failed")
            else:
                print("success")
        else:
            status = out.split()[0]
            if status in ["1", "2"]:
                print("running")
            elif status == "4":
                print("success")
            else:
                print("failed")
    except Exception:
        # Fallback to success to let Snakemake verify output files
        print("success")

if __name__ == "__main__":
    main()
