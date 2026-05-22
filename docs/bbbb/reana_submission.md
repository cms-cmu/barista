# REANA Submission Guide & Example

This directory contains a simple example workflow for running jobs on the REANA cluster, as well as instructions on how to validate, submit, monitor, and retrieve results for workflows.

---

## Example Workflow Files

1. **[Snakefile_reana_simple.smk](../../src/workflows/Snakefile_reana_simple.smk)**: A minimal Snakemake workflow that creates an output directory, prints a hello message, and logs the hostname and timestamp of the execution node.
2. **[reana_simple.yaml](../../src/workflows/reana_simple.yaml)**: A REANA specification file that configures the inputs, outputs, and computing resources (setting the compute backend to `slurmcern`).

---

## 1. Environment & Credentials Setup

To communicate with the REANA server, you need to configure your connection environment variables. In our CMU group, the REANA server is hosted on the Falcon cluster.

Run the following commands to export the required configuration variables:

```bash
export REANA_SERVER_URL=https://falcon.phys.cmu.edu
export REANA_ACCESS_TOKEN=<your_access_token>
```

> [!TIP]
> The orchestrator has a wrapper utility called `./run_container` which handles the REANA environment setup via Pixi automatically.

---

## 2. REANA Command Reference

You can run `reana-client` commands directly by prefixing them with `./run_container reana`:

### A. Validate Configuration
Check if your `reana.yaml` specification and paths are valid before submitting:
```bash
./run_container reana validate -f src/workflows/reana_simple.yaml
```

### B. Submit and Run Workflow (All-in-One)
The `run` command is a shortcut that automatically performs `create`, `upload`, and `start`:
```bash
./run_container reana run -f src/workflows/reana_simple.yaml -w simple-test
```
*Note: The `-w simple-test` flag assigns a custom name to this run.*

### C. Step-by-Step Submission (Alternative)
If you prefer to submit, upload files, and start manually:
```bash
# 1. Create the workflow structure on the server
./run_container reana create -f src/workflows/reana_simple.yaml -w simple-test

# 2. Upload input files to the server workspace
./run_container reana upload -w simple-test

# 3. Start the workflow run
./run_container reana start -w simple-test
```

### D. Monitor Execution Status
Check the status of your running workflow:
```bash
./run_container reana status -w simple-test
```
You can also view the live execution logs of the scheduler or specific tasks:
```bash
./run_container reana logs -w simple-test
```

### E. List All Workflows
To see all your workflows and their statuses:
```bash
./run_container reana list
```

### F. Download Outputs
Once the workflow status reaches `finished`, retrieve the outputs from the remote workspace to your local directory:
```bash
./run_container reana download reana_output -w simple-test
```

### G. Clean Up Workflows
To delete a workflow and clean up its space:
```bash
./run_container reana delete -w simple-test --destroy
```
