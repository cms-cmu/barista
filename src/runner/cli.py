from __future__ import annotations
import argparse
import sys
import os
import yaml

def make_parser() -> argparse.ArgumentParser:
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

def parse_args() -> argparse.Namespace:
    parser = make_parser()
    
    # Check if we were passed a single configuration YAML file
    if len(sys.argv) == 2 and (sys.argv[1].endswith('.yml') or sys.argv[1].endswith('.yaml')) and os.path.exists(sys.argv[1]):
        with open(sys.argv[1], 'r') as f:
            yaml_config = yaml.safe_load(f)
            
        # Parse standard defaults first
        args = parser.parse_args([])
        
        # Merge YAML properties into namespace
        for key, val in yaml_config.items():
            setattr(args, key, val)
            
        # Store path to the job configuration
        args.job_yaml_path = sys.argv[1]
        return args
    else:
        args = parser.parse_args()
        args.job_yaml_path = None
        return args
