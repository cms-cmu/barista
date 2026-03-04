"""
Performance profiler for runner.py + processor_HH4b.py.

Instruments key stages of the processor pipeline with per-stage timing
and memory (RSS) snapshots. Outputs a human-readable report and a CSV
for easy before/after comparison.

Usage (inside container):
    python src/scripts/memory/perf_profile.py \\
        -o output/perf_profile \\
        --script runner.py -t -d GluGluToHHTo4B_cHHH1 \\
            -p coffea4bees/analysis/processors/processor_HH4b.py \\
            -y UL18 -op output/perf_test \\
            -m coffea4bees/metadata/datasets_HH4b_Run2/ \\
            -c coffea4bees/analysis/metadata/HH4b_signals.yml
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import textwrap


def main():
    parser = argparse.ArgumentParser(
        description="Profile runner.py with per-stage timing and memory snapshots.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("-o", "--output", default="perf_profile",
                        help="Output base path (no extension)")
    parser.add_argument("--script", nargs=argparse.REMAINDER, required=True,
                        help="runner.py and its arguments")
    args = parser.parse_args()

    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    script_file = args.script[0] if args.script else "runner.py"
    script_args = args.script[1:] if len(args.script) > 1 else []

    # We generate a small wrapper script that:
    # 1. Sets up sys.argv
    # 2. Installs the perf hook (patches processor on import)
    # 3. Executes runner.py
    wrapper_code = textwrap.dedent(f"""\
        import sys, os

        # Setup argv as if runner.py was called directly
        sys.argv = {[script_file] + script_args!r}

        # Set output path for the hook
        os.environ["PERF_PROFILE_OUTPUT"] = {args.output!r}

        # Install the hook before runner.py imports the processor
        from src.scripts.memory._perf_hook import install_hook
        install_hook()

        # Execute runner.py
        with open({script_file!r}) as _f:
            _code = _f.read()
        exec(compile(_code, {script_file!r}, "exec"))
    """)

    print(f"=== Performance Profiler ===")
    print(f"Output: {args.output}.{{txt,csv}}")
    print(f"Running: python {script_file} {' '.join(script_args)}")
    print()

    result = subprocess.run(
        [sys.executable, "-c", wrapper_code],
    )
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
