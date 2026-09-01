import subprocess
import sys
import argparse
import os

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                        description='Check memory usage with mprof.')
    parser.add_argument('--threshold', type=float, default=0, help='Memory threshold in MB')
    parser.add_argument('--tolerance', type=float, default=10, help='Tolerance percentage')
    parser.add_argument('-o', '--output', type=str, default="mprofile_test", help='Name of the outputs')
    parser.add_argument('-s', '--script', nargs=argparse.REMAINDER, required=True, help='Script to run with mprof')
    args = parser.parse_args()
    print(args)

    THRESHOLD_MB = args.threshold
    TOLERANCE_PERCENT = args.tolerance

    # Extract the directory part of the output path
    output_dir = os.path.dirname(args.output)

    # Check if the output directory exists, if not create it
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    # Run the application with mprof
    result = subprocess.run(['mprof', 'run', '-C', '-o', f'{args.output}.dat', 'python'] + args.script)
    if result.returncode != 0:
        print("Error running mprof:", result.stderr)
        sys.exit(1)

    # Generate the memory usage plot
    result = subprocess.run(['mprof', 'plot', '-o', f'{args.output}.png', f'{args.output}.dat'], capture_output=True, text=True)
    if result.returncode != 0:
        print("Error generating memory usage plot:", result.stderr)
        sys.exit(1)
    print("Memory usage plot generated successfully.")

    # Get the peak memory usage
    result = subprocess.run(['mprof', 'peak', f'{args.output}.dat'], capture_output=True, text=True)
    if result.returncode != 0:
        print("Error getting peak memory usage:", result.stderr)
        sys.exit(1)

    # Parse the peak memory usage
    for line in result.stdout.splitlines():
        print(line)
        if args.output in line:
            peak_memory = float(line.split()[-2])
            tolerance = THRESHOLD_MB * (TOLERANCE_PERCENT / 100)
            lower_bound = THRESHOLD_MB - tolerance
            upper_bound = THRESHOLD_MB + tolerance

            if THRESHOLD_MB == 0:
                print(f"Peak memory usage: {peak_memory} MB")
            else:
                if lower_bound <= peak_memory <= upper_bound:
                    print(f"Memory usage is within the tolerance range: {peak_memory} MB (threshold: {THRESHOLD_MB} MB ± {TOLERANCE_PERCENT}%)")
                else:
                    print(f"Memory usage is outside the tolerance range: {peak_memory} MB (threshold: {THRESHOLD_MB} MB ± {TOLERANCE_PERCENT}%)")
                    sys.exit(1)

if __name__ == '__main__':
    main()