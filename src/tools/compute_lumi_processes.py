#!/usr/bin/env python3
"""
Convert YAML lumi file to JSON text file and run brilcalc.

Usage: python convert_yaml_to_json_lumi.py -i <input.yml> [-o <output.txt>]
"""

import yaml
import json
import sys
import subprocess
import argparse
import tempfile
import os
from pathlib import Path


def group_lumis_into_ranges(lumi_list):
    """Convert a list of lumi sections into [start, end] ranges."""
    if not lumi_list:
        return []
    
    sorted_lumis = sorted(lumi_list)
    ranges = []
    start = end = sorted_lumis[0]
    
    for lumi in sorted_lumis[1:]:
        if lumi == end + 1:
            end = lumi
        else:
            ranges.append([start, end])
            start = end = lumi
    
    ranges.append([start, end])
    return ranges


def convert_yaml_to_json(yaml_file):
    """Convert YAML lumi file to JSON format."""
    print(f"Reading: {yaml_file}")
    
    with open(yaml_file, 'r') as f:
        data = yaml.unsafe_load(f)
    
    dataset_name = list(data.keys())[0]
    lumis_processed = data[dataset_name].get('lumis_processed', {})
    
    print(f"Dataset: {dataset_name}, Runs: {len(lumis_processed)}")
    
    # Convert to JSON format: {"run": [[start, end], ...]}
    json_output = {str(run): group_lumis_into_ranges(lumis) 
                   for run, lumis in lumis_processed.items()}
    
    return json_output


def run_brilcalc(json_data, output_file=None):
    """Run brilcalc command with the JSON data and optionally save output."""
    # Create temporary JSON file
    temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False)
    json.dump(json_data, temp_file, indent=2)
    temp_file.close()
    
    cmd = [
        "brilcalc", "lumi",
        "-c", "web",
        "-b", "STABLE BEAMS",
        "-u", "/fb",
        "-i", temp_file.name
    ]
    
    print(f"\nRunning: {' '.join(cmd)}\n")
    
    try:
        if output_file:
            # Save brilcalc output to file
            with open(output_file, 'w') as f:
                result = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, text=True)
            print(f"\nBrilcalc output saved to: {output_file}")
        else:
            # Print to console
            subprocess.run(cmd)
    finally:
        # Clean up temporary JSON file
        try:
            os.unlink(temp_file.name)
        except:
            pass


def main():
    parser = argparse.ArgumentParser(description='Convert YAML to JSON and run brilcalc')
    parser.add_argument('-i', '--input', required=True, help='Input YAML file')
    parser.add_argument('-o', '--output', help='Save brilcalc output to file (optional)')
    
    args = parser.parse_args()
    
    # Check input exists
    if not Path(args.input).exists():
        print(f"Error: '{args.input}' not found!")
        sys.exit(1)
    
    # Convert YAML to JSON data
    json_data = convert_yaml_to_json(args.input)
    
    # Run brilcalc and optionally save output
    run_brilcalc(json_data, args.output)


if __name__ == "__main__":
    main()
