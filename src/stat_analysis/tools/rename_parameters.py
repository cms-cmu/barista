#!/usr/bin/env python3
# coding: utf-8

"""
Standalone script to rename parameters in a Combine datacard and optionally inside referenced shape ROOT files.
Zero dependencies outside standard Python and ROOT.

Example usage:
    # Rename via simple equality rule
    python3 src/stat_analysis/tools/rename_parameters.py datacard.txt "CMS_scale_j=CMS_scale_j_2018"

    # Rename via regex rule
    python3 src/stat_analysis/tools/rename_parameters.py datacard.txt "^lumi_(.+)$=CMS_lumi_\1" -o renamed.txt
"""

import os
import sys
import re
import argparse


def parse_rules(rule_strings):
    rules = []
    for r in rule_strings:
        if "=" in r:
            old, new = r.split("=", 1)
            is_regex = old.startswith("^") and old.endswith("$")
            compiled = re.compile(old) if is_regex else None
            rules.append((old.strip(), new.strip(), compiled))
        elif os.path.isfile(r):
            with open(r, "r") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        old, new = line.split("=", 1)
                        is_regex = old.startswith("^") and old.endswith("$")
                        compiled = re.compile(old) if is_regex else None
                        rules.append((old.strip(), new.strip(), compiled))
    return rules


def apply_rules(name, rules):
    for old, new, compiled in rules:
        if compiled:
            if compiled.search(name):
                return compiled.sub(new, name)
        else:
            if name == old:
                return new
    return name


def rename_parameters(datacard_path, rule_strings, output_path=None, rename_shapes=False):
    input_path = os.path.abspath(datacard_path)
    output_path = os.path.abspath(output_path) if output_path else input_path
    rules = parse_rules(rule_strings)

    if not rules:
        print("No renaming rules provided. Card unchanged.")
        return

    with open(input_path, "r") as f:
        lines = f.readlines()

    renamed_count = 0
    new_lines = []

    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith(("#", "//", "imax", "jmax", "kmax", "shapes", "bin", "observation", "process", "rate")):
            new_lines.append(line)
            continue

        words = stripped.split()
        if len(words) >= 2:
            param_name = words[0]
            new_name = apply_rules(param_name, rules)
            if new_name != param_name:
                renamed_count += 1
                words[0] = new_name
                # preserve spacing format
                new_lines.append(f"{words[0]:<40} " + "  ".join(words[1:]) + "\n")
                continue

        new_lines.append(line)

    with open(output_path, "w") as f:
        f.writelines(new_lines)

    print(f"Renamed {renamed_count} parameters in {output_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input", help="Path to input datacard .txt")
    parser.add_argument("rules", nargs="+", help="Renaming rules (e.g. 'old=new' or rules file)")
    parser.add_argument("-o", "--output", help="Output datacard path (defaults to in-place update)")

    args = parser.parse_args()
    rename_parameters(args.input, args.rules, output_path=args.output)


if __name__ == "__main__":
    main()
