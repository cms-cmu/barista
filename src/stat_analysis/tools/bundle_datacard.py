#!/usr/bin/env python3
# coding: utf-8

"""
Standalone script to bundle a Combine datacard into a self-contained folder.
Copies the datacard .txt and all referenced ROOT shape files into the target directory,
updating the shape paths inside the datacard so the bundle is completely portable.
Zero dependencies outside the Python standard library.

Example usage:
    python3 src/stat_analysis/tools/bundle_datacard.py datacard.txt bundled_card/
    python3 src/stat_analysis/tools/bundle_datacard.py datacard.txt bundled_card/ --shapes-dir shapes
"""

import os
import sys
import shutil
import argparse


def bundle_datacard(datacard_path, target_dir, shapes_subdir=".", skip_shapes=False):
    src_card = os.path.abspath(datacard_path)
    if not os.path.isfile(src_card):
        raise FileNotFoundError(f"Datacard not found: {src_card}")

    card_dir = os.path.dirname(src_card)
    target_dir = os.path.abspath(target_dir)
    shapes_dir = os.path.join(target_dir, shapes_subdir)

    os.makedirs(target_dir, exist_ok=True)
    if not skip_shapes:
        os.makedirs(shapes_dir, exist_ok=True)

    dest_card = os.path.join(target_dir, os.path.basename(src_card))

    copied_shapes = {}

    with open(src_card, "r") as f_in, open(dest_card, "w") as f_out:
        for line in f_in:
            words = line.strip().split()
            if len(words) >= 4 and words[0] == "shapes":
                # shapes <process> <channel> <file> <pattern> [<syst_pattern>]
                shape_file = words[3]
                if shape_file != "FAKE":
                    abs_shape = os.path.normpath(os.path.join(card_dir, shape_file))
                    if abs_shape not in copied_shapes:
                        if not os.path.exists(abs_shape):
                            print(f"Warning: referenced shape file '{abs_shape}' does not exist!")
                            copied_shapes[abs_shape] = shape_file
                        else:
                            if skip_shapes:
                                copied_shapes[abs_shape] = abs_shape
                            else:
                                base_name = os.path.basename(abs_shape)
                                dest_shape = os.path.join(shapes_dir, base_name)
                                shutil.copy2(abs_shape, dest_shape)
                                rel_path = os.path.relpath(dest_shape, target_dir)
                                copied_shapes[abs_shape] = rel_path
                                print(f"Bundled shape file: {base_name} -> {rel_path}")

                    words[3] = copied_shapes[abs_shape]
                    line = "  ".join(words) + "\n"

            f_out.write(line)

    print(f"Successfully bundled datacard to: {dest_card}")
    return dest_card


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input", help="Path to input datacard .txt")
    parser.add_argument("output_dir", help="Directory where bundled datacard and shapes will be saved")
    parser.add_argument("-s", "--shapes-dir", default=".", help="Subdirectory inside output_dir for shapes (default: '.')")
    parser.add_argument("--no-shapes", action="store_true", help="Do not copy shapes (leave original paths)")

    args = parser.parse_args()
    bundle_datacard(args.input, args.output_dir, shapes_subdir=args.shapes_dir, skip_shapes=args.no_shapes)


if __name__ == "__main__":
    main()
