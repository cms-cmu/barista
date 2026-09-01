#!/usr/bin/env python3

"""
Converts one or multiple pdf files to png using "pdftocairo".
"""

import os
import glob
import subprocess
from multiprocessing import Pool
from collections import deque
from typing import Sequence, Union, Tuple


this_dir = os.path.dirname(os.path.abspath(__file__))
repo_dir = os.path.dirname(this_dir)


def pdf_to_png(
    paths: Union[Sequence[str], str],
    recursive: bool = False,
    n_cores: int = 1,
) -> None:
    if isinstance(paths, str):
        paths = [paths]

    # find paths to all pdf files
    pdf_paths = []
    paths = deque(map(os.path.abspath, sum(map(glob.glob, paths), [])))
    seen = set()
    while paths:
        path = paths.popleft()

        # repetition guard
        if path in seen:
            continue
        seen.add(path)

        # handle files
        if os.path.isfile(path) and path.endswith(".pdf"):
            pdf_paths.append(path)

        # handle directories
        if os.path.isdir(path):
            for elem in os.listdir(path):
                elem = os.path.join(path, elem)
                if os.path.isfile(elem) or recursive:
                    paths.appendleft(elem)

    # convert files
    convert_pdfs(
        [(path, f"{os.path.splitext(path)[0]}.png") for path in pdf_paths],
        n_cores=n_cores,
    )


def _has_pdftocairo() -> bool:
    # check via "type"
    p = subprocess.run(
        "type pdftocairo",
        shell=True,
        executable="/bin/bash",
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    return p.returncode == 0


def convert_pdf(src: str, dst: str) -> None:
    # check if the source file exists
    if not os.path.exists(src):
        raise RuntimeError(f"source file {src} does not exist")
    if os.path.getsize(src) == 0:
        print(f"Warning: Skipping empty PDF file {src}")
        return

    # build the command
    src = os.path.abspath(src)
    dst = os.path.splitext(os.path.abspath(dst))[0]
    cmd = f"pdftocairo -singlefile -cropbox -png \"{src}\" \"{dst}\""

    # convert it
    p = subprocess.run(
        cmd,
        shell=True,
        executable="/bin/bash",
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    # handle response
    if p.returncode != 0:
        raise RuntimeError(f"command failed with exit code {p.returncode}: {p.stderr}")


def convert_pdf_mp(arg: Tuple[str, str]) -> None:
    try:
        convert_pdf(*arg)
    except Exception as e:
        print(f"Warning: Failed to convert {arg[0]}: {str(e)}")


def convert_pdfs(paths: Sequence[Tuple[str, str]], n_cores: int = 1) -> None:
    print(f"converting {len(paths)} pdf file(s) ...")

    if n_cores <= 1:
        # in-thread conversion
        for src, dst in paths:
            try:
                convert_pdf(src, dst)
            except Exception as e:
                print(f"Warning: Failed to convert {src}: {str(e)}")
    else:
        # multi-processing
        with Pool(n_cores) as pool:
            pool.map(convert_pdf_mp, paths)

    print("done")


def main() -> None:
    from argparse import ArgumentParser

    parser = ArgumentParser(description=__doc__)
    parser.add_argument(
        "paths",
        nargs="+",
        help="files to convert or directories to check for pdf files",
    )
    parser.add_argument(
        "--recursive",
        "-r",
        action="store_true",
        help="convert pdf files recursively in all subdirectories",
    )
    parser.add_argument(
        "--cores",
        "-j",
        type=int,
        default=1,
        help="number of cores to use for parallel conversion",
    )
    args = parser.parse_args()

    pdf_to_png(args.paths, recursive=args.recursive, n_cores=args.cores)


if __name__ == "__main__":
    main()

