import argparse
import json
import logging
import math

import fsspec
from classifier.task import ArgParser, Dataset, EntryPoint, parse

from ..setting import ResultKey


class cache(Dataset):
    argparser = ArgParser(description="Load the datasets saved by [blue]cache[/blue].")
    argparser.add_argument(
        "--input",
        default=argparse.SUPPRESS,
        required=True,
        help="a json file containing the metadata of the cached dataset",
    )
    argparser.add_argument(
        "--chunk",
        action="extend",
        nargs="+",
        default=[],
        help="if given, only load the selected chunks e.g. [yellow]--chunks 0-3 5[/yellow]",
    )

    def train(self):
        from src.storage.eos import EOS

        metafile = EOS(self.opts.input)
        base = metafile.parent
        with fsspec.open(metafile) as f:
            metadata = json.load(f)[ResultKey.cache]
        # load states
        states = metadata["states"]
        if states:
            logging.info(f"The following states will be loaded {sorted(states)}")
            for state in states:
                mod, var = state.rsplit(".", 1)
                mod = EntryPoint._fetch_module(mod, "state", True)[1]
                setattr(mod, var, states[state])
        # load datasets
        total = math.ceil(metadata["size"] / metadata["chunksize"])
        if self.opts.chunk:
            chunks = parse.intervals(self.opts.chunk, total)
        else:
            chunks = list(range(total))
        if len(chunks) == 0:
            logging.warning("No chunk to load")
        else:
            count = len(chunks) * metadata["chunksize"]
            if chunks[-1] == total - 1:
                count -= total * metadata["chunksize"] - metadata["size"]
            logging.info(
                f'Loading {count}/{metadata["size"]} entries from {len(chunks)}/{total} cached chunks (shuffle={metadata["shuffle"]}, compression={metadata["compression"]})'
            )
        return [
            _load_cache(str(base / f"chunk{i}.pt"), metadata["compression"])
            for i in chunks
        ]


class _load_cache:
    def __init__(self, path: str, compression: str):
        self.path = path
        self.compression = compression

    def __call__(self):
        import torch

        with fsspec.open(self.path, "rb", compression=self.compression) as f:
            return torch.load(f)
