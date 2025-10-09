import json
from argparse import ArgumentParser

import fsspec
from src.data_formats.root import Friend
from src.storage.eos import EOS, PathLike
from src.utils.argparser import DefaultFormatter
from src.utils.json import DefaultEncoder


def merge_friend_metas(output: PathLike, *metafiles: PathLike, cleanup: bool = True):
    merged: dict[str, Friend] = {}
    for metafile in metafiles:
        with fsspec.open(metafile) as f:
            meta: dict[str, dict] = json.load(f)
        for k, v in meta.items():
            friend = Friend.from_json(v)
            if k in merged:
                merged[k] += friend
            else:
                merged[k] = friend
    output = EOS(output)
    tmp = output.local_temp(dir=".")
    try:
        with fsspec.open(tmp, "wt") as f:
            json.dump(merged, f, cls=DefaultEncoder)
        tmp.move_to(output, parents=True, overwrite=True)
    except Exception as e:
        if tmp.exists:
            tmp.rm()
        raise e
    if cleanup:
        for metafile in metafiles:
            metafile = EOS(metafile)
            if metafile != output:
                metafile.rm()


if __name__ == "__main__":
    argparser = ArgumentParser(formatter_class=DefaultFormatter)
    argparser.add_argument(
        "-i",
        "--input",
        nargs="+",
        required=True,
        help="input metafiles",
        action="extend",
        default=[],
    )
    argparser.add_argument(
        "-o",
        "--output",
        required=True,
        help="output metafile",
    )
    argparser.add_argument(
        "--cleanup",
        action="store_true",
        help="remove input metafiles after merging",
    )
    args = argparser.parse_args()
    merge_friend_metas(args.output, *args.input, cleanup=args.cleanup)
