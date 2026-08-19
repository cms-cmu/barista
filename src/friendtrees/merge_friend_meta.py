import json
from argparse import ArgumentParser

import fsspec
from src.data_formats.root import Friend
from src.storage.eos import EOS, PathLike
from src.utils.argparser import DefaultFormatter
from src.utils.json import DefaultEncoder


_FRIEND_KEYS = {"name", "branches", "data"}


def _iter_friend_dicts(obj):
    """Recursively yield any dict shaped like a serialized Friend.

    A Friend dict is one with the keys Friend.from_json consumes:
    'name', 'branches', and 'data'.
    """
    if isinstance(obj, dict):
        if _FRIEND_KEYS.issubset(obj.keys()):
            yield obj
            return
        for v in obj.values():
            yield from _iter_friend_dicts(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from _iter_friend_dicts(v)


def merge_friend_metas(output: PathLike, *metafiles: PathLike, cleanup: bool = True):
    with fsspec.open(metafiles[0]) as f:
        merged = json.load(f)
    anchors: dict[str, tuple[dict, Friend]] = {}
    for v in _iter_friend_dicts(merged):
        friend = Friend.from_json(v)
        if friend.name not in anchors:
            anchors[friend.name] = (v, friend)
    if not anchors:
        raise ValueError(f'no friends found in "{metafiles[0]}"')
    updated: set[str] = set()
    for metafile in metafiles[1:]:
        with fsspec.open(metafile) as f:
            meta = json.load(f)
        friends = [Friend.from_json(v) for v in _iter_friend_dicts(meta)]
        names = {friend.name for friend in friends}
        if names != anchors.keys():
            raise ValueError(
                f'friends in "{metafile}" {sorted(names)} do not match'
                f' those in "{metafiles[0]}" {sorted(anchors)}'
            )
        for friend in friends:
            anchor = anchors[friend.name][1]
            if anchor._branches != friend._branches:
                only_anchor = sorted(anchor._branches - friend._branches)
                only_new = sorted(friend._branches - anchor._branches)
                print(f'Branch mismatch for friend "{friend.name}":')
                if only_anchor:
                    print(f'  missing in "{metafile}": {only_anchor}')
                if only_new:
                    print(f'  missing in "{metafiles[0]}": {only_new}')
                try:
                    answer = ""
                    while answer not in ("yes", "no"):
                        answer = input(
                            "continue merge without this branch? type 'yes' or 'no': "
                        ).strip().lower()
                except EOFError:
                    answer = "no"
                if answer != "yes":
                    raise ValueError(
                        f'merge aborted: friend "{friend.name}" has different branches'
                    )
                common = anchor._branches & friend._branches
                anchor._branches = common
                friend._branches = common
            anchor += friend
            updated.add(friend.name)
    for name in updated:
        v, friend = anchors[name]
        v.clear()
        v.update(friend.to_json())
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
