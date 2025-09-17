import awkward as ak
import src.data_formats.awkward as akext
from src.data_formats.root import Chunk, Friend
from src.storage.eos import PathLike

_NAMING = "{path1}/{name}_{uuid}_{start}_{stop}_{path0}"

def _build_cutflow(*selections):
    if len(selections) == 1:
        return selections[0]
    selection = akext.to_numpy(selections[0], copy=True).astype(bool)
    for s in selections[1:]:
        selection[selection] = s
    return ak.Array(selection)


def dump_friend(
    events: ak.Array,
    output: PathLike,
    name: str,
    data: ak.Array,
    dump_naming: str = _NAMING,
):
    chunk = Chunk.from_coffea_events(events)
    friend = Friend(name)
    friend.add(chunk, data)
    friend.dump(output, dump_naming)
    return {name: friend}

