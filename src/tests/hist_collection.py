import awkward as ak
import numpy as np
from hist import Hist

from ..hist_tools import Collection


def projection_test(hist: Hist, **axes: list[float]):
    for axis, values in axes.items():
        assert np.all(hist.project(axis).to_numpy()[0] == values)


def nested_hists():
    collection = Collection()
    fill = collection.add("all", ([1], "flat"), ([10], "nested1"), ([100], "nested2"))
    data = ak.zip(
        {
            "flat": [1, 2],
            "nested1": [[10, 20], [30]],
            "nested2": [[[100, 200, 300], [400]], [[500, 600]]],
        },
        depth_limit=1,
    )
    fill(data, weight=1.0)
    return collection.to_dict(nonempty=True)


if __name__ == "__main__":
    nested = nested_hists()
    projection_test(
        nested["hists"]["all"],
        flat=[4, 2],
        nested1=[3, 1, 2],
        nested2=[1, 1, 1, 1, 1, 1],
    )
