import logging
import warnings

import awkward as ak
import numpy as np
from src.storage.eos import PathLike

from src.skimmer.picoaod import PicoAOD, _branch_filter
from coffea4bees.analysis.helpers.cutflow import cutflow_4b

warnings.filterwarnings(
    "ignore", category=RuntimeWarning, message="Missing cross-reference .*"
)


class TestSkimmer(PicoAOD):
    def __init__(self, base_path: PathLike, step: int, **_):
        super().__init__(
            base_path,
            step,
            skip_branches=(
                _branch_filter(("Jet",), ("event",)),
            ),  # hacked, skip skip == keep
        )
        self._cutFlow = cutflow_4b()

    def select(self, events: ak.Array):
        selection = events.event % 11 == 0
        events = events[selection]
        n_jet = ak.num(events.Jet)
        total_jet = int(ak.sum(n_jet))
        branches = ak.Array(
            {
                # replace branch by using the same name as nanoAOD
                "Jet_pt": events.event % 53 + ak.local_index(events.Jet.pt),
                # replace another branch
                "Jet_phi": ak.unflatten(np.zeros(total_jet), n_jet),
                # create new branch
                "Jet_isGood": ak.unflatten(
                    np.repeat(events.event % 2 == 0, n_jet), n_jet
                ),
                # create new regular branch
                "isBad": events.event % 2 == 1,
            }
        )
        result = {"total_jet": total_jet}

        return (
            selection,  # used to filter events
            branches,  # added to output root file
            result,  # added to the returned dict
        )
        # all of the following are valid return values
        # return selection
        # return selection, branches
        # return selection, None, result


class TestAnalysis(TestSkimmer):
    def __init__(self, **_): ...

    def process(self, events):
        selection, branches, result = self.select(events)

        assert ak.all(selection)
        assert ak.all(events.Jet.pt == branches.Jet_pt)
        assert ak.all(events.Jet.phi == branches.Jet_phi)
        assert ak.all(events.Jet.isGood == branches.Jet_isGood)
        assert ak.all(events.isBad == branches.isBad)
        # assert events.metadata["total_jet"] == result["total_jet"]
        logging.debug("All tests passed.")

        return result
