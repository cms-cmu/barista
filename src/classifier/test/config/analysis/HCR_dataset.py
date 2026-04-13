from __future__ import annotations

from typing import TYPE_CHECKING, Callable

from src.classifier.config.setting import IO
from src.classifier.config.setting.cms import CollisionData
from src.classifier.config.setting.HCR import InputBranch
from src.classifier.task import Analysis, ArgParser, converter

if TYPE_CHECKING:
    import awkward as ak


def _merge_fields(
    data: ak.Array, to_merge: ak.Array, rename: Callable[[list[str]], list[str]]
):
    for k, v in zip(rename(to_merge.fields), to_merge.fields):
        data[k] = to_merge[v]
    return data


class RandomDataset(Analysis):
    argparser = ArgParser()
    argparser.add_argument(
        "--seed", type=int, default=0, help="Random seed to generate dataset."
    )
    argparser.add_argument(
        "--nevents", type=converter.int_pos, default=1, help="Total number of events."
    )
    argparser.add_argument(
        "--phi",
        type=converter.bounded(float, 0, 1, False, True),
        default=1,
        help="Phi range of the jets in radians/pi.",
    )
    argparser.add_argument(
        "--name", type=str, default="dataset", help="Name of the dataset."
    )

    def __init__(self):
        super().__init__()
        import numpy as np

        self._phi = np.pi * self.opts.phi

    def analyze(self, results):
        return [self._generate]

    def _generate(self):
        import awkward as ak
        import numpy as np

        from src.data_formats.root import TreeWriter

        from ...physics.random_object import RandomMPtEtaPhi, typed_uniform

        rng = np.random.Generator(np.random.PCG64(self.opts.seed))

        weight = typed_uniform(rng, 0.5, 1, self.opts.nevents, np.float32)
        data = ak.zip(
            {"weight": weight, "event": np.arange(self.opts.nevents, dtype=np.uint64)}
        )

        max_jets = InputBranch.n_CanJet + InputBranch.n_NotCanJet
        jets = RandomMPtEtaPhi(
            pt_range=(40, 1000),
            mass_pt_ratio=(0.03, 0.15),
            eta_range=(-5, 5),
            phi_range=(-self._phi, self._phi),
            count_range=(
                InputBranch.n_CanJet,
                max_jets,
            ),
        ).generate(self.opts.nevents, self.opts.seed + 1)
        canjets = jets[:, : InputBranch.n_CanJet]
        _merge_fields(data, canjets, InputBranch.get__feature_CanJet)
        notcanjets = jets[:, InputBranch.n_CanJet :]
        n_notcanjets = ak.num(notcanjets)
        notcanjets["isSelJet"] = ak.unflatten(
            rng.choice(np.array([True, False], dtype=bool), ak.sum(n_notcanjets)),
            n_notcanjets,
        )
        _merge_fields(data, notcanjets, InputBranch.get__feature_NotCanJet)
        if "year" in InputBranch.feature_ancillary:
            data["year"] = rng.choice(
                [*map(int, CollisionData.years)], self.opts.nevents
            )
        if "nSelJets" in InputBranch.feature_ancillary:
            data["nSelJets"] = (
                ak.sum(notcanjets["isSelJet"], axis=1) + InputBranch.n_CanJet
            )
        if "xW" in InputBranch.feature_ancillary:
            data["xW"] = typed_uniform(rng, 0, 1, self.opts.nevents, np.float32)
        if "xbW" in InputBranch.feature_ancillary:
            data["xbW"] = typed_uniform(rng, 0, 1, self.opts.nevents, np.float32)

        path = IO.output / f"{self.opts.name}.root"
        with TreeWriter()(path) as writer:
            writer.extend(data)

        return {
            "path": str(path),
            "n_events": self.opts.nevents,
            "seed": self.opts.seed,
        }
